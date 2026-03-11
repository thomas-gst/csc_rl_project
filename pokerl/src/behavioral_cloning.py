"""
behavioral_cloning.py — Module de Behavioral Cloning et DAgger pour PokeRL.

Permet de pré-entraîner la politique d'un agent MaskablePPO par imitation
(Behavioral Cloning) puis d'affiner les distributions via DAgger (Dataset
Aggregation) avant de basculer sur l'apprentissage par renforcement.

Flux typique :
    1. Collecte initiale (expert joue dans l'environnement) → cache .npz
    2. Entraînement supervisé (cross-entropy) du réseau acteur  [round 0 BC]
    3. Rollout de la politique courante + labelisation par l'expert [DAgger]
    4. Agrégation du dataset + ré-entraînement                [rounds DAgger]
    5. Reprise en mode RL avec le modèle pré-entraîné

Classes / fonctions publiques :
    ExpertDemonstrationCollector  — collecte (obs, mask, action) de l'expert
    PolicyRolloutCollector        — DAgger : politique → états, expert → labels
    save_demonstrations          — sérialisation .npz
    load_demonstrations          — chargement .npz
    aggregate_demonstrations     — union de deux jeux de démonstrations
    train_bc                     — entraînement supervisé de la politique
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Collecte de démonstrations
# ─────────────────────────────────────────────────────────────────────────────

class ExpertDemonstrationCollector:
    """Collecte des triplets ``(observation, action_mask, expert_action)``
    en faisant jouer un expert dans l'environnement de combat.

    L'expert (ex. ``SimpleHeuristicsPlayer``) fournit l'action, tandis que
    l'environnement fournit l'observation et le masque d'action.  Seules
    les décisions *valides* (action dans le masque) sont conservées.
    """

    def __init__(self, env: Any, expert_player: Any):
        """
        Args:
            env:            MaskableSingleAgentWrapper — environnement Gymnasium
                            mono-agent avec action_masks().
            expert_player:  Instance de ``Player`` poke-env dont on utilisera
                            ``choose_move(battle)`` pour obtenir les décisions.
        """
        self.env = env
        self.expert = expert_player

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_expert_action(self, battle: Any) -> Optional[int]:
        """Interroge l'expert et convertit sa décision en index d'action."""
        if battle is None or battle.finished:
            return None
        try:
            expert_order = self.expert.choose_move(battle)
            action = self.env._pokerl_env.order_to_action(
                expert_order,
                battle,
                fake=self.env._pokerl_env.fake,
                strict=self.env._pokerl_env.strict,
            )
            action_int = int(action)
            return action_int if action_int >= 0 else None
        except (ValueError, Exception) as exc:
            logger.debug("Action expert invalide : %s", exc)
            return None

    @staticmethod
    def _is_default_only(battle: Any) -> bool:
        """Détecte un tour « /choose default » sans décision à prendre."""
        if battle is None:
            return False
        if len(battle.valid_orders) != 1:
            return False
        return str(battle.valid_orders[0]) == "/choose default"

    # ── Collecte principale ──────────────────────────────────────────────

    def collect(self, n_episodes: int, *, verbose: bool = True) -> dict:
        """Joue *n_episodes* épisodes avec l'expert et renvoie les démonstrations.

        Returns:
            dict avec les clés :
                observations  — np.ndarray (N, obs_dim), float32
                action_masks  — np.ndarray (N, act_dim), bool
                actions       — np.ndarray (N,),         int64
        """
        observations: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []

        iterator = range(n_episodes)
        if verbose:
            iterator = tqdm(iterator, desc="Collecte BC", unit="ep")

        skipped = 0
        wins = 0

        for _ep in iterator:
            obs, _info = self.env.reset()
            done = False

            while not done:
                mask = self.env.action_masks()
                battle = self.env._pokerl_env.battle1

                # Tour par défaut (pas de décision) → skip
                if self._is_default_only(battle):
                    valid = np.where(mask)[0]
                    obs, _, terminated, truncated, _ = self.env.step(
                        np.int64(valid[0])
                    )
                    done = terminated or truncated
                    continue

                expert_action = self._get_expert_action(battle)

                if (
                    expert_action is not None
                    and 0 <= expert_action < len(mask)
                    and mask[expert_action]
                ):
                    # Action valide de l'expert → enregistrer
                    observations.append(obs.copy())
                    masks.append(mask.copy())
                    actions.append(expert_action)
                    step_action = np.int64(expert_action)
                else:
                    # Fallback : action masquée aléatoire (non enregistrée)
                    valid = np.where(mask)[0]
                    step_action = (
                        np.int64(np.random.choice(valid)) if len(valid) > 0 else np.int64(0)
                    )
                    skipped += 1

                obs, _, terminated, truncated, _ = self.env.step(step_action)
                done = terminated or truncated

            # Comptabiliser la victoire de l'expert
            battle = self.env._pokerl_env.battle1
            if battle is not None and battle.won:
                wins += 1

            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    steps=len(actions), skip=skipped, wins=wins
                )

        if not actions:
            raise RuntimeError(
                "Aucune démonstration collectée ! "
                "Vérifiez l'expert et l'environnement."
            )

        result = {
            "observations": np.array(observations, dtype=np.float32),
            "action_masks": np.array(masks, dtype=np.bool_),
            "actions": np.array(actions, dtype=np.int64),
        }
        print(
            f"\n[BC] Collecte terminée : {len(actions)} steps utiles "
            f"sur {n_episodes} épisodes ({skipped} steps ignorés, "
            f"{wins} victoires expert)"
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Collecte DAgger — la politique courante explore, l'expert labelise
# ─────────────────────────────────────────────────────────────────────────────

class PolicyRolloutCollector:
    """Collecte DAgger : la **politique courante** joue les épisodes mais
    l'expert *labelise* chaque état visité.

    Contrairement à ``ExpertDemonstrationCollector`` (qui fait jouer l'expert),
    ici c'est la politique qui choisit l'action de déplacement.  L'expert ne
    sert qu'à annoter les états rencontrés.  Cela corrige le distributional
    shift : la politique apprend sur les états *qu'elle visite réellement*.
    """

    def __init__(self, env: Any, expert_player: Any):
        self.env = env
        self.expert = expert_player
        # Réutiliser les helpers de la collecte experte
        self._bc_collector = ExpertDemonstrationCollector(env, expert_player)

    def collect(
        self,
        model: Any,
        n_episodes: int,
        *,
        verbose: bool = True,
    ) -> dict:
        """Joue *n_episodes* épisodes avec la politique et labelise avec l'expert.

        Args:
            model:      Politique courante (MaskablePPO).
            n_episodes: Nombre d'épisodes à dérouler.
            verbose:    Affichage tqdm.
        """
        observations: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []

        iterator = range(n_episodes)
        if verbose:
            iterator = tqdm(iterator, desc="DAgger rollout", unit="ep")

        skipped = 0
        wins = 0

        for _ep in iterator:
            obs, _info = self.env.reset()
            done = False

            while not done:
                mask = self.env.action_masks()
                battle = self.env._pokerl_env.battle1

                # Tour par défaut (pas de décision) → skip
                if self._bc_collector._is_default_only(battle):
                    valid = np.where(mask)[0]
                    obs, _, terminated, truncated, _ = self.env.step(
                        np.int64(valid[0])
                    )
                    done = terminated or truncated
                    continue

                # --- Labelisation par l'expert ---
                expert_action = self._bc_collector._get_expert_action(battle)
                if (
                    expert_action is not None
                    and 0 <= expert_action < len(mask)
                    and mask[expert_action]
                ):
                    observations.append(obs.copy())
                    masks.append(mask.copy())
                    actions.append(expert_action)
                else:
                    skipped += 1

                # --- Action de déplacement : politique courante ---
                action_pred, _ = model.predict(
                    obs, deterministic=False, action_masks=mask
                )
                step_action = np.int64(int(action_pred))

                obs, _, terminated, truncated, _ = self.env.step(step_action)
                done = terminated or truncated

            battle = self.env._pokerl_env.battle1
            if battle is not None and battle.won:
                wins += 1
            if verbose and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(steps=len(actions), skip=skipped, wins=wins)

        if not actions:
            logger.warning("[DAgger] Aucun step utile collecté dans ce round.")
            return {
                "observations": np.empty((0,), dtype=np.float32),
                "action_masks": np.empty((0,), dtype=np.bool_),
                "actions": np.empty((0,), dtype=np.int64),
            }

        print(
            f"\n[DAgger] Rollout terminé : {len(actions)} steps utiles "
            f"sur {n_episodes} épisodes ({skipped} ignorés, {wins} victoires politique)"
        )
        return {
            "observations": np.array(observations, dtype=np.float32),
            "action_masks": np.array(masks, dtype=np.bool_),
            "actions": np.array(actions, dtype=np.int64),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sérialisation des démonstrations
# ─────────────────────────────────────────────────────────────────────────────

def save_demonstrations(data: dict, path: Path | str) -> None:
    """Sauvegarde les démonstrations au format ``.npz`` compressé."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **data)
    print(f"[BC] Démonstrations sauvegardées : {path}")


def load_demonstrations(path: Path | str) -> dict:
    """Charge les démonstrations depuis un fichier ``.npz``."""
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def aggregate_demonstrations(base: dict, new: dict) -> dict:
    """Concatène deux jeux de démonstrations le long de l'axe 0.

    Utilisé par DAgger pour agréger les rollouts de la politique courante
    aux données existantes.
    """
    if not base.get("actions", np.array([])).size:
        return new
    if not new.get("actions", np.array([])).size:
        return base
    return {
        key: np.concatenate([base[key], new[key]], axis=0)
        for key in base
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement Behavioral Cloning
# ─────────────────────────────────────────────────────────────────────────────

def _extract_logits(policy: Any, obs: torch.Tensor) -> torch.Tensor:
    """Forward-pass à travers le réseau acteur pour obtenir les logits bruts.

    Gère les deux versions de l'API SB3 (v1 et v2+) de manière transparente.
    """
    # SB3 v2+ : extract_features(obs, features_extractor)
    if hasattr(policy, "pi_features_extractor"):
        features = policy.extract_features(obs, policy.pi_features_extractor)
    else:
        features = policy.extract_features(obs)

    # MLP extractor : forward_actor (v2+) ou forward complet
    if hasattr(policy.mlp_extractor, "forward_actor"):
        latent_pi = policy.mlp_extractor.forward_actor(features)
    else:
        latent_pi, _ = policy.mlp_extractor(features)

    return policy.action_net(latent_pi)


def _run_eval(
    model: Any,
    eval_env: Any,
    n_episodes: int,
) -> float:
    """Joue *n_episodes* épisodes avec la politique courante et retourne le win rate."""
    wins = 0
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action_masks = eval_env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=action_masks,
            )
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = bool(terminated or truncated)
        battle = eval_env._pokerl_env.battle1
        if battle is not None and battle.won:
            wins += 1
    return wins / max(n_episodes, 1)


def train_bc(
    model: Any,
    demonstrations: dict,
    *,
    n_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    verbose: bool = True,
    wandb_module: Any = None,
    global_step_offset: int = 0,
    eval_env: Any = None,
    n_eval_episodes: int = 50,
    phase_label: str = "bc",
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 1e-4,
) -> dict:
    """Entraîne la politique du modèle par clonage comportemental (BC ou round DAgger).

    Minimise la cross-entropy entre les logits masqués du réseau acteur
    et les actions de l'expert.

    :param model:                       Instance MaskablePPO (sb3-contrib).
    :param demonstrations:              Dict contenant 'observations', 'action_masks', 'actions'.
    :param n_epochs:                    Nombre de passes complètes sur les données.
    :param batch_size:                  Taille des mini-batchs.
    :param learning_rate:               Taux d'apprentissage pour Adam (BC seulement).
    :param verbose:                     Affichage époque par époque.
    :param wandb_module:                Module ``wandb`` pour le logging (ou ``None``).
    :param global_step_offset:          Offset pour le step W&B (utile si le RL suit).
    :param phase_label:                 Préfixe W&B (\"bc\" pour le round initial, \"dagger\" pour
                                        les rounds DAgger).
    :param early_stopping_patience:     Époques sans amélioration avant arrêt (0 = désactivé).
                                        Métrique surveillée : win rate si eval_env fourni,
                                        sinon loss.
    :param early_stopping_min_delta:    Amélioration minimale considérée comme significative.
    :return:                            Historique ``{'loss': [...], 'accuracy': [...]}``.
    """
    policy = model.policy
    policy.set_training_mode(True)
    device = policy.device

    # ── Préparation des tensors ──
    obs_t = torch.as_tensor(demonstrations["observations"]).float().to(device)
    mask_t = torch.as_tensor(demonstrations["action_masks"]).bool().to(device)
    act_t = torch.as_tensor(demonstrations["actions"]).long().to(device)

    n_samples = len(act_t)
    n_actions = int(model.action_space.n)
    label_up = phase_label.upper()

    print(f"\n[{label_up}] Entraînement sur {n_samples} échantillons "
          f"({n_actions} actions, {n_epochs} epochs, batch={batch_size})")

    dataset = TensorDataset(obs_t, mask_t, act_t)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Optimiseur dédié au BC (indépendant de l'optimiseur PPO)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    history: dict[str, list[float]] = {"loss": [], "accuracy": []}

    # On initialise le compteur de steps du modèle pour s'aligner avec le RL
    if not hasattr(model, "num_timesteps"):
        model.num_timesteps = 0
    model.num_timesteps += global_step_offset

    # ── Early Stopping ──
    use_es = early_stopping_patience > 0
    # Surveille win_rate (max) si eval_env fourni, sinon loss (min)
    monitor_win_rate = (eval_env is not None)
    es_best_value = -math.inf if monitor_win_rate else math.inf
    es_wait = 0
    es_best_weights: Any = None  # deepcopy du state_dict au meilleur état

    def _is_improvement(current: float) -> bool:
        nonlocal es_best_value
        if monitor_win_rate:
            improved = current > es_best_value + early_stopping_min_delta
        else:
            improved = current < es_best_value - early_stopping_min_delta
        if improved:
            es_best_value = current
        return improved

    for epoch in range(n_epochs):
        losses: list[float] = []
        correct = 0
        total = 0

        for obs_b, mask_b, act_b in loader:
            logits = _extract_logits(policy, obs_b)

            # Masquer les actions invalides (logits → -inf)
            neg_inf = torch.finfo(logits.dtype).min
            logits_masked = logits.masked_fill(~mask_b, neg_inf)

            loss = F.cross_entropy(logits_masked, act_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            correct += (logits_masked.argmax(-1) == act_b).sum().item()
            total += len(act_b)
            # Imputer chaque observation comme un "timestep" d'environnement
            model.num_timesteps += len(act_b)

        mean_loss = sum(losses) / max(len(losses), 1)
        acc = correct / max(total, 1)
        history["loss"].append(mean_loss)
        history["accuracy"].append(acc)

        # ── Évaluation post-époque ──
        win_rate: float | None = None
        if eval_env is not None:
            policy.set_training_mode(False)
            win_rate = _run_eval(model, eval_env, n_eval_episodes)
            policy.set_training_mode(True)
            history.setdefault("win_rate", []).append(win_rate)

        # ── Early Stopping bookkeeping ──
        es_triggered = False
        if use_es:
            monitored = win_rate if monitor_win_rate else mean_loss
            if monitored is not None and _is_improvement(monitored):
                es_best_weights = copy.deepcopy(policy.state_dict())
                es_wait = 0
            else:
                es_wait += 1
                if es_wait >= early_stopping_patience:
                    es_triggered = True

        if verbose:
            es_str = ""
            if use_es:
                metric_name = "wr" if monitor_win_rate else "loss"
                es_str = f" [ES wait {es_wait}/{early_stopping_patience}, best {metric_name}: "
                es_str += f"{es_best_value:.4f}]"
            wr_str = f" — win rate: {win_rate:.1%}" if win_rate is not None else ""
            print(
                f"  [{label_up}] Epoch {epoch + 1:3d}/{n_epochs} — "
                f"loss: {mean_loss:.4f} — accuracy: {acc:.2%}{wr_str}{es_str}"
            )

        # ── Logging W&B ──
        if wandb_module is not None and wandb_module.run is not None:
            payload = {
                f"{phase_label}/loss": mean_loss,
                f"{phase_label}/accuracy": acc,
            }
            if win_rate is not None:
                payload["eval/win_rate"] = win_rate
            if use_es:
                payload[f"{phase_label}/es_wait"] = es_wait
            wandb_module.log(payload, step=model.num_timesteps)

        if es_triggered:
            print(
                f"  [{label_up}] Early stopping déclenché à l'époque {epoch + 1} "
                f"({early_stopping_patience} époques sans amélioration). "
                f"Meilleure {metric_name}: {es_best_value:.4f}"
            )
            break

    # Restaurer les meilleurs poids si early stopping activé
    if use_es and es_best_weights is not None:
        policy.load_state_dict(es_best_weights)
        if verbose:
            print(f"  [{label_up}] Poids restaurés au meilleur état sauvegardé.")

    policy.set_training_mode(False)
    print(
        f"[{label_up}] Terminé — loss finale: {history['loss'][-1]:.4f} — "
        f"accuracy finale: {history['accuracy'][-1]:.2%}"
    )
    return history
