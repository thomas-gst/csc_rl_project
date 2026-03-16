#!/usr/bin/env python3
"""
train.py — Script principal d'entraînement pour PokeRL.

Algorithmes disponibles (via config.yaml → algorithm) :
    MaskablePPO   (sb3-contrib)        — action masking natif ✓
    RecurrentPPO  (sb3-contrib)        — mémoire LSTM + action masking ✓
    DQN           (stable-baselines3)  — replay buffer, PAS d'action masking ✗

Usage :
    python train.py
    python train.py algorithm=recurrentppo
    python train.py resume_path=models/best_model.zip
    python train.py wandb.project=my_project wandb.entity=my_team
"""

from __future__ import annotations
import torch as th
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
)

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration
from attention_model import AttentionPolicy

# Ajouter le dossier parent au path pour les imports locaux
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.environment import make_env, PokeRLEnv, MaskableSingleAgentWrapper
from src.rewards import build_reward



# ───────────────────────────────────────────────────────────────────────────────
# Factory : instanciation de l'algorithme
# ───────────────────────────────────────────────────────────────────────────────

# Registre des algorithmes disponibles :
#   ✓ Action Masking natif  — MaskablePPO, RecurrentPPO (sb3-contrib)
#   ✗ PAS d'Action Masking  — DQN (stable-baselines3)
ALGO_REGISTRY: dict[str, type] = {
    "MaskablePPO":  MaskablePPO,   # ✓ on-policy  | masking ✓ | MLP
    "RecurrentPPO": RecurrentPPO,  # ✓ on-policy  | masking ✓ | LSTM
    "DQN":          DQN,  
    "AttentionPPO": MaskablePPO, 
}

# Algorithmes qui supportent l'Action Masking (MaskableEvalCallback requis)
MASKABLE_ALGOS = {"MaskablePPO", "RecurrentPPO", "AttentionPPO"}


class WandbMetricsCallback(BaseCallback):
    """Envoie périodiquement les métriques SB3 (rollout/train/time) vers W&B."""

    def __init__(self, enabled: bool, log_freq_steps: int):
        super().__init__()
        self.enabled = enabled
        self.log_freq_steps = max(1, int(log_freq_steps))

    def _log_metrics(self) -> None:
        if not self.enabled or wandb is None or wandb.run is None:
            return

        logger_values = getattr(self.model, "logger", None)
        name_to_value = getattr(logger_values, "name_to_value", None)
        if not isinstance(name_to_value, dict):
            return

        payload: dict[str, float] = {}
        for key, value in name_to_value.items():
            if not isinstance(key, str):
                continue
            if not (key.startswith("rollout/") or key.startswith("train/") or key.startswith("time/")):
                continue
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                continue

        if payload:
            wandb.log(payload, step=self.num_timesteps)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq_steps == 0:
            self._log_metrics()
        return True

    def _on_training_end(self) -> None:
        self._log_metrics()


class EvalBestOnlyCallback(BaseCallback):
    """Évalue périodiquement et sauvegarde uniquement le meilleur modèle."""

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_path: Path,
        deterministic: bool,
        maskable: bool,
        wandb_enabled: bool,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_model_path = Path(best_model_path)
        self.deterministic = deterministic
        self.maskable = maskable
        self.wandb_enabled = wandb_enabled
        self.best_mean_reward = -math.inf
        self._has_evaluated = False

    def _on_training_start(self) -> None:
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

    def _collect_requested_train_metrics(self) -> dict[str, float]:
        """Récupère certaines métriques ``train/*`` de SB3."""
        logger_values = getattr(self.model, "logger", None)
        name_to_value = getattr(logger_values, "name_to_value", None)
        if not isinstance(name_to_value, dict):
            return {}

        requested_keys = {
            "train/explained_variance",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
        }
        metrics: dict[str, float] = {}
        for key in requested_keys:
            value = name_to_value.get(key)
            if value is not None:
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue
        return metrics

    def _evaluate_and_log(self) -> None:
        rewards: list[float] = []
        lengths: list[int] = []
        wins = 0

        for _ in range(self.n_eval_episodes):
            obs, _info = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                if self.maskable:
                    action_masks = self.eval_env.action_masks()
                    action, _ = self.model.predict(
                        obs,
                        deterministic=self.deterministic,
                        action_masks=action_masks,
                    )
                else:
                    action, _ = self.model.predict(obs, deterministic=self.deterministic)

                obs, reward, terminated, truncated, _step_info = self.eval_env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)

            battle = self.eval_env._pokerl_env.battle1
            if battle is not None and battle.won:
                wins += 1

            rewards.append(episode_reward)
            lengths.append(episode_length)

        mean_reward = sum(rewards) / len(rewards)
        mean_ep_len = sum(lengths) / len(lengths)
        win_rate = wins / self.n_eval_episodes
        self._has_evaluated = True

        if self.wandb_enabled and wandb is not None and wandb.run is not None:
            payload = {
                "eval/mean_reward": mean_reward,
                "eval/mean_ep_length": mean_ep_len,
                "eval/win_rate": win_rate,
            }
            payload.update(self._collect_requested_train_metrics())
            wandb.log(payload, step=self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(str(self.best_model_path))
            if self.verbose > 0:
                print(
                    f"[Eval] Nouveau best @ {self.num_timesteps} steps | "
                    f"mean_reward={mean_reward:.3f} | sauvegardé: {self.best_model_path}.zip"
                )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_log()
        return True

    def _on_training_end(self) -> None:
        if not self._has_evaluated:
            self._evaluate_and_log()


def build_model(
    algo_name: str,
    env: Any,
    cfg: DictConfig,
    log_dir: Path,
    resume_path: str | None = None,
) -> Any:
    """Factory qui instancie (ou recharge) l'algorithme RL demandé.

    :param algo_name:    Clé dans ALGO_REGISTRY ("MaskablePPO" | "RecurrentPPO" | "DQN").
    :param env:          Environnement Gymnasium wrappé.
    :param cfg:          Dictionnaire de configuration complet (config.yaml).
    :param log_dir:      Répertoire TensorBoard.
    :param resume_path:  Chemin d'un .zip à recharger (None = nouveau modèle).
    :return:             Instance de l'algorithme prête à ``learn()``.
    """
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(
            f"Algorithme inconnu : '{algo_name}'. "
            f"Choix possibles : {list(ALGO_REGISTRY.keys())}"
        )

    cls = ALGO_REGISTRY[algo_name]

    # --- Reprise d'un entraînement existant ---
    if resume_path:
        print(f"Reprise de l'entraînement depuis {resume_path}")
        return cls.load(resume_path, env=env, tensorboard_log=None)

    # --- Nouveau modèle ---
    seed = cfg.training.seed
    if algo_name in ("MaskablePPO", "RecurrentPPO"):
        # Configuration commune PPO-like
        ppo = cfg.ppo
        net_arch = list(ppo.net_arch)
        return cls(
            policy=ppo.policy,
            env=env,
            learning_rate=float(ppo.learning_rate),
            n_steps=int(ppo.n_steps),
            batch_size=int(ppo.batch_size),
            n_epochs=int(ppo.n_epochs),
            gamma=float(ppo.gamma),
            gae_lambda=float(ppo.gae_lambda),
            clip_range=float(ppo.clip_range),
            ent_coef=float(ppo.ent_coef),
            vf_coef=float(ppo.vf_coef),
            max_grad_norm=float(ppo.max_grad_norm),
            policy_kwargs={"net_arch": net_arch},
            tensorboard_log=str(log_dir),
            seed=seed,
            verbose=0,
        )
    if algo_name == "AttentionPPO":
            # Configuration commune PPO-like
            attppo = cfg.attppo
            return cls(
                policy=AttentionPolicy,
                env=env,
                learning_rate=float(attppo.learning_rate),
                n_steps=int(attppo.n_steps),
                batch_size=int(attppo.batch_size),
                n_epochs=int(attppo.n_epochs),
                gamma=float(attppo.gamma),
                gae_lambda=float(attppo.gae_lambda),
                clip_range=float(attppo.clip_range),
                ent_coef=float(attppo.ent_coef),
                vf_coef=float(attppo.vf_coef),
                max_grad_norm=float(attppo.max_grad_norm),
                policy_kwargs={"cfg": attppo},
                tensorboard_log=str(log_dir),
                seed=seed,
                verbose=0,
            )   
    if algo_name == "DQN":
        #   DQN (stable-baselines3)   — PAS de masquage natif ; les actions
        #   invalides restent sélectionnables. L'env retombe sur un move
        #   aléatoire (strict=False), ce qui biaise le signal de récompense.
        #   Préférer MaskablePPO pour Pokémon Showdown.
        dqn = cfg.dqn
        net_arch = list(dqn.net_arch)
        return cls(
            policy=dqn.policy,
            env=env,
            learning_rate=float(dqn.learning_rate),
            buffer_size=int(dqn.buffer_size),
            learning_starts=int(dqn.learning_starts),
            batch_size=int(dqn.batch_size),
            gamma=float(dqn.gamma),
            tau=float(dqn.tau),
            target_update_interval=int(dqn.target_update_interval),
            exploration_fraction=float(dqn.exploration_fraction),
            exploration_initial_eps=float(dqn.exploration_initial_eps),
            exploration_final_eps=float(dqn.exploration_final_eps),
            policy_kwargs={"net_arch": net_arch},
            tensorboard_log=str(log_dir),
            seed=seed,
            verbose=0,
        )

    raise NotImplementedError(f"build_model non implémenté pour {algo_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_server_config(cfg: DictConfig) -> ServerConfiguration:
    host = cfg.server.host
    port = cfg.server.port
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        f"http://{host}:{port}/action.php?",
    )


def make_opponent(name: str, battle_format: str, server_cfg: ServerConfiguration) -> object:
    """Instancie l'adversaire selon la config."""
    opponents = {
        "random": RandomPlayer,
        "max_power": MaxBasePowerPlayer,
        "heuristic": SimpleHeuristicsPlayer,
    }
    cls = opponents.get(name, RandomPlayer)
    return cls(battle_format=battle_format, server_configuration=server_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────────────────────────────────────

def _reward_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg.reward, resolve=True)


def train(cfg: DictConfig, resume_path: str | None = None):
    """Lance l'entraînement avec l'algorithme défini dans cfg.algorithm.name."""

    server_cfg = make_server_config(cfg)
    battle_format = cfg.battle.format
    algo_name = cfg.algorithm.name

    # ── Récompense ──
    reward_fn = build_reward(_reward_dict(cfg))

    # ── Environnement d'entraînement ──
    train_opponent_name = cfg.training.opponent
    opponent = make_opponent(train_opponent_name, battle_format, server_cfg)
    env = make_env(
        reward_fn=reward_fn,
        opponent=opponent,
        battle_format=battle_format,
        server_configuration=server_cfg,
        log_level=logging.WARNING,
        team=cfg.battle.get("team"),
    )

    # ── Environnement d'évaluation ──
    eval_reward_fn = build_reward(_reward_dict(cfg))
    eval_opponent_name = cfg.eval.opponent
    eval_opponent = make_opponent(eval_opponent_name, battle_format, server_cfg)
    eval_env = make_env(
        reward_fn=eval_reward_fn,
        opponent=eval_opponent,
        battle_format=battle_format,
        server_configuration=server_cfg,
        log_level=logging.WARNING,
        team=cfg.battle.get("team"),
    )

    # ── Dossiers ──
    model_dir = Path(cfg.training.model_dir)
    base_log_dir = Path(cfg.training.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    base_log_dir.mkdir(parents=True, exist_ok=True)

    # ── Modèle (factory) ──
    if algo_name == "AttentionPPO":
        algo_cfg_key = "attppo"
    else:
        algo_cfg_key = "ppo" if algo_name in MASKABLE_ALGOS else "dqn"
    algo_section = cfg[algo_cfg_key]
    algo_learning_rate = algo_section.learning_rate
    ppo_ent_coef = algo_section.ent_coef
    run_name = (
        f"{algo_name}_lr{algo_learning_rate}"
        f"_entcoef{ppo_ent_coef}_opp{train_opponent_name}"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = base_log_dir / f"{run_name}_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = bool(cfg.wandb.enabled)
    if wandb_enabled:
        if wandb is None:
            raise ImportError(
                "wandb n'est pas installé. Installe-le avec: pip install wandb"
            )
        if not cfg.wandb.project:
            raise ValueError(
                "Configuration manquante: renseigne `wandb.project` dans "
                "configs/wandb/default.yaml ou via override CLI."
            )
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=cfg.wandb.name or f"{run_name}_{timestamp}",
            tags=list(cfg.wandb.tags),
            notes=cfg.wandb.notes or None,
            mode=cfg.wandb.mode,
            save_code=bool(cfg.wandb.save_code),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(run_log_dir),
            settings=wandb.Settings(quiet=True),
        )

    model = build_model(algo_name, env, cfg, run_log_dir, resume_path)

    # ── Callbacks ──
    eval_best_only_cb = EvalBestOnlyCallback(
        eval_env=eval_env,
        eval_freq=cfg.training.save_freq,
        n_eval_episodes=cfg.training.eval_episodes,
        best_model_path=model_dir / "best_model",
        deterministic=True,
        maskable=algo_name in MASKABLE_ALGOS,
        wandb_enabled=wandb_enabled,
    )
    callbacks: list[BaseCallback] = [
        eval_best_only_cb,
        WandbMetricsCallback(
            enabled=wandb_enabled,
            log_freq_steps=cfg.wandb.log_freq_steps,
        ),
    ]
    if wandb_enabled and WandbCallback is not None:
        callbacks.append(WandbCallback(verbose=0))
    callback = CallbackList(callbacks)

    # ── Lancement ──
    total_timesteps = int(cfg.training.total_timesteps)
    masking_label = "actif" if algo_name in MASKABLE_ALGOS else "inactif (DQN)"
    net_arch = list(algo_section.net_arch)
    print(f"\n{'='*60}")
    print(f"  PokeRL — Entraînement {algo_name}")
    print(f"  Format       : {battle_format}")
    print(f"  Timesteps    : {total_timesteps:,}")
    print(f"  Observation  : {env.observation_space.shape}")
    print(f"  Actions      : {env.action_space.n}")
    print(f"  Récompense   : {cfg['reward']['class']}")
    print(f"  Net arch     : {net_arch}")
    print(f"  Action mask  : {masking_label}")
    print(f"  Adv. train   : {train_opponent_name}")
    print(f"  Adv. eval    : {eval_opponent_name}")
    print(f"  Logs         : {run_log_dir}")
    print(f"  W&B          : {'actif' if wandb_enabled else 'inactif'}")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=int(cfg.training.log_interval),
        )
    except KeyboardInterrupt:
        print("\nInterruption — arrêt propre.")
    finally:
        try:
            if algo_name == "AttentionPPO":
                model_path = model_dir / f"{run_name}_{timestamp}.pt"
                th.save(model.policy.features_extractor.state_dict(), model_path)
                print(f"Dernier modèle sauvegardé : {model_path}")
            if algo_name == "MaskablePPO":
               model_path = model_dir / f"{run_name}_{timestamp}" 
               model.save(model_path)
               print(f"Dernier modèle sauvegardé : {model_path}") 
        except Exception as e:
            print(f"Aucun modèle best n'a été sauvegardé. Erreur :{e}")

        # Nettoyage des environnements
        env.close()
        eval_env.close()
        if wandb is not None and wandb.run is not None:
            try:
                wandb.finish()
            except Exception as err:
                print(f"[W&B] warning: échec de wandb.finish() ({err}).")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    train(cfg, resume_path=cfg.resume_path)


if __name__ == "__main__":
    main()
