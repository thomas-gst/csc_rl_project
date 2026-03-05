#!/usr/bin/env python3
"""
train.py — Script principal d'entraînement pour PokeRL.

Algorithmes disponibles (via config.yaml → algorithm) :
    MaskablePPO   (sb3-contrib)        — action masking natif ✓
    RecurrentPPO  (sb3-contrib)        — mémoire LSTM + action masking ✓
    DQN           (stable-baselines3)  — replay buffer, PAS d'action masking ✗

Usage :
    python train.py                        # lance avec config.yaml par défaut
    python train.py --config custom.yaml   # config personnalisée
    python train.py --resume models/best_model.zip  # reprendre un entraînement
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

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
    "DQN":          DQN,           # ✗ off-policy | masking ✗ | replay buffer
}

# Algorithmes qui supportent l'Action Masking (MaskableEvalCallback requis)
MASKABLE_ALGOS = {"MaskablePPO", "RecurrentPPO"}


class EvalTensorboardBestOnlyCallback(BaseCallback):
    """Évalue périodiquement, log dans TensorBoard et sauvegarde uniquement le meilleur modèle."""

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int,
        n_eval_episodes: int,
        tb_log_dir: Path,
        best_model_path: Path,
        deterministic: bool,
        maskable: bool,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = int(n_eval_episodes)
        self.tb_log_dir = Path(tb_log_dir)
        self.best_model_path = Path(best_model_path)
        self.deterministic = deterministic
        self.maskable = maskable
        self.best_mean_reward = -math.inf
        self._writer: SummaryWriter | None = None
        self._has_evaluated = False

    def _on_training_start(self) -> None:
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(self.tb_log_dir))

    def _log_requested_train_metrics(self) -> None:
        """Relaye certaines métriques `train/*` de SB3 vers TensorBoard (au pas d'évaluation)."""
        if self._writer is None:
            return

        logger_values = getattr(self.model, "logger", None)
        name_to_value = getattr(logger_values, "name_to_value", None)
        if not isinstance(name_to_value, dict):
            return

        requested_keys = {
            "train/explained_variance",
            "train/entropy_loss",
            "train/approx_kl",
            "train/clip_fraction",
        }
        for key in requested_keys:
            value = name_to_value.get(key)
            if value is not None:
                self._writer.add_scalar(key, float(value), self.num_timesteps)

    def _evaluate_and_log(self) -> None:
        if self.maskable:
            rewards, lengths = maskable_evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
                use_masking=True,
            )
        else:
            rewards, lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=False,
            )

        mean_reward = sum(rewards) / len(rewards)
        mean_ep_len = sum(lengths) / len(lengths)
        self._has_evaluated = True

        if self._writer is not None:
            self._writer.add_scalar("eval/mean_reward", mean_reward, self.num_timesteps)
            self._writer.add_scalar("eval/mean_ep_length", mean_ep_len, self.num_timesteps)
            self._log_requested_train_metrics()
            self._writer.flush()

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
        if self._writer is not None:
            self._writer.close()


def build_model(
    algo_name: str,
    env: Any,
    cfg: dict,
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
    seed = cfg["training"].get("seed")
    if algo_name in ("MaskablePPO", "RecurrentPPO"):
        # Configuration commune PPO-like
        ppo = cfg["ppo"]
        net_arch = ppo.get("net_arch", [256, 256])
        return cls(
            policy=ppo.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=float(ppo["learning_rate"]),
            n_steps=ppo["n_steps"],
            batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"],
            gamma=ppo["gamma"],
            gae_lambda=ppo["gae_lambda"],
            clip_range=ppo["clip_range"],
            ent_coef=ppo["ent_coef"],
            vf_coef=ppo["vf_coef"],
            max_grad_norm=ppo["max_grad_norm"],
            policy_kwargs={"net_arch": net_arch},
            tensorboard_log=None,
            seed=seed,
            verbose=1,
        )

    if algo_name == "DQN":
        #   DQN (stable-baselines3)   — PAS de masquage natif ; les actions
        #   invalides restent sélectionnables. L'env retombe sur un move
        #   aléatoire (strict=False), ce qui biaise le signal de récompense.
        #   Préférer MaskablePPO pour Pokémon Showdown.
        dqn = cfg["dqn"]
        net_arch = dqn.get("net_arch", [256, 256])
        return cls(
            policy=dqn.get("policy", "MlpPolicy"),
            env=env,
            learning_rate=float(dqn["learning_rate"]),
            buffer_size=dqn["buffer_size"],
            learning_starts=dqn["learning_starts"],
            batch_size=dqn["batch_size"],
            gamma=dqn["gamma"],
            tau=dqn["tau"],
            target_update_interval=dqn["target_update_interval"],
            exploration_fraction=dqn["exploration_fraction"],
            exploration_initial_eps=dqn["exploration_initial_eps"],
            exploration_final_eps=dqn["exploration_final_eps"],
            policy_kwargs={"net_arch": net_arch},
            tensorboard_log=None,
            seed=seed,
            verbose=1,
        )

    raise NotImplementedError(f"build_model non implémenté pour {algo_name}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_server_config(cfg: dict) -> ServerConfiguration:
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
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

def train(cfg: dict, resume_path: str | None = None):
    """Lance l'entraînement avec l'algorithme défini dans cfg['algorithm']."""

    server_cfg = make_server_config(cfg)
    battle_format = cfg["battle"]["format"]

    # ── Récompense ──
    reward_fn = build_reward(cfg["reward"])

    # ── Environnement d'entraînement ──
    train_opponent_name = cfg["training"].get("opponent", "random")
    opponent = make_opponent(train_opponent_name, battle_format, server_cfg)
    env = make_env(
        reward_fn=reward_fn,
        opponent=opponent,
        battle_format=battle_format,
        server_configuration=server_cfg,
    )

    # ── Environnement d'évaluation ──
    eval_reward_fn = build_reward(cfg["reward"])
    eval_opponent_name = cfg["eval"].get("opponent", "max_power")
    eval_opponent = make_opponent(eval_opponent_name, battle_format, server_cfg)
    eval_env = make_env(
        reward_fn=eval_reward_fn,
        opponent=eval_opponent,
        battle_format=battle_format,
        server_configuration=server_cfg,
    )

    # ── Dossiers ──
    model_dir = Path(cfg["training"]["model_dir"])
    base_log_dir = Path(cfg["training"]["log_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    base_log_dir.mkdir(parents=True, exist_ok=True)

    # ── Modèle (factory) ──
    algo_name = cfg.get("algorithm", "MaskablePPO")
    algo_cfg_key = "ppo" if algo_name in MASKABLE_ALGOS else "dqn"
    algo_learning_rate = cfg.get(algo_cfg_key, {}).get("learning_rate")
    ppo_ent_coef = cfg.get("ppo", {}).get("ent_coef", "na")
    run_name = (
        f"{algo_name}_lr{algo_learning_rate}"
        f"_entcoef{ppo_ent_coef}_opp{train_opponent_name}"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = base_log_dir / f"{run_name}_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(algo_name, env, cfg, run_log_dir, resume_path)

    # ── Callbacks ──
    eval_best_only_cb = EvalTensorboardBestOnlyCallback(
        eval_env=eval_env,
        eval_freq=cfg["training"]["save_freq"],
        n_eval_episodes=50,
        tb_log_dir=run_log_dir,
        best_model_path=model_dir / "best_model",
        deterministic=True,
        maskable=algo_name in MASKABLE_ALGOS,
    )

    # ── Lancement ──
    total_timesteps = int(cfg["training"]["total_timesteps"])
    masking_label = "actif" if algo_name in MASKABLE_ALGOS else "inactif (DQN)"
    net_arch = cfg.get("ppo" if algo_name in MASKABLE_ALGOS else "dqn", {}).get("net_arch", [])
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
    print(f"  TB logs      : {run_log_dir}")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_best_only_cb,
            log_interval=cfg["training"]["log_interval"],
        )
    except KeyboardInterrupt:
        print("\nInterruption — arrêt propre.")
    finally:
        best_path = model_dir / "best_model.zip"
        if best_path.exists():
            print(f"Meilleur modèle sauvegardé : {best_path}")
        else:
            print("Aucun modèle best n'a été sauvegardé.")

        # Nettoyage des environnements
        env.close()
        eval_env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PokeRL — Train (MaskablePPO | RecurrentPPO | DQN)")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(SCRIPT_DIR / "config.yaml"),
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Chemin vers un modèle .zip pour reprendre l'entraînement",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
