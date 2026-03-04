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
import sys
from pathlib import Path
from typing import Any

import yaml
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.type_aliases import MaybeCallback

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
        return cls.load(resume_path, env=env, tensorboard_log=str(log_dir))

    # --- Nouveau modèle ---
    seed = cfg["training"].get("seed")
    log_dir_str = str(log_dir)

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
            tensorboard_log=log_dir_str,
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
            tensorboard_log=log_dir_str,
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
    log_dir = Path(cfg["training"]["log_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Modèle (factory) ──
    algo_name = cfg.get("algorithm", "MaskablePPO")
    model = build_model(algo_name, env, cfg, log_dir, resume_path)

    # ── Callbacks ──
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["training"]["save_freq"],
        save_path=str(model_dir),
        name_prefix="pokerl",
    )

    # MaskableEvalCallback pour les algos supportant l'action masking,
    # EvalCallback standard pour DQN (pas de masquage).
    if algo_name in MASKABLE_ALGOS:
        eval_cb = MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=cfg["training"]["save_freq"],
            n_eval_episodes=5,
            deterministic=True,
        )
    else:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir / "eval"),
            eval_freq=cfg["training"]["save_freq"],
            n_eval_episodes=5,
            deterministic=True,
        )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

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
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=cfg["training"]["log_interval"],
        )
    except KeyboardInterrupt:
        print("\nInterruption — sauvegarde du modèle en cours…")
    finally:
        final_path = str(model_dir / "pokerl_final")
        model.save(final_path)
        print(f"Modèle sauvegardé : {final_path}.zip")

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
