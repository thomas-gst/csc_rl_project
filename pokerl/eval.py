#!/usr/bin/env python3
"""
eval.py — Script d'évaluation d'un modèle MaskablePPO entraîné.

Usage :
    python eval.py eval.model_path=models/best/best_model.zip
    python eval.py eval.model_path=models/pokerl_final.zip eval.opponent=heuristic eval.n_battles=200
    python eval.py eval.model_path=models/pokerl_final.zip eval.render=true
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sb3_contrib import MaskablePPO

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.environment import make_env
from src.rewards import build_reward


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


def make_opponent(name: str, battle_format: str, server_cfg: ServerConfiguration):
    opponents = {
        "random": RandomPlayer,
        "max_power": MaxBasePowerPlayer,
        "heuristic": SimpleHeuristicsPlayer,
    }
    cls = opponents.get(name, RandomPlayer)
    return cls(battle_format=battle_format, server_configuration=server_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    cfg: DictConfig,
    opponent_name: str = "random",
    n_battles: int = 100,
    render: bool = False,
):
    """Évalue un modèle entraîné sur *n_battles* combats."""

    server_cfg = make_server_config(cfg)
    battle_format = cfg.battle.format
    reward_fn = build_reward(OmegaConf.to_container(cfg.reward, resolve=True))

    opponent = make_opponent(opponent_name, battle_format, server_cfg)
    env = make_env(
        reward_fn=reward_fn,
        opponent=opponent,
        battle_format=battle_format,
        server_configuration=server_cfg,
        log_level=logging.WARNING,
    )

    model = MaskablePPO.load(model_path, env=env)

    print(f"\n{'='*60}")
    print(f"  PokeRL — Évaluation")
    print(f"  Modèle       : {model_path}")
    print(f"  Adversaire   : {opponent_name}")
    print(f"  Combats      : {n_battles}")
    print(f"  Format       : {battle_format}")
    print(f"{'='*60}\n")

    wins = 0
    total_rewards: list[float] = []

    for episode in range(1, n_battles + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if render:
                env.render()

        # Déterminer la victoire via l'environnement interne
        battle = env._pokerl_env.battle1
        if battle is not None and battle.won:
            wins += 1

        total_rewards.append(episode_reward)

        if episode % 10 == 0 or episode == n_battles:
            win_rate = wins / episode * 100
            avg_reward = np.mean(total_rewards)
            print(
                f"  [{episode:>4d}/{n_battles}]  "
                f"Victoires: {wins}/{episode} ({win_rate:.1f}%)  "
                f"Récompense moy: {avg_reward:.2f}"
            )

    print(f"\n{'─'*60}")
    print(f"  Résultat final : {wins}/{n_battles} victoires ({wins/n_battles*100:.1f}%)")
    print(f"  Récompense moy : {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"{'─'*60}\n")

    env.close()
    return wins, n_battles


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    model_path = cfg.eval.model_path
    if not model_path:
        raise ValueError(
            "Aucun modèle fourni. Lance: python eval.py eval.model_path=models/best_model.zip"
        )

    evaluate(
        model_path=model_path,
        cfg=cfg,
        opponent_name=cfg.eval.opponent,
        n_battles=int(cfg.eval.n_battles),
        render=bool(cfg.eval.render),
    )


if __name__ == "__main__":
    main()
