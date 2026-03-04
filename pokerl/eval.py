#!/usr/bin/env python3
"""
eval.py — Script d'évaluation d'un modèle MaskablePPO entraîné.

Usage :
    python eval.py --model models/best/best_model.zip
    python eval.py --model models/pokerl_final.zip --opponent heuristic --battles 200
    python eval.py --model models/pokerl_final.zip --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
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
    cfg: dict,
    opponent_name: str = "random",
    n_battles: int = 100,
    render: bool = False,
):
    """Évalue un modèle entraîné sur *n_battles* combats."""

    server_cfg = make_server_config(cfg)
    battle_format = cfg["battle"]["format"]
    reward_fn = build_reward(cfg["reward"])

    opponent = make_opponent(opponent_name, battle_format, server_cfg)
    env = make_env(
        reward_fn=reward_fn,
        opponent=opponent,
        battle_format=battle_format,
        server_configuration=server_cfg,
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

def main():
    parser = argparse.ArgumentParser(description="PokeRL — Evaluate MaskablePPO")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Chemin vers le modèle .zip entraîné",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=str(SCRIPT_DIR / "config.yaml"),
        help="Chemin vers le fichier de configuration YAML",
    )
    parser.add_argument(
        "--opponent", "-o",
        type=str,
        default=None,
        help="Type d'adversaire : random | max_power | heuristic",
    )
    parser.add_argument(
        "--battles", "-b",
        type=int,
        default=None,
        help="Nombre de combats d'évaluation",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Afficher le rendu du combat dans le terminal",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    opponent_name = args.opponent or cfg["eval"]["opponent"]
    n_battles = args.battles or cfg["eval"]["n_battles"]

    evaluate(
        model_path=args.model,
        cfg=cfg,
        opponent_name=opponent_name,
        n_battles=n_battles,
        render=args.render,
    )


if __name__ == "__main__":
    main()
