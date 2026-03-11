#!/usr/bin/env python3
"""
train_bc_rl.py — Entraînement en deux phases : Behavioral Cloning → RL.

Phase 1 (Behavioral Cloning) :
    Collecte des démonstrations d'un expert (par défaut SimpleHeuristicsPlayer)
    puis pré-entraînement supervisé (cross-entropy) du réseau acteur de
    MaskablePPO afin de fournir une politique initiale raisonnable.

Phase 2 (Reinforcement Learning) :
    Fine-tuning avec MaskablePPO (PPO + action masking).

Usage :
    python train_bc_rl.py
    python train_bc_rl.py bc.n_episodes=500 bc.n_epochs=20
    python train_bc_rl.py training.total_timesteps=2000000
    python train_bc_rl.py bc.expert=max_power bc.opponent=heuristic
"""

from __future__ import annotations

import atexit
import logging
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
except ImportError:
    wandb = None
    WandbCallback = None

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

# Ajouter le dossier parent au path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.environment import make_env
from src.rewards import build_reward
from src.behavioral_cloning import (
    ExpertDemonstrationCollector,
    PolicyRolloutCollector,
    save_demonstrations,
    load_demonstrations,
    aggregate_demonstrations,
    train_bc,
)

# ───────────────────────────────────────────────────────────────────────────────
# Gestion propre de l'arrêt — fermeture des connexions websocket poke-env
# ───────────────────────────────────────────────────────────────────────────────

# Registre global des envs ouverts — pour nettoyage sur SIGINT/SIGTERM
_OPEN_ENVS: list[Any] = []


def _safe_close(*envs: Any, drain_seconds: float = 1.5) -> None:
    """Ferme les environnements poke-env et laisse le thread asyncio se vider.

    poke-env exécute ses websockets dans un thread de fond ; ``close()`` se
    contente de poster la coroutine de déconnexion — sans attendre qu'elle
    se termine.  Sans le drain, le serveur Showdown conserve les sessions
    fantômes et rejette la reconnexion suivante.
    """
    for env in envs:
        if env is None:
            continue
        try:
            env.close()
        except Exception as exc:
            # Ne pas planter le nettoyage si l'env est déjà fermé
            print(f"[cleanup] warning lors de close() : {exc}")
        finally:
            if env in _OPEN_ENVS:
                _OPEN_ENVS.remove(env)
    # Laisse le thread asyncio de poke-env traiter les messages de
    # déconnexion avant que le process ne se termine ou ne rouvre de
    # nouvelles connexions vers le même serveur.
    time.sleep(drain_seconds)


def _cleanup_all() -> None:
    """Ferme tous les envs enregistrés (appelé via atexit + gestionnaire signal)."""
    if _OPEN_ENVS:
        print("\n[cleanup] Fermeture des connexions poke-env...")
        _safe_close(*list(_OPEN_ENVS))  # copie : _safe_close modifie la liste


def _signal_handler(sig: int, frame: Any) -> None:
    """Gestionnaire SIGINT/SIGTERM : nettoie les envs puis quitte."""
    print(f"\n[signal] Signal {signal.Signals(sig).name} reçu — arrêt propre.")
    _cleanup_all()
    sys.exit(0)


# Enregistrement : nettoyage garanti même sur Ctrl+C ou kill
atexit.register(_cleanup_all)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# Réutilisation directe depuis train.py
from train import (
    ALGO_REGISTRY,
    MASKABLE_ALGOS,
    WandbMetricsCallback,
    EvalBestOnlyCallback,
    build_model,
    make_server_config,
    make_opponent,
    make_runtime_env,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

EXPERT_REGISTRY: dict[str, type] = {
    "heuristic": SimpleHeuristicsPlayer,
    "max_power": MaxBasePowerPlayer,
    "random": RandomPlayer,
}


def _make_expert(
    name: str,
    battle_format: str,
    server_cfg: ServerConfiguration,
) -> object:
    """Instancie un joueur expert (pour la collecte BC uniquement).

    Le joueur est créé avec ``start_listening=False`` : il n'est pas
    connecté au serveur, on utilise uniquement sa méthode ``choose_move``.
    """
    cls = EXPERT_REGISTRY.get(name, SimpleHeuristicsPlayer)
    return cls(
        battle_format=battle_format,
        server_configuration=server_cfg,
        start_listening=False,
    )


def _reward_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg.reward, resolve=True)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — DAgger (Dataset Aggregation)
# ─────────────────────────────────────────────────────────────────────────────

def _make_collect_env(cfg: DictConfig, opponent_name: str) -> Any:
    """Crée un environnement de collecte avec l'adversaire spécifié."""
    reward_fn = build_reward(_reward_dict(cfg))
    return make_runtime_env(cfg, opponent_name, reward_fn, use_pipeline=False)


def phase_dagger(
    cfg: DictConfig,
    model: Any,
    run_log_dir: Path,
    wandb_enabled: bool,
    eval_env: Any = None,
) -> dict:
    """Exécute la Phase 1 : collecte initiale (BC) puis itérations DAgger.

    Boucle DAgger :
        Round 0 : L'expert joue contre ``bc.opponent`` — données mises en cache.
        Round k : La **politique courante** joue, l'expert labelise les états.
                  L'adversaire cicle parmi ``bc.opponents``.
                  Le dataset est agrégé, puis le modèle est reentraîné.

    :return: Historique du dernier round d'entraînement.
    """
    bc_cfg = cfg.bc
    demos_path = Path(bc_cfg.demos_path)

    # Paramètres DAgger (rétrocompatibles)
    n_dagger_rounds    = int(getattr(bc_cfg, "n_dagger_rounds", 0))
    episodes_per_round = int(getattr(bc_cfg, "episodes_per_round", 1000))
    epochs_per_round   = int(getattr(bc_cfg, "epochs_per_round", 5))
    dagger_opponents   = list(getattr(bc_cfg, "opponents", ["heuristic", "max_power", "random"]))
    n_eval_eps         = int(getattr(bc_cfg, "eval_episodes", 50))
    es_patience        = int(getattr(bc_cfg, "early_stopping_patience", 0))
    es_min_delta       = float(getattr(bc_cfg, "early_stopping_min_delta", 1e-4))
    n_eval_eps        = int(getattr(bc_cfg, "eval_episodes", 50))

    # ── Round 0 : Collecte initiale (cache ou expert pur) ──
    print("\n" + "=" * 60)
    print("  Phase 1 — Round 0 : Collecte de démonstrations (expert)")
    print("=" * 60)

    if demos_path.exists():
        print(f"[BC] Chargement des démonstrations en cache depuis {demos_path}...")
        demonstrations = load_demonstrations(demos_path)
    else:
        print("[BC] Aucun cache trouvé. Début de la collecte...")
        server_cfg = make_server_config(cfg)

        collect_env = _make_collect_env(cfg, bc_cfg.opponent)
        expert = _make_expert(bc_cfg.expert, cfg.battle.format, server_cfg)
        collector = ExpertDemonstrationCollector(collect_env, expert)

        _OPEN_ENVS.append(collect_env)
        try:
            demonstrations = collector.collect(
                n_episodes=int(bc_cfg.n_episodes),
                verbose=True,
            )
        finally:
            _safe_close(collect_env)

        if bc_cfg.save_demos:
            save_demonstrations(demonstrations, demos_path)

    # ── Entraînement initial (BC, round 0) ──
    print("\n" + "=" * 60)
    print("  Phase 1 — Entraînement supervisé initial (BC)")
    print("=" * 60)

    history = train_bc(
        model,
        demonstrations,
        n_epochs=int(bc_cfg.n_epochs),
        batch_size=int(bc_cfg.batch_size),
        learning_rate=float(bc_cfg.learning_rate),
        verbose=True,
        wandb_module=wandb if wandb_enabled else None,
        eval_env=eval_env,
        n_eval_episodes=n_eval_eps,
        phase_label="bc",
        early_stopping_patience=es_patience,
        early_stopping_min_delta=es_min_delta,
    )

    # ── Rounds DAgger 1 .. N ──
    if n_dagger_rounds > 0:
        server_cfg = make_server_config(cfg)

        for round_idx in range(n_dagger_rounds):
            opponent_name = dagger_opponents[round_idx % len(dagger_opponents)]
            label = f"dagger"

            print("\n" + "=" * 60)
            print(f"  Phase 1 — DAgger round {round_idx + 1}/{n_dagger_rounds} "
                  f"(adversaire : {opponent_name})")
            print("=" * 60)

            collect_env = _make_collect_env(cfg, opponent_name)
            expert = _make_expert(bc_cfg.expert, cfg.battle.format, server_cfg)
            rollout_collector = PolicyRolloutCollector(collect_env, expert)

            _OPEN_ENVS.append(collect_env)
            try:
                new_data = rollout_collector.collect(
                    model,
                    n_episodes=episodes_per_round,
                    verbose=True,
                )
            finally:
                _safe_close(collect_env)

            # Agrégation du nouveau dataset
            demonstrations = aggregate_demonstrations(demonstrations, new_data)
            n_total = len(demonstrations["actions"])
            print(f"[DAgger] Dataset agrégé : {n_total} échantillons au total")

            # Wandb : signaler le changement de round
            if wandb_enabled and wandb is not None and wandb.run is not None:
                wandb.log(
                    {"dagger/round": round_idx + 1, "dagger/dataset_size": n_total},
                    step=model.num_timesteps,
                )

            # Re-entraînement sur dataset agrégé
            history = train_bc(
                model,
                demonstrations,
                n_epochs=epochs_per_round,
                batch_size=int(bc_cfg.batch_size),
                learning_rate=float(bc_cfg.learning_rate),
                verbose=True,
                wandb_module=wandb if wandb_enabled else None,
                eval_env=eval_env,
                n_eval_episodes=n_eval_eps,
                phase_label=label,
                early_stopping_patience=es_patience,
                early_stopping_min_delta=es_min_delta,
            )

    # Sauvegarder le modèle après tout le pré-entraînement (BC + DAgger)
    bc_model_path = run_log_dir / "model_after_bc"
    model.save(str(bc_model_path))
    rounds_label = f" + {n_dagger_rounds} rounds DAgger" if n_dagger_rounds > 0 else ""
    print(f"[BC{rounds_label}] Modèle pré-entraîné sauvegardé : {bc_model_path}.zip")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Reinforcement Learning
# ─────────────────────────────────────────────────────────────────────────────

def phase_rl(
    cfg: DictConfig,
    model: Any,
    env: Any,
    eval_env: Any,
    run_log_dir: Path,
    wandb_enabled: bool,
    run_name: str,
    timestamp: str,
):
    """Exécute la phase de fine-tuning RL (MaskablePPO)."""
    algo_name = cfg.algorithm.name
    model_dir = Path(cfg.training.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Callbacks ──
    eval_cb = EvalBestOnlyCallback(
        eval_env=eval_env,
        eval_freq=cfg.training.save_freq,
        n_eval_episodes=cfg.training.eval_episodes,
        best_model_path=model_dir / "best_model",
        deterministic=True,
        maskable=algo_name in MASKABLE_ALGOS,
        wandb_enabled=wandb_enabled,
    )
    callbacks: list[BaseCallback] = [
        eval_cb,
        WandbMetricsCallback(
            enabled=wandb_enabled,
            log_freq_steps=cfg.wandb.log_freq_steps,
        ),
    ]
    if wandb_enabled and WandbCallback is not None:
        callbacks.append(WandbCallback(verbose=0))
    callback = CallbackList(callbacks)

    # ── Lancement RL ──
    total_timesteps = int(cfg.training.total_timesteps)
    train_opponent_name = cfg.training.opponent
    eval_opponent_name = cfg.eval.opponent
    algo_cfg_key = "ppo" if algo_name in MASKABLE_ALGOS else "dqn"
    algo_section = cfg[algo_cfg_key]
    net_arch = list(algo_section.net_arch)

    print(f"\n{'='*60}")
    print(f"  Phase 2 — Fine-tuning RL ({algo_name})")
    print(f"  Format       : {cfg.battle.format}")
    print(f"  Timesteps    : {total_timesteps:,}")
    print(f"  Observation  : {env.observation_space}")
    print(f"  Actions      : {env.action_space.n}")
    print(f"  Récompense   : {cfg['reward']['class']}")
    print(f"  Net arch     : {net_arch}")
    print(f"  Action mask  : actif")
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
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\nInterruption — arrêt propre.")
    finally:
        best_path = model_dir / f"{run_name}_{timestamp}.zip"
        if best_path.exists():
            print(f"Meilleur modèle sauvegardé : {best_path}")
        else:
            print("Aucun modèle best n'a été sauvegardé.")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration des deux phases
# ─────────────────────────────────────────────────────────────────────────────

def train_bc_rl(cfg: DictConfig):
    """Lance l'entraînement complet : Phase 1 (BC) → Phase 2 (RL)."""

    algo_name = cfg.algorithm.name
    if algo_name not in MASKABLE_ALGOS:
        raise ValueError(
            f"Le Behavioral Cloning n'est supporté qu'avec les algorithmes "
            f"maskable ({list(MASKABLE_ALGOS)}), pas '{algo_name}'."
        )

    battle_format = cfg.battle.format
    use_pipeline = bool(getattr(cfg.ppo, "use_pipeline", False))

    # ── Dossiers ──
    model_dir = Path(cfg.training.model_dir)
    base_log_dir = Path(cfg.training.log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    base_log_dir.mkdir(parents=True, exist_ok=True)

    algo_cfg_key = "ppo" if algo_name in MASKABLE_ALGOS else "dqn"
    algo_section = cfg[algo_cfg_key]
    algo_lr = algo_section.learning_rate
    ent_coef = cfg.ppo.ent_coef
    train_opponent_name = cfg.training.opponent
    algo_label = f"BC+{algo_name}"
    run_name = f"{algo_label}_lr{algo_lr}_entcoef{ent_coef}_opp{train_opponent_name}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_log_dir = base_log_dir / f"{run_name}_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──
    wandb_enabled = bool(cfg.wandb.enabled)
    if wandb_enabled:
        if wandb is None:
            raise ImportError("wandb non installé — pip install wandb")
        if not cfg.wandb.project:
            raise ValueError("wandb.project manquant dans la config.")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=cfg.wandb.name or f"{run_name}_{timestamp}",
            tags=list(cfg.wandb.tags) + ["bc+rl"],
            notes=cfg.wandb.notes or "Entraînement BC → RL",
            mode=cfg.wandb.mode,
            save_code=bool(cfg.wandb.save_code),
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(run_log_dir),
            settings=wandb.Settings(quiet=True),
        )

    # ── Environnement d'entraînement RL ──
    reward_fn = build_reward(_reward_dict(cfg))
    env = make_runtime_env(
        cfg,
        train_opponent_name,
        reward_fn,
        use_pipeline=use_pipeline,
    )
    _OPEN_ENVS.append(env)

    # ── Environnement d'évaluation ──
    eval_reward_fn = build_reward(_reward_dict(cfg))
    eval_env = make_runtime_env(
        cfg,
        cfg.eval.opponent,
        eval_reward_fn,
        use_pipeline=use_pipeline,
    )
    _OPEN_ENVS.append(eval_env)

    # ── Modèle ──
    model = build_model(algo_name, env, cfg, run_log_dir, resume_path=None)

    print(f"\n{'='*60}")
    print(f"  PokeRL — Entraînement BC → RL")
    print(f"  Algorithme   : {algo_name}")
    print(f"  Format       : {battle_format}")
    print(f"  Expert BC    : {cfg.bc.expert}")
    print(f"  Épisodes BC  : {cfg.bc.n_episodes}")
    print(f"  Epochs BC    : {cfg.bc.n_epochs}")
    print(f"  Timesteps RL : {cfg.training.total_timesteps:,}")
    print(f"{'='*60}")

    try:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 1 — DAgger (BC initial + itérations Dataset Aggregation)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        bc_history = phase_dagger(cfg, model, run_log_dir, wandb_enabled, eval_env=eval_env)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 2 — Reinforcement Learning
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        phase_rl(
            cfg, model, env, eval_env,
            run_log_dir, wandb_enabled,
            run_name, timestamp,
        )
    finally:
        # Nettoyage : drain asyncio pour que poke-env envoie ses messages
        # de déconnexion avant que le process se termine ou soit relancé.
        _safe_close(env, eval_env)
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
    train_bc_rl(cfg)


if __name__ == "__main__":
    main()
