#!/usr/bin/env python3
"""
diagnose.py — Script de diagnostic pour les "Invalid action" warnings dans PokeRL.

Stratégie :
  - Intercepte SinglesEnv.action_to_order pour capturer l'état exact
    du combat au moment où une action invalide est soumise.
  - Stocke un snapshot de l'état du combat au moment du calcul du masque
    (action_masks) pour comparer avec l'état au moment de l'exécution.
  - Utilise une politique aléatoire masquée (aucun modèle nécessaire).
  - Tourne en boucle jusqu'à N combats ou jusqu'à avoir accumulé assez
    d'erreurs pour les analyser.

Usage :
    python diagnose.py                  # 200 combats, config.yaml par défaut
    python diagnose.py --battles 500    # 500 combats
    python diagnose.py --log diag.log   # logs dans un fichier spécifique
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from poke_env.environment.singles_env import SinglesEnv
from poke_env.battle.battle import Battle
from poke_env.player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from src.environment import PokeRLEnv, MaskableSingleAgentWrapper, make_env
from src.rewards import DenseReward


# ─────────────────────────────────────────────────────────────────────────────
# Configuration du logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: str) -> logging.Logger:
    """Configure deux handlers : console (WARNING+) et fichier (DEBUG+)."""
    logger = logging.getLogger("diagnose")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S.%f",
    )

    # Console : uniquement les erreurs importantes
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Fichier : tout
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot de l'état d'un combat
# ─────────────────────────────────────────────────────────────────────────────

def battle_snapshot(battle: Optional[Battle], label: str = "") -> Dict[str, Any]:
    """Capture les champs clés du combat sous forme de dictionnaire sérialisable."""
    if battle is None:
        return {"error": "battle is None", "label": label}

    try:
        active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon

        # Moves disponibles
        avail_moves = [
            {"id": m.id, "type": str(m.type), "power": m.base_power}
            for m in (battle.available_moves or [])
        ]

        # Moves actifs du pokémon allié (ordre dict → même que action_to_order)
        active_moves_dict = []
        if active is not None:
            for i, (mid, mv) in enumerate(active.moves.items()):
                active_moves_dict.append({
                    "slot": i,
                    "id": mid,
                    "type": str(mv.type),
                    "power": mv.base_power,
                })

        # Switches disponibles
        avail_switches = [mon.species for mon in (battle.available_switches or [])]

        # Équipe (slot → espèce + HP)
        team_slots = [
            {
                "slot": i,
                "species": mon.species,
                "hp_frac": mon.current_hp_fraction,
                "fainted": mon.fainted,
                "is_active": (active is not None and mon.species == active.species),
            }
            for i, mon in enumerate(battle.team.values())
        ]

        # Valid orders
        try:
            valid_orders_str = [str(o) for o in battle.valid_orders]
        except Exception as e:
            valid_orders_str = [f"<error: {e}>"]

        return {
            "label": label,
            "battle_tag": battle.battle_tag,
            "turn": battle.turn,
            "force_switch": battle.force_switch,
            "finished": battle.finished,
            "active_pokemon": active.species if active else None,
            "active_status": str(active.status) if active else None,
            "active_fainted": active.fainted if active else None,
            "opponent_active": opp_active.species if opp_active else None,
            "available_moves": avail_moves,
            "active_moves_dict_order": active_moves_dict,
            "available_switches": avail_switches,
            "team_slots": team_slots,
            "valid_orders": valid_orders_str,
            "can_dynamax": battle.can_dynamax,
            "can_mega_evolve": battle.can_mega_evolve,
            "can_z_move": battle.can_z_move,
            "weather": {str(w): turns for w, turns in battle.weather.items()},
            "side_conditions": {str(sc): v for sc, v in battle.side_conditions.items()},
        }
    except Exception as exc:
        return {"label": label, "snapshot_error": str(exc), "tb": traceback.format_exc()}


def format_snapshot(snap: Dict[str, Any], indent: int = 4) -> str:
    """Formate un snapshot pour l'affichage dans les logs."""
    pad = " " * indent
    lines = [f"{pad}[Snapshot: {snap.get('label', '?')}]"]
    for k, v in snap.items():
        if k == "label":
            continue
        lines.append(f"{pad}  {k}: {v}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Compteur d'erreurs global
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiagStats:
    total_battles: int = 0
    total_steps: int = 0
    total_invalid_actions: int = 0
    invalid_action_reasons: Dict[str, int] = field(default_factory=dict)

    def record_invalid(self, reason: str):
        self.total_invalid_actions += 1
        self.invalid_action_reasons[reason] = (
            self.invalid_action_reasons.get(reason, 0) + 1
        )

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "RÉSUMÉ DIAGNOSTIC",
            f"  Combats    : {self.total_battles}",
            f"  Étapes     : {self.total_steps}",
            f"  Actions invalides : {self.total_invalid_actions}",
        ]
        if self.invalid_action_reasons:
            lines.append("  Répartition par cause :")
            for reason, cnt in sorted(
                self.invalid_action_reasons.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    [{cnt:4d}] {reason}")
        lines.append("=" * 60)
        return "\n".join(lines)


STATS = DiagStats()


# ─────────────────────────────────────────────────────────────────────────────
# Patch de action_to_order — cœur du diagnostic
# ─────────────────────────────────────────────────────────────────────────────

# On garde la référence vers la méthode originale
# En Python 3, accéder à un staticmethod via la classe retourne directement la fonction
_original_action_to_order = SinglesEnv.action_to_order


def _patched_action_to_order(
    action: np.int64, battle: Battle, fake: bool = False, strict: bool = True
):
    """Remplace SinglesEnv.action_to_order pour intercepter les vraies erreurs.

    Stratégie : on tente l'action avec strict=True dans un try/except.
    Si poke-env lève une ValueError, on a une VRAIE action invalide
    (pas un faux positif de heuristique) — on logue tout le contexte,
    puis on appelle avec strict=False pour le fallback normal.
    """
    diag_log = logging.getLogger("diagnose")

    # poke-env appelle action.item() → garantir np.int64
    action = np.int64(action)

    # ── Tenter l'action « strictement » pour détecter les vraies erreurs ──
    try:
        result = _original_action_to_order(action, battle, fake=fake, strict=True)
        # Pas d'erreur → retourner directement le résultat
        return result

    except ValueError as e:
        error_msg = str(e)

        # ── Identifier la cause à partir du message d'erreur et de l'état ──
        if "not in valid orders []" in error_msg or "valid orders []" in error_msg:
            reason = "valid_orders_empty"
        elif battle.force_switch and int(action) >= 6:
            reason = "move_during_force_switch"
        elif int(action) < 6:
            reason = "switch_not_in_valid_orders"
        elif "out of bounds" in error_msg:
            reason = "move_index_out_of_bounds"
        else:
            reason = "move_not_in_valid_orders"

        STATS.record_invalid(reason)

        # ── Snapshots ────────────────────────────────────────────────────
        snap_now = battle_snapshot(battle, label="at_action_to_order")
        stored = _LAST_MASK_SNAPSHOT.get(battle.battle_tag)

        try:
            valid_orders_str = [str(o) for o in battle.valid_orders]
        except Exception:
            valid_orders_str = ["<erreur lors du listage>"]

        # Moves du pokémon actif dans l'ordre du dict (= ordre utilisé par poke-env)
        active_moves_order = []
        if battle.active_pokemon:
            for i, (mid, mv) in enumerate(battle.active_pokemon.moves.items()):
                active_moves_order.append(f"[{i}] {mid}")

        # Quel slot le masque pensait cibler
        if int(action) < 6:
            target_desc = f"switch → team_slot[{action}] = {list(battle.team.values())[int(action)].species if int(action) < len(battle.team) else '?'}"
        else:
            move_idx = (int(action) - 6) % 4
            gimmick_idx = (int(action) - 6) // 4
            gimmick_name = ["normal", "mega", "z-move", "dynamax"][gimmick_idx] if gimmick_idx < 4 else "?"
            move_name = active_moves_order[move_idx] if move_idx < len(active_moves_order) else "?"
            target_desc = f"move_{gimmick_name} → move_slot[{move_idx}] = {move_name}"

        mask_info = ""
        if stored:
            dt = time.time() - stored["ts"]
            mask_info = (
                f"\n  ── Masque calculé {dt:.3f}s avant l'exécution ──\n"
                + format_snapshot(stored["snap"])
                + f"\n  Masque : {stored['mask']}"
            )

        diag_log.warning(
            f"\n{'═'*60}\n"
            f"[ACTION INVALIDE RÉELLE]\n"
            f"  Action soumise    : {action}  →  {target_desc}\n"
            f"  Cause identifiée  : {reason}\n"
            f"  Erreur poke-env   : {error_msg}\n"
            f"  force_switch      : {battle.force_switch}\n"
            f"  valid_orders ({len(valid_orders_str)}) : {valid_orders_str}\n"
            f"  available_moves   : {[m.id for m in (battle.available_moves or [])]}\n"
            f"  available_switches: {[m.species for m in (battle.available_switches or [])]}\n"
            f"  active_moves_dict : {active_moves_order}\n"
            f"  active_pokemon    : {battle.active_pokemon.species if battle.active_pokemon else 'None'}\n"
            f"  active_fainted    : {battle.active_pokemon.fainted if battle.active_pokemon else 'N/A'}\n"
            f"  team fainted      : {[m.species for m in battle.team.values() if m.fainted]}\n"
            f"  turn              : {battle.turn}\n"
            f"  finished          : {battle.finished}\n"
            + format_snapshot(snap_now)
            + mask_info
            + f"\n{'═'*60}"
        )

    # ── Fallback : laisser poke-env choisir un move aléatoire ────────────
    return _original_action_to_order(action, battle, fake=fake, strict=False)


# Applique le patch
SinglesEnv.action_to_order = staticmethod(_patched_action_to_order)


# ─────────────────────────────────────────────────────────────────────────────
# Stockage du dernier masque calculé (par tag de combat)
# ─────────────────────────────────────────────────────────────────────────────

_LAST_MASK_SNAPSHOT: Dict[str, Dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Environnement de diagnostic
# ─────────────────────────────────────────────────────────────────────────────

class DiagnosticPokeRLEnv(PokeRLEnv):
    """Enveloppe PokeRLEnv qui stocke un snapshot à chaque calcul de masque."""

    def action_masks(self) -> np.ndarray:
        mask = super().action_masks()

        battle = self.battle1
        if battle is not None and not battle.finished:
            _LAST_MASK_SNAPSHOT[battle.battle_tag] = {
                "snap": battle_snapshot(battle, label="at_action_masks"),
                "mask": mask.tolist(),
                "ts": time.time(),
            }

        return mask


class DiagnosticWrapper(MaskableSingleAgentWrapper):
    """Wrapper qui utilise DiagnosticPokeRLEnv et expose les mêmes méthodes."""

    def action_masks(self) -> np.ndarray:
        return self._pokerl_env.action_masks()


# ─────────────────────────────────────────────────────────────────────────────
# Politique aléatoire masquée (pas de modèle nécessaire)
# ─────────────────────────────────────────────────────────────────────────────

def random_masked_action(mask: np.ndarray) -> np.int64:
    """Choisit uniformément parmi les actions valides du masque.

    Retourne un ``np.int64`` (requis par poke-env qui appelle ``.item()``
    sur l'action dans ``action_to_order``).
    """
    valid = np.where(mask)[0]
    if len(valid) == 0:
        logging.getLogger("diagnose").error(
            "Masque entièrement vide — fallback action 6 !"
        )
        return np.int64(6)
    return np.int64(np.random.choice(valid))


# ─────────────────────────────────────────────────────────────────────────────
# Boucle de diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def make_diagnostic_env(
    battle_format: str, server_cfg: ServerConfiguration
) -> DiagnosticWrapper:
    """Crée l'environnement de diagnostic avec des noms uniques."""
    # Noms uniques pour éviter les conflits avec train.py ou d'autres sessions
    uid = uuid.uuid4().hex[:6]
    acc1 = AccountConfiguration(f"DiagAgent_{uid}", None)
    acc2 = AccountConfiguration(f"DiagOpp_{uid}", None)

    env = DiagnosticPokeRLEnv(
        reward_fn=DenseReward(),
        battle_format=battle_format,
        server_configuration=server_cfg,
        account_configuration1=acc1,
        account_configuration2=acc2,
        start_listening=True,
    )
    opponent = RandomPlayer(
        battle_format=battle_format,
        server_configuration=server_cfg,
        account_configuration=AccountConfiguration(f"DiagRnd_{uid}", None),
    )
    return DiagnosticWrapper(env, opponent)


def run_diagnostic(
    env: DiagnosticWrapper,
    n_battles: int,
    logger: logging.Logger,
    max_errors: int = 50,
):
    """Lance n_battles combats et collecte les diagnostics."""
    logger.info(f"Démarrage du diagnostic — {n_battles} combats cibles")
    logger.info(f"Observation space : {env.observation_space.shape}")
    logger.info(f"Action space      : {env.action_space.n}")

    battles_done = 0
    steps_total = 0

    while battles_done < n_battles:
        obs, info = env.reset()
        done = False
        steps = 0

        while not done:
            mask = env.action_masks()
            action = random_masked_action(mask)

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception as exc:
                logger.error(
                    f"Exception lors de env.step(action={action}) : {exc}\n"
                    + traceback.format_exc()
                )
                break

            done = terminated or truncated
            steps += 1
            steps_total += 1

            STATS.total_steps = steps_total

        battles_done += 1
        STATS.total_battles = battles_done

        if battles_done % 10 == 0:
            logger.info(
                f"  Bataille {battles_done}/{n_battles} | "
                f"Steps : {steps_total} | "
                f"Erreurs : {STATS.total_invalid_actions}"
            )

        if STATS.total_invalid_actions >= max_errors:
            logger.warning(
                f"Limite d'erreurs atteinte ({max_errors}) — arrêt anticipé."
            )
            break

    logger.info(STATS.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_server_config(cfg: dict) -> ServerConfiguration:
    host = cfg["server"]["host"]
    port = cfg["server"]["port"]
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        f"http://{host}:{port}/action.php?",
    )


def main():
    parser = argparse.ArgumentParser(
        description="PokeRL Diagnostic — détecte les causes des Invalid action warnings"
    )
    parser.add_argument(
        "--config", "-c",
        default=str(SCRIPT_DIR / "config.yaml"),
        help="Chemin vers config.yaml",
    )
    parser.add_argument(
        "--battles", "-n",
        type=int,
        default=200,
        help="Nombre de combats à jouer (défaut : 200)",
    )
    parser.add_argument(
        "--log", "-l",
        default=str(SCRIPT_DIR / "logs" / "diagnose.log"),
        help="Fichier de logs détaillés",
    )
    parser.add_argument(
        "--max-errors", "-e",
        type=int,
        default=50,
        help="Arrêter après N erreurs collectées (défaut : 50)",
    )
    args = parser.parse_args()

    # Logs
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.log)

    logger.info(f"Log détaillé → {args.log}")
    logger.info(f"Combats cibles : {args.battles}   Max erreurs : {args.max_errors}")

    # Config
    cfg = load_config(args.config)
    server_cfg = make_server_config(cfg)
    battle_format = cfg["battle"]["format"]

    logger.info(f"Serveur : {cfg['server']['host']}:{cfg['server']['port']}")
    logger.info(f"Format  : {battle_format}")

    # Env
    env = make_diagnostic_env(battle_format, server_cfg)

    try:
        run_diagnostic(env, args.battles, logger, max_errors=args.max_errors)
    except KeyboardInterrupt:
        logger.info("Interruption clavier.")
        logger.info(STATS.summary())
    finally:
        try:
            env.close()
        except Exception as e:
            logger.debug(f"env.close() ignoré : {e}")


if __name__ == "__main__":
    main()
