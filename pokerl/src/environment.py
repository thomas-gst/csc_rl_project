"""
environment.py — Environnement Gymnasium pour Pokémon Showdown.

Hérite de ``SinglesEnv`` (poke-env) et implémente :
    • ``embed_battle``  → délègue à ``features.py``
    • ``calc_reward``   → délègue à ``rewards.py``
    • ``action_masks``  → masque booléen pour MaskablePPO (sb3-contrib)

L'environnement est un **PettingZoo ParallelEnv** côté poke-env.
On l'enrobe ensuite d'un ``SingleAgentWrapper`` pour obtenir un ``gymnasium.Env``
compatible Stable Baselines 3.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from gymnasium.spaces import Box, Discrete

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.environment.singles_env import SinglesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player import Player, RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder.teambuilder import Teambuilder

from .features import embed_battle, OBSERVATION_SIZE
from .rewards import BaseReward, DenseReward


# ─────────────────────────────────────────────────────────────────────────────
# Environnement PettingZoo (2 agents)
# ─────────────────────────────────────────────────────────────────────────────
class PokeRLEnv(SinglesEnv[np.ndarray]):
    """Environnement RL pour combats Pokémon singles (gen 8 random battle).

    Utilise :
        - ``features.embed_battle`` pour construire l'observation,
        - un objet ``BaseReward`` pour le calcul de récompense,
        - ``action_masks()`` pour le masquage MaskablePPO.
    """

    def __init__(
        self,
        reward_fn: Optional[BaseReward] = None,
        *,
        account_configuration1: Optional[AccountConfiguration] = None,
        account_configuration2: Optional[AccountConfiguration] = None,
        avatar: Optional[int] = None,
        battle_format: str = "gen8randombattle",
        log_level: Optional[int] = None,
        save_replays: Union[bool, str] = False,
        server_configuration: Optional[ServerConfiguration] = LocalhostServerConfiguration,
        accept_open_team_sheet: Optional[bool] = False,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        open_timeout: Optional[float] = 10.0,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        challenge_timeout: Optional[float] = 60.0,
        team: Optional[Union[str, Teambuilder]] = None,
    ):
        super().__init__(
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            save_replays=save_replays,
            server_configuration=server_configuration,
            accept_open_team_sheet=accept_open_team_sheet,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            open_timeout=open_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            challenge_timeout=challenge_timeout,
            team=team,
            fake=False,
            strict=False,  # On ne veut pas crash sur action illégale → fallback random
        )

        self._reward_fn: BaseReward = reward_fn or DenseReward()

        # Observation space : vecteur continu de taille fixe
        self.observation_spaces = {
            agent: Box(
                low=-1.0,
                high=1.0,
                shape=(OBSERVATION_SIZE,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    # ── Interface poke-env ────────────────────────────────────────────────

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """Construit l'observation à partir de l'état de combat."""
        assert isinstance(battle, Battle)
        return embed_battle(battle)

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Calcule la récompense dense."""
        return self._reward_fn(battle)

    # ── Action masking ───────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Retourne un masque booléen (shape = action_space.n) indiquant
        quelles actions sont légales pour l'agent 1 (notre agent RL).

        Mapping rappel (gen8, act_size = 22) :
            0-5   → switch vers le pokémon du slot i
            6-9   → move 1-4
            10-13 → move 1-4 + mega
            14-17 → move 1-4 + z-move
            18-21 → move 1-4 + dynamax

        Compatible ``sb3_contrib.common.maskable.utils.get_action_masks``.
        """
        battle = self.battle1
        if battle is None or battle.finished:
            # Pas de combat en cours → tout est autorisé (sera ignoré)
            act_size = list(self.action_spaces.values())[0].n
            return np.ones(act_size, dtype=np.bool_)

        return self._build_action_mask(battle)

    @staticmethod
    def _build_action_mask(battle: Battle) -> np.ndarray:
        """Construit le masque pour un état de combat donné."""
        # Taille de l'espace d'action (gen8 → 22)
        # 6 switches + 4 moves × (1 normal + 1 mega + 1 z + 1 dmax) = 22
        act_size = 6 + 4 * 4  # 22 pour gen8
        mask = np.zeros(act_size, dtype=np.bool_)

        # ── Switches (actions 0 à 5) ──
        team_pokemon = list(battle.team.values())
        available_switch_species = {mon.species for mon in battle.available_switches}
        for i, mon in enumerate(team_pokemon):
            if mon.species in available_switch_species:
                mask[i] = True

        # ── Moves (actions 6 à 9) ──
        if battle.active_pokemon is not None:
            all_moves = list(battle.active_pokemon.moves.values())
            available_move_ids = {m.id for m in battle.available_moves}

            for i, move in enumerate(all_moves):
                if i >= 4:
                    break
                if move.id in available_move_ids:
                    # Move normal
                    mask[6 + i] = True

                    # Mega
                    if battle.can_mega_evolve:
                        mask[10 + i] = True

                    # Z-move
                    if battle.can_z_move and battle.active_pokemon.available_z_moves:
                        mask[14 + i] = True

                    # Dynamax
                    if battle.can_dynamax:
                        mask[18 + i] = True

        # Struggle / recharge — si aucun move dispo, seul struggle est "available"
        if battle.available_moves and battle.available_moves[0].id in ("struggle", "recharge"):
            mask[6] = True
            # Pas de gimmicks sur struggle
            mask[10:22] = False

        # Force switch → seuls les switches sont actifs (pas de moves)
        if battle.force_switch:
            mask[6:22] = False

        # S'assurer qu'au moins une action est vraie (fallback)
        if not mask.any():
            mask[6] = True  # struggle / default

        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper mono-agent avec action masking pour SB3
# ─────────────────────────────────────────────────────────────────────────────
class MaskableSingleAgentWrapper(SingleAgentWrapper):
    """``gymnasium.Env`` mono-agent qui expose ``action_masks()`` pour
    ``sb3_contrib.MaskablePPO``.

    Hérite du ``SingleAgentWrapper`` de poke-env et ajoute simplement
    la méthode ``action_masks()`` qui délègue à ``PokeRLEnv.action_masks()``.
    """

    def __init__(self, env: PokeRLEnv, opponent: Player):
        super().__init__(env, opponent)
        self._pokerl_env = env

    def action_masks(self) -> np.ndarray:
        """Retourne le masque d'action pour l'agent principal."""
        return self._pokerl_env.action_masks()


# ─────────────────────────────────────────────────────────────────────────────
# Factory : crée l'environnement prêt à l'emploi
# ─────────────────────────────────────────────────────────────────────────────
def make_env(
    reward_fn: Optional[BaseReward] = None,
    opponent: Optional[Player] = None,
    battle_format: str = "gen8randombattle",
    server_configuration: Optional[ServerConfiguration] = None,
) -> MaskableSingleAgentWrapper:
    """Crée un ``gymnasium.Env`` mono-agent avec action masking.

    :param reward_fn: Fonction de récompense (voir ``rewards.py``).
    :param opponent: Adversaire (``Player``). Par défaut ``RandomPlayer``.
    :param battle_format: Format de combat Showdown.
    :param server_configuration: Configuration du serveur.
    :return: Environnement Gymnasium prêt pour SB3 / MaskablePPO.
    """
    server_cfg = server_configuration or LocalhostServerConfiguration

    env = PokeRLEnv(
        reward_fn=reward_fn,
        battle_format=battle_format,
        server_configuration=server_cfg,
        start_listening=True,
    )

    if opponent is None:
        opponent = RandomPlayer(
            battle_format=battle_format,
            server_configuration=server_cfg,
        )

    return MaskableSingleAgentWrapper(env, opponent)
