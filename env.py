# env.py

# setup des environnement d'entrainement 

from typing import Any, Dict
import numpy as np
import os
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1" 
os.environ["RAY_CPP_LOG_LEVEL"] = "3" # Only show FATAL C++ errors
import logging

import numpy.typing as npt
from gymnasium.spaces import Box, Dict as GymDict
from ray.rllib.env import ParallelPettingZooEnv

from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.battle_order import BattleOrder

from feature_extractor import TransformerFeatureExtractor

PORTS = [8000 + i for i in range(5)]

class PokeTransformerEnv(SinglesEnv[npt.NDArray[np.float32]]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extractor = TransformerFeatureExtractor(num_tokens=13, features_per_token=200)
        
        self.observation_spaces = {
            agent: GymDict({
                "observations": Box(low=-10.0, high=np.inf, shape=(13, 200), dtype=np.float32),
                "action_mask": Box(0.0, 1.0, shape=(26,), dtype=np.float32),
            })
            for agent in self.possible_agents
        }
    # c'est ici qu'on crée plusieurs environnements et que j'assigne a chaque port
    
    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        worker_idx = getattr(config, "worker_index", 0)
        ports = PORTS
        assigned_port = ports[worker_idx % len(ports)]

        server_config = ServerConfiguration(
            f'ws://localhost:{assigned_port}/showdown/websocket', 
            f'http://localhost:{assigned_port}/'
        )

        env = cls(battle_format=config["battle_format"], log_level=logging.CRITICAL, strict=False, server_configuration=server_config)
        opponent = SimpleHeuristicsPlayer(start_listening=False, server_configuration=server_config, log_level=logging.CRITICAL)
        return SingleAgentWrapper(env, opponent)

    def action_to_order(self, action: int, battle: AbstractBattle, **kwargs) -> BattleOrder:
        """Catches poke-env kwargs (like fake=True) and prevents RLlib crashes."""
        try:
            return super().action_to_order(action, battle, **kwargs)
        except (KeyError, ValueError, AssertionError, TypeError):
            if battle.available_moves:
                return BattleOrder(battle.available_moves[0])
            elif battle.available_switches:
                return BattleOrder(battle.available_switches[0])
            return BattleOrder("random")

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0)

    def create_action_mask(self, battle: AbstractBattle) -> np.ndarray:
        mask = np.zeros(26, dtype=np.float32)
        active_mon = battle.active_pokemon
        if not active_mon:
            mask[0] = 1.0 
            return mask

        is_reviving = getattr(battle, 'reviving', False)
        force_switch = getattr(battle, 'force_switch', False)
        avail_moves = battle.available_moves
        avail_species = {p.species for p in battle.available_switches}

        if not force_switch and not is_reviving:
            for i, move in enumerate(list(active_mon.moves.values())[:4]):
                if move in avail_moves:
                    mask[i] = 1.0
                    if getattr(battle, 'can_tera', False):
                        mask[i + 16] = 1.0
            if not mask[:4].any() and avail_moves:
                mask[0] = 1.0 

        is_trapped = getattr(battle, 'trapped', False) or getattr(battle, 'maybe_trapped', False)
        can_switch = force_switch or is_reviving or not is_trapped
        
        if can_switch or avail_species:
            for i, mon in enumerate(list(battle.team.values())[:6]):
                if mon.species in avail_species:
                    if is_reviving:
                        if mon.fainted: mask[20 + i] = 1.0
                    else:
                        if not mon.fainted and not mon.active: mask[20 + i] = 1.0

        if not mask.any(): mask[0] = 1.0
        return mask

    def embed_battle(self, battle: AbstractBattle):
        return {
            "observations": self.extractor.extract(battle),
            "action_mask": self.create_action_mask(battle)
        }