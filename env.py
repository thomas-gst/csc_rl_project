# env.py

import os
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1" 
os.environ["RAY_CPP_LOG_LEVEL"] = "3" 
import logging
import torch

import numpy as np
import numpy.typing as npt
from typing import Any, Dict
from gymnasium.spaces import Box, Discrete, Dict as GymDict
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.core.columns import Columns

from poke_env.battle import AbstractBattle
from poke_env.environment import SingleAgentWrapper, SinglesEnv
from poke_env.player import Player, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.player.battle_order import BattleOrder

from feature_extractor import TransformerFeatureExtractor
from model import PokeTransformerModule

PORTS = [8000 + i for i in range(5)]

# --- THE NEW OPPONENT: PLAYS USING YOUR PYTORCH MODEL ---
class SelfPlayOpponent(Player):
    def __init__(self, model_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cpu") # Keep opponent on CPU to save GPU RAM
        self.extractor = TransformerFeatureExtractor(num_tokens=13, features_per_token=200)
        
        obs_space = GymDict({
            "observations": Box(low=-10.0, high=np.inf, shape=(13, 200), dtype=np.float32),
            "action_mask": Box(0.0, 1.0, shape=(26,), dtype=np.float32),
        })
        act_space = Discrete(26)
        
        self.model = PokeTransformerModule(
            observation_space=obs_space, action_space=act_space,
            inference_only=True, model_config={}, catalog_class=None,
        ).to(self.device)
        
        print(f"[Self-Play Bot] Booting up with weights from {model_path}...")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.model.eval()

    def _get_action_mask_and_mapping(self, battle):
        mask = np.zeros(26, dtype=np.float32)
        idx_to_order = {}
        if battle.available_moves:
            active_moves = list(battle.active_pokemon.moves.values())
            for move in battle.available_moves:
                try: move_idx = active_moves.index(move)
                except ValueError: move_idx = 0
                mask[move_idx] = 1.0
                idx_to_order[move_idx] = self.create_order(move)
                if getattr(battle, 'can_tera', False):
                    mask[move_idx + 16] = 1.0
                    idx_to_order[move_idx + 16] = self.create_order(move, terastallize=True)
                    
        if battle.available_switches:
            team = list(battle.team.values())
            for switch in battle.available_switches:
                try: switch_idx = team.index(switch)
                except ValueError: switch_idx = 0
                mask[20 + switch_idx] = 1.0
                idx_to_order[20 + switch_idx] = self.create_order(switch)
        return mask, idx_to_order

    def choose_move(self, battle):
        try:
            obs_matrix = self.extractor.extract(battle)
            mask, mapping = self._get_action_mask_and_mapping(battle)
        except Exception:
            return self.choose_random_move(battle)

        obs_tensor = torch.tensor(obs_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)
        batch_dict = {Columns.OBS: {"observations": obs_tensor, "action_mask": mask_tensor}}
        
        with torch.no_grad():
            out = self.model._forward(batch_dict)
            logits = out[Columns.ACTION_DIST_INPUTS]
            best_action_idx = torch.argmax(logits, dim=-1).item()
            
        return mapping.get(best_action_idx, self.choose_random_move(battle))


class PokeTransformerEnv(SinglesEnv[npt.NDArray[np.float32]]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.extractor = TransformerFeatureExtractor(num_tokens=13, features_per_token=200)
        self.observation_spaces = {
            agent: GymDict({
                "observations": Box(low=-10.0, high=np.inf, shape=(13, 200), dtype=np.float32),
                "action_mask": Box(0.0, 1.0, shape=(26,), dtype=np.float32),
            }) for agent in self.possible_agents
        }
    
    @classmethod
    def create_single_agent_env(cls, config: Dict[str, Any]) -> SingleAgentWrapper:
        worker_idx = getattr(config, "worker_index", 0)
        assigned_port = PORTS[worker_idx % len(PORTS)]

        server_config = ServerConfiguration(
            f'ws://localhost:{assigned_port}/showdown/websocket', 
            f'http://localhost:{assigned_port}/'
        )

        env = cls(battle_format=config["battle_format"], log_level=logging.CRITICAL, strict=False, server_configuration=server_config)
        
        # --- THE FIX: Spawn the Self-Play Opponent instead of the Heuristic Bot ---
        pretrained_path = config.get("pretrained_weights", "pretrained_pokeformer_final.pt")
        opponent = SelfPlayOpponent(
            model_path=pretrained_path,
            start_listening=False, 
            server_configuration=server_config, 
            log_level=logging.CRITICAL
        )
        return SingleAgentWrapper(env, opponent)

    def action_to_order(self, action: int, battle: AbstractBattle, **kwargs) -> BattleOrder:
        try:
            return super().action_to_order(action, battle, **kwargs)
        except (KeyError, ValueError, AssertionError, TypeError):
            if battle.available_moves: return BattleOrder(battle.available_moves[0])
            elif battle.available_switches: return BattleOrder(battle.available_switches[0])
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