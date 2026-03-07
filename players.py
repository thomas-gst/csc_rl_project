# OLD USELESS
from poke_env.player import Player
from poke_env.player import Gen9EnvSinglePlayer
from gymnnasium.spaces import Box
import numpy as np

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # Chooses a move with the highest base power when possible
        if battle.available_moves:
            # Iterating over available moves to find the one with the highest base power
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            # Creating an order for the selected move
            return self.create_order(best_move)
        else:
            # If no attacking move is available, perform a random switch
            # This involves choosing a random move, which could be a switch or another available action
            return self.choose_random_move(battle)


class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize any additional attributes for the RL player here
        low = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([3, 3, 3, 3, 4, 4, 4, 4, 1, 1], dtype=np.float32)

        self._observation_space = Box(low=low, high=high, dtype=np.float32)
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return super().action_space

    def embed_battle(self, battle):
        moves_base_power = -np.ones(4, dtype=np.float32)
        moves_dmg_multiplier = np.ones(4, dtype=np.float32)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power/100 if move.base_power is not None else -1
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
        
        hp_percentage = battle.active_pokemon.current_hp / battle.active_pokemon.max_hp
        opponent_hp_percentage = battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp

        return np.concatenate((moves_base_power, moves_dmg_multiplier, [hp_percentage, opponent_hp_percentage])).astype(np.float32)

        def compute_reward(self, battle) ->float:
            return self.reward_computing_helper(battle, fainted_value =2.0, hp_value=1.0, victory_value=30.0)

        