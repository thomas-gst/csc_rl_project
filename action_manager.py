# action_manager.py
import numpy as np
from poke_env.battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

class ActionManager:
    """Handles all legal move logic, masking, and RLlib failsafes."""
    
    def __init__(self, action_space_size: int = 26):
        self.action_space_size = action_space_size

    def get_action_mask(self, battle: AbstractBattle) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        active_mon = battle.active_pokemon

        if not active_mon:
            mask[0] = 1.0 
            return mask

        is_reviving = getattr(battle, 'reviving', False)
        force_switch = getattr(battle, 'force_switch', False)
        avail_moves = battle.available_moves
        avail_species = {p.species for p in battle.available_switches}

        # 1. Attack & Tera (0-3, 16-19)
        if not force_switch and not is_reviving:
            for i, move in enumerate(list(active_mon.moves.values())[:4]):
                if move in avail_moves:
                    mask[i] = 1.0
                    if getattr(battle, 'can_tera', False):
                        mask[i + 16] = 1.0
            if not mask[:4].any() and avail_moves:
                mask[0] = 1.0 

        # 2. Switches (20-25)
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

    def get_failsafe_order(self, battle: AbstractBattle) -> BattleOrder:
        """If RLlib hallucinates a bad move, this safely catches it."""
        if battle.available_moves:
            return BattleOrder(battle.available_moves[0])
        elif battle.available_switches:
            return BattleOrder(battle.available_switches[0])
        return BattleOrder("random")