"""
rewards.py — Logique modulaire de calcul des récompenses.

Chaque classe de récompense hérite de ``BaseReward`` et implémente ``__call__``.
On peut permuter de stratégie de récompense juste en changeant ``reward.class``
dans ``config.yaml``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict
from weakref import WeakKeyDictionary

from poke_env.battle.abstract_battle import AbstractBattle
# ─────────────────────────────────────────────────────────────────────────────
# Classe abstraite
# ─────────────────────────────────────────────────────────────────────────────
class BaseReward(ABC):
    """Interface commune pour toutes les stratégies de récompense."""

    def __init__(self, **kwargs):
        # Buffer pour stocker la dernière valeur de chaque bataille
        # (permet le calcul en *différence* d'état).
        self._buffer: WeakKeyDictionary[AbstractBattle, float] = WeakKeyDictionary()

    @abstractmethod
    def _state_value(self, battle: AbstractBattle) -> float:
        """Calcule la *valeur absolue* de l'état courant (non différentielle)."""
        ...

    def __call__(self, battle: AbstractBattle) -> float:
        """Renvoie la récompense = variation de la valeur d'état depuis le
        dernier appel pour cette bataille."""
        current = self._state_value(battle)
        prev = self._buffer.get(battle, 0.0)
        self._buffer[battle] = current
        return current - prev


# ─────────────────────────────────────────────────────────────────────────────
# Récompense dense (par défaut)
# ─────────────────────────────────────────────────────────────────────────────
class DenseReward(BaseReward):
    """Récompense dense prenant en compte :
    - les PV restants (alliés vs adverses)
    - les K.O.
    - les statuts (brûlure, paralysie, …)
    - la victoire / défaite

    Paramètres configurables via ``config.yaml`` :
        fainted_value   — poids par pokémon mis K.O.
        hp_value        — poids des PV normalisés (0-1) par pokémon
        status_value    — poids par statut infligé / subi
        victory_value   — bonus / malus pour la fin du combat
    """

    def __init__(
        self,
        fainted_value: float = 2.0,
        hp_value: float = 1.0,
        status_value: float = 0.5,
        victory_value: float = 15.0,
        num_pokemon: int = 6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fainted_value = fainted_value
        self.hp_value = hp_value
        self.status_value = status_value
        self.victory_value = victory_value
        self.num_pokemon = num_pokemon

    def _state_value(self, battle: AbstractBattle) -> float:
        value = 0.0

        # ── Notre équipe ──
        for mon in battle.team.values():
            value += mon.current_hp_fraction * self.hp_value
            if mon.fainted:
                value -= self.fainted_value
            elif mon.status is not None:
                value -= self.status_value

        # Pokémon pas encore révélés → on part du principe qu'ils ont 100 % PV
        value += (self.num_pokemon - len(battle.team)) * self.hp_value

        # ── Équipe adverse ──
        for mon in battle.opponent_team.values():
            value -= mon.current_hp_fraction * self.hp_value
            if mon.fainted:
                value += self.fainted_value
            elif mon.status is not None:
                value += self.status_value

        value -= (self.num_pokemon - len(battle.opponent_team)) * self.hp_value

        # ── Issue du combat ──
        if battle.won:
            value += self.victory_value
        elif battle.lost:
            value -= self.victory_value

        return value


# ─────────────────────────────────────────────────────────────────────────────
# Récompense agressive (exemple d'extension)
# ─────────────────────────────────────────────────────────────────────────────
class AggressiveReward(DenseReward):
    """Variante qui récompense davantage les K.O. adverses et pénalise
    moins les PV perdus. Utile pour encourager le jeu offensif."""

    def __init__(self, **kwargs):
        kwargs.setdefault("fainted_value", 4.0)
        kwargs.setdefault("hp_value", 0.5)
        kwargs.setdefault("status_value", 0.25)
        kwargs.setdefault("victory_value", 15.0)
        super().__init__(**kwargs)
        
# ─────────────────────────────────────────────────────────────────────────────
# Récompense d'imitation du SimpleHeuristicsPlayer
# ─────────────────────────────────────────────────────────────────────────────
class HeuristicExpertReward(BaseReward):
    """Récompense d'imitation : +match_reward si l'agent choisit la même action
    que ``SimpleHeuristicsPlayer`` aurait choisie dans le même état.

    Fonctionnement :
        L'action de l'agent et la recommandation de l'expert sont capturées
        *avant* l'appel à ``env.step()`` via ``set_pre_step()``, qui est
        invoqué par ``MaskableSingleAgentWrapper.step()`` lorsqu'un oracle
        est configuré.  ``__call__`` lit ensuite ces valeurs stockées.

    Paramètres :
        match_reward  — récompense quand l'agent imite l'expert (défaut 1.0).
    """

    def __init__(self, match_reward: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.match_reward = float(match_reward)
        # battle → (agent_action, expert_action) avant chaque step
        self._pre_step: WeakKeyDictionary[AbstractBattle, tuple[int, int]] = WeakKeyDictionary()

    def set_pre_step(
        self,
        battle: AbstractBattle,
        agent_action: int,
        expert_action: int,
    ) -> None:
        """Appelé par ``MaskableSingleAgentWrapper.step()`` AVANT ``env.step()``.

        :param battle:        État de combat actuel (point de vue de l'agent).
        :param agent_action:  Action entière choisie par l'agent RL.
        :param expert_action: Action entière que l'oracle aurait choisie
                              (-2 = ordre par défaut / conversion impossible).
        """
        self._pre_step[battle] = (int(agent_action), int(expert_action))

    def _state_value(self, battle: AbstractBattle) -> float:
        # Non utilisé : on surcharge __call__ directement.
        return 0.0

    def __call__(self, battle: AbstractBattle) -> float:
        """Retourne match_reward si l'agent a imité l'expert, sinon 0."""
        data = self._pre_step.pop(battle, None)
        if data is None:
            return 0.0
        agent_action, expert_action = data
        # expert_action == -2 signifie que la conversion a échoué → pas de signal
        if expert_action == -2:
            return 0.0
        return self.match_reward if agent_action == expert_action else 0.0



# ─────────────────────────────────────────────────────────────────────────────
# Registre (pour instanciation dynamique depuis la config)
# ─────────────────────────────────────────────────────────────────────────────
REWARD_REGISTRY: Dict[str, type] = {
    "DenseReward": DenseReward,
    "AggressiveReward": AggressiveReward,
    "HeuristicExpertReward": HeuristicExpertReward,
}


def build_reward(cfg: dict) -> BaseReward:
    """Instancie la bonne classe de récompense à partir du bloc ``reward``
    de ``config.yaml``."""
    cls_name = cfg.get("class", "DenseReward")
    if cls_name not in REWARD_REGISTRY:
        raise ValueError(
            f"Reward class inconnue : {cls_name}. "
            f"Choix possibles : {list(REWARD_REGISTRY.keys())}"
        )
    # On passe tous les autres champs comme kwargs
    params = {k: v for k, v in cfg.items() if k != "class"}
    return REWARD_REGISTRY[cls_name](**params)
