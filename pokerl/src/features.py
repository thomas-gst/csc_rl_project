"""
features.py — Extraction de l'observation (embed_battle).

Construit un vecteur NumPy plat à partir d'un objet ``Battle``.
Séparé de environment.py pour pouvoir grossir librement (météo, terrain,
move-set adverse, etc.) sans polluer la classe d'environnement.

──────────────────────────────────────────────────────────────────────────
Vue d'ensemble du vecteur d'observation  (taille totale : 490) :
  • Pokémon actif allié       →  HP, type, boosts, statut              (33)
  • Moves alliés (4 slots)    →  type, power, accuracy, catégorie, PP  (92 = 4×23)
  • Pokémon actif adverse     →  HP, type, boosts, statut              (33)
  • Moves adverses connus     →  type, power, accuracy, catégorie      (220 = 10×22, PP inconnus)
  • Équipe alliée (bench)     →  HP + statut pour chaque slot (×5)     (40)
  • Équipe adverse (bench)    →  HP + statut pour chaque slot (×5)     (40)
  • Side conditions (hazards) →  nos hazards + les leurs               (14)
  • Météo                     →  one-hot                                (8)
  • Terrain                   →  one-hot                                (6)
  • Gimmicks disponibles      →  dynamax, mega, tera, z-move            (4)
──────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
NUM_TYPES = 18  # Normal → Fairy (on exclut STELLAR et ???)
TYPE_LIST: List[PokemonType] = [
    PokemonType.NORMAL, PokemonType.FIRE, PokemonType.WATER,
    PokemonType.ELECTRIC, PokemonType.GRASS, PokemonType.ICE,
    PokemonType.FIGHTING, PokemonType.POISON, PokemonType.GROUND,
    PokemonType.FLYING, PokemonType.PSYCHIC, PokemonType.BUG,
    PokemonType.ROCK, PokemonType.GHOST, PokemonType.DRAGON,
    PokemonType.DARK, PokemonType.STEEL, PokemonType.FAIRY,
]

STAT_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
NUM_BOOSTS = len(STAT_KEYS)  # 7

STATUS_LIST: List[Optional[Status]] = [
    None, Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX,
]

WEATHER_LIST: List[Weather] = [
    Weather.SUNNYDAY, Weather.RAINDANCE, Weather.SANDSTORM,
    Weather.HAIL, Weather.SNOWSCAPE,
    Weather.DESOLATELAND, Weather.PRIMORDIALSEA, Weather.DELTASTREAM,
]

FIELD_LIST: List[Field] = [
    Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
    Field.TRICK_ROOM, Field.GRAVITY,
]

HAZARDS_LIST: List[SideCondition] = [
    SideCondition.STEALTH_ROCK,
    SideCondition.SPIKES,
    SideCondition.TOXIC_SPIKES,
    SideCondition.STICKY_WEB,
]

SCREENS_LIST: List[SideCondition] = [
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.AURORA_VEIL,
]

MAX_MOVES = 4
MAX_TEAM = 6
MAX_OPP_MOVES = 10  # on garde les 10 derniers moves adverses connus

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _type_one_hot(pokemon: Optional[Pokemon]) -> np.ndarray: # TAILLE 18
    """One-hot (2 pour double-type) d'un Pokémon."""
    vec = np.zeros(NUM_TYPES, dtype=np.float32)
    if pokemon is None:
        return vec
    for t in pokemon.types:
        if t is not None and t in TYPE_LIST:
            vec[TYPE_LIST.index(t)] = 1.0
    return vec


def _move_type_one_hot(move: Optional[Move]) -> np.ndarray: # TAILLE 18
    vec = np.zeros(NUM_TYPES, dtype=np.float32)
    if move is None:
        return vec
    if move.type is not None and move.type in TYPE_LIST:
        vec[TYPE_LIST.index(move.type)] = 1.0
    return vec


def _status_one_hot(status: Optional[Status]) -> np.ndarray: # TAILLE 7
    """One-hot du statut principal (brûlure, gel, etc.)."""
    vec = np.zeros(len(STATUS_LIST), dtype=np.float32)
    idx = STATUS_LIST.index(status) if status in STATUS_LIST else 0
    vec[idx] = 1.0
    return vec


def _boosts(pokemon: Optional[Pokemon]) -> np.ndarray: # TAILLE 7
    """Boosts normalisés entre -1 et 1 (un boost de ±6 → ±1)."""
    if pokemon is None:
        return np.zeros(NUM_BOOSTS, dtype=np.float32)
    return np.array(
        [pokemon.boosts.get(k, 0) / 6.0 for k in STAT_KEYS],
        dtype=np.float32,
    )


def _move_features(move: Optional[Move], include_pp: bool = False) -> np.ndarray:
    """Vecteur caractérisant un move : type (one-hot), power, accuracy, catégorie [+ PP].

    :param include_pp: Si True, ajoute 1 valeur PP normalisée (current_pp / max_pp).
                       Activé uniquement pour les moves alliés dont on connaît les PP.
                       Taille : 22 (sans PP) ou 23 (avec PP).
    """
    size = NUM_TYPES + 4 + (1 if include_pp else 0)
    if move is None:
        return np.zeros(size, dtype=np.float32)
    power = (move.base_power or 0) / 250.0
    accuracy = (move.accuracy if move.accuracy is not None else 100.0) / 100.0
    # Catégorie : physical / special (one-hot 2 bits, 00 = status)
    cat = np.zeros(2, dtype=np.float32)
    cat_name = str(move.category).lower()
    if "physical" in cat_name:
        cat[0] = 1.0
    elif "special" in cat_name:
        cat[1] = 1.0
    parts = [
        _move_type_one_hot(move),                       # 18
        np.array([power, accuracy], dtype=np.float32),  # 2
        cat,                                             # 2
    ]
    if include_pp:
        max_pp = move.max_pp or 1  # évite une division par zéro
        pp_fraction = np.clip(move.current_pp / max_pp, 0.0, 1.0)
        parts.append(np.array([pp_fraction], dtype=np.float32))  # 1
    return np.concatenate(parts)


def _pokemon_bench_features(mon: Optional[Pokemon]) -> np.ndarray: # TAILLE 8
    """Features réduites pour un Pokémon sur le banc (HP + statut)."""
    if mon is None:
        return np.zeros(1 + len(STATUS_LIST), dtype=np.float32)
    hp = np.array([mon.current_hp_fraction], dtype=np.float32)
    return np.concatenate([hp, _status_one_hot(mon.status)])


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────────────────────────────────────────

def _active_pokemon_features(pokemon: Optional[Pokemon], battle: Battle) -> np.ndarray:
    """Vecteur complet pour le Pokémon actif (allié ou adverse)."""
    if pokemon is None:
        # HP(1) + type(18) + boosts(7) + statut(7) = 33
        return np.zeros(1 + NUM_TYPES + NUM_BOOSTS + len(STATUS_LIST), dtype=np.float32)

    hp = np.array([pokemon.current_hp_fraction], dtype=np.float32)  # 1
    types = _type_one_hot(pokemon)  # 18
    boosts = _boosts(pokemon)  # 7
    status = _status_one_hot(pokemon.status)  # 7
    return np.concatenate([hp, types, boosts, status])  # 33


def _active_moves_features(battle: Battle) -> np.ndarray:
    """Features des 4 moves du Pokémon actif allié (slots fixes).

    Inclut les PP normalisés (current_pp / max_pp) car le serveur nous les
    communique via les requêtes. Taille par slot : 23 (type+power+acc+cat+PP).
    """
    parts: list[np.ndarray] = []
    allied_move_size = NUM_TYPES + 5  # 23 (avec PP)

    if battle.active_pokemon is not None:
        all_moves = list(battle.active_pokemon.moves.values())
    else:
        all_moves = []

    for i in range(MAX_MOVES):
        if i < len(all_moves):
            parts.append(_move_features(all_moves[i], include_pp=True))
        else:
            parts.append(np.zeros(allied_move_size, dtype=np.float32))
    return np.concatenate(parts)  # 4 × 23 = 92


def _opponent_known_moves(battle: Battle) -> np.ndarray:
    """Encode les moves connus de l'adversaire (jusqu'à MAX_OPP_MOVES moves).

    Les PP adverses sont inconnus (le serveur ne les communique pas).
    Taille par slot : 22 (pas de PP).
    """
    opp_move_size = NUM_TYPES + 4  # 22 (sans PP)
    opp = battle.opponent_active_pokemon
    parts: list[np.ndarray] = []

    known: list[Move] = []
    if opp is not None:
        known = list(opp.moves.values())

    for i in range(MAX_OPP_MOVES):
        if i < len(known):
            parts.append(_move_features(known[i], include_pp=False))
        else:
            parts.append(np.zeros(opp_move_size, dtype=np.float32))
    return np.concatenate(parts)  # 10 × 22 = 220


def _side_conditions_features(
    our_sc: dict, opp_sc: dict
) -> np.ndarray:
    """Encode les entry hazards + écrans pour les deux côtés."""
    parts: list[float] = []

    # Hazards — valeur = nombre de couches (spikes 0-3, toxic_spikes 0-2, etc.)
    for sc in HAZARDS_LIST:
        parts.append(float(our_sc.get(sc, 0)))
    for sc in SCREENS_LIST:
        parts.append(1.0 if sc in our_sc else 0.0)

    for sc in HAZARDS_LIST:
        parts.append(float(opp_sc.get(sc, 0)))
    for sc in SCREENS_LIST:
        parts.append(1.0 if sc in opp_sc else 0.0)

    return np.array(parts, dtype=np.float32)  # 2 × (4 + 3) = 14
"""Indices 0-3  : nos hazards    [SR_couches, Spikes_couches, ToxicSpikes_couches, StickyWeb]
Indices 4-6  : nos écrans     [Reflect, LightScreen, AuroraVeil]
Indices 7-10 : leurs hazards  [SR_couches, Spikes_couches, ToxicSpikes_couches, StickyWeb]
Indices 11-13: leurs écrans   [Reflect, LightScreen, AuroraVeil]
exemple : [0, 0, 0, 0,   1, 0, 0,    1, 2, 0, 0,   0, 0, 0]"""


def _weather_features(battle: Battle) -> np.ndarray:
    """One-hot de la météo active."""
    vec = np.zeros(len(WEATHER_LIST), dtype=np.float32)
    for w in WEATHER_LIST:
        if w in battle.weather:
            vec[WEATHER_LIST.index(w)] = 1.0
    return vec  # 8


def _field_features(battle: Battle) -> np.ndarray:
    """One-hot des terrains / effets de champ actifs."""
    vec = np.zeros(len(FIELD_LIST), dtype=np.float32)
    for f in FIELD_LIST:
        if f in battle.fields:
            vec[FIELD_LIST.index(f)] = 1.0
    return vec  # 6


def _bench_features(battle: Battle, ours: bool = True) -> np.ndarray:
    """Features des pokémon sur le banc (max 5 slots, hors actif)."""
    bench_size = 1 + len(STATUS_LIST)  # 8 per slot
    team = battle.team if ours else battle.opponent_team
    active = battle.active_pokemon if ours else battle.opponent_active_pokemon

    bench_mons = [
        mon for mon in team.values()
        if mon != active
    ]

    parts: list[np.ndarray] = []
    for i in range(MAX_TEAM - 1):  # 5 slots
        if i < len(bench_mons):
            parts.append(_pokemon_bench_features(bench_mons[i]))
        else:
            parts.append(np.zeros(bench_size, dtype=np.float32))
    return np.concatenate(parts)  # 5 × 8 = 40


def _gimmick_features(battle: Battle) -> np.ndarray:
    """4 bits : [can_dynamax, can_mega_evolve, can_tera, can_z_move]."""
    return np.array([
        float(battle.can_dynamax),
        float(battle.can_mega_evolve),
        float(battle.can_tera is not False and battle.can_tera is not None),
        float(battle.can_z_move),
    ], dtype=np.float32)  # 4


# ─────────────────────────────────────────────────────────────────────────────
# Assemblage final
# ─────────────────────────────────────────────────────────────────────────────

def embed_battle(battle: Battle) -> np.ndarray:
    """Produit le vecteur d'observation complet pour un état de combat.

    Retourne un ``np.ndarray`` de shape ``(OBSERVATION_SIZE,)`` avec des
    valeurs float32.
    """
    parts = [
        # Pokémon actif allié
        _active_pokemon_features(battle.active_pokemon, battle),  # 33
        # Moves du pokémon actif allié
        _active_moves_features(battle),  # 88
        # Pokémon actif adverse
        _active_pokemon_features(battle.opponent_active_pokemon, battle),  # 33
        # Moves connus de l'adversaire
        _opponent_known_moves(battle),  # 220
        # Banc allié
        _bench_features(battle, ours=True),  # 40
        # Banc adverse
        _bench_features(battle, ours=False),  # 40
        # Side conditions (hazards + écrans)
        _side_conditions_features(battle.side_conditions, battle.opponent_side_conditions),  # 14
        # Météo
        _weather_features(battle),  # 8
        # Terrain
        _field_features(battle),  # 6
        # Gimmicks
        _gimmick_features(battle),  # 4
    ]
    return np.concatenate(parts)


# Taille totale du vecteur d'observation
# 33 + 92 + 33 + 220 + 40 + 40 + 14 + 8 + 6 + 4 = 490
#                ↑
#        moves alliés : 4 × 23 = 92  (type+power+acc+cat+PP)
#        moves adverses : 10 × 22 = 220 (PP inconnus, non inclus)
OBSERVATION_SIZE: int = (
    (1 + NUM_TYPES + NUM_BOOSTS + len(STATUS_LIST))        # active allié   = 33
    + MAX_MOVES * (NUM_TYPES + 5)                          # moves alliés   = 92  (avec PP)
    + (1 + NUM_TYPES + NUM_BOOSTS + len(STATUS_LIST))      # active adverse  = 33
    + MAX_OPP_MOVES * (NUM_TYPES + 4)                      # moves adverses = 220 (sans PP)
    + (MAX_TEAM - 1) * (1 + len(STATUS_LIST))              # banc allié     = 40
    + (MAX_TEAM - 1) * (1 + len(STATUS_LIST))              # banc adverse   = 40
    + 2 * (len(HAZARDS_LIST) + len(SCREENS_LIST))          # side conds     = 14
    + len(WEATHER_LIST)                                    # météo          = 8
    + len(FIELD_LIST)                                      # terrain        = 6
    + 4                                                    # gimmicks       = 4
)
# OBSERVATION_SIZE == 490
