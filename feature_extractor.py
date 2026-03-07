# feature_extractor.py

#self explanatory, normalement ca marche
import json
import numpy as np
from poke_env.battle import AbstractBattle

# Strict orderings for one-hot matrices
TYPES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]
STATUSES = ["brn", "par", "slp", "frz", "psn", "tox"]

class TransformerFeatureExtractor:
    def __init__(self, num_tokens: int = 13, features_per_token: int = 200):
        self.num_tokens = num_tokens
        self.features_per_token = features_per_token
        
        # Build ultra-fast O(1) lookup dictionaries from the JSON lists
        self.vocab_maps = {}
        try:
            with open("vocab.json", "r") as f:
                raw_vocab = json.load(f)
                for category, items_list in raw_vocab.items():
                    # Map the string to its index + 1
                    self.vocab_maps[category] = {str(item).lower(): idx + 1 for idx, item in enumerate(items_list)}
        except FileNotFoundError:
            print("Warning: vocab.json not found.")

    def _safe_get_id(self, category: str, name: str) -> int:
        if not name: return 0
        clean_name = str(name).lower().replace(" ", "").replace("-", "")
        return self.vocab_maps.get(category, {}).get(clean_name, 0)

    def _get_field_features(self, battle: AbstractBattle) -> np.ndarray:
        features = np.zeros(self.features_per_token, dtype=np.float32)
        idx = 7 # Indices 0-6 are reserved for IDs (left as 0.0 for the Field token)
        
        features[idx] = battle.turn / 100.0; idx += 1
        
        w_strs = [str(w).lower() for w in battle.weather.keys()]
        features[idx] = 1.0 if any('sun' in w or 'desolateland' in w for w in w_strs) else 0.0; idx += 1
        features[idx] = 1.0 if any('rain' in w or 'primordialsea' in w for w in w_strs) else 0.0; idx += 1
        features[idx] = 1.0 if any('sand' in w for w in w_strs) else 0.0; idx += 1
        features[idx] = 1.0 if any('snow' in w or 'hail' in w for w in w_strs) else 0.0; idx += 1
        
        t_strs = [str(f).lower() for f in battle.fields.keys()]
        features[idx] = 1.0 if any('electric' in f for f in t_strs) else 0.0; idx += 1
        features[idx] = 1.0 if any('grassy' in f for f in t_strs) else 0.0; idx += 1
        features[idx] = 1.0 if any('misty' in f for f in t_strs) else 0.0; idx += 1
        features[idx] = 1.0 if any('psychic' in f for f in t_strs) else 0.0; idx += 1
        
        asc = [str(c).lower() for c in battle.side_conditions.keys()]
        features[idx] = 1.0 if any('stealthrock' in c for c in asc) else 0.0; idx += 1
        features[idx] = 1.0 if any('spikes' in c and 'toxic' not in c for c in asc) else 0.0; idx += 1
        features[idx] = 1.0 if any('toxicspikes' in c for c in asc) else 0.0; idx += 1
        
        esc = [str(c).lower() for c in battle.opponent_side_conditions.keys()]
        features[idx] = 1.0 if any('stealthrock' in c for c in esc) else 0.0; idx += 1
        features[idx] = 1.0 if any('spikes' in c and 'toxic' not in c for c in esc) else 0.0; idx += 1
        features[idx] = 1.0 if any('toxicspikes' in c for c in esc) else 0.0; idx += 1

        return features

    def _encode_pokemon(self, mon, is_active: bool) -> np.ndarray:
        features = np.zeros(self.features_per_token, dtype=np.float32)
        if mon is None: return features
            
        # --- 7 CATEGORICAL IDs ---
        features[0] = self._safe_get_id("pokemon.species", mon.species)
        features[1] = self._safe_get_id("pokemon.item", mon.item)
        features[2] = self._safe_get_id("pokemon.ability", mon.ability)
        
        moves = list(mon.moves.values())[:4]
        for i, move in enumerate(moves):
            features[3 + i] = self._safe_get_id("move.id", move.id)

        # --- CONTINUOUS SCALARS ---
        idx = 7
        features[idx] = 1.0 if is_active else 0.0; idx += 1
        features[idx] = 1.0 if mon.fainted else 0.0; idx += 1
        features[idx] = mon.current_hp_fraction; idx += 1
        
        features[idx] = mon.base_stats.get('atk', 0) / 150.0; idx += 1
        features[idx] = mon.base_stats.get('def', 0) / 150.0; idx += 1
        features[idx] = mon.base_stats.get('spa', 0) / 150.0; idx += 1
        features[idx] = mon.base_stats.get('spd', 0) / 150.0; idx += 1
        features[idx] = mon.base_stats.get('spe', 0) / 150.0; idx += 1
        features[idx] = getattr(mon, 'weight', 0) / 100.0; idx += 1
        
        if mon.status:
            st = str(mon.status.name).lower()
            if st in STATUSES: features[idx + STATUSES.index(st)] = 1.0
        idx += len(STATUSES)
        
        boosts = mon.boosts if mon.boosts else {}
        features[idx] = boosts.get('atk', 0) / 6.0; idx += 1
        features[idx] = boosts.get('def', 0) / 6.0; idx += 1
        features[idx] = boosts.get('spa', 0) / 6.0; idx += 1
        features[idx] = boosts.get('spe', 0) / 6.0; idx += 1
        
        if mon.type_1:
            t1 = str(mon.type_1.name).lower()
            if t1 in TYPES: features[idx + TYPES.index(t1)] = 1.0
        if mon.type_2:
            t2 = str(mon.type_2.name).lower()
            if t2 in TYPES: features[idx + TYPES.index(t2)] = 1.0
        idx += len(TYPES)
        
        features[idx] = 1.0 if getattr(mon, 'is_terastallized', False) else 0.0; idx += 1

        for move in moves:
            features[idx] = move.base_power / 150.0; idx += 1
            features[idx] = 1.0 if move.accuracy is True else (move.accuracy / 100.0 if move.accuracy else 0.0); idx += 1
            if move.type:
                mt = str(move.type.name).lower()
                if mt in TYPES: features[idx + TYPES.index(mt)] = 1.0
            idx += len(TYPES)

        return features

    def extract(self, battle: AbstractBattle) -> np.ndarray:
        matrix = np.zeros((self.num_tokens, self.features_per_token), dtype=np.float32)
        matrix[0] = self._get_field_features(battle)

        for i, mon in enumerate(battle.team.values()):
            if i >= 6: break 
            matrix[1 + i] = self._encode_pokemon(mon, mon == battle.active_pokemon)

        for i, mon in enumerate(battle.opponent_team.values()):
            if i >= 6: break
            matrix[7 + i] = self._encode_pokemon(mon, mon == battle.opponent_active_pokemon)

        return matrix