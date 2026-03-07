# collect_bc_data.py

# Script de collecte de données en Behavioral Cloning
# on lance plein de combats a fond et on enregistre les données

import asyncio
import numpy as np
import logging
import os
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from poke_env.player import SimpleHeuristicsPlayer, RandomPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

from feature_extractor import TransformerFeatureExtractor

NUM_SERVERS = 15
PORTS = [8000 + i for i in range(NUM_SERVERS)]

class HarvestingPlayer(SimpleHeuristicsPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = TransformerFeatureExtractor(num_tokens=13, features_per_token=200)
        # THE FIX: Buffer turns by battle tag instead of flat lists
        self.battle_history = {} 

    def _reverse_engineer_action(self, battle_order, battle) -> int:
        action_idx = 0 
        target = getattr(battle_order, 'order', None)

        if hasattr(target, 'base_power'):  
            tera_offset = 16 if getattr(battle_order, 'terastallize', False) else 0
            for i, move in enumerate(list(battle.active_pokemon.moves.values())[:4]):
                if move.id == target.id:
                    action_idx = i + tera_offset
                    break
        elif hasattr(target, 'species'):  
            for i, mon in enumerate(list(battle.team.values())[:6]):
                if mon.species == target.species:
                    action_idx = 20 + i
                    break
        return action_idx

    def choose_move(self, battle):
        expert_order = super().choose_move(battle)
        try:
            matrix = self.extractor.extract(battle)
            action_idx = self._reverse_engineer_action(expert_order, battle)
            
            # THE FIX: Save to the specific battle's buffer
            if battle.battle_tag not in self.battle_history:
                self.battle_history[battle.battle_tag] = []
                
            self.battle_history[battle.battle_tag].append((matrix, action_idx))
        except Exception as e:
            pass
            
        return expert_order


async def monitor_progress(harvester, port: int, total_games: int):
    while harvester.n_finished_battles < total_games:
        print(f"[Port {port}] Progress: {harvester.n_finished_battles} / {total_games} games completed.", flush=True)
        await asyncio.sleep(15)

async def run_harvester_node_async(port: int, num_games: int, vs_heuristic: bool):
    server_config = ServerConfiguration(
        f'ws://localhost:{port}/showdown/websocket', 
        f'http://localhost:{port}/'
    )
    
    harvester = HarvestingPlayer(
        battle_format="gen9randombattle",
        max_concurrent_battles=10, 
        server_configuration=server_config, 
        log_level=logging.CRITICAL
    )
    
    opponent_type = "Heuristic" if vs_heuristic else "Random"
    print(f"[Port {port}] Started {num_games} games vs {opponent_type}...", flush=True)
    
    if vs_heuristic:
        opponent = SimpleHeuristicsPlayer(
            battle_format="gen9randombattle", max_concurrent_battles=15, 
            server_configuration=server_config, log_level=logging.CRITICAL
        )
    else:
        opponent = RandomPlayer(
            battle_format="gen9randombattle", max_concurrent_battles=15, 
            server_configuration=server_config, log_level=logging.CRITICAL
        )
        
    monitor_task = asyncio.create_task(monitor_progress(harvester, port, num_games))
    
    await harvester.battle_against(opponent, n_battles=num_games)
    monitor_task.cancel()
    
    # --- THE FIX: RETROACTIVE VALUE ASSIGNMENT ---
    all_obs, all_actions, all_values = [], [], []
    
    for battle_tag, history in harvester.battle_history.items():
        battle = harvester.battles.get(battle_tag)
        if battle is None: continue
            
        # Your specific env.py scale!
        if battle.won: game_value = 30.0
        elif battle.lost: game_value = -30.0
        else: continue 
            
        for obs, act in history:
            all_obs.append(obs)
            all_actions.append(act)
            all_values.append(game_value)
    
    np_obs = np.array(all_obs, dtype=np.float32)
    np_actions = np.array(all_actions, dtype=np.int64)
    np_values = np.array(all_values, dtype=np.float32)
    
    chunk_path = f"chunk_{port}.npz"
    np.savez_compressed(chunk_path, obs=np_obs, actions=np_actions, values=np_values)
    print(f"[Port {port}] Finished and saved chunk with {len(np_obs)} turns!", flush=True)

def run_harvester_worker(port: int, num_games: int, vs_heuristic: bool):
    try:
        asyncio.run(run_harvester_node_async(port, num_games, vs_heuristic))
    except Exception as e:
        print(f"CRITICAL ERROR on Port {port}: {e}", flush=True)

def main():
    multiprocessing.set_start_method('spawn', force=True)
    print(f"Initializing Multi-Core Harvester across {NUM_SERVERS} servers...", flush=True)
    
    games_per_port = 4000 
    
    with ProcessPoolExecutor(max_workers=len(PORTS)) as executor:
        futures = []
        for i, port in enumerate(PORTS):
            vs_heuristic = True 
            futures.append(executor.submit(run_harvester_worker, port, games_per_port, vs_heuristic))
        
        for future in futures:
            try: future.result() 
            except Exception as e: print(f"A worker process crashed: {e}", flush=True)

    aggregate_existing_chunks()

def aggregate_existing_chunks():
    print("\n--- Aggregating Existing Chunk Files ---", flush=True)
    chunk_files = glob.glob("chunk_*.npz")
    
    if not chunk_files:
        print("ERROR: No chunks found in the current directory.", flush=True)
        return
        
    print(f"Found {len(chunk_files)} chunk files. Merging...", flush=True)
    
    all_obs, all_actions, all_values = [], [], []
    
    for file in chunk_files:
        try:
            data = np.load(file)
            all_obs.append(data['obs'])
            all_actions.append(data['actions'])
            # THE FIX: Load the values array too!
            all_values.append(data['values'])
            print(f"Loaded {file} - Turns: {len(data['obs'])}", flush=True)
            os.remove(file) # Clean up as we go
        except Exception as e:
            print(f"Failed to load {file}: {e}", flush=True)
            
    final_obs = np.concatenate(all_obs, axis=0)
    final_actions = np.concatenate(all_actions, axis=0)
    final_values = np.concatenate(all_values, axis=0)
    
    print(f"\nTotal turns recorded: {len(final_obs)}", flush=True)
    print(f"Observation Matrix Shape: {final_obs.shape}", flush=True) 
    print(f"Value Array Shape: {final_values.shape}", flush=True)
    
    save_path = "expert_data_combined.npz"
    np.savez_compressed(save_path, obs=final_obs, actions=final_actions, values=final_values)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Successfully saved {size_mb:.2f} MB expert dataset to {save_path}!", flush=True)

if __name__ == "__main__":
    #main()
    aggregate_existing_chunks()