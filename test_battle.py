# test.py
import os
import torch
import config
from env import ExampleEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.core import Columns

def watch_bot_play():
    # Register the environment
    register_env("showdown", ExampleEnv.create_single_agent_env)
    
    save_path = os.path.abspath("poke_model_checkpoint")
    print(f"Loading the trained brain from: {save_path}")
    
    # 1. Load the algorithm
    algo = Algorithm.from_checkpoint(save_path)
    
    # NEW: Extract the actual PyTorch Neural Network (RLModule)
    rl_module = algo.get_module("default_policy")
    # Put it in evaluation mode
    rl_module.eval()

    print("Brain loaded! Setting up the arena...")
    # 2. Create the environment
    env = ExampleEnv.create_single_agent_env({"battle_format": "gen9randombattle"})

    # 3. Start the battle
    obs, info = env.reset()
    done = False
    turn = 1
    total_reward = 0.0

    print("\n" + "="*35)
    print("🥊 BATTLE START 🥊")
    print("="*35)
    
    while not done:
        # 4. Talk directly to the PyTorch network!
        # We have to wrap the single observation in a list [] to create a "Batch of 1"
        batch = {
            Columns.OBS: {
                "observations": torch.tensor([obs["observations"]]),
                "action_mask": torch.tensor([obs["action_mask"]])
            }
        }
        
        # Pass it through the network without calculating gradients
        with torch.no_grad():
            out = rl_module.forward_inference(batch)
            logits = out[Columns.ACTION_DIST_INPUTS]
            
            # The network outputs 26 scores. argmax() picks the index with the highest score.
            action = torch.argmax(logits, dim=1).item()
        
        # 5. Execute the action in Showdown
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        # 6. Print what just happened
        print(f"Turn {turn:02d} | Action Picked: {action:02d} | Turn Reward: {reward:5.2f} | Total: {total_reward:5.2f}")
        turn += 1

    print("="*35)
    print(f"🏁 BATTLE FINISHED! Final Score: {total_reward:.2f}")

if __name__ == "__main__":
    watch_bot_play()