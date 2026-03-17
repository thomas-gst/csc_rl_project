# train.py

import os
os.environ["RAY_DISABLE_METRICS_COLLECTION"] = "1" 
os.environ["RAY_CPP_LOG_LEVEL"] = "3" 

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from gymnasium.spaces import Dict as GymDict
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from env import PokeTransformerEnv
from model import PokeTransformerModule

class WinRateCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        if episode.get_return() > 0:
            metrics_logger.log_value("win_rate", 1.0, reduce="mean")
        else:
            metrics_logger.log_value("win_rate", 0.0, reduce="mean")

# --- THE MAGIC SYNC FUNCTION ---
def sync_opponents(env_runner):
    """Grabs the latest PPO weights and copies them into the Opponent Bot."""
    try:
        latest_state = env_runner.module.state_dict()
    except AttributeError:
        latest_state = env_runner.module["default_policy"].state_dict()
        
    for env_wrapper in env_runner.environments:
        # Load the updated weights but keep the opponent frozen on the CPU
        env_wrapper.opponent.model.load_state_dict(latest_state, strict=False)

def single_agent_train():
    register_env("showdown", PokeTransformerEnv.create_single_agent_env)
    PRETRAINED_PATH = os.path.abspath("pretrained_pokeformer_final.pt")

    algo_config = (
        PPOConfig()
        .environment(
            "showdown",
            # Pass the pretrained path so the Env can boot up the Self-Play bot!
            env_config={"battle_format": "gen9randombattle", "pretrained_weights": PRETRAINED_PATH}, 
            disable_env_checking=True,
        )
        .learners(num_learners=1, num_gpus_per_learner=1)
        .env_runners(
            num_env_runners=10,           
            num_envs_per_env_runner=2,
            sample_timeout_s=800.0,       
            rollout_fragment_length="auto"
        )
        .callbacks(WinRateCallback)
        .training(
            gamma=0.99,
            lr=1e-5, # Drop the LR down. PPO Self-Play requires tiny adjustments.
            entropy_coeff=0.01, # Allow 1% random moves so it discovers new strategies!
            train_batch_size_per_learner=512, 
            minibatch_size=64
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PokeTransformerModule,
                observation_space=GymDict({
                    "observations": Box(low=-10.0, high=np.inf, shape=(13, 200), dtype=np.float32),
                    "action_mask": Box(0.0, 1.0, shape=(26,), dtype=np.float32),
                }),
                action_space=Discrete(26),
                model_config={
                    "pretrained_weights": PRETRAINED_PATH,
                    "freeze_actor": False 
                },
            )
        )
    )
    
    algo = algo_config.build_algo()

    print("Starting Self-Play Training Loop...")
    for i in range(1000): 
        result = algo.train()
        
        # --- THE SYNC TRIGGER ---
        # Every 10 iterations, update the opponent to match the new, smarter AI
        if i % 10 == 0 and i > 0:
            print(f"\n[!] Iteration {i}: Syncing latest brain to Self-Play Opponents! Win rate will drop to 50%.\n")
            algo.env_runner_group.foreach_env_runner(sync_opponents)
        # ------------------------

        runners_stats = result.get("env_runners", {})
        agent_returns = runners_stats.get("agent_episode_return_mean", {})
        reward = agent_returns.get("default_agent", 0.0)
        
        win_rate_mean = runners_stats.get("win_rate", 0.0)
        win_percentage = win_rate_mean * 100 
        
        if isinstance(reward, str):
            print(f"Iter {i}: Waiting for games to finish...")
        else:
            print(f"Iter {i}: Mean Return = {reward:.3f} | Win Rate vs Self = {win_percentage:.1f}%")

    print("Saving Self-Play model...")
    save_path = os.path.abspath("poke_model_checkpoint")
    algo.save(save_path)
    print(f"Model successfully saved at: {save_path}")
    
    algo.stop()

if __name__ == "__main__":
    single_agent_train()