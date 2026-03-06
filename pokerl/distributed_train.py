#!/usr/bin/env python3
"""
distributed_train.py — Entraînement distribué PPO avec Ray RLlib.

Usage:
    python distributed_train.py
    python distributed_train.py distributed.ray.address=auto
    python distributed_train.py distributed.num_rollout_workers=8 training.total_timesteps=2000000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from gymnasium import spaces, Wrapper

try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
    from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
    from ray.rllib.utils.torch_utils import FLOAT_MIN
    from ray.tune.registry import register_env
except ImportError as exc:
    raise ImportError(
        "Ray/RLlib n'est pas installé. Installe avec: pip install 'ray[rllib]'"
    ) from exc

try:
    import wandb
except ImportError:
    wandb = None

import torch
import torch.nn as nn

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.environment import make_env
from src.rewards import build_reward


ENV_NAME = "pokerl_env"
MASKED_MODEL_NAME = "pokerl_masked_ppo"
_ENV_REGISTERED = False
_MODEL_REGISTERED = False


class RLlibActionMaskWrapper(Wrapper):
    """Expose l'observation au format RLlib: {observations, action_mask}."""

    def __init__(self, env):
        super().__init__(env)
        action_n = int(env.action_space.n)
        self.action_space = env.action_space
        self.observation_space = spaces.Dict(
            {
                "observations": env.observation_space,
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(action_n,),
                    dtype=np.float32,
                ),
            }
        )

    def _mask(self) -> np.ndarray:
        mask = np.asarray(self.env.action_masks(), dtype=np.float32)
        if mask.sum() <= 0:
            mask = np.ones_like(mask, dtype=np.float32)
        return mask

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {"observations": obs, "action_mask": self._mask()}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped_obs = {"observations": obs, "action_mask": self._mask()}
        return wrapped_obs, reward, terminated, truncated, info


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """Model RLlib appliquant un masque d'action sur les logits de politique."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Handle both Dict (with masking) and Box (without masking) obs spaces
        if isinstance(obs_space, spaces.Dict):
            base_obs_space = obs_space["observations"]
            self.use_masking = True
        else:
            base_obs_space = obs_space
            self.use_masking = False
        
        self.internal_model = TorchFC(
            base_obs_space,
            action_space,
            num_outputs,
            model_config,
            f"{name}_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        if self.use_masking:
            observations = input_dict["obs"]["observations"]
            action_mask = input_dict["obs"]["action_mask"]
            
            logits, _ = self.internal_model({"obs": observations}, state, seq_lens)
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            masked_logits = logits + inf_mask
            return masked_logits, state
        else:
            # No masking - obs is already a tensor, pass through normally
            return self.internal_model(input_dict, state, seq_lens)

    def value_function(self):
        return self.internal_model.value_function()


def _reward_dict(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg.reward, resolve=True)


def make_server_config(cfg: DictConfig) -> ServerConfiguration:
    host = cfg.server.host
    port = cfg.server.port
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        f"http://{host}:{port}/action.php?",
    )


def make_opponent(name: str, battle_format: str, server_cfg: ServerConfiguration):
    opponents = {
        "random": RandomPlayer,
        "max_power": MaxBasePowerPlayer,
        "heuristic": SimpleHeuristicsPlayer,
    }
    cls = opponents.get(name, RandomPlayer)
    return cls(battle_format=battle_format, server_configuration=server_cfg)


def env_creator(env_config: dict[str, Any]):
    server_cfg = ServerConfiguration(
        env_config["server_ws_url"],
        env_config["server_action_url"],
    )
    reward_fn = build_reward(env_config["reward_cfg"])
    opponent = make_opponent(
        env_config["opponent_name"],
        env_config["battle_format"],
        server_cfg,
    )
    env = make_env(
        reward_fn=reward_fn,
        opponent=opponent,
        battle_format=env_config["battle_format"],
        server_configuration=server_cfg,
        log_level=getattr(logging, env_config.get("log_level", "WARNING"), logging.WARNING),
    )
    # Apply action masking wrapper if explicitly enabled
    if env_config.get("use_action_masking", False):
        return RLlibActionMaskWrapper(env)
    return env


def _ensure_env_registered() -> None:
    global _ENV_REGISTERED
    if _ENV_REGISTERED:
        return
    register_env(ENV_NAME, env_creator)
    _ENV_REGISTERED = True


def _ensure_model_registered() -> None:
    global _MODEL_REGISTERED
    if _MODEL_REGISTERED:
        return
    ModelCatalog.register_custom_model(MASKED_MODEL_NAME, TorchActionMaskModel)
    _MODEL_REGISTERED = True


def _init_ray(cfg: DictConfig) -> None:
    if ray.is_initialized():
        return

    init_kwargs: dict[str, Any] = {
        "ignore_reinit_error": bool(cfg.distributed.ray.ignore_reinit_error),
        "include_dashboard": bool(cfg.distributed.ray.include_dashboard),
        "log_to_driver": bool(cfg.distributed.ray.log_to_driver),
    }

    if cfg.distributed.ray.address:
        init_kwargs["address"] = str(cfg.distributed.ray.address)
    if cfg.distributed.ray.num_cpus is not None:
        init_kwargs["num_cpus"] = int(cfg.distributed.ray.num_cpus)
    if cfg.distributed.ray.num_gpus is not None:
        init_kwargs["num_gpus"] = float(cfg.distributed.ray.num_gpus)

    ray.init(**init_kwargs)


def _build_rllib_config(cfg: DictConfig) -> PPOConfig:
    server_cfg = make_server_config(cfg)
    env_cfg = {
        "battle_format": cfg.battle.format,
        "opponent_name": cfg.training.opponent,
        "reward_cfg": _reward_dict(cfg),
        "server_ws_url": server_cfg.websocket_url,
        "server_action_url": server_cfg.authentication_url,
        "log_level": str(cfg.distributed.env_log_level),
        "use_action_masking": bool(cfg.distributed.use_action_masking),
    }

    model_cfg: dict[str, Any] = {"fcnet_hiddens": list(cfg.ppo.net_arch)}
    if bool(cfg.distributed.use_action_masking):
        _ensure_model_registered()
        model_cfg["custom_model"] = MASKED_MODEL_NAME

    ppo_cfg = (
        PPOConfig()
        .environment(env=ENV_NAME, env_config=env_cfg)
        .framework(str(cfg.distributed.framework))
        .resources(num_gpus=float(cfg.distributed.trainer_num_gpus))
        .training(
            lr=float(cfg.ppo.learning_rate),
            gamma=float(cfg.ppo.gamma),
            lambda_=float(cfg.ppo.gae_lambda),
            clip_param=float(cfg.ppo.clip_range),
            entropy_coeff=float(cfg.ppo.ent_coef),
            vf_loss_coeff=float(cfg.ppo.vf_coef),
            train_batch_size=int(cfg.distributed.train_batch_size),
            minibatch_size=int(cfg.distributed.sgd_minibatch_size),
            num_sgd_iter=int(cfg.distributed.num_sgd_iter),
            grad_clip=float(cfg.ppo.max_grad_norm),
            model=model_cfg,
        )
    )

    if hasattr(ppo_cfg, "api_stack"):
        try:
            ppo_cfg = ppo_cfg.api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
        except TypeError:
            pass

    if hasattr(ppo_cfg, "env_runners"):
        ppo_cfg = ppo_cfg.env_runners(
            num_env_runners=int(cfg.distributed.num_rollout_workers),
            num_envs_per_env_runner=int(cfg.distributed.num_envs_per_worker),
            rollout_fragment_length=int(cfg.distributed.rollout_fragment_length),
        )
    else:
        ppo_cfg = ppo_cfg.rollouts(
            num_rollout_workers=int(cfg.distributed.num_rollout_workers),
            num_envs_per_worker=int(cfg.distributed.num_envs_per_worker),
            rollout_fragment_length=int(cfg.distributed.rollout_fragment_length),
        )

    return ppo_cfg


def train_distributed(cfg: DictConfig) -> None:
    _init_ray(cfg)
    _ensure_env_registered()

    model_dir = Path(cfg.training.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = model_dir / str(cfg.distributed.checkpoint_subdir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_enabled = bool(cfg.wandb.enabled and cfg.distributed.wandb_sync)
    if wandb_enabled:
        if wandb is None:
            raise ImportError("wandb n'est pas installé. Installe-le avec: pip install wandb")
        if not cfg.wandb.project:
            raise ValueError("Renseigne wandb.project dans configs/wandb/default.yaml")
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=cfg.wandb.name or "rllib_distributed_ppo",
            tags=list(cfg.wandb.tags) + ["rllib", "distributed"],
            notes=cfg.wandb.notes or None,
            mode=cfg.wandb.mode,
            save_code=bool(cfg.wandb.save_code),
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    trainer = _build_rllib_config(cfg).build()

    total_target = int(cfg.training.total_timesteps)
    max_iterations = int(cfg.distributed.max_iterations)
    checkpoint_freq = max(1, int(cfg.distributed.checkpoint_freq_iters))

    print(f"\n{'=' * 64}")
    print("PokeRL — RLlib Distributed PPO")
    print(f"Ray address      : {cfg.distributed.ray.address or 'local'}")
    print(f"Rollout workers  : {int(cfg.distributed.num_rollout_workers)}")
    print(f"Envs / worker    : {int(cfg.distributed.num_envs_per_worker)}")
    print(f"Train batch size : {int(cfg.distributed.train_batch_size)}")
    print(f"Action masking   : {'on' if bool(cfg.distributed.use_action_masking) else 'off'}")
    print(f"Target timesteps : {total_target:,}")
    print(f"Max iterations   : {max_iterations}")
    print(f"Checkpoints      : {ckpt_dir} (every {checkpoint_freq} iters)")
    print(f"{'=' * 64}\n")

    try:
        for iteration in range(1, max_iterations + 1):
            result = trainer.train()
            timesteps_total = int(result.get("timesteps_total", 0))
            reward_mean = float(result.get("episode_reward_mean", 0.0))

            print(
                f"[iter {iteration:04d}] "
                f"timesteps={timesteps_total:>10,d} | "
                f"reward_mean={reward_mean:>8.3f}"
            )

            if wandb_enabled and wandb is not None and wandb.run is not None:
                payload = {
                    "rllib/episode_reward_mean": reward_mean,
                    "rllib/episode_len_mean": float(result.get("episode_len_mean", 0.0)),
                    "rllib/timesteps_total": timesteps_total,
                    "rllib/training_iteration": iteration,
                }
                learner = result.get("info", {}).get("learner", {})
                default_policy = learner.get("default_policy", {})
                if isinstance(default_policy, dict):
                    for key in ("policy_loss", "vf_loss", "entropy", "kl"):
                        value = default_policy.get(key)
                        if value is not None:
                            payload[f"rllib/{key}"] = float(value)
                wandb.log(payload, step=timesteps_total)

            if iteration % checkpoint_freq == 0:
                ckpt = trainer.save(checkpoint_dir=str(ckpt_dir))
                print(f"Checkpoint saved: {ckpt}")

            if timesteps_total >= total_target:
                print(
                    f"Reached training.total_timesteps={total_target:,} "
                    f"at iteration {iteration}."
                )
                break

    except KeyboardInterrupt:
        print("\nInterruption clavier — arrêt propre.")
    finally:
        final_ckpt = trainer.save(checkpoint_dir=str(ckpt_dir))
        print(f"Final checkpoint: {final_ckpt}")
        trainer.stop()
        if wandb_enabled and wandb is not None and wandb.run is not None:
            wandb.finish()
        if bool(cfg.distributed.shutdown_ray_on_exit) and ray.is_initialized():
            ray.shutdown()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_distributed(cfg)


if __name__ == "__main__":
    main()
    