import warnings
from functools import partial
from typing import Any

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from sb3_contrib.common.maskable.distributions import MaskableDistribution, make_masked_proba_distribution


class AttentionExtractor(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.features_dim = 490
        self.hidden_dim = cfg.hidden_dim
        
        # Define architecture 
        self.pokemon_embedding = nn.Sequential(
            nn.Linear(33, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.allied_move_embedding = nn.Sequential(
            nn.Linear(92, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.enemy_moves_embedding = nn.Sequential(
            nn.Linear(220, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.bench_embedding = nn.Sequential(
            nn.Linear(40, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        self.globals_embedding = nn.Sequential(
            nn.Linear(32, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim,nhead=cfg.num_heads, dim_feedforward = 2*cfg.hidden_dim, batch_first=True, norm_first=True, activation="gelu")
        self.transformer_layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
        )
        
        self.token_type_embedding = nn.Embedding(9, cfg.hidden_dim)
        
        mask = th.zeros(9, 9)
        # actor and critic cls can't see each other
        mask[0, 1] = float("-inf")
        mask[1, 0] = float("-inf")

        # other tokens can't see the cls
        mask[2:, 0] = float("-inf")
        self.register_buffer("attention_mask", mask)
        
        self.actor_cls = nn.Parameter(th.randn(1,cfg.hidden_dim))
        self.critic_cls = nn.Parameter(th.randn(1,cfg.hidden_dim))
        
    def forward(self, obs: th.Tensor):
        batch_size = obs.size(0)
        
        active_allied_pokemon = obs[:,:33]
        moves_allied = obs[:,33:125]
        active_enemy_pokemon = obs[:,125:158]
        moves_enemy = obs[:,158:378]
        allied_bench = obs[:,378:418]
        enemy_bench = obs[:,418:458]
        globals = obs[:,458:]
        
        embedded_ally_pokemon = self.pokemon_embedding(active_allied_pokemon)
        embedded_ally_moves = self.allied_move_embedding(moves_allied)
        embedded_enemy_pokemon = self.pokemon_embedding(active_enemy_pokemon)
        embedded_enemy_moves = self.enemy_moves_embedding(moves_enemy)
        embedded_ally_bench = self.bench_embedding(allied_bench)
        embedded_enemy_bench = self.bench_embedding(enemy_bench)
        embedded_globals = self.globals_embedding(globals)
        
        actor_cls = self.actor_cls.expand(batch_size,-1)
        critic_cls = self.critic_cls.expand(batch_size,-1) 
        
        input_seq = th.stack([
            actor_cls, 
            critic_cls, 
            embedded_ally_pokemon, 
            embedded_ally_moves,
            embedded_enemy_pokemon,
            embedded_enemy_moves,
            embedded_ally_bench,
            embedded_enemy_bench,
            embedded_globals,
        ], dim=1)
        
        # add token type embeddings
        token_type_ids = th.arange(9, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        input_seq = input_seq + self.token_type_embedding(token_type_ids)
        
        output_seq = self.transformer_layers(input_seq, mask=self.attention_mask)
        output_actor_cls = output_seq[:,0,:]
        output_critic_cls = output_seq[:,1,:]
        
        return output_actor_cls, output_critic_cls
        
        
        

class AttentionPolicy(MaskableActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        cfg = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
                
        if cfg is None:
            raise RuntimeError("No config dictionary passed for attention policy")
        self.cfg = cfg
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )
        
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        
        # Action distribution
        self.action_dist = make_masked_proba_distribution(action_space)
        self._build(lr_schedule)



    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        latent_pi, latent_vf = self.attention_extractor(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _build_attention_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        self.attention_extractor = AttentionExtractor(
            cfg=self.cfg
        ).to(self.cfg.device)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_attention_extractor()

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=self.attention_extractor.hidden_dim)
        self.value_net = nn.Linear(self.attention_extractor.hidden_dim, 1)


        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MaskableDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def _predict(  # type: ignore[override]
        self,
        observation: PyTorchObs,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :param action_masks: Action masks to apply to the action distribution
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation, action_masks).get_actions(deterministic=deterministic)



    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :param action_masks: Action masks to apply to the action distribution
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        # Check for common mistake that the user does not mix Gym/VecEnv API
        # Tuple obs are not supported by SB3, so we can safely do that check
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[1], dict):
            raise ValueError(
                "You have passed a tuple to the predict() function instead of a Numpy array or a Dict. "
                "You are probably mixing Gym API with SB3 VecEnv API: `obs, info = env.reset()` (Gym) "
                "vs `obs = vec_env.reset()` (SB3 VecEnv). "
                "See related issue https://github.com/DLR-RM/stable-baselines3/issues/1694 "
                "and documentation for more information: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api"
            )

        obs_tensor, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            actions = self._predict(obs_tensor, deterministic=deterministic, action_masks=action_masks)
            # Convert to numpy
            actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[assignment, misc]

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        if not vectorized_env:
            assert isinstance(actions, np.ndarray)
            actions = actions.squeeze(axis=0)

        return actions, state  # type: ignore[return-value]




    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        latent_pi, latent_vf = self.attention_extractor(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()




    def get_distribution(self, obs: PyTorchObs, action_masks: np.ndarray | None = None) -> MaskableDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :param action_masks: Actions' mask
        :return: the action distribution.
        """
        latent_pi, _ = self.attention_extractor(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution




    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, latent_vf = self.attention_extractor(obs)
        return self.value_net(latent_vf)




