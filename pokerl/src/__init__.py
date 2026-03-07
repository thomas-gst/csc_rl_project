"""PokeRL — Apprentissage par renforcement pour Pokémon Showdown."""

from .environment import PokeRLEnv
from .features import embed_battle, OBSERVATION_SIZE
from .pipeline_ppo import (
    ObservationUnpacker,
    PipelineMaskablePolicy,
    PipelineObservationWrapper,
    PokeEnvKnowledgeBase,
    PokeTransformer,
    PokemonEncoderConfig,
    enrich_obs_with_poke_env,
)
from .rewards import DenseReward, BaseReward

__all__ = [
    "PokeRLEnv",
    "embed_battle",
    "OBSERVATION_SIZE",
    "PokemonEncoderConfig",
    "PokeTransformer",
    "ObservationUnpacker",
    "PokeEnvKnowledgeBase",
    "enrich_obs_with_poke_env",
    "PipelineObservationWrapper",
    "PipelineMaskablePolicy",
    "DenseReward",
    "BaseReward",
]
