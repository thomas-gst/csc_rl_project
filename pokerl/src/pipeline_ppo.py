from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig
from poke_env.battle.battle import Battle
from poke_env.battle.pokemon import Pokemon
from poke_env.data import GenData, to_id_str
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import Tensor, nn

from .features import STATUS_LIST, TYPE_LIST, embed_battle

LOGGER = logging.getLogger(__name__)
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
MOVE_CATEGORY_TO_INDEX = {"PHYSICAL": 0, "SPECIAL": 1, "STATUS": 2}
PRIORITY_OFFSET = 6
PRIORITY_BINS = 13


@dataclass(frozen=True)
class PokemonEncoderConfig:
    move_vocab_size: int = 2048
    ability_vocab_size: int = 512
    species_vocab_size: int = 2048
    value_buckets: int = 101
    power_buckets: int = 256
    stat_buckets: int = 1024
    move_embed_dim: int = 32
    ability_embed_dim: int = 24
    species_embed_dim: int = 48
    value_embed_dim: int = 8
    power_embed_dim: int = 12
    stat_embed_dim: int = 12
    move_type_dim: int = 18
    move_priority_dim: int = 13
    move_category_dim: int = 3
    boosts_dim: int = 7
    pokemon_flag_dim: int = 40
    n_pok: int = 12
    moves_per_pokemon: int = 4
    abilities_per_pokemon: int = 4
    move_hidden_dims: tuple[int, ...] = (128, 96)
    move_output_dim: int = 48
    ability_hidden_dims: tuple[int, ...] = (128, 96)
    ability_output_dim: int = 64
    pokemon_hidden_dim: int = 1024
    pokemon_token_dim: int = 1024
    field_hidden_dim: int = 1024
    d_model: int = 1024
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_dim: int = 2048
    transformer_dropout: float = 0.0
    readout_heads: int = 8
    readout_ff_dim: int = 2048
    n_actions: int = 22
    value_bins: int = 51
    value_min: float = -1.6
    value_max: float = 1.6

    @property
    def move_scalar_dim(self) -> int:
        return 3 + self.move_type_dim + self.move_priority_dim + self.move_category_dim

    @property
    def pokemon_scalar_dim(self) -> int:
        return 2 + 8 + self.boosts_dim

    @property
    def move_input_dim(self) -> int:
        return (
            self.move_embed_dim
            + self.value_embed_dim
            + self.value_embed_dim
            + self.power_embed_dim
            + self.move_type_dim
            + self.move_priority_dim
            + self.move_category_dim
        )

    @property
    def ability_input_dim(self) -> int:
        return self.abilities_per_pokemon * self.ability_embed_dim

    @property
    def embedded_stat_count(self) -> int:
        return 8

    @property
    def pokemon_input_dim(self) -> int:
        return (
            self.species_embed_dim
            + self.value_embed_dim
            + self.value_embed_dim
            + self.embedded_stat_count * self.stat_embed_dim
            + self.ability_output_dim
            + self.moves_per_pokemon * self.move_output_dim
            + self.boosts_dim
            + self.pokemon_flag_dim
        )

    @property
    def sequence_length(self) -> int:
        return 3 + self.n_pok


@dataclass
class PokemonTokenOutput:
    move_vectors: Tensor
    ability_vectors: Tensor
    pokemon_input: Tensor
    pokemon_tokens: Tensor


@dataclass
class FieldEncoderOutput:
    field_input: Tensor
    field_token: Tensor


@dataclass
class TransformerForwardOutput:
    obs: dict[str, Tensor]
    pokemon: PokemonTokenOutput
    field: FieldEncoderOutput
    sequence_input: Tensor
    encoded_sequence: Tensor
    readout_tokens: Tensor
    policy_logits: Tensor
    masked_policy_logits: Tensor
    value_logits: Tensor
    value_probs: Tensor
    value: Tensor
    attention_mask: Tensor


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if not is_last:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MoveSubnet(nn.Module):
    def __init__(self, cfg: PokemonEncoderConfig) -> None:
        super().__init__()
        self.move_net = MLP(cfg.move_input_dim, cfg.move_hidden_dims, cfg.move_output_dim)

    def forward(self, move_features: Tensor) -> Tensor:
        return self.move_net(move_features)


class AbilitySubnet(nn.Module):
    def __init__(self, cfg: PokemonEncoderConfig) -> None:
        super().__init__()
        self.ability_net = MLP(cfg.ability_input_dim, cfg.ability_hidden_dims, cfg.ability_output_dim)

    def forward(self, ability_features: Tensor) -> Tensor:
        return self.ability_net(ability_features)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class PokeTransformerEncoder(nn.Module):
    def __init__(self, cfg: PokemonEncoderConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.transformer_heads, cfg.transformer_ff_dim, cfg.transformer_dropout)
                for _ in range(cfg.transformer_layers)
            ]
        )

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class PokemonIdentityEncoder(nn.Module):
    def __init__(
        self,
        cfg: PokemonEncoderConfig,
        move_emb: nn.Embedding,
        ability_emb: nn.Embedding,
        pokemon_id_emb: nn.Embedding,
        val_100_emb: nn.Embedding,
        power_emb: nn.Embedding,
        stat_emb: nn.Embedding,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.move_emb = move_emb
        self.ability_emb = ability_emb
        self.pokemon_id_emb = pokemon_id_emb
        self.val_100_emb = val_100_emb
        self.power_emb = power_emb
        self.stat_emb = stat_emb
        self.move_subnet = MoveSubnet(cfg)
        self.ability_subnet = AbilitySubnet(cfg)
        self.pokemon_net = nn.Sequential(
            nn.Linear(cfg.pokemon_input_dim, cfg.pokemon_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.pokemon_hidden_dim, cfg.pokemon_token_dim),
            nn.LayerNorm(cfg.pokemon_token_dim),
        )

    def build_move_vectors(self, obs: Mapping[str, Tensor]) -> Tensor:
        move_ids = obs["move_ids"].long().clamp(0, self.cfg.move_vocab_size - 1)
        move_scalars = obs["move_scalars"].float()
        batch_size = move_ids.shape[0]
        accuracy_int = move_scalars[..., 0].clamp(0, self.cfg.value_buckets - 1).long()
        pp_int = move_scalars[..., 1].clamp(0, self.cfg.value_buckets - 1).long()
        power_int = move_scalars[..., 2].clamp(0, self.cfg.power_buckets - 1).long()
        type_onehot = move_scalars[..., 3 : 3 + self.cfg.move_type_dim]
        priority_start = 3 + self.cfg.move_type_dim
        priority_end = priority_start + self.cfg.move_priority_dim
        priority_onehot = move_scalars[..., priority_start:priority_end]
        category_onehot = move_scalars[..., priority_end : priority_end + self.cfg.move_category_dim]
        m_combined = torch.cat(
            [
                self.move_emb(move_ids),
                self.val_100_emb(accuracy_int),
                self.val_100_emb(pp_int),
                self.power_emb(power_int),
                type_onehot,
                priority_onehot,
                category_onehot,
            ],
            dim=-1,
        )
        move_flat = m_combined.reshape(batch_size * self.cfg.n_pok * self.cfg.moves_per_pokemon, -1)
        move_encoded = self.move_subnet(move_flat)
        return move_encoded.reshape(batch_size, self.cfg.n_pok, -1)

    def build_ability_vectors(self, obs: Mapping[str, Tensor]) -> Tensor:
        ability_ids = obs["ability_ids"].long().clamp(0, self.cfg.ability_vocab_size - 1)
        batch_size = ability_ids.shape[0]
        ability_emb = self.ability_emb(ability_ids)
        ability_flat = ability_emb.reshape(batch_size * self.cfg.n_pok, -1)
        ability_encoded = self.ability_subnet(ability_flat)
        return ability_encoded.reshape(batch_size, self.cfg.n_pok, -1)

    def build_pokemon_tokens(self, obs: Mapping[str, Tensor]) -> PokemonTokenOutput:
        batch_size = obs["pokemon_species_ids"].shape[0]
        species_ids = obs["pokemon_species_ids"].long().clamp(0, self.cfg.species_vocab_size - 1)
        pokemon_scalars = obs["pokemon_scalars"].float()
        pokemon_flags = obs["pokemon_flags"].float()
        m_vecs = self.build_move_vectors(obs)
        a_vecs = self.build_ability_vectors(obs)
        hp_int = pokemon_scalars[..., 0].clamp(0, self.cfg.value_buckets - 1).long()
        level_int = pokemon_scalars[..., 1].clamp(0, self.cfg.value_buckets - 1).long()
        stat_values = pokemon_scalars[..., 2:10].clamp(0, self.cfg.stat_buckets - 1).long()
        boosts_raw = pokemon_scalars[..., 10 : 10 + self.cfg.boosts_dim]
        species_vec = self.pokemon_id_emb(species_ids)
        hp_vec = self.val_100_emb(hp_int)
        level_vec = self.val_100_emb(level_int)
        stat_vec = self.stat_emb(stat_values).flatten(2)
        p_in = torch.cat(
            [species_vec, hp_vec, level_vec, stat_vec, a_vecs, m_vecs, boosts_raw, pokemon_flags],
            dim=-1,
        )
        p_flat = p_in.reshape(batch_size * self.cfg.n_pok, -1)
        p_tokens = self.pokemon_net(p_flat).reshape(batch_size, self.cfg.n_pok, -1)
        return PokemonTokenOutput(
            move_vectors=m_vecs,
            ability_vectors=a_vecs,
            pokemon_input=p_in,
            pokemon_tokens=p_tokens,
        )

    def forward(self, obs: Mapping[str, Tensor]) -> PokemonTokenOutput:
        return self.build_pokemon_tokens(obs)


class FieldTokenEncoder(nn.Module):
    def __init__(self, cfg: PokemonEncoderConfig, val_100_emb: nn.Embedding) -> None:
        super().__init__()
        self.cfg = cfg
        self.val_100_emb = val_100_emb
        self.field_net: nn.Sequential | None = None
        self._field_input_dim: int | None = None

    def _build_subnet(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _get_field_net(self, input_dim: int, device: torch.device) -> nn.Sequential:
        if self.field_net is None or self._field_input_dim != input_dim:
            self.field_net = self._build_subnet(input_dim, self.cfg.field_hidden_dim, self.cfg.d_model).to(device)
            self._field_input_dim = input_dim
        return self.field_net

    def forward(self, obs: Mapping[str, Tensor], g_map: Mapping[str, int | tuple[int, ...] | list[int]]) -> FieldEncoderOutput:
        global_scalars = obs["global_scalars"].float()
        turn_idx = int(g_map["turn_int"])
        remainder_info = g_map["remainder_raw"]
        remainder_start = int(remainder_info[0] if isinstance(remainder_info, (tuple, list)) else remainder_info)
        turn_bucket = (global_scalars[:, turn_idx] * 100.0).round()
        turn_bucket = turn_bucket.clamp(0, self.cfg.value_buckets - 1).long()
        turn_emb = self.val_100_emb(turn_bucket)
        field_in = torch.cat([turn_emb, global_scalars[:, remainder_start:]], dim=-1)
        field_net = self._get_field_net(field_in.shape[-1], field_in.device)
        field_token = field_net(field_in).unsqueeze(1)
        return FieldEncoderOutput(field_input=field_in, field_token=field_token)


class ReadoutBlock(nn.Module):
    def __init__(self, cfg: PokemonEncoderConfig) -> None:
        super().__init__()
        self.readout_norm_attn = nn.LayerNorm(cfg.d_model)
        self.readout_norm_ff = nn.LayerNorm(cfg.d_model)
        self.readout_mha = nn.MultiheadAttention(
            cfg.d_model,
            cfg.readout_heads,
            dropout=cfg.transformer_dropout,
            batch_first=True,
        )
        self.readout_net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.readout_ff_dim),
            nn.GELU(),
            nn.Linear(cfg.readout_ff_dim, cfg.d_model),
        )

    def forward(self, query_tokens: Tensor, full_sequence: Tensor, attn_mask: Tensor) -> Tensor:
        q = self.readout_norm_attn(query_tokens)
        kv = self.readout_norm_attn(full_sequence)
        attended, _ = self.readout_mha(query=q, key=kv, value=kv, attn_mask=attn_mask, need_weights=False)
        q_out = query_tokens + attended
        q_out = q_out + self.readout_net(self.readout_norm_ff(q_out))
        return q_out


class PokeTransformer(nn.Module):
    def __init__(self, cfg: PokemonEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.move_emb = nn.Embedding(cfg.move_vocab_size, cfg.move_embed_dim)
        self.ability_emb = nn.Embedding(cfg.ability_vocab_size, cfg.ability_embed_dim)
        self.pokemon_id_emb = nn.Embedding(cfg.species_vocab_size, cfg.species_embed_dim)
        self.val_100_emb = nn.Embedding(cfg.value_buckets, cfg.value_embed_dim)
        self.power_emb = nn.Embedding(cfg.power_buckets, cfg.power_embed_dim)
        self.stat_emb = nn.Embedding(cfg.stat_buckets, cfg.stat_embed_dim)
        self.encoder = PokemonIdentityEncoder(
            cfg=cfg,
            move_emb=self.move_emb,
            ability_emb=self.ability_emb,
            pokemon_id_emb=self.pokemon_id_emb,
            val_100_emb=self.val_100_emb,
            power_emb=self.power_emb,
            stat_emb=self.stat_emb,
        )
        self.field_encoder = FieldTokenEncoder(cfg=cfg, val_100_emb=self.val_100_emb)
        self.actor_tok = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.critic_tok = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.backbone = PokeTransformerEncoder(cfg)
        self.readout = ReadoutBlock(cfg)
        self.pi_head = nn.Linear(cfg.d_model, cfg.n_actions)
        self.v_head = nn.Linear(cfg.d_model, cfg.value_bins)
        self.register_buffer(
            "value_support",
            torch.linspace(cfg.value_min, cfg.value_max, cfg.value_bins),
            persistent=False,
        )
        self.register_buffer(
            "poke_mask",
            self._build_poke_mask(cfg.sequence_length),
            persistent=False,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.actor_tok, std=0.02)
        nn.init.normal_(self.critic_tok, std=0.02)

    def _build_poke_mask(self, sequence_length: int) -> Tensor:
        mask = torch.zeros(sequence_length, sequence_length, dtype=torch.float32)
        state_start = 2
        mask[state_start:, 0:2] = float("-inf")
        mask[0, 1] = float("-inf")
        mask[1, 0] = float("-inf")
        return mask

    def _expand_special_tokens(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        actor = self.actor_tok.expand(batch_size, -1, -1).to(device)
        critic = self.critic_tok.expand(batch_size, -1, -1).to(device)
        return actor, critic

    def _apply_action_mask(self, logits: Tensor, action_mask: Tensor | None) -> Tensor:
        if action_mask is None:
            return logits
        mask = action_mask if action_mask.dtype == torch.bool else action_mask > 0.5
        return logits.masked_fill(~mask, -1e4)

    def build_pokemon_tokens(self, obs: Mapping[str, Tensor]) -> PokemonTokenOutput:
        return self.encoder(obs)

    def build_field_token(self, obs: Mapping[str, Tensor], g_map: Mapping[str, int | tuple[int, ...] | list[int]]) -> FieldEncoderOutput:
        return self.field_encoder(obs, g_map)

    def build_token_sequence(self, pokemon_output: PokemonTokenOutput, field_output: FieldEncoderOutput) -> Tensor:
        batch_size = pokemon_output.pokemon_tokens.shape[0]
        device = pokemon_output.pokemon_tokens.device
        actor_tok, critic_tok = self._expand_special_tokens(batch_size, device)
        return torch.cat([actor_tok, critic_tok, field_output.field_token, pokemon_output.pokemon_tokens], dim=1)

    def forward_from_obs(
        self,
        obs: Mapping[str, Tensor],
        g_map: Mapping[str, int | tuple[int, ...] | list[int]],
        action_mask: Tensor | None = None,
    ) -> TransformerForwardOutput:
        pokemon_output = self.build_pokemon_tokens(obs)
        field_output = self.build_field_token(obs, g_map)
        sequence_input = self.build_token_sequence(pokemon_output, field_output)
        attn_mask = self.poke_mask.to(sequence_input.device)
        encoded_sequence = self.backbone(sequence_input, attn_mask)
        query_tokens = encoded_sequence[:, 0:2, :]
        readout_tokens = self.readout(query_tokens, encoded_sequence, attn_mask[:2, :])
        actor_token = readout_tokens[:, 0, :]
        critic_token = readout_tokens[:, 1, :]
        policy_logits = self.pi_head(actor_token)
        masked_policy_logits = self._apply_action_mask(policy_logits, action_mask)
        value_logits = self.v_head(critic_token)
        value_probs = torch.softmax(value_logits, dim=-1)
        value = (value_probs * self.value_support.to(value_probs.device)).sum(dim=-1)
        return TransformerForwardOutput(
            obs=dict(obs),
            pokemon=pokemon_output,
            field=field_output,
            sequence_input=sequence_input,
            encoded_sequence=encoded_sequence,
            readout_tokens=readout_tokens,
            policy_logits=policy_logits,
            masked_policy_logits=masked_policy_logits,
            value_logits=value_logits,
            value_probs=value_probs,
            value=value,
            attention_mask=attn_mask,
        )


class ObservationUnpacker(nn.Module):
    ACTIVE_SIZE = 33
    ALLY_MOVES_SIZE = 92
    OPP_ACTIVE_SIZE = 33
    OPP_MOVES_SIZE = 220
    BENCH_SIZE = 40
    SIDE_CONDITIONS_SIZE = 14
    WEATHER_SIZE = 8
    TERRAIN_SIZE = 6
    GIMMICKS_SIZE = 4
    OBS_SIZE = 490
    ALLY_ACTIVE_SLICE = slice(0, 33)
    ALLY_MOVES_SLICE = slice(33, 125)
    OPP_ACTIVE_SLICE = slice(125, 158)
    OPP_MOVES_SLICE = slice(158, 378)
    ALLY_BENCH_SLICE = slice(378, 418)
    OPP_BENCH_SLICE = slice(418, 458)
    SIDE_CONDITIONS_SLICE = slice(458, 472)
    WEATHER_SLICE = slice(472, 480)
    TERRAIN_SLICE = slice(480, 486)
    GIMMICKS_SLICE = slice(486, 490)

    def __init__(self, default_level: int = 50) -> None:
        super().__init__()
        self.default_level = default_level

    def _ensure_batched(self, obs_flat: Tensor) -> Tensor:
        return obs_flat.unsqueeze(0) if obs_flat.ndim == 1 else obs_flat

    def _make_move_scalars(self, move_block: Tensor, *, include_pp: bool) -> Tensor:
        batch_size, num_moves, _ = move_block.shape
        out = torch.zeros(batch_size, num_moves, 37, device=move_block.device, dtype=torch.float32)
        out[..., 0] = (move_block[..., 19] * 100.0).clamp(0, 100)
        out[..., 1] = (move_block[..., 22] * 100.0).clamp(0, 100) if include_pp else 0.0
        out[..., 2] = (move_block[..., 18] * 250.0).clamp(0, 255)
        out[..., 3:21] = move_block[..., :18]
        has_move = move_block.abs().sum(dim=-1) > 0
        physical = move_block[..., 20] > 0.5
        special = move_block[..., 21] > 0.5
        out[..., 34] = physical.float()
        out[..., 35] = special.float()
        out[..., 36] = (has_move & ~physical & ~special).float()
        return out

    def _fill_active_slot(self, pokemon_scalars: Tensor, pokemon_flags: Tensor, slot: int, active_block: Tensor, *, is_ally: bool) -> None:
        hp = (active_block[:, 0] * 100.0).clamp(0, 100)
        pokemon_scalars[:, slot, 0] = hp
        pokemon_scalars[:, slot, 1] = float(self.default_level)
        pokemon_scalars[:, slot, 10:17] = (active_block[:, 19:26] * 6.0).clamp(-6, 6)
        pokemon_flags[:, slot, :18] = active_block[:, 1:19]
        pokemon_flags[:, slot, 18:25] = active_block[:, 26:33]
        pokemon_flags[:, slot, 25] = 1.0
        pokemon_flags[:, slot, 26] = 1.0 if is_ally else 0.0
        pokemon_flags[:, slot, 32] = 1.0

    def _fill_bench_slots(self, pokemon_scalars: Tensor, pokemon_flags: Tensor, start_slot: int, bench_block: Tensor, *, is_ally: bool) -> None:
        bench = bench_block.view(bench_block.shape[0], 5, 8)
        pokemon_scalars[:, start_slot:start_slot + 5, 0] = (bench[..., 0] * 100.0).clamp(0, 100)
        pokemon_scalars[:, start_slot:start_slot + 5, 1] = float(self.default_level)
        pokemon_flags[:, start_slot:start_slot + 5, 18:25] = bench[..., 1:8]
        pokemon_flags[:, start_slot:start_slot + 5, 26] = 1.0 if is_ally else 0.0
        pokemon_flags[:, start_slot:start_slot + 5, 32] = (bench[..., 0] > 0).float()

    def forward(self, obs_flat: Tensor) -> dict[str, Tensor]:
        obs_flat = self._ensure_batched(obs_flat.float())
        if obs_flat.shape[-1] < self.OBS_SIZE:
            raise ValueError(f"Observation trop petite: {obs_flat.shape[-1]} < {self.OBS_SIZE}")
        batch_size = obs_flat.shape[0]
        ally_active = obs_flat[:, self.ALLY_ACTIVE_SLICE]
        ally_moves = obs_flat[:, self.ALLY_MOVES_SLICE].view(batch_size, 4, 23)
        opp_active = obs_flat[:, self.OPP_ACTIVE_SLICE]
        opp_moves = obs_flat[:, self.OPP_MOVES_SLICE].view(batch_size, 10, 22)
        ally_bench = obs_flat[:, self.ALLY_BENCH_SLICE]
        opp_bench = obs_flat[:, self.OPP_BENCH_SLICE]
        side_conditions = obs_flat[:, self.SIDE_CONDITIONS_SLICE]
        weather = obs_flat[:, self.WEATHER_SLICE]
        terrain = obs_flat[:, self.TERRAIN_SLICE]
        gimmicks = obs_flat[:, self.GIMMICKS_SLICE]
        pokemon_species_ids = torch.zeros(batch_size, 12, device=obs_flat.device, dtype=torch.long)
        ability_ids = torch.zeros(batch_size, 12, 4, device=obs_flat.device, dtype=torch.long)
        move_ids = torch.zeros(batch_size, 12, 4, device=obs_flat.device, dtype=torch.long)
        move_scalars = torch.zeros(batch_size, 12, 4, 37, device=obs_flat.device, dtype=torch.float32)
        pokemon_scalars = torch.zeros(batch_size, 12, 17, device=obs_flat.device, dtype=torch.float32)
        pokemon_flags = torch.zeros(batch_size, 12, 40, device=obs_flat.device, dtype=torch.float32)
        move_scalars[:, 0] = self._make_move_scalars(ally_moves, include_pp=True)
        move_scalars[:, 6] = self._make_move_scalars(opp_moves[:, :4], include_pp=False)
        self._fill_active_slot(pokemon_scalars, pokemon_flags, 0, ally_active, is_ally=True)
        self._fill_active_slot(pokemon_scalars, pokemon_flags, 6, opp_active, is_ally=False)
        self._fill_bench_slots(pokemon_scalars, pokemon_flags, 1, ally_bench, is_ally=True)
        self._fill_bench_slots(pokemon_scalars, pokemon_flags, 7, opp_bench, is_ally=False)
        global_scalars = torch.zeros(batch_size, 33, device=obs_flat.device, dtype=torch.float32)
        global_scalars[:, 1:15] = side_conditions
        global_scalars[:, 15:23] = weather
        global_scalars[:, 23:29] = terrain
        global_scalars[:, 29:33] = gimmicks
        return {
            "obs_flat": obs_flat,
            "global_scalars": global_scalars,
            "pokemon_species_ids": pokemon_species_ids,
            "ability_ids": ability_ids,
            "move_ids": move_ids,
            "move_scalars": move_scalars,
            "pokemon_scalars": pokemon_scalars,
            "pokemon_flags": pokemon_flags,
        }


@dataclass
class PokeEnvAvailability:
    species_from_pokedex: bool
    moves_from_gen_data: bool
    abilities_from_pokedex: bool
    exact_items_used: bool
    battle_history_used: bool


class DynamicStringIndexer:
    def __init__(self, initial_tokens: list[str] | None = None) -> None:
        tokens = ["<unk>"] if initial_tokens is None else ["<unk>", *initial_tokens]
        self.stoi: dict[str, int] = {}
        for token in tokens:
            self.encode(token)

    def encode(self, value: str | None) -> int:
        key = to_id_str(value or "<unk>") or "<unk>"
        if key not in self.stoi:
            self.stoi[key] = len(self.stoi)
        return self.stoi[key]


class PokeEnvKnowledgeBase:
    def __init__(self, gen: int = 9) -> None:
        gen_data = GenData.from_gen(gen)
        species_tokens = sorted(gen_data.pokedex.keys())
        move_tokens = sorted(gen_data.moves.keys())
        ability_tokens = sorted(
            {
                to_id_str(ability)
                for dex_entry in gen_data.pokedex.values()
                for ability in dex_entry.get("abilities", {}).values()
                if ability
            }
        )
        self.species_index = DynamicStringIndexer(species_tokens)
        self.move_index = DynamicStringIndexer(move_tokens)
        self.ability_index = DynamicStringIndexer(ability_tokens)
        self.availability = PokeEnvAvailability(
            species_from_pokedex=True,
            moves_from_gen_data=True,
            abilities_from_pokedex=True,
            exact_items_used=False,
            battle_history_used=False,
        )

    def encode_species(self, pokemon: Pokemon | None) -> int:
        return self.species_index.encode(None if pokemon is None else pokemon.species)

    def encode_abilities(self, pokemon: Pokemon | None, max_slots: int = 4) -> list[int]:
        values: list[int] = []
        if pokemon is not None:
            candidates: list[str | None] = []
            if pokemon.ability is not None:
                candidates.append(pokemon.ability)
            candidates.extend(list(pokemon.possible_abilities))
            seen: set[str] = set()
            for ability in candidates:
                key = to_id_str(ability or "")
                if not key or key in seen:
                    continue
                seen.add(key)
                values.append(self.ability_index.encode(key))
        values = values[:max_slots]
        return values + [0] * (max_slots - len(values))

    def encode_moves(self, pokemon: Pokemon | None, max_slots: int = 4) -> list[int]:
        values: list[int] = []
        if pokemon is not None:
            for move in list(pokemon.moves.values())[:max_slots]:
                values.append(self.move_index.encode(move.id))
        return values + [0] * (max_slots - len(values))


def ordered_battle_slots(battle: Battle) -> list[Pokemon | None]:
    ally_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    ally_bench = [mon for mon in battle.team.values() if mon is not ally_active][:5]
    opp_bench = [mon for mon in battle.opponent_team.values() if mon is not opp_active][:5]
    return [ally_active, *ally_bench, *([None] * (5 - len(ally_bench))), opp_active, *opp_bench, *([None] * (5 - len(opp_bench)))]


def estimate_stat(mon: Pokemon, stat_name: str) -> int:
    base = mon.base_stats.get(stat_name, 100)
    level = int(mon.level or 50)
    iv, ev, nature_mult = 31, 84, 1.0
    if stat_name == "spe":
        for move_id in (mon.moves or {}):
            if move_id in {"trickroom", "gyroball"}:
                iv, ev, nature_mult = 0, 0, 0.9
                break
    if stat_name == "hp":
        return int(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10
    raw_stat = int(((2 * base + iv + (ev // 4)) * level) / 100) + 5
    return int(raw_stat * nature_mult)


def stat_estimate(pokemon: Pokemon | None) -> tuple[list[int], dict[str, bool]]:
    if pokemon is None:
        return [0] * 8, {"species_known": False, "stats_exact": False, "stats_estimated": False}
    species_known = bool(pokemon.species)
    exact_stats = pokemon.stats if any(value is not None for value in pokemon.stats.values()) else None
    values: list[int] = []
    used_exact = False
    used_estimate = False
    for key in STAT_KEYS:
        if exact_stats is not None and exact_stats.get(key) is not None:
            value = int(exact_stats[key])
            used_exact = True
        elif species_known:
            value = estimate_stat(pokemon, key)
            used_estimate = True
        else:
            value = 0
        values.append(value)
    estimated_hp = values[0]
    max_hp = int(pokemon.max_hp or estimated_hp or 0)
    current_hp = int(pokemon.current_hp or round(float(pokemon.current_hp_fraction or 0.0) * max(1, max_hp)))
    values.extend([max_hp, current_hp])
    return values[:8], {
        "species_known": species_known,
        "stats_exact": used_exact,
        "stats_estimated": used_estimate and not used_exact,
    }


def level_estimate(pokemon: Pokemon | None, default_level: int = 50) -> int:
    if pokemon is None or pokemon.level is None:
        return default_level
    return int(pokemon.level)


def encode_move_scalars(pokemon: Pokemon | None, move_scalar_dim: int = 37) -> Tensor:
    block = torch.zeros(4, move_scalar_dim, dtype=torch.float32)
    if pokemon is None:
        return block
    for move_idx, move in enumerate(list(pokemon.moves.values())[:4]):
        acc = getattr(move, "accuracy", 100)
        if acc is True:
            acc_int = 100
        elif isinstance(acc, (int, float)):
            acc_int = int(acc if acc > 1.0 else acc * 100)
        else:
            acc_int = 100
        block[move_idx, 0] = float(max(0, min(100, acc_int)))
        block[move_idx, 1] = float(max(0, int(getattr(move, "current_pp", 0) or 0)))
        block[move_idx, 2] = float(max(0, int(getattr(move, "base_power", 0) or 0)))
        if getattr(move, "type", None) in TYPE_LIST:
            block[move_idx, 3 + TYPE_LIST.index(move.type)] = 1.0
        priority = int(getattr(move, "priority", 0) or 0)
        prio_idx = max(0, min(PRIORITY_BINS - 1, priority + PRIORITY_OFFSET))
        block[move_idx, 21 + prio_idx] = 1.0
        category = getattr(move, "category", None)
        category_name = category.name.upper() if category is not None else "STATUS"
        block[move_idx, 34 + MOVE_CATEGORY_TO_INDEX.get(category_name, 2)] = 1.0
    return block


def encode_status_flags(pokemon: Pokemon | None) -> Tensor:
    status_block = torch.zeros(7, dtype=torch.float32)
    if pokemon is None:
        return status_block
    status = pokemon.status if pokemon.status in STATUS_LIST else None
    status_block[STATUS_LIST.index(status)] = 1.0
    return status_block


def encode_type_flags(pokemon: Pokemon | None) -> Tensor:
    type_block = torch.zeros(18, dtype=torch.float32)
    if pokemon is None:
        return type_block
    for pokemon_type in [pokemon.type_1, pokemon.type_2]:
        if pokemon_type in TYPE_LIST:
            type_block[TYPE_LIST.index(pokemon_type)] = 1.0
    return type_block


def enrich_obs_with_poke_env(battle: Battle, unpacker: ObservationUnpacker, kb: PokeEnvKnowledgeBase) -> dict[str, Tensor]:
    obs_flat_np = embed_battle(battle)
    obs_flat = torch.from_numpy(obs_flat_np).float().unsqueeze(0)
    obs = unpacker(obs_flat)
    obs["global_scalars"][0, 0] = float(getattr(battle, "turn", 0)) * 0.01
    species_ids = torch.zeros(1, 12, dtype=torch.long)
    ability_ids = torch.zeros(1, 12, 4, dtype=torch.long)
    move_ids = torch.zeros(1, 12, 4, dtype=torch.long)
    slots = ordered_battle_slots(battle)
    for slot_idx, pokemon in enumerate(slots[:12]):
        is_ally = slot_idx < 6
        species_ids[0, slot_idx] = kb.encode_species(pokemon)
        ability_ids[0, slot_idx] = torch.tensor(kb.encode_abilities(pokemon), dtype=torch.long)
        move_ids[0, slot_idx] = torch.tensor(kb.encode_moves(pokemon), dtype=torch.long)
        obs["move_scalars"][0, slot_idx] = encode_move_scalars(pokemon)
        obs["pokemon_scalars"][0, slot_idx, 1] = float(level_estimate(pokemon, unpacker.default_level))
        stats_8, meta = stat_estimate(pokemon)
        obs["pokemon_scalars"][0, slot_idx, 2:10] = torch.tensor(stats_8, dtype=torch.float32)
        obs["pokemon_flags"][0, slot_idx, :18] = encode_type_flags(pokemon)
        obs["pokemon_flags"][0, slot_idx, 18:25] = encode_status_flags(pokemon)
        obs["pokemon_flags"][0, slot_idx, 25] = float(pokemon is not None and bool(getattr(pokemon, "active", False)))
        obs["pokemon_flags"][0, slot_idx, 26] = float(is_ally)
        obs["pokemon_flags"][0, slot_idx, 27] = float(meta["species_known"])
        obs["pokemon_flags"][0, slot_idx, 28] = float(meta["stats_exact"])
        obs["pokemon_flags"][0, slot_idx, 29] = float(meta["stats_estimated"])
        obs["pokemon_flags"][0, slot_idx, 30] = float(pokemon is not None and bool(getattr(pokemon, "fainted", False)))
        obs["pokemon_flags"][0, slot_idx, 31] = float(pokemon is not None and bool(getattr(pokemon, "terastallized", False)))
        obs["pokemon_flags"][0, slot_idx, 32] = float(pokemon is not None)
        if pokemon is not None:
            status_counter = int(max(0, min(6, int(getattr(pokemon, "status_counter", 0) or 0))))
            obs["pokemon_flags"][0, slot_idx, 33 + status_counter] = 1.0
    obs["pokemon_species_ids"] = species_ids.to(obs_flat.device)
    obs["ability_ids"] = ability_ids.to(obs_flat.device)
    obs["move_ids"] = move_ids.to(obs_flat.device)
    return obs


class PipelineObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, kb: PokeEnvKnowledgeBase, unpacker: ObservationUnpacker) -> None:
        super().__init__(env)
        self.kb = kb
        self.unpacker = unpacker
        self._pokerl_env = getattr(env, "_pokerl_env", None)
        self.observation_space = spaces.Dict(
            {
                "global_scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32),
                "pokemon_species_ids": spaces.Box(low=0, high=max(1, len(kb.species_index.stoi)), shape=(12,), dtype=np.int64),
                "ability_ids": spaces.Box(low=0, high=max(1, len(kb.ability_index.stoi)), shape=(12, 4), dtype=np.int64),
                "move_ids": spaces.Box(low=0, high=max(1, len(kb.move_index.stoi)), shape=(12, 4), dtype=np.int64),
                "move_scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(12, 4, 37), dtype=np.float32),
                "pokemon_scalars": spaces.Box(low=-np.inf, high=np.inf, shape=(12, 17), dtype=np.float32),
                "pokemon_flags": spaces.Box(low=-np.inf, high=np.inf, shape=(12, 40), dtype=np.float32),
            }
        )
        self.action_space = env.action_space

    def _zero_obs(self) -> dict[str, np.ndarray]:
        return {
            "global_scalars": np.zeros((33,), dtype=np.float32),
            "pokemon_species_ids": np.zeros((12,), dtype=np.int64),
            "ability_ids": np.zeros((12, 4), dtype=np.int64),
            "move_ids": np.zeros((12, 4), dtype=np.int64),
            "move_scalars": np.zeros((12, 4, 37), dtype=np.float32),
            "pokemon_scalars": np.zeros((12, 17), dtype=np.float32),
            "pokemon_flags": np.zeros((12, 40), dtype=np.float32),
        }

    def _current_battle(self) -> Battle | None:
        if self._pokerl_env is None:
            return None
        return self._pokerl_env.battle1

    def _convert_obs(self) -> dict[str, np.ndarray]:
        battle = self._current_battle()
        if battle is None:
            return self._zero_obs()
        structured = enrich_obs_with_poke_env(battle, self.unpacker, self.kb)
        return {
            "global_scalars": structured["global_scalars"][0].detach().cpu().numpy().astype(np.float32),
            "pokemon_species_ids": structured["pokemon_species_ids"][0].detach().cpu().numpy().astype(np.int64),
            "ability_ids": structured["ability_ids"][0].detach().cpu().numpy().astype(np.int64),
            "move_ids": structured["move_ids"][0].detach().cpu().numpy().astype(np.int64),
            "move_scalars": structured["move_scalars"][0].detach().cpu().numpy().astype(np.float32),
            "pokemon_scalars": structured["pokemon_scalars"][0].detach().cpu().numpy().astype(np.float32),
            "pokemon_flags": structured["pokemon_flags"][0].detach().cpu().numpy().astype(np.float32),
        }

    def reset(self, **kwargs):
        _obs, info = self.env.reset(**kwargs)
        return self._convert_obs(), info

    def step(self, action):
        _obs, reward, terminated, truncated, info = self.env.step(action)
        return self._convert_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()


class PipelineStatsCallback(BaseCallback):
    def __init__(self, enabled: bool, log_freq_steps: int, wandb_module: Any | None = None):
        super().__init__()
        self.enabled = enabled
        self.log_freq_steps = max(1, int(log_freq_steps))
        self.wandb_module = wandb_module

    def _collect_stats(self) -> dict[str, float]:
        policy = self.model.policy
        payload: dict[str, float] = {}
        stats = getattr(policy, "last_stats", {})
        if isinstance(stats, dict):
            for key, value in stats.items():
                try:
                    payload[f"pipeline/{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
        sq_norm = 0.0
        for param in policy.pipeline.parameters():
            sq_norm += float(param.detach().pow(2).sum().item())
        payload["pipeline/param_norm"] = math.sqrt(max(sq_norm, 0.0))
        return payload

    def _log_stats(self) -> None:
        if not self.enabled:
            return
        payload = self._collect_stats()
        for key, value in payload.items():
            self.logger.record(key, value)
        if self.wandb_module is not None and self.wandb_module.run is not None and payload:
            self.wandb_module.log(payload, step=self.num_timesteps)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq_steps == 0:
            self._log_stats()
        return True

    def _on_training_end(self) -> None:
        self._log_stats()


class PipelineMaskablePolicy(MaskableActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *args,
        pipeline_cfg: PokemonEncoderConfig | None = None,
        g_map: dict[str, int | tuple[int, ...] | list[int]] | None = None,
        **kwargs,
    ):
        self.base_pipeline_cfg = pipeline_cfg or PokemonEncoderConfig()
        self.g_map = g_map or {"turn_int": 0, "remainder_raw": (1, None)}
        self.last_stats: dict[str, float] = {}
        super().__init__(observation_space, action_space, lr_schedule, *args, net_arch=[], features_extractor_class=CombinedExtractor, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = nn.Identity()

    def _resolved_cfg(self) -> PokemonEncoderConfig:
        return replace(self.base_pipeline_cfg, n_actions=int(self.action_space.n))

    def _build(self, lr_schedule: Schedule) -> None:
        self.action_dist = make_masked_proba_distribution(self.action_space)
        self.pipeline = PokeTransformer(self._resolved_cfg())
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _obs_to_pipeline(self, obs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {
            "global_scalars": obs["global_scalars"].float(),
            "pokemon_species_ids": obs["pokemon_species_ids"].long(),
            "ability_ids": obs["ability_ids"].long(),
            "move_ids": obs["move_ids"].long(),
            "move_scalars": obs["move_scalars"].float(),
            "pokemon_scalars": obs["pokemon_scalars"].float(),
            "pokemon_flags": obs["pokemon_flags"].float(),
        }

    def _update_stats(self, output: TransformerForwardOutput, action_masks: Optional[np.ndarray] = None) -> None:
        with torch.no_grad():
            probs = torch.softmax(output.policy_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
            mask_density = float(np.mean(action_masks)) if action_masks is not None else 1.0
            self.last_stats = {
                "value_mean": float(output.value.mean().detach().cpu()),
                "value_std": float(output.value.std(unbiased=False).detach().cpu()),
                "policy_entropy": float(entropy.detach().cpu()),
                "pokemon_token_norm": float(output.pokemon.pokemon_tokens.norm(dim=-1).mean().detach().cpu()),
                "field_token_norm": float(output.field.field_token.norm(dim=-1).mean().detach().cpu()),
                "mask_density": mask_density,
            }

    def _distribution_from_output(self, output: TransformerForwardOutput, action_masks: Optional[np.ndarray] = None):
        distribution = self.action_dist.proba_distribution(action_logits=output.policy_logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(self, obs: Mapping[str, Tensor], deterministic: bool = False, action_masks: Optional[np.ndarray] = None):
        output = self.pipeline.forward_from_obs(self._obs_to_pipeline(obs), g_map=self.g_map, action_mask=None)
        distribution = self._distribution_from_output(output, action_masks=action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = output.value.unsqueeze(-1)
        self._update_stats(output, action_masks=action_masks)
        return actions, values, log_prob

    def evaluate_actions(self, obs: Mapping[str, Tensor], actions: Tensor, action_masks: Optional[Tensor] = None):
        output = self.pipeline.forward_from_obs(self._obs_to_pipeline(obs), g_map=self.g_map, action_mask=None)
        distribution = self._distribution_from_output(output, action_masks=action_masks.detach().cpu().numpy() if isinstance(action_masks, Tensor) else action_masks)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = output.value.unsqueeze(-1)
        self._update_stats(output, action_masks=action_masks.detach().cpu().numpy() if isinstance(action_masks, Tensor) else action_masks)
        return values, log_prob, entropy

    def predict_values(self, obs: Mapping[str, Tensor]) -> Tensor:
        output = self.pipeline.forward_from_obs(self._obs_to_pipeline(obs), g_map=self.g_map, action_mask=None)
        self._update_stats(output)
        return output.value.unsqueeze(-1)

    def get_distribution(self, obs: Mapping[str, Tensor], action_masks: Optional[np.ndarray] = None):
        output = self.pipeline.forward_from_obs(self._obs_to_pipeline(obs), g_map=self.g_map, action_mask=None)
        self._update_stats(output, action_masks=action_masks)
        return self._distribution_from_output(output, action_masks=action_masks)

    def _predict(self, observation: Mapping[str, Tensor], deterministic: bool = False, action_masks: Optional[np.ndarray] = None) -> Tensor:
        return self.get_distribution(observation, action_masks=action_masks).get_actions(deterministic=deterministic)


def make_pipeline_policy_kwargs(action_space_n: int, kb: PokeEnvKnowledgeBase | None = None) -> dict[str, Any]:
    kb = kb or PokeEnvKnowledgeBase(gen=9)
    return {
        "pipeline_cfg": replace(
            PokemonEncoderConfig(),
            move_vocab_size=max(2048, len(kb.move_index.stoi) + 1),
            ability_vocab_size=max(512, len(kb.ability_index.stoi) + 1),
            species_vocab_size=max(2048, len(kb.species_index.stoi) + 1),
            n_actions=int(action_space_n),
        ),
        "g_map": {"turn_int": 0, "remainder_raw": (1, None)},
    }
