# model.py

# LE MODEL transformer imitant celui du mec, si gemini s'est pas branlé jai pas rechecké cetait pas le but

import json
import torch
import torch.nn as nn
import os
from typing import Any, Dict, Optional
from gymnasium.spaces import Space
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule

class PokeTransformerModule(TorchRLModule, ValueFunctionAPI):
    def __init__(self, observation_space: Space, action_space: Space, inference_only: bool, model_config: Dict[str, Any], catalog_class: Any):
        super().__init__(observation_space=observation_space, action_space=action_space, inference_only=inference_only, model_config=model_config, catalog_class=catalog_class)
        
        self.d_model = 128
        
        try:
            with open("vocab.json", "r") as f:
                vocab = json.load(f)
        except FileNotFoundError:
            vocab = {}

        num_species = len(vocab.get("pokemon.species", [])) + 1
        num_items = len(vocab.get("pokemon.item", [])) + 1
        num_abilities = len(vocab.get("pokemon.ability", [])) + 1
        num_moves = len(vocab.get("move.id", [])) + 1

        emb_dim = 16
        self.pokemon_emb = nn.Embedding(num_embeddings=num_species, embedding_dim=emb_dim)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=emb_dim)
        self.ability_emb = nn.Embedding(num_embeddings=num_abilities, embedding_dim=emb_dim)
        self.move_emb = nn.Embedding(num_embeddings=num_moves, embedding_dim=emb_dim)

        self.field_net = nn.Sequential(
            nn.Linear(200, self.d_model * 2), nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model), nn.LayerNorm(self.d_model)
        )
        
        pok_in_dim = (emb_dim * 7) + (200 - 7)
        self.pokemon_net = nn.Sequential(
            nn.Linear(pok_in_dim, self.d_model * 2), nn.GELU(),
            nn.Linear(self.d_model * 2, self.d_model), nn.LayerNorm(self.d_model)
        )

        self.actor_tok = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.critic_tok = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.total_tokens = 15 
        self.register_buffer("attn_mask", self._build_poke_mask())

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=self.d_model * 2, 
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.pi_head = nn.Linear(self.d_model, 26)
        self.v_head = nn.Linear(self.d_model, 1) 

    def _build_poke_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.total_tokens, self.total_tokens)
        mask[2:, 0:2] = float('-inf')  
        mask[0, 1] = float('-inf')     
        mask[1, 0] = float('-inf')     
        return mask

    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs = batch[Columns.OBS]["observations"] 
        action_mask = batch[Columns.OBS]["action_mask"]
        B = obs.shape[0]

        field_token = self.field_net(obs[:, 0, :]).unsqueeze(1) 

        pok_obs = obs[:, 1:, :] 
        categorical_ids = pok_obs[:, :, 0:7].long()
        continuous_features = pok_obs[:, :, 7:]

        sp_emb = self.pokemon_emb(categorical_ids[:, :, 0])
        it_emb = self.item_emb(categorical_ids[:, :, 1])
        ab_emb = self.ability_emb(categorical_ids[:, :, 2])
        m1_emb = self.move_emb(categorical_ids[:, :, 3])
        m2_emb = self.move_emb(categorical_ids[:, :, 4])
        m3_emb = self.move_emb(categorical_ids[:, :, 5])
        m4_emb = self.move_emb(categorical_ids[:, :, 6])

        pok_combined = torch.cat([sp_emb, it_emb, ab_emb, m1_emb, m2_emb, m3_emb, m4_emb, continuous_features], dim=-1)
        pok_tokens = self.pokemon_net(pok_combined) 

        act_t = self.actor_tok.expand(B, -1, -1)
        crit_t = self.critic_tok.expand(B, -1, -1)
        seq = torch.cat([act_t, crit_t, field_token, pok_tokens], dim=1) 
        
        transformed_seq = self.transformer(seq, mask=self.attn_mask)

        logits = self.pi_head(transformed_seq[:, 0, :])
        masked_logits = torch.where(action_mask > 0.5, logits, torch.tensor(-1e8, device=logits.device, dtype=logits.dtype))

        return {Columns.EMBEDDINGS: transformed_seq, Columns.ACTION_DIST_INPUTS: masked_logits}

    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        if embeddings is None:
            _ = self._forward(batch)
            embeddings = batch.get(Columns.EMBEDDINGS)
        return self.v_head(embeddings[:, 1, :]).squeeze(-1)

class MiniPokeformerModule(TorchRLModule, ValueFunctionAPI):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        inference_only: bool,
        model_config: Dict[str, Any],
        catalog_class: Any,
    ):
        # --- THE FIX: We MUST use keyword arguments (name=value) here ---
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            model_config=model_config,
            catalog_class=catalog_class,
        )
        
        # 1. THE EMBEDDER: Upgrades our 144 features into a rich 64-dimension token
        self.embedder = nn.Linear(144, 64)
        
        # 2. THE TRANSFORMER: 2 layers of Self-Attention (batch_first=True is required!)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. THE HEADS: We flatten the 13 tokens (13 * 64 = 832) to make the final decisions
        self.actor = nn.Linear(13 * 64, 26)
        self.critic = nn.Linear(13 * 64, 1)

    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        obs = batch[Columns.OBS]["observations"] # Shape: [Batch, 13, 144]
        action_mask = batch[Columns.OBS]["action_mask"]

        # Pass the 13x144 matrix through the Transformer
        x = self.embedder(obs)           # Shape becomes: [Batch, 13, 64]
        x = self.transformer(x)          # Shape remains: [Batch, 13, 64]
        
        # Flatten the sequence to feed into the Actor and Critic
        flat_embeddings = x.reshape(x.shape[0], -1) # Shape becomes: [Batch, 832]

        logits = self.actor(flat_embeddings)

        masked_logits = torch.where(
            action_mask > 0.5,
            logits,
            torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
        )

        return {Columns.EMBEDDINGS: flat_embeddings, Columns.ACTION_DIST_INPUTS: masked_logits}

    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        if embeddings is None:
            obs = batch[Columns.OBS]["observations"]
            x = self.embedder(obs)
            x = self.transformer(x)
            embeddings = x.reshape(x.shape[0], -1)
            
        return self.critic(embeddings).squeeze(-1)