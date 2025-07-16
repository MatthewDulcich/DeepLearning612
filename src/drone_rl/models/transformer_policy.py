"""Transformer-based Actor-Critic policy for Stable-Baselines3.

This module defines:
    - TransformerFeaturesExtractor: converts state dict to embedding sequence
    - TransformerActorCritic: heads for action logits + value

Design:
    * Multi-scale (hierarchical) encoder: windowed self-attention at lower levels +
      global attention at the top.
    * Relative rotary position encodings via xformers.ops.
    * Memory from previous timesteps (Transformer-XL style) for long horizon.

Note: For brevity, some parts are simplified but hooks for additional depth are provided.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from xformers.components.attention import MultiHeadAttention
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import RotaryEmbeddingBias
    XFORMERS_AVAILABLE = True
except ImportError:
    MultiHeadAttention = None  # type: ignore
    XFORMERS_AVAILABLE = False


def _build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]) -> nn.Sequential:  # noqa: B006
    layers: List[nn.Module] = []
    last = input_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(last, h), nn.ReLU(inplace=True)])
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings for relative attention."""
    
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Initialize frequency for each dimension
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
        
        # Pre-compute positional embeddings
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().view(max_seq_len, 1, dim))
        self.register_buffer("sin", emb.sin().view(max_seq_len, 1, dim))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin embeddings for the input sequence."""
        seq_len = x.size(1)
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # q, k: [batch, seq_len, n_heads, head_dim]
    # cos, sin: [seq_len, 1, dim]
    
    # Reshape for broadcasting
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with optional memory."""

    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        use_rotary: bool = True,
        window_size: Optional[int] = None,
        use_memory: bool = False,
        memory_size: int = 0
    ) -> None:
        super().__init__()
        if not XFORMERS_AVAILABLE:
            raise ImportError("Install xformers for MultiHeadAttention")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_rotary = use_rotary
        self.window_size = window_size
        self.use_memory = use_memory
        self.memory_size = memory_size
        
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        
        if use_rotary:
            self.rope = RotaryPositionalEmbedding(embed_dim // num_heads, max_seq_len=1024)
        
        # Memory for Transformer-XL style recurrence
        if use_memory:
            self.register_buffer("memory", torch.zeros(1, memory_size, embed_dim))
            
    def _create_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create attention mask for windowed attention."""
        if self.window_size is None or self.window_size >= seq_len:
            return None
            
        # Create mask where each token can only attend to nearby tokens
        mask = torch.ones(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 0
        
        # Convert to boolean mask where True blocks attention
        return mask.bool()

    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True
    ) -> torch.Tensor:  # noqa: D401
        batch_size, seq_len, _ = x.shape
        
        # Combine with memory for Transformer-XL style recurrence
        if self.use_memory and self.memory.size(0) == batch_size:
            full_seq = torch.cat([self.memory, x], dim=1)
            mem_len = self.memory.size(1)
        else:
            full_seq = x
            mem_len = 0
            
        # Create windowed attention mask if needed
        if self.window_size is not None:
            window_mask = self._create_window_mask(seq_len, x.device)
            if attn_mask is not None:
                attn_mask = attn_mask | window_mask
            else:
                attn_mask = window_mask
        
        # Apply rotary position embeddings
        if self.use_rotary:
            cos, sin = self.rope(full_seq)
            
            # Reshape for attention computation
            q = full_seq.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads)
            k = q.clone()
            
            # Apply rotary embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
            # Reshape back
            q = q.view(batch_size, -1, self.embed_dim)
            k = k.view(batch_size, -1, self.embed_dim)
            
            # Compute attention with rotary embeddings
            h, _ = self.attn(
                query=q[:, -seq_len:], 
                key=k, 
                value=full_seq, 
                attn_mask=attn_mask, 
                need_weights=False
            )
        else:
            # Standard attention without rotary embeddings
            h, _ = self.attn(
                query=full_seq[:, -seq_len:], 
                key=full_seq, 
                value=full_seq, 
                attn_mask=attn_mask, 
                need_weights=False
            )
        
        # Residual connection and layer norm
        x = x + h
        x = self.ln1(x)
        
        # Feed-forward network
        x = x + self.ff(x)
        x = self.ln2(x)
        
        # Update memory for next forward pass
        if self.use_memory and update_memory:
            with torch.no_grad():
                self.memory = x[:, -self.memory_size:].detach() if x.size(1) >= self.memory_size else \
                              torch.cat([self.memory[:, -(self.memory_size - x.size(1)):], x], dim=1).detach()
                
        return x


class HierarchicalTransformer(nn.Module):
    """Multi-scale hierarchical transformer with local and global attention."""
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rotary: bool = True,
        use_memory: bool = True,
        memory_size: int = 64,
    ) -> None:
        super().__init__()
        
        # Lower layers use windowed attention (local)
        # Upper layers use full attention (global)
        self.local_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_rotary=use_rotary,
                window_size=min(16, 2**(i+2)),  # Increasing window sizes: 4, 8, 16
                use_memory=use_memory,
                memory_size=memory_size // 2,  # Smaller memory for local layers
            )
            for i in range(num_layers // 2)
        ])
        
        # Global attention layers
        self.global_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_rotary=use_rotary,
                window_size=None,  # Full attention
                use_memory=use_memory,
                memory_size=memory_size,
            )
            for i in range(num_layers - num_layers // 2)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply local attention layers
        for layer in self.local_layers:
            x = layer(x)
            
        # Apply global attention layers
        for layer in self.global_layers:
            x = layer(x)
            
        return x
    
    def reset_memory(self) -> None:
        """Reset memory for all layers."""
        for layer in self.local_layers:
            if layer.use_memory:
                layer.memory.zero_()
        for layer in self.global_layers:
            if layer.use_memory:
                layer.memory.zero_()


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor using hierarchical transformer with memory."""
    
    def __init__(
        self,
        observation_space: spaces.Dict,  # expects dict of sensor arrays
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        seq_len: int = 20,
        dropout: float = 0.1,
        use_rotary: bool = True,
        use_memory: bool = True,
        memory_size: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim=embed_dim)
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.use_memory = use_memory

        # Flatten numeric observations into one vector per timestep
        obs_size = 0
        for key, space in observation_space.items():
            if isinstance(space, spaces.Box):
                obs_size += int(np.prod(space.shape))
            else:
                raise ValueError(f"Unsupported sub-space {key}: {space}")

        # Embedding layer for input observations
        self.input_linear = nn.Linear(obs_size, embed_dim)
        
        # Standard positional embeddings (used alongside rotary embeddings)
        self.pos_emb = nn.Parameter(torch.zeros(seq_len, embed_dim))
        nn.init.normal_(self.pos_emb, std=0.01)  # Initialize with small random values
        
        # Hierarchical transformer encoder
        self.transformer = HierarchicalTransformer(
            embed_dim=embed_dim,
            num_layers=depth,
            num_heads=num_heads,
            dropout=dropout,
            use_rotary=use_rotary,
            use_memory=use_memory,
            memory_size=memory_size,
        )
        
        self.final_ln = nn.LayerNorm(embed_dim)
        
        # Optional projection for output features
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore  # noqa: D401
        # observations: dict of [B, seq, dim]
        seq_tensors: List[torch.Tensor] = []
        
        # Concatenate all observation components
        for k in sorted(observations.keys()):
            # Handle different observation shapes
            obs = observations[k]
            if len(obs.shape) > 3:  # For image-like observations
                b, seq, *spatial = obs.shape
                obs = obs.view(b, seq, -1)  # Flatten spatial dimensions
            seq_tensors.append(obs)
            
        x = torch.cat(seq_tensors, dim=-1)  # [B, seq, raw_dim]
        
        # Project to embedding dimension
        x = self.input_linear(x)  # [B, seq, embed]
        
        # Add positional embeddings
        x = x + self.pos_emb[:x.size(1)]  # Add positional embeddings (broadcast over batch)
        
        # Apply transformer layers
        x = self.transformer(x)
        
        # Apply final layer norm
        x = self.final_ln(x)
        
        # We have several options for the output:
        # 1. Last token only (like BERT [CLS])
        # 2. Mean pooling across sequence
        # 3. Weighted attention pooling
        
        # Using mean pooling for more stable training
        x_mean = x.mean(dim=1)  # [B, embed]
        x_last = x[:, -1]  # [B, embed]
        
        # Combine mean and last token with learnable projection
        x_combined = (x_mean + x_last) / 2
        return self.output_projection(x_combined)
    
    def reset_memory(self) -> None:
        """Reset memory for the transformer."""
        if self.use_memory:
            self.transformer.reset_memory()


class TransformerActorCritic(ActorCriticPolicy):
    """SB3 policy wrapper that plugs in our Transformer extractor."""

    def __init__(
        self, 
        *args: Any, 
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        # Set default feature extractor parameters
        if transformer_kwargs is None:
            transformer_kwargs = {}
            
        features_kwargs = {
            "embed_dim": 256,
            "depth": 6,
            "num_heads": 8,
            "seq_len": 20,
            "dropout": 0.1,
            "use_rotary": True,
            "use_memory": True,
            "memory_size": 64,
        }
        features_kwargs.update(transformer_kwargs)
        
        kwargs.setdefault("features_extractor_class", TransformerFeaturesExtractor)
        kwargs.setdefault("features_extractor_kwargs", features_kwargs)
        
        super().__init__(*args, **kwargs)

        # Actor & critic MLP heads with larger capacity
        latent_dim = self.features_extractor.features_dim
        
        # Handle different action spaces
        if isinstance(self.action_space, spaces.Discrete):
            action_dim = self.action_space.n
            self.action_net = _build_mlp(latent_dim, action_dim, [256, 256])
        elif isinstance(self.action_space, spaces.Box):
            action_dim = int(np.prod(self.action_space.shape))
            # For continuous actions, output mean and log_std
            self.action_mean = _build_mlp(latent_dim, action_dim, [256, 256])
            self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")
            
        # Value network with deeper architecture for better value estimation
        self.value_net = _build_mlp(latent_dim, 1, [256, 256, 128])
        
        # Initialize weights with orthogonal initialization
        self.apply(self._weights_init)
        
    @staticmethod
    def _weights_init(module: nn.Module) -> None:
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Override predict for SB3."""
        features = self.extract_features(obs)
        
        # Get action distribution
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(features)
            dist = self.action_dist.proba_distribution(action_logits=logits)
        else:  # Continuous actions
            mean_actions = self.action_mean(features)
            # Use state-independent log std
            log_std = self.action_log_std.expand_as(mean_actions)
            dist = self.action_dist.proba_distribution(mean_actions, log_std)
            
        actions = dist.get_actions(deterministic=deterministic)
        values = self.value_net(features)
        return actions, values, dist.distribution.logits if hasattr(dist.distribution, "logits") else mean_actions

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """SB3 prediction method."""
        actions, _, _ = self.forward(observation, deterministic)
        return actions

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions according to the current policy given observations."""
        features = self.extract_features(obs)
        
        # Get values
        values = self.value_net(features).flatten()
        
        # Get action distribution
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(features)
            dist = self.action_dist.proba_distribution(action_logits=logits)
        else:  # Continuous actions
            mean_actions = self.action_mean(features)
            log_std = self.action_log_std.expand_as(mean_actions)
            dist = self.action_dist.proba_distribution(mean_actions, log_std)
            
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_prob, entropy
    
    def reset_memory(self) -> None:
        """Reset memory for the transformer feature extractor."""
        if hasattr(self.features_extractor, "reset_memory"):
            self.features_extractor.reset_memory()
