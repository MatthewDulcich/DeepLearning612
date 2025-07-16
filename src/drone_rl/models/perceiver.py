"""PerceiverIO-based Actor-Critic policy with sensor-fusion cross-attention.

This module implements a PerceiverIO architecture for sensor fusion in drone control:
1. Each input modality (position, velocity, lidar rays) is embedded separately
2. Cross-attention fuses modalities into a fixed-size latent array
3. Self-attention processes the latent array
4. Final latent is used for policy and value prediction

Features:
- Modality-specific embeddings
- Cross-attention for sensor fusion
- Fixed compute regardless of input size
- Compatible with SB3 ActorCriticPolicy interface

We use lucidrains' perceiver-io if available, with fallback to a
simplified cross-attention implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

try:
    from perceiver_pytorch import PerceiverIO  # type: ignore
    PERCEIVER_AVAILABLE = True
except ImportError:
    PERCEIVER_AVAILABLE = False


def _build_mlp(inp: int, out: int, h: int = 256) -> nn.Sequential:  # noqa: D401
    """Build a simple MLP with GELU activation.
    
    Parameters
    ----------
    inp : int
        Input dimension
    out : int
        Output dimension
    h : int
        Hidden dimension
        
    Returns
    -------
    nn.Sequential
        MLP module
    """
    return nn.Sequential(nn.Linear(inp, h), nn.GELU(), nn.Linear(h, out))


class SimpleCrossAttention(nn.Module):
    """Simple cross-attention module as fallback when perceiver-pytorch is not available."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        num_latents: int = 64,
        dropout: float = 0.1
    ):
        """Initialize cross-attention module.
        
        Parameters
        ----------
        dim : int
            Embedding dimension
        num_heads : int
            Number of attention heads
        num_latents : int
            Number of latent tokens
        dropout : float
            Dropout probability
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.head_dim = dim // num_heads
        
        # Learnable latent array
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        
        # Projection matrices
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention between latents and inputs.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch_size, seq_len, dim]
            
        Returns
        -------
        torch.Tensor
            Updated latent array [batch_size, num_latents, dim]
        """
        batch_size = x.size(0)
        
        # Expand latents to batch size
        latents = self.latents.expand(batch_size, -1, -1)
        
        # Cross-attention: latents attend to inputs
        q = self.q_proj(latents)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, self.num_latents, self.dim)
        
        # Output projection
        out = self.o_proj(out)
        
        # Residual connection and layer norm
        latents = latents + self.dropout(out)
        latents = self.norm1(latents)
        
        # Feed-forward network
        latents = latents + self.ff(latents)
        latents = self.norm2(latents)
        
        return latents


class PerceiverFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor using PerceiverIO for sensor fusion.
    
    This extractor:
    1. Embeds each input modality separately
    2. Uses cross-attention to fuse modalities into a fixed-size latent array
    3. Processes the latent array with self-attention
    4. Returns the final latent for policy and value prediction
    """
    
    def __init__(
        self, 
        observation_space: spaces.Dict, 
        latent_dim: int = 256,
        num_latents: int = 64,
        num_cross_heads: int = 1,
        num_self_heads: int = 8,
        num_self_layers: int = 4,
        dropout: float = 0.1
    ):
        """Initialize PerceiverIO feature extractor.
        
        Parameters
        ----------
        observation_space : spaces.Dict
            Observation space (dictionary of spaces)
        latent_dim : int
            Dimension of latent vectors
        num_latents : int
            Number of latent vectors
        num_cross_heads : int
            Number of cross-attention heads
        num_self_heads : int
            Number of self-attention heads
        num_self_layers : int
            Number of self-attention layers
        dropout : float
            Dropout probability
        """
        super().__init__(observation_space, latent_dim)

        # Flatten each modality separately
        self.keys: List[str] = sorted(observation_space.keys())
        
        # Determine input dimensions for each modality
        input_dims = {}
        for k, space in observation_space.items():
            if isinstance(space, spaces.Box):
                # Handle different shapes
                if len(space.shape) == 1:  # Vector
                    input_dims[k] = int(space.shape[0])
                else:  # Sequence or image
                    flat_dim = int(np.prod(space.shape))
                    input_dims[k] = flat_dim
            else:
                raise ValueError(f"Unsupported space type for {k}: {space}")
        
        # Create embedding layers for each modality
        self.embed_layers = nn.ModuleDict({
            k: nn.Linear(d, latent_dim) for k, d in input_dims.items()
        })
        
        # Position embeddings for sequences
        self.max_seq_len = 100  # Maximum sequence length
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, latent_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        # Create PerceiverIO or fallback
        if PERCEIVER_AVAILABLE:
            self.perceiver = PerceiverIO(
                dim=latent_dim,                      # Dimension of embeddings
                queries_dim=latent_dim,              # Dimension of output queries
                logits_dim=None,                     # No classification head
                depth=num_self_layers,               # Number of self-attention layers
                num_latents=num_latents,             # Number of latent vectors
                latent_dim=latent_dim,               # Dimension of latent vectors
                cross_heads=num_cross_heads,         # Number of cross-attention heads
                latent_heads=num_self_heads,         # Number of self-attention heads
                cross_dim_head=latent_dim // num_cross_heads,  # Dimension per cross-attention head
                latent_dim_head=latent_dim // num_self_heads,  # Dimension per self-attention head
                dropout=dropout,                     # Dropout probability
            )
        else:
            # Fallback to simplified implementation
            self.cross_attn = SimpleCrossAttention(
                dim=latent_dim,
                num_heads=num_cross_heads,
                num_latents=num_latents,
                dropout=dropout
            )
            
            # Self-attention layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_self_heads,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            )
            self.self_attn = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_self_layers
            )
            
            # Output query
            self.query = nn.Parameter(torch.zeros(1, 1, latent_dim))
            nn.init.normal_(self.query, std=0.02)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process observations through PerceiverIO.
        
        Parameters
        ----------
        obs : Dict[str, torch.Tensor]
            Dictionary of observation tensors
            
        Returns
        -------
        torch.Tensor
            Extracted features [batch_size, latent_dim]
        """
        batch_size = next(iter(obs.values())).size(0)
        
        # Process each modality
        embeddings = []
        for k in self.keys:
            x = obs[k]
            
            # Reshape if needed
            if len(x.shape) > 3:  # For image-like observations
                b, *spatial = x.shape
                x = x.reshape(b, -1)
            
            # Apply embedding
            emb = self.embed_layers[k](x)
            
            # Add positional embedding for sequences
            if len(emb.shape) > 2:  # [batch, seq, dim]
                seq_len = min(emb.size(1), self.max_seq_len)
                emb[:, :seq_len] = emb[:, :seq_len] + self.pos_embed[:, :seq_len]
            
            embeddings.append(emb)
        
        # Concatenate all embeddings
        if any(len(e.shape) > 2 for e in embeddings):
            # Handle sequence inputs by flattening
            flat_embeddings = []
            for emb in embeddings:
                if len(emb.shape) > 2:  # [batch, seq, dim]
                    flat_embeddings.append(emb.reshape(batch_size, -1, emb.size(-1)))
                else:  # [batch, dim]
                    # Add sequence dimension
                    flat_embeddings.append(emb.unsqueeze(1))
            
            # Concatenate along sequence dimension
            x = torch.cat(flat_embeddings, dim=1)
        else:
            # Simple concatenation for non-sequence inputs
            x = torch.cat(embeddings, dim=1)
        
        # Process through PerceiverIO or fallback
        if PERCEIVER_AVAILABLE:
            # Create output queries (just one query for final representation)
            queries = torch.zeros(batch_size, 1, x.size(-1), device=x.device)
            
            # Apply PerceiverIO
            out = self.perceiver(x, queries=queries)
            
            # Return the single query output
            return out.squeeze(1)
        else:
            # Apply cross-attention between latents and inputs
            latents = self.cross_attn(x)
            
            # Apply self-attention to process latents
            latents = self.self_attn(latents)
            
            # Use mean pooling for the final representation
            return latents.mean(dim=1)


class ACTLayer(nn.Module):
    """Adaptive Computation Time layer for early exit.
    
    This layer implements the ACT mechanism from "Adaptive Computation Time for
    Recurrent Neural Networks" (Graves, 2016). It allows the model to decide
    how many processing steps to perform based on the input complexity.
    """
    
    def __init__(self, hidden_dim: int, max_steps: int = 10, threshold: float = 0.05):
        """Initialize ACT layer.
        
        Parameters
        ----------
        hidden_dim : int
            Dimension of hidden state
        max_steps : int
            Maximum number of computation steps
        threshold : float
            Halting threshold
        """
        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        
        # Halting probability predictor
        self.halt_predictor = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor, processor: nn.Module) -> Tuple[torch.Tensor, int, float]:
        """Apply adaptive computation.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch_size, hidden_dim]
        processor : nn.Module
            Processing module to apply repeatedly
            
        Returns
        -------
        Tuple[torch.Tensor, int, float]
            Processed tensor, number of steps, ponder cost
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize state
        state = x
        
        # Initialize halting and outputs
        halting_prob = torch.zeros(batch_size, 1, device=device)
        remainders = torch.ones(batch_size, 1, device=device)
        n_updates = torch.zeros(batch_size, 1, device=device)
        
        # Initialize weighted sum of states
        weighted_state = torch.zeros_like(state)
        
        # Adaptive computation loop
        for step in range(self.max_steps):
            # Predict halting probability
            p = torch.sigmoid(self.halt_predictor(state))
            
            # Update halting probability
            halting_prob += p * remainders
            
            # Compute weights for state update
            update_weights = torch.min(p, remainders)
            
            # Update state
            weighted_state += state * update_weights
            
            # Update counters
            n_updates += update_weights
            remainders -= update_weights
            
            # Check if all samples have halted
            if (halting_prob >= (1.0 - self.threshold)).all():
                break
                
            # Apply processor for next step
            state = processor(state)
        
        # Add remainder
        weighted_state += state * remainders
        
        # Compute ponder cost (N + r)
        ponder_cost = n_updates.mean().item() + remainders.mean().item()
        
        return weighted_state, step + 1, ponder_cost


class PerceiverActorCritic(ActorCriticPolicy):
    """PerceiverIO-based actor-critic policy with adaptive computation time.
    
    This policy uses:
    1. PerceiverIO for sensor fusion
    2. Adaptive Computation Time for early exit
    3. MLP heads for policy and value prediction
    """
    
    def __init__(
        self, 
        *args: Any, 
        perceiver_kwargs: Optional[Dict[str, Any]] = None,
        use_act: bool = True,
        act_max_steps: int = 10,
        act_threshold: float = 0.05,
        **kwargs: Any
    ):
        """Initialize PerceiverIO-based actor-critic policy.
        
        Parameters
        ----------
        *args : Any
            Arguments for ActorCriticPolicy
        perceiver_kwargs : Optional[Dict[str, Any]]
            Arguments for PerceiverFeatureExtractor
        use_act : bool
            Whether to use Adaptive Computation Time
        act_max_steps : int
            Maximum number of computation steps for ACT
        act_threshold : float
            Halting threshold for ACT
        **kwargs : Any
            Additional arguments for ActorCriticPolicy
        """
        # Set default feature extractor
        if perceiver_kwargs is None:
            perceiver_kwargs = {}
            
        kwargs.setdefault("features_extractor_class", PerceiverFeatureExtractor)
        kwargs.setdefault("features_extractor_kwargs", perceiver_kwargs)
        
        super().__init__(*args, **kwargs)
        
        # Get latent dimension from feature extractor
        latent_dim = self.features_extractor.features_dim
        
        # Create policy and value heads
        if isinstance(self.action_space, spaces.Discrete):
            self.policy_net = _build_mlp(latent_dim, self.action_space.n)
        else:
            action_dim = int(np.prod(self.action_space.shape))
            self.policy_net = _build_mlp(latent_dim, action_dim * 2)  # Mean and log_std
            
        self.value_net = _build_mlp(latent_dim, 1)
        
        # Optional ACT module
        self.use_act = use_act
        if use_act:
            self.act = ACTLayer(
                hidden_dim=latent_dim,
                max_steps=act_max_steps,
                threshold=act_threshold
            )
            
            # Processor module for ACT
            self.processor = nn.Sequential(
                nn.Linear(latent_dim, latent_dim * 2),
                nn.GELU(),
                nn.Linear(latent_dim * 2, latent_dim)
            )
            
        # Track computation steps and ponder cost
        self.last_steps = 0
        self.last_ponder = 0.0

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through policy.
        
        Parameters
        ----------
        obs : Dict[str, torch.Tensor]
            Observation dictionary
        deterministic : bool
            Whether to sample deterministically
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Actions, values, and action parameters
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Apply ACT if enabled
        if self.use_act:
            features, steps, ponder = self.act(features, self.processor)
            self.last_steps = steps
            self.last_ponder = ponder
        
        # Compute policy and value
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.policy_net(features)
            dist = self.action_dist.proba_distribution(action_logits=logits)
            action_params = logits
        else:
            policy_params = self.policy_net(features)
            mu, log_std = torch.chunk(policy_params, 2, dim=-1)
            dist = self.action_dist.proba_distribution(mu, log_std)
            action_params = mu
            
        actions = dist.get_actions(deterministic=deterministic)
        values = self.value_net(features).flatten()
        
        return actions, values, action_params

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """Predict action given observation.
        
        Parameters
        ----------
        observation : Dict[str, torch.Tensor]
            Observation dictionary
        deterministic : bool
            Whether to sample deterministically
            
        Returns
        -------
        torch.Tensor
            Predicted actions
        """
        actions, _, _ = self.forward(observation, deterministic)
        return actions

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions according to current policy.
        
        Parameters
        ----------
        obs : Dict[str, torch.Tensor]
            Observation dictionary
        actions : torch.Tensor
            Actions to evaluate
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Values, log probabilities, and entropy
        """
        # Extract features
        features = self.extract_features(obs)
        
        # Apply ACT if enabled
        if self.use_act:
            features, steps, ponder = self.act(features, self.processor)
            self.last_steps = steps
            self.last_ponder = ponder
        
        # Compute values
        values = self.value_net(features).flatten()
        
        # Compute action distribution
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.policy_net(features)
            dist = self.action_dist.proba_distribution(action_logits=logits)
        else:
            policy_params = self.policy_net(features)
            mu, log_std = torch.chunk(policy_params, 2, dim=-1)
            dist = self.action_dist.proba_distribution(mu, log_std)
            
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return values, log_prob, entropy
