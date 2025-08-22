
"""Transformer-based Actor-Critic policy for Stable-Baselines3.

Defines:
    - TransformerFeaturesExtractor
    - TransformerActorCritic

Key fixes vs your last version:
    * features_dim now matches the projector output (no 64 vs 256 mismatch)
    * cleaned memory handling
    * summed log_prob/entropy for continuous actions (SB3 buffer expects 1-D)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="xformers")

# ---------------- xformers guard ---------------- #
HAVE_XFORMERS = False
try:
    import xformers.ops as xops  # type: ignore
    HAVE_XFORMERS = True
except Exception:
    xops = None  # type: ignore

try:
    # deprecated path – keep if user has old xformers
    from xformers.components.attention import MultiHeadAttention as XfMultiHeadAttention  # type: ignore
except Exception:
    XfMultiHeadAttention = None  # type: ignore


# ---------------- small utils ---------------- #

def _build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 128]) -> nn.Sequential:  # noqa: B006
    layers: List[nn.Module] = []
    last = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), nn.ReLU(inplace=True)]
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings for relative attention."""

    def __init__(self, dim: int, max_seq_len: int = 512) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().view(max_seq_len, 1, dim), persistent=False)
        self.register_buffer("sin", emb.sin().view(max_seq_len, 1, dim), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(1)
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------- core blocks ---------------- #

class TransformerBlock(nn.Module):
    """Single transformer encoder block with optional memory and xformers fallback."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rotary: bool = True,
        window_size: Optional[int] = None,
        use_memory: bool = False,
        memory_size: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_rotary = use_rotary
        self.window_size = window_size
        self.use_memory = use_memory
        self.memory_size = memory_size

        self.use_xformers = HAVE_XFORMERS and XfMultiHeadAttention is not None

        if self.use_xformers:
            self.attn = XfMultiHeadAttention(embed_dim, num_heads, dropout=dropout)  # type: ignore
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )

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

        if use_memory and memory_size > 0:
            self.register_buffer("memory", torch.zeros(1, memory_size, embed_dim), persistent=False)

    def _create_window_mask(self, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.window_size is None or self.window_size >= seq_len:
            return None
        mask = torch.ones(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 0
        return mask.bool()

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # ---- memory handling (batch aligned) ----
        if self.use_memory and self.memory_size > 0:
            if self.memory.size(0) != batch_size:
                self.memory = x.new_zeros(batch_size, self.memory_size, self.embed_dim)
            full_seq = torch.cat([self.memory, x], dim=1)  # [B, mem+seq, D]
        else:
            full_seq = x

        # Windowed mask
        if self.window_size is not None:
            window_mask = self._create_window_mask(seq_len, x.device)
            attn_mask = window_mask if attn_mask is None else (attn_mask | window_mask)

        # Rotary
        if self.use_rotary:
            cos, sin = self.rope(full_seq)
            q = full_seq.view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads)
            k = q.clone()
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q = q.view(batch_size, -1, self.embed_dim)
            k = k.view(batch_size, -1, self.embed_dim)
        else:
            q = k = full_seq

        # Attention
        if self.use_xformers:
            h, _ = self.attn(
                query=q[:, -seq_len:],
                key=k,
                value=full_seq,
                attn_mask=attn_mask,
                need_weights=False,
            )
        else:
            h, _ = self.attn(q[:, -seq_len:], k, full_seq, attn_mask=attn_mask)

        # Residual + norm
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))

        # Update memory
        if self.use_memory and self.memory_size > 0 and update_memory:
            with torch.no_grad():
                if x.size(1) >= self.memory_size:
                    self.memory = x[:, -self.memory_size:].detach()
                else:
                    keep = self.memory_size - x.size(1)
                    self.memory = torch.cat([self.memory[:, -keep:], x], dim=1).detach()
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

        self.local_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rotary=use_rotary,
                    window_size=min(16, 2 ** (i + 2)),
                    use_memory=use_memory,
                    memory_size=memory_size // 2,
                )
                for i in range(num_layers // 2)
            ]
        )

        self.global_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_rotary=use_rotary,
                    window_size=None,
                    use_memory=use_memory,
                    memory_size=memory_size,
                )
                for _ in range(num_layers - num_layers // 2)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.local_layers:
            x = layer(x)
        for layer in self.global_layers:
            x = layer(x)
        return x

    def reset_memory(self) -> None:
        for layer in (*self.local_layers, *self.global_layers):
            if getattr(layer, "use_memory", False) and hasattr(layer, "memory"):
                layer.memory.zero_()


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """Feature extractor using hierarchical transformer with memory."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        seq_len: int = 20,
        dropout: float = 0.1,
        use_rotary: bool = True,
        use_memory: bool = True,
        memory_size: int = 64,
        proj_out_dim: int = 64,  # <<<<<< you set this to 64 before
    ) -> None:
        super().__init__(observation_space, features_dim=proj_out_dim)

        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.use_memory = use_memory
        # self.features_dim = proj_out_dim  # SB3 reads this

        obs_size = 0
        for space in observation_space.spaces.values():
            if isinstance(space, spaces.Box):
                obs_size += int(np.prod(space.shape))
            else:
                raise ValueError(f"Unsupported sub-space: {space}")

        self.input_linear = nn.Linear(obs_size, embed_dim)

        max_len = 1024
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.normal_(self.pos_emb, std=0.01)

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
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, proj_out_dim),
            nn.GELU(),
            nn.Linear(proj_out_dim, proj_out_dim),
        )
        self._features_dim = proj_out_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq_tensors: List[torch.Tensor] = []
        for k in sorted(observations.keys()):
            obs = observations[k]
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)  # [B, 1, feat]
            elif obs.dim() > 3:
                b, seq, *spatial = obs.shape
                obs = obs.view(b, seq, -1)
            seq_tensors.append(obs)

        x = torch.cat(seq_tensors, dim=-1)  # [B, seq, raw_dim]
        x = self.input_linear(x)            # [B, seq, embed]

        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :].to(x.device)

        x = self.transformer(x)
        x = self.final_ln(x)

        x_mean = x.mean(dim=1)
        x_last = x[:, -1]
        x_combined = 0.5 * (x_mean + x_last)
        return self.output_projection(x_combined)

    def reset_memory(self) -> None:
        if self.use_memory:
            self.transformer.reset_memory()


class StateSequencePredictor(nn.Module):
    """Autoregressive GRU decoder that generates a sequence of future states.

    This is intentionally separate from the actor/critic path and is meant
    to be used as an auxiliary head during training only.
    """

    def __init__(
        self,
        embed_dim: int,
        state_dim: int,
        horizon: int = 200,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        # Project encoder embedding to initial GRU hidden state
        self.init_proj = nn.Linear(embed_dim, self.hidden_dim * self.num_layers)

        # GRU decoder consumes previous predicted state (or start token)
        self.gru = nn.GRU(
            input_size=self.state_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Output projection from GRU hidden -> predicted state
        self.out_proj = nn.Linear(self.hidden_dim, self.state_dim)

        # Learnable start token for the autoregressive decoder
        self.register_parameter("start_token", nn.Parameter(torch.zeros(self.state_dim)))

    def forward(self, encoder_embedding: torch.Tensor, teacher_forcing: bool = False, teacher_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        encoder_embedding: [B, embed_dim]
        teacher_seq (optional): [B, H, state_dim] if teacher_forcing True

        returns: preds [B, H, state_dim]
        """
        B = encoder_embedding.size(0)

        # init hidden: [num_layers, B, hidden_dim]
        hidden = self.init_proj(encoder_embedding).view(self.num_layers, B, self.hidden_dim)

        prev = self.start_token.unsqueeze(0).expand(B, -1).to(encoder_embedding.device)

        preds: List[torch.Tensor] = []
        for t in range(self.horizon):
            out, hidden = self.gru(prev.unsqueeze(1), hidden)
            next_state = self.out_proj(out.squeeze(1))
            preds.append(next_state)

            if teacher_forcing and teacher_seq is not None:
                prev = teacher_seq[:, t]
            else:
                # detach to avoid gradients flowing through predictions
                prev = next_state.detach()

        return torch.stack(preds, dim=1)


# ---------------- Policy ---------------- #

class TransformerActorCritic(ActorCriticPolicy):
    """SB3 policy wrapper that plugs in our Transformer extractor."""

    def __init__(
        self,
        *args: Any,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
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
            "proj_out_dim": 64,  # keep consistent!
        }
        features_kwargs.update(transformer_kwargs)

        kwargs.setdefault("features_extractor_class", TransformerFeaturesExtractor)
        kwargs.setdefault("features_extractor_kwargs", features_kwargs)

        super().__init__(*args, **kwargs)

        latent_dim = self.features_extractor.features_dim  # should be 64 now

        if isinstance(self.action_space, spaces.Discrete):
            action_dim = self.action_space.n
            self.action_net = _build_mlp(latent_dim, action_dim, [256, 256])
        elif isinstance(self.action_space, spaces.Box):
            action_dim = int(np.prod(self.action_space.shape))
            self.action_mean = _build_mlp(latent_dim, action_dim, [256, 256])
            self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")

        self.value_net = _build_mlp(latent_dim, 1, [256, 256, 128])
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        features = self.extract_features(obs)
        values = self.value_net(features).flatten()

        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(features)
            dist = self.action_dist.proba_distribution(action_logits=logits)
            actions = dist.get_actions(deterministic=deterministic)
            log_prob = dist.log_prob(actions)
        else:
            mean_actions = self.action_mean(features)
            log_std = self.action_log_std.expand_as(mean_actions)
            dist = self.action_dist.proba_distribution(mean_actions, log_std)
            actions = dist.get_actions(deterministic=deterministic)
            log_prob = dist.log_prob(actions)

        # rollout buffer expects shape (batch,)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(-1)
        return actions, values, log_prob

    # ----- Predictor helper API (auxiliary head) -----
    def create_predictor_head(self, horizon: int, state_dim: int, hidden_dim: int = 256, num_layers: int = 1) -> None:
        """Attach a StateSequencePredictor auxiliary head to this policy.

        This does not affect the forward/action selection path.
        """
        if hasattr(self, "predictor") and self.predictor is not None:
            return
        embed_dim = getattr(self.features_extractor, "_features_dim", None) or self.features_extractor.features_dim
        self.predictor = StateSequencePredictor(embed_dim=embed_dim, state_dim=state_dim, horizon=horizon, hidden_dim=hidden_dim, num_layers=num_layers)
        # place on same device as policy
        try:
            self.predictor.to(next(self.parameters()).device)
        except StopIteration:
            pass

    def predict_future(self, obs: Dict[str, torch.Tensor], teacher_forcing: bool = False, teacher_seq: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Run the predictor head (if present) to produce future state sequence.

        Returns None if no predictor is attached.
        """
        if not hasattr(self, "predictor") or self.predictor is None:
            return None
        with torch.no_grad() if not teacher_forcing else torch.enable_grad():
            features = self.extract_features(obs)
            return self.predictor(features, teacher_forcing=teacher_forcing, teacher_seq=teacher_seq)

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        actions, _, _ = self.forward(observation, deterministic)
        return actions

    def evaluate_actions(
        self, obs: Dict[str, torch.Tensor], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        values = self.value_net(features).flatten()
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(features)
            dist = self.action_dist.proba_distribution(action_logits=logits)
        else:
            mean_actions = self.action_mean(features)
            log_std = self.action_log_std.expand_as(mean_actions)
            dist = self.action_dist.proba_distribution(mean_actions, log_std)

        log_prob = dist.log_prob(actions)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(-1)
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(-1)
        return values, log_prob, entropy

    def reset_memory(self) -> None:
        if hasattr(self.features_extractor, "reset_memory"):
            self.features_extractor.reset_memory()

    def get_action_logits(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return raw action logits for discrete actions or action means for continuous.

        This provides a consistent interface for KD. Expects `obs` as the same
        format accepted by `forward`/`evaluate_actions` (dict of tensors).
        """
        features = self.extract_features(obs)
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(features)
            return logits
        else:
            # For continuous actions we return the mean as a proxy for logits.
            mean_actions = self.action_mean(features)
            return mean_actions