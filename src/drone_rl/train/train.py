"""End-to-end training entry-point."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from stable_baselines3.common.utils import set_random_seed, get_linear_fn
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# local
from drone_rl.models.transformer_policy import TransformerActorCritic
from drone_rl.models.baselines import SimpleLSTMPolicy, DronePositionController
from drone_rl.utils.metrics import count_parameters, estimate_flops

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.cuda import is_available as cuda_available  # noqa: WPS433
except ImportError:  # pragma: no cover
    cuda_available = lambda: False  # type: ignore

# Optional performer/perceiver
try:
    from drone_rl.models.performer import PerformerActorCritic
    PERFORMER_AVAILABLE = True
except ImportError:
    PERFORMER_AVAILABLE = False
    PerformerActorCritic = None  # type: ignore

try:
    from drone_rl.models.perceiver import PerceiverActorCritic
    PERCEIVER_AVAILABLE = True
except ImportError:
    PERCEIVER_AVAILABLE = False
    PerceiverActorCritic = None  # type: ignore


POLICIES = {
    "transformer": TransformerActorCritic,
    "lstm": SimpleLSTMPolicy,
}
if PERFORMER_AVAILABLE:
    POLICIES["performer"] = PerformerActorCritic
if PERCEIVER_AVAILABLE:
    POLICIES["perceiver"] = PerceiverActorCritic


# ---------------- utils ---------------- #
def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def make_env(env_id: str, seed: int, rank: int = 0, capture_video: bool = False, run_dir: Optional[Path] = None) -> Callable:
    def _init() -> gym.Env:
        try:
            import flycraft  # noqa: F401
        except ImportError:
            pass
        env = gym.make(env_id, max_episode_steps=1000)
        env.reset(seed=seed + rank)

        if capture_video and run_dir is not None:
            from gymnasium.wrappers import RecordVideo
            video_dir = run_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            env = RecordVideo(
                env,
                video_dir,
                episode_trigger=lambda x: x % 100 == 0,
                name_prefix=f"{env_id.split('-')[0]}",
            )
        return env

    return _init


# ---------- Callbacks ---------- #
class SequencePredictionCallback(BaseCallback):
    """Evaluate model's next-state sequence prediction (if implemented)."""

    def __init__(
        self,
        eval_env,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        horizon: int = 200,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.horizon = horizon
        self.best_mse = float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        if not hasattr(self.model.policy, "predict_next_states"):
            self.logger.record("eval/seq_prediction_mse", 0.0)
            return True

        mse_values: List[float] = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            # VecEnv returns np.ndarray or tuple: (obs, info) only on gym Env
            if isinstance(obs, tuple):
                obs, _ = obs
            done = False
            true_states, observations = [], []
            steps = 0
            while not done and steps < self.horizon:
                observations.append(obs)
                action, _ = self.model.predict(obs, deterministic=True)
                next_obs, rewards, dones, infos = self.eval_env.step(action)
                info = infos[0]
                if "state" in info:
                    true_states.append(info["state"])
                else:
                    true_states.append(next_obs if not isinstance(next_obs, tuple) else next_obs[0])
                obs = next_obs
                done = dones[0]
                steps += 1

            preds = self.model.policy.predict_next_states(observations[0], self.horizon)
            L = min(len(true_states), len(preds))
            if L > 0:
                true_np = np.array(true_states[:L])
                pred_np = np.array(preds[:L])
                true_np = true_np.reshape(L, -1)
                pred_np = pred_np.reshape(L, -1)

                mse_values.append(float(np.mean((true_np - pred_np) ** 2)))

        avg_mse = np.mean(mse_values) if mse_values else float("inf")
        self.logger.record("eval/seq_prediction_mse", avg_mse)
        if avg_mse < self.best_mse:
            self.best_mse = avg_mse
            self.logger.record("eval/best_seq_prediction_mse", self.best_mse)
        return True


class ModelComplexityCallback(BaseCallback):
    def __init__(self, sample_input: Optional[Dict[str, torch.Tensor]] = None, verbose: int = 1):
        super().__init__(verbose)
        self.sample_input = sample_input

    def _on_training_start(self) -> None:
        param_count = count_parameters(self.model.policy)
        self.logger.record("model/parameters", param_count)

        if self.sample_input is not None:
            flops = estimate_flops(self.model.policy, self.sample_input)
            if flops > 0:
                self.logger.record("model/flops_per_forward", flops)

        if hasattr(self.model.policy, "features_extractor"):
            self.logger.record("model/extractor_type", type(self.model.policy.features_extractor).__name__)

    def _on_step(self) -> bool:
        return True


# ---------- KD utils ---------- #
def student_loss(student_out: Tuple, teacher_out: Tuple, temperature: float = 1.0, alpha: float = 0.5) -> torch.Tensor:
    logits_s, values_s = student_out
    logits_t, values_t = teacher_out

    kl = F.kl_div(
        F.log_softmax(logits_s / temperature, dim=-1),
        F.softmax(logits_t / temperature, dim=-1).detach(),
        reduction="batchmean",
    )
    mse = F.mse_loss(values_s, values_t.detach())
    return alpha * kl + (1 - alpha) * mse


class StateSequencePredictor(nn.Module):
    """Autoregressive GRU decoder for future state prediction."""

    def __init__(self, embed_dim: int, state_dim: int, horizon: int = 10, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.horizon = horizon

        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.decoder = nn.GRU(input_size=state_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, embedding: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(embedding).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        current_state = initial_state
        preds = []
        for _ in range(self.horizon):
            out, hidden = self.decoder(current_state.unsqueeze(1), hidden)
            next_state = self.output_proj(out.squeeze(1))
            preds.append(next_state)
            current_state = next_state
        return torch.stack(preds, dim=1)  # [B, horizon, state_dim]


# ---------------- main ---------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train drone navigation policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--sweep", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_random_seed(seed)

    env_id = cfg.get("env_id", "FlyCraft-v0")
    n_envs = cfg.get("n_envs", 8)

    output_dir = Path(cfg.get("output_dir", "runs"))
    run_name = cfg.get("run_name", f"{env_id.split('-')[0]}_{int(time.time())}")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.get("wandb_project", "drone-transformer-rl"),
            name=run_name,
            config=cfg,
            sync_tensorboard=True,
            monitor_gym=True,
        )

    # Vec envs
    env_fns = [make_env(env_id, seed, i, args.capture_video, run_dir) for i in range(n_envs)]
    env = VecMonitor(SubprocVecEnv(env_fns))

    eval_env = VecMonitor(SubprocVecEnv([make_env(env_id, seed + 1000, 0, args.capture_video, run_dir)]))

    device = args.device or ("cuda" if cuda_available() else "cpu")

    policy_name = cfg.get("policy", "transformer")
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(POLICIES.keys())}")
    policy_cls = POLICIES[policy_name]

    # performer/perceiver switches
    if policy_name == "transformer" and cfg.get("use_performer", False) and PERFORMER_AVAILABLE:
        policy_cls = POLICIES["performer"]

    policy_kwargs = cfg.get("policy_kwargs", {})
    ppo_kwargs = cfg.get("ppo_kwargs", {})
    ppo_kwargs.setdefault("verbose", 1)
    ppo_kwargs.setdefault("target_kl", None)
    ppo_kwargs.setdefault("device", device)

    ppo_kwargs["learning_rate"] = get_linear_fn(3e-4, 5e-6, 1.0)  # start→end over training frac
    ppo_kwargs["clip_range"]    = get_linear_fn(0.2, 0.05, 1.0)

    # clean fx kwargs
    fx_kwargs = policy_kwargs.get("features_extractor_kwargs", {})
    fx_kwargs.pop("use_spatio_temporal", None)
    policy_kwargs["features_extractor_kwargs"] = fx_kwargs
    policy_kwargs.setdefault("transformer_kwargs", {})
    policy_kwargs["transformer_kwargs"].setdefault("attn_backend", "torch")

    model = PPO(
        policy=policy_cls,
        env=env,
        tensorboard_log=str(run_dir / "tb"),
        policy_kwargs=policy_kwargs,
        **ppo_kwargs,
    )

    # -------- Attach sequence predictor (FIXED embed_dim) -------- #
    if cfg.get("predict_sequence", False):
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            state_dim = int(sum(np.prod(sp.shape) for sp in obs_space.spaces.values()))
        else:
            state_dim = int(np.prod(obs_space.shape))

        embed_dim_for_seq = model.policy.features_extractor.features_dim  # <<<<< 64 (matches projector)
        seq_predictor = StateSequencePredictor(
            embed_dim=embed_dim_for_seq,
            state_dim=state_dim,
            horizon=cfg.get("prediction_horizon", 200),
            hidden_dim=cfg.get("decoder_hidden_dim", 256),
            num_layers=cfg.get("decoder_layers", 2),
        ).to(model.policy.device)

        def _flatten_first_env(obs_t):
            if isinstance(obs_t, dict):
                return torch.cat([v[0].reshape(-1) for v in obs_t.values()], dim=0)
            return obs_t[0].reshape(-1)

        @torch.no_grad()
        def predict_next_states(obs0, horizon=None):
            if horizon is None:
                horizon = cfg.get("prediction_horizon", 200)
            obs_t, _ = model.policy.obs_to_tensor(obs0)
            emb = model.policy.extract_features(obs_t)
            assert emb.shape[-1] == embed_dim_for_seq, f"Embed dim mismatch {emb.shape[-1]} vs {embed_dim_for_seq}"
            init_state = _flatten_first_env(obs_t)
            preds = seq_predictor(embedding=emb[0].unsqueeze(0), initial_state=init_state.unsqueeze(0))
            return preds.cpu().numpy()[0][:horizon]

        model.policy.state_predictor = seq_predictor
        model.policy.predict_next_states = predict_next_states  # type: ignore

    # logger
    model.set_logger(configure(str(run_dir), ["stdout", "csv", "tensorboard"]))

    # KD
    if args.teacher:
        print(f"Loading teacher model from {args.teacher} for knowledge distillation")
        teacher = PPO.load(args.teacher, env=env, device=device)

        def custom_distillation_loss(model_outputs):
            pi_logits, values, old_logprobs, advantages, returns, entropy_coef, vf_coef = model_outputs
            with torch.no_grad():
                t_actions, t_values, t_log_probs = teacher.policy.evaluate_actions(model.policy._last_obs)
            kd_loss = student_loss(
                (pi_logits, values),
                (t_log_probs, t_values),
                temperature=cfg.get("distillation_temperature", 2.0),
                alpha=cfg.get("distillation_alpha", 0.5),
            )
            kd_weight = cfg.get("distillation_weight", 0.5)
            ppo_loss = model.policy.loss(pi_logits, values, old_logprobs, advantages, returns, entropy_coef, vf_coef)
            return (1 - kd_weight) * ppo_loss + kd_weight * kd_loss

        model.policy.custom_loss = custom_distillation_loss  # type: ignore

    # callbacks
    callbacks: List[BaseCallback] = [
        CheckpointCallback(
            save_freq=cfg.get("save_freq", 10000),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=run_name,
            save_replay_buffer=False,
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval"),
            eval_freq=cfg.get("eval_freq", 10000),
            n_eval_episodes=cfg.get("n_eval_episodes", 5),
            deterministic=True,
            render=args.capture_video,
        ),
        ModelComplexityCallback(),
    ]

    if cfg.get("predict_sequence", False):
        callbacks.append(
            SequencePredictionCallback(
                eval_env=eval_env,
                eval_freq=cfg.get("eval_freq", 10000),
                n_eval_episodes=cfg.get("n_eval_episodes", 5),
                horizon=cfg.get("prediction_horizon", 200),
            )
        )

    if args.wandb and WANDB_AVAILABLE:
        callbacks.append(WandbCallback())

    # Train
    if not args.sweep:
        timesteps = cfg.get("timesteps", 1_000_000)
        model.learn(total_timesteps=timesteps, callback=callbacks)
    else:
        raise NotImplementedError("Sweep path trimmed; plug in your Optuna code if needed.")

    model.save(run_dir / cfg.get("save_name", "final_model"))

    env.close()
    eval_env.close()
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    print(f"Training complete. Model saved to {run_dir}")


if __name__ == "__main__":
    main()