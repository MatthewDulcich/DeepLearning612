"""End-to-end training entry-point."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn, set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize

# Local
from drone_rl.models.transformer_policy import TransformerActorCritic
from drone_rl.models.baselines import SimpleLSTMPolicy, DronePositionController  # optional
from drone_rl.utils.metrics import count_parameters, estimate_flops
from drone_rl.train.lstm_callback import LSTMResetCallback

# wandb optional
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# cuda check
try:
    from torch.cuda import is_available as cuda_available  # noqa: WPS433
except ImportError:
    cuda_available = lambda: False  # type: ignore

# Optional other policies
try:
    from drone_rl.models.performer import PerformerActorCritic
    PERFORMER_AVAILABLE = True
except Exception:
    PERFORMER_AVAILABLE = False
    PerformerActorCritic = None  # type: ignore

try:
    from drone_rl.models.perceiver import PerceiverActorCritic
    PERCEIVER_AVAILABLE = True
except Exception:
    PERCEIVER_AVAILABLE = False
    PerceiverActorCritic = None  # type: ignore

POLICIES: Dict[str, Any] = {
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
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def make_env(
    env_id: str,
    seed: int,
    rank: int = 0,
    capture_video: bool = False,
    run_dir: Optional[Path] = None,
    max_episode_steps: int = 1000,
    # NEW curriculum args:
    step_frequence: int = 10,
    control_mode: str = "guidance_law_mode",
    reward_mode: str = "dense",
    goal_cfg: Optional[dict] = None,
    debug: bool = False,  # NEW: Add debug wrapper option
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        try:
            import flycraft  # noqa: F401
        except ImportError:
            pass
        env = gym.make(
            env_id,
            max_episode_steps=max_episode_steps,
            step_frequence=step_frequence,
            control_mode=control_mode,
            reward_mode=reward_mode,
            goal_cfg=goal_cfg or {"type": "fixed_short", "distance_m": 200},
        )
        env.reset(seed=seed + rank)

        # Add debug wrapper for first environment
        if debug and rank == 0:
            from drone_rl.utils.flycraft_debug_wrapper import FlyCraftDebugWrapper, RewardAnalysisWrapper
            env = FlyCraftDebugWrapper(env, verbose=True)
            env = RewardAnalysisWrapper(env)

        if capture_video and rank == 0 and run_dir is not None:
            env = gym.wrappers.RecordVideo(env, str(run_dir / "videos"))

        return env

    return _init


def _flatten_np(obs: Any) -> np.ndarray:
    """Flatten dict or array obs/state to 1D numpy array."""
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            v = np.asarray(v)
            parts.append(v.reshape(-1))
        return np.concatenate(parts, axis=0)
    arr = np.asarray(obs)
    return arr.reshape(-1)


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
            return True

        mse_values: List[float] = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs

            dones = [False]
            true_states: List[np.ndarray] = []
            observations: List[Any] = []
            steps = 0
            while not dones[0] and steps < self.horizon:
                observations.append(obs)
                action, _ = self.model.predict(obs, deterministic=True)
                next_obs, _, dones, infos = self.eval_env.step(action)

                info = infos[0]
                state_ref = info.get("state", None)
                if state_ref is None:
                    state_ref = next_obs if not isinstance(next_obs, tuple) else next_obs[0]
                true_states.append(_flatten_np(state_ref))

                obs = next_obs
                steps += 1

            try:
                preds = self.model.policy.predict_next_states(observations[0], steps)
            except Exception as e:
                self.logger.record("eval/seq_prediction_error", str(e))
                continue

            preds = np.asarray(preds).reshape(steps, -1)
            L = min(len(true_states), preds.shape[0])
            if L == 0:
                continue
            true_np = np.stack(true_states[:L], axis=0)
            pred_np = preds[:L]
            if true_np.shape != pred_np.shape:
                m = min(true_np.shape[1], pred_np.shape[1])
                true_np = true_np[:, :m]
                pred_np = pred_np[:, :m]

            mse_values.append(float(np.mean((true_np - pred_np) ** 2)))

        if mse_values:
            avg_mse = float(np.mean(mse_values))
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

    def __init__(self, embed_dim: int, state_dim: int, horizon: int = 10,
                 hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.decoder = nn.GRU(state_dim, hidden_dim, num_layers=num_layers, batch_first=True)
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
    max_episode_steps = cfg.get("max_episode_steps", 1000)

    # Curriculum scheduler for step frequency
    step_frequencies = [10, 20, 50, 100]
    success_threshold = 0.8
    curriculum_log = []

    for freq in step_frequencies:
        print(f"\n=== Training with step_frequency={freq}Hz ===")
        
        # Get stage-specific curriculum settings from config (with fallbacks)
        goal_cfg = cfg.get("goal_cfg_by_freq", {}).get(str(freq), {"type": "fixed_short", "distance_m": 200})
        control_mode = cfg.get("control_mode_by_freq", {}).get(str(freq), "guidance_law_mode")
        reward_mode = cfg.get("reward_mode_by_freq", {}).get(str(freq), "dense")
        
        print(f"[Curriculum] Using: freq={freq}Hz, control_mode={control_mode}, reward_mode={reward_mode}, goal_cfg={goal_cfg}")
        
        # dirs
        output_dir = Path(cfg.get("output_dir", "runs"))
        run_name = cfg.get("run_name", f"{env_id.split('-')[0]}_{int(time.time())}_f{freq}")
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

        # Vec envs + VecNormalize - now with curriculum args
        env_fns = [make_env(
            env_id, seed, i, args.capture_video, run_dir, max_episode_steps,
            step_frequence=freq,
            control_mode=control_mode,
            reward_mode=reward_mode,
            goal_cfg=goal_cfg,
            debug=(i == 0)  # Enable debug wrapper for first env only
        ) for i in range(n_envs)]
        # Use DummyVecEnv instead of SubprocVecEnv to avoid multiprocessing issues
        train_env = DummyVecEnv(env_fns)
        train_env = VecMonitor(train_env)
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            gamma=cfg.get("ppo_kwargs", {}).get("gamma", 0.99),
        )

        eval_env_fns = [make_env(
            env_id, seed + 1000, 0, args.capture_video, run_dir, max_episode_steps,
            step_frequence=freq,
            control_mode=control_mode,
            reward_mode=reward_mode,
            goal_cfg=goal_cfg
        )]
        eval_env = DummyVecEnv(eval_env_fns)
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            training=False,
            gamma=cfg.get("ppo_kwargs", {}).get("gamma", 0.99),
        )
        # share obs stats
        eval_env.obs_rms = train_env.obs_rms

        device = args.device or ("cuda" if cuda_available() else "cpu")

        # --------- policy/model setup (as before) ---------
        policy_name = cfg.get("policy", "transformer")
        if policy_name not in POLICIES:
            raise ValueError(f"Unknown policy: {policy_name}. Available: {list(POLICIES.keys())}")
        policy_cls = POLICIES[policy_name]
        if policy_name == "transformer" and cfg.get("use_performer", False) and PERFORMER_AVAILABLE:
            policy_cls = POLICIES["performer"]

        policy_kwargs = cfg.get("policy_kwargs", {})
        if isinstance(train_env.action_space, gym.spaces.Box):
            policy_kwargs["log_std_init"] = -0.3
        fx_kwargs = policy_kwargs.get("features_extractor_kwargs", {})
        fx_kwargs.pop("use_spatio_temporal", None)
        policy_kwargs["features_extractor_kwargs"] = fx_kwargs

        # Only merge transformer_kwargs for transformer/performer
        merged_policy_kwargs = policy_kwargs
        if policy_name in ["transformer", "performer"]:
            transformer_kwargs = cfg.get("transformer_kwargs", {})
            transformer_kwargs.setdefault("attn_backend", "torch")
            merged_policy_kwargs = {**policy_kwargs, **transformer_kwargs}

        ppo_kwargs = cfg.get("ppo_kwargs", {})
        ppo_kwargs.setdefault("verbose", 1)
        ppo_kwargs.setdefault("device", device)
        ppo_kwargs.setdefault("vf_coef", 1.0)
        ppo_kwargs.setdefault("ent_coef", 0.01)
        ppo_kwargs.setdefault("max_grad_norm", 0.5)
        ppo_kwargs.setdefault("n_epochs", 2)
        ppo_kwargs.setdefault("batch_size", 2048)
        ppo_kwargs["learning_rate"] = get_linear_fn(1e-4, 5e-6, 1.0)
        ppo_kwargs["clip_range"] = get_linear_fn(0.2, 0.1, 1.0)
        ppo_kwargs["target_kl"] = None

        model = PPO(
            policy=policy_cls,
            env=train_env,
            tensorboard_log=str(run_dir / "tb"),
            policy_kwargs=merged_policy_kwargs,
            **ppo_kwargs,
        )

        # logger
        model.set_logger(configure(str(run_dir), ["stdout", "csv", "tensorboard"]))

        # callbacks
        save_freq = max(cfg.get("save_freq", 10000) // n_envs, 1)
        eval_freq = max(cfg.get("eval_freq", 20000) // n_envs, 1)
        callbacks: List[BaseCallback] = [
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(run_dir / "checkpoints"),
                name_prefix=run_name,
                save_replay_buffer=False,
                save_vecnormalize=True,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir / "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=cfg.get("n_eval_episodes", 5),
                deterministic=True,
                render=args.capture_video,
            ),
            ModelComplexityCallback(),
        ]
        
        # Add LSTM reset callback if using LSTM policy
        if policy_name == "lstm":
            callbacks.append(LSTMResetCallback(verbose=1))
        if args.wandb and WANDB_AVAILABLE:
            callbacks.append(WandbCallback())

        # train
        timesteps = cfg.get("timesteps", 1_000_000)
        model.learn(total_timesteps=timesteps, callback=callbacks)

        # Evaluate success rate (VecEnv safe)
        n_eval = cfg.get("n_eval_episodes", 10)
        successes = 0
        for _ in range(n_eval):
            obs = eval_env.reset()
            done = np.array([False])
            while not done[0]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)  # VecEnv returns arrays
            info0 = info[0]
            if info0.get("success", False) or info0.get("is_success", False):
                successes += 1
        success_rate = successes / n_eval
        curriculum_log.append({"step_frequency": freq, "success_rate": success_rate})
        print(f"[Curriculum] step_frequency={freq}Hz, success_rate={success_rate:.2f}")
        
        # ENHANCED DEBUGGING: Log detailed eval results
        current_stage = step_frequencies.index(freq) + 1
        total_stages = len(step_frequencies)
        print(f"🔍 DETAILED EVALUATION RESULTS for {freq}Hz (Stage {current_stage}/{total_stages}):")
        print(f"  - Episodes evaluated: {n_eval}")
        print(f"  - Successful episodes: {successes}")
        print(f"  - Success rate: {success_rate:.2%}")
        print(f"  - Success threshold: {success_threshold:.2%}")
        
        # Log individual episode details if success rate is low
        if success_rate < 0.3:
            print("⚠️  LOW SUCCESS RATE - Debug info:")
            print(f"   - Training timesteps: {timesteps}")
            print(f"   - Reward mode: {reward_mode}")
            print(f"   - Goal config: {goal_cfg}")

        # Save model and env
        model.save(run_dir / cfg.get("save_name", f"final_model_f{freq}"))
        train_env.save(str(run_dir / f"vecnormalize_f{freq}.pkl"))
        train_env.close()
        eval_env.close()
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

        if success_rate < success_threshold:
            print(f"❌ [Curriculum] Stopping: success_rate {success_rate:.2f} < threshold {success_threshold}")
            remaining_stages = total_stages - current_stage
            print(f"   Failed at stage {current_stage}/{total_stages} ({remaining_stages} stages remaining)")
            break
        else:
            remaining_stages = total_stages - current_stage
            if current_stage < total_stages:
                next_freq = step_frequencies[current_stage] if current_stage < len(step_frequencies) else "FINAL"
                print(f"✅ [Curriculum] SUCCESS! Advancing from {freq}Hz to {next_freq}Hz")
                print(f"   ({remaining_stages} stages remaining)")
            else:
                print(f"🎉 [Curriculum] FINAL STAGE COMPLETED! Training successful at {freq}Hz")

    print(f"[Curriculum] Training completed. Log: {curriculum_log}")


if __name__ == "__main__":
    main()
