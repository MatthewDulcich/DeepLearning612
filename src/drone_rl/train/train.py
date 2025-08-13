"""End-to-end training entry-point.

Supports:
• TransformerActorCritic (spatio-temporal, performer switch) – default
• SimpleLSTMPolicy / other SB3 policy names for baselines
• Knowledge-distillation (teacher -> student)
• Optuna sweeps via --sweep flag
• Memory-compressed decoder for next-state prediction

Usage examples
--------------
python -m drone_rl.train.train --config configs/teacher_large.yaml
python -m drone_rl.train.train --config configs/student_distilled.yaml --teacher runs/teacher/best.zip
python -m drone_rl.train.train --config configs/baseline_lstm.yaml
python -m drone_rl.train.train --config configs/teacher_large.yaml --sweep 32
"""
from __future__ import annotations


import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

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

# Local imports
from drone_rl.models.transformer_policy import TransformerActorCritic  # noqa: E402
from drone_rl.models.baselines import SimpleLSTMPolicy, DronePositionController  # noqa: E402
from drone_rl.utils.metrics import count_parameters, estimate_flops  # noqa: E402

# Optional performer import
try:
    from drone_rl.models.performer import PerformerActorCritic
    PERFORMER_AVAILABLE = True
except ImportError:
    PERFORMER_AVAILABLE = False
    PerformerActorCritic = None  # type: ignore

# Optional perceiver import
try:
    from drone_rl.models.perceiver import PerceiverActorCritic
    PERCEIVER_AVAILABLE = True
except ImportError:
    PERCEIVER_AVAILABLE = False
    PerceiverActorCritic = None  # type: ignore

# Map string names to policy classes (extendable)
POLICIES = {
    "transformer": TransformerActorCritic,
    "lstm": SimpleLSTMPolicy,
}

# Add performer if available
if PERFORMER_AVAILABLE:
    POLICIES["performer"] = PerformerActorCritic

# Add perceiver if available
if PERCEIVER_AVAILABLE:
    POLICIES["perceiver"] = PerceiverActorCritic


# ---------------- utils ---------------- #

def load_config(path: str | Path) -> Dict[str, Any]:  # noqa: D401
    """Load configuration from YAML or JSON file.
    
    Parameters
    ----------
    path : str | Path
        Path to configuration file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def make_env(env_id: str, seed: int, rank: int = 0, capture_video: bool = False,
             run_dir: Optional[Path] = None) -> Callable:  # noqa: D401
    """Create a function that returns a configured environment.
    
    Parameters
    ----------
    env_id : str
        Environment ID
    seed : int
        Random seed
    rank : int
        Process rank for vectorized environments
    capture_video : bool
        Whether to capture videos of episodes
    run_dir : Optional[Path]
        Directory to save videos
        
    Returns
    -------
    Callable
        Function that creates and returns an environment
    """
    def _init() -> gym.Env:
        # Import here to avoid importing gym when not needed
        import gymnasium as gym
        try:
            import flycraft_gym  # noqa: F401
        except ImportError:
            pass  # Allow other environments
            
        # Create environment with appropriate render mode
        render_mode = "rgb_array" if capture_video else None
        env = gym.make(env_id, render_mode=render_mode, max_episode_steps=1000)
        
        # Set environment seed
        env.reset(seed=seed + rank)
        
        # Wrap for video recording if requested
        if capture_video and run_dir is not None:
            from gymnasium.wrappers import RecordVideo
            video_dir = run_dir / "videos"
            video_dir.mkdir(exist_ok=True)
            env = RecordVideo(
                env, 
                video_dir,
                episode_trigger=lambda x: x % 100 == 0,  # Record every 100 episodes
                name_prefix=f"{env_id.split('-')[0]}"
            )
            
        return env
        
    return _init


class SequencePredictionCallback(BaseCallback):
    """Callback to evaluate sequence prediction performance.
    
    This callback evaluates the model's ability to predict future states,
    which is a key requirement for the time-sequence prediction task.
    """
    
    def __init__(self, eval_env: gym.Env, eval_freq: int = 1000, 
                 n_eval_episodes: int = 5, horizon: int = 200,
                 verbose: int = 1):
        """Initialize callback.
        
        Parameters
        ----------
        eval_env : gym.Env
            Environment for evaluation
        eval_freq : int
            Evaluation frequency in timesteps
        n_eval_episodes : int
            Number of episodes for evaluation
        horizon : int
            Prediction horizon in timesteps
        verbose : int
            Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.horizon = horizon
        self.best_mse = float("inf")
        
    def _on_step(self) -> bool:
        """Evaluate sequence prediction at regular intervals.
        
        Returns
        -------
        bool
            Whether training should continue
        """
        if self.n_calls % self.eval_freq != 0:
            return True
            
        # Only evaluate if model has a state prediction head
        if not hasattr(self.model.policy, "predict_next_states"):
            self.logger.record("eval/seq_prediction_mse", 0.0)
            return True
            
        mse_values = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            
            # Collect true trajectory
            true_states = []
            observations = []
            
            for _ in range(self.horizon):
                if done:
                    break
                    
                observations.append(obs)
                action, _ = self.model.predict(obs, deterministic=True)
                next_obs, _, terminated, truncated, info = self.eval_env.step(action)
                
                # Extract state from info if available, otherwise use observation
                if "state" in info:
                    true_states.append(info["state"])
                else:
                    # Fallback to observation if state not provided
                    true_states.append(next_obs)
                    
                obs = next_obs
                done = terminated or truncated
                
            # Predict future states
            predicted_states = self.model.policy.predict_next_states(observations[0], self.horizon)
            
            # Compute MSE
            min_len = min(len(true_states), len(predicted_states))
            if min_len > 0:
                # Convert to numpy arrays if they aren't already
                true_np = np.array(true_states[:min_len])
                pred_np = np.array(predicted_states[:min_len])
                
                # Compute MSE
                mse = np.mean((true_np - pred_np) ** 2)
                mse_values.append(mse)
        
        # Log average MSE
        avg_mse = np.mean(mse_values) if mse_values else float("inf")
        self.logger.record("eval/seq_prediction_mse", avg_mse)
        
        # Track best MSE
        if avg_mse < self.best_mse:
            self.best_mse = avg_mse
            self.logger.record("eval/best_seq_prediction_mse", self.best_mse)
            
        return True


class ModelComplexityCallback(BaseCallback):
    """Callback to log model complexity metrics.
    
    This callback computes and logs:
    - Parameter count
    - FLOPs per forward pass
    - Inference time
    """
    
    def __init__(self, sample_input: Optional[Dict[str, torch.Tensor]] = None, 
                 verbose: int = 1):
        """Initialize callback.
        
        Parameters
        ----------
        sample_input : Optional[Dict[str, torch.Tensor]]
            Sample input for FLOPs estimation
        verbose : int
            Verbosity level
        """
        super().__init__(verbose)
        self.sample_input = sample_input
        
    def _on_training_start(self) -> None:
        """Log model complexity metrics at training start."""
        # Count parameters
        param_count = count_parameters(self.model.policy)
        self.logger.record("model/parameters", param_count)
        
        # Estimate FLOPs if sample input provided
        if self.sample_input is not None:
            flops = estimate_flops(self.model.policy, self.sample_input)
            if flops > 0:  # Valid estimation
                self.logger.record("model/flops_per_forward", flops)
                
        # Log model type and configuration
        if hasattr(self.model.policy, "features_extractor"):
            extractor_type = type(self.model.policy.features_extractor).__name__
            self.logger.record("model/extractor_type", extractor_type)


# ------------- KD helper -------------- #

def student_loss(student_out: Tuple, teacher_out: Tuple, 
                 temperature: float = 1.0, alpha: float = 0.5) -> torch.Tensor:
    """Knowledge distillation loss combining KL divergence and MSE.
    
    Parameters
    ----------
    student_out : Tuple
        Student model outputs (logits, values)
    teacher_out : Tuple
        Teacher model outputs (logits, values)
    temperature : float
        Temperature for softmax
    alpha : float
        Weight for KL divergence vs MSE
        
    Returns
    -------
    torch.Tensor
        Combined loss
    """
    logits_s, values_s = student_out
    logits_t, values_t = teacher_out
    
    # KL divergence for action distribution matching
    kl = F.kl_div(
        F.log_softmax(logits_s / temperature, dim=-1),
        F.softmax(logits_t / temperature, dim=-1).detach(),
        reduction="batchmean",
    )
    
    # MSE for value function matching
    mse = F.mse_loss(values_s, values_t.detach())
    
    # Combined loss
    return alpha * kl + (1 - alpha) * mse


class StateSequencePredictor(nn.Module):
    """Decoder module for next-state sequence prediction.
    
    This module takes the current state embedding and predicts
    the next k states in the sequence, satisfying the time-sequence
    prediction requirement.
    """
    
    def __init__(self, embed_dim: int, state_dim: int, horizon: int = 10,
                 hidden_dim: int = 256, num_layers: int = 2):
        """Initialize state sequence predictor.
        
        Parameters
        ----------
        embed_dim : int
            Input embedding dimension
        state_dim : int
            State dimension to predict
        horizon : int
            Number of future states to predict
        hidden_dim : int
            Hidden dimension of decoder
        num_layers : int
            Number of decoder layers
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.horizon = horizon
        
        # Projection from embedding to hidden
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Decoder (autoregressive GRU)
        self.decoder = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, state_dim)
        
    def forward(self, embedding: torch.Tensor, initial_state: torch.Tensor) -> torch.Tensor:
        """Predict sequence of future states.
        
        Parameters
        ----------
        embedding : torch.Tensor
            Current state embedding [batch_size, embed_dim]
        initial_state : torch.Tensor
            Initial state [batch_size, state_dim]
            
        Returns
        -------
        torch.Tensor
            Predicted future states [batch_size, horizon, state_dim]
        """
        batch_size = embedding.shape[0]
        device = embedding.device
        
        # Project embedding to hidden state
        hidden = self.input_proj(embedding)
        hidden = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        # Initialize with current state
        current_state = initial_state
        
        # Generate future states autoregressively
        predicted_states = []
        for _ in range(self.horizon):
            # Reshape for GRU input [batch, 1, state_dim]
            decoder_input = current_state.unsqueeze(1)
            
            # Decode next state
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Project to state space
            next_state = self.output_proj(output.squeeze(1))
            
            # Save prediction
            predicted_states.append(next_state)
            
            # Update current state for next iteration
            current_state = next_state
            
        # Stack predictions [batch, horizon, state_dim]
        return torch.stack(predicted_states, dim=1)


# ------------- main ------------------- #

def main() -> None:  # noqa: D401
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train drone navigation policy")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--teacher", type=str, default=None, help="Path to teacher model for distillation")
    parser.add_argument("--sweep", type=int, default=None, help="Number of Optuna trials to run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--capture-video", action="store_true", help="Capture evaluation videos")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    
    # Set random seed
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_random_seed(seed)
    
    # Environment configuration
    env_id = cfg.get("env_id", "FlyCraft-Nav-v0")
    n_envs = cfg.get("n_envs", 8)
    
    # Output directory
    output_dir = Path(cfg.get("output_dir", "runs"))
    run_name = cfg.get("run_name", f"{env_id.split('-')[0]}_{int(time.time())}")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    
    # Initialize W&B if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project=cfg.get("wandb_project", "drone-transformer-rl"),
            name=run_name,
            config=cfg,
            sync_tensorboard=True,
            monitor_gym=True,
        )
    
    # Create vectorized environments
    env = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"max_episode_steps": cfg.get("max_episode_steps", 1000)},
    )
    env = VecMonitor(env)
    
    # Create evaluation environment
    eval_env = make_vec_env(
        env_id=env_id,
        n_envs=1,
        seed=seed + 1000,  # Different seed for evaluation
        env_kwargs={"max_episode_steps": cfg.get("max_episode_steps", 1000)},
    )
    
    # Device configuration
    device = args.device
    if device is None:
        device = "cuda" if cuda_available() else "cpu"
    
    # Policy selection and configuration
    policy_name = cfg.get("policy", "transformer")
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}. Available: {list(POLICIES.keys())}")
    
    policy_cls = POLICIES[policy_name]
    policy_kwargs = cfg.get("policy_kwargs", {})
    
    # Special handling for transformer variants
    if policy_name == "transformer":
        # Configure spatio-temporal attention if requested
        if cfg.get("use_spatio_temporal", False):
            policy_kwargs.setdefault("transformer_kwargs", {})
            policy_kwargs["transformer_kwargs"]["use_spatio_temporal"] = True
        
        # Configure performer/linear attention if requested
        if cfg.get("use_performer", False):
            if not PERFORMER_AVAILABLE:
                print("Warning: Performer requested but not available. Using standard transformer.")
            else:
                policy_cls = POLICIES["performer"]
                
        # Configure memory-compressed decoder for sequence prediction
        if cfg.get("predict_sequence", False):
            # Extract state dimension from environment
            sample_obs = env.reset()[0]
            if isinstance(sample_obs, dict):
                # Flatten observation to estimate state dimension
                state_dim = sum(np.prod(space.shape) for space in env.observation_space.spaces.values())
            else:
                state_dim = np.prod(env.observation_space.shape)
                
            # Create sequence predictor
            seq_predictor = StateSequencePredictor(
                embed_dim=policy_kwargs.get("features_extractor_kwargs", {}).get("embed_dim", 256),
                state_dim=int(state_dim),
                horizon=cfg.get("prediction_horizon", 200),
                hidden_dim=cfg.get("decoder_hidden_dim", 256),
                num_layers=cfg.get("decoder_layers", 2)
            )
            
            # Add to policy kwargs
            policy_kwargs["state_predictor"] = seq_predictor
    
    # PPO configuration
    ppo_kwargs = cfg.get("ppo_kwargs", {})
    ppo_kwargs.setdefault("verbose", 1)
    ppo_kwargs.setdefault("device", device)
    
    # Create model
    model = PPO(
        policy=policy_cls,
        env=env,
        tensorboard_log=str(run_dir / "tb"),
        **ppo_kwargs,
        policy_kwargs=policy_kwargs,
    )
    
    # Setup logger
    new_logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Knowledge distillation setup
    if args.teacher:
        print(f"Loading teacher model from {args.teacher} for knowledge distillation")
        teacher = PPO.load(args.teacher, env=env, device=device)
        
        # Custom loss function that incorporates teacher outputs
        def custom_distillation_loss(model_outputs):
            """Custom loss for knowledge distillation."""
            # Extract model outputs
            pi_logits, values, old_logprobs, advantages, returns, entropy_coef, vf_coef = model_outputs
            
            # Get teacher predictions for the same observations
            with torch.no_grad():
                t_actions, t_values, t_log_probs = teacher.policy.evaluate_actions(model.policy._last_obs)
                
            # Compute distillation loss
            kd_loss = student_loss(
                (pi_logits, values), 
                (t_log_probs, t_values),
                temperature=cfg.get("distillation_temperature", 2.0),
                alpha=cfg.get("distillation_alpha", 0.5)
            )
            
            # Combine with standard PPO loss (weighted)
            kd_weight = cfg.get("distillation_weight", 0.5)
            ppo_loss = model.policy.loss(pi_logits, values, old_logprobs, advantages, returns, entropy_coef, vf_coef)
            
            return (1 - kd_weight) * ppo_loss + kd_weight * kd_loss
        
        # Set custom loss function
        model.policy.custom_loss = custom_distillation_loss  # type: ignore
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.get("save_freq", 10000),
        save_path=str(run_dir / "checkpoints"),
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=cfg.get("eval_freq", 10000),
        n_eval_episodes=cfg.get("n_eval_episodes", 5),
        deterministic=True,
        render=args.capture_video,
    )
    callbacks.append(eval_callback)
    
    # Sequence prediction callback (if enabled)
    if cfg.get("predict_sequence", False):
        seq_callback = SequencePredictionCallback(
            eval_env=eval_env,
            eval_freq=cfg.get("eval_freq", 10000),
            n_eval_episodes=cfg.get("n_eval_episodes", 5),
            horizon=cfg.get("prediction_horizon", 200),
        )
        callbacks.append(seq_callback)
    
    # Model complexity callback
    complexity_callback = ModelComplexityCallback()
    callbacks.append(complexity_callback)
    
    # W&B callback
    if args.wandb and WANDB_AVAILABLE:
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)
    
    # Optuna hyperparameter sweep
    if args.sweep:
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError:
            raise ImportError("Optuna required for hyperparameter sweeps. Install with: pip install optuna")
        
        # Define the objective function for optimization
        def objective(trial: optuna.Trial) -> float:
            """Optimization objective for Optuna."""
            # Reset model with trial-specific hyperparameters
            
            # Sample hyperparameters based on policy type
            if policy_name == "transformer" or policy_name == "performer":
                # Transformer-specific hyperparameters
                transformer_kwargs = {
                    "embed_dim": trial.suggest_categorical("embed_dim", [128, 256, 512]),
                    "depth": trial.suggest_int("depth", 2, 8),
                    "num_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                }
                
                if policy_name == "performer":
                    # Performer-specific hyperparameters
                    transformer_kwargs["feature_redraw_interval"] = trial.suggest_categorical(
                        "feature_redraw_interval", [100, 1000, 5000]
                    )
                    transformer_kwargs["generalized_attention"] = trial.suggest_categorical(
                        "generalized_attention", [True, False]
                    )
                
                policy_kwargs["transformer_kwargs"] = transformer_kwargs
            
            elif policy_name == "lstm":
                # LSTM-specific hyperparameters
                policy_kwargs["lstm_hidden"] = trial.suggest_categorical("lstm_hidden", [128, 256, 512])
                policy_kwargs["lstm_layers"] = trial.suggest_int("lstm_layers", 1, 4)
            
            # Common PPO hyperparameters
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
            ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
            
            # Update model hyperparameters
            model.learning_rate = lr
            model.clip_range = clip_range
            model.ent_coef = ent_coef
            
            # Train for specified number of timesteps
            timesteps = cfg.get("sweep_timesteps", 500000)
            model.learn(total_timesteps=timesteps, callback=callbacks)
            
            # Evaluate final performance
            mean_reward = 0
            n_eval = 10
            for _ in range(n_eval):
                obs = eval_env.reset()[0]
                done = False
                episode_reward = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                mean_reward += episode_reward
            
            return float(mean_reward / n_eval)
        
        # Create and run study
        study_name = f"{run_name}_sweep"
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=TPESampler(seed=seed),
        )
        
        try:
            study.optimize(objective, n_trials=args.sweep)
        except KeyboardInterrupt:
            print("Sweep interrupted. Using best parameters found so far.")
        
        # Save best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"Best parameters: {best_params}")
        print(f"Best reward: {best_value}")
        
        # Save best parameters to file
        with open(run_dir / "best_params.json", "w", encoding="utf-8") as f:
            json.dump({
                "params": best_params,
                "value": best_value,
            }, f, indent=2)
        
        # Train final model with best parameters
        for param, value in best_params.items():
            if param.startswith("transformer_"):
                policy_kwargs.setdefault("transformer_kwargs", {})
                policy_kwargs["transformer_kwargs"][param.replace("transformer_", "")] = value
            elif param in ["learning_rate", "clip_range", "ent_coef"]:
                setattr(model, param, value)
            else:
                policy_kwargs[param] = value
        
        # Final training with best parameters
        model = PPO(
            policy=policy_cls,
            env=env,
            tensorboard_log=str(run_dir / "tb"),
            **ppo_kwargs,
            policy_kwargs=policy_kwargs,
        )
    
    # Train model (if not already trained in sweep)
    if not args.sweep:
        timesteps = cfg.get("timesteps", 1_000_000)
        model.learn(total_timesteps=timesteps, callback=callbacks)
    
    # Save final model
    model.save(run_dir / cfg.get("save_name", "final_model"))
    
    # Close environments
    env.close()
    eval_env.close()
    
    # Close W&B
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    print(f"Training complete. Model saved to {run_dir}")


if __name__ == "__main__":
    main()
