#!/usr/bin/env python3
"""
RecurrentPPO LSTM training script with comprehensive logging and evaluation.

Usage:
    python train_lstm_recurrent.py --total-timesteps 5000000 --n-envs 8
"""

import argparse
import csv
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

# Try different import paths for linear_schedule (version compatibility)
try:
    from stable_baselines3.common.utils import linear_schedule
    USE_SCHEDULE = True
except ImportError:
    try:
        from stable_baselines3.common.schedule import linear_schedule
        USE_SCHEDULE = True
    except ImportError:
        # Fallback: use constant learning rate
        print("Warning: linear_schedule not available, using constant learning rate")
        USE_SCHEDULE = False

from sb3_contrib import RecurrentPPO

# Try different import paths for MlpLstmPolicy (version compatibility)
try:
    from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
except ImportError:
    try:
        from sb3_contrib.common.recurrent.policies import MlpLstmPolicy
    except ImportError:
        # Use string policy name as fallback
        print("Warning: Could not import MlpLstmPolicy class, using string policy name")
        MlpLstmPolicy = "MlpLstmPolicy"

# Local imports
import sys
sys.path.append('.')
try:
    import flycraft
    gym.register_envs(flycraft)
except ImportError:
    print("Warning: FlyCraft not available")

from src.drone_rl.utils.metrics import time_to_collision, path_deviation, velocity_error


# ============================================================================
# HYPERPARAMETERS (tunable constants)
# ============================================================================

DEFAULT_HYPERPARAMS = {
    # Environment
    'n_envs': 8,
    'env_id': 'FlyCraft-v0',
    
    # PPO params
    'n_steps': 2048,  # Must be divisible by 16 for LSTM sequence handling
    'batch_size': 256,
    'n_epochs': 8,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'vf_coef': 0.5,
    'ent_coef': 0.0,
    'learning_rate': 3e-4,
    
    # LSTM policy params
    'lstm_hidden_size': 256,
    'n_lstm_layers': 1,
    'ortho_init': True,
    
    # Training
    'total_timesteps': 5_000_000,
    'eval_every': 100_000,
    'n_eval_episodes': 10,
    
    # Paths
    'save_dir': 'runs/baseline_lstm_quick',
}


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_episode_metrics(obs_history: List[np.ndarray], 
                          info_history: List[Dict], 
                          action_history: List[np.ndarray]) -> Dict[str, float]:
    """Compute comprehensive metrics for a single episode."""
    if not info_history:
        return {}
    
    metrics = {}
    
    # Episode-level flags
    final_info = info_history[-1]
    metrics['success'] = float(final_info.get('success', False))
    metrics['collision'] = float(final_info.get('collision', False))
    metrics['oob'] = float(final_info.get('out_of_bounds', False))
    
    # Step-wise metrics
    path_devs = []
    vel_errs = []
    ttcs = []
    
    # Accuracy-style counters
    steps_path_within_0p5m = 0
    steps_vel_within_10pct = 0
    steps_ttc_over_3s = 0
    total_steps = len(info_history)
    
    for i, info in enumerate(info_history):
        # Path deviation
        if 'path_deviation' in info:
            path_dev = info['path_deviation']
        else:
            # Compute from position if available
            pos = info.get('position', np.zeros(3))
            target_pos = info.get('target_position', np.zeros(3))
            path_dev = np.linalg.norm(pos - target_pos) if pos.any() else 0.0
        
        path_devs.append(path_dev)
        if path_dev < 0.5:
            steps_path_within_0p5m += 1
            
        # Velocity error
        if 'velocity_error' in info:
            vel_err = info['velocity_error']
        else:
            # Compute from velocity if available
            vel = info.get('velocity', np.zeros(3))
            target_vel = info.get('target_velocity', np.zeros(3))
            vel_err = np.linalg.norm(vel - target_vel) if vel.any() else 0.0
            
        vel_errs.append(vel_err)
        
        # Convert to percentage if target speed available
        target_speed = np.linalg.norm(target_vel) if 'target_velocity' in info else 1.0
        vel_err_pct = (vel_err / max(target_speed, 1e-6)) * 100
        if vel_err_pct < 10.0:
            steps_vel_within_10pct += 1
            
        # Time to collision
        if 'time_to_collision' in info:
            ttc = info['time_to_collision']
        else:
            # Compute TTC if position/velocity available
            pos = info.get('position', np.zeros(3))
            vel = info.get('velocity', np.zeros(3))
            if pos.any() and vel.any():
                # Simplified TTC computation (would need obstacle info for real TTC)
                ttc = 10.0  # Default safe value
            else:
                ttc = 10.0
                
        ttcs.append(ttc)
        if ttc > 3.0:
            steps_ttc_over_3s += 1
    
    # Aggregate metrics
    if path_devs:
        metrics['mean_path_dev'] = np.mean(path_devs)
        metrics['p50_path_dev'] = np.percentile(path_devs, 50)
        metrics['p95_path_dev'] = np.percentile(path_devs, 95)
    
    if vel_errs:
        target_speeds = [max(np.linalg.norm(info.get('target_velocity', [1.0])), 1e-6) 
                        for info in info_history]
        vel_err_pcts = [(vel_errs[i] / target_speeds[i]) * 100 
                       for i in range(len(vel_errs))]
        metrics['mean_vel_err_pct'] = np.mean(vel_err_pcts)
        metrics['p95_vel_err_pct'] = np.percentile(vel_err_pcts, 95)
    
    if ttcs:
        metrics['min_ttc'] = np.min(ttcs)
        metrics['mean_ttc'] = np.mean(ttcs)
    
    # Accuracy-style rates
    metrics['steps_path_within_0p5m'] = steps_path_within_0p5m / max(total_steps, 1)
    metrics['steps_vel_within_10pct'] = steps_vel_within_10pct / max(total_steps, 1)
    metrics['steps_ttc_over_3s'] = steps_ttc_over_3s / max(total_steps, 1)
    
    return metrics


def aggregate_episode_metrics(episode_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across multiple episodes."""
    if not episode_metrics_list:
        return {}
    
    aggregated = {}
    
    # Episode-level rates
    aggregated['success_rate'] = np.mean([m.get('success', 0) for m in episode_metrics_list])
    aggregated['collision_rate'] = np.mean([m.get('collision', 0) for m in episode_metrics_list])
    aggregated['oob_rate'] = np.mean([m.get('oob', 0) for m in episode_metrics_list])
    
    # Aggregate continuous metrics
    for key in ['mean_path_dev', 'p50_path_dev', 'p95_path_dev',
                'mean_vel_err_pct', 'p95_vel_err_pct', 'min_ttc', 'mean_ttc']:
        values = [m.get(key, 0) for m in episode_metrics_list if key in m]
        if values:
            aggregated[key] = np.mean(values)
    
    # Aggregate accuracy-style rates
    for key in ['steps_path_within_0p5m', 'steps_vel_within_10pct', 'steps_ttc_over_3s']:
        values = [m.get(key, 0) for m in episode_metrics_list if key in m]
        if values:
            aggregated[key] = np.mean(values)
    
    return aggregated


# ============================================================================
# CUSTOM CALLBACK
# ============================================================================

class MetricsCallback(BaseCallback):
    """Custom callback for comprehensive metrics logging."""
    
    def __init__(self, eval_env: VecNormalize, csv_path: Path, 
                 eval_freq: int = 10000, n_eval_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.csv_path = csv_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Initialize CSV file
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = [
                    'step', 'wall_time', 'mean_return', 'success_rate', 'collision_rate', 'oob_rate',
                    'mean_path_dev', 'p50_path_dev', 'p95_path_dev',
                    'mean_vel_err_pct', 'p95_vel_err_pct',
                    'min_ttc', 'mean_ttc',
                    'steps_path_within_0p5m', 'steps_vel_within_10pct', 'steps_ttc_over_3s'
                ]
                writer.writerow(header)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
        return True
    
    def _run_evaluation(self):
        """Run evaluation episodes and log metrics."""
        if self.verbose > 0:
            print(f"\nRunning evaluation at step {self.num_timesteps}...")
        
        episode_metrics_list = []
        episode_returns = []
        
        for episode in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            episode_return = 0
            obs_history = [obs]
            info_history = []
            action_history = []
            done = False
            
            # Reset LSTM states for evaluation
            lstm_states = None
            
            while not done:
                action, lstm_states = self.model.predict(
                    obs, state=lstm_states, deterministic=True
                )
                obs, reward, done, info = self.eval_env.step(action)
                
                episode_return += reward
                obs_history.append(obs)
                info_history.append(info[0] if isinstance(info, list) else info)
                action_history.append(action)
            
            episode_returns.append(episode_return)
            
            # Compute episode metrics
            ep_metrics = compute_episode_metrics(obs_history, info_history, action_history)
            episode_metrics_list.append(ep_metrics)
        
        # Aggregate metrics
        aggregated_metrics = aggregate_episode_metrics(episode_metrics_list)
        mean_return = np.mean(episode_returns)
        
        # Log to TensorBoard
        self.logger.record("eval/mean_return", mean_return)
        for key, value in aggregated_metrics.items():
            self.logger.record(f"eval/{key}", value)
        
        # Log to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                self.num_timesteps,
                time.time(),
                mean_return,
                aggregated_metrics.get('success_rate', 0),
                aggregated_metrics.get('collision_rate', 0),
                aggregated_metrics.get('oob_rate', 0),
                aggregated_metrics.get('mean_path_dev', 0),
                aggregated_metrics.get('p50_path_dev', 0),
                aggregated_metrics.get('p95_path_dev', 0),
                aggregated_metrics.get('mean_vel_err_pct', 0),
                aggregated_metrics.get('p95_vel_err_pct', 0),
                aggregated_metrics.get('min_ttc', 0),
                aggregated_metrics.get('mean_ttc', 0),
                aggregated_metrics.get('steps_path_within_0p5m', 0),
                aggregated_metrics.get('steps_vel_within_10pct', 0),
                aggregated_metrics.get('steps_ttc_over_3s', 0),
            ]
            writer.writerow(row)
        
        if self.verbose > 0:
            print(f"Evaluation complete. Mean return: {mean_return:.2f}, "
                  f"Success rate: {aggregated_metrics.get('success_rate', 0):.2f}")


# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def make_env(env_id: str, rank: int = 0, seed: int = 0):
    """Create a single environment."""
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vec_env(env_id: str, n_envs: int, seed: int = 0) -> VecNormalize:
    """Create vectorized environment with normalization."""
    if n_envs == 1:
        env = DummyVecEnv([make_env(env_id, 0, seed)])
    else:
        env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(n_envs)])
    
    # Wrap with VecNormalize
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return env


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_recurrent_ppo(args):
    """Main training function."""
    print("🚁 Starting RecurrentPPO LSTM training...")
    print(f"Configuration: {args.__dict__}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = save_dir / "tb"
    csv_path = save_dir / "metrics.csv"
    
    # Create environments
    print(f"Creating {args.n_envs} training environments...")
    train_env = create_vec_env(args.env_id, args.n_envs, seed=42)
    
    print("Creating evaluation environment...")
    eval_env = create_vec_env(args.env_id, 1, seed=1337)
    eval_env.training = False
    eval_env.norm_reward = False
    
    # Create model
    print("Initializing RecurrentPPO model...")
    policy_kwargs = {
        "lstm_hidden_size": args.lstm_hidden,
        "n_lstm_layers": args.lstm_layers,
        "ortho_init": args.ortho_init,
    }
    
    # Handle learning rate (with or without scheduling)
    if USE_SCHEDULE:
        learning_rate = linear_schedule(args.learning_rate)
    else:
        learning_rate = args.learning_rate
    
    model = RecurrentPPO(
        MlpLstmPolicy,
        train_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_log_dir),
        verbose=1,
    )
    
    # Create callbacks
    print("Setting up callbacks...")
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir),
        eval_freq=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    
    # Metrics callback
    metrics_callback = MetricsCallback(
        eval_env, csv_path, 
        eval_freq=args.eval_every, 
        n_eval_episodes=args.n_eval_episodes,
        verbose=1
    )
    
    callbacks = [eval_callback, metrics_callback]
    
    # Train the model
    print(f"🚀 Starting training for {args.total_timesteps:,} timesteps...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    training_time = time.time() - start_time
    
    # Save final model and environment
    print("💾 Saving final model and environment...")
    model.save(save_dir / "final_model")
    train_env.save(save_dir / "vecnormalize.pkl")
    
    # Print summary
    print("\n" + "="*60)
    print("🎉 Training completed!")
    print(f"⏱️  Total training time: {training_time:.1f}s ({training_time/60:.1f}m)")
    print(f"📁 Models saved to: {save_dir}")
    print(f"📊 TensorBoard logs: tensorboard --logdir {tb_log_dir}")
    print(f"📈 CSV metrics: {csv_path}")
    
    # Read and print final metrics
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                if len(last_line) >= 4:
                    final_return = float(last_line[2])
                    final_success = float(last_line[3])
                    print(f"🏆 Final performance: Return={final_return:.2f}, Success Rate={final_success:.2%}")
    except Exception as e:
        print(f"Could not read final metrics: {e}")
    
    print("="*60)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RecurrentPPO LSTM policy")
    
    # Configuration file option
    parser.add_argument("--config", type=str, default=None,
                       help="YAML config file path (overrides individual args)")
    
    # Training hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_HYPERPARAMS['total_timesteps'],
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_HYPERPARAMS['n_envs'],
                       help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_HYPERPARAMS['n_steps'],
                       help="Steps per rollout (must be divisible by 16)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_HYPERPARAMS['batch_size'],
                       help="Batch size for training")
    parser.add_argument("--n-epochs", type=int, default=DEFAULT_HYPERPARAMS['n_epochs'],
                       help="Number of epochs per update")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_HYPERPARAMS['learning_rate'],
                       help="Learning rate")
    
    # LSTM hyperparameters
    parser.add_argument("--lstm-hidden", type=int, default=DEFAULT_HYPERPARAMS['lstm_hidden_size'],
                       help="LSTM hidden size")
    parser.add_argument("--lstm-layers", type=int, default=DEFAULT_HYPERPARAMS['n_lstm_layers'],
                       help="Number of LSTM layers")
    parser.add_argument("--ortho-init", action="store_true", default=DEFAULT_HYPERPARAMS['ortho_init'],
                       help="Use orthogonal initialization")
    
    # Paths and evaluation
    parser.add_argument("--save-dir", type=str, default=DEFAULT_HYPERPARAMS['save_dir'],
                       help="Directory to save models and logs")
    parser.add_argument("--eval-every", type=int, default=DEFAULT_HYPERPARAMS['eval_every'],
                       help="Evaluation frequency (timesteps)")
    parser.add_argument("--n-eval-episodes", type=int, default=DEFAULT_HYPERPARAMS['n_eval_episodes'],
                       help="Number of evaluation episodes")
    
    # Environment
    parser.add_argument("--env-id", type=str, default=DEFAULT_HYPERPARAMS['env_id'],
                       help="Environment ID")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override args with config values
        if 'timesteps' in config:
            args.total_timesteps = config['timesteps']
        if 'n_envs' in config:
            args.n_envs = config['n_envs']
        if 'env_id' in config:
            args.env_id = config['env_id']
        if 'save_freq' in config:
            args.eval_every = config['save_freq']
        if 'n_eval_episodes' in config:
            args.n_eval_episodes = config['n_eval_episodes']
        
        # Override with recurrent_kwargs if present
        if 'recurrent_kwargs' in config:
            rk = config['recurrent_kwargs']
            if 'lstm_hidden_size' in rk:
                args.lstm_hidden = rk['lstm_hidden_size']
            if 'n_lstm_layers' in rk:
                args.lstm_layers = rk['n_lstm_layers']
        
        # Override with ppo_kwargs if present
        if 'ppo_kwargs' in config:
            pk = config['ppo_kwargs']
            if 'learning_rate' in pk:
                args.learning_rate = pk['learning_rate']
            if 'n_steps' in pk:
                args.n_steps = pk['n_steps']
            if 'batch_size' in pk:
                args.batch_size = pk['batch_size']
            if 'n_epochs' in pk:
                args.n_epochs = pk['n_epochs']
        
        print(f"Loaded config from {args.config}")
    
    # Validate n_steps is divisible by 16 for LSTM sequence handling
    if args.n_steps % 16 != 0:
        print(f"Warning: n_steps ({args.n_steps}) should be divisible by 16 for optimal LSTM performance")
        args.n_steps = ((args.n_steps // 16) + 1) * 16
        print(f"Adjusted n_steps to {args.n_steps}")
    
    train_recurrent_ppo(args)


if __name__ == "__main__":
    main()
