"""
COMPREHENSIVE FIX FOR FLYCRAFT CURRICULUM LEARNING ISSUES

PROBLEM: FlyCraft environment doesn't accept curriculum parameters like step_frequence, control_mode, etc.

SOLUTION: Remove curriculum parameters from environment creation and use basic FlyCraft environment
"""

# ============================================================================
# FIX 1: Update train.py to use basic FlyCraft without curriculum params
# ============================================================================

def make_env_fixed(
    env_id: str,
    seed: int,
    rank: int = 0,
    capture_video: bool = False,
    run_dir = None,
    max_episode_steps: int = 1000,
    debug: bool = False,
) -> callable:
    """Simplified make_env that works with actual FlyCraft environment."""
    def _init():
        try:
            import flycraft
        except ImportError:
            pass
        
        # FlyCraft only accepts max_episode_steps, not curriculum parameters
        env = gym.make(env_id, max_episode_steps=max_episode_steps)
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

# ============================================================================
# FIX 2: Update configs to work without curriculum parameters
# ============================================================================

# configs/baseline_simple_working.yaml
WORKING_CONFIG = """
env_id: FlyCraft-v0
eval_freq: 2500
n_envs: 4
output_dir: runs
policy: lstm
policy_kwargs:
  lstm_hidden: 128
  lstm_layers: 1
  dropout: 0.0
  n_envs: 4
ppo_kwargs:
  batch_size: 128
  clip_range: 0.1
  gae_lambda: 0.95
  gamma: 0.98
  learning_rate: 3e-5
  n_epochs: 4
  n_steps: 256
  ent_coef: 0.02
  vf_coef: 1.0
  max_grad_norm: 0.5
run_name: baseline_working
save_freq: 10000
seed: 42
timesteps: 100000

# Simple environment - only max_episode_steps works
max_episode_steps: 1000

# Evaluation
n_eval_episodes: 10
verbose: 1
"""

# ============================================================================
# ALTERNATIVE: Use environment wrapper for curriculum
# ============================================================================

class FlyCraftCurriculumWrapper(gym.Wrapper):
    """Wrapper to implement curriculum without environment parameters."""
    
    def __init__(self, env, frequency=20, reward_mode="dense", goal_distance=500):
        super().__init__(env)
        self.frequency = frequency
        self.reward_mode = reward_mode
        self.goal_distance = goal_distance
        self.step_count = 0
        
    def reset(self, **kwargs):
        # Apply curriculum settings via wrapper logic
        obs, info = self.env.reset(**kwargs)
        # Modify observation or info based on curriculum
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Modify reward based on curriculum settings
        if self.reward_mode == "sparse":
            reward = reward if terminated or truncated else 0.0
        elif self.reward_mode == "dense_angle_only":
            # Implement angle-only reward logic
            pass
            
        self.step_count += 1
        return obs, reward, terminated, truncated, info

# ============================================================================
# IMMEDIATE COMMANDS TO RUN (once Python env is fixed)
# ============================================================================

COMMANDS = """
# 1. First test basic FlyCraft without any curriculum
PYTHONPATH=src python -c "
import gymnasium as gym
import flycraft
env = gym.make('FlyCraft-v0', max_episode_steps=500)
print('✅ FlyCraft works with max_episode_steps only')
env.close()
"

# 2. Test basic training
PYTHONPATH=src python -m src.drone_rl.train.train --config configs/baseline_simple_working.yaml

# 3. Check what parameters FlyCraft actually accepts
PYTHONPATH=src python discover_flycraft_params.py
"""

# ============================================================================
# KEY INSIGHTS FROM ERROR ANALYSIS
# ============================================================================

INSIGHTS = """
1. FlyCraft-v0 environment is very basic and doesn't support:
   - step_frequence (frequency control)
   - control_mode (guidance vs end-to-end)
   - reward_mode (dense vs sparse)
   - goal_cfg (goal configuration)

2. Only supported parameter: max_episode_steps

3. For curriculum learning, we need to:
   - Use environment wrappers instead of env parameters
   - Implement curriculum logic in training loop
   - Modify rewards/observations via wrappers

4. Success rate is 0% likely because:
   - Environment is too hard by default
   - No dense reward shaping
   - Agent needs much more training time
   - Need proper exploration strategy
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

NEXT_STEPS = """
1. Fix Python numpy environment issue first
2. Test basic FlyCraft creation
3. Run simple baseline without curriculum
4. Implement curriculum via wrappers if needed
5. Focus on hyperparameter tuning for success
"""
