"""Enhanced debug wrapper for FlyCraft environment to diagnose training issues."""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple


class FlyCraftDebugWrapper(gym.Wrapper):
    """Debug wrapper specifically for FlyCraft to track success/failure patterns."""
    
    def __init__(self, env: gym.Env, verbose: bool = False):
        super().__init__(env)
        self.verbose = verbose
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.failure_reasons = {"timeout": 0, "crash": 0, "lost": 0}
        self.position_history = []
        self.distance_to_goal_history = []
        
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset and log previous episode stats."""
        if self.episode_count > 0:
            self._log_episode_summary()
        
        self.episode_count += 1
        self.position_history = []
        self.distance_to_goal_history = []
        
        obs, info = self.env.reset(**kwargs)
        
        if self.verbose and self.episode_count <= 3:
            print(f"\n=== Episode {self.episode_count} Start ===")
            print(f"Initial obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not dict'}")
            print(f"Initial info: {info}")
            
        return obs, info
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step and track detailed metrics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track position and goal distance if available
        if isinstance(info, dict):
            if "drone_position" in info:
                pos = np.array(info["drone_position"])
                self.position_history.append(pos)
                
                if "desired_goal" in info:
                    goal = np.array(info["desired_goal"])
                    distance = np.linalg.norm(pos - goal)
                    self.distance_to_goal_history.append(distance)
        
        # Log detailed step info for first few episodes
        if self.verbose and self.episode_count <= 2 and len(self.position_history) <= 10:
            step_num = len(self.position_history)
            print(f"  Step {step_num}: reward={reward:.3f}, dist_to_goal={self.distance_to_goal_history[-1]:.1f}m" 
                  if self.distance_to_goal_history else f"  Step {step_num}: reward={reward:.3f}")
        
        return obs, reward, terminated, truncated, info
    
    def _log_episode_summary(self):
        """Log detailed episode summary."""
        episode_length = len(self.position_history)
        
        # Determine success/failure reason
        is_success = self.success_count > 0  # Will be updated by caller
        
        if self.distance_to_goal_history:
            initial_dist = self.distance_to_goal_history[0]
            final_dist = self.distance_to_goal_history[-1]
            min_dist = min(self.distance_to_goal_history)
            
            # Classify failure reason
            if not is_success:
                if final_dist > initial_dist * 0.8:  # Barely moved towards goal
                    reason = "lost"
                elif episode_length >= 990:  # Near max steps
                    reason = "timeout"
                else:
                    reason = "crash"
                self.failure_reasons[reason] += 1
        
        # Log every 10 episodes or if success
        if self.episode_count % 10 == 0 or is_success:
            success_rate = self.success_count / self.episode_count
            avg_length = np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
            
            print(f"\n📊 Episode {self.episode_count} Summary:")
            print(f"  Success rate: {success_rate:.1%} ({self.success_count}/{self.episode_count})")
            print(f"  Avg length: {avg_length:.1f} steps")
            print(f"  Failure reasons: {dict(self.failure_reasons)}")
            
            if self.distance_to_goal_history:
                print(f"  Distance: {initial_dist:.1f}m → {final_dist:.1f}m (min: {min_dist:.1f}m)")
                
        self.episode_lengths.append(episode_length)


class RewardAnalysisWrapper(gym.Wrapper):
    """Wrapper to analyze reward patterns and detect sparse rewards."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_history = []
        self.positive_rewards = 0
        self.negative_rewards = 0
        self.zero_rewards = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.reward_history.append(reward)
        
        if reward > 0:
            self.positive_rewards += 1
        elif reward < 0:
            self.negative_rewards += 1
        else:
            self.zero_rewards += 1
            
        # Log reward analysis every 100 steps
        if len(self.reward_history) % 100 == 0:
            total_steps = len(self.reward_history)
            pos_pct = self.positive_rewards / total_steps * 100
            neg_pct = self.negative_rewards / total_steps * 100
            zero_pct = self.zero_rewards / total_steps * 100
            
            print(f"Reward Analysis (last 100 steps): "
                  f"Positive: {pos_pct:.1f}%, Negative: {neg_pct:.1f}%, Zero: {zero_pct:.1f}%")
                  
        return obs, reward, terminated, truncated, info
