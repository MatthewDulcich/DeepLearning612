"""Environment wrapper for debugging and reward analysis."""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, Optional


class DebugWrapper(gym.Wrapper):
    """Wrapper to debug environment behavior and reward structure."""
    
    def __init__(self, env: gym.Env, log_frequency: int = 100):
        super().__init__(env)
        self.log_frequency = log_frequency
        self.episode_count = 0
        self.step_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and log episode statistics."""
        if self.episode_count > 0:
            # Log previous episode
            total_reward = sum(self.episode_rewards)
            episode_length = len(self.episode_rewards)
            
            if self.episode_count % self.log_frequency == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_frequency:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-self.log_frequency:]) if self.episode_lengths else 0
                success_rate = self.success_count / self.episode_count
                
                print(f"Episode {self.episode_count}: "
                      f"Reward={total_reward:.2f}, "
                      f"Length={episode_length}, "
                      f"Avg_Reward={avg_reward:.2f}, "
                      f"Avg_Length={avg_length:.1f}, "
                      f"Success_Rate={success_rate:.3f}")
        
        self.episode_count += 1
        self.step_count = 0
        self.episode_rewards = []
        
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step environment and log reward information."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_rewards.append(reward)
        self.step_count += 1
        
        # Log detailed step info for first few episodes
        if self.episode_count <= 3 and self.step_count <= 20:
            print(f"  Step {self.step_count}: action={action}, reward={reward:.3f}, "
                  f"terminated={terminated}, truncated={truncated}")
            if isinstance(info, dict):
                print(f"    Info: {info}")
        
        # Check for success
        if terminated or truncated:
            episode_length = len(self.episode_rewards)
            total_reward = sum(self.episode_rewards)
            self.episode_lengths.append(episode_length)
            
            # Check if this was a success
            is_success = False
            if isinstance(info, dict):
                is_success = info.get('is_success', False)
            elif isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
                is_success = info[0].get('is_success', False)
            
            if is_success:
                self.success_count += 1
                print(f"SUCCESS! Episode {self.episode_count}: "
                      f"Reward={total_reward:.2f}, Length={episode_length}")
        
        return obs, reward, terminated, truncated, info


def create_debug_env(env_id: str, **kwargs) -> gym.Env:
    """Create environment with debug wrapper."""
    env = gym.make(env_id, **kwargs)
    env = DebugWrapper(env, log_frequency=10)
    return env
