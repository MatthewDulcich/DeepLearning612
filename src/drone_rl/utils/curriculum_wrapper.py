"""
FlyCraft Curriculum Learning Wrapper

This wrapper implements curriculum learning by modifying the environment behavior
without requiring FlyCraft to support curriculum parameters directly.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import random


class FlyCraftCurriculumWrapper(gym.Wrapper):
    """
    Curriculum learning wrapper for FlyCraft environment.
    
    Implements curriculum by:
    1. Modifying reward structure (dense vs sparse vs angle-only)
    2. Adjusting goal distances and configurations
    3. Controlling episode termination conditions
    4. Adding curriculum-specific observations
    """
    
    def __init__(
        self, 
        env: gym.Env,
        frequency: int = 10,
        control_mode: str = "guidance_law_mode", 
        reward_mode: str = "dense",
        goal_cfg: Optional[Dict] = None,
        verbose: bool = False
    ):
        super().__init__(env)
        
        self.frequency = frequency
        self.control_mode = control_mode
        self.reward_mode = reward_mode
        self.goal_cfg = goal_cfg or {"type": "fixed_short", "distance_m": 200}
        self.verbose = verbose
        
        # Curriculum state tracking
        self.episode_count = 0
        self.step_count = 0
        self.original_reward = 0.0
        self.cumulative_distance_reduction = 0.0
        self.initial_distance_to_goal = None
        self.current_distance_to_goal = None
        self.goal_position = None
        self.drone_position = None
        
        # Frequency-based step duration (higher frequency = shorter real-time steps)
        self.step_duration = 1.0 / self.frequency
        
        if self.verbose:
            print(f"🎯 Curriculum Wrapper initialized:")
            print(f"   Frequency: {frequency}Hz (step_duration: {self.step_duration:.3f}s)")
            print(f"   Control mode: {control_mode}")
            print(f"   Reward mode: {reward_mode}")
            print(f"   Goal config: {goal_cfg}")
    
    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and apply curriculum settings."""
        obs, info = self.env.reset(**kwargs)
        
        self.episode_count += 1
        self.step_count = 0
        self.original_reward = 0.0
        self.cumulative_distance_reduction = 0.0
        self.initial_distance_to_goal = None
        self.current_distance_to_goal = None
        
        # Apply goal configuration based on curriculum
        self._apply_goal_configuration(obs, info)
        
        # Extract initial positions
        self._extract_positions(obs, info)
        
        if self.verbose and self.episode_count <= 3:
            print(f"\n🎮 Episode {self.episode_count} Reset (Curriculum: {self.frequency}Hz)")
            if self.drone_position is not None and self.goal_position is not None:
                dist = np.linalg.norm(self.goal_position - self.drone_position)
                print(f"   Initial distance to goal: {dist:.1f}m")
        
        # Modify observation space if needed
        obs = self._modify_observation(obs)
        info = self._modify_info(info)
        
        return obs, info
    
    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Step environment and apply curriculum modifications."""
        # Apply control mode modifications to action
        modified_action = self._modify_action(action)
        
        # Step the base environment
        obs, reward, terminated, truncated, info = self.env.step(modified_action)
        
        self.step_count += 1
        self.original_reward = reward
        
        # Extract current positions
        self._extract_positions(obs, info)
        
        # Apply curriculum reward modifications
        curriculum_reward = self._modify_reward(reward, obs, info, terminated, truncated)
        
        # Apply curriculum termination conditions
        curriculum_terminated, curriculum_truncated = self._modify_termination(
            terminated, truncated, obs, info
        )
        
        # Modify observations and info
        obs = self._modify_observation(obs)
        info = self._modify_info(info)
        
        # Log detailed step info for first few episodes
        if self.verbose and self.episode_count <= 2 and self.step_count <= 10:
            print(f"  Step {self.step_count}: orig_reward={reward:.3f}, "
                  f"curriculum_reward={curriculum_reward:.3f}, "
                  f"dist_to_goal={self.current_distance_to_goal:.1f}m" 
                  if self.current_distance_to_goal else "")
        
        return obs, curriculum_reward, curriculum_terminated, curriculum_truncated, info
    
    def _apply_goal_configuration(self, obs: Dict, info: Dict) -> None:
        """Apply goal configuration based on curriculum settings."""
        goal_type = self.goal_cfg.get("type", "fixed_short")
        
        if goal_type == "fixed_short":
            # Fixed goal at specified distance
            distance_m = self.goal_cfg.get("distance_m", 200)
            heading_jitter = self.goal_cfg.get("heading_jitter_deg", 0)
            
            # Modify goal in info if available
            if "desired_goal" in info or "goal_position" in info:
                # Apply distance and jitter modifications
                if heading_jitter > 0:
                    jitter_rad = np.radians(np.random.uniform(-heading_jitter, heading_jitter))
                    # Apply jitter to goal position (simplified)
                    pass
        
        elif goal_type == "bucket_short":
            # Random goal from distance bins
            distance_bins = self.goal_cfg.get("distance_bins", [150, 250, 350])
            selected_distance = random.choice(distance_bins)
            
        elif goal_type == "bucket_med":
            # Medium range with yaw variations
            distance_bins = self.goal_cfg.get("distance_bins", [250, 400, 600])
            yaw_bins = self.goal_cfg.get("yaw_bins", [-30, 30])
            selected_distance = random.choice(distance_bins)
            selected_yaw = random.choice(yaw_bins) if yaw_bins else 0
            
        elif goal_type == "bucket_wide":
            # Wide range with large yaw variations
            distance_bins = self.goal_cfg.get("distance_bins", [300, 700, 1000])
            yaw_bins = self.goal_cfg.get("yaw_bins", [-60, 60])
            selected_distance = random.choice(distance_bins)
            selected_yaw = random.choice(yaw_bins) if yaw_bins else 0
    
    def _extract_positions(self, obs: Dict, info: Dict) -> None:
        """Extract drone and goal positions from observation/info."""
        # Try multiple possible keys for positions
        drone_pos_keys = ["drone_position", "position", "pos", "aircraft_position"]
        goal_pos_keys = ["desired_goal", "goal_position", "target_position", "goal"]
        
        # Extract drone position
        for key in drone_pos_keys:
            if key in info and info[key] is not None:
                self.drone_position = np.array(info[key])
                break
            elif key in obs and obs[key] is not None:
                if isinstance(obs[key], np.ndarray) and obs[key].shape[-1] >= 3:
                    self.drone_position = np.array(obs[key][:3])
                    break
        
        # Extract goal position  
        for key in goal_pos_keys:
            if key in info and info[key] is not None:
                self.goal_position = np.array(info[key])
                break
            elif key in obs and obs[key] is not None:
                if isinstance(obs[key], np.ndarray) and obs[key].shape[-1] >= 3:
                    self.goal_position = np.array(obs[key][:3])
                    break
        
        # Calculate distance if both positions available
        if self.drone_position is not None and self.goal_position is not None:
            if len(self.drone_position) >= 3 and len(self.goal_position) >= 3:
                self.current_distance_to_goal = np.linalg.norm(
                    self.goal_position[:3] - self.drone_position[:3]
                )
                
                if self.initial_distance_to_goal is None:
                    self.initial_distance_to_goal = self.current_distance_to_goal
    
    def _modify_action(self, action: np.ndarray) -> np.ndarray:
        """Modify action based on control mode curriculum."""
        if self.control_mode == "guidance_law_mode":
            # Higher-level guidance commands (default)
            return action
        elif self.control_mode == "end_to_end":
            # Direct low-level control - could add noise or constraints
            return action
        else:
            return action
    
    def _modify_reward(
        self, 
        original_reward: float, 
        obs: Dict, 
        info: Dict, 
        terminated: bool, 
        truncated: bool
    ) -> float:
        """Apply curriculum-specific reward modifications."""
        
        if self.reward_mode == "dense":
            # Dense reward with distance-based shaping
            reward = original_reward
            
            # Add distance-based reward if positions available
            if self.current_distance_to_goal is not None and self.initial_distance_to_goal is not None:
                # Progress reward: positive for getting closer
                if hasattr(self, '_previous_distance'):
                    distance_change = self._previous_distance - self.current_distance_to_goal
                    progress_reward = distance_change * 0.1  # Scale factor
                    reward += progress_reward
                
                # Distance penalty: small penalty for being far
                distance_penalty = -self.current_distance_to_goal * 0.001
                reward += distance_penalty
                
                self._previous_distance = self.current_distance_to_goal
            
            return reward
            
        elif self.reward_mode == "dense_angle_only":
            # Dense reward but only based on orientation/angle, not distance
            reward = original_reward
            
            # Remove distance-based components, keep only orientation rewards
            # This is a simplified version - in practice you'd extract angle components
            if self.current_distance_to_goal is not None:
                # Reduce distance influence, emphasize orientation
                reward *= 0.5  # Reduce overall reward magnitude
            
            return reward
            
        elif self.reward_mode == "sparse":
            # Sparse reward: only at episode end
            if terminated or truncated:
                # Check if goal was reached
                if self.current_distance_to_goal is not None:
                    if self.current_distance_to_goal < 10.0:  # Success threshold
                        return 100.0  # Large success reward
                    else:
                        return -10.0   # Failure penalty
                else:
                    return original_reward  # Fallback
            else:
                return 0.0  # No reward during episode
        
        else:
            return original_reward
    
    def _modify_termination(
        self, 
        terminated: bool, 
        truncated: bool, 
        obs: Dict, 
        info: Dict
    ) -> Tuple[bool, bool]:
        """Apply curriculum-specific termination conditions."""
        
        # Frequency-based episode length adjustment
        if self.frequency >= 50:
            # Higher frequency = shorter episodes for easier learning
            max_steps_adjusted = int(1000 * (20.0 / self.frequency))
            if self.step_count >= max_steps_adjusted:
                truncated = True
        
        # Early termination for very easy stages
        if self.frequency <= 10 and self.current_distance_to_goal is not None:
            if self.current_distance_to_goal < 5.0:  # Very close - success
                terminated = True
                if "is_success" not in info:
                    info["is_success"] = True
        
        return terminated, truncated
    
    def _modify_observation(self, obs: Dict) -> Dict:
        """Add curriculum-specific information to observations."""
        # Add curriculum metadata to observation
        if isinstance(obs, dict):
            obs["curriculum_frequency"] = np.array([self.frequency], dtype=np.float32)
            obs["curriculum_step_duration"] = np.array([self.step_duration], dtype=np.float32)
            
            if self.current_distance_to_goal is not None:
                obs["distance_to_goal"] = np.array([self.current_distance_to_goal], dtype=np.float32)
        
        return obs
    
    def _modify_info(self, info: Dict) -> Dict:
        """Add curriculum-specific information to info dict."""
        info["curriculum_frequency"] = self.frequency
        info["curriculum_reward_mode"] = self.reward_mode
        info["curriculum_control_mode"] = self.control_mode
        
        if self.current_distance_to_goal is not None:
            info["curriculum_distance_to_goal"] = self.current_distance_to_goal
            info["curriculum_progress"] = (
                (self.initial_distance_to_goal - self.current_distance_to_goal) / self.initial_distance_to_goal
                if self.initial_distance_to_goal and self.initial_distance_to_goal > 0 else 0.0
            )
        
        return info


def create_curriculum_env(
    env_id: str = "FlyCraft-v0",
    frequency: int = 10,
    control_mode: str = "guidance_law_mode",
    reward_mode: str = "dense", 
    goal_cfg: Optional[Dict] = None,
    max_episode_steps: int = 1000,
    verbose: bool = False
) -> gym.Env:
    """
    Factory function to create FlyCraft environment with curriculum wrapper.
    
    Parameters
    ----------
    env_id : str
        Environment ID
    frequency : int
        Control frequency in Hz (10, 20, 50, 100)
    control_mode : str
        Control interface mode
    reward_mode : str
        Reward structure mode
    goal_cfg : Dict
        Goal configuration
    max_episode_steps : int
        Maximum episode length
    verbose : bool
        Enable verbose logging
        
    Returns
    -------
    gym.Env
        Wrapped environment with curriculum learning
    """
    # Create base environment
    env = gym.make(env_id, max_episode_steps=max_episode_steps)
    
    # Wrap with curriculum
    env = FlyCraftCurriculumWrapper(
        env=env,
        frequency=frequency,
        control_mode=control_mode,
        reward_mode=reward_mode,
        goal_cfg=goal_cfg,
        verbose=verbose
    )
    
    return env
