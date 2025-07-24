"""Baseline controllers: PID, LSTM policy for ablation comparison.

This module provides baseline controllers for comparison with transformer models:
1. PIDController - Classic control approach for individual channels
2. DronePositionController - 3D position control using multiple PIDs
3. SimpleLSTMPolicy - Recurrent policy compatible with SB3 for sequence modeling

These baselines serve as benchmarks for ablation studies to quantify the
performance gains from transformer-based approaches.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PIDController:
    """PID controller for single-channel control.

    Implements a standard Proportional-Integral-Derivative controller
    with anti-windup protection for the integral term.
    """

    def __init__(
        self, 
        kp: float = 1.0, 
        ki: float = 0.0, 
        kd: float = 0.1,
        integral_limit: Optional[float] = None
    ):
        """Initialize PID controller.

        Parameters
        ----------
        kp : float
            Proportional gain
        ki : float
            Integral gain
        kd : float
            Derivative gain
        integral_limit : Optional[float]
            Maximum absolute value for integral term (anti-windup)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True

    def reset(self) -> None:
        """Reset controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_call = True

    def __call__(self, error: float, dt: float) -> float:
        """Compute control output based on error and time step.

        Parameters
        ----------
        error : float
            Current error (setpoint - measured_value)
        dt : float
            Time step in seconds

        Returns
        -------
        float
            Control output
        """
        # Handle first call (no derivative)
        if self.first_call:
            self.prev_error = error
            self.first_call = False
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / max(dt, 1e-6)

        # Update integral with anti-windup
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        # Store error for next iteration
        self.prev_error = error

        # Compute PID output
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class DronePositionController:
    """3D position controller for drones using separate PID controllers for each axis.

    This controller takes a target position and current state (position, velocity)
    and outputs control commands for the drone.
    """

    def __init__(
        self,
        position_gains: Dict[str, Tuple[float, float, float]] = None,
        velocity_gains: Dict[str, Tuple[float, float, float]] = None,
        integral_limits: Dict[str, float] = None,
        output_limits: Dict[str, Tuple[float, float]] = None
    ):
        """Initialize position controller with separate PID controllers.

        Parameters
        ----------
        position_gains : Dict[str, Tuple[float, float, float]]
            PID gains (kp, ki, kd) for position control per axis
        velocity_gains : Dict[str, Tuple[float, float, float]]
            PID gains for velocity control per axis
        integral_limits : Dict[str, float]
            Integral limits for anti-windup per axis
        output_limits : Dict[str, Tuple[float, float]]
            Min/max output limits per axis
        """
        # Default gains if not provided
        if position_gains is None:
            position_gains = {
                'x': (0.5, 0.0, 0.1),
                'y': (0.5, 0.0, 0.1),
                'z': (1.0, 0.1, 0.2),
                'yaw': (1.0, 0.0, 0.1)
            }
        
        if velocity_gains is None:
            velocity_gains = {
                'x': (0.8, 0.0, 0.05),
                'y': (0.8, 0.0, 0.05),
                'z': (1.2, 0.0, 0.1),
                'yaw': (0.5, 0.0, 0.0)
            }
            
        if integral_limits is None:
            integral_limits = {
                'x': 1.0, 'y': 1.0, 'z': 1.0, 'yaw': 0.5
            }
            
        if output_limits is None:
            output_limits = {
                'x': (-1.0, 1.0),
                'y': (-1.0, 1.0),
                'z': (-1.0, 1.0),
                'yaw': (-1.0, 1.0)
            }
        
        # Create position controllers (outer loop)
        self.position_controllers = {}
        for axis, (kp, ki, kd) in position_gains.items():
            self.position_controllers[axis] = PIDController(
                kp=kp, ki=ki, kd=kd, 
                integral_limit=integral_limits.get(axis, None)
            )
            
        # Create velocity controllers (inner loop)
        self.velocity_controllers = {}
        for axis, (kp, ki, kd) in velocity_gains.items():
            self.velocity_controllers[axis] = PIDController(
                kp=kp, ki=ki, kd=kd,
                integral_limit=integral_limits.get(axis, None)
            )
            
        self.output_limits = output_limits
        self.prev_time = None
        
    def reset(self) -> None:
        """Reset all controllers."""
        for controller in self.position_controllers.values():
            controller.reset()
        for controller in self.velocity_controllers.values():
            controller.reset()
        self.prev_time = None
        
    def __call__(
        self,
        target_position: np.ndarray,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        current_time: float,
        target_yaw: float = 0.0,
        current_yaw: float = 0.0
    ) -> np.ndarray:
        """Compute control outputs for position tracking.

        Uses a cascaded control approach:
        1. Position error → Velocity setpoint (outer loop)
        2. Velocity error → Control output (inner loop)

        Parameters
        ----------
        target_position : np.ndarray
            Target position [x, y, z]
        current_position : np.ndarray
            Current position [x, y, z]
        current_velocity : np.ndarray
            Current velocity [vx, vy, vz]
        current_time : float
            Current time in seconds
        target_yaw : float
            Target yaw angle in radians
        current_yaw : float
            Current yaw angle in radians

        Returns
        -------
        np.ndarray
            Control outputs [roll, pitch, thrust, yaw_rate]
        """
        # Compute time step
        if self.prev_time is None:
            dt = 0.01  # Default dt for first call
        else:
            dt = current_time - self.prev_time
        self.prev_time = current_time
        
        # Ensure dt is positive and reasonable
        dt = max(dt, 1e-6)
        dt = min(dt, 0.1)  # Cap at 100ms to prevent large steps
        
        # Position control (outer loop) - generates velocity setpoints
        velocity_setpoints = {}
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            error = target_position[i] - current_position[i]
            velocity_setpoints[axis] = self.position_controllers[axis](error, dt)
        
        # Handle yaw separately (angle wrapping)
        yaw_error = self._wrap_angle(target_yaw - current_yaw)
        velocity_setpoints['yaw'] = self.position_controllers['yaw'](yaw_error, dt)
        
        # Velocity control (inner loop) - generates control outputs
        control_outputs = {}
        for i, axis in enumerate(axes):
            vel_error = velocity_setpoints[axis] - current_velocity[i]
            control_outputs[axis] = self.velocity_controllers[axis](vel_error, dt)
        
        # Yaw rate control
        control_outputs['yaw'] = velocity_setpoints['yaw']  # Direct passthrough or use another PID
        
        # Apply output limits
        for axis, (min_val, max_val) in self.output_limits.items():
            control_outputs[axis] = max(min_val, min(control_outputs[axis], max_val))
        
        # Convert to control vector expected by drone
        # This mapping depends on the specific drone simulator's control scheme
        # Assuming: [roll_cmd, pitch_cmd, thrust_cmd, yaw_rate_cmd]
        control_vector = np.array([
            control_outputs['x'],    # roll command (affects y movement)
            -control_outputs['y'],   # pitch command (affects x movement, inverted)
            control_outputs['z'],    # thrust command
            control_outputs['yaw']   # yaw rate command
        ])
        
        return control_vector
        
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi] range."""
        return ((angle + math.pi) % (2 * math.pi)) - math.pi


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """LSTM-based feature extractor for sequential observations."""
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        lstm_hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize LSTM feature extractor.
        
        Parameters
        ----------
        observation_space : spaces.Dict
            Observation space (expects dictionary of arrays)
        lstm_hidden : int
            Hidden size of LSTM layers
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability between LSTM layers
        """
        super().__init__(observation_space, features_dim=lstm_hidden)
        
        # Determine input size from observation space
        self.input_size = 0
        for space in observation_space.values():
            if isinstance(space, spaces.Box):
                self.input_size += int(np.prod(space.shape))
            else:
                raise ValueError(f"Unsupported observation space: {space}")
        
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, obs):
        # Flatten and concatenate dict observation to tensor
        if isinstance(obs, dict):
            # Assume each value is a tensor of shape (batch_size, ...) or (...,)
            obs_list = []
            for k in sorted(obs.keys()):
                v = obs[k]
                if isinstance(v, np.ndarray):
                    v = torch.from_numpy(v)
                if not torch.is_tensor(v):
                    v = torch.tensor(v)
                # Flatten all but batch dimension
                if v.dim() > 1:
                    v = v.view(v.size(0), -1) if v.dim() == 2 else v.flatten()
                obs_list.append(v)
            x = torch.cat(obs_list, dim=-1)
        else:
            x = obs

        # Now x is a tensor, continue as before
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        batch_size = x.size(0)
        device = x.device
        h_0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden, device=device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        features = lstm_out[:, -1, :]
        return features


class SimpleLSTMPolicy(ActorCriticPolicy):
    """LSTM-based policy for sequential decision making.
    
    This policy uses an LSTM to process observation sequences and
    outputs actions and value estimates compatible with SB3.
    """
    
    def __init__(
        self, 
        *args, 
        lstm_hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs
    ):
        """Initialize LSTM policy.
        
        Parameters
        ----------
        *args : Any
            Arguments for parent class
        lstm_hidden : int
            Hidden size of LSTM
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        **kwargs : Any
            Keyword arguments for parent class
        """
        # Remove LSTM-specific kwargs from kwargs before calling super()
        features_extractor_class = LSTMFeatureExtractor
        features_extractor_kwargs = dict(
            lstm_hidden=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout,
        )
        kwargs["features_extractor_class"] = features_extractor_class
        kwargs["features_extractor_kwargs"] = features_extractor_kwargs

        # Remove LSTM-specific keys if present
        kwargs.pop("lstm_hidden", None)
        kwargs.pop("num_layers", None)
        kwargs.pop("dropout", None)

        super().__init__(*args, **kwargs)
    
    @staticmethod
    def _weights_init(module: nn.Module) -> None:
        """Initialize weights using orthogonal initialization.
        
        Parameters
        ----------
        module : nn.Module
            Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
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
        features = self.extract_features(obs)
        
        # Get action distribution
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(features)
            dist = self.action_dist.proba_distribution(action_logits=logits)
            action_params = logits  # For discrete actions, logits are the params
        else:  # Continuous actions
            mean_actions = self.action_mean(features)
            # Use state-independent log std
            log_std = self.action_log_std.expand_as(mean_actions)
            dist = self.action_dist.proba_distribution(mean_actions, log_std)
            action_params = mean_actions  # For continuous actions, means are the params
        
        # Sample actions
        actions = dist.get_actions(deterministic=deterministic)
        
        # Compute values
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
        features = self.extract_features(obs)
        
        # Compute values
        values = self.value_net(features).flatten()
        
        # Get action distribution
        if isinstance(self.action_space, spaces.Discrete):
            logits = self.action_net(features)
            dist = self.action_dist.proba_distribution(action_logits=logits)
        else:  # Continuous actions
            mean_actions = self.action_mean(features)
            log_std = self.action_log_std.expand_as(mean_actions)
            dist = self.action_dist.proba_distribution(mean_actions, log_std)
        
        # Compute log probabilities and entropy
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return values, log_prob, entropy
    
    def reset_hidden(self) -> None:
        """Reset LSTM hidden state between episodes."""
        if hasattr(self.features_extractor, "reset_hidden"):
            self.features_extractor.reset_hidden()
