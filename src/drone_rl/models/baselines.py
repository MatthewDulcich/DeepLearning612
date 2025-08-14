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
        features_dim: int = 128,
        lstm_hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize LSTM feature extractor.
        
        Parameters
        ----------
        observation_space : spaces.Dict
            Observation space (expects dictionary of arrays)
        features_dim : int
            Output feature dimension
        lstm_hidden : int
            Hidden size of LSTM layers
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability between LSTM layers
        """
        super().__init__(observation_space, features_dim=features_dim)
        
        # Determine input size from observation space
        self.input_size = 0
        for space in observation_space.values():
            if isinstance(space, spaces.Box):
                self.input_size += int(np.prod(space.shape))
            else:
                raise ValueError(f"Unsupported observation space: {space}")
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Layer normalization after LSTM
        self.layer_norm = nn.LayerNorm(lstm_hidden)
        # Output projection with dropout
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # Hidden state for recurrent processing
        self.hidden = None
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process observations through LSTM.
        
        Parameters
        ----------
        observations : Dict[str, torch.Tensor]
            Dictionary of observation tensors
            
        Returns
        -------
        torch.Tensor
            Extracted features
        """
        # Explicitly concatenate goal and achieved_goal first, then other obs
        features = []
        for k in ["desired_goal", "achieved_goal"]:
            if k in observations:
                obs = observations[k]
                if len(obs.shape) > 3:
                    b, seq, *spatial = obs.shape
                    obs = obs.reshape(b, seq, -1)
                features.append(obs)
        for k in sorted(observations.keys()):
            if k not in ("desired_goal", "achieved_goal"):
                obs = observations[k]
                if len(obs.shape) > 3:
                    b, seq, *spatial = obs.shape
                    obs = obs.reshape(b, seq, -1)
                features.append(obs)
        x = torch.cat(features, dim=-1)  # [batch_size, seq_len, input_size]
        # TODO: Add mask support for variable-length sequences
        # Process through LSTM
        if self.hidden is None or x.size(0) != self.hidden[0].size(1):
            # Initialize hidden state if needed
            self.reset_hidden(batch_size=x.size(0), device=x.device)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        # Use last timestep output
        last_output = lstm_out[:, -1]
        # Project to output dimension
        return self.fc(last_output)
    
    def reset_hidden(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> None:
        """Reset LSTM hidden state.
        
        Parameters
        ----------
        batch_size : int
            Batch size for hidden state
        device : torch.device
            Device for hidden state tensors
        """
        # Initialize hidden state (h0, c0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
        self.hidden = (h0, c0)


class SimpleLSTMPolicy(ActorCriticPolicy):
    """LSTM-based policy for sequential decision making.
    
    This policy uses an LSTM to process observation sequences and
    outputs actions and value estimates compatible with SB3.
    """
    
    def __init__(
        self, 
        *args: Any, 
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        **kwargs: Any
    ):
        """Initialize LSTM policy.
        
        Parameters
        ----------
        *args : Any
            Arguments for parent class
        lstm_hidden : int
            Hidden size of LSTM
        lstm_layers : int
            Number of LSTM layers
        dropout : float
            Dropout probability
        **kwargs : Any
            Keyword arguments for parent class
        """
        # Set custom feature extractor
        features_kwargs = {
            "features_dim": 128,
            "lstm_hidden": lstm_hidden,
            "num_layers": lstm_layers,
            "dropout": dropout
        }
        kwargs.setdefault("features_extractor_class", LSTMFeatureExtractor)
        kwargs.setdefault("features_extractor_kwargs", features_kwargs)
        
        super().__init__(*args, **kwargs)
        
        # Action network depends on action space type
        if isinstance(self.action_space, spaces.Discrete):
            self.action_net = nn.Linear(self.features_dim, self.action_space.n)
        else:  # Continuous actions (Box)
            action_dim = int(np.prod(self.action_space.shape))
            self.action_mean = nn.Linear(self.features_dim, action_dim)
            # State-dependent log_std
            self.action_log_std_layer = nn.Linear(self.features_dim, action_dim)
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize weights
        self.apply(self._weights_init)
    
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
            # State-dependent log std
            log_std = self.action_log_std_layer(features)
            log_std = torch.clamp(log_std, -5, 2)  # Clamp for stability
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
            log_std = self.action_log_std_layer(features)
            log_std = torch.clamp(log_std, -5, 2)
            dist = self.action_dist.proba_distribution(mean_actions, log_std)
        
        # Compute log probabilities and entropy
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return values, log_prob, entropy
    
    def reset_hidden(self) -> None:
        """Reset LSTM hidden state between episodes."""
        if hasattr(self.features_extractor, "reset_hidden"):
            self.features_extractor.reset_hidden()
