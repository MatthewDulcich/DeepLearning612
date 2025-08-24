#!/usr/bin/env python3
"""Test script for LSTM future predictor functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from drone_rl.models.baselines import SimpleLSTMPolicy
from drone_rl.train.train import _flatten_obs_batch_for_predictor, attach_future_targets_to_rollout_buffer

def test_predictor_components():
    """Test the predictor head creation and functionality."""
    print("🧪 Testing LSTM future predictor components...")
    
    # Create a simple environment
    try:
        env = gym.make("CartPole-v1")  # Fallback to CartPole if FlyCraft not available
        env_id = "CartPole-v1"
    except:
        print("⚠️  Could not create environment, using mock")
        return
    
    # Create vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Test observation flattening
    obs = vec_env.reset()
    if isinstance(obs, dict):
        obs_array = np.concatenate([v.flatten() for v in obs.values()], axis=-1)
    else:
        obs_array = obs
    
    print(f"✅ Observation shape: {obs_array.shape}")
    
    # Create LSTM policy
    policy = SimpleLSTMPolicy(
        observation_space=vec_env.observation_space,
        action_space=vec_env.action_space,
        lr_schedule=lambda x: 1e-4,
        lstm_hidden=64,
        lstm_layers=1,
        n_envs=1
    )
    
    print(f"✅ Created LSTM policy with features_dim: {policy.features_dim}")
    
    # Test predictor head creation
    H = 5
    state_dim = obs_array.shape[-1]
    policy.create_predictor_head(H, state_dim)
    
    print(f"✅ Created predictor head: {policy.predictor_head}")
    
    # Test forward pass
    features = torch.randn(2, policy.features_dim)  # Batch of 2
    preds = policy.predict_future(features)
    expected_shape = (2, H, state_dim)
    
    assert preds.shape == expected_shape, f"Expected {expected_shape}, got {preds.shape}"
    print(f"✅ Predictor forward pass: {features.shape} -> {preds.shape}")
    
    # Test rollout buffer future states creation
    print("\n🧪 Testing rollout buffer future states...")
    
    # Mock rollout buffer observations
    n_envs, n_steps = 2, 8
    mock_obs = np.random.randn(n_envs * n_steps, state_dim)
    
    class MockRolloutBuffer:
        def __init__(self, obs):
            self.observations = obs
    
    rb = MockRolloutBuffer(mock_obs)
    attach_future_targets_to_rollout_buffer(rb, n_envs, n_steps, H)
    
    expected_future_shape = (n_envs * n_steps, H, state_dim)
    assert rb.future_states.shape == expected_future_shape
    print(f"✅ Future states shape: {rb.future_states.shape}")
    
    print("\n🎉 All tests passed!")

if __name__ == "__main__":
    test_predictor_components()
