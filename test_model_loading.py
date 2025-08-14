"""Test script to verify model loading compatibility fix."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gymnasium as gym
from src.drone_rl.models.transformer_policy import TransformerActorCritic
from src.drone_rl.utils.model_compatibility import load_model_with_compatibility

def test_model_loading():
    """Test loading models with deprecated state_predictor components."""
    
    # Test paths - adjust these to actual model paths
    test_models = [
        "runs/teacher_large/best_model.zip",
        "runs/teacher_large_v2/best_model.zip", 
        "runs/teacher_large_v5/best_model.zip"
    ]
    
    try:
        # Create environment (this might fail without flycraft installed)
        env = gym.make("FlyCraft", max_episode_steps=1000)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        print("  This is expected if flycraft is not installed on this machine")
        return
    
    for model_path in test_models:
        if not os.path.exists(model_path):
            print(f"⚠ Model not found: {model_path}")
            continue
            
        print(f"\nTesting model: {model_path}")
        try:
            model = load_model_with_compatibility(
                model_path, 
                env, 
                TransformerActorCritic, 
                device="cpu"
            )
            print(f"✓ Successfully loaded: {model_path}")
            
            # Test a forward pass
            obs = env.reset()[0]
            if isinstance(obs, dict):
                obs = {k: v.unsqueeze(0) if hasattr(v, 'unsqueeze') else v for k, v in obs.items()}
            action, _ = model.predict(obs, deterministic=True)
            print(f"✓ Forward pass successful, action shape: {action.shape}")
            
        except Exception as e:
            print(f"✗ Failed to load {model_path}: {e}")
    
    env.close()

if __name__ == "__main__":
    test_model_loading()
