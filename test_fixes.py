#!/usr/bin/env python3
"""Quick test script to validate fixes and debug training issues."""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_environment_basic():
    """Test basic FlyCraft environment functionality."""
    print("🔧 Testing basic FlyCraft environment...")
    
    test_code = """
import gymnasium as gym
try:
    import flycraft
    env = gym.make("FlyCraft-v0", step_frequence=20, reward_mode="dense")
    obs, info = env.reset()
    print(f"✅ Environment created successfully")
    print(f"   Obs space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Test a few steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if i < 3:
            print(f"   Step {i}: reward={reward:.3f}")
    
    print(f"   Total reward (10 steps): {total_reward:.3f}")
    env.close()
    
except Exception as e:
    print(f"❌ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Environment test failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ Environment test timed out")
        return False
    except Exception as e:
        print(f"❌ Environment test error: {e}")
        return False

def test_training_quick():
    """Run a very short training test to validate config."""
    print("\n🚀 Testing training pipeline...")
    
    # Use the improved config for a quick test
    config_path = "configs/baseline_lstm_improved_fixed.yaml"
    
    # Modify config for quick test
    quick_config = f"""
env_id: FlyCraft-v0
eval_freq: 500
n_envs: 2
output_dir: runs
policy: lstm
policy_kwargs:
  lstm_hidden: 64
  lstm_layers: 1
  dropout: 0.0
  n_envs: 2
ppo_kwargs:
  batch_size: 64
  clip_range: 0.1
  gae_lambda: 0.95
  gamma: 0.98
  learning_rate: 3e-4
  n_epochs: 2
  n_steps: 128
  ent_coef: 0.02
  vf_coef: 1.0
  max_grad_norm: 0.5
run_name: quick_test_{int(time.time())}
save_freq: 1000
seed: 42
timesteps: 1000  # Very short for testing

# Fixed environment
step_frequence: 20
control_mode: guidance_law_mode
reward_mode: dense
goal_cfg:
  type: fixed_short
  distance_m: 400

n_eval_episodes: 5
verbose: 1
"""
    
    # Write quick config
    with open("configs/quick_test.yaml", "w") as f:
        f.write(quick_config)
    
    try:
        cmd = [sys.executable, "-m", "src.drone_rl.train.train", 
               "--config", "configs/quick_test.yaml"]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Training test completed successfully!")
            # Look for success rate in output
            if "success_rate" in result.stdout:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "success_rate" in line or "Success rate" in line:
                        print(f"   {line.strip()}")
            return True
        else:
            print(f"❌ Training test failed:")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Training test error: {e}")
        return False

def run_diagnosis():
    """Run comprehensive diagnosis."""
    print("🔬 RUNNING COMPREHENSIVE DIAGNOSIS")
    print("=" * 50)
    
    # Test 1: Basic environment
    env_ok = test_environment_basic()
    
    # Test 2: Quick training
    if env_ok:
        training_ok = test_training_quick()
    else:
        print("⏭️  Skipping training test due to environment issues")
        training_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DIAGNOSIS SUMMARY:")
    print(f"   Environment: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"   Training:    {'✅ PASS' if training_ok else '❌ FAIL'}")
    
    if env_ok and training_ok:
        print("\n🎉 DIAGNOSIS COMPLETE - Ready for full training!")
        print("   Recommended next steps:")
        print("   1. Run baseline: python -m src.drone_rl.train.train_lstm --config configs/baseline_lstm_improved_fixed.yaml")
        print("   2. Run curriculum: python -m src.drone_rl.train.train --config configs/curriculum_transformer_fixed.yaml")
    else:
        print("\n⚠️  ISSUES DETECTED - See errors above")

if __name__ == "__main__":
    # Change to project directory if needed
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    run_diagnosis()
