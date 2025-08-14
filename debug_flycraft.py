"""Debug script to understand FlyCraft environment behavior and success conditions."""

import gymnasium as gym
import numpy as np

def analyze_flycraft_environment():
    """Analyze the FlyCraft environment to understand success conditions."""
    try:
        import flycraft
        env = gym.make("FlyCraft-v0")
        
        print("=== FlyCraft Environment Analysis ===")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Run a few random episodes to understand the environment
        for episode in range(3):
            obs, info = env.reset()
            print(f"\n--- Episode {episode + 1} ---")
            print(f"Initial observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not dict'}")
            print(f"Initial info: {info}")
            
            total_reward = 0
            step_count = 0
            
            for step in range(100):  # Limit steps for analysis
                action = env.action_space.sample()  # Random action
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                step_count += 1
                
                # Print interesting step information
                if step < 5 or step % 20 == 0:
                    print(f"  Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
                    if isinstance(info, dict) and 'is_success' in info:
                        print(f"    Success flag: {info['is_success']}")
                    if isinstance(info, dict):
                        print(f"    Info keys: {list(info.keys())}")
                
                if terminated or truncated:
                    print(f"  Episode ended at step {step}")
                    break
            
            print(f"Episode {episode + 1} total reward: {total_reward:.3f}, steps: {step_count}")
            if isinstance(info, dict) and 'is_success' in info:
                print(f"Final success: {info['is_success']}")
        
        env.close()
        
    except ImportError:
        print("FlyCraft not available for analysis")
    except Exception as e:
        print(f"Error analyzing environment: {e}")

if __name__ == "__main__":
    analyze_flycraft_environment()
