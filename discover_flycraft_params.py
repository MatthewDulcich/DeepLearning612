#!/usr/bin/env python3
"""Quick script to discover FlyCraft environment parameters."""

import gymnasium as gym

def discover_flycraft_params():
    """Test what parameters FlyCraft actually accepts."""
    print("🔍 Discovering FlyCraft environment parameters...")
    
    try:
        import flycraft
        
        # Try basic creation first
        print("\n1. Testing basic FlyCraft-v0 creation:")
        try:
            env = gym.make("FlyCraft-v0")
            print("✅ Basic creation successful")
            print(f"   Observation space: {env.observation_space}")
            print(f"   Action space: {env.action_space}")
            env.close()
        except Exception as e:
            print(f"❌ Basic creation failed: {e}")
            return
        
        # Test different parameter names
        print("\n2. Testing parameter variations:")
        
        param_tests = [
            # Test frequency variations
            {"step_frequency": 20},
            {"frequency": 20}, 
            {"control_frequency": 20},
            {"sim_frequency": 20},
            {"hz": 20},
            
            # Test other common parameters
            {"max_episode_steps": 500},
            {"episode_length": 500},
            {"time_limit": 500},
            
            # Test control modes
            {"control_mode": "guidance_law_mode"},
            {"controller": "guidance_law_mode"},
            
            # Test reward modes  
            {"reward_mode": "dense"},
            {"reward_type": "dense"},
            
            # Test goal configurations
            {"goal_cfg": {"type": "fixed_short", "distance_m": 200}},
            {"goal_config": {"type": "fixed_short", "distance_m": 200}},
            {"target_config": {"type": "fixed_short", "distance_m": 200}},
        ]
        
        for i, params in enumerate(param_tests):
            param_name = list(params.keys())[0]
            try:
                env = gym.make("FlyCraft-v0", **params)
                print(f"✅ {param_name}: ACCEPTED")
                env.close()
            except Exception as e:
                if "unexpected keyword argument" in str(e):
                    print(f"❌ {param_name}: NOT ACCEPTED")
                else:
                    print(f"⚠️  {param_name}: ERROR - {e}")
        
        # Try to inspect FlyCraft source if possible
        print("\n3. Trying to inspect FlyCraft class:")
        try:
            from flycraft.envs import FlyCraftEnv
            import inspect
            sig = inspect.signature(FlyCraftEnv.__init__)
            print(f"✅ FlyCraftEnv.__init__ signature:")
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    default = param.default if param.default != inspect.Parameter.empty else 'NO DEFAULT'
                    print(f"   {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'} = {default}")
        except Exception as e:
            print(f"❌ Could not inspect FlyCraftEnv: {e}")
            
    except ImportError:
        print("❌ FlyCraft not available for testing")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    discover_flycraft_params()
