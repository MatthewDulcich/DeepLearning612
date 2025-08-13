#!/usr/bin/env python3
"""
Test script to verify stable-baselines3 imports work correctly.
"""

def test_imports():
    """Test critical imports for RecurrentPPO training."""
    print("Testing stable-baselines3 imports...")
    
    try:
        import stable_baselines3
        print(f"✅ stable-baselines3 version: {stable_baselines3.__version__}")
    except ImportError as e:
        print(f"❌ stable-baselines3 import failed: {e}")
        return False
    
    try:
        from stable_baselines3.common.utils import linear_schedule
        print("✅ linear_schedule imported from utils")
        USE_SCHEDULE = True
    except ImportError:
        try:
            from stable_baselines3.common.schedule import linear_schedule  
            print("✅ linear_schedule imported from schedule")
            USE_SCHEDULE = True
        except ImportError:
            print("⚠️  linear_schedule not available, will use constant learning rate")
            USE_SCHEDULE = False
    
    try:
        import sb3_contrib
        print(f"✅ sb3_contrib version: {sb3_contrib.__version__}")
    except ImportError as e:
        print(f"❌ sb3_contrib import failed: {e}")
        print("   Install with: pip install sb3-contrib")
        return False
    
    try:
        from sb3_contrib import RecurrentPPO
        print("✅ RecurrentPPO imported successfully")
    except ImportError as e:
        print(f"❌ RecurrentPPO import failed: {e}")
        return False
    
    try:
        from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
        print("✅ MlpLstmPolicy imported successfully")
    except ImportError as e:
        print(f"❌ MlpLstmPolicy import failed: {e}")
        print("   Try: from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy")
        return False
    
    print("\n✅ All critical imports successful!")
    return True

def test_simple_model():
    """Test creating a simple RecurrentPPO model."""
    print("\nTesting RecurrentPPO model creation...")
    
    try:
        import gymnasium as gym
        from sb3_contrib import RecurrentPPO
        
        # Use CartPole for simple test
        env = gym.make("CartPole-v1")
        
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            verbose=0
        )
        
        print("✅ RecurrentPPO model created successfully")
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        test_simple_model()
    else:
        print("\n❌ Import tests failed. Please install dependencies:")
        print("   pip install stable-baselines3 sb3-contrib gymnasium")
