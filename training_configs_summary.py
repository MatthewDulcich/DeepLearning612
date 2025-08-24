#!/usr/bin/env python3
"""
Training Configuration Summary for LSTM Predictor

This script shows the available training configurations and their trade-offs.
"""

configs = {
    "lstm_predictor_test.yaml": {
        "description": "Quick validation config",
        "horizon": 5,
        "timesteps": "250k",
        "n_envs": 4,
        "estimated_time": "1-2 hours",
        "memory_usage": "~1GB",
        "purpose": "Validate implementation works"
    },
    
    "lstm_predictor_medium.yaml": {
        "description": "Development config", 
        "horizon": 50,
        "timesteps": "1M",
        "n_envs": 6,
        "estimated_time": "6-8 hours",
        "memory_usage": "~3GB", 
        "purpose": "Test medium-term prediction"
    },
    
    "lstm_predictor_full.yaml": {
        "description": "Production config",
        "horizon": 200, 
        "timesteps": "2M",
        "n_envs": 8,
        "estimated_time": "12-16 hours",
        "memory_usage": "~6GB",
        "purpose": "Full 200-step prediction training"
    }
}

def print_config_summary():
    print("🚀 LSTM Predictor Training Configurations\n")
    
    for config_name, details in configs.items():
        print(f"📁 {config_name}")
        print(f"   📝 {details['description']}")
        print(f"   🔮 Horizon: {details['horizon']} steps")
        print(f"   ⏱️  Training: {details['timesteps']} timesteps ({details['estimated_time']})")
        print(f"   🔧 Resources: {details['n_envs']} envs, {details['memory_usage']}")
        print(f"   🎯 Purpose: {details['purpose']}")
        print()
    
    print("💡 Recommended progression:")
    print("   1. Start with 'test' config to validate")
    print("   2. Run 'medium' config for development") 
    print("   3. Scale to 'full' config for final training")
    print("   4. Compare against baseline LSTM without predictor")
    
    print("\n🧠 Key Features:")
    print("   ✅ Non-autoregressive prediction (predicts all H steps at once)")
    print("   ✅ Separate optimizer (doesn't interfere with PPO)")
    print("   ✅ Memory-safe sampling (adaptive max_samples)")
    print("   ✅ Curriculum learning compatible")
    print("   ✅ Conservative hyperparameters for stability")

if __name__ == "__main__":
    print_config_summary()
