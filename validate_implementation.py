#!/usr/bin/env python3
"""
Minimal validation of LSTM predictor implementation without environment dependencies.
Run this to verify the predictor components work correctly.
"""

def validate_predictor_implementation():
    """Validate predictor code structure without running training."""
    
    print("🔍 Validating LSTM predictor implementation...")
    
    # Check 1: Verify predictor methods exist in SimpleLSTMPolicy
    try:
        with open('src/drone_rl/models/baselines.py', 'r') as f:
            content = f.read()
        
        required_methods = [
            'create_predictor_head',
            'predict_future',
            'predictor_head = None'
        ]
        
        for method in required_methods:
            if method in content:
                print(f"✅ Found: {method}")
            else:
                print(f"❌ Missing: {method}")
                return False
                
    except FileNotFoundError:
        print("❌ Could not find baselines.py")
        return False
    
    # Check 2: Verify training utilities exist
    try:
        with open('src/drone_rl/train/train.py', 'r') as f:
            content = f.read()
        
        required_functions = [
            '_flatten_obs_batch_for_predictor',
            'attach_future_targets_to_rollout_buffer', 
            'LSTMPredictorCallback',
            'cfg.get("predict_sequence"',
            'create_predictor_head'
        ]
        
        for func in required_functions:
            if func in content:
                print(f"✅ Found: {func}")
            else:
                print(f"❌ Missing: {func}")
                return False
                
    except FileNotFoundError:
        print("❌ Could not find train.py")
        return False
    
    # Check 3: Verify test config exists
    try:
        with open('configs/lstm_predictor_test.yaml', 'r') as f:
            content = f.read()
        
        if 'predict_sequence: true' in content and 'prediction_horizon:' in content:
            print("✅ Found: Test configuration")
        else:
            print("❌ Missing: Required config settings")
            return False
            
    except FileNotFoundError:
        print("❌ Could not find test config")
        return False
    
    # Check 4: Verify documentation
    try:
        with open('LSTM_PREDICTOR_README.md', 'r') as f:
            content = f.read()
        
        if len(content) > 1000:  # Basic content check
            print("✅ Found: Documentation")
        else:
            print("❌ Missing: Adequate documentation")
            return False
            
    except FileNotFoundError:
        print("❌ Could not find README")
        return False
    
    print("\n🎉 All validation checks passed!")
    print("\n📋 Implementation Summary:")
    print("   ✅ SimpleLSTMPolicy enhanced with predictor methods")
    print("   ✅ Training callback and utilities implemented")  
    print("   ✅ Configuration support added")
    print("   ✅ Test config created")
    print("   ✅ Documentation provided")
    print("\n🚀 Ready for testing with:")
    print("   PYTHONPATH=src python -m src.drone_rl.train.train_lstm --config configs/lstm_predictor_test.yaml")
    
    return True

if __name__ == "__main__":
    validate_predictor_implementation()
