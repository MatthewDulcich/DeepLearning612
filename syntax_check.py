#!/usr/bin/env python3
"""
Simple syntax check for our LSTM predictor implementation.
This script verifies the code compiles without importing heavy dependencies.
"""

import ast
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_class_methods(file_path, class_name, required_methods):
    """Check if a class has required methods"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                missing = [m for m in required_methods if m not in methods]
                return len(missing) == 0, missing, methods
        
        return False, required_methods, []
    except Exception as e:
        return False, [], f"Error: {e}"

def main():
    print("🔍 Checking LSTM Predictor Implementation Syntax...")
    
    # Check main files
    files_to_check = [
        "src/drone_rl/models/baselines.py",
        "src/drone_rl/train/train.py"
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        print(f"\n📁 Checking {file_path}...")
        
        if not Path(file_path).exists():
            print(f"❌ File not found: {file_path}")
            all_good = False
            continue
            
        is_valid, error = check_syntax(file_path)
        if is_valid:
            print(f"✅ Syntax OK")
        else:
            print(f"❌ {error}")
            all_good = False
    
    # Check specific class methods
    print(f"\n🔍 Checking LSTMPredictorCallback class...")
    
    has_methods, missing, found = check_class_methods(
        "src/drone_rl/train/train.py", 
        "LSTMPredictorCallback",
        ["_on_step", "_on_rollout_end", "__init__"]
    )
    
    if has_methods:
        print(f"✅ LSTMPredictorCallback has required methods: {found}")
    else:
        print(f"❌ LSTMPredictorCallback missing methods: {missing}")
        print(f"Found methods: {found}")
        all_good = False
    
    # Check SimpleLSTMPolicy methods
    print(f"\n🔍 Checking SimpleLSTMPolicy predictor methods...")
    
    has_methods, missing, found = check_class_methods(
        "src/drone_rl/models/baselines.py", 
        "SimpleLSTMPolicy",
        ["create_predictor_head", "predict_future"]
    )
    
    if has_methods:
        print(f"✅ SimpleLSTMPolicy has predictor methods")
    else:
        print(f"❌ SimpleLSTMPolicy missing methods: {missing}")
        all_good = False
    
    if all_good:
        print(f"\n🎉 All syntax checks passed! Implementation should work.")
        print(f"💡 The error you saw was due to missing '_on_step' method, which has been fixed.")
        print(f"💡 Try running the training again with the drone-rl environment activated.")
    else:
        print(f"\n❌ Some issues found. Please review the errors above.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
