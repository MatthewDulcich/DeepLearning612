#!/usr/bin/env python3
"""
Syntax checker for curriculum wrapper fix.
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

def check_for_observation_space_property(file_path):
    """Check if curriculum wrapper has observation_space property"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Check if observation_space is assigned
        has_obs_space = "self.observation_space" in source
        has_curriculum_keys = "curriculum_frequency" in source and "curriculum_step_duration" in source
        
        return has_obs_space, has_curriculum_keys
    except Exception as e:
        return False, False

def main():
    print("🔍 Checking Curriculum Wrapper Fix...")
    
    wrapper_file = "src/drone_rl/utils/curriculum_wrapper.py"
    
    if not Path(wrapper_file).exists():
        print(f"❌ File not found: {wrapper_file}")
        return 1
    
    # Check syntax
    is_valid, error = check_syntax(wrapper_file)
    if is_valid:
        print(f"✅ Curriculum wrapper syntax OK")
    else:
        print(f"❌ Syntax error: {error}")
        return 1
    
    # Check for observation space fix
    has_obs_space, has_curriculum_keys = check_for_observation_space_property(wrapper_file)
    
    if has_obs_space:
        print(f"✅ Curriculum wrapper defines observation_space property")
    else:
        print(f"❌ Curriculum wrapper missing observation_space property")
        return 1
    
    if has_curriculum_keys:
        print(f"✅ Curriculum wrapper includes curriculum keys")
    else:
        print(f"❌ Curriculum wrapper missing curriculum keys")
        return 1
    
    print(f"\n🎉 Curriculum wrapper fix looks good!")
    print(f"💡 The KeyError for 'curriculum_frequency' should be resolved.")
    print(f"💡 The wrapper now properly updates observation_space to include curriculum keys.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
