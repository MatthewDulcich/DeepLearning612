#!/usr/bin/env python3
"""
Pre-flight validation script for RecurrentPPO training setup.
Validates syntax, imports, and configurations without requiring FlyCraft.
"""

import sys
import subprocess
from pathlib import Path

def check_python_syntax():
    """Check Python syntax of all training scripts."""
    print("🔍 Checking Python syntax...")
    
    files_to_check = [
        "train_lstm_recurrent.py",
        "src/drone_rl/train/train.py", 
        "demo/app.py",
        "demo/generate_acmi.py",
        "test_recurrent_setup.py"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    compile(f.read(), file_path, 'exec')
                print(f"  ✅ {file_path}")
            except SyntaxError as e:
                print(f"  ❌ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
                return False
        else:
            print(f"  ⚠️  {file_path}: File not found")
    
    return True

def check_config_files():
    """Validate YAML configuration files."""
    print("\n📋 Checking configuration files...")
    
    try:
        import yaml
    except ImportError:
        print("  ⚠️  PyYAML not available, skipping config validation")
        return True
    
    config_files = [
        "configs/baseline_lstm.yaml",
        "configs/teacher_large.yaml",
        "configs/student_distilled.yaml"
    ]
    
    for config_path in config_files:
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    yaml.safe_load(f)
                print(f"  ✅ {config_path}")
            except yaml.YAMLError as e:
                print(f"  ❌ {config_path}: YAML error: {e}")
                return False
        else:
            print(f"  ⚠️  {config_path}: File not found")
    
    return True

def check_core_imports():
    """Check availability of core packages (excluding FlyCraft)."""
    print("\n📦 Checking core package imports...")
    
    core_packages = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("gymnasium", "gymnasium"), 
        ("stable_baselines3", "stable_baselines3"),
    ]
    
    optional_packages = [
        ("sb3_contrib", "sb3-contrib"),
        ("yaml", "PyYAML"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
    ]
    
    all_good = True
    
    # Check core packages
    for package, pip_name in core_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (install with: pip install {pip_name})")
            all_good = False
    
    # Check optional packages
    for package, pip_name in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} (install with: pip install {pip_name})")
    
    # FlyCraft check (expected to fail on macOS)
    try:
        import flycraft
        print(f"  ✅ flycraft (Linux environment detected)")
    except ImportError:
        import platform
        if platform.system() == "Darwin":
            print(f"  ⚠️  flycraft (expected on macOS - training requires Linux)")
        else:
            print(f"  ❌ flycraft (required for training)")
            all_good = False
    
    return all_good

def check_script_executability():
    """Check that shell scripts are executable."""
    print("\n🚀 Checking script permissions...")
    
    shell_scripts = [
        "run_lstm_recurrent.sh"
    ]
    
    for script in shell_scripts:
        script_path = Path(script)
        if script_path.exists():
            if script_path.stat().st_mode & 0o111:  # Check execute permission
                print(f"  ✅ {script} (executable)")
            else:
                print(f"  ⚠️  {script} (not executable - run: chmod +x {script})")
        else:
            print(f"  ❌ {script} (not found)")

def check_directory_structure():
    """Verify expected directory structure."""
    print("\n📁 Checking directory structure...")
    
    expected_dirs = [
        "configs",
        "demo", 
        "src/drone_rl",
        "src/drone_rl/models",
        "src/drone_rl/train",
        "src/drone_rl/utils"
    ]
    
    expected_files = [
        "train_lstm_recurrent.py",
        "configs/baseline_lstm.yaml", 
        "demo/app.py",
        "demo/generate_acmi.py",
        "src/drone_rl/models/baselines.py",
        "src/drone_rl/train/train.py",
        "src/drone_rl/utils/metrics.py"
    ]
    
    for directory in expected_dirs:
        if Path(directory).is_dir():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ (missing)")
    
    for file_path in expected_files:
        if Path(file_path).is_file():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (missing)")

def run_quick_import_test():
    """Test importing key modules without FlyCraft dependency."""
    print("\n🧪 Running quick import tests...")
    
    try:
        # Test RecurrentPPO import 
        from sb3_contrib import RecurrentPPO
        print("  ✅ RecurrentPPO import successful")
    except ImportError as e:
        print(f"  ❌ RecurrentPPO import failed: {e}")
        return False
    
    try:
        # Test local modules (without FlyCraft)
        sys.path.append('.')
        from src.drone_rl.models.baselines import SimpleLSTMPolicy
        print("  ✅ SimpleLSTMPolicy import successful")
    except ImportError as e:
        print(f"  ❌ SimpleLSTMPolicy import failed: {e}")
        return False
    
    try:
        from src.drone_rl.utils.metrics import time_to_collision
        print("  ✅ Metrics utilities import successful")
    except ImportError as e:
        print(f"  ❌ Metrics utilities import failed: {e}")
        return False
    
    return True

def print_platform_specific_notes():
    """Print platform-specific setup notes."""
    import platform
    
    print(f"\n💡 Platform-specific notes:")
    print(f"   Detected OS: {platform.system()} {platform.release()}")
    
    if platform.system() == "Darwin":  # macOS
        print("""
   macOS Setup:
   - FlyCraft training is not supported on macOS
   - You can validate syntax and test imports locally
   - Use Linux (local/cloud) for actual training
   - Consider using Docker for consistent environment
        """)
    elif platform.system() == "Linux":
        print("""
   Linux Setup:
   - Install FlyCraft manually: 
     git clone https://github.com/flycraft-team/flycraft
     cd flycraft && pip install -e .
   - GPU support available with CUDA PyTorch
   - All features should work properly
        """)
    else:
        print(f"   - Platform {platform.system()} not explicitly tested")

def main():
    """Run all validation checks."""
    print("🛫 RecurrentPPO Training Pre-flight Check")
    print("=" * 50)
    
    checks = [
        ("Syntax Validation", check_python_syntax),
        ("Configuration Files", check_config_files), 
        ("Core Imports", check_core_imports),
        ("Directory Structure", check_directory_structure),
        ("Script Permissions", check_script_executability),
        ("Import Tests", run_quick_import_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Validation Summary:")
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False
    
    print_platform_specific_notes()
    
    if all_passed:
        print("\n🎉 All checks passed! Ready for training setup.")
        print("\nNext steps:")
        print("   1. Install missing optional dependencies if needed")
        print("   2. On Linux: Install FlyCraft manually") 
        print("   3. Run: ./run_lstm_recurrent.sh --config configs/baseline_lstm.yaml")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please address issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
