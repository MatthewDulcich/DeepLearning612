# Import Error Fix Guide

## Problem
Getting `ImportError` for `stable_baselines3.common.utils import linear_schedule` when running the LSTM training.

## Solution

### 1. Install Compatible Versions
```bash
# Install specific compatible versions
pip install stable-baselines3==2.0.0
pip install sb3-contrib==2.0.0  
pip install gymnasium==0.28.0

# Or install latest with version constraints
pip install "stable-baselines3>=2.0.0,<3.0.0"
pip install "sb3-contrib>=2.0.0,<3.0.0"
pip install "gymnasium>=0.28.0,<1.0.0"
```

### 2. Test Imports
```bash
# Run the import test
python test_imports.py

# Should show all green checkmarks
```

### 3. Common Import Issues & Fixes

#### A. `linear_schedule` Import Error
**Error**: `ImportError: cannot import name 'linear_schedule' from 'stable_baselines3.common.utils'`

**Fix**: The training script now handles this automatically with fallback to constant learning rate.

#### B. `MlpLstmPolicy` Import Error  
**Error**: `ImportError: cannot import name 'MlpLstmPolicy'`

**Fix**: The script tries multiple import paths and falls back to string policy name.

#### C. Version Compatibility Issues
**Error**: Various import errors between SB3 versions

**Fix**: 
```bash
# Uninstall and reinstall with specific versions
pip uninstall stable-baselines3 sb3-contrib gymnasium
pip install stable-baselines3==2.0.0 sb3-contrib==2.0.0 gymnasium==0.28.0
```

### 4. Alternative: Use Conda Environment
```bash
# Create isolated environment
conda create -n recurrent_ppo python=3.9
conda activate recurrent_ppo

# Install packages
pip install stable-baselines3==2.0.0
pip install sb3-contrib==2.0.0
pip install gymnasium==0.28.0
pip install torch numpy pandas pyyaml tensorboard
```

### 5. Verify Installation
```bash
# Test the setup
python test_imports.py

# If successful, run training
./run_lstm_recurrent.sh --config configs/baseline_lstm.yaml
```

## Updated Requirements

The `requirements_platform.txt` has been updated with version constraints to prevent compatibility issues:

```txt
stable-baselines3>=2.0.0,<3.0.0
sb3_contrib>=2.0.0,<3.0.0
gymnasium>=0.28.0,<1.0.0
```

## If Still Having Issues

1. **Check Python version**: Requires Python 3.8+
2. **Check environment**: Make sure you're in the right virtual environment
3. **Clean install**: Remove all RL packages and reinstall from scratch
4. **Use the test script**: `python test_imports.py` will diagnose the specific issue

The training script has been made robust to handle these import variations automatically.
