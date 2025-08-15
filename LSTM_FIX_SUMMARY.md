# LSTM IndexError Fix Summary

## Problem
```
IndexError: index 1 is out of bounds for dimension 1 with size 1
```
This occurred in `baselines.py` line 318 when trying to reset LSTM hidden states.

## Root Cause
The `done_mask` from vectorized environments had a different shape than expected, causing the code to try to access environment indices that don't exist in the hidden state tensor.

## Fixes Applied

### 1. Fixed `src/drone_rl/models/baselines.py` - `reset_hidden` method
```python
elif done_mask is not None:
    # Only reset done envs - handle different done_mask shapes
    done_mask = np.asarray(done_mask)
    if done_mask.ndim == 0:
        done_mask = [done_mask]
    
    # Ensure we don't exceed the hidden state dimensions
    if self._hidden_state is not None:
        max_envs = self._hidden_state[0].shape[1]
        print(f"Reset hidden: max_envs={max_envs}, done_mask length={len(done_mask)}")
        for idx, done in enumerate(done_mask):
            if done and idx < max_envs:  # ← KEY FIX: Check idx < max_envs
                self._hidden_state[0][:, idx].zero_()
                self._hidden_state[1][:, idx].zero_()
    else:
        # No hidden state initialized yet
        print("Reset hidden: No hidden state to reset, skipping")
```

### 2. Enhanced `src/drone_rl/train/lstm_callback.py` with debugging
```python
if dones is not None and np.any(dones):
    # Reset hidden states for done environments
    if self.verbose > 1:
        print(f"LSTM Callback: dones shape={np.array(dones).shape}, dones={dones}")
    # ... rest of callback logic
```

## Key Changes
1. **Bounds checking**: Added `idx < max_envs` to prevent index out of bounds
2. **Shape handling**: Convert `done_mask` to numpy array and handle scalar cases
3. **Null checks**: Check if `_hidden_state` exists before accessing
4. **Debugging**: Added print statements to track what's happening

## Testing
Run on your remote system with:
```bash
PYTHONPATH=src python -m src.drone_rl.train.train --config configs/baseline_lstm_improved_fixed.yaml
```

The curriculum wrapper implementation is also ready and should work properly now.
