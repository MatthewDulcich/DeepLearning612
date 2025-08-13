# RecurrentPPO LSTM Implementation

This document explains the RecurrentPPO LSTM baseline implementation that replaces the previous SimpleLSTMPolicy approach.

## Overview

The project now supports two LSTM implementations:

1. **SimpleLSTMPolicy** (Legacy): Custom LSTM policy using standard PPO
2. **RecurrentPPO** (New): SB3-Contrib's RecurrentPPO with proper sequence handling

## Key Improvements

### Proper Sequence Handling
- RecurrentPPO maintains hidden states across episode boundaries
- Better handling of partial observability in sequential environments
- Improved temporal credit assignment

### Advanced Metrics
- Episode success rate, collision rate, path deviation percentiles
- Action magnitude statistics and exploration metrics
- Comprehensive CSV and TensorBoard logging

### Robust Training
- VecNormalize for observation/reward normalization
- Custom evaluation callback with detailed metrics
- Automatic checkpoint saving and model evaluation

## Usage

### Training

Use the dedicated training script:
```bash
python train_lstm_recurrent.py --config configs/baseline_lstm.yaml --wandb
```

Or use the shell script:
```bash
./run_lstm_recurrent.sh --wandb --seed 42
```

### Configuration

The `configs/baseline_lstm.yaml` has been updated for RecurrentPPO:

```yaml
algorithm: RecurrentPPO
policy: MlpLstmPolicy

recurrent_kwargs:
  lstm_hidden_size: 256
  n_lstm_layers: 2
  shared_lstm: False
  enable_critic_lstm: True

ppo_kwargs:
  n_steps: 128          # Shorter sequences for LSTM
  batch_size: 64        # Smaller batches
  learning_rate: 3.0e-4
```

### Demo Compatibility

The Streamlit demo (`demo/app.py`) automatically detects and loads RecurrentPPO models:

- Maintains identical user interface
- Automatically handles hidden state management
- Falls back to SimpleLSTMPolicy if RecurrentPPO not available

## Model Loading

The demo app intelligently detects model types:

```python
# Auto-detection based on filename
is_recurrent = "recurrent" in model_path.lower()

if is_recurrent and RECURRENT_PPO_AVAILABLE:
    model = RecurrentPPO.load(model_path, env=env)
    model._is_recurrent = True
else:
    # Fallback to regular PPO + SimpleLSTMPolicy
    model = PPO.load(model_path, env=env, custom_objects={"policy_class": SimpleLSTMPolicy})
```

## Dependencies

Install the required packages:

```bash
pip install -r requirements_recurrent.txt
```

Key dependencies:
- `sb3-contrib>=2.0.0` - RecurrentPPO implementation
- `stable-baselines3>=2.0.0` - Base RL algorithms
- `flycraft` - Flight simulation environment

## Metrics and Logging

The RecurrentPPO implementation includes comprehensive metrics:

### Episode Metrics
- `episode_success_rate`: Percentage of successful missions
- `episode_collision_rate`: Percentage of episodes ending in collisions
- `episode_length_mean/std`: Episode duration statistics

### Performance Metrics
- `path_deviation_p50/p90/p95`: Path following accuracy percentiles
- `action_magnitude_mean/std`: Control effort statistics
- `exploration_rate`: Action entropy for exploration measurement

### Training Metrics
- Standard PPO metrics (policy loss, value loss, explained variance)
- Learning rate and clip range schedules
- Gradient norms and policy entropy

## Files Modified

### Core Training
- `train_lstm_recurrent.py`: New dedicated RecurrentPPO training script
- `src/drone_rl/train/train.py`: Updated to support both PPO and RecurrentPPO
- `configs/baseline_lstm.yaml`: Updated for RecurrentPPO configuration

### Demo Application
- `demo/app.py`: Enhanced to support both LSTM implementations
- Added hidden state management for RecurrentPPO models
- Backward compatibility with SimpleLSTMPolicy models

### Configuration and Scripts
- `requirements_recurrent.txt`: Dependencies for RecurrentPPO
- `run_lstm_recurrent.sh`: Convenient training script
- `test_recurrent_setup.py`: Dependency verification script

## Migration Guide

### From SimpleLSTMPolicy to RecurrentPPO

1. **Update Config**: Change `policy: lstm` to `algorithm: RecurrentPPO`
2. **Add RecurrentPPO kwargs**: Configure LSTM parameters in `recurrent_kwargs`
3. **Adjust hyperparameters**: Use smaller batch sizes and shorter sequences
4. **Install dependencies**: `pip install sb3-contrib`

### Model Compatibility

- Old SimpleLSTMPolicy models work with updated demo app
- New RecurrentPPO models are auto-detected and loaded correctly
- Model type is preserved in checkpoint metadata

## Performance Comparison

Expected improvements with RecurrentPPO:
- Better sample efficiency for sequential tasks
- Improved handling of partial observability
- More stable training with proper hidden state management
- Enhanced temporal credit assignment

## Troubleshooting

### Import Errors
```bash
# Install missing dependencies
pip install sb3-contrib flycraft

# Check installation
python test_recurrent_setup.py
```

### Model Loading Issues
- Ensure correct model path in demo app
- Check that RecurrentPPO models use `.zip` extension
- Verify model was saved with `RecurrentPPO.save()`

### Training Issues
- Reduce batch size if memory constrained
- Adjust `n_steps` for different sequence lengths
- Monitor metrics in TensorBoard for convergence
