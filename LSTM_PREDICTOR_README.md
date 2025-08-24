# LSTM Future Predictor Implementation

## Ov## Usage

### Quick Testing (5-step prediction)
```bash
# Test the implementation with a small horizon
PYTHONPATH=src python -m src.drone_rl.train.train --config configs/lstm_predictor_test.yaml
```

### Medium-Scale Training (50-step prediction)  
```bash
# Medium-scale training for development
PYTHONPATH=src python -m src.drone_rl.train.train --config configs/lstm_predictor_medium.yaml
```

### Full-Scale Training (200-step prediction)
```bash
# Full production training with 200-step prediction
PYTHONPATH=src python -m src.drone_rl.train.train --config configs/lstm_predictor_full.yaml
```

### Baseline Comparison
```bash
# Train LSTM without predictor for comparison
PYTHONPATH=src python -m src.drone_rl.train.train_lstm --config configs/baseline_lstm.yaml
```implementation adds a non-autoregressive 200-step future predictor to the LSTM policy as an auxiliary task alongside PPO training. The predictor learns to predict the next H future states from the current LSTM features, providing benefits for representation learning and potential model-based planning.

## Implementation Details

### 1. Predictor Head Architecture

The predictor is implemented as a simple linear layer that maps from LSTM features to H×state_dim outputs:

- **Input**: LSTM features tensor [B, features_dim] (e.g., 128D)
- **Output**: Predicted future states [B, H, state_dim] (e.g., [B, 200, obs_dim])
- **Architecture**: Single linear layer with no activation (regression task)

### 2. Training Process

The predictor is trained using a separate optimizer and callback:

1. **Data Collection**: After each PPO rollout, stacked future states are computed from rollout_buffer.observations
2. **Target Generation**: For each timestep t, collect the next H observations as ground truth targets
3. **Training**: MSE loss between predicted and actual future states, with gradient clipping
4. **Isolation**: Separate optimizer and optional feature extractor freezing to avoid destabilizing PPO

### 3. Key Functions Added

#### SimpleLSTMPolicy Methods
- `create_predictor_head(horizon, state_dim, device)`: Creates the predictor linear layer
- `predict_future(features)`: Forward pass returning [B, H, state_dim] predictions

#### Training Utilities  
- `_flatten_obs_batch_for_predictor(obs_batch)`: Robust flattening for dict observations
- `attach_future_targets_to_rollout_buffer(rb, n_envs, n_steps, H)`: Creates stacked future targets
- `LSTMPredictorCallback`: Callback that trains predictor at rollout end

## Training Configuration Scaling

The system automatically adjusts predictor training parameters based on horizon:

- **H ≤ 10**: max_samples=1024, lr=1e-4 (more samples, standard learning rate)
- **H ≤ 50**: max_samples=512, lr=1e-4 (moderate samples, standard learning rate)  
- **H > 50**: max_samples=256, lr=5e-5 (fewer samples, lower learning rate for stability)

### Memory Considerations

- **H=5**: ~0.5GB predictor memory
- **H=50**: ~2.5GB predictor memory
- **H=200**: ~5GB predictor memory

Monitor memory usage and reduce `n_envs` or `max_samples` if needed.

### Training Progression

1. **Start with H=5** to validate implementation
2. **Scale to H=50** to test medium-term prediction  
3. **Full H=200** for production training
4. **Compare against baseline** LSTM without predictor

## Expected Training Times

- **Test (H=5, 250k steps)**: ~1-2 hours
- **Medium (H=50, 1M steps)**: ~6-8 hours  
- **Full (H=200, 2M steps)**: ~12-16 hours

Times depend on hardware and environment complexity.

## Usage Examples

### Testing with Small Horizon
```bash
# Test with H=5 for quick validation
PYTHONPATH=src python -m src.drone_rl.train.train --config configs/lstm_predictor_test.yaml
```

### Production Training with H=200
```bash
# Modify teacher_large.yaml to use policy: lstm, then:
PYTHONPATH=src python -m src.drone_rl.train.train --config configs/teacher_large.yaml
```

## Tuning Recommendations

### Conservative Settings (Recommended)
- `max_samples: 512` - Limit memory usage and training time
- `lr: 1e-4` - Small learning rate for auxiliary task
- `loss_weight: 1.0` - Can be reduced if PPO becomes unstable
- `freeze_extractor: true` - Only train predictor head, keep LSTM frozen

### Advanced Settings
- `freeze_extractor: false` - Allow gradients into LSTM (monitor PPO stability)
- `loss_weight: 0.1` - Reduce weight if joint training destabilizes policy
- Increase `max_samples` if memory allows and you want more training signal

### Memory Considerations

For H=200 with typical observation dimensions:
- Memory usage ≈ `n_envs × n_steps × H × obs_dim × 4 bytes`
- Example: 16 envs × 2048 steps × 200 × 20D = ~5GB
- Use `max_samples=512` to limit batch size during training

## Monitoring

The implementation logs:
- `predictor/loss` - MSE loss during training
- Memory warnings if future_states > 1GB
- Callback status messages during setup and training

## Benefits

1. **Representation Learning**: Auxiliary prediction task improves learned features
2. **Model-based Planning**: Predicted trajectories can be used for collision avoidance
3. **Diagnostics**: Compare predicted vs actual futures to detect model drift
4. **Transfer Learning**: Pre-trained dynamics can transfer to new tasks

## Safety Features

- **Separate Optimizer**: Predictor training isolated from PPO updates
- **Memory Limits**: Configurable sampling to prevent OOM
- **Error Handling**: Graceful degradation if observation flattening fails
- **Compatibility**: Falls back gracefully when predictor disabled

## Next Steps

1. Start with `prediction_horizon: 10` to validate implementation
2. Monitor `predictor/loss` convergence and PPO stability
3. Gradually increase horizon to 200 once validated
4. Experiment with joint training by setting `freeze_extractor: false`
5. Use predicted trajectories for model-based control if desired

## File Changes Made

- `src/drone_rl/models/baselines.py`: Added predictor methods to SimpleLSTMPolicy
- `src/drone_rl/train/train.py`: Added helper functions and callback integration
- `configs/lstm_predictor_test.yaml`: Test configuration with H=5
