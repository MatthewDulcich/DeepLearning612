import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbStepLoggingCallback(BaseCallback):
    """Logs metrics to wandb every N steps, including LSTM diagnostics if available."""
    def __init__(self, log_freq=5, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            metrics = {}
            # Log episode reward, length, loss, etc. if available
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                for k, v in self.model.logger.name_to_value.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = v
            # Log LSTM hidden state size if available
            policy = getattr(self.model, 'policy', None)
            if policy is not None and hasattr(policy, 'features_extractor'):
                fx = policy.features_extractor
                if hasattr(fx, 'lstm'):
                    lstm = fx.lstm
                    if hasattr(lstm, 'hidden_size'):
                        metrics['lstm_hidden_size'] = lstm.hidden_size
                    if hasattr(lstm, 'num_layers'):
                        metrics['lstm_num_layers'] = lstm.num_layers
            # Log current timestep
            metrics['global_step'] = self.num_timesteps
            if metrics:
                wandb.log(metrics, step=self.num_timesteps)
        return True
