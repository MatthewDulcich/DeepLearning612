import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbLoggingCallback(BaseCallback):
    """Logs key RL metrics to wandb at every rollout."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log all keys in the logger's latest dict to wandb
        if self.locals.get('infos'):
            infos = self.locals['infos']
            if isinstance(infos, list) and len(infos) > 0:
                info = infos[0]
                if isinstance(info, dict):
                    for k, v in info.items():
                        if isinstance(v, (int, float)):
                            wandb.log({k: v}, step=self.num_timesteps)
        # Log SB3 logger metrics
        if hasattr(self.model, 'logger'):
            log_dict = self.model.logger.name_to_value
            for k, v in log_dict.items():
                if isinstance(v, (int, float)):
                    wandb.log({k: v}, step=self.num_timesteps)
        return True
