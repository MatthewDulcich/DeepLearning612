"""Callback for managing LSTM hidden states during training."""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class LSTMResetCallback(BaseCallback):
    """Callback to reset LSTM hidden states when episodes end.
    
    This is crucial for LSTM policies as hidden states should be reset
    between episodes to prevent information leakage across episode boundaries.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check if any episodes are done
        if hasattr(self.training_env, 'buf_dones'):
            dones = self.training_env.buf_dones
        else:
            # Fallback for different vectorized env types
            dones = getattr(self.training_env, '_dones', None)
        
        if dones is not None and np.any(dones):
            # Reset hidden states for done environments
            if self.verbose > 1:
                print(f"LSTM Callback: dones shape={np.array(dones).shape}, dones={dones}")
                
            if hasattr(self.model.policy, 'reset_hidden'):
                self.model.policy.reset_hidden(done_mask=dones)
            elif hasattr(self.model.policy.features_extractor, 'reset_hidden'):
                self.model.policy.features_extractor.reset_hidden(done_mask=dones)
                
            if self.verbose > 0:
                n_done = np.sum(dones)
                print(f"Reset LSTM hidden states for {n_done} done environments")
        
        return True
    
    def _on_rollout_start(self) -> None:
        """Called before collecting a new rollout."""
        # Reset all hidden states at the start of each rollout
        if hasattr(self.model.policy, 'reset_hidden'):
            self.model.policy.reset_hidden()
        elif hasattr(self.model.policy.features_extractor, 'reset_hidden'):
            self.model.policy.features_extractor.reset_hidden()
            
        if self.verbose > 0:
            print("Reset all LSTM hidden states at rollout start")
