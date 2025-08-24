
import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo import PPO
from drone_rl.models.transformer_policy import TransformerActorCritic

class PPOWithMSE(PPO):
    """
    PPO extension that adds an auxiliary MSE loss (e.g., for sequence prediction) to the PPO loss.
    Assumes the policy has a `state_predictor` and a `get_seq_prediction_targets` method or similar.
    """
    def __init__(self, *args, aux_coef=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coef = aux_coef

    def train(self):
        # Copied from the original SB3 github repo
        #----------------------------------------------------------------------------------------------------------------
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Checks to see if MSE loss should be added (This is added and not from the SB3 repo)
                if hasattr(self.policy, 'state_predictor') and hasattr(self.policy, 'get_seq_prediction_targets'):
                    # Get a batch of data for auxiliary prediction. The helper should
                    # return (embedding, target_sequence) where embedding is the
                    # encoder embedding [B, embed_dim] and target_sequence is
                    # [B, H, state_dim]. This keeps the predictor API consistent.
                    batch = self.policy.get_seq_prediction_targets(batch_size=self.batch_size)
                    if batch is not None:
                        try:
                            embedding, target_sequence = batch
                            # Predict future states using the state_predictor
                            # StateSequencePredictor.forward expects (encoder_embedding, teacher_forcing=False, teacher_seq=None)
                            pred_sequence = self.policy.state_predictor(embedding)

                            # Compute MSE loss between predicted and true future states
                            mse_loss = F.mse_loss(pred_sequence, target_sequence)

                            # Weighted auxiliary loss
                            aux_term = float(self.aux_coef) * mse_loss

                            # Log both raw and weighted losses
                            if hasattr(self, 'logger'):
                                try:
                                    self.logger.record("train/mse_loss", float(mse_loss.item()))
                                    self.logger.record("train/mse_loss_weighted", float(aux_term.item()))
                                except Exception:
                                    # In some contexts logger.record may raise for non-scalar types
                                    pass

                            # Add the auxiliary loss to the PPO loss
                            loss = loss + aux_term
                        except Exception:
                            # If anything in the aux loss computation fails, skip it
                            if hasattr(self, 'logger'):
                                try:
                                    self.logger.record("train/mse_error", 1)
                                except Exception:
                                    pass

                # Optimization step
                # If knowledge distillation is configured on the outer model, compute KD loss
                # Only run KD when both student and teacher are transformer policies to avoid
                # accidentally distilling from/to baseline LSTM policies.
                if (
                    hasattr(self, "teacher")
                    and hasattr(self, "kd_params")
                    and self.teacher is not None
                    and isinstance(self.policy, TransformerActorCritic)
                    and isinstance(self.teacher.policy, TransformerActorCritic)
                ):
                    try:
                        # Evaluate teacher values for the same observations/actions (no grad)
                        with th.no_grad():
                            t_values, _, _ = self.teacher.policy.evaluate_actions(rollout_data.observations, rollout_data.actions)

                        # Try to obtain logits from teacher and student via get_action_logits
                        s_logits = None
                        t_logits = None
                        try:
                            if hasattr(self.policy, "get_action_logits"):
                                s_logits = self.policy.get_action_logits(rollout_data.observations)
                        except Exception:
                            s_logits = None
                        try:
                            if hasattr(self.teacher.policy, "get_action_logits"):
                                with th.no_grad():
                                    t_logits = self.teacher.policy.get_action_logits(rollout_data.observations)
                        except Exception:
                            t_logits = None

                        # Compute KL only if both logits available
                        if (s_logits is not None) and (t_logits is not None):
                            temperature = float(self.kd_params.get("temperature", 1.0))
                            alpha = float(self.kd_params.get("alpha", 0.5))
                            kl = F.kl_div(
                                F.log_softmax(s_logits / temperature, dim=-1),
                                F.softmax(t_logits / temperature, dim=-1).detach(),
                                reduction="batchmean",
                            )
                        else:
                            kl = th.tensor(0.0, device=loss.device)

                        # Value MSE (student values are `values` computed earlier)
                        value_mse = F.mse_loss(values, t_values.detach())
                        kd_loss = alpha * kl + (1.0 - alpha) * value_mse
                        kd_weight = float(self.kd_params.get("weight", 0.7))
                        if hasattr(self, 'logger'):
                            try:
                                self.logger.record("train/kd_kl", float(kl.item()))
                                self.logger.record("train/kd_value_mse", float(value_mse.item()))
                                self.logger.record("train/kd_loss", float(kd_loss.item()))
                            except Exception:
                                pass

                        loss = loss + kd_weight * kd_loss
                    except Exception:
                        # If KD fails, continue without it
                        pass
                else:
                    # Not performing KD for non-transformer policies; optionally log skip.
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.record("train/kd_skipped", 1)
                        except Exception:
                            pass

                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        #---------------------------------------------------------------------------------------
        # Above is code straight from SB3 github repo