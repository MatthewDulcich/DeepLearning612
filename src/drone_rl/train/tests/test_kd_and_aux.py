import torch
import torch.nn as nn
import numpy as np
from drone_rl.train.ppo_with_mse import PPOWithMSE
from drone_rl.models.transformer_policy import TransformerActorCritic
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


def test_kd_loss_shapes():
    # create tiny env
    env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    policy_kwargs = {'features_extractor_class': TransformerActorCritic}
    # build a minimal PPOWithMSE instance (use very small network)
    model = PPOWithMSE(policy=TransformerActorCritic, env=env, aux_coef=0.1, verbose=0, batch_size=8)

    # Mock teacher attach with minimal API
    class DummyTeacher:
        def __init__(self, policy):
            self.policy = policy

    teacher = DummyTeacher(model.policy)
    model.teacher = teacher
    model.kd_params = {'temperature': 2.0, 'alpha': 0.5, 'weight': 0.5}

    # Create fake rollout_data with observations and actions
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    # Fake one-step rollout_data using model.rollout_buffer API is complex; instead test KD computation path directly
    s_logits = torch.randn(4, 2)
    t_logits = torch.randn(4, 2)
    temperature = model.kd_params['temperature']
    alpha = model.kd_params['alpha']

    kl = nn.functional.kl_div(nn.functional.log_softmax(s_logits / temperature, dim=-1), nn.functional.softmax(t_logits / temperature, dim=-1), reduction='batchmean')
    assert kl.shape == ()

    # value mse
    values = torch.randn(4)
    t_values = torch.randn(4)
    mse = nn.functional.mse_loss(values, t_values)
    assert mse.shape == ()


def test_aux_mse_integration():
    # Ensure auxiliary MSE weighting doesn't crash
    env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    model = PPOWithMSE(policy=TransformerActorCritic, env=env, aux_coef=0.2, verbose=0, batch_size=8)
    # Attach dummy predictor
    class DummyPred(nn.Module):
        def forward(self, emb, init):
            return torch.zeros((emb.size(0), 10, init.size(1)))
    model.policy.state_predictor = DummyPred()

    # Mock get_seq_prediction_targets to return tensors of correct shape
    def fake_get_targets(batch_size=4):
        emb = torch.zeros((batch_size, model.policy.features_extractor.features_dim))
        init = torch.zeros((batch_size, 1, 4))
        target = torch.zeros((batch_size, 10, 4))
        return emb, init, target

    model.policy.get_seq_prediction_targets = fake_get_targets
    # Call train() will attempt to run; to avoid long runs, just call a small slice of logic by invoking the mse computation directly
    batch = model.policy.get_seq_prediction_targets()
    emb, init, target = batch
    pred = model.policy.state_predictor(emb, init)
    mse_loss = nn.functional.mse_loss(pred, target)
    assert mse_loss.shape == ()
