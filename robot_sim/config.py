"""Shared PPO defaults and method-specific settings; no task physics here."""
from copy import deepcopy

PPO_DEFAULTS = dict(num_learning_epochs=5, num_mini_batches=4, clip_param=0.2,
    gamma=0.99, lam=0.95, value_loss_coef=1.0, entropy_coef=0.01,
    learning_rate=1e-3, max_grad_norm=1.0, desired_kl=0.005, schedule='adaptive')


def training_config(method='ppo'):
    if method not in ('ppo', 'him', 'waq'):
        raise ValueError('method must be ppo, him, or waq')
    return dict(method=method, seed=1, steps_per_env=48, iterations=10000,
        save_interval=100, history_length=0 if method == 'ppo' else 5,
        history_stride=1, history_fill='repeat_initial_observation',
        network=dict(actor_hidden_dims=[128, 64, 32], critic_hidden_dims=[256, 128, 64],
                     activation='elu', init_noise_std=0.5, latent_dim=16,
                     encoder_hidden_dims=[128, 64]),
        algorithm=deepcopy(PPO_DEFAULTS),
        auxiliary=dict(learning_rate=1e-3, velocity_coef=1.0, swap_coef=1.0,
                       reconstruction_coef=1.0, beta=0.01, prototypes=32,
                       temperature=3.0),
        # Explicitly named variant: actor always uses predictions, no AdaBoot.
        waq_variant='without_adaboot')
