"""HIM velocity and prototype representations (HIMLoco/SwAV formulation)."""
import torch
from torch import nn
from torch.nn import functional as F
from robot_sim.modules.actor_critic import get_activation


def mlp(input_dim, hidden, output_dim, activation='elu'):
    layers = []
    for width in hidden:
        layers += [nn.Linear(input_dim, width), get_activation(activation)]
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class HIMEstimator(nn.Module):
    def __init__(self, obs_dim, frames, hidden, latent_dim, prototypes, activation):
        super().__init__()
        self.encoder = mlp(obs_dim * frames, hidden, 3 + latent_dim, activation)
        self.target = mlp(obs_dim, hidden, latent_dim, activation)
        self.prototypes = nn.Parameter(torch.randn(prototypes, latent_dim))

    def forward(self, history):
        encoded = self.encoder(history)
        return encoded[:, :3], F.normalize(encoded[:, 3:], dim=-1)

    def pair_scores(self, history, next_obs):
        velocity, source = self(history)
        target = F.normalize(self.target(next_obs), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return velocity, source @ prototypes.T, target @ prototypes.T
