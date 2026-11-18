"""DreamWaQ context-aided velocity estimator and beta-VAE decoder."""
import torch
from torch import nn
from robot_sim.modules.him_estimator import mlp


class WAQEstimator(nn.Module):
    def __init__(self, obs_dim, frames, hidden, latent_dim, activation):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = mlp(obs_dim * frames, hidden, 3 + 2 * latent_dim, activation)
        self.decoder = mlp(3 + latent_dim, list(reversed(hidden)), obs_dim, activation)

    def encode(self, history):
        out = self.encoder(history)
        velocity = out[:, :3]
        mean = out[:, 3:3 + self.latent_dim]
        logvar = out[:, 3 + self.latent_dim:].clamp(-10, 10)
        return velocity, mean, logvar

    def forward(self, history):
        velocity, mean, _ = self.encode(history)
        # Deterministic context in the policy path keeps PPO log-probability
        # recomputation and ONNX inference conditioned on the same latent.
        return velocity, mean

    def reconstruct(self, history):
        velocity, mean, logvar = self.encode(history)
        latent = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        prediction = self.decoder(torch.cat((velocity, latent), -1))
        return velocity, prediction, mean, logvar
