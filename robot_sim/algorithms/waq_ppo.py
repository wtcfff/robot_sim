"""DreamWaQ CENet auxiliary optimization; explicit without-AdaBoot variant."""
import torch
from torch.nn import functional as F
from robot_sim.algorithms.him_ppo import HIMPPO


class WAQPPO(HIMPPO):
    def auxiliary_update(self, batch):
        valid = ~batch['dones'].bool()
        if not valid.any():
            return dict(velocity_loss=0.0, reconstruction_loss=0.0, latent_kl=0.0)
        velocity, prediction, mean, logvar = self.model.estimator.reconstruct(batch['history'][valid])
        cfg = self.config['auxiliary']
        velocity_loss = F.mse_loss(velocity, batch['velocity'][valid])
        reconstruction = F.mse_loss(prediction, batch['next_obs'][valid])
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(-1).mean()
        loss = cfg['velocity_coef'] * velocity_loss + cfg['reconstruction_coef'] * reconstruction + cfg['beta'] * kl
        self.aux_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.estimator.parameters(), 10.0)
        self.aux_optimizer.step()
        return dict(velocity_loss=velocity_loss.item(), reconstruction_loss=reconstruction.item(), latent_kl=kl.item())
