"""HIM auxiliary optimization, kept separate from simulator and network code."""
import torch
from torch.nn import functional as F
from robot_sim.algorithms.ppo import PPO


@torch.no_grad()
def assignments(scores, eps=0.05, iterations=3):
    # Center before exponentiation to avoid overflow at low temperatures.
    logits = scores / eps
    q = (logits - logits.max()).exp().T.clamp_min(1e-12)
    prototypes, batch = q.shape
    q /= q.sum()
    for _ in range(iterations):
        q /= q.sum(1, keepdim=True).clamp_min(1e-12)
        q /= prototypes
        q /= q.sum(0, keepdim=True).clamp_min(1e-12)
        q /= batch
    return (q * batch).T


class HIMPPO(PPO):
    def __init__(self, model, config, device):
        super().__init__(model, config, device)
        self.aux_optimizer = torch.optim.Adam(model.estimator.parameters(), lr=config['auxiliary']['learning_rate'])

    def auxiliary_update(self, batch):
        # Exclude transitions ending an episode from temporal pairing.
        valid = ~batch['dones'].bool()
        if valid.sum() < 2:
            return dict(velocity_loss=0.0, swap_loss=0.0)
        velocity, source, target = self.model.estimator.pair_scores(batch['history'][valid], batch['next_obs'][valid])
        cfg = self.config['auxiliary']
        a, b = assignments(source), assignments(target)
        swap = -0.5 * (a * F.log_softmax(target/cfg['temperature'], -1) + b * F.log_softmax(source/cfg['temperature'], -1)).sum(-1).mean()
        velocity_loss = F.mse_loss(velocity, batch['velocity'][valid])
        self.aux_optimizer.zero_grad()
        (cfg['velocity_coef'] * velocity_loss + cfg['swap_coef'] * swap).backward()
        torch.nn.utils.clip_grad_norm_(self.model.estimator.parameters(), 10.0)
        self.aux_optimizer.step()
        return dict(velocity_loss=velocity_loss.item(), swap_loss=swap.item())
