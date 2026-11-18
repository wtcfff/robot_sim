# PPO clipping, value clipping and adaptive KL schedule adapted from rsl_rl.
# Copyright 2021 ETH Zurich, NVIDIA CORPORATION; SPDX-License-Identifier: BSD-3-Clause
import torch
from torch import nn
from robot_sim.simulation.storage import RolloutStorage


class PPO:
    def __init__(self, model, config, device):
        self.model, self.config, self.device = model, config, device
        self.cfg = config['algorithm']
        params = list(model.actor.parameters()) + list(model.critic.parameters()) + [model.std]
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=self.cfg['learning_rate'])
        self.storage = RolloutStorage()
        self.aux_optimizer = None

    def distribution(self, obs, history):
        if self.model.is_sequence:
            self.model.act(obs, history)
        else:
            self.model.act(obs)

    def auxiliary_update(self, batch):
        return {}

    def update(self):
        cfg = self.cfg
        metrics, count = {}, 0
        for batch in self.storage.batches(cfg['num_learning_epochs'], cfg['num_mini_batches']):
            self.distribution(batch['obs'], batch['history'])
            logprob = self.model.get_actions_log_prob(batch['actions'])
            values = self.model.evaluate(batch['critic']).squeeze(-1)
            ratio = (logprob - batch['logprob']).exp()
            advantage = batch['advantages']
            policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1-cfg['clip_param'],1+cfg['clip_param']) * advantage).mean()
            clipped = batch['values'] + (values - batch['values']).clamp(-cfg['clip_param'], cfg['clip_param'])
            value_loss = torch.maximum((values-batch['returns']).square(), (clipped-batch['returns']).square()).mean()
            entropy = self.model.entropy.mean()
            with torch.no_grad():
                mu, sigma = self.model.action_mean, self.model.action_std
                kl = (torch.log(sigma / batch['sigma']) + (batch['sigma'].square() + (batch['mu']-mu).square())/(2*sigma.square()) - 0.5).sum(-1).mean()
                if cfg['schedule'] == 'adaptive' and cfg['desired_kl'] is not None:
                    lr = self.optimizer.param_groups[0]['lr']
                    if kl > cfg['desired_kl'] * 2:
                        lr = max(1e-5, lr / 1.5)
                    elif 0 < kl < cfg['desired_kl'] / 2:
                        lr = min(1e-2, lr * 1.5)
                    for group in self.optimizer.param_groups:
                        group['lr'] = lr
            self.optimizer.zero_grad()
            loss = policy_loss + cfg['value_loss_coef'] * value_loss - cfg['entropy_coef'] * entropy
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, cfg['max_grad_norm'])
            self.optimizer.step()
            with torch.no_grad():
                self.model.std.clamp_(min=1e-4)
            result = dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(), entropy=entropy.item(), kl=kl.item())
            result.update(self.auxiliary_update(batch))
            if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
                raise FloatingPointError('Non-finite training metric: ' + str(result))
            for key,value in result.items():
                metrics[key] = metrics.get(key, 0.0) + value
            count += 1
        self.storage.clear()
        return {k: v / count for k,v in metrics.items()}
