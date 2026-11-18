"""Owned rollout snapshots, episode-safe history, and mini-batch iteration."""
import torch


class History:
    def __init__(self, num_envs, frames, obs_dim, stride=1, device='cpu'):
        self.frames, self.obs_dim, self.stride = frames, obs_dim, stride
        self.data = torch.zeros(num_envs, frames, obs_dim, device=device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, obs, ids=None):
        if ids is None:
            self.data[:] = obs[:, None, :]
            self.age.zero_()
        else:
            self.data[ids] = obs[ids, None, :]
            self.age[ids] = 0

    def advance(self, obs, dones):
        self.age += 1
        ids = ((self.age % self.stride == 0) & ~dones.bool()).nonzero().flatten()
        if self.frames and ids.numel():
            self.data[ids] = torch.cat((self.data[ids, 1:], obs[ids, None]), 1)
        self.reset(obs, dones.nonzero().flatten())

    @property
    def flat(self):
        return self.data.flatten(1)


class RolloutStorage:
    def __init__(self):
        self.rows = []

    def add(self, **data):
        self.rows.append({k: v.detach().clone() for k,v in data.items()})

    def compute_returns(self, last_values, gamma, lam):
        self.data = {k: torch.stack([row[k] for row in self.rows]) for k in self.rows[0]}
        values, rewards, dones = (self.data[k] for k in ('values','rewards','dones'))
        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(last_values)
        # rewards already include timeout value bootstrapping. Recursion stops
        # at every episode boundary, including truncations.
        for step in reversed(range(len(self.rows))):
            next_value = last_values if step == len(self.rows)-1 else values[step+1]
            mask = (~dones[step].bool()).float()
            delta = rewards[step] + gamma * next_value * mask - values[step]
            gae = delta + gamma * lam * mask * gae
            advantages[step] = gae
        self.data['returns'] = advantages + values
        self.data['advantages'] = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.data = {k: v.flatten(0,1) for k,v in self.data.items()}

    def batches(self, epochs, count):
        size = self.data['actions'].shape[0]
        if count > size:
            raise ValueError('num_mini_batches exceeds rollout size')
        for _ in range(epochs):
            for ids in torch.randperm(size, device=self.data['actions'].device).tensor_split(count):
                yield {k: v[ids] for k,v in self.data.items()}

    def clear(self):
        self.rows.clear()
        self.data = {}
