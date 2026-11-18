"""Actor-critic with a method-specific history estimator; no simulator imports."""
import torch
from robot_sim.modules.actor_critic import ActorCritic
from robot_sim.modules.him_estimator import HIMEstimator
from robot_sim.modules.waq_estimator import WAQEstimator


class HistoryActorCritic(ActorCritic):
    is_sequence = True

    def __init__(self, spec):
        cfg = spec['network']
        latent = cfg['latent_dim']
        common = {k: v for k,v in cfg.items() if k not in ('latent_dim', 'encoder_hidden_dims')}
        super().__init__(spec['obs_dim'] + 3 + latent, spec['critic_dim'], spec['action_dim'], **common)
        args = (spec['obs_dim'], spec['history_length'], cfg['encoder_hidden_dims'], latent)
        if spec['method'] == 'him':
            self.estimator = HIMEstimator(*args, spec['auxiliary']['prototypes'], cfg['activation'])
        else:
            self.estimator = WAQEstimator(*args, cfg['activation'])

    def policy_features(self, obs, history):
        velocity, latent = self.estimator(history)
        return torch.cat((obs, velocity.detach(), latent.detach()), dim=-1)

    def act(self, observations, history, **kwargs):
        return super().act(self.policy_features(observations, history))

    def act_inference(self, observations, history):
        return self.actor(self.policy_features(observations, history))
