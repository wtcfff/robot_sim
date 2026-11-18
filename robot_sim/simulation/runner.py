"""Shared on-policy interaction, method selection, logging and PT checkpoints."""
import json
import time
from pathlib import Path
import torch
from robot_sim.registry import make_model, make_algorithm
from robot_sim.simulation.storage import History
from robot_sim.utils.helpers import class_to_dict


class Runner:
    def __init__(self, env, task, config, log_dir, device='cuda:0'):
        self.env, self.task, self.config, self.device = env, task, config, device
        if config['method'] != 'ppo' and not hasattr(env, 'base_lin_vel'):
            raise ValueError('HIM/WAQ require task-provided body velocity supervision')
        critic_dim = env.num_obs if config['method'] == 'ppo' else env.num_privileged_obs
        if critic_dim is None:
            raise ValueError('History methods require task-provided privileged critic input')
        self.spec = dict(method=config['method'], obs_dim=env.num_obs, action_dim=env.num_actions,
            critic_dim=critic_dim, history_length=config['history_length'],
            network=config['network'], auxiliary=config['auxiliary'])
        self.model = make_model(self.spec).to(device)
        self.algorithm = make_algorithm(self.model, config, device)
        self.history = History(env.num_envs, config['history_length'], env.num_obs,
                               config['history_stride'], device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.iteration, self.total_steps = 0, 0
        self.task_config = class_to_dict(env.cfg)
        self.deployment = env.deployment_config() if hasattr(env, 'deployment_config') else None
        (self.log_dir/'config.json').write_text(json.dumps(dict(task=task, task_config=self.task_config, training=config, model=self.spec), indent=2), encoding='utf-8')

    def critic(self, obs, privileged):
        # Plain PPO cannot accidentally receive simulator truth.
        return obs if self.config['method'] == 'ppo' else privileged

    def save(self):
        torch.save(dict(format_version=1, task=self.task, model_spec=self.spec,
            model_state_dict=self.model.state_dict(), optimizer=self.algorithm.optimizer.state_dict(),
            auxiliary_optimizer=self.algorithm.aux_optimizer.state_dict() if self.algorithm.aux_optimizer else None,
            training_config=self.config, task_config=self.task_config,
            deployment=self.deployment, iteration=self.iteration, total_steps=self.total_steps),
            self.log_dir/('model_%d.pt' % self.iteration))

    def load(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved['model_spec'] != self.spec or saved['task_config'] != self.task_config:
            raise ValueError('Resume config differs from checkpoint; use its saved task/method settings')
        self.model.load_state_dict(saved['model_state_dict'])
        self.algorithm.optimizer.load_state_dict(saved['optimizer'])
        if self.algorithm.aux_optimizer and saved['auxiliary_optimizer']:
            self.algorithm.aux_optimizer.load_state_dict(saved['auxiliary_optimizer'])
        self.iteration, self.total_steps = saved['iteration'], saved['total_steps']

    def learn(self, iterations):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(self.log_dir))
        obs, privileged = self.env.reset()
        self.history.reset(obs)
        start = time.time()
        try:
            for _ in range(iterations):
                reward_sum = 0.0
                for _ in range(self.config['steps_per_env']):
                    with torch.no_grad():
                        history = self.history.flat.clone()
                        old_obs = obs.clone()
                        critic = self.critic(obs, privileged).clone()
                        velocity = self.env.base_lin_vel.clone() if self.config['method'] != 'ppo' else torch.zeros(obs.shape[0],3,device=self.device)
                        self.algorithm.distribution(obs, history)
                        actions = self.model.distribution.sample()
                        logprob = self.model.get_actions_log_prob(actions)
                        values = self.model.evaluate(critic).squeeze(-1)
                        mu, sigma = self.model.action_mean.clone(), self.model.action_std.clone()
                        obs, privileged, rewards, dones, infos, _ = self.env.step(actions)
                        rewards = rewards.clone()
                        truncated = infos['truncated']
                        final_critic = self.critic(infos['final_obs'], infos['final_privileged'])
                        final_values = self.model.evaluate(final_critic).squeeze(-1)
                        rewards += self.config['algorithm']['gamma'] * truncated.float() * final_values
                        self.algorithm.storage.add(obs=old_obs, history=history, critic=critic,
                            actions=actions, logprob=logprob, values=values, mu=mu, sigma=sigma,
                            rewards=rewards, dones=dones.bool(), velocity=velocity, next_obs=infos['final_obs'])
                        self.history.advance(obs, dones)
                        reward_sum += self.env.rew_buf.mean().item()
                with torch.no_grad():
                    last_values = self.model.evaluate(self.critic(obs, privileged)).squeeze(-1)
                    cfg = self.config['algorithm']
                    self.algorithm.storage.compute_returns(last_values, cfg['gamma'], cfg['lam'])
                metrics = self.algorithm.update()
                self.iteration += 1
                self.total_steps += self.env.num_envs * self.config['steps_per_env']
                metrics['reward_per_step'] = reward_sum / self.config['steps_per_env']
                for key,value in metrics.items():
                    writer.add_scalar('train/'+key, value, self.total_steps)
                print('iteration=%d steps=%d elapsed=%.1fs %s' % (self.iteration,self.total_steps,time.time()-start,json.dumps(metrics)), flush=True)
                with (self.log_dir/'metrics.jsonl').open('a') as out:
                    out.write(json.dumps(dict(iteration=self.iteration, steps=self.total_steps, **metrics))+'\n')
                if self.iteration % self.config['save_interval'] == 0:
                    self.save()
            self.save()
        finally:
            writer.close()
