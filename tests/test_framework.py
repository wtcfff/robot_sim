"""Configuration isolation, environment boundaries and cross-backend contracts."""
import ast
import copy
from pathlib import Path
import unittest
import isaacgym  # Must precede torch and all Isaac Gym task imports.
import numpy as np
import torch
from robot_sim import ROOT_DIR
from robot_sim.registry import make_config,make_env,make_model,make_algorithm
from robot_sim.config import training_config
from robot_sim.simulation.storage import History,RolloutStorage
from robot_sim.simulation.terrain import Terrain,resolve_terrain
from robot_sim.simulation.envs.base.base_config import BaseEnvCfg
from robot_sim.utils.helpers import class_to_dict
from scripts.sim2sim import observation,control


class ConfigurationTests(unittest.TestCase):
    def test_mutable_defaults_are_isolated(self):
        a,b=make_config('wheelleg'),make_config('wheelleg')
        snapshot=class_to_dict(b)
        a.terrain.selected_ids.append(8)
        a.control.stiffness['hip_Joint']=1
        a.commands.ranges.lin_vel_x[0]=-100
        self.assertEqual(class_to_dict(b),snapshot)
        self.assertIsNone(BaseEnvCfg().terrain.selected_ids)

    def test_ids_and_all_terrain_geometry(self):
        cfg=make_config('wheelleg').terrain
        cfg.selected_ids=None;cfg.num_rows=2;cfg.num_cols=9;cfg.border_size=0;cfg.mesh_type='heightfield'
        terrain=Terrain(cfg,0)
        self.assertEqual(cfg.resolved_ids,list(range(9)))
        self.assertEqual(terrain.column_ids.tolist(),list(range(9)))
        self.assertEqual(terrain.height_field_raw.shape,(160,720))
        for col in range(1,9):
            self.assertTrue(np.any(terrain.height_field_raw[80:160,col*80:(col+1)*80]),str(col))
        for invalid in ([],[1,1],[-1],[9],[True]):
            bad=copy.deepcopy(cfg);bad.selected_ids=invalid
            with self.assertRaises(ValueError): resolve_terrain(bad)

    def test_reference_configuration_and_reward_code_preserved(self):
        reference=Path('/home/xsl/桌面/Wheel-Legged-Gym-master/wheel_legged_gym')
        if not reference.is_dir(): self.skipTest('Reference checkout is optional outside migration audit')
        from wheel_legged_gym.envs.wheelleg.wheelleg_config import WheellegCfg as Original
        old,new=class_to_dict(Original()),class_to_dict(make_config('wheelleg'))
        for group in ('domain_rand','noise','commands','control','init_state','rewards','normalization','sim'):
            self.assertEqual(old[group],new[group],group)
        for relative in ('base/legged_robot.py','wheelleg/wheelleg.py'):
            def reward_nodes(path):
                tree=ast.parse(path.read_text())
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                        if isinstance(n,ast.FunctionDef) and (n.name.startswith('_reward_') or n.name=='compute_reward')}
            self.assertEqual(reward_nodes(reference/'envs'/relative),
                             reward_nodes(Path(ROOT_DIR)/'robot_sim/simulation/envs'/relative))

    def test_method_information_access_dimensions(self):
        for method in ('ppo','him','waq'):
            cfg=training_config(method)
            spec=dict(method=method,obs_dim=32,action_dim=8,critic_dim=32 if method=='ppo' else 259,
                      history_length=cfg['history_length'],network=cfg['network'],auxiliary=cfg['auxiliary'])
            net=make_model(spec)
            self.assertEqual(net.actor[0].in_features,32 if method=='ppo' else 51)
            self.assertEqual(net.critic[0].in_features,32 if method=='ppo' else 259)


class DataTests(unittest.TestCase):
    def test_auxiliary_updates_and_policy_determinism(self):
        for method in ('him','waq'):
            torch.manual_seed(10)
            cfg=training_config(method)
            spec=dict(method=method,obs_dim=32,action_dim=8,critic_dim=259,
                      history_length=5,network=cfg['network'],auxiliary=cfg['auxiliary'])
            net=make_model(spec)
            algorithm=make_algorithm(net,cfg,'cpu')
            batch=dict(history=torch.randn(64,160),next_obs=torch.randn(64,32),
                       velocity=torch.randn(64,3),dones=torch.zeros(64,dtype=torch.bool))
            before=[p.detach().clone() for p in net.estimator.parameters()]
            metrics=algorithm.auxiliary_update(batch)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,net.estimator.parameters())))
            if method=='him':
                # Cross entropy sums over prototypes, rather than shrinking
                # its weight as the number of prototypes increases.
                self.assertGreater(metrics['swap_loss'],1.0)
            obs=torch.randn(64,32)
            torch.testing.assert_close(net.act_inference(obs,batch['history']),
                                       net.act_inference(obs,batch['history']))

    def test_history_reset_and_sampling_stride(self):
        history=History(2,3,2,stride=2)
        initial=torch.tensor([[1.,2.],[3.,4.]])
        history.reset(initial)
        history.advance(initial+1,torch.tensor([False,False]))
        torch.testing.assert_close(history.data[:,0],initial)
        history.advance(initial+2,torch.tensor([True,False]))
        torch.testing.assert_close(history.data[0],(initial[0]+2).expand(3,2))
        torch.testing.assert_close(history.data[1,-1],initial[1]+2)
        self.assertEqual(history.age.tolist(),[0,2])

    def test_returns_stop_at_episode_boundaries(self):
        storage=RolloutStorage()
        storage.add(values=torch.tensor([2.]),rewards=torch.tensor([5.]),dones=torch.tensor([True]),actions=torch.zeros(1,1))
        storage.add(values=torch.tensor([100.]),rewards=torch.tensor([1.]),dones=torch.tensor([True]),actions=torch.zeros(1,1))
        storage.compute_returns(torch.tensor([200.]),gamma=.99,lam=.95)
        torch.testing.assert_close(storage.data['returns'],torch.tensor([5.,1.]))


class SimulatorTests(unittest.TestCase):
    def test_gpu_reset_timeout_and_deployment_alignment(self):
        cfg=make_config('wheelleg');cfg.env.num_envs=4;cfg.env.episode_length_s=.01
        cfg.terrain.num_rows=2;cfg.terrain.num_cols=4;cfg.terrain.max_init_terrain_level=1;cfg.terrain.border_size=2
        cfg.disable_training_randomization()
        cfg.commands.curriculum=False;cfg.commands.heading_command=False
        env=make_env('wheelleg',cfg,headless=True)
        try:
            obs,priv=env.reset()
            self.assertEqual(tuple(obs.shape),(4,32));self.assertEqual(tuple(priv.shape),(4,259))
            deploy=env.deployment_config()
            env.set_external_commands([.4,0.,-.2,.36])
            current=env.compute_proprioception_observations()[0].cpu().numpy()
            mujoco_obs=observation(deploy,env.base_ang_vel[0].cpu().numpy(),env.projected_gravity[0].cpu().numpy(),
                env.dof_pos[0].cpu().numpy(),env.dof_vel[0].cpu().numpy(),env.actions[0].cpu().numpy(),np.array([.4,0.,-.2,.36]))
            np.testing.assert_allclose(current,mujoco_obs,atol=1e-6)
            action=torch.tensor([[.1,.2,.3,.4,-.1,-.2,-.3,-.4]],device=env.device).repeat(4,1)
            expected=env._compute_torques(action)[0].cpu().numpy()
            actual=control(deploy,env.dof_pos[0].cpu().numpy(),env.dof_vel[0].cpu().numpy(),action[0].cpu().numpy())
            np.testing.assert_allclose(expected,actual,atol=1e-5)
            history=History(4,5,32,device=env.device);history.reset(env.obs_buf)
            saw_timeout=False
            for _ in range(3):
                obs,priv,reward,done,info,_=env.step(action)
                history.advance(obs,done)
                self.assertTrue(torch.isfinite(priv).all())
                if done.any():
                    saw_timeout=True
                    self.assertTrue(info['truncated'][done.bool()].all())
                    self.assertFalse(info['terminated'].any())
                    self.assertFalse(torch.equal(info['final_obs'],obs))
                    for i in done.nonzero().flatten():
                        torch.testing.assert_close(history.data[i],obs[i].expand(5,32))
            self.assertTrue(saw_timeout)
        finally: env.close()


if __name__=='__main__': unittest.main()
