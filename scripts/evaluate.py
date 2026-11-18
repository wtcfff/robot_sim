"""Fixed-seed Isaac Gym benchmark; task supplies commands and metric definitions."""
import argparse
import json
from pathlib import Path
import isaacgym
import torch
from robot_sim.registry import make_config,make_env,make_model
from robot_sim.simulation.storage import History
from robot_sim.utils.helpers import set_seed
from robot_sim.utils.helpers import class_to_dict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--num-envs',type=int,default=16)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output')
    args=parser.parse_args()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=make_config(saved['task'],saved['task_config'])
    cfg.env.num_envs=args.num_envs
    cfg.disable_training_randomization()
    cfg.commands.curriculum=False;cfg.commands.heading_command=False;cfg.commands.resampling_time=1e9
    # Build the same fixed terrain bank for all methods, then stop adaptation.
    set_seed(args.seed)
    env=make_env(saved['task'],cfg,args.device,True)
    env.cfg.terrain.curriculum=False
    model=make_model(saved['model_spec']).to(args.device).eval()
    model.load_state_dict(saved['model_state_dict'])
    spec=saved['model_spec'];train=saved['training_config']
    history=History(args.num_envs,spec['history_length'],spec['obs_dim'],train['history_stride'],args.device)
    metrics={};episodes=0;failures=0
    try:
        if not hasattr(env,'evaluation_commands') or not hasattr(env,'evaluation_metrics'):
            raise ValueError('Task must supply evaluation_commands and evaluation_metrics')
        obs,_=env.reset()
        env.set_external_commands(env.evaluation_commands(0.0));obs=env.obs_buf
        history.reset(obs)
        with torch.no_grad():
            for step in range(args.steps):
                env.set_external_commands(env.evaluation_commands(step*env.dt));obs=env.obs_buf
                if history.frames:
                    ids=(history.age%history.stride==0).nonzero().flatten()
                    history.data[ids,-1]=obs[ids]
                action=model.act_inference(obs,history.flat) if history.frames else model.act_inference(obs)
                obs,_,reward,done,info,_=env.step(action)
                # Capture metrics before reset inside env; do not measure new
                # episode state against the previous episode's action.
                values=info['evaluation_metrics']
                for key,value in values.items(): metrics[key]=metrics.get(key,0.)+value.mean().item()
                metrics['reward']=metrics.get('reward',0.)+reward.mean().item()
                episodes+=int(done.sum());failures+=int(info['terminated'].sum())
                history.advance(obs,done)
        result=dict(task=saved['task'],method=spec['method'],seed=args.seed,
            steps=args.steps,num_envs=args.num_envs,ended_episodes=episodes,failures=failures,
            metrics={k:v/args.steps for k,v in metrics.items()},
            training_task_config=saved['task_config'], evaluation_task_config=class_to_dict(cfg))
        output=Path(args.output) if args.output else Path(args.checkpoint).with_suffix('.evaluation.json')
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if not k.endswith('task_config')},indent=2))
        print('Saved',output)
    finally: env.close()


if __name__=='__main__': main()
