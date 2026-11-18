"""Train an independent task with ppo, him, or waq; PT checkpoints are retained."""
import argparse
from datetime import datetime
from pathlib import Path
import isaacgym  # Import before torch.
import torch
from robot_sim.registry import make_env, make_config
from robot_sim.config import training_config
from robot_sim.simulation.runner import Runner
from robot_sim.utils.helpers import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='wheelleg')
    parser.add_argument('--method', choices=['ppo','him','waq'], default='ppo')
    parser.add_argument('--num-envs', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--steps-per-env', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--terrain-ids', type=int, nargs='+')
    parser.add_argument('--log-dir')
    parser.add_argument('--resume')
    args = parser.parse_args()
    saved = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if saved:
        args.task, args.method = saved['task'], saved['model_spec']['method']
    cfg = make_config(args.task, saved['task_config'] if saved else None)
    train_cfg = saved['training_config'] if saved else training_config(args.method)
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    if args.terrain_ids is not None:
        cfg.terrain.selected_ids = args.terrain_ids
        cfg.terrain.weights = None
    if args.steps_per_env is not None:
        train_cfg['steps_per_env'] = args.steps_per_env
    if not saved:
        train_cfg['seed'] = args.seed
    set_seed(train_cfg['seed'])
    log_dir = args.log_dir or str(Path('logs')/args.task/args.method/datetime.now().strftime('%Y%m%d_%H%M%S'))
    env = make_env(args.task, cfg, args.device, args.headless)
    try:
        runner = Runner(env, args.task, train_cfg, log_dir, args.device)
        if saved:
            runner.load(args.resume)
        runner.learn(args.iterations if args.iterations is not None else train_cfg['iterations'])
    finally:
        env.close()


if __name__ == '__main__':
    main()
