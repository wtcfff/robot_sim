"""Factories: tasks and methods remain independent; imports are lazy."""
from robot_sim.config import training_config

_TASKS = {'wheelleg': ('robot_sim.simulation.envs.wheelleg.wheelleg', 'Wheelleg',
                      'robot_sim.simulation.envs.wheelleg.wheelleg_config', 'WheellegCfg')}


def register_task(name, env_class, config_class):
    if name in _TASKS:
        raise ValueError('Task already registered: ' + name)
    _TASKS[name] = (env_class, config_class)


def task_classes(name):
    import importlib
    entry = _TASKS[name]
    if len(entry) == 2:
        return entry
    module, cls, cfg_module, cfg_cls = entry
    return getattr(importlib.import_module(module), cls), getattr(importlib.import_module(cfg_module), cfg_cls)


def make_config(name, snapshot=None):
    from robot_sim.utils.helpers import update_config
    _, config_class = task_classes(name)
    cfg = config_class()
    if snapshot is not None:
        update_config(cfg, snapshot)
    return cfg


def make_env(name, cfg=None, device='cuda:0', headless=True):
    from isaacgym import gymapi
    from robot_sim.simulation.terrain import resolve_terrain
    env_class, _ = task_classes(name)
    cfg = cfg or make_config(name)
    resolve_terrain(cfg.terrain)
    params = gymapi.SimParams()
    params.dt = cfg.sim.dt
    params.substeps = cfg.sim.substeps
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(*cfg.sim.gravity)
    params.use_gpu_pipeline = device.startswith('cuda')
    for key, value in vars(cfg.sim.physx).items():
        if key == 'contact_collection':
            value = gymapi.ContactCollection(value)
        setattr(params.physx, key, value)
    params.physx.use_gpu = device.startswith('cuda')
    return env_class(cfg, params, gymapi.SIM_PHYSX, device, headless)


def make_model(spec):
    method = spec['method']
    if method == 'ppo':
        from robot_sim.modules.actor_critic import ActorCritic
        kwargs = {k: v for k, v in spec['network'].items() if k not in ('latent_dim','encoder_hidden_dims')}
        return ActorCritic(spec['obs_dim'], spec['obs_dim'], spec['action_dim'], **kwargs)
    from robot_sim.modules.him_actor_critic import HistoryActorCritic
    return HistoryActorCritic(spec)


def make_algorithm(model, config, device):
    if config['method'] == 'ppo':
        from robot_sim.algorithms.ppo import PPO
        cls = PPO
    elif config['method'] == 'him':
        from robot_sim.algorithms.him_ppo import HIMPPO
        cls = HIMPPO
    else:
        from robot_sim.algorithms.waq_ppo import WAQPPO
        cls = WAQPPO
    return cls(model, config, device)
