"""Isaac Gym keyboard deployment from a PT checkpoint, optionally using ONNX.
W/S forward/reverse, A/D turn, SPACE stop, R reset. Keys are held, not toggled.
"""
import argparse
import time
import numpy as np
import isaacgym
from isaacgym import gymapi
import torch
from robot_sim.registry import make_config, make_env, make_model
from robot_sim.simulation.storage import History
from robot_sim.utils.helpers import set_seed


def main(keyboard=True):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--onnx')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=0)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--speed',type=float,default=0.5)
    parser.add_argument('--yaw-speed',type=float,default=0.6)
    parser.add_argument('--command-ramp-time',type=float,default=0.5)
    args=parser.parse_args()
    if args.headless and args.steps<=0: parser.error('--headless requires positive --steps')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=make_config(saved['task'],saved['task_config'])
    cfg.env.num_envs=1
    cfg.disable_training_randomization()
    # Retain the checkpoint's height scan layout even on a flat plane: the
    # environment still constructs its privileged observations during playback.
    cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False
    cfg.commands.curriculum=False;cfg.commands.heading_command=False;cfg.commands.resampling_time=1e9
    cfg.env.episode_length_s=1e9
    set_seed(1)
    env=make_env(saved['task'],cfg,args.device,args.headless)
    spec=saved['model_spec'];train=saved['training_config']
    model=make_model(spec).to(args.device).eval()
    model.load_state_dict(saved['model_state_dict'])
    session=None
    if args.onnx:
        import onnxruntime as ort
        session=ort.InferenceSession(args.onnx,providers=['CPUExecutionProvider'])
        expected={'obs':spec['obs_dim']}
        if spec['history_length']: expected['obs_history']=spec['obs_dim']*spec['history_length']
        if {i.name:i.shape[-1] for i in session.get_inputs()} != expected:
            raise ValueError('ONNX input signature differs from checkpoint')
    history=History(1,spec['history_length'],spec['obs_dim'],train['history_stride'],args.device)
    pressed=set();reset_pending=[False]
    def handler(event):
        if event.action=='reset' and event.value>0: reset_pending[0]=True
        elif event.value>0: pressed.add(event.action)
        else: pressed.discard(event.action)
    if keyboard and env.viewer:
        for key,name in [(gymapi.KEY_W,'forward'),(gymapi.KEY_S,'back'),(gymapi.KEY_A,'left'),
                         (gymapi.KEY_D,'right'),(gymapi.KEY_SPACE,'stop'),(gymapi.KEY_R,'reset')]:
            env.gym.subscribe_viewer_keyboard_event(env.viewer,key,name)
        env.keyboard_event_handler=handler
    command=np.array(saved['deployment']['initialization']['commands'],dtype=float)
    target=command.copy()
    if not keyboard: target[0]=args.speed
    try:
        obs,_=env.reset()
        env.set_external_commands(command)
        obs=env.obs_buf
        history.reset(obs)
        step=0
        while not args.steps or step<args.steps:
            if env.viewer and env.gym.query_viewer_has_closed(env.viewer): break
            start=time.perf_counter()
            # Poll before building observation; normal step rendering also polls.
            if env.viewer: env.render()
            if reset_pending[0]:
                obs,_=env.reset();command[:3]=0;target[:3]=0
                env.set_external_commands(command);obs=env.obs_buf
                history.reset(obs);reset_pending[0]=False
            if keyboard:
                target[0]=args.speed*(int('forward' in pressed)-int('back' in pressed))
                target[2]=args.yaw_speed*(int('left' in pressed)-int('right' in pressed))
                if 'stop' in pressed: target[:3]=0
            dt=cfg.sim.dt*cfg.control.decimation
            alpha=1.0 if args.command_ramp_time<=0 else min(1.,dt/args.command_ramp_time)
            # Bounded-rate interpolation over the configured transition duration.
            delta=target-command
            limit=np.array([max(abs(args.speed),1e-6),0,max(abs(args.yaw_speed),1e-6),0])*alpha
            command+=np.clip(delta,-limit,limit)
            env.set_external_commands(command);obs=env.obs_buf
            # Latest frame must reflect the command actually used for inference.
            if history.frames and history.age[0] % history.stride==0:
                history.data[:, -1]=obs
            with torch.no_grad():
                if session:
                    feed={'obs':obs.cpu().numpy()}
                    if history.frames: feed['obs_history']=history.flat.cpu().numpy()
                    action=torch.from_numpy(session.run(None,feed)[0]).to(args.device)
                else:
                    action=model.act_inference(obs,history.flat) if spec['history_length'] else model.act_inference(obs)
            obs,_,_,done,_,_=env.step(action)
            history.advance(obs,done)
            step+=1
            if step%50==0:
                print('step',step,'command',command.tolist(),'velocity',env.base_lin_vel[0].tolist(),flush=True)
            if env.viewer:
                position=env.root_states[0,:3].cpu().numpy()
                env.set_camera(position+np.array([2.,-3.,1.5]),position)
                time.sleep(max(0.,dt-(time.perf_counter()-start)))
        print('Isaac Gym playback complete:',step,'steps')
    finally: env.close()


if __name__=='__main__': main()
