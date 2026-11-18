"""Run ONNX in MuJoCo using an explicit generated migration YAML.

W/S set forward/reverse, A/D set yaw, SPACE stops, R resets, ESC exits.
Uses only the ONNX and YAML, never training task imports or PT weights.
"""
import argparse
from pathlib import Path
import time
import numpy as np
import yaml
import mujoco
import onnxruntime as ort
from robot_sim import ROOT_DIR


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    h = cfg['history']
    if h['order'] != 'oldest_to_newest' or not h['include_current_observation']:
        raise ValueError('Unsupported history ordering/timing')
    if h['startup_fill'] != 'repeat_initial_observation' or h['reset_fill'] != 'repeat_initial_observation':
        raise ValueError('Unsupported history fill rule')
    if h['frame_dim'] != cfg['obs_dim'] or h['input_dim'] != h['frames'] * cfg['obs_dim']:
        raise ValueError('Invalid history dimensions')
    if h['enabled'] != bool(h['frames']) or h['sample_interval_policy_steps'] < 1:
        raise ValueError('Invalid history enable/stride')
    if len(cfg['joints']) != cfg['action_dim'] or sum(b['size'] for b in cfg['observations']) != cfg['obs_dim']:
        raise ValueError('Action/observation layout dimension mismatch')
    if cfg['initialization']['settling_policy_steps'] != 0:
        raise ValueError('This runtime requires explicit zero settling steps')
    cfg['_directory'] = str(path.parent)
    return cfg


def rotate_inverse(quat_wxyz, vector):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat_wxyz, dtype=np.float64))
    return matrix.reshape(3,3).T @ vector


def observation(cfg, angular_velocity, gravity, q, dq, previous_action, command):
    default = np.asarray([j['default_position'] for j in cfg['joints']])
    sources = dict(angular_velocity=angular_velocity, gravity=gravity,
                   joint_position_offset=q-default, joint_velocity=dq,
                   previous_action=previous_action, commands=command)
    blocks=[]
    for block in cfg['observations']:
        if block['source'] == 'constant':
            value = np.asarray(block['value'])
        else:
            value = sources[block['source']]
        if 'indices' in block:
            value = value[block['indices']]
        value = np.asarray(value) * np.asarray(block['scale'])
        if value.shape != (block['size'],):
            raise ValueError('Observation block shape mismatch: '+block['source'])
        blocks.append(value)
    return np.clip(np.concatenate(blocks), -cfg['clip_observations'], cfg['clip_observations']).astype(np.float32)


def control(cfg, q, dq, actions):
    actions = np.clip(actions,-cfg['clip_actions'],cfg['clip_actions'])
    torques=[]
    for i,joint in enumerate(cfg['joints']):
        target = actions[i]*joint['action_scale']
        if joint['target_limit'] is not None:
            target = np.clip(target,-joint['target_limit'],joint['target_limit'])
        if joint['mode'] == 'position':
            torque = joint['kp']*(target+joint['default_position']-q[i])-joint['kd']*dq[i]
        elif joint['mode'] == 'velocity':
            torque = joint['kp']*(target-dq[i])
        elif joint['mode'] == 'torque':
            torque = target
        else:
            raise ValueError('Unknown control mode: '+joint['mode'])
        torques.append(np.clip(torque,-joint['effort_limit'],joint['effort_limit']))
    return np.asarray(torques)


class Simulation:
    def __init__(self, cfg, policy=None, xml=None):
        self.cfg=cfg
        xml=Path(xml or cfg['target']['xml'])
        if not xml.is_absolute():
            xml=Path(ROOT_DIR)/xml
        self.model=mujoco.MjModel.from_xml_path(str(xml))
        self.data=mujoco.MjData(self.model)
        self.model.opt.timestep=cfg['simulation_dt']
        self.session=ort.InferenceSession(str(policy or Path(cfg['_directory'])/cfg['policy']),providers=['CPUExecutionProvider'])
        expected={'obs':cfg['obs_dim']}
        if cfg['history']['enabled']:
            expected['obs_history']=cfg['history']['input_dim']
        if set(i.name for i in self.session.get_inputs()) != set(expected):
            raise ValueError('ONNX inputs do not match configured history')
        for item in self.session.get_inputs():
            if item.shape[-1] != expected[item.name]:
                raise ValueError('ONNX input dimension mismatch: '+item.name)
        if self.session.get_outputs()[0].shape[-1] != cfg['action_dim']:
            raise ValueError('ONNX action dimension mismatch')
        mapping=cfg['target']['joint_mapping']
        ids=[]; actuators=[];sign=[];offset=[]
        for joint in cfg['joints']:
            entry=mapping[joint['name']]
            ids.append(self.id(mujoco.mjtObj.mjOBJ_JOINT,entry['joint']))
            actuators.append(self.id(mujoco.mjtObj.mjOBJ_ACTUATOR,entry['actuator']))
            sign.append(entry['sign']); offset.append(entry['offset'])
        self.qids=self.model.jnt_qposadr[ids]
        self.vids=self.model.jnt_dofadr[ids]
        self.actuators=np.asarray(actuators)
        self.sign=np.asarray(sign); self.offset=np.asarray(offset)
        if not np.all(np.isin(self.sign,[-1.0,1.0])):
            raise ValueError('Joint sign must be +1 or -1')
        # Runtime writes joint torque directly; actuator gear must be unity.
        if not np.allclose(self.model.actuator_gear[self.actuators,0],1.0):
            raise ValueError('Direct torque runtime requires unity actuator gear')
        self.base=self.id(mujoco.mjtObj.mjOBJ_BODY,cfg['target']['base_body'])
        self.reset()

    def id(self, kind, name):
        result=mujoco.mj_name2id(self.model,kind,name)
        if result < 0:
            raise ValueError('MuJoCo object missing: '+name)
        return result

    def state(self):
        q=self.data.qpos[self.qids]*self.sign+self.offset
        dq=self.measured_velocity if self.cfg['velocity_source']=='wrapped_position_difference' else self.data.qvel[self.vids]*self.sign
        velocity=np.zeros(6)
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.base,velocity,0)
        quat=self.data.xquat[self.base]
        return q,dq,rotate_inverse(quat,velocity[:3]),rotate_inverse(quat,np.array([0.,0.,-1.]))

    def obs(self):
        q,dq,gyro,gravity=self.state()
        return observation(self.cfg,gyro,gravity,q,dq,self.previous_action,self.command)

    def reset(self):
        mujoco.mj_resetData(self.model,self.data)
        init=self.cfg['initialization']
        base_joint=self.model.body_jntadr[self.base]
        if base_joint >= 0 and self.model.jnt_type[base_joint] == mujoco.mjtJoint.mjJNT_FREE:
            addr=self.model.jnt_qposadr[base_joint]
            self.data.qpos[addr:addr+3]=init['base_position']
            x,y,z,w=init['base_quaternion_xyzw']
            self.data.qpos[addr+3:addr+7]=[w,x,y,z]
        self.data.qpos[self.qids]=(np.array([j['default_position'] for j in self.cfg['joints']])-self.offset)/self.sign
        self.data.qvel[:]=0
        mujoco.mj_forward(self.model,self.data)
        self.command=np.array(init['commands'],dtype=float)
        self.previous_action=np.zeros(self.cfg['action_dim'])
        self.measured_velocity=np.zeros(self.cfg['action_dim'])
        self.previous_q=self.data.qpos[self.qids]*self.sign+self.offset
        self.age=0
        self.history=np.tile(self.obs(),(self.cfg['history']['frames'],1))

    def step(self):
        obs=self.obs()
        h=self.cfg['history']
        if h['enabled'] and self.age > 0 and self.age % h['sample_interval_policy_steps']==0:
            self.history=np.concatenate((self.history[1:],obs[None]),axis=0)
        elif h['enabled'] and self.age == 0:
            self.history[-1]=obs
        # Initial row is already present exactly once after reset.
        feeds={'obs':obs[None]}
        if h['enabled']:
            feeds['obs_history']=self.history.reshape(1,-1).astype(np.float32)
        action=self.session.run(None,feeds)[0][0]
        if not np.isfinite(action).all():
            raise FloatingPointError('Non-finite ONNX action')
        for _ in range(self.cfg['control_decimation']):
            q,dq,_,_=self.state()
            self.data.ctrl[self.actuators]=control(self.cfg,q,dq,action)*self.sign
            mujoco.mj_step(self.model,self.data)
            current=self.data.qpos[self.qids]*self.sign+self.offset
            delta=(current-self.previous_q+np.pi)%(2*np.pi)-np.pi
            self.measured_velocity=delta/self.cfg['simulation_dt']
            self.previous_q=current.copy()
        self.previous_action=np.clip(action,-self.cfg['clip_actions'],self.cfg['clip_actions'])
        self.age+=1
        return action


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--policy')
    parser.add_argument('--xml')
    parser.add_argument('--steps',type=int,default=0,help='0 runs until viewer closed')
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args()
    if args.headless and args.steps <= 0:
        parser.error('--headless requires positive --steps')
    cfg=load_config(args.config)
    sim=Simulation(cfg,args.policy,args.xml)
    print('Alignment:',cfg['task'],cfg['method'],'obs',cfg['obs_dim'],'actions',cfg['action_dim'],'history',cfg['history'],flush=True)
    reset_pending=[False]
    def key(code):
        if len(sim.command)<3:
            return
        if code==ord('W'): sim.command[0]=0.5
        elif code==ord('S'): sim.command[0]=-0.5
        elif code==ord('A'): sim.command[2]=0.6
        elif code==ord('D'): sim.command[2]=-0.6
        elif code==32: sim.command[:3]=0
        elif code==ord('R'): reset_pending[0]=True
    viewer=None
    if not args.headless:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=key)
    try:
        step=0
        while (not args.steps or step<args.steps) and (viewer is None or viewer.is_running()):
            start=time.perf_counter()
            if reset_pending[0]:
                sim.reset();reset_pending[0]=False
            sim.step();step+=1
            if viewer:
                viewer.sync()
                time.sleep(max(0.,cfg['simulation_dt']*cfg['control_decimation']-(time.perf_counter()-start)))
        print('MuJoCo completed',step,'steps; finite state:',bool(np.isfinite(sim.data.qpos).all()))
    finally:
        if viewer: viewer.close()


if __name__=='__main__': main()
