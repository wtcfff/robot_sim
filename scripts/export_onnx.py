"""Reconstruct a PT checkpoint, export the complete policy, write sim2sim YAML."""
import argparse
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
import yaml
from robot_sim import ROOT_DIR
from robot_sim.registry import make_model


class InferencePolicy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, obs_history=None):
        if self.model.is_sequence:
            return self.model.act_inference(obs, obs_history)
        return self.model.act_inference(obs)


def export(checkpoint, output=None, template=None):
    checkpoint = Path(checkpoint).resolve()
    saved = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
    spec = saved['model_spec']
    model = make_model(spec).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    output = Path(output) if output else checkpoint.parent/'exported'/(checkpoint.stem+'.onnx')
    output.parent.mkdir(parents=True, exist_ok=True)
    wrapper = InferencePolicy(model).eval()
    obs = torch.zeros(1, spec['obs_dim'])
    inputs, names = (obs,), ['obs']
    if spec['history_length']:
        inputs += (torch.zeros(1, spec['obs_dim']*spec['history_length']),)
        names += ['obs_history']
    with torch.no_grad():
        torch.onnx.export(wrapper, inputs, str(output), input_names=names, output_names=['actions'],
            dynamic_axes={name:{0:'batch'} for name in names+['actions']},
            opset_version=13, do_constant_folding=True, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(output)))
    session = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    rng = np.random.RandomState(123)
    max_error = 0.0
    for batch in (1, 7):
        values = [torch.from_numpy(rng.normal(size=(batch,*v.shape[1:])).astype(np.float32)) for v in inputs]
        with torch.no_grad():
            expected = wrapper(*values).numpy()
        actual = session.run(None,dict(zip(names,[v.numpy() for v in values])))[0]
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
        max_error = max(max_error, float(np.max(np.abs(actual-expected))))
    cfg = copy.deepcopy(saved.get('deployment'))
    if cfg is not None:
        template_path = Path(template) if template else Path(ROOT_DIR)/'scripts/sim2sim_configs'/(saved['task']+'.yaml')
        if not template_path.is_file():
            raise FileNotFoundError('ONNX exported, but no target mapping template: '+str(template_path))
        target = yaml.safe_load(template_path.read_text(encoding='utf-8'))
        cfg.update(target=target, format_version=1, task=saved['task'], method=spec['method'],
            policy=output.name, source_checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            obs_dim=spec['obs_dim'], action_dim=spec['action_dim'])
        train = saved['training_config']
        if spec['method'] == 'waq':
            cfg['method_variant'] = train['waq_variant']
        cfg['history'] = dict(enabled=bool(spec['history_length']), frames=spec['history_length'],
            frame_dim=spec['obs_dim'], input_dim=spec['history_length']*spec['obs_dim'],
            order='oldest_to_newest', include_current_observation=True,
            sample_interval_policy_steps=train['history_stride'],
            startup_fill=train['history_fill'], reset_fill=train['history_fill'])
        yaml_path=output.with_suffix('.yaml')
        yaml_path.write_text('# Generated from training snapshot plus target mapping. Explicit migration contract.\n'+yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        print('Migration config:',yaml_path)
    print('Exported:',output,'maximum absolute error:',max_error)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output')
    parser.add_argument('--template')
    args=parser.parse_args()
    export(args.checkpoint,args.output,args.template)
