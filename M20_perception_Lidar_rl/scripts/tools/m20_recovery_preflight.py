"""CPU-only migration/ABI checks; never creates an Isaac app or trains a policy."""
import argparse
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--validation-root', type=Path, default=ROOT.parent/'mujoco_validation')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    base = a.validation_root/'deploy/mujoco_v135'
    required = [a.checkpoint, a.teacher, a.replay,
        ROOT/'deep_robotics_model/M20/usd/M20.usd',
        ROOT/'deep_robotics_model/DR02/urdf/pro/DR02-pro.urdf', base/'policy.py',
        base/'assets/m20/mjcf/M20.xml', base/'terrains/stairs_ascent_walls.xml']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError('Migration is missing required files: '+', '.join(missing))
    versions = {name: metadata.version(name) for name in ('torch', 'rsl-rl-lib', 'mujoco', 'numpy')}
    if int(versions['rsl-rl-lib'].split('.')[0]) != 5:
        raise ValueError('Recovery PPO currently supports the RSL-RL 5.x API; revalidate another major version')
    if util.find_spec('isaaclab') is None:
        raise ValueError('Activate the target host Isaac Lab environment first')
    sys.path.insert(0, str(base.resolve()))
    from policy import SavedPolicy
    saved, teacher = SavedPolicy(a.checkpoint), SavedPolicy(a.teacher)
    for field in ('nominal', 'action_scale', 'kp', 'kd', 'effort', 'armature'):
        if not np.array_equal(getattr(saved, field), getattr(teacher, field)):
            raise ValueError('Reference/current execution contract mismatch: '+field)
    if saved.obs_cfg != teacher.obs_cfg:
        raise ValueError('Reference/current observation contract mismatch')
    for path in (a.checkpoint, a.teacher):
        weights = torch.load(path, weights_only=True, map_location='cpu')
        critic = weights['critic_state_dict']
        if critic['mlp.0.weight'].shape[1] != 568:
            raise ValueError('Expected the native 568D critic; refusing guessed training ABI')
        if not all(torch.isfinite(value).all() for value in critic.values() if torch.is_tensor(value)):
            raise ValueError('Checkpoint critic contains nonfinite weights')
    with np.load(a.replay, allow_pickle=False) as data:
        obs = data['observation']
        if obs.dtype != np.float32 or obs.ndim != 2 or obs.shape[1] != 63 or not len(obs) or not np.isfinite(obs).all():
            raise ValueError('Reference observations must be nonempty, finite float32 N x 63')
        samples = len(obs)
    replay_sha = hashlib.sha256(a.replay.read_bytes()).hexdigest()
    bundled = ROOT/'data/m20_recovery/reference_observations_v1.npz'
    if a.replay.resolve() == bundled.resolve():
        expected = json.loads(bundled.with_suffix('.json').read_text())['replay_sha256']
        if replay_sha != expected:
            raise ValueError('Bundled reference replay SHA-256 mismatch')
    result = dict(status='cpu_preflight_passed', versions=versions, checkpoint=saved.metadata,
        teacher=teacher.metadata, replay=str(a.replay.resolve()), replay_samples=samples,
        replay_sha256=replay_sha,
        skipped=['Isaac runtime wiring, PPO update and training: user requested no training',
                 'Target-host physics performance: rerun after migration'],
        no_training_started=True)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], versions=versions, replay_samples=samples,
                         no_training_started=True), indent=2))


if __name__ == '__main__':
    main()
