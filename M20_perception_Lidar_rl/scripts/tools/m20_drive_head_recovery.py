"""Fit selected output rows against a drive teacher on measured MuJoCo states.

The hidden layers and every other output row remain byte-for-byte unchanged.
This produces a normal 63->16 MLP; no runtime controller or observation changes.
Every candidate needs independent closed-loop validation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch


def entry_released(labels, frames_required):
    frames = np.zeros(2, dtype=int)
    supported = np.zeros(2, dtype=bool)
    released = []
    for label in labels:
        frames = np.where(label == 1, frames+1, 0)
        supported |= frames >= frames_required
        released.append(bool(supported.all()))
    return np.asarray(released)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ordered-checkpoint', type=Path, required=True)
    p.add_argument('--drive-checkpoint', type=Path, required=True)
    p.add_argument('--replay', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, nargs='+', default=list(range(6, 12)))
    p.add_argument('--fractions', type=float, nargs='+', default=[.25, .5, .75, 1.])
    p.add_argument('--ridge', type=float, default=10.)
    a = p.parse_args()
    if any(i < 0 or i >= 16 for i in a.rows) or len(set(a.rows)) != len(a.rows):
        raise ValueError('Distinct output rows in 0..15 required')
    if a.ridge <= 0 or any(not 0 <= f <= 1 for f in a.fractions):
        raise ValueError('Positive ridge and fractions in [0,1] required')
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo.parent/'mujoco_validation/deploy/mujoco_v135'))
    from policy import SavedPolicy
    torch.set_num_threads(1)
    ordered = SavedPolicy(a.ordered_checkpoint)
    drive = SavedPolicy(a.drive_checkpoint)
    source = torch.load(a.ordered_checkpoint, map_location='cpu', weights_only=True)
    if not np.array_equal(ordered.action_scale, drive.action_scale):
        raise ValueError('Action scale mismatch')
    observations, phase = [], []
    for path in sorted(p for directory in a.replay for p in directory.glob('*.npz')):
        with np.load(path) as d:
            active = d['time'] >= 1.
            observations.append(d['observation'][active])
            # Preserve entry; the projection must learn the distinction from
            # the existing state, not an extra runtime geometry input.
            criterion = json.loads((path.parent/'manifest.json').read_text())['support_frames']
            phase.append(entry_released(d['supported_tread'], criterion)[active]
                if path.name.startswith('ascent_') else np.zeros(active.sum(), bool))
    obs = torch.tensor(np.concatenate(observations))
    after_entry = torch.tensor(np.concatenate(phase))
    with torch.inference_mode():
        hidden = ordered.actor[:-1](obs)
        original = ordered.actor(obs)
        target = drive.actor(obs)
        desired_delta = (target-original)[:, a.rows]*after_entry[:, None]
    # Solve for a residual to the saved decoder, rather than refitting it.
    design = torch.cat((hidden, torch.ones((len(hidden), 1))), 1).double()
    gram = design.T@design+a.ridge*torch.eye(design.shape[1], dtype=torch.double)
    residual = torch.linalg.solve(gram, design.T@desired_delta.double()).float()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(ordered_checkpoint=str(a.ordered_checkpoint.resolve()),
        drive_checkpoint=str(a.drive_checkpoint.resolve()), rows=a.rows, replay=[str(p.resolve()) for p in a.replay],
        samples=len(obs), after_entry_samples=int(after_entry.sum()), ridge=a.ridge,
        entry_mask='both front wheels previously supported B1 for the benchmark confirmation frames',
        method='ridge projection of drive teacher into selected existing decoder rows',
        training='CPU supervised projection, no PPO updates', candidates=[])
    for fraction in a.fractions:
        directory = a.output/f'fraction_{fraction:g}'
        (directory/'params').mkdir(parents=True)
        for name, src in [('env.yaml', ordered.config_path), ('agent.yaml', Path(ordered.metadata['agent_path']))]:
            shutil.copyfile(src, directory/'params'/name)
        candidate = copy.deepcopy(source)
        state = candidate['actor_state_dict']
        state['mlp.6.weight'][a.rows] += fraction*residual[:-1].T
        state['mlp.6.bias'][a.rows] += fraction*residual[-1]
        # New policy, so saved Adam moments cannot be reused.
        candidate['optimizer_state_dict']['state'] = {}
        candidate['infos'] = dict(offline_recovery=True, source_iteration=source['iter'], fraction=fraction)
        path = directory/f'model_{source["iter"]}.pt'
        torch.save(candidate, path)
        item = dict(checkpoint=str(path), fraction=fraction,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            train_output_rmse=float(((fraction*design.float()@residual-desired_delta).square().mean()).sqrt()))
        manifest['candidates'].append(item)
        print(item, flush=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
