"""Build physically labeled Recovery replay from a 35300 benchmark suite.

This reads recorded MuJoCo trajectories only; it performs no optimization and
does not infer flat ground from a missing B feature. Labels describe the actual
terrain case, while phase labels come from the production interaction loop.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(baseline_dir, checkpoint, output):
    baseline_dir, checkpoint, output = map(lambda p: Path(p).resolve(), (baseline_dir, checkpoint, output))
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Refusing to replace a reference replay or its metadata')
    source_sha = sha256(checkpoint)
    arrays = {key: [] for key in ('observation', 'scene', 'phase', 'case_id')}
    cases = []
    for manifest_path in sorted(baseline_dir.rglob('manifest.json')):
        results_path = manifest_path.parent/'results.json'
        if not results_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('checkpoint_sha256') != source_sha or not manifest.get('physics_enabled'):
            raise ValueError('Replay source must be actual physics rollouts of the selected teacher')
        if manifest.get('schema_version') != 2:
            raise ValueError('Replay requires benchmark v2 with recorded policy phases')
        for row in json.loads(results_path.read_text()):
            terrain = row['terrain']
            if terrain not in ('flat', 'ascent'):
                raise ValueError('Unsupported replay terrain case: '+str(terrain))
            tag = f"{terrain}_{row['tread']}_{row['command_speed']}"
            trajectory = manifest_path.parent/(tag+'.npz')
            with np.load(trajectory, allow_pickle=False) as data:
                observation = data['observation']
                active = data['policy_active']
                phase = data['phase']
                if (observation.dtype != np.float32 or observation.ndim != 2 or observation.shape[1] != 63
                    or active.shape != (len(observation),) or active.dtype.kind != 'b'
                    or phase.shape != active.shape or phase.dtype.kind not in 'iu'):
                    raise ValueError('Invalid recorded observations or physical phase labels: '+str(trajectory))
                keep = active & np.isin(phase, [1, 2, 3])
                if not keep.any() or not np.isfinite(observation[keep]).all():
                    raise ValueError('Replay case has no finite policy-active observations')
                count = int(keep.sum())
                arrays['observation'].append(observation[keep].copy())
                arrays['scene'].append(np.full(count, 0 if terrain == 'flat' else 1, dtype=np.int64))
                arrays['phase'].append(phase[keep].astype(np.int64))
                arrays['case_id'].append(np.full(count, len(cases), dtype=np.int64))
            cases.append(dict(case_id=len(cases), terrain=terrain, profile=manifest.get('profile'),
                tread=row['tread'], command_speed=row['command_speed'], samples=count,
                trajectory=str(trajectory), trajectory_sha256=sha256(trajectory),
                manifest=str(manifest_path), manifest_sha256=sha256(manifest_path)))
    if not cases or {case['terrain'] for case in cases} != {'flat', 'ascent'}:
        raise ValueError('Recovery replay needs both flat and ascent physical cases')
    if not any(case['profile'] == 'z_c_8' for case in cases):
        raise ValueError('Recovery replay must include the Z-to-C-to-8 production interaction profile')
    packed = {key: np.concatenate(value) for key, value in arrays.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    try:
        checkpoint_label = str(checkpoint.relative_to(ROOT))
        replay_label = str(output.relative_to(ROOT))
    except ValueError:
        checkpoint_label, replay_label = str(checkpoint), str(output)
    metadata = dict(schema_version=2, source_checkpoint=checkpoint_label,
        source_sha256=source_sha, replay_path=replay_label, replay_sha256=sha256(output),
        samples=len(packed['observation']), shape=list(packed['observation'].shape), dtype='float32',
        scene_names={'0': 'flat_route', '1': 'ascent_route'},
        phase_names={'1': 'policy_idle', '2': 'moving', '3': 'after_motion'},
        source='Recorded teacher MuJoCo physics, physical case labels and production policy phases; no optimization',
        scene_scope='Terrain-case identity, not an inference from actor B flags. Ascent approach/exit retain the ascent case label.',
        policy_abi='v11_signed_edges_v2; 63 observations / 16 actions', cases=cases)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'interview_prep/model_35300.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'data/m20_recovery/reference_observations_v2.npz')
    args = parser.parse_args()
    metadata = build(args.baseline_dir, args.checkpoint, args.output)
    print(json.dumps({key: metadata[key] for key in ('schema_version', 'samples', 'replay_path', 'replay_sha256')}, indent=2))


if __name__ == '__main__':
    main()
