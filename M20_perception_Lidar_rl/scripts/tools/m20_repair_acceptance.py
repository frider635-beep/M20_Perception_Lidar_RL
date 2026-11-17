"""Evaluate all saved stage candidates serially; never choose by total reward."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT/'scripts/repair_acceptance.json'


def assess(rows, criteria):
    expected = {('flat', None, v) for v in criteria['expected_speeds']}
    expected |= {('ascent', w, v) for w in criteria['expected_treads'] for v in criteria['expected_speeds']}
    actual = {(r['terrain'], r['tread'], r['command_speed']) for r in rows}
    failures = []
    if actual != expected or len(rows) != len(expected):
        failures.append('Missing, extra or duplicate benchmark cases')
    for row in rows:
        name = f"{row['terrain']}/{row['tread']}/{row['command_speed']}"
        if row['fell']:
            failures.append(name+': fall')
        if row['terrain'] == 'ascent':
            if not row.get('all_treads_supported') or row['termination'] != 'flight_complete':
                failures.append(name+': incomplete or skipped tread sequence')
            speed = row['later_stairs_mean_velocity_x']
        else:
            speed = row['mean_velocity_x']
            reach = row['p95_max_abs_reach']
            if reach is None or not reach <= criteria['maximum_flat_reach_p95']:
                failures.append(name+': flat wheel workspace')
            if speed is None or not speed <= criteria['maximum_flat_speed_ratio']*row['command_speed']:
                failures.append(name+': flat overspeed')
        if speed is None or not speed >= criteria['minimum_speed_ratio']*row['command_speed']:
            failures.append(name+': speed deficit')
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation-root', type=Path, default=ROOT.parent/'mujoco_validation')
    a = p.parse_args()
    criteria = json.loads(CRITERIA.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(criteria=criteria, accepted_checkpoint=None, candidates=[])
    candidates = sorted(a.run_dir.glob('model_*.pt'), key=lambda path: int(path.stem.split('_')[-1]), reverse=True)
    if not candidates:
        raise ValueError('Stage produced no saved checkpoints')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for checkpoint in candidates:
        output = a.output/checkpoint.stem
        command = [sys.executable, str(ROOT/'scripts/tools/m20_entry_flat_benchmark.py'),
            '--checkpoint', str(checkpoint.resolve()), '--validation-root', str(a.validation_root.resolve()),
            '--height', str(criteria['stair_height']), '--speeds', *map(str, criteria['expected_speeds']),
            '--treads', *map(str, criteria['expected_treads']), '--output', str(output.resolve())]
        with (a.output/(checkpoint.stem+'.log')).open('w') as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        rows = json.loads((output/'results.json').read_text())
        failures = assess(rows, criteria)
        report['candidates'].append(dict(checkpoint=str(checkpoint.resolve()), failures=failures))
        if not failures:
            report['accepted_checkpoint'] = str(checkpoint.resolve())
        (a.output/'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
        if not failures:
            break
    if report['accepted_checkpoint'] is None:
        print('NO ACCEPTED CHECKPOINT: inspect acceptance.json; training will not continue automatically', file=sys.stderr)
        return 2
    print('Accepted for the fixed MuJoCo ascent/flat suite: '+report['accepted_checkpoint'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
