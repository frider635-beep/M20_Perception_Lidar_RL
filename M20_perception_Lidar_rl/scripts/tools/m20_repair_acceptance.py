"""Independent MuJoCo capability retention and strict repair gates.

The training monitor records failures and continues by user instruction. A repair
pass certifies the fixed benchmark; neither gate certifies real-robot readiness.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CRITERIA = ROOT/'scripts/repair_acceptance.json'


def case_key(row):
    return (row.get('profile', 'standing'), row['terrain'], row['tread'], row['command_speed'])


def expected_cases(criteria):
    cases = {('standing', 'flat', None, v) for v in criteria['expected_speeds']}
    cases |= {('standing', 'ascent', w, v) for w in criteria['expected_treads'] for v in criteria['expected_speeds']}
    if criteria.get('require_keyboard_profile', False):
        cases |= {('z_c_8', 'flat', None, .7)}
        cases |= {('z_c_8', 'ascent', w, .7) for w in criteria['expected_treads']}
    return cases


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def validate_rows(rows, criteria):
    failures = []
    try:
        actual = {case_key(row) for row in rows}
    except (KeyError, TypeError):
        return ['Missing benchmark case identity']
    if actual != expected_cases(criteria) or len(rows) != len(actual):
        failures.append('Missing, extra or duplicate benchmark cases')
    for row in rows:
        name = '/'.join(map(str, case_key(row)))
        if row.get('schema_version') != 2:
            failures.append(name+': requires benchmark schema 2; recollect baseline')
        for field in ('fell', 'observed_bypass', 'policy_entered', 'stall_detected', 'whole_body_top_supported'):
            if not isinstance(row.get(field), bool):
                failures.append(name+': missing boolean '+field)
        # An early physical failure can occur before a speed window or stair
        # segment exists. Those explicit nulls describe a failed behavior,
        # whereas absent fields or NaN still indicate malformed evidence.
        failed_early = row.get('fell') is True or row.get('policy_entered') is False
        for field in ('command_yaw_max_abs', 'forward_yaw_max_change_deg', 'after_c_idle_yaw_max_change_deg',
                      'forward_lateral_max_abs', 'minimum_route_speed_ratio', 'progress_windows',
                      'route_mean_velocity_x', 'p95_max_abs_reach'):
            if failed_early and field in row and row[field] is None and field not in ('command_yaw_max_abs', 'progress_windows'):
                continue
            if not finite(row.get(field)):
                failures.append(name+': missing/nonfinite '+field)
        if (row.get('terrain') == 'ascent' and not finite(row.get('later_stairs_route_velocity_x'))
            and not (failed_early and 'later_stairs_route_velocity_x' in row and row['later_stairs_route_velocity_x'] is None)):
            failures.append(name+': missing/nonfinite later_stairs_route_velocity_x')
        if row.get('progress_windows', 0) == 0 and not failed_early:
            failures.append(name+': no complete post-entry progress window')
    return failures


def assess(rows, criteria):
    """Strict final gate; original models with known defects are expected to fail."""
    failures = validate_rows(rows, criteria)
    if failures:
        return failures
    for row in rows:
        name = '/'.join(map(str, case_key(row)))
        if row['fell'] or row['observed_bypass'] or not row['policy_entered']:
            failures.append(name+': fall, bypass or startup failure')
            continue
        if row['stall_detected']:
            failures.append(name+': post-entry route stall')
        if row['minimum_route_speed_ratio'] < criteria['minimum_window_speed_ratio']:
            failures.append(name+': local speed deficit')
        if row['command_yaw_max_abs'] > 1.e-7:
            failures.append(name+': nonzero yaw command')
        if row['after_c_idle_yaw_max_change_deg'] > criteria['maximum_idle_yaw_deg']:
            failures.append(name+': C idle yaw drift')
        ascent = row['terrain'] == 'ascent'
        if row['forward_yaw_max_change_deg'] > criteria['maximum_ascent_yaw_deg' if ascent else 'maximum_flat_yaw_deg']:
            failures.append(name+': moving yaw drift')
        if row['forward_lateral_max_abs'] > criteria['maximum_ascent_lateral_m' if ascent else 'maximum_flat_lateral_m']:
            failures.append(name+': moving lateral drift')
        if ascent:
            if not row.get('all_treads_supported') or row['termination'] != 'flight_complete' or not row['whole_body_top_supported']:
                failures.append(name+': incomplete support or skipped tread sequence')
            speed = row['later_stairs_route_velocity_x']
        else:
            speed = row['route_mean_velocity_x']
            if row['p95_max_abs_reach'] > criteria['maximum_flat_reach_p95']:
                failures.append(name+': flat wheel workspace')
            if speed > criteria['maximum_flat_speed_ratio']*row['command_speed']:
                failures.append(name+': flat overspeed')
        if speed < criteria['minimum_speed_ratio']*row['command_speed']:
            failures.append(name+': speed deficit')
    return failures


def assess_retention(rows, baseline_rows, criteria):
    """Do not demand immediate repair of defects already present in the baseline."""
    failures = validate_rows(rows, criteria)
    failures += ['baseline: '+f for f in validate_rows(baseline_rows, criteria)]
    if failures:
        return failures
    baseline = {case_key(row): row for row in baseline_rows}
    for row in rows:
        key = case_key(row)
        original = baseline[key]
        name = '/'.join(map(str, key))
        if row['fell'] or row['observed_bypass'] or not row['policy_entered']:
            failures.append(name+': fall, bypass or startup failure')
            continue
        if row['terrain'] == 'ascent' and original['whole_body_top_supported'] and not row['whole_body_top_supported']:
            failures.append(name+': lost complete ascent')
        field = 'later_stairs_route_velocity_x' if row['terrain'] == 'ascent' else 'route_mean_velocity_x'
        if finite(original[field]) and row[field] < original[field]*criteria['retention_minimum_speed_ratio']:
            failures.append(name+': retained route speed below baseline limit')
        if row['stall_detected'] and not original['stall_detected']:
            failures.append(name+': new post-entry stall')
        if (finite(original['minimum_route_speed_ratio']) and
            row['minimum_route_speed_ratio'] < original['minimum_route_speed_ratio']-criteria['retention_window_speed_tolerance']):
            failures.append(name+': local progress regressed')
        for field in ('forward_yaw_max_change_deg', 'after_c_idle_yaw_max_change_deg'):
            if finite(original[field]) and row[field] > original[field]+criteria['retention_yaw_tolerance_deg']:
                failures.append(name+': yaw regression '+field)
        if (finite(original['forward_lateral_max_abs']) and
            row['forward_lateral_max_abs'] > original['forward_lateral_max_abs']+criteria['retention_lateral_tolerance_m']):
            failures.append(name+': lateral drift regression')
        if row['command_yaw_max_abs'] > 1.e-7:
            failures.append(name+': nonzero yaw command')
    return failures


def benchmark(checkpoint, output, criteria, validation_root, *, no_trajectories=False):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    rows = []
    for profile in ('standing', 'keyboard'):
        if profile == 'keyboard' and not criteria.get('require_keyboard_profile'):
            continue
        target = output/profile
        command = [sys.executable, str(ROOT/'scripts/tools/m20_entry_flat_benchmark.py'),
            '--checkpoint', str(checkpoint.resolve()), '--validation-root', str(validation_root.resolve()),
            '--height', str(criteria['stair_height']), '--speeds',
            *map(str, [.7] if profile == 'keyboard' else criteria['expected_speeds']),
            '--treads', *map(str, criteria['expected_treads']), '--output', str(target.resolve())]
        if profile == 'keyboard':
            command += ['--startup', 'z', '--keyboard-profile']
        if no_trajectories:
            command.append('--no-trajectories')
        with (output/(profile+'.log')).open('w') as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        rows.extend(json.loads((target/'results.json').read_text()))
    (output/'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation-root', type=Path, default=ROOT.parent/'mujoco_validation')
    p.add_argument('--baseline-results', type=Path)
    p.add_argument('--baseline-only', action='store_true')
    p.add_argument('--gate', choices=('retention', 'repair'), default='repair')
    p.add_argument('--no-trajectories', action='store_true')
    a = p.parse_args()
    if a.baseline_only and a.checkpoint is None:
        p.error('--baseline-only requires --checkpoint')
    if a.gate == 'retention' and not (a.baseline_results or a.baseline_only):
        p.error('--gate retention requires --baseline-results')
    criteria = json.loads(CRITERIA.read_text())
    baseline = json.loads(a.baseline_results.read_text()) if a.baseline_results else None
    candidates = [a.checkpoint] if a.checkpoint else sorted(a.run_dir.glob('model_*.pt'),
        key=lambda path: int(path.stem.split('_')[-1]), reverse=True)
    if not candidates:
        raise ValueError('Stage produced no saved checkpoints')
    report = dict(schema_version=2, criteria=criteria, gate=a.gate, baseline_only=a.baseline_only,
        baseline_results=str(a.baseline_results.resolve()) if a.baseline_results else None,
        baseline_sha256=hashlib.sha256(a.baseline_results.read_bytes()).hexdigest() if a.baseline_results else None,
        accepted_checkpoint=None, retention_checkpoint=None, candidates=[])
    if a.run_dir:
        a.output.mkdir(parents=True, exist_ok=False)
    passed = False
    for checkpoint in candidates:
        output = a.output/checkpoint.stem if a.run_dir else a.output
        rows = benchmark(checkpoint, output, criteria, a.validation_root, no_trajectories=a.no_trajectories)
        validation = validate_rows(rows, criteria)
        repair_failures = assess(rows, criteria)
        retention_failures = assess_retention(rows, baseline, criteria) if baseline is not None else ['No baseline supplied']
        candidate = dict(checkpoint=str(checkpoint.resolve()), results=str((output/'results.json').resolve()),
            valid_baseline=not validation, validation_failures=validation,
            retention_passed=not retention_failures, repair_passed=not repair_failures,
            retention_failures=retention_failures, repair_failures=repair_failures)
        report['candidates'].append(candidate)
        if candidate['retention_passed']:
            report['retention_checkpoint'] = str(checkpoint.resolve())
        if candidate['repair_passed']:
            report['accepted_checkpoint'] = str(checkpoint.resolve())
        report.update({key: candidate[key] for key in ('retention_passed', 'repair_passed', 'valid_baseline',
            'retention_failures', 'repair_failures', 'validation_failures')})
        (a.output/'acceptance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        passed = candidate['valid_baseline'] if a.baseline_only else candidate[a.gate+'_passed']
        if passed:
            break
    print(json.dumps(dict(output=str(a.output.resolve()), gate=a.gate, passed=passed,
        baseline_only=a.baseline_only, retention_passed=report['retention_passed'], repair_passed=report['repair_passed'])))
    return 0 if passed else 2


if __name__ == '__main__':
    sys.exit(main())
