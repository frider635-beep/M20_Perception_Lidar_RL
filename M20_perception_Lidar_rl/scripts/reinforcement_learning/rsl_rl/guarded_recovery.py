"""PPO segments with independent behavior reports; no simulator imports."""
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / 'scripts/repair_workstation_limits.json'
TASK = 'M20-Stair-V11.1-AscentRecovery-v0'


class TrainingInterrupted(KeyboardInterrupt):
    """Interrupt training without entering Isaac Kit's native signal teardown."""

    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'User requested {signal.Signals(signum).name}')


def install_interrupt_handlers(cleanup_seconds=20):
    """Call after AppLauncher installs its own handlers; headless repair only.

    The first signal unwinds Python so status/logs and evaluation children can
    be finalized. A second signal, or the deadline, terminates native cleanup
    even if Kit stops returning to Python. No partial candidate is promoted.
    """
    def interrupt(signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(cleanup_seconds)
        raise TrainingInterrupted(signum)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)


def run_evaluation(command, *, env, stdout, timeout=600):
    """Keep the evaluator and its benchmark children in one cancellable group."""
    process = subprocess.Popen(command, env=env, stdout=stdout,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A benchmark descendant may survive even after its parent exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
        raise
    return subprocess.CompletedProcess(command, returncode)


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def validate_session(task, total_updates, segment_updates, baseline_path, output_path):
    limits = json.loads(LEDGER.read_text())
    if task != TASK:
        raise ValueError('Guarded training is restricted to AscentRecovery')
    if type(total_updates) is not int or not 1 <= total_updates <= limits['max_guarded_total_updates']:
        raise ValueError('Invalid guarded total update count')
    if type(segment_updates) is not int or not 1 <= segment_updates <= limits['max_guard_interval']:
        raise ValueError('Guard interval exceeds the workstation ledger')
    if not baseline_path or not Path(baseline_path).is_file():
        raise ValueError('Guarded training requires original-model baseline results')
    baseline = Path(baseline_path)
    rows = json.loads(baseline.read_text())
    gate_spec = importlib.util.spec_from_file_location('m20_session_acceptance', ROOT/'scripts/tools/m20_repair_acceptance.py')
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    if not isinstance(rows, list):
        raise ValueError('Baseline must be a list of physical benchmark cases')
    failures = gate.validate_rows(rows, json.loads(gate.CRITERIA.read_text()))
    if failures:
        raise ValueError('Invalid original-model baseline: ' + '; '.join(failures))
    source = json.loads((ROOT/'interview_prep/CHECKPOINT.json').read_text())['sha256']
    for profile in ('standing', 'keyboard'):
        manifest = json.loads((baseline.parent/profile/'manifest.json').read_text())
        if manifest.get('checkpoint_sha256') != source:
            raise ValueError('Retention baseline must come from the original selected 35300')
    if not output_path:
        raise ValueError('Guarded training requires a session output directory')
    output = Path(output_path)
    if (output / 'status.json').exists():
        raise ValueError('Refusing to overwrite a guarded session')
    return limits


def resource_check(path, limits):
    free_gib = shutil.disk_usage(path).free / 2**30
    available_gib = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available_gib = int(line.split()[1]) / 2**20
    if free_gib < limits['minimum_free_disk_gib']:
        raise RuntimeError(f'Disk reserve reached: {free_gib:.2f} GiB free')
    if available_gib is None or available_gib < limits['minimum_available_ram_gib']:
        raise RuntimeError(f'Insufficient available RAM: {available_gib} GiB')
    return dict(free_disk_gib=free_gib, available_ram_gib=available_gib)


def evaluation_summary(report, baseline_path):
    """Validate results and retain case-level trends without stopping for behavior.

    Every behavior failure is advisory by user instruction. Report new falls,
    bypass, startup failures, stalls and lost baseline-complete ascents with
    greater visibility so a long run cannot hide them behind total reward.
    """
    candidates = report.get('candidates') or []
    if len(candidates) != 1:
        raise RuntimeError('Behavior report must contain one candidate for guarded training')
    candidate = candidates[0]
    results_path = Path(candidate.get('results', ''))
    baseline_path = Path(baseline_path)
    if not results_path.is_file() or not baseline_path.is_file():
        raise RuntimeError('Behavior report is missing candidate or baseline results')
    rows = json.loads(results_path.read_text())
    baseline_rows = json.loads(baseline_path.read_text())
    gate_spec = importlib.util.spec_from_file_location('m20_summary_acceptance', ROOT/'scripts/tools/m20_repair_acceptance.py')
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    criteria = json.loads(gate.CRITERIA.read_text())
    invalid = gate.validate_rows(rows, criteria) + gate.validate_rows(baseline_rows, criteria)
    if invalid:
        raise RuntimeError('Invalid behavior result data: ' + '; '.join(invalid))
    baseline_by_key = {gate.case_key(row): row for row in baseline_rows}
    failures, cases = [], []
    for row in rows:
        key = gate.case_key(row)
        name = '/'.join(map(str, key))
        original = baseline_by_key[key]
        if row['fell'] or row['observed_bypass'] or not row['policy_entered']:
            failures.append(name + ': fall, bypass or startup failure')
        if row['stall_detected'] and not original['stall_detected']:
            failures.append(name + ': new post-entry route stall')
        if row['terrain'] == 'ascent':
            if original.get('whole_body_top_supported') and not row['whole_body_top_supported']:
                failures.append(name + ': lost complete ascent')
            if original.get('all_treads_supported') and not row.get('all_treads_supported'):
                failures.append(name + ': lost complete tread support')
        speed_field = 'later_stairs_route_velocity_x' if row['terrain'] == 'ascent' else 'route_mean_velocity_x'
        speed, baseline_speed = row.get(speed_field), original.get(speed_field)
        cases.append(dict(case=name, speed=speed, baseline_speed=baseline_speed,
                          speed_to_command=speed/row['command_speed'] if speed is not None else None,
                          speed_to_baseline=speed/baseline_speed
                              if speed is not None and baseline_speed is not None and baseline_speed > 0 else None,
                          reach_p95=row['p95_max_abs_reach'],
                          idle_yaw_deg=row['after_c_idle_yaw_max_change_deg'],
                          moving_yaw_deg=row['forward_yaw_max_change_deg'],
                          lateral_m=row['forward_lateral_max_abs'],
                          fell=row['fell'], stalled=row['stall_detected'],
                          whole_body_top_supported=row['whole_body_top_supported'],
                          all_treads_supported=row.get('all_treads_supported'),
                          skipped_treads=row.get('skipped_treads')))
    return dict(capability_regressions=failures, cases=cases)


def learn_guarded(runner, *, total_updates, segment_updates, baseline_path, output_path, log_dir):
    """Preserve optimizer and continue learning through valid behavior failures.

    RSL 5 records the last update index, not the next index. We explicitly
    advance it between calls so segment boundaries never repeat an iteration.
    Only segment-final weights are written; baseline/user assets are untouched.
    """
    limits = validate_session(TASK, total_updates, segment_updates, baseline_path, output_path)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir).resolve()
    baseline_path = Path(baseline_path).resolve()
    source_iteration = int(runner.current_learning_iteration)
    initial_iteration = source_iteration + 1
    status = dict(status='starting', pid=os.getpid(), started=timestamp(), updated=timestamp(),
                  total_updates=total_updates, completed_updates=0, segment_updates=segment_updates,
                  initial_iteration=initial_iteration, source_iteration=source_iteration, log_dir=str(log_dir),
                  baseline_results=str(baseline_path),
                  baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  last_retained_checkpoint=None, last_checkpoint=None, repair_passed=False,
                  behavior_failure_policy='record_and_continue',
                  retention_failures_seen=False, retention_history=[])
    write_json(output / 'status.json', status)
    # Built-in interval saves are redundant with our segment-final save.
    runner.cfg['save_interval'] = 10**12
    completed = 0
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    started = time.monotonic()
    try:
        while completed < total_updates:
            resources = resource_check(output, limits)
            count = min(segment_updates, total_updates - completed)
            runner.current_learning_iteration = initial_iteration + completed
            status.update(status='training', updated=timestamp(), segment_start=completed,
                          segment_target=completed + count, resources=resources)
            write_json(output / 'status.json', status)
            print(f'[RECOVERY SESSION] additional updates {completed}/{total_updates}; segment {count}', flush=True)
            runner.learn(num_learning_iterations=count, init_at_random_ep_len=(completed == 0))
            writer = getattr(runner.logger, 'writer', None)
            if writer is not None:
                writer.flush()
                writer.close()
                runner.logger.writer = None
            expected_iteration = initial_iteration + completed + count - 1
            if runner.current_learning_iteration != expected_iteration:
                raise RuntimeError('Unexpected RSL iteration accounting at segment boundary')
            checkpoint = log_dir / f'model_{expected_iteration}.pt'
            if not checkpoint.is_file():
                raise RuntimeError('Segment did not save its final checkpoint')
            completed += count
            evaluation = output / f'eval_{completed:06d}'
            if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != status['baseline_sha256']:
                raise RuntimeError('Original-model baseline changed during training')
            status.update(status='evaluating', completed_updates=completed, updated=timestamp(),
                          candidate_checkpoint=str(checkpoint), evaluation_dir=str(evaluation),
                          last_checkpoint=str(checkpoint),
                          elapsed_seconds=time.monotonic() - started)
            write_json(output / 'status.json', status)
            command = [sys.executable, str(ROOT / 'scripts/tools/m20_repair_acceptance.py'),
                       '--checkpoint', str(checkpoint), '--output', str(evaluation),
                       '--baseline-results', str(baseline_path), '--gate', 'retention']
            # Keep periodic full evidence; every interval retains metrics and sequences.
            if completed % 250:
                command.append('--no-trajectories')
            with (output / f'eval_{completed:06d}.log').open('w') as stream:
                result = run_evaluation(command, env=cpu_env, stdout=stream)
            report_path = evaluation / 'acceptance.json'
            if not report_path.is_file():
                raise RuntimeError(f'Evaluation produced no report (exit {result.returncode})')
            report = json.loads(report_path.read_text())
            write_json(output / 'latest_acceptance.json', report)
            if report.get('validation_failures') or report.get('valid_baseline') is not True:
                raise RuntimeError(f'Behavior evaluation is invalid; inspect {report_path}')
            if report.get('retention_passed') is True and result.returncode != 0:
                raise RuntimeError(f'Evaluation exit {result.returncode} disagrees with its passing report: {report_path}')
            summary = evaluation_summary(report, baseline_path)
            summary.update(completed_updates=completed, checkpoint=str(checkpoint), report=str(report_path),
                           retention_passed=report.get('retention_passed'), repair_passed=report.get('repair_passed'))
            write_json(output/'latest_metrics.json', summary)
            with (output/'metrics_history.jsonl').open('a') as stream:
                stream.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + '\n')
            if report.get('retention_passed') is not True:
                failures = report.get('retention_failures')
                if result.returncode != 2 or not failures:
                    raise RuntimeError(f'Behavior evaluation failed without a valid rejection; inspect {report_path}')
                capability_failures = summary['capability_regressions']
                status['retention_failures_seen'] = True
                status['retention_history'].append(dict(
                    completed_updates=completed, checkpoint=str(checkpoint),
                    report=str(report_path), failures=failures, capability_regressions=capability_failures))
                status.update(last_retention_failures=failures, capability_regressions=capability_failures,
                              updated=timestamp())
                status['status'] = 'training_after_retention_failure'
                write_json(output / 'status.json', status)
                if capability_failures:
                    print('[RECOVERY CAPABILITY REGRESSION] continuing PPO as requested; '
                          'candidate is not accepted:\n'
                          + '\n'.join(f'  - {failure}' for failure in capability_failures), flush=True)
                print('[RECOVERY WARNING] retention failed; continuing PPO with recorded failures:\n'
                      + '\n'.join(f'  - {failure}' for failure in failures), flush=True)
            else:
                status.update(last_retained_checkpoint=str(checkpoint),
                              last_retention_failures=[], capability_regressions=[], updated=timestamp())
            if report.get('repair_passed'):
                status['last_repair_passed_checkpoint'] = str(checkpoint)
            status['repair_passed'] = bool(report.get('repair_passed'))
            status['retention_passed'] = bool(report.get('retention_passed'))
            status['last_checkpoint'] = str(checkpoint)
            write_json(output / 'status.json', status)
            print(f'[RECOVERY SEGMENT COMPLETE] {completed}/{total_updates}; '
                  f'retention_passed={report.get("retention_passed")}; repair_passed={status["repair_passed"]}', flush=True)
        status.update(status=('completed_with_retention_failures' if status['retention_failures_seen'] else 'completed'),
                      updated=timestamp(), elapsed_seconds=time.monotonic() - started)
        write_json(output / 'status.json', status)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            status.update(status='stopped_by_user', stop_reason=str(error) or 'KeyboardInterrupt',
                          signal=getattr(error, 'signum', signal.SIGINT), updated=timestamp())
            write_json(output / 'status.json', status)
        elif not str(status.get('status', '')).startswith('stopped_'):
            status.update(status='stopped_error', error=f'{type(error).__name__}: {error}', updated=timestamp())
            write_json(output / 'status.json', status)
        raise
