"""No simulator or PPO: guard limits and segment failure/iteration accounting."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('guarded_test', ROOT/'scripts/reinforcement_learning/rsl_rl/guarded_recovery.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def fixture(tmp_path, monkeypatch, fail_at=None):
    baseline = tmp_path/'baseline.json'
    other_spec = importlib.util.spec_from_file_location('acceptance_fixture', ROOT/'scripts/tools/test_m20_repair_acceptance.py')
    other = importlib.util.module_from_spec(other_spec)
    other_spec.loader.exec_module(other)
    rows, _ = other.accepted_rows()
    baseline.write_text(json.dumps(rows))
    identity = json.loads((ROOT/'interview_prep/CHECKPOINT.json').read_text())['sha256']
    for profile in ('standing', 'keyboard'):
        (tmp_path/profile).mkdir()
        (tmp_path/profile/'manifest.json').write_text(json.dumps(dict(checkpoint_sha256=identity)))
    logs = tmp_path/'training'
    logs.mkdir()
    output = tmp_path/'session'
    calls = []
    runner = SimpleNamespace(current_learning_iteration=35300, cfg={}, logger=SimpleNamespace(writer=None))
    def learn(num_learning_iterations, init_at_random_ep_len):
        calls.append((runner.current_learning_iteration, num_learning_iterations, init_at_random_ep_len))
        runner.current_learning_iteration += num_learning_iterations-1
        (logs/f'model_{runner.current_learning_iteration}.pt').write_bytes(b'fake-test-checkpoint')
    runner.learn = learn
    monkeypatch.setattr(guard, 'resource_check', lambda *args: {})
    def evaluate(command, **kwargs):
        target = Path(command[command.index('--output')+1])
        target.mkdir()
        passed = len(calls) != fail_at
        (target/'results.json').write_text(json.dumps(rows))
        (target/'acceptance.json').write_text(json.dumps(dict(
            retention_passed=passed, repair_passed=False, valid_baseline=True,
            validation_failures=[], retention_failures=[] if passed else ['test route speed regressed'],
            baseline_results=str(baseline), candidates=[dict(results=str(target/'results.json'))])))
        return SimpleNamespace(returncode=0 if passed else 2)
    monkeypatch.setattr(guard, 'run_evaluation', evaluate)
    return runner, calls, baseline, logs, output


def test_segments_preserve_exact_total_without_duplicate_iterations(tmp_path, monkeypatch):
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch)
    guard.learn_guarded(runner, total_updates=207, segment_updates=100,
                        baseline_path=baseline, output_path=output, log_dir=logs)
    assert calls == [(35301,100,True), (35401,100,False), (35501,7,False)]
    status = json.loads((output/'status.json').read_text())
    assert status['status'] == 'completed' and status['completed_updates'] == 207


def test_retention_failure_continues_and_is_recorded(tmp_path, monkeypatch):
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch, fail_at=2)
    guard.learn_guarded(runner, total_updates=207, segment_updates=100,
                        baseline_path=baseline, output_path=output, log_dir=logs)
    assert calls == [(35301,100,True), (35401,100,False), (35501,7,False)]
    status = json.loads((output/'status.json').read_text())
    assert status['status'] == 'completed_with_retention_failures'
    assert status['last_retained_checkpoint'].endswith('model_35507.pt')
    assert status['retention_failures_seen'] is True
    assert len(status['retention_history']) == 1
    assert status['retention_history'][0]['completed_updates'] == 200
    assert status['retention_history'][0]['capability_regressions'] == []
    history = [json.loads(line) for line in (output/'metrics_history.jsonl').read_text().splitlines()]
    assert [entry['completed_updates'] for entry in history] == [100, 200, 207]
    assert len(history[1]['cases']) == 12


def test_final_failed_candidate_is_saved_without_promoting_it(tmp_path, monkeypatch):
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch, fail_at=2)
    guard.learn_guarded(runner, total_updates=200, segment_updates=100,
                        baseline_path=baseline, output_path=output, log_dir=logs)
    status = json.loads((output/'status.json').read_text())
    assert len(calls) == 2 and status['completed_updates'] == 200
    assert status['status'] == 'completed_with_retention_failures'
    assert status['last_checkpoint'].endswith('model_35500.pt')
    assert status['last_retained_checkpoint'].endswith('model_35400.pt')
    assert status['retention_passed'] is False and status['repair_passed'] is False
    assert 'last_repair_passed_checkpoint' not in status
    assert status['behavior_failure_policy'] == 'record_and_continue'


@pytest.mark.parametrize('change,expected', [
    ({'fell': True}, 'fall'),
    ({'observed_bypass': True}, 'bypass'),
    ({'policy_entered': False}, 'startup'),
    ({'stall_detected': True}, 'stall'),
    ({'whole_body_top_supported': False}, 'lost complete ascent'),
    ({'all_treads_supported': False}, 'lost complete tread support'),
    ({'fell': True, 'whole_body_top_supported': False, 'termination': 'fall',
      'progress_windows': 0, 'minimum_route_speed_ratio': None,
      'route_mean_velocity_x': None, 'later_stairs_route_velocity_x': None,
      'p95_max_abs_reach': None}, 'fall'),
])
def test_capability_regressions_are_highlighted_but_learning_continues(
        tmp_path, monkeypatch, capsys, change, expected):
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch, fail_at=1)
    original_evaluate = guard.run_evaluation
    def evaluate(command, **kwargs):
        result = original_evaluate(command, **kwargs)
        if len(calls) == 1:
            results_path = Path(command[command.index('--output')+1])/'results.json'
            rows = json.loads(results_path.read_text())
            next(row for row in rows if row['terrain'] == 'ascent').update(change)
            results_path.write_text(json.dumps(rows))
        return result
    monkeypatch.setattr(guard, 'run_evaluation', evaluate)
    guard.learn_guarded(runner, total_updates=201, segment_updates=100,
                        baseline_path=baseline, output_path=output, log_dir=logs)
    status = json.loads((output/'status.json').read_text())
    assert len(calls) == 3 and status['completed_updates'] == 201
    assert status['status'] == 'completed_with_retention_failures'
    assert any(expected in failure for failure in status['retention_history'][0]['capability_regressions'])
    assert '[RECOVERY CAPABILITY REGRESSION]' in capsys.readouterr().out


def test_repeated_failures_continue_without_resetting_optimizer_or_promoting_model(tmp_path, monkeypatch):
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch)
    original_evaluate = guard.run_evaluation
    optimizer_identity = object()
    runner.alg = SimpleNamespace(optimizer=optimizer_identity)
    original_learn = runner.learn
    def learn(**kwargs):
        assert runner.alg.optimizer is optimizer_identity
        original_learn(**kwargs)
    runner.learn = learn
    def failed_evaluation(command, **kwargs):
        original_evaluate(command, **kwargs)
        target = Path(command[command.index('--output')+1])
        report = json.loads((target/'acceptance.json').read_text())
        report.update(retention_passed=False, retention_failures=['speed regression'])
        (target/'acceptance.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(guard, 'run_evaluation', failed_evaluation)
    guard.learn_guarded(runner, total_updates=301, segment_updates=100,
                        baseline_path=baseline, output_path=output, log_dir=logs)
    status = json.loads((output/'status.json').read_text())
    assert len(calls) == 4 and status['completed_updates'] == 301
    assert status['last_retained_checkpoint'] is None
    assert status['last_checkpoint'].endswith('model_35601.pt')
    assert len(status['retention_history']) == 4
    assert not status['repair_passed']


@pytest.mark.parametrize('invalid_report,exit_code', [
    (dict(retention_passed=False, valid_baseline=False, validation_failures=['missing case']), 2),
    (dict(retention_passed=False, valid_baseline=True, validation_failures=[], retention_failures=[]), 2),
    (dict(retention_passed=True, valid_baseline=True, validation_failures=[]), 1),
])
def test_evaluation_errors_remain_errors_and_never_continue(tmp_path, monkeypatch, invalid_report, exit_code):
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch)
    def invalid_evaluation(command, **kwargs):
        target = Path(command[command.index('--output')+1])
        target.mkdir()
        (target/'acceptance.json').write_text(json.dumps(invalid_report))
        return SimpleNamespace(returncode=exit_code)
    monkeypatch.setattr(guard, 'run_evaluation', invalid_evaluation)
    with pytest.raises(RuntimeError) as error:
        guard.learn_guarded(runner, total_updates=30000, segment_updates=25,
                            baseline_path=baseline, output_path=output, log_dir=logs)
    assert len(calls) == 1
    status = json.loads((output/'status.json').read_text())
    assert status['status'] == 'stopped_error'
    assert status['last_retained_checkpoint'] is None


def test_continuation_finishes_with_cleanup_and_zero_worker_exit(monkeypatch, tmp_path, capsys):
    """Exercise the real loop and train.py handlers without Isaac/Hydra/PPO."""
    source = ROOT/'scripts/reinforcement_learning/rsl_rl/train.py'
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    training_try = next(node for node in reversed(main.body) if isinstance(node, ast.Try))
    entrypoint = tree.body[-1]
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch, fail_at=1)
    monkeypatch.setitem(sys.modules, 'guarded_recovery', guard)
    cleaned = []
    namespace = dict(
        __name__='__main__', args_cli=SimpleNamespace(
            headless=True, task=guard.TASK, guarded_total_updates=201,
            guard_baseline=baseline, guard_output=output),
        REPAIR_TASKS=[guard.TASK], TrainingInterrupted=guard.TrainingInterrupted,
        runner=runner, agent_cfg=SimpleNamespace(max_iterations=100), log_dir=logs,
        env=SimpleNamespace(close=lambda: cleaned.append('env closed')),
        v135_report=False, run_status='completed', sys=sys,
    )
    def run_handler():
        exec(compile(ast.Module(body=[training_try], type_ignores=[]), str(source), 'exec'), namespace)
    class WorkerExit(BaseException):
        pass
    def exit_worker(code):
        cleaned.append(('exit', code))
        raise WorkerExit
    namespace.update(main=run_handler, os=SimpleNamespace(_exit=exit_worker))
    with pytest.raises(WorkerExit):
        exec(compile(ast.Module(body=[entrypoint], type_ignores=[]), str(source), 'exec'), namespace)
    assert cleaned == ['env closed', ('exit', 0)]
    assert len(calls) == 3
    assert namespace['run_status'] == 'completed'
    captured = capsys.readouterr()
    assert 'continuing PPO' in captured.out
    assert 'Traceback' not in captured.err


def test_invalid_baseline_prevents_any_update(tmp_path, monkeypatch):
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch)
    rows = json.loads(baseline.read_text())
    rows[0]['schema_version'] = 1
    baseline.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='Invalid original-model baseline'):
        guard.learn_guarded(runner, total_updates=30000, segment_updates=25,
                            baseline_path=baseline, output_path=output, log_dir=logs)
    assert not calls


@pytest.mark.parametrize('total,interval', [(30001,100),(30000,101),(0,100),(30000,0)])
def test_session_limits_fail_before_simulator(tmp_path,total,interval):
    with pytest.raises(ValueError):
        guard.validate_session(guard.TASK,total,interval,tmp_path/'missing',tmp_path/'session')


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_interrupt_stops_without_promoting_partial_candidate(tmp_path, monkeypatch, signum):
    runner, calls, baseline, logs, output = fixture(tmp_path, monkeypatch)
    def interrupted_learn(**kwargs):
        calls.append(kwargs)
        raise guard.TrainingInterrupted(signum)
    runner.learn = interrupted_learn
    with pytest.raises(guard.TrainingInterrupted):
        guard.learn_guarded(runner, total_updates=30000, segment_updates=25,
                            baseline_path=baseline, output_path=output, log_dir=logs)
    status = json.loads((output/'status.json').read_text())
    assert len(calls) == 1
    assert status['status'] == 'stopped_by_user'
    assert status['signal'] == signum
    assert status['completed_updates'] == 0
    assert status['last_retained_checkpoint'] is None
    assert not list(output.glob('eval_*'))


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_real_signal_uses_python_cleanup_and_signal_exit_code(signum):
    program = '''
import os, signal, sys
from guarded_recovery import TrainingInterrupted, install_interrupt_handlers
signal.signal(signal.SIGTERM, lambda *args: os._exit(99))
signal.signal(signal.SIGINT, lambda *args: os._exit(99))
install_interrupt_handlers()
try:
    try:
        os.kill(os.getpid(), int(sys.argv[1]))
    finally:
        print('cleanup reached', flush=True)
except TrainingInterrupted as error:
    sys.exit(128 + error.signum)
'''
    result = subprocess.run([sys.executable, '-c', program, str(signum)],
                            cwd=Path(guard.__file__).parent, capture_output=True, text=True, timeout=5)
    assert result.returncode == 128 + signum
    assert result.stdout.strip() == 'cleanup reached'


def test_native_cleanup_deadline_cannot_hang_indefinitely():
    program = '''
import os, signal, time
from guarded_recovery import TrainingInterrupted, install_interrupt_handlers
install_interrupt_handlers(cleanup_seconds=1)
try:
    os.kill(os.getpid(), signal.SIGTERM)
except TrainingInterrupted:
    time.sleep(30)
'''
    result = subprocess.run([sys.executable, '-c', program],
                            cwd=Path(guard.__file__).parent, timeout=5)
    assert result.returncode == -signal.SIGALRM


def test_evaluation_timeout_cleans_descendant_even_after_parent_exits(tmp_path):
    child_pid_file = tmp_path/'child.pid'
    # The parent accepts TERM, but its benchmark child ignores it.
    child = "import os,signal,time,pathlib,sys; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(30)"
    child_pid = None
    try:
        with (tmp_path/'eval.log').open('w') as stream:
            with pytest.raises(subprocess.TimeoutExpired):
                guard.run_evaluation([sys.executable, '-c', parent, child, str(child_pid_file)],
                                     env=os.environ.copy(), stdout=stream, timeout=1)
        child_pid = int(child_pid_file.read_text())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stat = Path(f'/proc/{child_pid}/stat')
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                break
            time.sleep(.02)
        else:
            pytest.fail('Benchmark descendant survived evaluation cancellation')
    finally:
        if child_pid is None and child_pid_file.exists():
            child_pid = int(child_pid_file.read_text())
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
