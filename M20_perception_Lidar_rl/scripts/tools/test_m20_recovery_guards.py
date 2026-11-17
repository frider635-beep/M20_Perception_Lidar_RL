"""Check actual launcher failures and replay phase boundaries without Isaac."""
import importlib.util
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


limits = load('repair_limits_test', ROOT/'scripts/reinforcement_learning/rsl_rl/repair_workstation_limits.py')


@pytest.mark.parametrize('n,k', [(4097, 100), (4096, 50000), (0, 1), (1, 0)])
def test_invalid_workstation_stage(n, k):
    with pytest.raises(ValueError):
        limits.validate(n, k)


def test_user_requested_environment_bound_and_default():
    assert limits.validate(None, None) == (4096, 100)
    assert limits.validate(4096, 100) == (4096, 100)
    # Smaller counts remain available for explicitly selected CPU/runtime probes.
    assert limits.validate(2048, 25) == (2048, 25)
    assert limits.validate(256, 25) == (256, 25)


def test_distributed_remains_rejected_at_requested_count():
    with pytest.raises(ValueError, match='one GPU process'):
        limits.validate(4096, 100, distributed=True)


def test_missing_limits_fail_closed(tmp_path):
    with pytest.raises(OSError):
        limits.validate(1, 1, ledger=tmp_path/'absent.json')


@pytest.mark.parametrize('script', ['train_m20_v111_entry_flat.sh', 'train_m20_v111_adjacent.sh'])
@pytest.mark.parametrize('arguments,reason', [
    (['--num_envs', '4097', '--max_iterations', '100'], 'environments must'),
    (['--num_envs=4096', '--max_iterations=50000'], 'stages are limited'),
    (['--num_envs', '4096', '--max_iterations', '100', '--distributed'], 'one GPU process'),
    (['--max_iterations', '100', '--max_iterations', '50000'], 'stages are limited')])
def test_real_launchers_reject_before_importing_isaac(script, arguments, reason):
    env = dict(os.environ, M20_PYTHON=sys.executable)
    proc = subprocess.run(['bash', str(ROOT/'scripts'/script), *arguments],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert proc.returncode != 0 and reason in proc.stderr
    assert 'isaaclab' not in proc.stderr, 'guard was bypassed and training imports were reached'


def test_support_phase_uses_each_front_wheel():
    module = load('head_recovery_test', ROOT/'scripts/tools/m20_drive_head_recovery.py')
    labels = np.array([[1, -1]]*3+[[2, 1]]*2+[[2, -1]]+[[2, 1]]*3)
    phase = module.entry_released(labels, 3)
    assert not phase[:-1].any() and phase[-1]
