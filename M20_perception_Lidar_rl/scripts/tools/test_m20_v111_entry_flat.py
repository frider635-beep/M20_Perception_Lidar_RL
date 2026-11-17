"""CPU positive/negative entry trajectories; no Isaac startup required."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
VELOCITY = ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity'
spec = importlib.util.spec_from_file_location('entry_repair_test', VELOCITY/'mdp/v111_entry_flat.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
tree = ast.parse((VELOCITY/'config/wheeled/deeprobotics_m20/v111_entry_flat_env_cfg.py').read_text())
ledger = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'M20V111EntryFlatCfg')
CFG = SimpleNamespace(**{node.target.id: ast.literal_eval(node.value) for node in ledger.body if isinstance(node, ast.AnnAssign)})


def fixture():
    s = {}
    repair.reset_entry(s, torch.tensor([False]), SimpleNamespace(num_envs=1, device='cpu'))
    def frame(x, bottom, force=(100., 100.), pending=True):
        s.update(repair.entry_step(s['entry_certified'], s['entry_approached'], s['entry_support_steps'],
            s['entry_skip_steps'], s['entry_failed'], torch.tensor([x]), torch.tensor([bottom]),
            torch.tensor([force]), torch.tensor([0.]), torch.tensor([.16]), torch.tensor([.26]),
            torch.tensor([pending]), CFG, .02))
        return s
    return s, frame


@pytest.mark.parametrize('leader', [0, 1])
def test_each_leader_can_support_b1_then_follow(leader):
    s, f = fixture()
    f([-.15, -.15], [0., 0.])
    f([.05, .05], [.23, .23], [0., 0.])
    x, z, force = [-.05, -.05], [.24, .24], [0., 0.]
    x[leader], z[leader], force[leader] = .12, .16, 100.
    for _ in range(3):
        f(x, z, force)
    assert s['entry_certified'][0, leader]
    assert not s['entry_certified'][0, 1-leader]
    for _ in range(3):
        f([.12, .12], [.16, .16])
    assert s['entry_certified'].all() and not s['entry_failed'].any()
    f([.4, .4], [.32, .32])
    assert not s['entry_failed'].any(), 'normal later steps treated as an entry skip'
    assert not s['entry_event'].any(), 'support event paid more than once'


@pytest.mark.parametrize('skipper', [0, 1])
def test_one_front_wheel_skip_is_persistent_failure(skipper):
    s, f = fixture()
    f([-.15, -.15], [0., 0.])
    x, z = [-.05, -.05], [0., 0.]
    x[skipper], z[skipper] = .4, .32
    f(x, z)
    f(x, z)
    assert s['entry_failed'].all() and not s['entry_certified'].any()
    assert s['entry_fail_event'].all()
    for _ in range(4):
        f([.12, .12], [.16, .16])
    assert s['entry_failed'].all() and not s['entry_event'].any()
    assert not s['entry_fail_event'].any(), 'failure event paid repeatedly'
    repair.reset_entry(s, torch.tensor([True]), SimpleNamespace(num_envs=1, device='cpu'))
    assert not s['entry_failed'].any() and not s['entry_approached'].any()


def test_airborne_and_no_prior_approach_never_pay_support():
    s, f = fixture()
    for _ in range(5):
        f([.12, .12], [.16, .16])
    assert not s['entry_certified'].any()
    f([-.15, -.15], [0., 0.])
    for _ in range(5):
        f([.4, .4], [.4, .4], [0., 0.])
    assert not s['entry_certified'].any() and s['entry_overshoot'].item() > 0
    for _ in range(5):
        f([.12, .12], [.16, .16], [0., 0.])
    assert not s['entry_certified'].any()


def test_non_ascent_is_inactive_and_nonfinite_is_rejected():
    s, f = fixture()
    for _ in range(5):
        f([.4, .4], [.32, .32], pending=False)
    assert not s['entry_failed'].any() and not s['entry_certified'].any()
    assert s['entry_overshoot'].item() == 0
    with pytest.raises(RuntimeError, match='Nonfinite'):
        f([float('nan'), 0.], [0., 0.])


def test_no_entry_slowdown_on_later_steps(monkeypatch):
    # Call the production reward functions with actual frozen entry evidence.
    # Persistent failure history must not suppress drive after leaving B1.
    import sys
    import types
    repair.__package__ = 'entry_repair_test_package'
    package = types.ModuleType(repair.__package__)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, repair.__package__, package)
    native = types.ModuleType(repair.__package__+'.v111_rear_posture')
    native.v111_world_forward_deficit_with_descent_support = lambda *a: torch.tensor([.8])
    monkeypatch.setitem(sys.modules, native.__name__, native)
    env = SimpleNamespace(cfg=SimpleNamespace(v111_entry_flat=CFG))
    monkeypatch.setattr(repair, 'flat_state', lambda env: dict(workspace_cost=torch.tensor([9.]), blend=torch.tensor([0.])))
    monkeypatch.setattr(repair, 'entry_state', lambda env: dict(entry_near=torch.tensor([False]), entry_failed=torch.tensor([True])))
    assert repair.drive_multiplier(env).item() == 1.
    assert repair.forward_deficit(env).item() == pytest.approx(.8)
    monkeypatch.setattr(repair, 'entry_state', lambda env: dict(entry_near=torch.tensor([True])))
    assert repair.forward_deficit(env).item() == 0.


def test_benchmark_contact_debounce_and_immutable_first_support():
    import numpy as np
    spec = importlib.util.spec_from_file_location('entry_benchmark_test', ROOT/'scripts/tools/m20_entry_flat_benchmark.py')
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    prior, frames, first = np.zeros(2, dtype=int), np.zeros(2, dtype=int), np.zeros(2, dtype=int)
    for index in range(3):
        supported = np.array([1, 2])
        frames, first = benchmark.update_touchdown(supported, prior, frames, first)
        prior = supported
        np.testing.assert_array_equal(first, [1, 2] if index == 2 else [0, 0])
    for _ in range(4):
        supported = np.array([3, 3])
        frames, first = benchmark.update_touchdown(supported, prior, frames, first)
        prior = supported
    np.testing.assert_array_equal(first, [1, 2])
