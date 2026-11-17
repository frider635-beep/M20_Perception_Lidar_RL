"""Exercise physical support ledger independently of Isaac and reward wiring."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V = ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity'
spec = importlib.util.spec_from_file_location('adjacent_test', V/'mdp/v111_adjacent.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tree = ast.parse((V/'config/wheeled/deeprobotics_m20/v111_adjacent_env_cfg.py').read_text())
ledger = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'M20V111AdjacentCfg')
CFG = SimpleNamespace(**{n.target.id: ast.literal_eval(n.value) for n in ledger.body if isinstance(n, ast.AnnAssign)})


def fixture():
    env = SimpleNamespace(num_envs=1, device='cpu')
    m.reset_sequence(env, torch.tensor([False]))
    def step(indices, *, force=100., dz=0., lateral=0., active=True):
        i = torch.tensor([indices], dtype=torch.float)
        result = m.sequence_step(env._v111_adjacent, (i-.5)*.26, torch.full((1, 2), lateral),
            i*.16+dz, torch.full((1, 2), force), torch.tensor([0.]), torch.tensor([.16]),
            torch.tensor([.26]), torch.tensor([.7]), torch.tensor([12]),
            torch.full((1, 2), active), CFG, .02)
        env._v111_adjacent.update(result)
        return result
    return env, step


@pytest.mark.parametrize('leader', [0, 1])
def test_full_flight_independent_lead_and_no_repeat_credit(leader):
    env, f = fixture()
    count = 0
    for index in range(1, 13):
        pair = [index-1, index-1]
        pair[leader] = index
        for _ in range(3):
            count += int(f(pair)['support_event'].sum())
        for _ in range(3):
            s = f([index, index])
            count += int(s['support_event'].sum())
        assert s['last_index'].tolist() == [[index, index]]
        assert not f([index, index])['support_event'].any()
    assert count == 22 and s['complete'].item()
    assert not f([12, 12])['complete_event'].any()


def test_skip_survives_backtrack_and_resynchronizes():
    env, f = fixture()
    for _ in range(3):
        f([1, 1])
    for _ in range(3):
        s = f([3, 2])
    assert s['skip_event'].tolist() == [[1, 0]]
    assert s['support_event'].tolist() == [[False, True]]
    for _ in range(3):
        s = f([2, 2])
    assert not s['support_event'].any() and s['missed_count'].tolist() == [[1, 0]]
    for _ in range(3):
        s = f([4, 3])
    assert s['support_event'].all()
    for index in range(4, 13):
        for _ in range(3):
            s = f([index, index])
    assert not s['complete'].item()
    m.reset_sequence(env, torch.tensor([True]))
    assert not env._v111_adjacent['ever_skipped'].any()
    assert not env._v111_adjacent['last_index'].any()


@pytest.mark.parametrize('kwargs', [dict(force=0.), dict(dz=.08), dict(lateral=4.), dict(active=False)])
def test_false_support_never_earns_credit(kwargs):
    env, f = fixture()
    for _ in range(5):
        s = f([2, 2], **kwargs)
    assert not s['last_index'].any() and not s['skip_event'].any()


def test_nonfinite_is_rejected():
    _, f = fixture()
    with pytest.raises(RuntimeError, match='Nonfinite'):
        f([float('nan'), 1])


def test_benchmark_remembers_missing_b2_and_later_missing_treads():
    spec = importlib.util.spec_from_file_location('benchmark_test', ROOT/'scripts/tools/m20_entry_flat_benchmark.py')
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    labels = np.repeat([[1, 1], [3, 2], [2, 3], [4, 4], [6, 5]], 3, axis=0)
    report = b.support_sequence_report(labels)
    assert report['skipped_treads'] == [[2, 5], []]
    assert not report['no_observed_tread_skip']
