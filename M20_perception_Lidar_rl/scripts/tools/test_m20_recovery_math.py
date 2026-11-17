"""Recovery accounting and acceptance checks; no simulator or PPO updates."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
V = ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load('recovery_math', V/'mdp/v111_recovery.py')
gate = load('recovery_acceptance', ROOT/'scripts/tools/m20_repair_acceptance.py')


def test_swing_cost_has_finite_budget_per_target():
    target = torch.ones(1, 2, dtype=torch.long)
    peak = torch.zeros(1, 2)
    total = 0.
    for height in (.1, .8, .2, 2., 2., .5, 1.):
        target, peak, cost = m.budget_step(target, peak, target, torch.full((1, 2), height))
        total += cost.item()
    assert total == pytest.approx(2.)
    _, _, cost = m.budget_step(target, peak, target+1, torch.ones(1, 2))
    assert cost.item() == 2.


def test_progress_never_pays_backtracking_or_entry_catchup():
    best = torch.tensor([float('nan')])
    for x, active, expected in [(0., True, 0.), (1., True, 1.), (.5, True, 0.),
                                (1., True, 0.), (1.5, True, .5), (4., False, 0.), (4., True, 0.)]:
        best, delta = m.progress_step(best, torch.tensor([x]), torch.tensor([active]))
        assert delta.item() == expected


def fixture():
    c = NS(rear_progress_scale=.5, contact_force=15., plane_tolerance=.035,
           top_support_seconds=.06, minimum_upright_cosine=.8, minimum_top_clearance=.30)
    data = NS(body_pos_w=torch.tensor([[[3., 0., 2.]]*4]),
              root_pos_w=torch.tensor([[3.1, 0., 2.5]]), projected_gravity_b=torch.tensor([[0., 0., -1.]]))
    robot = NS(data=data, find_bodies=lambda name: ([['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'].index(name)], [name]))
    forces = torch.zeros(1, 4, 3); forces[:, :, 2] = 30.
    sensor = NS(data=NS(net_forces_w=forces), find_bodies=robot.find_bodies)
    entry = dict(anchor=torch.zeros(1, 2), forward=torch.tensor([[1., 0.]]),
                 height_valid=torch.tensor([True]), tread=torch.tensor([.26]),
                 entry_near=torch.tensor([False]), lower=torch.tensor([0.]), height=torch.tensor([.16]))
    adjacent = dict(overshoot_target=torch.ones(1, 2, dtype=torch.long),
                    overshoot_side=torch.zeros(1, 2), complete=torch.tensor([True]))
    env = NS(num_envs=1, device='cpu', step_dt=.02,
             cfg=NS(v111_recovery=c, v11_perception=NS(wheel_radius=.08, forward_command_threshold=.05)),
             scene=NS(sensors={'contact_forces': sensor}),
             command_manager=NS(get_command=lambda _: torch.tensor([[1., 0., 0.]])),
             _v111_first_tread=entry, _v111_adjacent=adjacent)
    geometry = dict(ascent=torch.tensor([True]), flight_riser_count=torch.tensor([12]))
    m.reset_recovery(env, torch.tensor([True]))
    return env, robot, geometry


def test_whole_completion_is_once_after_sustained_four_wheel_support():
    env, robot, g = fixture()
    events = []
    for _ in range(5):
        m.finish_recovery(env, robot, g)
        events.append(env._v111_recovery['whole_event'].item())
    assert events == [False, False, True, False, False]
    assert env._v111_recovery['whole_flight'].item()


@pytest.mark.parametrize('fault', ['rear_airborne', 'rear_below_top', 'tilted', 'low_body', 'front_incomplete'])
def test_front_only_or_fallen_completion_is_rejected(fault):
    env, robot, g = fixture()
    if fault == 'rear_airborne':
        env.scene.sensors['contact_forces'].data.net_forces_w[:, 2, 2] = 0
    elif fault == 'rear_below_top':
        robot.data.body_pos_w[:, 2, 2] -= .16
    elif fault == 'tilted':
        robot.data.projected_gravity_b[:, 2] = -.4
    elif fault == 'low_body':
        robot.data.root_pos_w[:, 2] = 2.
    else:
        env._v111_adjacent['complete'][:] = False
    for _ in range(5):
        m.finish_recovery(env, robot, g)
    assert not env._v111_recovery['whole_flight'].any()


def test_reference_anchor_leaves_front_adaptation_on_actor_visible_ascent():
    tree = ast.parse((ROOT/'source/rl_training/rl_training/v111_recovery_runner.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'reference_anchor_loss')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<anchor>', 'exec'), scope)
    x = torch.zeros(2, 63); x[1, 57:59] = 1.
    predicted = torch.ones(2, 16, requires_grad=True)
    target = torch.zeros_like(predicted, requires_grad=True)
    loss = scope['reference_anchor_loss'](predicted, target, x, .04, 2.)
    loss.backward()
    assert target.grad is None
    for row in (1, 2, 4, 5):
        assert predicted.grad[1, row] == pytest.approx(.04*predicted.grad[0, row].item())
    for row in (0, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        assert predicted.grad[1, row] == predicted.grad[0, row]


def accepted_rows():
    c = json.loads(gate.CRITERIA.read_text())
    rows = [dict(terrain='flat', tread=None, command_speed=v, fell=False,
                 mean_velocity_x=v, p95_max_abs_reach=.05) for v in c['expected_speeds']]
    rows += [dict(terrain='ascent', tread=w, command_speed=v, fell=False,
                  all_treads_supported=True, termination='flight_complete',
                  later_stairs_mean_velocity_x=v) for w in c['expected_treads'] for v in c['expected_speeds']]
    return rows, c


def test_acceptance_rejects_missing_case_slowdown_skip_and_nan():
    rows, c = accepted_rows()
    assert not gate.assess(rows, c)
    assert gate.assess(rows[:-1], c)
    for field, value in [('later_stairs_mean_velocity_x', .2), ('later_stairs_mean_velocity_x', float('nan')),
                         ('all_treads_supported', False), ('fell', True)]:
        changed = [dict(row) for row in rows]
        changed[-1][field] = value
        assert gate.assess(changed, c)
