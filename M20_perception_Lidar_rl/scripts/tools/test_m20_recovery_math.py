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


def test_inactive_backtracking_never_reopens_old_progress():
    best = torch.tensor([float('nan')])
    total = 0.
    for x, active in [(0., True), (1., True), (.5, False), (1., True),
                      (2., False), (1., False), (2., True), (2.5, True)]:
        best, delta = m.progress_step(best, torch.tensor([x]), torch.tensor([active]))
        total += delta.item()
    assert total == pytest.approx(1.5)
    assert best.item() == 2.5


@pytest.mark.parametrize('dt', [.01, .02, .04])
def test_entry_overshoot_budget_does_not_grow_with_dwell_or_step_rate(dt):
    peak = torch.zeros(1)
    total = 0.
    for x in [.2, .1, .8, 2.] + [2.] * round(5/dt):
        peak, event = m.entry_budget_step(peak, torch.tensor([x]))
        total += (event/dt*dt).item()
    assert total == pytest.approx(1.)


def quality_cfg():
    return NS(straight_lateral_deadband=.05, straight_yaw_deadband=.03,
              straight_yaw_scale=.08, straight_lateral_scale=.12,
              flat_quality_speed_scale=.5)


def test_flat_quality_requires_motion_and_rewards_compact_straight_tracking():
    command = torch.tensor([[1., 0., 0.]]*5 + [[0., 0., 0.], [1., 0., .6], [0., .5, 0.]])
    velocity = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 0., 0.],
                             [-1., 0., 0.], [1., .2, 0.], [0., 0., 0.],
                             [1., 0., 0.], [0., .5, 0.]])
    workspace = torch.tensor([0., 0., 4., 0., 0., 0., 0., 0.])
    reward = m.flat_quality_score(command, velocity, torch.zeros(8), workspace,
                                  torch.ones(8), quality_cfg())
    assert reward[1].item() == pytest.approx(1.)
    assert 0 < reward[2] < reward[1]
    assert 0 < reward[4] < reward[1]
    assert reward[[0, 3, 5, 6, 7]].tolist() == [0.]*5
    assert m.flat_quality_score(command, velocity, torch.zeros(8), workspace,
                                torch.zeros(8), quality_cfg()).tolist() == [0.]*8


def test_straight_cost_is_bounded_and_does_not_fight_requested_turns():
    command = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., .6], [0., .5, 0.]])
    velocity = torch.tensor([[0., 0., 0.], [1., 10., 0.], [0., 0., 0.], [0., .5, 0.]])
    cost = m.straight_cost(command, velocity, torch.tensor([.05, 10., .6, 0.]),
                          torch.ones(4), quality_cfg())
    assert -1 <= cost[0] < 0
    assert cost[1].item() == -1
    assert cost[2:].tolist() == [0., 0.]


def test_recovery_keeps_drive_reward_even_with_bad_flat_posture(monkeypatch):
    entry = load('entry_reward_math', V/'mdp/v111_entry_flat.py')
    env = NS(num_envs=2, device='cpu', cfg=NS(v111_recovery=NS(flat_cost_cap=2.)))
    monkeypatch.setattr(entry, 'flat_state', lambda _: dict(
        workspace_cost=torch.tensor([0., 100.]), cost=torch.zeros(2), blend=torch.ones(2)))
    assert entry.drive_multiplier(env).tolist() == [1., 1.]
    assert entry.flat_penalty(env).tolist() == pytest.approx([0., -200./102.])


def test_new_flat_cost_budget_cannot_exceed_native_drive_gain_at_benchmark_speeds():
    """A rate-budget guard, not a claim about total PPO return or convergence."""
    tree = ast.parse((V/'config/wheeled/deeprobotics_m20/v111_recovery_env_cfg.py').read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'M20V111RecoveryCfg')
    values = {n.target.id: ast.literal_eval(n.value) for n in cfg.body if isinstance(n, ast.AnnAssign)}
    new_cost_bound = values['flat_cost_cap']*values['flat_posture_weight']+values['straight_stability_weight']
    # Saved V11.1 native tracking is 5 * exp(-velocity_error^2 / .5).
    speeds = torch.tensor([.5, .7, 1.])
    native_gain_over_rest = 5*(1-torch.exp(-speeds.square()/.5))
    assert (native_gain_over_rest > new_cost_bound).all()


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
                 entry_near=torch.tensor([False]), entry_overshoot=torch.zeros(1),
                 lower=torch.tensor([0.]), height=torch.tensor([.16]))
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


def test_failed_sequence_can_still_earn_new_progress():
    env, robot, geometry = fixture()
    env._v111_adjacent['complete'][:] = False
    robot.data.root_pos_w[:, 0] = 1.
    m.finish_recovery(env, robot, geometry)
    robot.data.root_pos_w[:, 0] = 1.1
    m.finish_recovery(env, robot, geometry)
    assert env._v111_recovery['progress'].item() > 0
    assert not env._v111_recovery['whole_flight'].any()


def accepted_rows():
    fixtures = load('acceptance_fixtures', ROOT/'scripts/tools/test_m20_repair_acceptance.py')
    return fixtures.accepted_rows()


def test_acceptance_rejects_missing_case_slowdown_skip_and_nan():
    rows, c = accepted_rows()
    assert not gate.assess(rows, c)
    assert gate.assess(rows[:-1], c)
    for field, value in [('later_stairs_route_velocity_x', .2), ('later_stairs_route_velocity_x', float('nan')),
                         ('all_treads_supported', False), ('fell', True)]:
        changed = [dict(row) for row in rows]
        next(row for row in changed if row['terrain'] == 'ascent')[field] = value
        assert gate.assess(changed, c)
