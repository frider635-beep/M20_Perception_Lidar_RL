"""Capability retention must not certify repair or accept missing physical evidence."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts/tools'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load('acceptance_v2', 'm20_repair_acceptance.py')
bench = load('benchmark_v2', 'm20_entry_flat_benchmark.py')


def accepted_rows():
    criteria = json.loads(gate.CRITERIA.read_text())
    rows = []
    for profile, terrain, tread, speed in sorted(gate.expected_cases(criteria), key=str):
        rows.append(dict(schema_version=2, profile=profile, terrain=terrain, tread=tread, command_speed=speed,
            fell=False, observed_bypass=False, policy_entered=True, stall_detected=False,
            whole_body_top_supported=terrain == 'ascent', termination='flight_complete' if terrain == 'ascent' else 'duration',
            command_yaw_max_abs=0., after_c_idle_yaw_max_change_deg=.5,
            forward_yaw_max_change_deg=1., forward_lateral_max_abs=.02,
            minimum_route_speed_ratio=.9, progress_windows=30,
            route_mean_velocity_x=speed, mean_velocity_x=speed,
            later_stairs_route_velocity_x=speed if terrain == 'ascent' else None,
            later_stairs_mean_velocity_x=speed if terrain == 'ascent' else None,
            p95_max_abs_reach=.05, all_treads_supported=terrain == 'ascent'))
    return rows, criteria


def test_retention_is_not_repair():
    rows, criteria = accepted_rows()
    rows[0]['all_treads_supported'] = False
    assert not gate.assess_retention(rows, rows, criteria)
    assert gate.assess(rows, criteria)


@pytest.mark.parametrize('change', [
    {'fell': True}, {'stall_detected': True}, {'whole_body_top_supported': False},
    {'later_stairs_route_velocity_x': .1}, {'forward_yaw_max_change_deg': 7.},
    {'minimum_route_speed_ratio': .4}, {'forward_lateral_max_abs': .4}])
def test_retention_reports_loss_of_capability(change):
    rows, criteria = accepted_rows()
    baseline = copy.deepcopy(rows)
    ascent = next(row for row in rows if row['terrain'] == 'ascent')
    ascent.update(change)
    assert gate.assess_retention(rows, baseline, criteria)


def test_early_fall_is_valid_failed_behavior_with_no_speed_window():
    rows, criteria = accepted_rows()
    baseline = copy.deepcopy(rows)
    fallen = next(row for row in rows if row['terrain'] == 'ascent')
    fallen.update(fell=True, termination='fall', whole_body_top_supported=False,
                  progress_windows=0, minimum_route_speed_ratio=None,
                  route_mean_velocity_x=None, later_stairs_route_velocity_x=None,
                  p95_max_abs_reach=None)
    assert not gate.validate_rows(rows, criteria)
    assert any('fall' in failure for failure in gate.assess(rows, criteria))
    assert any('fall' in failure for failure in gate.assess_retention(rows, baseline, criteria))
    # Recovery from that failed baseline is valid evidence, not None arithmetic.
    assert not gate.assess_retention(baseline, rows, criteria)


@pytest.mark.parametrize('field', ['forward_yaw_max_change_deg', 'route_mean_velocity_x',
    'minimum_route_speed_ratio', 'after_c_idle_yaw_max_change_deg'])
def test_nonfinite_or_missing_evidence_fails_closed(field):
    rows, criteria = accepted_rows()
    rows[0][field] = float('nan')
    assert gate.assess(rows, criteria)
    assert gate.assess_retention(rows, rows, criteria)
    del rows[0][field]
    assert gate.assess(rows, criteria)


def test_keyboard_cases_are_required():
    rows, criteria = accepted_rows()
    assert not gate.assess(rows, criteria)
    assert gate.assess([row for row in rows if row['profile'] == 'standing'], criteria)
    assert gate.assess(rows + [rows[0]], criteria)


def test_yaw_idle_and_motion_are_independent():
    rows, criteria = accepted_rows()
    keyboard = next(row for row in rows if row['profile'] == 'z_c_8' and row['terrain'] == 'flat')
    keyboard['after_c_idle_yaw_max_change_deg'] = 3.
    assert any('idle yaw' in failure for failure in gate.assess(rows, criteria))
    keyboard['after_c_idle_yaw_max_change_deg'] = 0.
    keyboard['forward_yaw_max_change_deg'] = 4.
    assert any('moving yaw' in failure for failure in gate.assess(rows, criteria))


def test_top_crossing_requires_rear_support_and_clearance():
    wheels = np.array([[4., .2, 2.01], [4., -.2, 2.01], [3.8, .2, 2.01], [3.8, -.2, 2.01]])
    forces = np.zeros((4, 3)); forces[:, 2] = 30.
    args = dict(top_height=1.92, goal_x=3.37, wheel_radius=.09, width=1.4)
    assert bench.top_support_candidate(wheels, forces, [4., 0., 2.45], 1., **args)
    forces[2, 2] = 0.
    assert not bench.top_support_candidate(wheels, forces, [4., 0., 2.45], 1., **args)
    forces[2, 2] = 30.
    assert not bench.top_support_candidate(wheels, forces, [4., 0., 2.1], 1., **args)
    assert not bench.top_support_candidate(wheels, forces, [4., 0., 2.45], .5, **args)


def test_local_stall_cannot_be_hidden_by_later_sprint():
    time = np.arange(0., 6., .02)
    moving = np.ones(len(time), dtype=bool)
    position = np.where(time < 2., time, np.where(time < 4., 2., 2.+2*(time-4.)))
    result = bench.route_progress_metrics(time, position, position-.5, moving, 1., .26)
    assert result['stall_detected']
    assert result['minimum_route_speed_ratio'] == pytest.approx(0.)
    result = bench.route_progress_metrics(time, time, time-.5, moving, 1., .26)
    assert not result['stall_detected']
    assert result['minimum_route_speed_ratio'] == pytest.approx(1.)


def test_rear_sequences_do_not_replace_front_sequences():
    labels = np.repeat([[1, 1, 1, 1], [2, 2, 3, 3], [3, 3, 3, 3]], 3, axis=0)
    assert bench.support_sequence_report(labels[:, :2])['no_observed_tread_skip']
    assert not bench.support_sequence_report(labels[:, 2:])['no_observed_tread_skip']
