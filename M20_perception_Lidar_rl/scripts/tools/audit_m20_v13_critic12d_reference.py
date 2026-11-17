#!/usr/bin/env python3
"""Fail closed if Critic12D drifts from native V13 outside its allow-list."""

from __future__ import annotations

import argparse
import copy
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

NATIVE = "M20-Stair-V13-V11FD-Alternation-v0"
CANDIDATE = "M20-Stair-V13-V11FD-Alternation-Critic12D-v0"


def _remove(mapping, path):
    node = mapping
    for key in path[:-1]:
        node = node[key]
    if path[-1] not in node:
        raise AssertionError(f"expected allow-listed path is absent: {'.'.join(path)}")
    del node[path[-1]]


def _diff(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [f"{path}: keys {sorted(set(left) ^ set(right))}"]
        out = []
        for key in sorted(left):
            out.extend(_diff(left[key], right[key], f"{path}.{key}" if path else key))
        return out
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out.extend(_diff(a, b, f"{path}[{index}]"))
        return out
    try:
        equal = left == right
        if isinstance(equal, bool) and equal:
            return []
    except Exception:
        pass
    return [] if repr(left) == repr(right) else [f"{path}: {left!r} != {right!r}"]


def main():
    native = copy.deepcopy(parse_env_cfg(NATIVE, device=args.device, num_envs=8).to_dict())
    candidate = copy.deepcopy(parse_env_cfg(CANDIDATE, device=args.device, num_envs=8).to_dict())
    for name in ("v13_reward_state", "v13_descent_state"):
        _remove(candidate, ("observations", "critic", name))
    for key in (
        "course_active_rate", "live_edge_rate", "edge_distance_mean",
        "drop_height_mean", "rear_extension_max_mean", "rear_contact_fraction",
    ):
        _remove(candidate, ("curriculum", f"v13_critic_descent_{key}"))
    _remove(candidate, ("curriculum", "v13_ascent_route_engagement"))
    _remove(candidate, ("rewards", "v13_descent_rear_extension"))
    _remove(candidate, ("rewards", "v13_descent_rear_roll"))
    # The behavioral allow-list is deliberately explicit: moderate ascent
    # commands, reduced ascent speed pressure, and no other native V13 drift.
    for key in (
        "soft_alternation_command_split", "stair_forward_command_min",
        "stair_forward_command_max",
    ):
        candidate["v11_perception"][key] = copy.deepcopy(native["v11_perception"][key])
    for key in (
        "non_ascending_stair_forward_command_min",
        "non_ascending_stair_forward_command_max",
    ):
        _remove(candidate, ("v11_perception", key))
    for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "world_forward_deficit"):
        candidate["rewards"][name] = copy.deepcopy(native["rewards"][name])
    _remove(candidate, ("v13_critic12d_balance",))
    differences = _diff(native, candidate)
    if differences:
        raise AssertionError("unexpected environment drift:\n" + "\n".join(differences[:50]))

    native_runner = copy.deepcopy(load_cfg_from_registry(NATIVE, "rsl_rl_cfg_entry_point").to_dict())
    candidate_runner = copy.deepcopy(load_cfg_from_registry(CANDIDATE, "rsl_rl_cfg_entry_point").to_dict())
    assert native_runner["num_steps_per_env"] == 24
    assert candidate_runner["num_steps_per_env"] == 24
    assert candidate_runner["max_iterations"] == 20000
    for key in ("experiment_name", "max_iterations"):
        native_runner.pop(key)
        candidate_runner.pop(key)
    differences = _diff(native_runner, candidate_runner)
    if differences:
        raise AssertionError("unexpected PPO drift:\n" + "\n".join(differences[:50]))
    print(
        "V13_CRITIC12D_REFERENCE_AUDIT: PASS "
        "env_delta=critic12d+route_coupled_ascent+rear_motion ppo_delta=limit20k+logroot",
        flush=True,
    )
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
