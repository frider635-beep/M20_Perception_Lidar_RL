#!/usr/bin/env python3
"""Fail closed unless V13.4 differs from V13 only by the reviewed allow-list."""

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

BASE = "M20-Stair-V13-V11FD-Alternation-v0"
CANDIDATE = "M20-Stair-V13.4-V11FD-PerceptiveFollow-v0"


def _remove(mapping, path):
    node = mapping
    for key in path[:-1]:
        node = node[key]
    if path[-1] not in node:
        raise AssertionError(f"missing allow-listed path: {'.'.join(path)}")
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
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            out.extend(_diff(a, b, f"{path}[{i}]"))
        return out
    try:
        if isinstance(left == right, bool) and left == right:
            return []
    except Exception:
        pass
    return [] if repr(left) == repr(right) else [f"{path}: {left!r} != {right!r}"]


def main() -> None:
    base = copy.deepcopy(parse_env_cfg(BASE, device=args.device, num_envs=8).to_dict())
    candidate = copy.deepcopy(parse_env_cfg(CANDIDATE, device=args.device, num_envs=8).to_dict())

    ledger = candidate["v133_transition"]
    expected = {
        "require_first_riser_seed": True,
        "b1_settlement": 1.0,
        "adjacent_transition_settlement": 2.0,
        "chain_settlement": 6.0,
        "adjacent_progress_weight": 0.5,
        "repeat_transition_cost": 0.0,
        "skip_transition_cost": 0.0,
        "simultaneous_transition_cost": 0.0,
        "rear_thigh_parallel_free_cosine": 0.75,
        "rear_thigh_parallel_scale": 0.20,
        "rear_thigh_parallel_max_penalty": 1.0,
        "rear_thigh_parallel_weight": 0.50,
        "platform_target_loss_release_steps": 30,
        "platform_body_clearance_free_height": 0.38,
        "platform_body_clearance_scale": 0.10,
        "platform_body_clearance_max_penalty": 1.0,
        "platform_body_clearance_weight": 1.0,
    }
    for key, value in expected.items():
        if ledger[key] != value:
            raise AssertionError(f"V13.4 {key}={ledger[key]!r}, expected {value!r}")

    _remove(candidate, ("v133_transition",))
    _remove(candidate, ("observations", "critic", "v134_task_state"))
    if candidate["v11_perception"]["platform_target_loss_release_steps"] != 30:
        raise AssertionError("V13.4 platform target-loss hold must be 30 control frames.")
    candidate["v11_perception"]["platform_target_loss_release_steps"] = (
        base["v11_perception"]["platform_target_loss_release_steps"]
    )
    for name in (
        "v134_ascent_chain", "v134_descent_rear_extension",
        "v134_descent_rear_knee_flexion", "v134_descent_rear_roll",
        "v134_descent_rear_thigh_parallel", "v134_platform_body_clearance",
    ):
        _remove(candidate, ("rewards", name))
    for name in list(candidate["curriculum"]):
        if name.startswith("v134_"):
            _remove(candidate, ("curriculum", name))
    differences = _diff(base, candidate)
    if differences:
        raise AssertionError("unexpected V13.4 environment drift:\n" + "\n".join(differences[:50]))

    base_runner = copy.deepcopy(load_cfg_from_registry(BASE, "rsl_rl_cfg_entry_point").to_dict())
    candidate_runner = copy.deepcopy(load_cfg_from_registry(CANDIDATE, "rsl_rl_cfg_entry_point").to_dict())
    if candidate_runner["num_steps_per_env"] != 24:
        raise AssertionError("V13.4 rollout must be 24.")
    if candidate_runner["max_iterations"] != 30000:
        raise AssertionError("V13.4 run must be 30000 iterations.")
    if candidate_runner["algorithm"]["entropy_coef"] != 1.0e-3:
        raise AssertionError("V13.4 style retention must use V13's entropy coefficient.")
    if candidate_runner["algorithm"]["learning_rate"] != 1.0e-4:
        raise AssertionError("V13.4 style retention must use the reviewed fine-tuning rate.")
    if candidate_runner["class_name"] != "rl_training.v134_style_prior_runner:V134StylePriorRunner":
        raise AssertionError("V13.4 must use the ascent-only teacher runner.")
    if candidate_runner["algorithm"]["class_name"] != "rl_training.v134_style_prior_runner:V134StylePriorPPO":
        raise AssertionError("V13.4 must use the ascent-teacher PPO wrapper.")
    if candidate_runner.pop("require_v134_style_actor") is not True:
        raise AssertionError("V13.4 must fail closed without the reviewed actor.")
    expected_prior = {
        "v134_teacher_burst_start_probability": 0.025,
        "v134_teacher_burst_end_probability": 0.005,
        "v134_teacher_burst_decay_updates": 6000,
        "v134_teacher_burst_steps": 8,
    }
    for key, expected_value in expected_prior.items():
        actual = candidate_runner.pop(key)
        if actual != expected_value:
            raise AssertionError(f"V13.4 {key}={actual!r}, expected {expected_value!r}.")
    for key in ("experiment_name", "num_steps_per_env", "max_iterations", "save_interval"):
        base_runner[key] = candidate_runner[key]
    base_runner["algorithm"]["entropy_coef"] = candidate_runner["algorithm"]["entropy_coef"]
    base_runner["algorithm"]["learning_rate"] = candidate_runner["algorithm"]["learning_rate"]
    base_runner["algorithm"]["class_name"] = candidate_runner["algorithm"]["class_name"]
    base_runner["class_name"] = candidate_runner["class_name"]
    differences = _diff(base_runner, candidate_runner)
    if differences:
        raise AssertionError("unexpected V13.4 PPO drift:\n" + "\n".join(differences[:50]))

    print(
        "V134_REFERENCE_AUDIT: PASS base=V13 "
        "delta=fresh_student+sparse_model5000_ascent_rollouts+reward_driven_student+separate_descent_top+lockedB1+early_platform_posture+rollout24",
        flush=True,
    )
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
