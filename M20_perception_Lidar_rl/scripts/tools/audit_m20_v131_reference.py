#!/usr/bin/env python3
"""Fail-closed V13.1 audit against the constructed V13 authority."""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher


V131_REWARDS = {
    "v131_b1_progress": 0.5,
    "v131_b1_velocity_coupled_progress": 1.5,
    "v131_expected_progress": 0.5,
    "v131_velocity_coupled_expected_progress": 1.5,
    "v131_same_side_progress": -0.75,
    "v131_simultaneous_progress": -0.5,
    "v131_simultaneous_b1_lift": -0.5,
    "v131_alternating_completion": 1.0,
    "v131_nonordered_completion": -0.5,
    "v131_skip_riser": -1.0,
    "v131_skip_progress": -1.0,
    "v131_top_follow_progress": 1.0,
    "v131_top_follow_completion": 3.0,
    "v131_post_top_front_lift": -0.5,
}
REPOINTED_REWARDS = {
    "action_rate_l2": (
        "v131_expected_swing_action_rate_l2",
        {"expected_hipy_scale": 0.35, "expected_knee_scale": 0.50},
    ),
    "action_smooth_l2": (
        "v131_expected_swing_action_smooth_l2",
        {"expected_hipy_scale": 0.50, "expected_knee_scale": 0.50},
    ),
}


def differences(current, expected, path="", limit=30):
    result = []
    if isinstance(current, dict) and isinstance(expected, dict):
        for key in sorted(set(current) | set(expected), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in current:
                result.append(f"{child}: missing-current; expected={expected[key]!r}")
            elif key not in expected:
                result.append(f"{child}: current-only={current[key]!r}")
            else:
                result.extend(differences(current[key], expected[key], child, limit))
            if len(result) >= limit:
                break
    elif isinstance(current, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(current) != len(expected):
            result.append(f"{path}: lengths current={len(current)} expected={len(expected)}")
        for index, (left, right) in enumerate(zip(current, expected)):
            result.extend(differences(left, right, f"{path}[{index}]", limit))
            if len(result) >= limit:
                break
    elif current != expected:
        result.append(f"{path}: current={current!r}; expected={expected!r}")
    return result[:limit]


def assert_equal(label, current, expected):
    if current != expected:
        raise AssertionError(label + "\n" + "\n".join(differences(current, expected)))


def run_audit(args) -> None:
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (
        DeeproboticsM20V13V11FDAlternationPPORunnerCfg,
        DeeproboticsM20V131RetentionPPORunnerCfg,
        DeeproboticsM20V131V11FDOrderedAlternationPPORunnerCfg,
    )
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v13_v11fd_alternation_env_cfg import (
        DeeproboticsM20V13V11FDAlternationEnvCfg,
    )
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v131_v11fd_ordered_alternation_env_cfg import (
        DeeproboticsM20V131V11FDOrderedAlternationEnvCfg,
        DeeproboticsM20V131RetentionEnvCfg,
    )

    if args.self_test_failure:
        raise AssertionError("intentional V13.1 audit failure-path self-test")

    base_env = DeeproboticsM20V13V11FDAlternationEnvCfg().to_dict()
    current_env = DeeproboticsM20V131V11FDOrderedAlternationEnvCfg().to_dict()
    base_agent = DeeproboticsM20V13V11FDAlternationPPORunnerCfg().to_dict()
    current_agent = DeeproboticsM20V131V11FDOrderedAlternationPPORunnerCfg().to_dict()
    retention_env = DeeproboticsM20V131RetentionEnvCfg().to_dict()
    retention_agent = DeeproboticsM20V131RetentionPPORunnerCfg().to_dict()

    for section in ("actions", "commands", "observations", "events", "terminations"):
        assert_equal(f"V13.1 drifted in frozen section {section}", current_env[section], base_env[section])
    current_scene = dict(current_env["scene"])
    current_scene["terrain"] = dict(current_scene["terrain"])
    if current_scene["terrain"]["max_init_terrain_level"] != 1:
        raise AssertionError("V13.1 from-scratch max_init_terrain_level must be 1.")
    current_scene["terrain"]["max_init_terrain_level"] = base_env["scene"]["terrain"]["max_init_terrain_level"]
    assert_equal("V13.1 scene changed outside initial terrain level", current_scene, base_env["scene"])
    current_perception = dict(current_env["v11_perception"])
    current_perception["soft_alternation_curriculum_gate"] = (
        base_env["v11_perception"]["soft_alternation_curriculum_gate"]
    )
    assert_equal("V13.1 changed V11 physical perception", current_perception, base_env["v11_perception"])

    base_rewards = dict(base_env["rewards"])
    current_rewards = dict(current_env["rewards"])
    for native_name in ("track_lin_vel_xy_exp", "world_forward_progress", "world_forward_deficit"):
        assert_equal(
            f"native reward changed: {native_name}",
            current_rewards[native_name], base_rewards[native_name],
        )
    for name, weight in V131_REWARDS.items():
        term = current_rewards.pop(name, None)
        if term is None:
            raise AssertionError(f"V13.1 reward is disconnected: {name}")
        assert_equal(f"{name}.weight", term["weight"], weight)
    for name in tuple(base_rewards):
        if name.startswith("v13_"):
            base_rewards.pop(name)
            if current_rewards.pop(name, None) is not None:
                raise AssertionError(f"permissive V13 reward remains active: {name}")
    for name, (expected_func_name, expected_extras) in REPOINTED_REWARDS.items():
        base_term = base_rewards.pop(name)
        current_term = current_rewards.pop(name)
        assert_equal(f"{name}.weight", current_term["weight"], base_term["weight"])
        func_name = getattr(current_term["func"], "__name__", str(current_term["func"]))
        if expected_func_name not in func_name:
            raise AssertionError(
                f"{name} did not route to {expected_func_name}: {func_name}"
            )
        current_params = dict(current_term["params"])
        for key, expected in expected_extras.items():
            assert_equal(f"{name}.params.{key}", current_params.pop(key, None), expected)
        assert_equal(f"{name} native params", current_params, base_term["params"])
    assert_equal("rewards outside ordered whitelist", current_rewards, base_rewards)

    base_curriculum = {
        name: term for name, term in base_env["curriculum"].items() if not name.startswith("v13_")
    }
    current_curriculum = dict(current_env["curriculum"])
    active_v131_metrics = {
        name for name, term in current_curriculum.items() if name.startswith("v131_") and term is not None
    }
    if len(active_v131_metrics) != 69:
        raise AssertionError(f"expected 69 active V13.1 metrics, got {sorted(active_v131_metrics)}")
    current_curriculum = {
        name: term for name, term in current_curriculum.items()
        if not name.startswith("v13_") and not name.startswith("v131_")
    }
    terrain_params = current_curriculum["terrain_levels"]["params"]
    if terrain_params.pop("strict_ordered_gate", None) is not True:
        raise AssertionError("V13.1 terrain promotion is not strict-ledger gated.")
    assert_equal("curriculum outside V13.1 metrics", current_curriculum, base_curriculum)

    for section in ("policy", "algorithm"):
        assert_equal(f"agent.{section}", current_agent[section], base_agent[section])
    for key, value in base_agent.items():
        if key == "experiment_name" or key in ("policy", "algorithm"):
            continue
        assert_equal(f"agent.{key}", current_agent[key], value)
    assert_equal(
        "agent.experiment_name", current_agent["experiment_name"],
        "M20-V13.1-V11FD-OrderedAlternation",
    )
    fresh_current_env = DeeproboticsM20V131V11FDOrderedAlternationEnvCfg().to_dict()
    for section in ("actions", "commands", "observations", "events", "terminations", "rewards"):
        assert_equal(
            f"retention task drifted from V13.1 in {section}",
            retention_env[section], fresh_current_env[section],
        )
    retention_terrain = retention_env["scene"]["terrain"]
    if retention_terrain["terrain_generator"]["curriculum"] is not False:
        raise AssertionError("retention terrain must be fixed, not curriculum driven.")
    sub_terrains = retention_terrain["terrain_generator"]["sub_terrains"]
    if sub_terrains["pyramid_stairs_inv"]["proportion"] != 1.0:
        raise AssertionError("retention terrain must contain only ascending stairs.")
    if tuple(sub_terrains["pyramid_stairs_inv"]["step_height_range"]) != (0.12, 0.12):
        raise AssertionError("retention stair height must be fixed at the medium fixture.")
    if any(
        value["proportion"] != 0.0
        for name, value in sub_terrains.items()
        if name != "pyramid_stairs_inv"
    ):
        raise AssertionError("retention terrain leaked a non-ascending terrain type.")
    if retention_env["curriculum"]["terrain_levels"] is not None:
        raise AssertionError("retention test must not update terrain levels.")
    for section in ("policy", "algorithm"):
        assert_equal(
            f"retention agent changed {section}",
            retention_agent[section], current_agent[section],
        )
    if retention_agent["max_iterations"] != 400:
        raise AssertionError("retention test must default to 400 iterations.")
    if retention_agent["require_v131_retention_actor"] is not True:
        raise AssertionError("retention test must fail closed without its actor.")

    print("V131_STRICT_V13_REFERENCE_AUDIT: PASS", flush=True)
    print("FROZEN=actions,commands,observations,events,terminations,scene_except_init_level,v11_perception,ppo,network", flush=True)
    print("REPOINTED=" + ",".join(sorted(REPOINTED_REWARDS)), flush=True)
    print("V131_REWARDS=" + ",".join(sorted(V131_REWARDS)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test-failure", action="store_true", help=argparse.SUPPRESS)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app
    try:
        run_audit(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()


if __name__ == "__main__":
    main()
