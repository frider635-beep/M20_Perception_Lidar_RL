#!/usr/bin/env python3
"""Fail-closed config audit against the named 2026-09-02 V11-FD run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch/params"
V13_REWARDS = {
    "v13_opposite_progress": 0.5,
    "v13_velocity_coupled_opposite_progress": 1.5,
    "v13_same_side_progress": -0.75,
    "v13_simultaneous_progress": -0.5,
    "v13_alternating_completion": 1.0,
    "v13_same_side_completion": -0.5,
}
V13_REWARD_FUNCTIONS = {
    "v13_opposite_progress": "v13_opposite_progress_reward",
    "v13_velocity_coupled_opposite_progress": "v13_velocity_coupled_opposite_progress_reward",
    "v13_same_side_progress": "v13_same_side_progress_reward",
    "v13_simultaneous_progress": "v13_simultaneous_progress_reward",
    "v13_alternating_completion": "v13_alternating_completion_reward",
    "v13_same_side_completion": "v13_same_side_completion_reward",
}
V13_CURRICULUM = {
    "v13_potential_active_rate",
    "v13_opposite_progress_delta_mean",
    "v13_same_side_progress_delta_mean",
    "v13_simultaneous_progress_delta_mean",
    "v13_opposite_potential_mean",
    "v13_seed_event_rate",
    "v13_compliant_event_rate",
    "v13_violation_event_rate",
    "v13_simultaneous_completion_event_rate",
    "v13_seed_fl_fraction",
    "v13_natural_compliance",
    "v13_progress_precision",
    "v13_simultaneous_progress_fraction",
}


def describe_differences(current, saved, path="", limit=20):
    """Return a bounded, recursive diff so a failed audit is actionable."""
    differences = []
    if isinstance(current, dict) and isinstance(saved, dict):
        for key in sorted(set(current) | set(saved), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in current:
                differences.append(f"{child_path}: missing-current; saved={saved[key]!r}")
            elif key not in saved:
                differences.append(f"{child_path}: current-only={current[key]!r}")
            else:
                differences.extend(describe_differences(current[key], saved[key], child_path, limit))
            if len(differences) >= limit:
                break
    elif isinstance(current, (list, tuple)) and isinstance(saved, (list, tuple)):
        if len(current) != len(saved):
            differences.append(f"{path}: lengths current={len(current)} saved={len(saved)}")
        for index, (current_item, saved_item) in enumerate(zip(current, saved)):
            differences.extend(describe_differences(current_item, saved_item, f"{path}[{index}]", limit))
            if len(differences) >= limit:
                break
    elif current != saved:
        differences.append(f"{path}: current={current!r}; saved={saved!r}")
    return differences[:limit]


def assert_equal(label: str, current, saved) -> None:
    if current != saved:
        details = "\n".join(describe_differences(current, saved))
        raise AssertionError(f"V13 drifted from saved V11-FD section: {label}\n{details}")


def run_audit(args) -> None:
    """Construct the real configs and prove that only the declared V13 delta exists."""
    import yaml

    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (
        DeeproboticsM20V13V11FDAlternationPPORunnerCfg,
    )
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v13_v11fd_alternation_env_cfg import (
        DeeproboticsM20V13V11FDAlternationEnvCfg,
    )

    def load(name: str):
        path = REFERENCE / name
        if not path.is_file():
            raise RuntimeError(f"V13 reference config is missing: {path}")
        return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)

    if args.self_test_failure:
        raise AssertionError("intentional V13 audit failure-path self-test")

    saved_env, saved_agent = load("env.yaml"), load("agent.yaml")
    current_env = DeeproboticsM20V13V11FDAlternationEnvCfg().to_dict()
    current_agent = DeeproboticsM20V13V11FDAlternationPPORunnerCfg().to_dict()

    if args.show_all_diffs:
        print("V13_RAW_ENV_DIFF_BEGIN", flush=True)
        print("\n".join(describe_differences(current_env, saved_env, limit=200)), flush=True)
        print("V13_RAW_ENV_DIFF_END", flush=True)

    # Full behavioral managers: V13 changes no action, command, observation,
    # event or termination manager entry.
    for section in ("actions", "commands", "observations", "events", "terminations"):
        assert_equal(section, current_env[section], saved_env[section])

    current_rewards = dict(current_env["rewards"])
    for name, expected_weight in V13_REWARDS.items():
        term = current_rewards.pop(name, None)
        if term is None:
            raise AssertionError(f"V13 reward is not connected: {name}")
        assert_equal(f"{name}.weight", term["weight"], expected_weight)
        actual_function = str(term["func"]).rsplit(":", 1)[-1]
        assert_equal(f"{name}.func", actual_function, V13_REWARD_FUNCTIONS[name])
    assert_equal(
        "velocity-coupling.params",
        current_env["rewards"]["v13_velocity_coupled_opposite_progress"]["params"],
        {"command_name": "base_velocity", "std": 0.7071067811865476},
    )
    # Preserve the saved generic completion term's +5 weight and params, but
    # repoint it from bilateral-same-riser completion to every safe one-leg /
    # one-distinct-riser completion. This is the explicit V13 gait-unit delta.
    landing_term = current_rewards["landing_event"]
    assert_equal("landing_event.weight", landing_term["weight"], saved_env["rewards"]["landing_event"]["weight"])
    assert_equal("landing_event.params", landing_term["params"], saved_env["rewards"]["landing_event"]["params"])
    assert_equal(
        "landing_event.func",
        str(landing_term["func"]).rsplit(":", 1)[-1],
        "v13_any_front_riser_completion_reward",
    )
    landing_term["func"] = saved_env["rewards"]["landing_event"]["func"]
    center_term = current_rewards["landing_center"]
    assert_equal("landing_center.weight", center_term["weight"], saved_env["rewards"]["landing_center"]["weight"])
    assert_equal("landing_center.params", center_term["params"], saved_env["rewards"]["landing_center"]["params"])
    assert_equal(
        "landing_center.func",
        str(center_term["func"]).rsplit(":", 1)[-1],
        "v13_any_front_riser_center_reward",
    )
    center_term["func"] = saved_env["rewards"]["landing_center"]["func"]
    # These terms were added to the V11 source after the reference run. V13
    # forces them to None before comparing with the older snapshot.
    for name in ("safe_front_landing_progress", "single_front_hard_landing", "descent_rear_support_progress"):
        if name not in saved_env["rewards"]:
            if current_rewards.get(name, "missing") is not None:
                raise AssertionError(f"post-run reward must stay disabled: {name}")
            current_rewards.pop(name, None)
    assert_equal("rewards-minus-v13", current_rewards, saved_env["rewards"])

    current_curriculum = dict(current_env["curriculum"])
    v13_curriculum = {name for name in current_curriculum if name.startswith("v13_")}
    assert_equal("V13 diagnostic set", v13_curriculum, V13_CURRICULUM)
    for name in v13_curriculum:
        current_curriculum.pop(name)
    for name in (
        "v11_descent_rear_support_event_rate",
        "v11_fd_safe_support_event_rate",
        "v11_fd_hard_candidate_event_rate",
    ):
        if name not in saved_env["curriculum"]:
            if current_curriculum.get(name, "missing") is not None:
                raise AssertionError(f"post-run metric must stay disabled: {name}")
            current_curriculum.pop(name, None)
    assert_equal("curriculum-minus-v13", current_curriculum, saved_env["curriculum"])

    # Compare every saved perception field. The later compatibility-only
    # persistent descent path must stay disabled to match this run.
    for key, value in saved_env["v11_perception"].items():
        if key == "front_completion_mode":
            continue
        assert_equal(f"v11_perception.{key}", current_env["v11_perception"][key], value)
    assert_equal(
        "v11_perception.front_completion_mode",
        current_env["v11_perception"]["front_completion_mode"],
        "alternating_risers",
    )
    assert_equal("v11_perception.soft_alternation", current_env["v11_perception"]["soft_alternation"], True)
    assert_equal(
        "v11_perception.soft_alternation_command_split",
        current_env["v11_perception"]["soft_alternation_command_split"],
        False,
    )
    assert_equal(
        "v11_perception.soft_alternation_curriculum_gate",
        current_env["v11_perception"]["soft_alternation_curriculum_gate"],
        False,
    )
    if current_env["v11_perception"].get("enable_persistent_descent_rear_phase") is not False:
        raise AssertionError("V13 must retain the saved non-persistent descent lifecycle")

    assert_equal(
        "terrain_generator",
        current_env["scene"]["terrain"]["terrain_generator"],
        saved_env["scene"]["terrain"]["terrain_generator"],
    )
    assert_equal(
        "max_init_terrain_level",
        current_env["scene"]["terrain"]["max_init_terrain_level"],
        saved_env["scene"]["terrain"]["max_init_terrain_level"],
    )

    # V13 keeps the saved model and PPO algorithm byte-for-byte at config level.
    for section in ("policy", "algorithm"):
        for key, value in saved_agent[section].items():
            assert_equal(f"agent.{section}.{key}", current_agent[section][key], value)
    for key in (
        "num_steps_per_env", "max_iterations", "save_interval", "resume", "load_run", "load_checkpoint",
    ):
        assert_equal(f"agent.{key}", current_agent[key], saved_agent[key])

    print("V13_STRICT_V11FD_REFERENCE_AUDIT: PASS", flush=True)
    print(f"REFERENCE_ENV={REFERENCE / 'env.yaml'}", flush=True)
    print(f"REFERENCE_AGENT={REFERENCE / 'agent.yaml'}", flush=True)
    print("V13_REWARD_DELTA=" + ",".join(sorted(V13_REWARDS)), flush=True)
    print("V13_COMPLETION_DELTA=one-safe-front-side-per-distinct-riser,soft-style", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-all-diffs", action="store_true")
    parser.add_argument("--self-test-failure", action="store_true", help=argparse.SUPPRESS)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    simulation_app = AppLauncher(args).app
    try:
        run_audit(args)
    except BaseException:
        traceback.print_exc()
        # Kit's close path can terminate the process with status 0 before a
        # surrounding finally block runs. Flush and exit nonzero first; the OS
        # releases this short-lived audit process's simulator resources.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()


if __name__ == "__main__":
    main()
