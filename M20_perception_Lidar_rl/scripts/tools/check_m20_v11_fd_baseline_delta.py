#!/usr/bin/env python3
"""Fail-closed audit of a saved V11-FD run against the 2026-08-27 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "logs/rsl_rl/M20-V11-Continuous-Landing/2026-08-27_18-14-26_v11_from_scratch"
ALLOWED_REWARD_DIFFS = {
    "descent_rear_follow_delay",
    "descent_rear_roll_consistency",
    "descent_step_complete",
    "feet_air_time_ang_z_M20",
    "flat_orientation_l2",
    "lin_vel_z_l2",
    "pure_yaw_xy_drift",
    "rotation_gait_status",
    "rotation_gait_symmetry",
    "safe_front_landing_progress",
    "single_front_hard_landing",
}
EXPECTED_TERRAIN = {
    "flat": 0.20,
    "pyramid_stairs": 0.20,
    "pyramid_stairs_inv": 0.30,
    "boxes": 0.10,
    "random_rough": 0.10,
    "hf_pyramid_slope": 0.05,
    "hf_pyramid_slope_inv": 0.05,
}


def load(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader)


def reward_signature(cfg):
    return {
        name: None if term is None else (term.get("weight"), term.get("func"), term.get("params"))
        for name, term in cfg["rewards"].items()
    }


parser = argparse.ArgumentParser()
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
run = args.run_dir.resolve()
base_env, fd_env = load(BASELINE / "params/env.yaml"), load(run / "params/env.yaml")
base_agent, fd_agent = load(BASELINE / "params/agent.yaml"), load(run / "params/agent.yaml")

assert base_env["commands"] == fd_env["commands"], "V11-FD command distribution drifted from baseline"
base_rewards, fd_rewards = reward_signature(base_env), reward_signature(fd_env)
actual_reward_diffs = {
    name for name in set(base_rewards) | set(fd_rewards)
    if base_rewards.get(name) != fd_rewards.get(name)
}
assert actual_reward_diffs == ALLOWED_REWARD_DIFFS, (
    f"unexpected reward delta: missing={sorted(ALLOWED_REWARD_DIFFS - actual_reward_diffs)}, "
    f"extra={sorted(actual_reward_diffs - ALLOWED_REWARD_DIFFS)}"
)

terrain = fd_env["scene"]["terrain"]["terrain_generator"]["sub_terrains"]
actual_terrain = {name: float(item["proportion"]) for name, item in terrain.items()}
assert actual_terrain == EXPECTED_TERRAIN, f"terrain mix mismatch: {actual_terrain}"
assert fd_env["scene"]["terrain"]["max_init_terrain_level"] == 1

cfg = fd_env["v11_perception"]
assert cfg["front_completion_mode"] == "bilateral_same_riser"
assert cfg["step_feature_mode"] == "signed_edges"
assert cfg["feature_contract"] == "v11_signed_edges_v2"
assert cfg["require_clearance_at_crossing"] is True
assert abs(cfg["wheel_radius"] + cfg["clearance_margin"] - 0.14) < 1.0e-9

assert fd_agent["seed"] == 42
assert fd_agent["experiment_name"] == "M20-V11-Flat-Descent"
assert fd_agent["save_interval"] == 100
assert fd_agent["resume"] is False
assert fd_agent["load_run"] is None and fd_agent["load_checkpoint"] is None
assert fd_agent["policy"] == base_agent["policy"]
assert fd_agent["algorithm"] == base_agent["algorithm"]

print("V11_FD_BASELINE_DELTA_AUDIT: PASS")
print(f"allowed_reward_diffs={','.join(sorted(actual_reward_diffs))}")
