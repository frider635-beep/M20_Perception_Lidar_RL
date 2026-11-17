#!/usr/bin/env python3
"""Offline event-weight calibration using a 57-D blind locomotion actor."""

import argparse
import csv
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--envs_per_height", type=int, default=8)
parser.add_argument("--steps", type=int, default=1100)
parser.add_argument("--step_height", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import torch.nn as nn

import rl_training.tasks  # noqa: F401,E402
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.hr_bridge_env_cfg import (  # noqa: E402
    DeeproboticsM20HRBridgeEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.hr_bridge_terrains import (  # noqa: E402
    make_hr_bridge_fixed_terrain,
)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.hr_bridge import HRBridgeTracker, STATE_HR_BRIDGE  # noqa: E402

HEIGHTS = (args.step_height,) if args.step_height is not None else (0.05, 0.08, 0.11, 0.14, 0.17, 0.20)


def actor_network():
    return nn.Sequential(
        nn.Linear(57, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16),
    )


def configure():
    cfg = DeeproboticsM20HRBridgeEnvCfg()
    cfg.seed = 241
    cfg.scene.num_envs = args.envs_per_height * len(HEIGHTS)
    cfg.scene.terrain.terrain_generator = make_hr_bridge_fixed_terrain(
        HEIGHTS, args.envs_per_height, 1.10
    )
    cfg.scene.terrain.max_init_terrain_level = len(HEIGHTS) - 1
    cfg.curriculum.terrain_levels = None
    cfg.terminations.terrain_out_of_bounds = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
    for name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia", "randomize_com_positions", "randomize_apply_external_force_torque",
        "randomize_reset_joints", "randomize_actuator_gains", "randomize_push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events.randomize_reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    return cfg


def median(values, default=0.0):
    if not values:
        return default
    return float(torch.tensor(values).median())


def main():
    cfg = configure()
    env = gym.make("M20-Stair-HRBridge-v0", cfg=cfg)
    raw = env.unwrapped
    actor = actor_network().to(raw.device).eval()
    loaded = torch.load(os.path.abspath(args.checkpoint), map_location=raw.device, weights_only=False)
    state = loaded["actor_state_dict"]
    actor.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")})
    tracker = HRBridgeTracker(raw)
    obs, _ = env.reset()
    n = raw.num_envs
    returns = torch.zeros(n, device=raw.device)
    s1_returns = torch.zeros_like(returns)
    success_latched = torch.zeros(n, dtype=torch.bool, device=raw.device)
    rows = []
    stair_cfg = cfg.scene.terrain.terrain_generator.sub_terrains["hr_bridge_fixed_stair"]
    success_x = stair_cfg.first_riser_from_spawn + stair_cfg.levels * stair_cfg.step_depth

    for step_index in range(args.steps):
        with torch.inference_mode():
            policy_obs = obs["policy"]
            action = actor(policy_obs[:, :57])
        with torch.no_grad():
            controller_state, riser, activation = tracker.pre_action_state()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            step = tracker.post_step(controller_state, riser, activation)
        if step_index < 3:
            term_counts = {
                name: int(raw.termination_manager.get_term(name).sum())
                for name in raw.termination_manager.active_terms
            }
            print(
                f"CALIBRATION_DIAGNOSTIC step={step_index} action_max={float(action.abs().max()):.4f} "
                f"terminated={int(terminated.sum())} truncated={int(truncated.sum())} terms={term_counts}",
                flush=True,
            )
        returns += reward
        s1_returns += reward * (controller_state == STATE_HR_BRIDGE)
        root_progress = raw.scene["robot"].data.root_pos_w[:, 0] - raw.scene.env_origins[:, 0]
        stable = raw.scene["robot"].data.projected_gravity_b[:, 2] < -0.65
        success_latched |= (root_progress > success_x) & stable
        for env_id in done.nonzero(as_tuple=False).flatten().tolist():
            level = int(raw.scene.terrain.terrain_levels[env_id].clamp(0, len(HEIGHTS) - 1))
            rows.append({
                "env": env_id,
                "height_m": HEIGHTS[level],
                "rough_return": float(returns[env_id]),
                "s1_rough_return": float(s1_returns[env_id]),
                "full_stair_success": int(success_latched[env_id]),
                "perception_miss": int(tracker.miss_latched[env_id]),
            })
        returns[done] = 0.0
        s1_returns[done] = 0.0
        success_latched[done] = False
        tracker.reset(done)

    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)
    with open(os.path.join(output, "calibration_episodes.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
            "env", "height_m", "rough_return", "s1_rough_return", "full_stair_success", "perception_miss"
        ])
        writer.writeheader(); writer.writerows(rows)
    success_returns = [row["rough_return"] for row in rows if row["full_stair_success"]]
    failure_returns = [row["rough_return"] for row in rows if not row["full_stair_success"]]
    s1_scale = median([abs(row["s1_rough_return"]) for row in rows], 1.0)
    return_gap = abs(median(success_returns) - median(failure_returns))
    # Base and Bridge optimize separate returns. Summing them for calibration
    # double-counts JOINT_SUCCESS and forces event weights to exceed the whole
    # official success/failure gap.
    hr_weight = max(s1_scale, 1.0)
    miss_weight = 2.0 * hr_weight
    joint_weight = max(6.0 * hr_weight, 0.20 * max(return_gap, 1.0))
    success_median = median(success_returns)
    failure_median = median(failure_returns)
    diagnostic_scores = {
        # Count every environment event once. These values are for reporting,
        # not a replacement for the two PPO returns.
        "score_A_hr_and_stair_success": success_median + hr_weight + joint_weight,
        "score_B_hr_success_stair_fail": failure_median + hr_weight,
        "score_C_perception_miss_blind_success": success_median - miss_weight,
        "score_D_both_fail": failure_median,
    }
    recommendation = {
        "episodes": len(rows),
        "success_episodes": len(success_returns),
        "failure_episodes": len(failure_returns),
        "median_success_return": success_median,
        "median_failure_return": failure_median,
        "median_abs_s1_return": s1_scale,
        "return_gap": return_gap,
        "HR_SUCCESS_WEIGHT": hr_weight,
        "PERCEPTION_MISS_WEIGHT": miss_weight,
        "JOINT_SUCCESS_WEIGHT": joint_weight,
        "base_A_return": success_median + joint_weight,
        "bridge_A_return": hr_weight + joint_weight,
        "base_B_return": failure_median,
        "bridge_B_return": hr_weight,
        "base_C_return": success_median,
        "bridge_C_return": -miss_weight,
        "base_D_return": failure_median,
        "bridge_D_return": 0.0,
        **diagnostic_scores,
        "SCORE_A_HIGHEST": diagnostic_scores["score_A_hr_and_stair_success"]
        > max(
            diagnostic_scores["score_B_hr_success_stair_fail"],
            diagnostic_scores["score_C_perception_miss_blind_success"],
            diagnostic_scores["score_D_both_fail"],
        ),
    }
    with open(os.path.join(output, "calibration_report.json"), "w") as handle:
        json.dump(recommendation, handle, indent=2)
    with open(os.path.join(output, "calibration_report.md"), "w") as handle:
        handle.write("# M20 HR Bridge Reward Calibration\n\n")
        for key, value in recommendation.items():
            handle.write(f"- {key}: {value}\n")
    print(json.dumps(recommendation, indent=2))
    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
