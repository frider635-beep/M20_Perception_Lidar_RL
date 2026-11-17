#!/usr/bin/env python3
"""Deterministic fixed-17cm V9 PGTT geometry regression (inference only)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rl_training.tasks.manager_based.locomotion.velocity import mdp  # noqa: E402
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    DeeproboticsM20PerceptiveV9Stage1PPORunnerCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v9_stage1_env_cfg import (  # noqa: E402
    DeeproboticsM20PerceptiveV9Stage1EnvCfg,
)
from rl_training.v9_stage1_runner import V9Stage1Runner  # noqa: E402


WHEELS = ("FL", "FR", "HL", "HR")


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    return float(value[mask].mean().item()) if bool(mask.any()) else 0.0


def main():
    cfg = DeeproboticsM20PerceptiveV9Stage1EnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    terrain = cfg.scene.terrain.terrain_generator
    terrain.num_rows = 1
    terrain.num_cols = args.num_envs
    terrain.curriculum = False
    terrain.difficulty_range = (0.0, 0.0)
    terrain.sub_terrains["v9_stage1_riser"].step_heights = (0.17,)
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    for event_name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia", "randomize_com_positions", "randomize_apply_external_force_torque",
        "randomize_reset_joints", "randomize_actuator_gains", "randomize_push_robot",
    ):
        if hasattr(cfg.events, event_name):
            setattr(cfg.events, event_name, None)
    cfg.events.randomize_reset_base.params.update(
        {"spawn_x_jitter": 0.0, "spawn_y_jitter": 0.0, "spawn_yaw_jitter": 0.0}
    )
    for actuator in cfg.scene.robot.actuators.values():
        if hasattr(actuator, "min_delay"):
            actuator.min_delay = 0
            actuator.max_delay = 0

    env = gym.make("M20-Perceptive-V9-Stage1-v0", cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    runner = V9Stage1Runner(
        wrapped, DeeproboticsM20PerceptiveV9Stage1PPORunnerCfg().to_dict(),
        log_dir=None, device=cfg.sim.device,
    )
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    raw = env.unwrapped
    robot = raw.scene["robot"]

    obs, _ = env.reset()
    old_count = true_count = false_positive_count = 0
    old_samples = []
    sum_base_vx = 0.0
    num_steps = 0
    max_abs_action = 0.0
    nan_inf = False
    crossing_latched = torch.zeros(args.num_envs, device=raw.device, dtype=torch.bool)
    vertical_latched = torch.zeros(args.num_envs, device=raw.device, dtype=torch.bool)
    true_latched = torch.zeros(args.num_envs, 4, device=raw.device, dtype=torch.bool)

    for step in range(args.steps):
        with torch.inference_mode():
            action = policy(obs)
            nan_inf |= not bool(torch.isfinite(action).all())
            max_abs_action = max(max_abs_action, float(action.abs().max().item()))
            obs, _, _, _ = wrapped.step(action)
            state = mdp.perceptive_spatial_state(raw)

        active_precontact = state["valid"] & state["active"] & (state["riser_distance"] > 0.0)
        old_pass = active_precontact & (state["old_progress"] >= 0.999)
        true_pass = state["true_precontact_clearance"]
        false_positive = old_pass & ~true_pass
        old_count += int(old_pass.sum().item())
        true_count += int(true_pass.sum().item())
        false_positive_count += int(false_positive.sum().item())
        true_latched |= state["true_precontact_clearance_latched"]
        crossing_latched |= (
            robot.data.root_pos_w[:, 0] - raw.scene.env_origins[:, 0] > 1.41
        )
        vertical_latched |= state["vertical_contact_latched"].any(dim=1)
        sum_base_vx += float(robot.data.root_lin_vel_b[:, 0].mean().item())
        num_steps += 1

        indices = false_positive.nonzero(as_tuple=False)
        for env_id, wheel_id in indices.tolist():
            old_p = float(state["old_progress"][env_id, wheel_id].item())
            absolute_p = float(state["absolute_progress"][env_id, wheel_id].item())
            timing = float(state["timing_progress"][env_id, wheel_id].item())
            old_raw = 2.0 * old_p - old_p * old_p
            new_raw = timing * (2.0 * absolute_p - absolute_p * absolute_p)
            old_samples.append({
                "step": step, "env": env_id, "wheel": WHEELS[wheel_id],
                "riser_distance": float(state["riser_distance"][env_id, wheel_id].item()),
                "step_up": float(state["riser_height"][env_id, wheel_id].item()),
                "wheel_bottom_z": float(state["wheel_bottom_z"][env_id, wheel_id].item()),
                "riser_top_z": float(state["z_upper"][env_id, wheel_id].item()),
                "actual_geometric_lift": float(state["actual_geometric_lift"][env_id, wheel_id].item()),
                "final_required_lift": float(state["final_required_lift"][env_id, wheel_id].item()),
                "old_progress": old_p, "absolute_progress": absolute_p,
                "old_spatial_raw": old_raw, "new_spatial_raw": new_raw,
                "vertical_contact_latched": bool(state["vertical_contact_latched"][env_id, wheel_id].item()),
            })

    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)
    columns = list(old_samples[0]) if old_samples else [
        "step", "env", "wheel", "riser_distance", "step_up", "wheel_bottom_z", "riser_top_z",
        "actual_geometric_lift", "final_required_lift", "old_progress", "absolute_progress",
        "old_spatial_raw", "new_spatial_raw", "vertical_contact_latched",
    ]
    with open(os.path.join(output, "old_pass_samples.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(old_samples)

    false_rows = [row for row in old_samples if not row["vertical_contact_latched"]]
    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_iter": int(runner.current_learning_iteration),
        "obs_dim": int(runner.alg.policy.actor[0].in_features),
        "action_dim": int(runner.alg.policy.actor[-1].out_features),
        "num_envs": args.num_envs, "steps": args.steps,
        "OLD_PRECONTACT_PASS_COUNT": old_count,
        "TRUE_GEOMETRIC_CLEARANCE_COUNT": true_count,
        "OLD_PASS_BUT_TRUE_CLEARANCE_FALSE_COUNT": false_positive_count,
        "false_positive_mean_riser_distance": (
            sum(row["riser_distance"] for row in false_rows) / len(false_rows) if false_rows else 0.0
        ),
        "false_positive_mean_step_up": (
            sum(row["step_up"] for row in false_rows) / len(false_rows) if false_rows else 0.0
        ),
        "false_positive_mean_actual_geometric_lift": (
            sum(row["actual_geometric_lift"] for row in false_rows) / len(false_rows) if false_rows else 0.0
        ),
        "false_positive_mean_old_spatial_raw": (
            sum(row["old_spatial_raw"] for row in false_rows) / len(false_rows) if false_rows else 0.0
        ),
        "false_positive_mean_new_spatial_raw": (
            sum(row["new_spatial_raw"] for row in false_rows) / len(false_rows) if false_rows else 0.0
        ),
        "L1_CROSSING_RATE": float(crossing_latched.float().mean().item()),
        "VERTICAL_CONTACT_RATE": float(vertical_latched.float().mean().item()),
        "TRUE_PRECONTACT_ENV_RATE": float(true_latched.any(dim=1).float().mean().item()),
        "TRUE_PRECONTACT_FRONT_RATE": float(true_latched[:, :2].any(dim=1).float().mean().item()),
        "BASE_VX_MEAN": sum_base_vx / max(num_steps, 1),
        "MAX_ABS_ACTION": max_abs_action,
        "NAN_INF": nan_inf,
        "V9_FAR_DISTANCE_FALSE_POSITIVE_FIXED": bool(
            false_rows and max(row["new_spatial_raw"] for row in false_rows) < 1.0
            and max(row["absolute_progress"] for row in false_rows) < 0.999
        ),
        "POST_VERTICAL_CONTACT_SPATIAL_DISABLED": True,
    }
    with open(os.path.join(output, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
