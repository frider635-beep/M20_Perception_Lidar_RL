#!/usr/bin/env python3
"""Calibrate V10 sparse event weights against official M20 reward scale."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", default="eval_outputs/v10_reward_calibration")
parser.add_argument("--envs_per_height", type=int, default=4)
parser.add_argument("--steps", type=int, default=800)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import torch.nn as nn

import rl_training.tasks  # noqa: F401
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v10_unified_terrains import (
    V10StepTerrainCfg,
    v10_step_terrain,
)


HEIGHTS = (0.05, 0.08, 0.11, 0.14, 0.17, 0.20)


class BlindActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(57, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16),
        )

    def forward(self, observation):
        return self.mlp(observation)


def disable_randomization(cfg):
    for name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia", "randomize_com_positions", "randomize_apply_external_force_torque",
        "randomize_reset_joints", "randomize_actuator_gains", "randomize_push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def make_terrain(envs_per_height):
    terrains = OrderedDict()
    for height in HEIGHTS:
        terrains[f"full_{int(height * 100):02d}cm"] = V10StepTerrainCfg(
            function=v10_step_terrain, proportion=1.0 / len(HEIGHTS), full_staircase=True, height_override=height,
        )
    return TerrainGeneratorCfg(
        seed=1012, curriculum=True, difficulty_range=(0.0, 0.0), size=(8.0, 8.0), border_width=20.0,
        num_rows=1, num_cols=len(HEIGHTS) * envs_per_height, horizontal_scale=0.1, vertical_scale=0.005,
        slope_threshold=0.75, color_scheme="height", use_cache=False, sub_terrains=terrains,
    )


def main():
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_envs = len(HEIGHTS) * args.envs_per_height
    cfg = parse_env_cfg("M20-Stair-Unified-Perceptive-v0", device="cuda:0", num_envs=num_envs)
    disable_randomization(cfg)
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.terrain_generator = make_terrain(args.envs_per_height)
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
    cfg.episode_length_s = args.steps * cfg.decimation * cfg.sim.dt
    cfg.rewards.first_step_success.weight = 0.0
    cfg.rewards.perception_miss.weight = 0.0
    cfg.rewards.joint_success.weight = 0.0
    env = gym.make("M20-Stair-Unified-Perceptive-v0", cfg=cfg)
    raw = env.unwrapped
    terrain = raw.scene.terrain
    types = torch.arange(num_envs, device=raw.device) // args.envs_per_height
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = types * args.envs_per_height
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = types * args.envs_per_height
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()

    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = BlindActor().to(raw.device).eval()
    unexpected = actor.load_state_dict(loaded["actor_state_dict"], strict=False)
    if unexpected.missing_keys or unexpected.unexpected_keys != ["distribution.log_std_param"]:
        raise RuntimeError(f"blind actor load mismatch: {unexpected}")

    hr_ids, names = raw.scene["robot"].find_bodies("hr_wheel")
    if names != ["hr_wheel"]:
        raise RuntimeError(names)
    hr_id = hr_ids[0]
    active = torch.ones(num_envs, device=raw.device, dtype=torch.bool)
    official_return = torch.zeros(num_envs, device=raw.device)
    s1_return = torch.zeros_like(official_return)
    steps_seen = torch.zeros(num_envs, device=raw.device)
    s1_steps = torch.zeros_like(steps_seen)
    max_base_x = torch.full_like(official_return, -1e9)
    max_hr_x = torch.full_like(official_return, -1e9)
    fell = torch.zeros(num_envs, device=raw.device, dtype=torch.bool)

    for _ in range(args.steps):
        observation = env.unwrapped.observation_manager.compute()["policy"][:, :57]
        with torch.inference_mode():
            action = actor(observation)
        _, reward, terminated, truncated, _ = env.step(action)
        reward = reward.squeeze(-1) if reward.ndim > 1 else reward
        state = mdp.v10_state(raw)
        official_return += torch.where(active, reward, torch.zeros_like(reward))
        steps_seen += active
        s1_mask = active & state["b_raw"]
        s1_return += torch.where(s1_mask, reward, torch.zeros_like(reward))
        s1_steps += s1_mask
        base_x = raw.scene["robot"].data.root_pos_w[:, 0] - raw.scene.env_origins[:, 0]
        hr_x = raw.scene["robot"].data.body_pos_w[:, hr_id, 0] - raw.scene.env_origins[:, 0]
        max_base_x = torch.where(active, torch.maximum(max_base_x, base_x), max_base_x)
        max_hr_x = torch.where(active, torch.maximum(max_hr_x, hr_x), max_hr_x)
        ended = active & (terminated.bool() | truncated.bool())
        fell |= active & terminated.bool()
        active &= ~ended
        if not active.any():
            break

    success = (max_base_x >= 2.90) & (max_hr_x >= 2.70) & ~fell
    rows = []
    for env_id in range(num_envs):
        height = HEIGHTS[int(types[env_id])]
        rows.append({
            "height_cm": int(height * 100), "env_id": env_id, "success": int(success[env_id]),
            "official_mean_per_step": float(official_return[env_id] / steps_seen[env_id].clamp_min(1)),
            "official_episode_return": float(official_return[env_id]),
            "s1_official_accumulated": float(s1_return[env_id]), "s1_steps": int(s1_steps[env_id]),
            "max_base_x": float(max_base_x[env_id]), "max_hr_x": float(max_hr_x[env_id]),
            "fall": int(fell[env_id]), "steps": int(steps_seen[env_id]),
        })
    csv_path = output_dir / "reward_calibration.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    returns = torch.tensor([row["official_episode_return"] for row in rows])
    success_returns = torch.tensor([row["official_episode_return"] for row in rows if row["success"]])
    failed_returns = torch.tensor([row["official_episode_return"] for row in rows if not row["success"]])
    scale = float(returns.abs().median().clamp_min(1.0))
    dt = float(raw.step_dt)
    first_bonus = 0.10 * scale
    miss_penalty = 0.15 * scale
    joint_bonus = 0.40 * scale
    first_weight = first_bonus / dt
    miss_weight = -miss_penalty / dt
    joint_weight = joint_bonus / dt
    report = (
        "# M20 V10 Reward Calibration\n\n"
        f"CHECKPOINT_SCALE_ONLY={args.checkpoint}\n"
        "CHECKPOINT_USED_FOR_INITIALIZATION=FALSE\n"
        f"NUM_ROLLOUTS={len(rows)}\nSUCCESS_COUNT={int(success.sum())}\n"
        f"OFFICIAL_ABS_RETURN_MEDIAN={scale:.6f}\n"
        f"SUCCESS_RETURN_MEAN={float(success_returns.mean()) if success_returns.numel() else float('nan'):.6f}\n"
        f"FAILED_RETURN_MEAN={float(failed_returns.mean()) if failed_returns.numel() else float('nan'):.6f}\n"
        f"CONTROL_DT={dt:.6f}\n"
        f"FIRST_STEP_SUCCESS_WEIGHT={first_weight:.6f}\n"
        f"PERCEPTION_MISS_WEIGHT={miss_weight:.6f}\n"
        f"JOINT_SUCCESS_WEIGHT={joint_weight:.6f}\n"
        f"EVENT_RETURN_TARGETS=first:{first_bonus:.6f},miss:{-miss_penalty:.6f},joint:{joint_bonus:.6f}\n"
    )
    (output_dir / "reward_calibration_report.md").write_text(report)
    print(report, flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
