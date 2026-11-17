#!/usr/bin/env python3
"""Export V10 M20 wheel-position derivatives for joint-sign alignment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import parse_env_cfg

import rl_training.tasks  # noqa: F401


JOINTS = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
WHEELS = ["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"]


def wheel_positions_base(robot, body_ids: list[int]) -> torch.Tensor:
    delta = robot.data.body_pos_w[:, body_ids] - robot.data.root_pos_w[:, None, :]
    quat = robot.data.root_quat_w[:, None, :].expand(-1, len(body_ids), -1)
    return math_utils.quat_apply_inverse(quat.reshape(-1, 4), delta.reshape(-1, 3)).reshape(1, len(body_ids), 3)


def main() -> None:
    cfg = parse_env_cfg("M20-Stair-Unified-Perceptive-v0", device=args.device, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.events.randomize_reset_base = None
    env = gym.make("M20-Stair-Unified-Perceptive-v0", cfg=cfg)
    raw = env.unwrapped
    env.reset()
    robot = raw.scene["robot"]
    joint_ids, joint_names = robot.find_joints(JOINTS, preserve_order=True)
    body_ids, body_names = robot.find_bodies(WHEELS, preserve_order=True)
    if joint_names != JOINTS or body_names != WHEELS:
        raise RuntimeError(f"mapping mismatch joints={joint_names} wheels={body_names}")

    default = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(robot.data.default_joint_vel)

    def set_pose(q: torch.Tensor) -> torch.Tensor:
        robot.write_joint_state_to_sim(q, zero_vel)
        raw.scene.write_data_to_sim()
        raw.sim.forward()
        raw.scene.update(dt=0.0)
        return wheel_positions_base(robot, body_ids)[0].clone()

    baseline = set_pose(default)
    rows = []
    for joint_id, joint_name in zip(joint_ids, joint_names):
        q = default.clone()
        q[:, joint_id] += 0.1
        derivative = (set_pose(q) - baseline) / 0.1
        row = {"joint": joint_name}
        for wheel_index, wheel_name in enumerate(WHEELS):
            for axis_index, axis_name in enumerate("xyz"):
                row[f"{wheel_name}_{axis_name}"] = float(derivative[wheel_index, axis_index].item())
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["joint"] + [f"{wheel}_{axis}" for wheel in WHEELS for axis in "xyz"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"ISAAC_KINEMATIC_SIGNS_CSV={output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
