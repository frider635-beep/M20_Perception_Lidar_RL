#!/usr/bin/env python3
"""Audit deployable PGTT D around and beyond the first riser."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="eval_outputs/v10_lifecycle/d_lifecycle_before_fix.csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import csv
import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import (
    _elevation_points_yaw,
    extract_nearest_riser_points,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v10_unified_terrains import (
    V10StepTerrainCfg,
    v10_step_terrain,
)


HEIGHTS = (0.05, 0.08, 0.11, 0.14, 0.17, 0.20)
OFFSETS = tuple(round(0.20 + 0.05 * index, 2) for index in range(29))


def terrain_cfg():
    terrains = OrderedDict()
    for height in HEIGHTS:
        terrains[f"full_{int(height*100):02d}cm"] = V10StepTerrainCfg(
            function=v10_step_terrain, proportion=1 / 6, full_staircase=True, height_override=height,
        )
    return TerrainGeneratorCfg(
        seed=1013, curriculum=True, difficulty_range=(0.0, 0.0), size=(8.0, 8.0), border_width=20.0,
        num_rows=1, num_cols=6, horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
        color_scheme="height", use_cache=False, sub_terrains=terrains,
    )


def teleport(raw, offset):
    robot = raw.scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += raw.scene.env_origins
    root[:, 0] += offset
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    sensor = raw.scene.sensors["v10_pgtt_scanner"]
    sensor.reset()
    sensor.update(raw.physics_dt, force_recompute=True)


def main():
    cfg = parse_env_cfg("M20-Stair-Unified-Perceptive-v0", device="cuda:0", num_envs=6)
    cfg.scene.terrain.terrain_generator = terrain_cfg()
    cfg.curriculum.terrain_levels = None
    for name in vars(cfg.events):
        if name.startswith("randomize_"):
            setattr(cfg.events, name, None)
    env = gym.make("M20-Stair-Unified-Perceptive-v0", cfg=cfg)
    raw = env.unwrapped
    terrain = raw.scene.terrain
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = torch.arange(6, device=raw.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = torch.arange(6, device=raw.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    rows = []
    virtual_base = torch.zeros((6, 4, 2), device=raw.device)
    for offset in OFFSETS:
        teleport(raw, offset)
        points = _elevation_points_yaw(raw, SceneEntityCfg("v10_pgtt_scanner"), raw.scene["robot"])
        riser = extract_nearest_riser_points(
            points, virtual_base, corridor_half_width=0.20, h_min=0.048, d_end=-0.30,
        )
        for env_id, height in enumerate(HEIGHTS):
            rows.append({
                "height_cm": int(height * 100), "root_offset": offset,
                "valid": int(riser["valid"][env_id, 0]),
                "H": float(riser["riser_height"][env_id, 0]),
                "D_nearest": float(riser["riser_distance"][env_id, 0]),
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    for height in HEIGHTS:
        selected = [row for row in rows if row["height_cm"] == int(height * 100)]
        print(f"D_LIFECYCLE_{int(height*100):02d}CM=" + ",".join(f"{r['root_offset']:.2f}:{r['D_nearest']:.3f}" for r in selected), flush=True)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
