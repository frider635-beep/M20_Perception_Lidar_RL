#!/usr/bin/env python3
"""Exact fixed-stair runtime audit of the deployable V10 lifecycle."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="eval_outputs/v10_lifecycle/fixed_lifecycle_after_fix.csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v10_unified_terrains import (
    V10StepTerrainCfg, v10_step_terrain,
)


HEIGHTS = (0.05, 0.08, 0.11, 0.14, 0.17, 0.20)
OFFSETS = tuple(round(0.20 + 0.05 * index, 2) for index in range(29))


def terrain_cfg():
    terrains = OrderedDict()
    for height in HEIGHTS:
        terrains[f"full_{int(height*100):02d}cm"] = V10StepTerrainCfg(
            function=v10_step_terrain, proportion=1/6, full_staircase=True, height_override=height,
        )
    return TerrainGeneratorCfg(
        seed=1014, curriculum=True, difficulty_range=(0.0, 0.0), size=(8.0, 8.0), border_width=20.0,
        num_rows=1, num_cols=6, horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
        color_scheme="height", use_cache=False, sub_terrains=terrains,
    )


def disable_randomization(cfg):
    for name in vars(cfg.events):
        if name.startswith("randomize_"):
            setattr(cfg.events, name, None)


def teleport(raw, offset):
    robot = raw.scene["robot"]
    root = robot.data.default_root_state.clone()
    root[:, :3] += raw.scene.env_origins
    root[:, 0] += offset
    root[:, 7:] = 0.0
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    raw.scene.write_data_to_sim(); raw.sim.forward()
    sensor = raw.scene.sensors["v10_pgtt_scanner"]
    sensor.reset(); sensor.update(raw.physics_dt, force_recompute=True)
    raw.episode_length_buf[:] = 1


def sample(raw, repeats=3):
    state = None
    for _ in range(repeats):
        raw.common_step_counter += 1
        raw._v10_last_step = -1
        state = mdp.v10_state(raw)
    return state


def main():
    cfg = parse_env_cfg("M20-Stair-Unified-Perceptive-v0", device="cuda:0", num_envs=6)
    disable_randomization(cfg)
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.terrain_generator = terrain_cfg()
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
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

    rows, previous_locked_d = [], [None] * 6
    target_switch_count = [0] * 6
    for offset in OFFSETS:
        teleport(raw, offset)
        state = sample(raw)
        for env_id, height in enumerate(HEIGHTS):
            locked_d = float(state["locked_d_raw"][env_id])
            if (
                previous_locked_d[env_id] is not None and not bool(state["first_riser_consumed"][env_id])
                and locked_d - previous_locked_d[env_id] > raw.cfg.v10_perception.track_max_forward_jump + 1e-6
            ):
                target_switch_count[env_id] += 1
            if bool(state["first_riser_locked"][env_id]):
                previous_locked_d[env_id] = locked_d
            rows.append({
                "height_cm": int(height*100), "root_offset": offset,
                "D_detected": float(state["detected_d"][env_id]),
                "D_locked_raw": locked_d, "D_filtered": float(state["locked_d_filtered"][env_id]),
                "H_locked": float(state["locked_h"][env_id]),
                "visible": int(state["first_riser_visible"][env_id]),
                "locked": int(state["first_riser_locked"][env_id]),
                "consumed": int(state["first_riser_consumed"][env_id]),
                "b_raw": int(state["b_raw"][env_id]),
                "b_actor": float(state["actor_feature"][env_id, 0]),
                "teacher_success": int(state["first_step_event"][env_id]),
                "teacher_miss": int(state["perception_miss_event"][env_id]),
            })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    passed = True
    for env_id, height in enumerate(HEIGHTS):
        selected = [row for row in rows if row["height_cm"] == int(height*100)]
        consumed = any(row["consumed"] for row in selected)
        l2_rows = [row for row in selected if row["root_offset"] >= 1.30]
        l2_block = all(not row["b_raw"] and row["b_actor"] == 0 for row in l2_rows)
        ok = target_switch_count[env_id] == 0 and consumed and l2_block
        passed &= ok
        print(
            f"FIXED_{int(height*100):02d}CM switch={target_switch_count[env_id]} "
            f"consumed={consumed} L2_L6_block={l2_block}", flush=True
        )
    print(f"FIRST_RISER_LOCK={'PASS' if all(v == 0 for v in target_switch_count) else 'FAIL'}", flush=True)
    print(f"L2_L6_REACTIVATION_BLOCK={'PASS' if passed else 'FAIL'}", flush=True)
    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
