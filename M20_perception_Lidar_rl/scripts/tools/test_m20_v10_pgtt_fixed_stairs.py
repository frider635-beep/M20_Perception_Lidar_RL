#!/usr/bin/env python3
"""Exact 5/8/11/14/17/20cm V10 PGTT H/D approach audit."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="/tmp/m20_v10_pgtt_fixed.txt")
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
    V10StepTerrainCfg,
    v10_step_terrain,
)


HEIGHTS = (0.05, 0.08, 0.11, 0.14, 0.17, 0.20)
OFFSETS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def disable_randomization(cfg):
    for name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia", "randomize_com_positions", "randomize_apply_external_force_torque",
        "randomize_reset_joints", "randomize_actuator_gains", "randomize_push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    # This audit replaces the official terrain dictionary with fixed named
    # steps, so the production pyramid_stairs_inv-specific reset is inapplicable.
    cfg.events.randomize_reset_base = None


def make_terrain():
    terrains = OrderedDict()
    for height in HEIGHTS:
        terrains[f"single_{int(height * 100):02d}cm"] = V10StepTerrainCfg(
            function=v10_step_terrain,
            proportion=1.0 / len(HEIGHTS),
            full_staircase=False,
            height_override=height,
        )
    return TerrainGeneratorCfg(
        seed=1011, curriculum=True, difficulty_range=(0.0, 0.0), size=(8.0, 8.0), border_width=20.0,
        num_rows=1, num_cols=len(HEIGHTS), horizontal_scale=0.1, vertical_scale=0.005,
        slope_threshold=0.75, color_scheme="height", use_cache=False, sub_terrains=terrains,
    )


def teleport(raw, offset):
    robot = raw.scene["robot"]
    state = robot.data.default_root_state.clone()
    state[:, :3] += raw.scene.env_origins
    state[:, 0] += offset
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    sensor = raw.scene.sensors["v10_pgtt_scanner"]
    sensor.reset()
    sensor.update(raw.physics_dt, force_recompute=True)
    raw.episode_length_buf[:] = 1
    sampled = None
    for _ in range(3):
        raw.common_step_counter += 1
        raw._v10_last_step = -1
        sampled = mdp.v10_state(raw)
    return sampled


def main():
    cfg = parse_env_cfg("M20-Stair-Unified-Perceptive-v0", device="cuda:0", num_envs=len(HEIGHTS))
    disable_randomization(cfg)
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.terrain_generator = make_terrain()
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
    env = gym.make("M20-Stair-Unified-Perceptive-v0", cfg=cfg)
    raw = env.unwrapped
    terrain = raw.scene.terrain
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = torch.arange(len(HEIGHTS), device=raw.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()
    terrain.terrain_levels[:] = 0
    terrain.terrain_types[:] = torch.arange(len(HEIGHTS), device=raw.device)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

    all_d = []
    all_h = []
    all_valid = []
    all_active = []
    for offset in OFFSETS:
        state = teleport(raw, offset)
        all_d.append(state["d_raw"].clone())
        all_h.append(state["h_raw"].clone())
        all_valid.append(state["valid"].clone())
        all_active.append(state["b_raw"].clone())
    distances = torch.stack(all_d)
    measured_h = torch.stack(all_h)
    valid = torch.stack(all_valid)
    active = torch.stack(all_active)
    expected_h = torch.tensor(HEIGHTS, device=raw.device)
    height_error = (measured_h - expected_h).abs()
    monotonic = True
    max_jump_error = 0.0
    height_ok = True
    valid_ok = True
    lines = []
    for index, height in enumerate(HEIGHTS):
        selected = active[:, index]
        selected_d = distances[selected, index]
        selected_h = measured_h[selected, index]
        selected_error = height_error[selected, index]
        delta_d = selected_d[1:] - selected_d[:-1]
        monotonic &= bool(selected_d.numel() >= 2 and (delta_d <= 0.015).all())
        max_jump_error = max(max_jump_error, float((delta_d + 0.10).abs().max()))
        height_ok &= bool(selected_error.numel() > 0 and (selected_error <= 0.015).all())
        valid_ok &= bool(valid[selected, index].all())
        lines.append(
            f"FIXED_{int(height*100):02d}CM active_N={int(selected.sum())} valid={bool(valid[selected, index].all())} "
            f"H_mean={float(selected_h.mean()):.6f} H_max_error={float(selected_error.max()):.6f} "
            f"D={selected_d.tolist()}"
        )
    passed = monotonic and max_jump_error <= 0.04 and height_ok and valid_ok
    lines.extend((
        f"D_MONOTONIC={'PASS' if monotonic else 'FAIL'}",
        f"D_MAX_STEP_ERROR={max_jump_error:.6f}",
        f"HEIGHT_EXTRACTION={'PASS' if height_ok else 'FAIL'}",
        f"PGTT_DEPLOYABLE_INPUT={'PASS' if passed else 'FAIL'}",
    ))
    report = "\n".join(lines) + "\n"
    Path(args.output).write_text(report)
    print(report, end="", flush=True)
    env.close()
    simulation_app.close()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
