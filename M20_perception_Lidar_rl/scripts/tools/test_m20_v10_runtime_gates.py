#!/usr/bin/env python3
"""Isaac runtime gates for V10 PGTT and first-riser lifecycle."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--flat_steps", type=int, default=2000)
parser.add_argument("--output", default="/tmp/m20_v10_runtime_gates.txt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp


TASK = "M20-Stair-Unified-Perceptive-v0"


def disable_randomization(cfg):
    for name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia", "randomize_com_positions", "randomize_apply_external_force_torque",
        "randomize_reset_joints", "randomize_actuator_gains", "randomize_push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def set_layout(raw):
    terrain = raw.scene.terrain
    levels = torch.tensor([0, 0, 3, 4, 3], device=raw.device)
    types = torch.tensor([0, 12, 13, 14, 28], device=raw.device)
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = types
    terrain.env_origins[:] = terrain.terrain_origins[levels, types]


def teleport(raw, env_id: int, x_offset: float):
    robot = raw.scene["robot"]
    state = robot.data.default_root_state[env_id : env_id + 1].clone()
    state[:, :3] += raw.scene.env_origins[env_id : env_id + 1]
    state[:, 0] += x_offset
    state[:, 7:] = 0.0
    ids = torch.tensor([env_id], device=raw.device, dtype=torch.long)
    robot.write_root_state_to_sim(state, env_ids=ids)
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos[ids], torch.zeros_like(robot.data.default_joint_vel[ids]), env_ids=ids
    )
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    sensor = raw.scene.sensors["v10_pgtt_scanner"]
    sensor.reset(ids)
    sensor.update(raw.physics_dt, force_recompute=True)


def sample(raw, repeats=3):
    state = None
    raw.episode_length_buf[:] = 1
    for _ in range(repeats):
        raw.common_step_counter += 1
        raw._v10_last_step = -1
        state = mdp.v10_state(raw)
    return state


def main():
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=5)
    disable_randomization(cfg)
    cfg.curriculum.terrain_levels = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    set_layout(raw)
    env.reset()
    set_layout(raw)

    zeros = torch.zeros((raw.num_envs, 16), device=raw.device)
    flat_ok = True
    for _ in range(args.flat_steps):
        env.step(zeros)
        state = mdp.v10_state(raw)
        flat_ok &= bool(
            (state["actor_feature"][0] == 0).all()
            and not state["b_raw"][0]
            and state["h_raw"][0] == 0
            and state["d_raw"][0] == 0
        )

    # Put the three single-step cases 0.50m before L1 and confirm the gate.
    for env_id in (1, 2, 3):
        teleport(raw, env_id, 0.50)
    state = sample(raw)
    expected = {1: "05CM", 2: "APPROX17CM", 3: "20CM"}
    single_ok = {}
    for env_id, label in expected.items():
        ok = bool(state["valid"][env_id] and state["b_raw"][env_id] and state["h_raw"][env_id] >= 0.05)
        single_ok[label] = ok
        print(
            f"{label}_GATE valid={bool(state['valid'][env_id])} b={bool(state['b_raw'][env_id])} "
            f"H={float(state['h_raw'][env_id]):.6f} D={float(state['d_raw'][env_id]):.6f}"
        )

    teleport(raw, 4, 0.50)
    state = sample(raw)
    first_trigger = bool(state["b_raw"][4])
    state["first_riser_consumed"][4] = True
    later_b = []
    # L2-L6 planes are each another 0.31m forward.
    for level in range(2, 7):
        teleport(raw, 4, 0.50 + (level - 1) * 0.31)
        state = sample(raw)
        later_b.append(bool(state["b_raw"][4]))
    first_only = first_trigger and not any(later_b)

    lines = [
        f"FLAT_STEPS={args.flat_steps}",
        f"FLAT_GATE={'PASS' if flat_ok else 'FAIL'}",
        f"SINGLE_STEP_05_GATE={'PASS' if single_ok['05CM'] else 'FAIL'}",
        f"SINGLE_STEP_17_GATE={'PASS' if single_ok['APPROX17CM'] else 'FAIL'}",
        f"SINGLE_STEP_20_GATE={'PASS' if single_ok['20CM'] else 'FAIL'}",
        f"FULL_FIRST_TRIGGER={first_trigger}",
        f"FULL_L2_L6_B={later_b}",
        f"FIRST_RISER_ONLY={'PASS' if first_only else 'FAIL'}",
    ]
    report = "\n".join(lines) + "\n"
    print(report, end="", flush=True)
    Path(args.output).write_text(report)
    env.close()
    simulation_app.close()
    passed = flat_ok and all(single_ok.values()) and first_only
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
