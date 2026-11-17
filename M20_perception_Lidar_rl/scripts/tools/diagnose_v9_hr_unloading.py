#!/usr/bin/env python3
"""Read-only V9 inference diagnostic for rear-right unloading before a riser."""

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
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1200)
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
LEG_JOINTS = (
    ("fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint"),
    ("fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint"),
    ("hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint"),
    ("hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint"),
)


class Accumulator:
    def __init__(self):
        self.count = 0
        self.values: dict[str, float] = {}

    def add(self, mask: torch.Tensor, values: dict[str, torch.Tensor]):
        n = int(mask.sum().item())
        if n == 0:
            return
        self.count += n
        for key, value in values.items():
            self.values[key] = self.values.get(key, 0.0) + float(value[mask].sum().item())

    def means(self):
        return {key: value / max(self.count, 1) for key, value in self.values.items()}


def main():
    cfg = DeeproboticsM20PerceptiveV9Stage1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    # Keep the exact Stage-1 command, terrain, observation and actuator setup.
    env = gym.make("M20-Perceptive-V9-Stage1-v0", cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    agent_cfg = DeeproboticsM20PerceptiveV9Stage1PPORunnerCfg().to_dict()
    runner = V9Stage1Runner(wrapped, agent_cfg, log_dir=None, device=cfg.sim.device)
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    contact_sensor = raw.scene.sensors["contact_forces"]

    body_ids, contact_ids, leg_ids = [], [], []
    for wheel in mdp.WHEEL_NAMES:
        ids, names = robot.find_bodies(wheel)
        cids, cnames = contact_sensor.find_bodies(wheel)
        if len(ids) != 1 or names != [wheel] or len(cids) != 1 or cnames != [wheel]:
            raise RuntimeError(f"wheel/contact order mismatch for {wheel}: {names}, {cnames}")
        body_ids.append(ids[0])
        contact_ids.append(cids[0])
    for expected in LEG_JOINTS:
        ids, names = robot.find_joints(list(expected), preserve_order=True)
        if names != list(expected):
            raise RuntimeError(f"leg joint order mismatch: {names}")
        leg_ids.append(ids)
    contact_ids_t = torch.tensor(contact_ids, device=raw.device, dtype=torch.long)
    leg_ids_t = torch.tensor(leg_ids, device=raw.device, dtype=torch.long)

    accumulators = {
        "HR_PREPARE_BASELINE": Accumulator(),
        "HR_PRELIFT_ACTIVE": Accumulator(),
        "HR_PRELIFT_NOT_STARTED": Accumulator(),
        "HR_LIFT_STARTED": Accumulator(),
    }
    for label in ("30_20cm", "20_10cm", "10_05cm", "05_00cm"):
        accumulators[f"HR_ACTIVE_{label}"] = Accumulator()

    obs, _ = env.reset()
    rows = []
    for step in range(args.steps):
        with torch.inference_mode():
            policy_obs = runner.alg.policy.get_actor_obs(obs)
            action = policy(obs)
            base_action = runner._v9_base_actor(policy_obs)
            obs, _, _, _ = wrapped.step(action)
            state = mdp.perceptive_spatial_state(raw)

        forces = contact_sensor.data.net_forces_w[:, contact_ids_t]
        force_norm = forces.norm(dim=-1)
        force_z_up = forces[..., 2].clamp_min(0.0)
        contact = force_norm >= 1.0
        vertical = mdp.classify_vertical_riser_contact(
            state["wheel_yaw"][..., :2], state["z_actual_world"], forces,
            state["riser_x"], state["z_lower"], state["z_upper"],
            wheel_radius=raw.cfg.spatial_guidance.wheel_collision_radius,
        ) & state["valid"]
        distance = state["riser_distance"][:, 3]
        hr_active = state["valid"][:, 3] & state["active"][:, 3] & (distance > 0.0) & ~vertical[:, 3]
        baseline = state["valid"][:, 3] & ~state["active"][:, 3] & (distance > 0.35) & (distance <= 0.45)
        lift_started = hr_active & (state["actual_lift"][:, 3] > 0.005)
        lift_not_started = hr_active & ~lift_started

        gravity = robot.data.projected_gravity_b
        roll = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
        pitch = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
        joint_delta = (robot.data.joint_pos[:, leg_ids_t] - robot.data.default_joint_pos[:, leg_ids_t]).norm(dim=2)
        hr_action_norm = action[:, 9:12].norm(dim=1)
        hr_action_delta = (action[:, 9:12] - base_action[:, 9:12]).norm(dim=1)
        values = {
            "HR_actual_lift_m": state["actual_lift"][:, 3],
            "HR_lift_ref_m": state["lift_ref"][:, 3],
            "HR_clearance_progress": state["progress"][:, 3],
            "HR_leg_action_norm": hr_action_norm,
            "HR_leg_action_delta_norm": hr_action_delta,
            "HR_joint_position_delta_norm": joint_delta[:, 3],
            "base_roll_rad": roll,
            "base_pitch_rad": pitch,
            "base_vx_mps": robot.data.root_lin_vel_b[:, 0],
        }
        for index, wheel in enumerate(WHEELS):
            values[f"{wheel}_contact"] = contact[:, index].float()
            values[f"{wheel}_force_z_up_N"] = force_z_up[:, index]
            values[f"{wheel}_force_norm_N"] = force_norm[:, index]

        accumulators["HR_PREPARE_BASELINE"].add(baseline, values)
        accumulators["HR_PRELIFT_ACTIVE"].add(hr_active, values)
        accumulators["HR_PRELIFT_NOT_STARTED"].add(lift_not_started, values)
        accumulators["HR_LIFT_STARTED"].add(lift_started, values)
        for label, lower, upper in (
            ("30_20cm", 0.20, 0.30), ("20_10cm", 0.10, 0.20),
            ("10_05cm", 0.05, 0.10), ("05_00cm", 0.00, 0.05),
        ):
            accumulators[f"HR_ACTIVE_{label}"].add(hr_active & (distance > lower) & (distance <= upper), values)

        if step % 10 == 0:
            rows.append({
                "step": step,
                "hr_active_fraction": float(hr_active.float().mean()),
                "hr_lift_started_fraction": float(lift_started.float().mean()),
                **{f"active_{k}": v for k, v in AccumulatorValues(hr_active, values).items()},
            })

    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_iter": int(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["iter"]),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "definitions": {
            "normal_force": "world-up max(Fz,0), appropriate for horizontal tread support",
            "HR_PREPARE_BASELINE": "valid & !active & 0.35m < riser_distance <= 0.45m",
            "HR_PRELIFT_ACTIVE": "valid & active & riser_distance>0 & !vertical_contact",
            "HR_LIFT_STARTED": "HR_PRELIFT_ACTIVE & actual_lift>0.005m",
        },
        "groups": {
            name: {"sample_count": acc.count, **acc.means()} for name, acc in accumulators.items()
        },
    }
    active = result["groups"]["HR_PRELIFT_ACTIVE"]
    baseline_result = result["groups"]["HR_PREPARE_BASELINE"]
    result["diagnosis"] = {
        "HR_PRELIFT_OBSERVED": active["sample_count"] > 0,
        "HR_CONTACT_REMAINS_HIGH": active.get("HR_contact", 0.0) >= 0.40,
        "HR_FORCE_REMAINS_HIGH": active.get("HR_force_z_up_N", 0.0) >= 0.70 * max(
            baseline_result.get("HR_force_z_up_N", 0.0), 1e-6
        ),
        "HR_LIFT_REMAINS_LOW": active.get("HR_actual_lift_m", 0.0) <= 0.02,
    }
    result["diagnosis"]["HR_UNLOADING_FAILURE"] = all(result["diagnosis"].values())

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "summary.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    if rows:
        with open(os.path.join(args.output, "timeseries_summary.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))
    env.close()


def AccumulatorValues(mask: torch.Tensor, values: dict[str, torch.Tensor]):
    n = int(mask.sum().item())
    if n == 0:
        return {key: 0.0 for key in values}
    return {key: float(value[mask].mean().item()) for key, value in values.items()}


try:
    main()
finally:
    simulation_app.close()
