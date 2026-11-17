#!/usr/bin/env python3
"""Read-only, term-by-term reward ledger for M20-Stair-HRBridge-v0."""

import argparse
import csv
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--step_height", type=float, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1100)
parser.add_argument("--mode", choices=("actor", "bridge_hold", "stop_on_s1"), default="actor")
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
from rl_training.tasks.manager_based.locomotion.velocity.mdp.hr_bridge import (  # noqa: E402
    HRBridgeTracker,
    STATE_HR_BRIDGE,
)


def actor_network():
    return nn.Sequential(
        nn.Linear(57, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16),
    )


def configure():
    cfg = DeeproboticsM20HRBridgeEnvCfg()
    cfg.seed = 241
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.terrain_generator = make_hr_bridge_fixed_terrain(
        (args.step_height,), args.num_envs, 1.10
    )
    cfg.scene.terrain.max_init_terrain_level = 0
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
    term_names = list(raw.reward_manager.active_terms)
    term_count = len(term_names)
    episode_terms = torch.zeros((n, term_count), device=raw.device)
    s1_terms = torch.zeros_like(episode_terms)
    episode_return = torch.zeros(n, device=raw.device)
    s1_return = torch.zeros(n, device=raw.device)
    episode_steps = torch.zeros(n, dtype=torch.long, device=raw.device)
    s1_steps = torch.zeros_like(episode_steps)
    s1_vx_sum = torch.zeros(n, device=raw.device)
    s1_cmd_sum = torch.zeros(n, device=raw.device)
    full_success = torch.zeros(n, dtype=torch.bool, device=raw.device)
    stopped = torch.zeros_like(full_success)
    previous_action = torch.zeros((n, 16), device=raw.device)
    rows = []
    stair_cfg = cfg.scene.terrain.terrain_generator.sub_terrains["hr_bridge_fixed_stair"]
    success_x = stair_cfg.first_riser_from_spawn + stair_cfg.levels * stair_cfg.step_depth
    step_dt = float(raw.step_dt)

    for _ in range(args.steps):
        with torch.inference_mode():
            action = actor(obs["policy"][:, :57])
        # Actor output is an inference tensor; clone it under normal tensor
        # semantics before applying evaluator-only counterfactual overrides.
        with torch.inference_mode(False):
            action = action.clone()
        with torch.no_grad():
            controller_state, riser, activation = tracker.pre_action_state()
            stopped |= activation
            if args.mode == "bridge_hold":
                active = controller_state == STATE_HR_BRIDGE
                action[active, 9:12] = previous_action[active, 9:12]
            elif args.mode == "stop_on_s1":
                action[stopped] = 0.0
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            tracker.post_step(controller_state, riser, activation)
        step_terms = raw.reward_manager._step_reward * step_dt
        in_s1 = controller_state == STATE_HR_BRIDGE
        episode_terms += step_terms
        s1_terms += step_terms * in_s1[:, None]
        episode_return += reward
        s1_return += reward * in_s1
        episode_steps += 1
        s1_steps += in_s1.long()
        s1_vx_sum += raw.scene["robot"].data.root_lin_vel_b[:, 0] * in_s1
        s1_cmd_sum += raw.command_manager.get_command("base_velocity")[:, 0] * in_s1
        root_progress = raw.scene["robot"].data.root_pos_w[:, 0] - raw.scene.env_origins[:, 0]
        stable = raw.scene["robot"].data.projected_gravity_b[:, 2] < -0.65
        full_success |= (root_progress > success_x) & stable
        for env_id in done.nonzero(as_tuple=False).flatten().tolist():
            s1_n = int(s1_steps[env_id])
            row = {
                "mode": args.mode,
                "height_m": args.step_height,
                "env": env_id,
                "episode_steps": int(episode_steps[env_id]),
                "s1_steps": s1_n,
                "episode_return": float(episode_return[env_id]),
                "s1_return": float(s1_return[env_id]),
                "s1_mean_base_vx": float(s1_vx_sum[env_id] / max(s1_n, 1)),
                "s1_mean_command_x": float(s1_cmd_sum[env_id] / max(s1_n, 1)),
                "full_stair_success": int(full_success[env_id]),
                "hr_bridge_success": int(tracker.success_latched[env_id]),
                "perception_miss": int(tracker.miss_latched[env_id]),
                "joint_success": int(tracker.joint_success_latched[env_id]),
                "final_root_progress": float(root_progress[env_id]),
            }
            for idx, name in enumerate(term_names):
                row[f"episode_term__{name}"] = float(episode_terms[env_id, idx])
                row[f"s1_term__{name}"] = float(s1_terms[env_id, idx])
            rows.append(row)
        previous_action = action.detach().clone()
        previous_action[done] = 0.0
        episode_terms[done] = 0.0
        s1_terms[done] = 0.0
        episode_return[done] = 0.0
        s1_return[done] = 0.0
        episode_steps[done] = 0
        s1_steps[done] = 0
        s1_vx_sum[done] = 0.0
        s1_cmd_sum[done] = 0.0
        full_success[done] = False
        stopped[done] = False
        tracker.reset(done)

    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, f"ledger_{args.mode}_{int(round(args.step_height * 100)):02d}cm.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "mode": args.mode,
        "height_m": args.step_height,
        "checkpoint": os.path.abspath(args.checkpoint),
        "num_envs": n,
        "steps": args.steps,
        "step_dt": step_dt,
        "gamma": 0.99,
        "reward_terms": term_names,
        "episodes_written": len(rows),
    }
    with open(path.replace(".csv", ".json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
