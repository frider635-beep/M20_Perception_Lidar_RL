#!/usr/bin/env python3
"""Run the final V10 checkpoint at fixed +/-1 rad/s yaw in the original Isaac task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.v10_unified_policy import UnifiedActor
from rl_training.tasks.manager_based.locomotion.velocity import mdp


def load_actor(path: str, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = UnifiedActor().to(device).eval()
    state = {
        key.removeprefix("actor."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("actor.")
    }
    actor.load_state_dict(state, strict=True)
    return actor, int(checkpoint.get("iter", -1))


def main() -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg("M20-Stair-Unified-Perceptive-v0", device=args.device, num_envs=2)
    cfg.seed = 29999
    cfg.scene.num_envs = 2
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
    cfg.commands.base_velocity.rel_only_lin_y_envs = 0.0
    cfg.commands.base_velocity.rel_only_lin_x_envs = 0.0
    cfg.commands.base_velocity.rel_only_ang_z_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (100.0, 100.0)
    env = gym.make("M20-Stair-Unified-Perceptive-v0", cfg=cfg)
    raw = env.unwrapped
    actor, iteration = load_actor(args.checkpoint, raw.device)
    env.reset()
    commands = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], device=raw.device)
    rows: list[dict[str, float | int]] = []
    first_actions = None
    for step in range(args.steps):
        raw.command_manager.get_command("base_velocity")[:] = commands
        policy_obs = raw.observation_manager.compute()["policy"]
        with torch.inference_mode():
            actions = actor(policy_obs)
        if first_actions is None:
            first_actions = actions.detach().clone()
        env.step(actions)
        raw.command_manager.get_command("base_velocity")[:] = commands
        next_obs = raw.observation_manager.compute()["policy"]
        base_wz = raw.scene["robot"].data.root_ang_vel_b[:, 2]
        reward = mdp.track_ang_vel_z_exp(raw, std=0.5, command_name="base_velocity")
        for env_id in range(2):
            rows.append(
                {
                    "step": step,
                    "env": env_id,
                    "raw_command_wz": float(commands[env_id, 2]),
                    "obs_command_wz": float(next_obs[env_id, 8]),
                    "base_wz": float(base_wz[env_id]),
                    "track_ang_vel_z": float(reward[env_id]),
                    "b": float(next_obs[env_id, 57]),
                    "H": float(next_obs[env_id, 58]),
                    "D": float(next_obs[env_id, 59]),
                }
            )
    with (output / "timeseries.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {"checkpoint_iteration": iteration}
    for env_id, label in ((0, "positive"), (1, "negative")):
        tail = [row for row in rows if row["env"] == env_id and row["step"] >= args.steps - 100]
        summary[label] = {
            "raw_command_wz": sum(float(row["raw_command_wz"]) for row in tail) / len(tail),
            "obs_command_wz": sum(float(row["obs_command_wz"]) for row in tail) / len(tail),
            "mean_base_wz": sum(float(row["base_wz"]) for row in tail) / len(tail),
            "mean_track_reward": sum(float(row["track_ang_vel_z"]) for row in tail) / len(tail),
        }
    assert first_actions is not None
    summary["first_action_max_abs_diff"] = float((first_actions[0] - first_actions[1]).abs().max())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    env.close()
    app.close()


if __name__ == "__main__":
    main()
