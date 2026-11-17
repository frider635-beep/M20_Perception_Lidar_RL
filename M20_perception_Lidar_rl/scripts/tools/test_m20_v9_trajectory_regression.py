#!/usr/bin/env python3
"""Same-seed physical trajectory regression for official 57D and V9 372D actors."""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--official", required=True)
parser.add_argument("--v9", required=True)
parser.add_argument("--terrain", choices=("flat", "stair"), required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seconds", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from isaaclab.terrains import TerrainGeneratorCfg

import rl_training.tasks  # noqa: F401,E402
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v9_stage1_env_cfg import (  # noqa: E402
    DeeproboticsM20PerceptiveV9Stage1EnvCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v9_stage1_terrains import (  # noqa: E402
    V9Stage1RiserCfg,
    v9_stage1_riser_terrain,
)


def terrain_cfg(height: float) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=9109, curriculum=False, difficulty_range=(0.0, 1.0), size=(8.0, 8.0),
        border_width=1.0, num_rows=1, num_cols=1, color_scheme="height", use_cache=False,
        sub_terrains={
            "v9_regression": V9Stage1RiserCfg(
                function=v9_stage1_riser_terrain, proportion=1.0, step_heights=(height,)
            )
        },
    )


class Actor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16),
        )

    def forward(self, x):
        return self.mlp(x)


def load_actor(path, input_dim):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("actor_state_dict", checkpoint)
    actor = Actor(input_dim).eval()
    actor.load_state_dict({
        "mlp.0.weight": state["mlp.0.weight"], "mlp.0.bias": state["mlp.0.bias"],
        "mlp.2.weight": state["mlp.2.weight"], "mlp.2.bias": state["mlp.2.bias"],
        "mlp.4.weight": state["mlp.4.weight"], "mlp.4.bias": state["mlp.4.bias"],
        "mlp.6.weight": state["mlp.6.weight"], "mlp.6.bias": state["mlp.6.bias"],
    })
    return actor


def main():
    cfg = DeeproboticsM20PerceptiveV9Stage1EnvCfg()
    cfg.seed = 9109
    cfg.scene.num_envs = 2
    cfg.scene.terrain.terrain_generator = terrain_cfg(0.0 if args.terrain == "flat" else 0.14)
    cfg.observations.policy.enable_corruption = False
    command = cfg.commands.base_velocity
    command.ranges.lin_vel_x = (0.4, 0.4)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_zero_vel_envs = 0.0
    command.perceptive_metrics = False
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
    raw = env.unwrapped
    official_path = args.official.replace("_actor57.onnx", ".pt")
    v9_path = args.v9.replace(".onnx", ".pt")
    if not os.path.isfile(official_path):
        official_path = "/home/ubuntu/m20_perceptive_final/eval_outputs/v9_initialization_gate/colleague_model9999/model_9999.pt"
    if not os.path.isfile(v9_path):
        v9_path = "/home/ubuntu/m20_perceptive_final/eval_outputs/v9_initialization_gate/colleague_model9999/transferred_actor/v9_colleague_model9999_57_to_372.pt"
    official = load_actor(official_path, 57).to(raw.device)
    v9 = load_actor(v9_path, 372).to(raw.device)
    obs, _ = env.reset()
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    if tuple(policy_obs.shape) != (2, 372):
        raise RuntimeError(f"expected [2,372] policy obs, got {tuple(policy_obs.shape)}")
    max_action_error = 0.0
    max_root_pos_error = 0.0
    max_root_quat_error = 0.0
    max_joint_error = 0.0
    origin = raw.scene.env_origins
    steps = int(args.seconds / raw.step_dt)
    for _ in range(steps):
        array = policy_obs.detach().cpu().numpy().astype(np.float32)
        with torch.inference_mode():
            a57_t = official(policy_obs[0:1, :57])
            a372_t = v9(policy_obs[1:2])
        max_action_error = max(max_action_error, float((a57_t - a372_t).abs().max().item()))
        action = torch.cat((a57_t, a372_t), dim=0)
        obs, _, terminated, truncated, _ = env.step(action)
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        robot = raw.scene["robot"]
        local_pos = robot.data.root_pos_w - origin
        max_root_pos_error = max(max_root_pos_error, float((local_pos[0] - local_pos[1]).abs().max()))
        max_root_quat_error = max(
            max_root_quat_error, float((robot.data.root_quat_w[0] - robot.data.root_quat_w[1]).abs().max())
        )
        max_joint_error = max(
            max_joint_error, float((robot.data.joint_pos[0] - robot.data.joint_pos[1]).abs().max())
        )
        if bool(terminated.any()) or bool(truncated.any()):
            break
    env.close()
    passed = max_action_error <= 2.0e-5 and max_root_pos_error <= 2.0e-4 and max_joint_error <= 2.0e-4
    result = {
        "terrain": args.terrain, "steps": steps, "max_action_error": max_action_error,
        "max_root_pos_error": max_root_pos_error, "max_root_quat_error": max_root_quat_error,
        "max_joint_error": max_joint_error, "pass": passed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


try:
    main()
finally:
    simulation_app.close()
