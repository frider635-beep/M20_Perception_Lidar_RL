"""Runtime dump of the final M20 PPO terrain composition at 4096 envs."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


try:
    cfg = parse_env_cfg("M20-Stair-Perceptive-v0", device=args.device, num_envs=4096)
    curriculum_configured = cfg.curriculum.terrain_levels is not None
    env = gym.make("M20-Stair-Perceptive-v0", cfg=cfg)
    env.reset()
    raw = env.unwrapped
    terrain = raw.scene.terrain
    types, type_counts = torch.unique(terrain.terrain_types, return_counts=True)
    levels, level_counts = torch.unique(terrain.terrain_levels, return_counts=True)
    print(f"NUM_ENVS={raw.num_envs}")
    print(f"TERRAIN_GENERATOR={type(terrain.cfg.terrain_generator).__name__}")
    print(f"SUB_TERRAINS={list(terrain.cfg.terrain_generator.sub_terrains)}")
    print("TERRAIN_TYPE_DISTRIBUTION=" + repr({int(k): int(v) for k, v in zip(types, type_counts)}))
    print("TERRAIN_LEVEL_DISTRIBUTION=" + repr({int(k): int(v) for k, v in zip(levels, level_counts)}))
    print(f"CURRICULUM_CONFIGURED={curriculum_configured}")
    env.close()
finally:
    app.close()
