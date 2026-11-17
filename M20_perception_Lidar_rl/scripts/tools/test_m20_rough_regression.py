"""Regression smoke for the untouched authoritative Rough M20 task."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg("Rough-Deeprobotics-M20-v0", device=args.device, num_envs=args.num_envs)
    env = gym.make("Rough-Deeprobotics-M20-v0", cfg=cfg)
    observations, _ = env.reset()
    raw = env.unwrapped
    dims = (
        raw.observation_manager.group_obs_dim["policy"][0],
        raw.observation_manager.group_obs_dim["critic"][0],
        raw.action_manager.total_action_dim,
    )
    action = torch.zeros((args.num_envs, dims[2]), device=raw.device)
    for _ in range(args.steps):
        observations, reward, _, _, _ = env.step(action)
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
    print(f"ROUGH_POLICY_DIM={dims[0]} ROUGH_CRITIC_DIM={dims[1]} ROUGH_ACTION_DIM={dims[2]}")
    print(f"ROUGH_REGRESSION_ENVS={args.num_envs} STEPS={args.steps} NO_NAN=True NO_INF=True")
    env.close()


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
