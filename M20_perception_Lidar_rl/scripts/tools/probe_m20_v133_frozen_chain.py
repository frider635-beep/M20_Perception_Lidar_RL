#!/usr/bin/env python3
"""Run a frozen 63D actor and report V13.3 chain events on fixed ascent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_training.v133_transition_runner import V133TransitionRunner


TASK = "M20-Stair-V13.3-V11FD-FromScratch-v0"


def main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum = False
    generator.num_rows = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    for terrain in generator.sub_terrains.values():
        terrain.proportion = 0.0
    ascent = generator.sub_terrains["pyramid_stairs_inv"]
    ascent.proportion = 1.0
    ascent.step_height_range = (0.12, 0.12)
    cfg.curriculum.terrain_levels = None

    env = gym.make(TASK, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    runner = V133TransitionRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    payload = torch.load(args.checkpoint, map_location=runner.device, weights_only=False)
    runner.alg.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    obs, _ = wrapped.reset()
    finite = True
    episodes = 0
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = policy(obs)
            obs, reward, dones, _ = wrapped.step(actions)
            finite = finite and all(torch.isfinite(value).all().item() for value in (
                obs["policy"], actions, reward,
            ))
            episodes += int(dones.sum())
            if not finite:
                break
    state = wrapped.unwrapped._v133_chain
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "steps": args.steps,
        "num_envs": args.num_envs,
        "episodes": episodes,
        "finite": finite,
    }
    for name in (
        "seed", "adjacent", "chain", "repeat", "follower", "skip", "simultaneous",
        "chain_fl", "chain_fr",
    ):
        result[name] = int(state[f"{name}_count"])
    result["adjacent_progress_delta_sum"] = float(state["adjacent_progress_delta_sum"])
    result["adjacent_progress_clawback_sum"] = float(state["adjacent_progress_clawback_sum"])
    result["adjacent_progress_active_samples"] = int(state["adjacent_progress_active_samples"])
    print("V133_FROZEN_CHAIN_PROBE=" + json.dumps(result, sort_keys=True), flush=True)
    wrapped.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
