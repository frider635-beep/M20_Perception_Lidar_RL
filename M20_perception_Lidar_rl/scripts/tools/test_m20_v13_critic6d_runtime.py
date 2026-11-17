#!/usr/bin/env python3
"""Isaac runtime smoke for native V13 plus critic-only reward state."""

from __future__ import annotations

import argparse
import os
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_training.v11_continuous_runner import V11ContinuousRunner


TASK = "M20-Stair-V13-V11FD-Alternation-Critic6D-v0"


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    assert raw.action_manager.total_action_dim == 16
    assert "v133_style" not in raw.reward_manager.active_terms
    assert not any(name.startswith("v133_official_descent") for name in raw.reward_manager.active_terms)
    assert raw.cfg.v11_perception.front_completion_mode == "alternating_risers"
    assert raw.cfg.v11_perception.soft_alternation

    obs = raw.observation_manager.compute()
    assert obs["policy"].shape == (args.num_envs, 63)
    assert obs["critic"].shape == (args.num_envs, 574)
    assert torch.isfinite(obs["critic"]).all()
    reward_state = raw.observation_manager._group_obs_term_names["critic"]
    assert reward_state[-1] == "v13_reward_state"

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = V11ContinuousRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    obs = wrapped.get_observations()
    assert runner.alg.actor.obs_dim == 63 and runner.alg.critic.obs_dim == 574
    with torch.inference_mode():
        actions = runner.alg.actor(obs)
    for _ in range(args.steps):
        obs, reward, dones, _ = wrapped.step(torch.zeros_like(actions))
        assert obs["policy"].shape == (args.num_envs, 63)
        assert obs["critic"].shape == (args.num_envs, 574)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
        assert torch.isfinite(reward).all() and dones.shape == (args.num_envs,)
    print(
        f"V13_CRITIC6D_RUNTIME: PASS envs={args.num_envs} steps={args.steps} "
        "actor_dim=63 critic_dim=574 action_dim=16 native_v13=yes",
        flush=True,
    )
    wrapped.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
