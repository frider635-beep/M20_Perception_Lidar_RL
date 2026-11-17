#!/usr/bin/env python3
"""Isaac runtime smoke for balanced V13 with a 580D asymmetric critic."""

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
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_training.v11_continuous_runner import V11ContinuousRunner

TASK = "M20-Stair-V13-V11FD-Alternation-Critic12D-v0"


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    # Preserve V13's resolved terrain-column topology, then make its existing
    # stair samples start close enough for a deterministic forward-command
    # smoke. Reweighting sub-terrains here would invalidate the already
    # resolved terrain type IDs used by the conditioned reset.
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.v11_perception.stair_reset_distance_range = (0.25, 0.50)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    obs = raw.observation_manager.compute()
    assert raw.action_manager.total_action_dim == 16
    assert obs["policy"].shape == (args.num_envs, 63)
    assert obs["critic"].shape == (args.num_envs, 580)
    assert torch.isfinite(obs["critic"]).all()
    descent_obs = obs["critic"][:, -6:]
    active_seen = bool((descent_obs[:, 0] > 0.5).any())
    names = raw.observation_manager._group_obs_term_names["critic"]
    assert names[-2:] == ["v13_reward_state", "v13_descent_state"]
    assert raw.cfg.v11_perception.front_completion_mode == "alternating_risers"
    assert raw.cfg.v11_perception.soft_alternation
    assert raw.cfg.v11_perception.soft_alternation_command_split
    assert not raw.cfg.v11_perception.enable_persistent_descent_rear_phase
    assert raw.reward_manager.get_term_cfg("v13_descent_rear_extension").weight > 0.0
    assert raw.reward_manager.get_term_cfg("v13_descent_rear_roll").weight > 0.0
    assert raw.reward_manager.get_term_cfg("track_lin_vel_xy_exp").func is mdp.v11_ascent_route_coupled_track_lin_vel_xy_exp
    assert raw.reward_manager.get_term_cfg("track_ang_vel_z_exp").func is mdp.v11_ascent_route_coupled_track_ang_vel_z_exp
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    assert agent_cfg.num_steps_per_env == 24
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = V11ContinuousRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    assert runner.alg.actor.obs_dim == 63 and runner.alg.critic.obs_dim == 580
    action = torch.zeros((args.num_envs, 16), device=raw.device)
    for _ in range(args.steps):
        obs, reward, dones, _ = wrapped.step(action)
        assert obs["policy"].shape == (args.num_envs, 63)
        assert obs["critic"].shape == (args.num_envs, 580)
        assert torch.isfinite(obs["critic"]).all() and torch.isfinite(reward).all()
        descent_obs = obs["critic"][:, -6:]
        active_seen |= bool((descent_obs[:, 0] > 0.5).any())
        assert torch.all(descent_obs[:, 4] >= 0.0)
        assert torch.all((descent_obs[:, 5] == 0.0) | (descent_obs[:, 5] == 0.5) | (descent_obs[:, 5] == 1.0))
    assert active_seen, "descending samples never activated critic course state"
    print(
        f"V13_CRITIC12D_RUNTIME: PASS envs={args.num_envs} steps={args.steps} "
        "actor_dim=63 critic_dim=580 action_dim=16 tread_lock=no rollout=24",
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
