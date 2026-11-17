#!/usr/bin/env python3
"""Isaac runtime smoke for native V13.4 actor, critic and terrain routing."""

from __future__ import annotations

import argparse
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--ppo_update", action="store_true")
parser.add_argument("--release_actor", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_training.v134_style_prior_runner import V134StylePriorRunner

TASK = "M20-Stair-V13.4-V11FD-PerceptiveFollow-v0"


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    obs = raw.observation_manager.compute()
    assert raw.action_manager.total_action_dim == 16
    assert obs["policy"].shape == (args.num_envs, 63)
    assert obs["critic"].shape == (args.num_envs, 580)
    assert torch.isfinite(obs["critic"]).all()
    assert raw.observation_manager._group_obs_term_names["critic"][-1] == "v134_task_state"
    assert raw.cfg.v11_perception.front_completion_mode == "alternating_risers"
    assert raw.cfg.v11_perception.soft_alternation
    assert not raw.cfg.v11_perception.enable_persistent_descent_rear_phase
    assert raw.cfg.v11_perception.platform_target_loss_release_steps == 30
    assert raw.cfg.v133_transition.require_first_riser_seed
    assert raw.reward_manager.get_term_cfg("v134_ascent_chain").func is mdp.v134_positive_ascent_chain_reward
    assert raw.reward_manager.get_term_cfg("v134_descent_rear_extension").func is mdp.v13_descent_rear_extension_penalty
    assert raw.reward_manager.get_term_cfg("v134_descent_rear_knee_flexion").func is mdp.v134_descent_rear_knee_flexion_penalty
    assert raw.reward_manager.get_term_cfg("v134_descent_rear_roll").func is mdp.v13_descent_rear_roll_penalty
    assert raw.reward_manager.get_term_cfg("v134_descent_rear_thigh_parallel").func is mdp.v134_descent_rear_thigh_parallel_penalty
    assert raw.reward_manager.get_term_cfg("v134_platform_body_clearance").func is mdp.v134_platform_body_clearance_penalty
    assert not any(name.startswith("v134_descent_follow") for name in raw.reward_manager.active_terms)
    assert "v133_style" not in raw.reward_manager.active_terms
    assert not any(name.startswith("v133_official_descent") for name in raw.reward_manager.active_terms)

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    assert agent_cfg.num_steps_per_env == 24 and agent_cfg.max_iterations == 30000
    assert agent_cfg.algorithm.entropy_coef == 1.0e-3
    assert agent_cfg.algorithm.learning_rate == 1.0e-4
    assert agent_cfg.v134_teacher_burst_start_probability == 0.025
    assert agent_cfg.v134_teacher_burst_end_probability == 0.005
    assert agent_cfg.v134_teacher_burst_decay_updates == 6000
    assert agent_cfg.v134_teacher_burst_steps == 8
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = V134StylePriorRunner(
        wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device,
    )
    assert runner.alg.actor.obs_dim == 63 and runner.alg.critic.obs_dim == 580
    assert runner.alg._v134_teacher_burst_start_probability == 0.025
    assert runner.alg._v134_teacher_burst_end_probability == 0.005
    assert runner.alg._v134_teacher_burst_decay_updates == 6000
    assert runner.alg._v134_teacher_burst_steps == 8
    assert all(not p.requires_grad for p in runner.alg._v134_teacher_actor.parameters())
    assert any(
        not torch.equal(runner.alg.actor.state_dict()[key], value)
        for key, value in runner.alg._v134_teacher_actor.state_dict().items()
    )
    runner_obs = wrapped.get_observations()
    with torch.inference_mode():
        sampled = runner.alg.actor(runner_obs)
    assert sampled.shape == (args.num_envs, 16) and torch.isfinite(sampled).all()
    if args.ppo_update:
        actor_before = {
            key: value.detach().clone() for key, value in runner.alg.actor.state_dict().items()
        }
        # Force burst starts in the smoke so the actor-mask path is exercised
        # independently of random terrain allocation and Bernoulli sampling.
        configured_start = runner.alg._v134_teacher_burst_start_probability
        configured_end = runner.alg._v134_teacher_burst_end_probability
        runner.alg._v134_teacher_burst_start_probability = 1.0
        runner.alg._v134_teacher_burst_end_probability = 1.0
        obs = runner_obs
        for _ in range(agent_cfg.num_steps_per_env):
            with torch.inference_mode():
                actions = runner.alg.act(obs)
            obs, rewards, dones, extras = wrapped.step(actions)
            runner.alg.process_env_step(obs, rewards, dones, extras)
        teacher_count = int(runner.alg.storage.v134_teacher_action_masks.sum().item())
        assert teacher_count > 0
        with torch.inference_mode():
            runner.alg.compute_returns(obs)
        losses = runner.alg.update()
        changed = any(
            not torch.equal(value, actor_before[key])
            for key, value in runner.alg.actor.state_dict().items()
        )
        assert changed
        assert losses["v134_teacher_action_rate"] > 0.0
        assert losses["v134_teacher_action_rate_given_eligible"] > 0.0
        assert losses["v134_teacher_burst_count"] > 0.0
        assert losses["v134_teacher_actor_samples_masked"] == float(teacher_count)
        assert losses["v134_teacher_burst_probability"] == 1.0
        assert not runner.alg.storage.v134_teacher_action_masks.any()
        runner.alg._v134_teacher_burst_start_probability = configured_start
        runner.alg._v134_teacher_burst_end_probability = configured_end
        print(
            "V134_STYLE_PPO_UPDATE: PASS "
            f"changed={int(changed)} teacher_actions={teacher_count} "
            f"action_rate={losses['v134_teacher_action_rate']:.6g} "
            f"eligible_rate={losses['v134_teacher_action_rate_given_eligible']:.6g} "
            f"bursts={losses['v134_teacher_burst_count']:.0f}",
            flush=True,
        )
    action = torch.zeros_like(sampled)
    for _ in range(args.steps):
        obs, reward, dones, _ = wrapped.step(action)
        assert obs["policy"].shape == (args.num_envs, 63)
        assert obs["critic"].shape == (args.num_envs, 580)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
        assert torch.isfinite(reward).all() and dones.shape == (args.num_envs,)
    print(
        f"V134_RUNTIME: PASS envs={args.num_envs} steps={args.steps} "
        "actor=63 critic=580 action=16 runner=sparse_ascent_teacher_rollout rollout=24 "
        "ascent=positive_adjacent_chain descent=natural_posture_envelope",
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
