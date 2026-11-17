#!/usr/bin/env python3
"""Small Isaac runtime and fresh-network smoke for V13.3 (no learning)."""

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
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (
    DeeproboticsM20V133AdjacentTransitionPPORunnerCfg,
)
from rl_training.v133_transition_runner import V133TransitionRunner


TASK = "M20-Stair-V13.3-V11FD-FromScratch-v0"


def main() -> None:
    os.environ.pop("V133_MOTOR_PRIOR_CHECKPOINT", None)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    assert raw.action_manager.total_action_dim == 16
    assert raw.scene.terrain.cfg.max_init_terrain_level == 5
    assert raw.cfg.scene.terrain.terrain_generator.curriculum is True
    terrain = raw.cfg.scene.terrain.terrain_generator.sub_terrains
    assert abs(sum(float(value.proportion) for value in terrain.values()) - 1.0) < 1.0e-9
    for term in (
        "v13_opposite_progress", "v13_velocity_coupled_opposite_progress",
        "v13_same_side_progress", "v13_simultaneous_progress",
        "v13_alternating_completion", "v13_same_side_completion",
    ):
        assert term in raw.reward_manager.active_terms
    style_cfg = raw.reward_manager.get_term_cfg("v133_style")
    assert style_cfg.func is mdp.v133_three_step_chain_reward and float(style_cfg.weight) == 1.0
    assert float(raw.cfg.v133_transition.adjacent_progress_weight) == 0.50
    assert not raw.cfg.v11_perception.enable_persistent_descent_rear_phase
    assert "descent_step_complete" not in raw.reward_manager.active_terms
    assert "descent_rear_support_progress" not in raw.reward_manager.active_terms
    assert "descent_rear_follow_delay" not in raw.reward_manager.active_terms
    assert "descent_rear_roll_consistency" not in raw.reward_manager.active_terms
    assert raw.reward_manager.get_term_cfg("flat_orientation_l2").func is mdp.v11_phase_flat_orientation_l2
    assert raw.reward_manager.get_term_cfg("lin_vel_z_l2").func is mdp.v11_phase_lin_vel_z_l2
    assert raw.reward_manager.get_term_cfg("action_rate_l2").func is mdp.v133_route_action_rate
    assert raw.reward_manager.get_term_cfg("action_smooth_l2").func is mdp.v133_route_action_smooth
    assert raw.reward_manager.get_term_cfg("hipy_joint_pos_penalty").func is mdp.v133_route_front_joint_pos_penalty
    assert raw.reward_manager.get_term_cfg("knee_joint_pos_penalty").func is mdp.v133_route_front_joint_pos_penalty
    correction_weights = {
        "v133_official_descent_flat_orientation": -1.0,
        "v133_official_descent_lin_vel_z": -1.0,
        "v133_official_descent_action_rate": 0.004,
        "v133_official_descent_ang_vel_xy": 0.04,
        "v133_official_descent_hipx": -2.6,
        "v133_official_descent_hipy": -1.4,
        "v133_official_descent_knee": -0.65,
        "v133_official_descent_joint_limit_cancel": 5.0,
    }
    for term_name, weight in correction_weights.items():
        term_cfg = raw.reward_manager.get_term_cfg(term_name)
        assert float(term_cfg.weight) == weight
        assert torch.isfinite(term_cfg.func(raw, **term_cfg.params)).all()
    routed = {
        "world_forward_deficit": mdp.v133_ascent_world_forward_deficit,
        "world_forward_progress": mdp.v133_ascent_world_forward_progress,
        "path_lateral_deviation": mdp.v133_ascent_path_lateral_deviation,
        "path_heading_error": mdp.v133_ascent_path_heading_error,
        "non_wheel_leg_contact": mdp.v133_ascent_non_wheel_contact,
        "non_wheel_leg_scrape": mdp.v133_ascent_non_wheel_scrape,
        "wheel_impact": mdp.v133_ascent_wheel_impact,
        "stair_straightness": mdp.v133_ascent_stair_straightness,
        "front_riser_contact": mdp.v133_ascent_front_riser_contact,
    }
    for term_name, routed_func in routed.items():
        assert raw.reward_manager.get_term_cfg(term_name).func is routed_func
    cancelled = (
        "is_terminated", "landing_event", "platform_exit_success", "clearance_progress",
        "pure_yaw_xy_drift", "front_hipy_participation_settlement", "front_knee_excess",
        "landing_center", "landing_edge", "v13_opposite_progress",
        "v13_velocity_coupled_opposite_progress", "v13_same_side_progress",
        "v13_simultaneous_progress", "v13_alternating_completion",
        "v13_same_side_completion", "v133_style",
    )
    for source_name in cancelled:
        source = raw.reward_manager.get_term_cfg(source_name)
        cancel = raw.reward_manager.get_term_cfg(f"v133_official_descent_cancel_{source_name}")
        assert cancel.func is mdp.v133_cancel_nonofficial_descent_reward
        assert float(cancel.weight) == -float(source.weight)
    assert raw.reward_manager.get_term_cfg("joint_mirror").func is mdp.joint_mirror
    for term_name in ("action_rate_l2", "action_smooth_l2"):
        assert "descent_rear_hipy_scale" not in raw.reward_manager.get_term_cfg(term_name).params
        assert "descent_rear_knee_scale" not in raw.reward_manager.get_term_cfg(term_name).params
    terrain_params = raw.cfg.curriculum.terrain_levels.params
    assert terrain_params["official_descent_curriculum"] is True
    assert raw.cfg.terminations.illegal_contact.func is mdp.v133_nonofficial_illegal_contact_termination
    assert raw.cfg.terminations.front_riser_collision.func is mdp.v133_ascent_front_riser_collision_termination

    # Force one sample onto a descending column and verify the algebraic sum
    # of inherited plus correction terms equals the official M20 weights.
    terrain_types = raw.scene.terrain.terrain_types
    saved_type = terrain_types[0].clone()
    descending_ids = sorted(
        set(raw.cfg.v11_perception.stair_terrain_type_ids)
        - set(raw.cfg.v11_perception.ascending_stair_terrain_type_ids)
    )
    terrain_types[0] = descending_ids[0]
    assert mdp.v133_descending_stair_mask(raw)[0]
    assert not mdp.v133_nonofficial_illegal_contact_termination(
        raw, **raw.cfg.terminations.illegal_contact.params
    )[0]
    assert not mdp.v133_ascent_front_riser_collision_termination(
        raw, **raw.cfg.terminations.front_riser_collision.params
    )[0]
    command_term = raw.command_manager.get_term("base_velocity")
    command_term._resample(torch.tensor([0], device=raw.device, dtype=torch.long))
    torch.testing.assert_close(
        raw.command_manager.get_command("base_velocity")[0, 1:],
        torch.zeros(2, device=raw.device),
    )

    def term_value(name):
        term_cfg = raw.reward_manager.get_term_cfg(name)
        return term_cfg.func(raw, **term_cfg.params)

    official_flat = mdp.flat_orientation_l2(raw)[0]
    actual_flat = -0.5 * term_value("flat_orientation_l2")[0] - term_value(
        "v133_official_descent_flat_orientation"
    )[0]
    torch.testing.assert_close(actual_flat, -50.0 * official_flat)
    official_lin_z = mdp.lin_vel_z_l2(raw)[0]
    actual_lin_z = -2.0 * term_value("lin_vel_z_l2")[0] - term_value(
        "v133_official_descent_lin_vel_z"
    )[0]
    torch.testing.assert_close(actual_lin_z, -2.0 * official_lin_z)
    torch.testing.assert_close(
        -0.06 * mdp.ang_vel_xy_l2(raw)[0]
        + 0.04 * term_value("v133_official_descent_ang_vel_xy")[0],
        -0.02 * mdp.ang_vel_xy_l2(raw)[0],
    )
    torch.testing.assert_close(
        -0.014 * term_value("action_rate_l2")[0]
        + 0.004 * term_value("v133_official_descent_action_rate")[0],
        -0.01 * term_value("v133_official_descent_action_rate")[0],
    )
    for base_name, correction_name, base_weight, official_weight in (
        ("hipx_joint_pos_penalty", "v133_official_descent_hipx", -0.4, -3.0),
        ("hipy_joint_pos_penalty", "v133_official_descent_hipy", -0.1, -1.5),
        ("knee_joint_pos_penalty", "v133_official_descent_knee", -0.1, -0.75),
    ):
        official_cost = term_value(correction_name)[0]
        torch.testing.assert_close(
            base_weight * term_value(base_name)[0]
            + (official_weight - base_weight) * official_cost,
            official_weight * official_cost,
        )
    torch.testing.assert_close(
        -5.0 * term_value("joint_pos_limits")[0]
        + 5.0 * term_value("v133_official_descent_joint_limit_cancel")[0],
        torch.zeros((), device=raw.device),
    )
    for term_name in routed:
        torch.testing.assert_close(term_value(term_name)[0], torch.zeros((), device=raw.device))
    for source_name in cancelled:
        source_cfg = raw.reward_manager.get_term_cfg(source_name)
        cancel_name = f"v133_official_descent_cancel_{source_name}"
        torch.testing.assert_close(
            float(source_cfg.weight) * term_value(source_name)[0]
            - float(source_cfg.weight) * term_value(cancel_name)[0],
            torch.zeros((), device=raw.device),
        )

    # Descent corrections must be zero on ascent, and chain teaching must be
    # zero on every terrain column other than the configured ascent column.
    ascending_ids = sorted(raw.cfg.v11_perception.ascending_stair_terrain_type_ids)
    terrain_types[0] = ascending_ids[0]
    assert mdp.v133_ascending_stair_mask(raw)[0] and not mdp.v133_descending_stair_mask(raw)[0]
    for term_name in correction_weights:
        torch.testing.assert_close(term_value(term_name)[0], torch.zeros((), device=raw.device))
    non_stair_ids = sorted(
        set(range(raw.scene.terrain.cfg.terrain_generator.num_cols))
        - set(raw.cfg.v11_perception.stair_terrain_type_ids)
    )
    terrain_types[0] = non_stair_ids[0]
    assert not mdp.v133_ascending_stair_mask(raw)[0] and not mdp.v133_descending_stair_mask(raw)[0]
    chain_state = mdp.v133_three_step_chain_state(raw)
    saved_adjacent = chain_state["adjacent_event"][0].clone()
    saved_chain = chain_state["chain_event"][0].clone()
    chain_state["adjacent_event"][0] = True
    chain_state["chain_event"][0] = True
    torch.testing.assert_close(term_value("v133_style")[0], torch.zeros((), device=raw.device))
    chain_state["adjacent_event"][0] = saved_adjacent
    chain_state["chain_event"][0] = saved_chain
    terrain_types[0] = saved_type

    agent_cfg = DeeproboticsM20V133AdjacentTransitionPPORunnerCfg()
    assert agent_cfg.num_steps_per_env == 32 and agent_cfg.max_iterations == 20000
    assert agent_cfg.algorithm.num_mini_batches == 8 and agent_cfg.save_interval == 250
    assert agent_cfg.algorithm.learning_rate == 1.0e-3 and agent_cfg.algorithm.desired_kl == 0.01
    assert not agent_cfg.resume and agent_cfg.load_run is None and agent_cfg.load_checkpoint is None
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = V133TransitionRunner(
        wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device,
    )
    assert not runner.alg.optimizer.state
    obs = wrapped.get_observations()
    assert obs["policy"].shape == (args.num_envs, 63)
    with torch.inference_mode():
        actions = runner.alg.actor(obs)
    assert actions.shape == (args.num_envs, 16) and torch.isfinite(actions).all()

    for _ in range(args.steps):
        obs, reward, dones, _ = wrapped.step(torch.zeros_like(actions))
        transition = mdp.v133_three_step_chain_state(raw)
        assert obs["policy"].shape == (args.num_envs, 63)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        for event in (
            "seed_event", "adjacent_event", "chain_event", "repeat_event",
            "follower_event", "skip_event", "simultaneous_event",
        ):
            assert transition[event].dtype == torch.bool
        assert torch.isin(transition["last_side"], torch.tensor([-1, 0, 1], device=raw.device)).all()
        for value in (
            "adjacent_progress_best", "adjacent_progress_escrow",
            "adjacent_progress_delta", "adjacent_progress_clawback",
        ):
            assert torch.isfinite(transition[value]).all()
        assert torch.isfinite(runner.alg.actor(obs)).all()
        assert dones.shape == (args.num_envs,)

    print(
        f"V133_RUNTIME_STEP_SMOKE: PASS envs={args.num_envs} steps={args.steps} "
        "actor_dim=63 action_dim=16 from_scratch=yes rollout=32",
        flush=True,
    )
    wrapped.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        # Isaac standalone shutdown can hang or rewrite an assertion to status
        # 0.  A failed smoke owns no persistent state, so exit immediately.
        os._exit(1)
