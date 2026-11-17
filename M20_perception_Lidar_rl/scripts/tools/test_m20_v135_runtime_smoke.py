#!/usr/bin/env python3
"""Check V13.5 registration, native ABI/PPO and real Isaac steps."""

import argparse
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--ppo_update", action="store_true")
parser.add_argument("--stage_restore_smoke", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rl_training.v11_continuous_runner import V11ContinuousRunner

TASK = "M20-Stair-V13.5-V11FD-NativeAlternation-v0"


def main():
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    assert agent.class_name == "rl_training.v11_continuous_runner:V11ContinuousRunner"
    assert agent.algorithm.class_name == "PPO"
    assert not agent.resume and agent.load_run is None and agent.load_checkpoint is None
    assert agent.num_steps_per_env == 24 and agent.max_iterations == 30000 and agent.save_interval == 100
    assert agent.policy.class_name == "V11ContinuousActorCritic"
    assert agent.algorithm.learning_rate == 1.0e-3
    assert agent.algorithm.entropy_coef == 3.0e-3
    assert agent.algorithm.desired_kl == 1.0e-2

    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    assert cfg.v11_perception.front_completion_mode == "alternating_risers"
    assert cfg.v11_perception.soft_alternation
    assert not hasattr(cfg, "v133_transition")
    assert cfg.scene.terrain.max_init_terrain_level == 3
    assert cfg.v135_longtrain.rollout_steps_per_update == 24
    assert cfg.v135_longtrain.anneal_updates == 1000
    cfg.v135_stage_starts.enabled = args.stage_restore_smoke
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import v135_terrain_geometry, column_names
    geometry = v135_terrain_geometry(raw)
    assert geometry["step_width"].shape == (args.num_envs,)
    assert geometry["flight_riser_count"].shape == (args.num_envs,)
    assert geometry["max_risers"] == 8
    columns = column_names(cfg.scene.terrain.terrain_generator)
    ascent_columns = cfg.v11_perception.ascending_stair_terrain_type_ids
    actual_widths = tuple(cfg.scene.terrain.terrain_generator.sub_terrains[columns[i]].step_width for i in ascent_columns)
    assert actual_widths == cfg.v135_riser_identity.ascent_tread_widths
    assert cfg.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_width == .30
    print(f"V135_TREAD_COLUMN_WIRING: PASS widths={actual_widths}", flush=True)
    observations = raw.observation_manager.compute()
    assert observations["policy"].shape == (args.num_envs, 63)
    assert observations["critic"].shape == (args.num_envs, 574)
    assert raw.action_manager.total_action_dim == 16
    joint_pos_names = raw.action_manager.get_term("joint_pos")._joint_names
    assert joint_pos_names[:6] == [
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    ], joint_pos_names
    required = {
        "v135_post_stair_flat_progress", "v135_post_stair_flat_completion",
        "v135_post_stair_flat_crawl", "v135_ascent_leader_progress",
        "v135_ascent_leader_velocity_coupled", "v135_descent_rear_transfer_progress",
        "v135_descent_rear_transfer_completion",
        "v135_descent_rear_recovery",
        "v135_adjacent_chain",
        "v135_alternation_cycle",
    }
    assert required.issubset(raw.reward_manager.active_terms)
    expected_weights = {
        "v135_post_stair_flat_progress": 0.5,
        "v135_post_stair_flat_completion": 1.0,
        "v135_post_stair_flat_crawl": 0.5,
        "v135_ascent_leader_progress": 0.3,
        "v135_ascent_leader_velocity_coupled": 0.7,
        "v135_adjacent_chain": 1.0,
        "v135_alternation_cycle": 1.5,
        "v135_descent_rear_transfer_progress": 0.8,
        "v135_descent_rear_transfer_completion": 1.0,
        "v135_valid_riser_completion": 1.5,
        "v135_wrong_side_riser": -0.15,
        "v135_duplicate_riser": -0.10,
        "v135_skip_riser": -0.35,
        "v135_dual_skip_riser": -0.20,
        "v135_descent_rear_recovery": 0.5,
        "landing_event": 5.0,
        "platform_exit_success": 5.0,
    }
    for name, weight in expected_weights.items():
        assert raw.reward_manager.get_term_cfg(name).weight == weight
    for retired in (
        "v135_descent_body_rear_lag",
        "v135_descent_premature_collapse",
        "v135_descent_rear_early_extension",
        "v135_descent_persistent_extension",

        "descent_step_complete", "descent_rear_follow_delay",
        "descent_rear_roll_consistency", "v135_cancel_native_descent_step_completion",
        "v135_cancel_native_descent_follow_delay", "v135_cancel_native_descent_roll_consistency",
    ):
        assert retired not in raw.reward_manager.active_terms, retired
    module = __import__(
        "rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_descent_stance",
        fromlist=["v135_post_stair_flat_state"],
    )
    recovery = module.v135_post_stair_flat_state(raw)
    for name in ("quality", "quality_delta", "clearance_ratio", "completion_event", "flat_blend"):
        assert recovery[name].shape == (args.num_envs,)
    assert torch.isfinite(recovery["quality"]).all()
    assert torch.isfinite(recovery["clearance_ratio"]).all()
    rear = __import__(
        "rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_descent_support_transfer",
        fromlist=["v135_descent_rear_support_transfer_state"],
    ).v135_descent_rear_support_transfer_state(raw)
    assert rear["P_rear"].shape == (args.num_envs,)
    assert (rear["P_rear"] >= 0).all() and (rear["P_rear"] <= 1).all()
    assert rear["rear_extension_ratio"].shape == (args.num_envs, 2)
    assert torch.isfinite(rear["rear_extension_ratio"]).all()
    assert rear["support_quality"].shape == (args.num_envs,)
    assert (rear["support_quality"] >= 0).all() and (rear["support_quality"] <= 1).all()
    for name in ("rear_load_share", "body_path_speed", "body_pitch_rate_abs", "body_rear_lag"):
        assert rear[name].shape == (args.num_envs,)
        assert torch.isfinite(rear[name]).all(), name
    assert (rear["rear_load_share"] >= 0).all() and (rear["rear_load_share"] <= 1).all()
    assert rear["four_wheel_support"].shape == (args.num_envs,)

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    if args.ppo_update:
        runner = V11ContinuousRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
        assert runner.alg.actor.obs_dim == 63 and runner.alg.critic.obs_dim == 574
        assert not runner.alg.actor.is_recurrent and not runner.alg.critic.is_recurrent
        observations = wrapped.get_observations()
        with torch.inference_mode():
            for _ in range(agent.num_steps_per_env):
                actions = runner.alg.act(observations)
                observations, rewards, dones, extras = wrapped.step(actions)
                runner.alg.process_env_step(observations, rewards, dones, extras)
            runner.alg.compute_returns(observations)
        runner.alg.update()
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        from rl_training.v135_training_report import write_manifest
        with TemporaryDirectory() as folder:
            manifest = json.loads(Path(write_manifest(folder, env, runner, agent)).read_text())
            assert manifest["actor_dim"] == 63 and manifest["critic_dim"] == 574
            assert manifest["action_dim"] == 16 and manifest["rollout"] == 24
            assert manifest["reward_weights"]["track_lin_vel_xy_exp"] == 5.0
            assert manifest["source_sha256"]
        print("V135_NATIVE_MLP_PPO_UPDATE: PASS", flush=True)
    actions = torch.zeros((args.num_envs, 16), device=raw.device)
    with torch.inference_mode():
        for _ in range(args.steps):
            observations, rewards, dones, extras = wrapped.step(actions)
            assert observations["policy"].shape == (args.num_envs, 63)
            assert observations["critic"].shape == (args.num_envs, 574)
            assert torch.isfinite(observations["policy"]).all()
            assert torch.isfinite(observations["critic"]).all()
            assert torch.isfinite(rewards).all()
            diag = extras.get("log", {})
            prefix = "Diag/V135Behavior/"
            assert prefix + "Ascent/full_episodes" in diag
            assert prefix + "Descent/frames" in diag
            assert prefix + "Flat/censored_exits" in diag
            assert all(torch.isfinite(value).all() for key, value in diag.items() if key.startswith(prefix))
    if args.stage_restore_smoke:
        with torch.inference_mode():
            # This validates restoration mechanics only, not a successful stair prefix.
            snapshot = raw._capture_stage(0)
            def tensor_bytes(value):
                if isinstance(value, torch.Tensor):
                    return value.numel() * value.element_size()
                if isinstance(value, dict):
                    return sum(tensor_bytes(child) for child in value.values())
                if isinstance(value, (tuple, list)):
                    return sum(tensor_bytes(child) for child in value)
                return 0
            assert tensor_bytes(snapshot) < 2_000_000
            print(f"V135_STAGE_SNAPSHOT_BYTES={tensor_bytes(snapshot)}", flush=True)
            untouched = raw.scene["robot"].data.root_state_w[1:].clone()
            action_before = raw.action_manager.action[0].clone()
            raw.action_manager._action[0] += 0.123
            raw._restore_stage(0, snapshot)
            assert torch.equal(raw.action_manager.action[0], action_before)
            assert torch.equal(raw.scene["robot"].data.root_state_w[1:], untouched)
            assert raw._v135_stage_started[0]
            assert not raw._v135_riser_identity["episode_ordered_valid"][0]
            assert raw._v11_last_episode_length[0] == 0
            for _ in range(8):
                observations, rewards, dones, extras = wrapped.step(actions)
                assert torch.isfinite(observations["policy"]).all() and torch.isfinite(rewards).all()
            print("V135_STAGE_RESTORE_MECHANICS: PASS (not behavioral prefix validation)", flush=True)
    wrapped.close()
    print(
        f"V135_RUNTIME_SMOKE: PASS envs={args.num_envs} steps={args.steps} "
        f"actor=63 critic=574 action=16 ppo_update={'YES' if args.ppo_update else 'NO'}",
        flush=True,
    )
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
