#!/usr/bin/env python3
"""Attribute first-riser reverse motion without changing the environment or policy."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--envs-per-condition", type=int, default=4)
parser.add_argument("--command-x", type=float, default=0.4)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--stair-height", type=float, default=0.14)
parser.add_argument("--train-faithful", action="store_true", help="Use training observation/lifecycle/actuator settings; normal perception only.")
parser.add_argument("--trials", type=int, default=5)
parser.add_argument("--mask-b-after-first", action="store_true")
parser.add_argument("--force-b-after-first", action="store_true")
parser.add_argument("--pause-after-first-delay-steps", type=int, default=-1)
parser.add_argument("--pause-duration-steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import rl_training.tasks  # noqa: F401
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.v10_unified_policy import UnifiedActor


HEIGHT = args.stair_height
def fixed_official_stair_terrain(cfg, num_cols: int):
    terrain = copy.deepcopy(cfg.scene.terrain.terrain_generator)
    stair = copy.deepcopy(terrain.sub_terrains["pyramid_stairs_inv"])
    stair.proportion = 1.0
    stair.step_height_range = (HEIGHT, HEIGHT)
    terrain.sub_terrains = {"pyramid_stairs_inv": stair}
    terrain.seed = 23000
    terrain.curriculum = False
    terrain.difficulty_range = (0.0, 0.0)
    terrain.num_rows = 1
    terrain.num_cols = num_cols
    terrain.use_cache = False
    return terrain


def disable_randomization_except_stair_reset(cfg) -> None:
    for name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
        "randomize_reset_joints",
        "randomize_actuator_gains",
        "randomize_push_robot",
        "push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)


def load_actor(checkpoint_path: str, device: str) -> tuple[UnifiedActor, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = UnifiedActor().to(device).eval()
    state = {
        key.removeprefix("actor."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("actor.")
    }
    actor.load_state_dict(state, strict=True)
    return actor, int(checkpoint.get("iter", -1))


def main() -> None:
    if args.mask_b_after_first and args.force_b_after_first:
        raise ValueError("choose at most one b intervention")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_each = args.trials if args.train_faithful else args.envs_per_condition
    num_envs = n_each if args.train_faithful else 2 * n_each

    cfg = parse_env_cfg("M20-Stair-Unified-Perceptive-v0", device=args.device, num_envs=num_envs)
    if not args.train_faithful:
        disable_randomization_except_stair_reset(cfg)
    cfg.seed = 23000
    cfg.scene.num_envs = num_envs
    cfg.scene.terrain.terrain_generator = fixed_official_stair_terrain(cfg, num_envs)
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.gait_level = None
    if hasattr(cfg.curriculum, "command_levels"):
        cfg.curriculum.command_levels = None

    # All generated columns are full ascending stairs for V10 topology labels.
    cfg.v10_perception.single_stair_column_start = -1
    cfg.v10_perception.single_stair_column_end = -1
    cfg.v10_perception.full_stair_column_start = 0
    cfg.v10_perception.full_stair_column_end = num_envs
    cfg.v10_perception.num_terrain_columns = num_envs
    if not args.train_faithful:
        cfg.v10_perception.height_noise_std = 0.0
        cfg.v10_perception.distance_noise_std = 0.0
        cfg.v10_perception.frame_hold_probability = 0.0
        cfg.v10_perception.stair_reset_distance_range = (1.10, 1.10)
        cfg.v10_perception.stair_reset_lateral_range = (0.0, 0.0)
        cfg.v10_perception.stair_reset_yaw_jitter_range = (0.0, 0.0)
        cfg.events.randomize_reset_base.params.update({
            "stair_distance_range": (1.10, 1.10),
            "stair_lateral_range": (0.0, 0.0),
            "stair_yaw_jitter_range": (0.0, 0.0),
        })
        cfg.observations.policy.enable_corruption = False
        for actuator in cfg.scene.robot.actuators.values():
            if hasattr(actuator, "min_delay"):
                actuator.min_delay = 0
            if hasattr(actuator, "max_delay"):
                actuator.max_delay = 0

    cfg.commands.base_velocity.ranges.lin_vel_x = (args.command_x, args.command_x)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
    cfg.commands.base_velocity.rel_only_lin_y_envs = 0.0
    cfg.commands.base_velocity.rel_only_lin_x_envs = 0.0
    cfg.commands.base_velocity.rel_only_ang_z_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (100.0, 100.0)
    cfg.episode_length_s = args.steps * cfg.decimation * cfg.sim.dt

    env = gym.make("M20-Stair-Unified-Perceptive-v0", cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset()
    actor, checkpoint_iteration = load_actor(args.checkpoint, raw.device)
    policy_obs = obs["policy"]
    if tuple(policy_obs.shape) != (num_envs, 60) or raw.action_manager.total_action_dim != 16:
        raise RuntimeError(
            f"alignment failure obs={tuple(policy_obs.shape)} action={raw.action_manager.total_action_dim}"
        )

    robot = raw.scene["robot"]
    policy_joint_ids, policy_joint_names = robot.find_joints(raw.cfg.joint_names, preserve_order=True)
    if policy_joint_names != list(raw.cfg.joint_names):
        raise RuntimeError(f"policy joint mapping failure: {policy_joint_names}")
    print(
        "ISAAC_INITIAL_STATE "
        f"joint_names={policy_joint_names} "
        f"joint_pos={robot.data.joint_pos[0, policy_joint_ids].detach().cpu().tolist()} "
        f"default_joint_pos={robot.data.default_joint_pos[0, policy_joint_ids].detach().cpu().tolist()} "
        f"root_pos={robot.data.root_pos_w[0].detach().cpu().tolist()} "
        f"root_quat={robot.data.root_quat_w[0].detach().cpu().tolist()} "
        f"projected_gravity={robot.data.projected_gravity_b[0].detach().cpu().tolist()}",
        flush=True,
    )
    initial_state = {
        "joint_names": policy_joint_names,
        "joint_pos": robot.data.joint_pos[0, policy_joint_ids].detach().cpu().tolist(),
        "joint_vel": robot.data.joint_vel[0, policy_joint_ids].detach().cpu().tolist(),
        "root_pos_local": (
            robot.data.root_pos_w[0] - raw.scene.env_origins[0]
        ).detach().cpu().tolist(),
        "root_quat_wxyz": robot.data.root_quat_w[0].detach().cpu().tolist(),
        "root_lin_vel_w": robot.data.root_lin_vel_w[0].detach().cpu().tolist(),
        "root_ang_vel_w": robot.data.root_ang_vel_w[0].detach().cpu().tolist(),
        "projected_gravity": robot.data.projected_gravity_b[0].detach().cpu().tolist(),
    }
    initial_state_path = output_dir / "isaac_initial_state.json"
    initial_state_path.write_text(json.dumps(initial_state, indent=2) + "\n")
    print(f"ISAAC_INITIAL_STATE_JSON={initial_state_path}", flush=True)
    fr_ids, fr_names = robot.find_bodies("fr_wheel")
    hr_ids, hr_names = robot.find_bodies("hr_wheel")
    if fr_names != ["fr_wheel"] or hr_names != ["hr_wheel"]:
        raise RuntimeError(f"wheel mapping failure: FR={fr_names}, HR={hr_names}")
    fr_id, hr_id = fr_ids[0], hr_ids[0]

    wheel_joint_ids, wheel_joint_names = robot.find_joints(raw.cfg.wheel_joint_names, preserve_order=True)
    if wheel_joint_names != list(raw.cfg.wheel_joint_names):
        raise RuntimeError(f"wheel joint mapping failure: {wheel_joint_names}")
    wheel_term = raw.action_manager.get_term("joint_vel")
    leg_term = raw.action_manager.get_term("joint_pos")
    reward_names = list(raw.reward_manager._term_names)
    reward_indices = {name: reward_names.index(name) for name in (
        "first_step_success", "perception_miss", "joint_success",
        "fr_clearance_progress", "track_lin_vel_xy_exp",
    )}
    print(
        "ACTION_ALIGNMENT "
        f"terms={raw.action_manager.active_terms} dims={raw.action_manager.action_term_dim} "
        f"wheel_names={wheel_joint_names} wheel_scale={raw.cfg.actions.joint_vel.scale}",
        flush=True,
    )

    active = torch.ones(num_envs, device=raw.device, dtype=torch.bool)
    first_seen_step = torch.full((num_envs,), -1, device=raw.device, dtype=torch.long)
    rows = []

    for step in range(args.steps):
        state = mdp.v10_state(raw)
        new_first = state["first_step_success_latched"] & (first_seen_step < 0)
        first_seen_step = torch.where(new_first, torch.full_like(first_seen_step, step), first_seen_step)
        pause_active = (
            (args.pause_after_first_delay_steps >= 0)
            & (first_seen_step >= 0)
            & (step >= first_seen_step + args.pause_after_first_delay_steps)
            & (step < first_seen_step + args.pause_after_first_delay_steps + args.pause_duration_steps)
        )
        command_buffer = raw.command_manager.get_command("base_velocity")
        command_buffer[:, 0] = args.command_x
        command_buffer[:, 1:3] = 0.0
        command_buffer[pause_active] = 0.0
        actor_obs = policy_obs.clone()
        actor_obs[:, 6:9] = command_buffer
        if args.mask_b_after_first:
            actor_obs[state["first_step_success_latched"], 57] = 0.0
        if args.force_b_after_first:
            actor_obs[state["first_step_success_latched"], 57] = 1.0
        with torch.inference_mode():
            action = actor(actor_obs)

            # Strict intervention: identical 59 dimensions, change b only.
            obs_b0 = actor_obs.clone()
            obs_b1 = actor_obs.clone()
            obs_b0[:, 57] = 0.0
            obs_b1[:, 57] = 1.0
            action_b0 = actor(obs_b0)
            action_b1 = actor(obs_b1)

        command = raw.command_manager.get_command("base_velocity")
        command_norm = torch.linalg.norm(command[:, :2], dim=1).clamp_min(1.0e-6)
        command_unit = command[:, :2] / command_norm[:, None]
        base_v_before = (robot.data.root_lin_vel_b[:, :2] * command_unit).sum(dim=1)
        if "tracked_x" in state:
            tracked_xy = torch.stack((state["tracked_x"], state["tracked_y"]), dim=1)
            tracked_local = mdp.v10_world_target_relative(
                robot.data.root_pos_w[:, :2], robot.data.heading_w, tracked_xy
            )
            target_norm = torch.linalg.norm(tracked_local, dim=1).clamp_min(1.0e-6)
            approach_speed = (command[:, :2] * tracked_local).sum(dim=1) / target_norm
            in_window = (
                (state["locked_d_filtered"] >= float(raw.cfg.v10_perception.anticipation_distance_min))
                & (state["locked_d_filtered"] <= float(raw.cfg.v10_perception.anticipation_distance_max))
                & (tracked_local[:, 1].abs() <= float(raw.cfg.v10_perception.corridor_half_width))
            )
            phase_tensor = state["target_phase"]
        else:
            # Legacy one-shot lifecycle on the successful vx-only checkpoint host.
            tracked_local = torch.stack(
                (state["locked_d_filtered"], torch.zeros_like(state["locked_d_filtered"])), dim=1
            )
            approach_speed = command[:, 0]
            in_window = (
                (state["locked_d_filtered"] >= float(raw.cfg.v10_perception.anticipation_distance_min))
                & (state["locked_d_filtered"] <= float(raw.cfg.v10_perception.anticipation_distance_max))
            )
            phase_tensor = torch.where(
                state["first_riser_consumed"], torch.full_like(state["valid_count"], 3),
                torch.where(
                    state["first_riser_triggered"], torch.full_like(state["valid_count"], 2),
                    torch.where(state["first_riser_locked"], torch.ones_like(state["valid_count"]), state["valid_count"] * 0),
                ),
            )

        next_obs, total_reward, terminated, truncated, _ = env.step(action)
        policy_obs = next_obs["policy"]
        processed_wheel = wheel_term.processed_actions.detach().clone()
        processed_leg = leg_term.processed_actions.detach().clone()
        actual_wheel = robot.data.joint_vel[:, wheel_joint_ids].detach().clone()
        base_v_after = (robot.data.root_lin_vel_b[:, :2] * command_unit).sum(dim=1)
        state_after = mdp.v10_state(raw)

        for env_id in range(num_envs):
            if not bool(active[env_id].item()):
                continue
            row = {
                "step": step,
                "env_id": env_id,
                "command_vx": float(command[env_id, 0].item()),
                "command_vy": float(command[env_id, 1].item()),
                "command_wz": float(command[env_id, 2].item()),
                "pause_active": int(pause_active[env_id].item()),
                "base_v_along_command_before": float(base_v_before[env_id].item()),
                "base_v_along_command_after": float(base_v_after[env_id].item()),
                "base_vx_body_after": float(robot.data.root_lin_vel_b[env_id, 0].item()),
                "base_vy_body_after": float(robot.data.root_lin_vel_b[env_id, 1].item()),
                "target_phase": int(phase_tensor[env_id].item()),
                "locked": int(state["first_riser_locked"][env_id].item()),
                "visible": int(state["first_riser_visible"][env_id].item()),
                "consumed": int(state["first_riser_consumed"][env_id].item()),
                "triggered": int(state["first_riser_triggered"][env_id].item()),
                "b_raw": int(state["b_raw"][env_id].item()),
                "actor_b": float(actor_obs[env_id, 57].item()),
                "actor_h": float(actor_obs[env_id, 58].item()),
                "actor_d": float(actor_obs[env_id, 59].item()),
                "locked_h": float(state["locked_h"][env_id].item()),
                "locked_d": float(state["locked_d_filtered"][env_id].item()),
                "tracked_local_x": float(tracked_local[env_id, 0].item()),
                "tracked_local_y": float(tracked_local[env_id, 1].item()),
                "approach_speed": float(approach_speed[env_id].item()),
                "in_window": int(in_window[env_id].item()),
                "first_success_before": int(state["first_step_success_latched"][env_id].item()),
                "first_success_after": int(state_after["first_step_success_latched"][env_id].item()),
                "full_success_after": int(state_after["full_stair_success_latched"][env_id].item()),
                "first_step_event": int(state_after["first_step_event"][env_id].item()),
                "joint_event": int(state_after["joint_event"][env_id].item()),
                "total_reward": float(total_reward[env_id].item()),
                "terminated": int(terminated[env_id].item()),
                "truncated": int(truncated[env_id].item()),
            }
            for reward_name, reward_index in reward_indices.items():
                row[f"reward_{reward_name}"] = float(
                    (raw.reward_manager._step_reward[env_id, reward_index] * raw.step_dt).item()
                )
            for index, name in enumerate(wheel_joint_names):
                short = name.replace("_wheel_joint", "")
                row[f"raw_{short}"] = float(action[env_id, 12 + index].item())
                row[f"processed_{short}"] = float(processed_wheel[env_id, index].item())
                row[f"actual_vel_{short}"] = float(actual_wheel[env_id, index].item())
                row[f"cf_b0_raw_{short}"] = float(action_b0[env_id, 12 + index].item())
                row[f"cf_b1_raw_{short}"] = float(action_b1[env_id, 12 + index].item())
                row[f"cf_delta_raw_{short}"] = float(
                    (action_b1[env_id, 12 + index] - action_b0[env_id, 12 + index]).item()
                )
            rows.append(row)

        ended = active & (terminated.bool() | truncated.bool())
        active &= ~ended
        if not active.any():
            break

    csv_path = output_dir / "reverse_attribution_timeseries.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    wheel_keys = [name.replace("_wheel_joint", "") for name in wheel_joint_names]
    lines = [
        "REVERSE_ATTRIBUTION_RUNTIME=PASS",
        f"CHECKPOINT_ITERATION={checkpoint_iteration}",
        f"TERRAIN=fixed_official_stair_{HEIGHT:.3f}m",
        f"COMMAND=({args.command_x:.3f},0,0)_fixed_100s",
        f"MASK_B_AFTER_FIRST={int(args.mask_b_after_first)}",
        f"FORCE_B_AFTER_FIRST={int(args.force_b_after_first)}",
        "BASE_V_SIGN=projection of body-frame XY velocity onto normalized XY command; negative means physical motion opposite command",
        "PROCESSED_WHEEL_SIGN=joint-coordinate velocity target exactly held by ActionManager; M20 training diagnostic documents negative as normal +X travel",
        "THRESHOLDS=-0.1/-0.3 m/s are reporting levels only, not lifecycle parameters",
    ]
    summary_rows = []
    for env_id in range(num_envs):
        selected = [row for row in rows if row["env_id"] == env_id]
        if not selected:
            continue
        min_row = min(selected, key=lambda row: row["base_v_along_command_after"])
        first_on = next((row["step"] for row in selected if row["actor_b"] > 0.5), -1)
        first_success = next((row["step"] for row in selected if row["first_success_after"]), -1)
        first_reverse_01 = next((row["step"] for row in selected if row["base_v_along_command_after"] < -0.1), -1)
        first_reverse_03 = next((row["step"] for row in selected if row["base_v_along_command_after"] < -0.3), -1)
        b_toggles = sum(
            int((selected[i]["actor_b"] > 0.5) != (selected[i - 1]["actor_b"] > 0.5))
            for i in range(1, len(selected))
        )
        processed = [min_row[f"processed_{key}"] for key in wheel_keys]
        actual = [min_row[f"actual_vel_{key}"] for key in wheel_keys]
        raw_wheel = [min_row[f"raw_{key}"] for key in wheel_keys]
        cf_delta = [min_row[f"cf_delta_raw_{key}"] for key in wheel_keys]
        summary = {
            "env_id": env_id,
            "first_actor_b_on_step": first_on,
            "actor_b_toggle_count": b_toggles,
            "first_success_step": first_success,
            "first_reverse_below_0p1_step": first_reverse_01,
            "first_reverse_below_0p3_step": first_reverse_03,
            "min_base_v_along_command": min_row["base_v_along_command_after"],
            "min_base_v_step": min_row["step"],
            "at_min_actor_b": min_row["actor_b"],
            "at_min_b_raw": min_row["b_raw"],
            "at_min_phase": min_row["target_phase"],
            "at_min_first_success": min_row["first_success_after"],
            "at_min_raw_wheels": json.dumps(raw_wheel),
            "at_min_processed_wheels": json.dumps(processed),
            "at_min_actual_wheel_velocities": json.dumps(actual),
            "at_min_counterfactual_b1_minus_b0_raw_wheels": json.dumps(cf_delta),
        }
        summary_rows.append(summary)
        lines.append(
            f"ENV{env_id} first_b={first_on} b_toggles={b_toggles} first_success={first_success} "
            f"first_v<-0.1={first_reverse_01} first_v<-0.3={first_reverse_03} "
            f"min_v={min_row['base_v_along_command_after']:.6f}@{min_row['step']} "
            f"at_min(b_actor/raw/phase/first)={min_row['actor_b']:.0f}/{min_row['b_raw']}/{min_row['target_phase']}/{min_row['first_success_after']} "
            f"raw_wheel={raw_wheel} processed_wheel={processed} actual_wheel={actual} cf_delta={cf_delta}"
        )
    summary_path = output_dir / "reverse_attribution_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    report = "\n".join(lines) + "\n"
    (output_dir / "DIAGNOSIS.txt").write_text(report)
    print(report, end="", flush=True)
    print(f"TIMESERIES_CSV={csv_path}", flush=True)
    print(f"SUMMARY_CSV={summary_path}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
