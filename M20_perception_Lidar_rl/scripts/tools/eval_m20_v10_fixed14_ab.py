#!/usr/bin/env python3
"""Side-car IsaacLab A/B evaluation for V10 PGTT on fixed 14 cm stairs."""

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

    active = torch.ones(num_envs, device=raw.device, dtype=torch.bool)
    fell = torch.zeros_like(active)
    max_base_x = torch.full((num_envs,), -1.0e9, device=raw.device)
    max_hr_x = torch.full_like(max_base_x, -1.0e9)
    max_fr_active_lift = torch.zeros_like(max_base_x)
    max_clearance = torch.zeros_like(max_base_x)
    max_fr_action = torch.zeros((num_envs, 3), device=raw.device)
    b_seen = torch.zeros_like(active)
    first_seen = torch.zeros_like(active)
    miss_seen = torch.zeros_like(active)
    stable_at_completion = torch.zeros_like(active)
    full_seen = torch.zeros_like(active)
    end_step = torch.full((num_envs,), -1, device=raw.device, dtype=torch.long)
    observation_frames = {}
    normal_timeseries = []
    best_d_error = float("inf")
    previous_normal_frame = None
    first_b_seen = False

    for step in range(args.steps):
        state = mdp.v10_state(raw)
        root_x = robot.data.root_pos_w[:, 0] - raw.scene.env_origins[:, 0]
        hr_x = robot.data.body_pos_w[:, hr_id, 0] - raw.scene.env_origins[:, 0]
        max_base_x = torch.where(active, torch.maximum(max_base_x, root_x), max_base_x)
        max_hr_x = torch.where(active, torch.maximum(max_hr_x, hr_x), max_hr_x)
        max_fr_active_lift = torch.where(
            active, torch.maximum(max_fr_active_lift, state["max_active_fr_lift"]), max_fr_active_lift
        )
        max_clearance = torch.where(
            active, torch.maximum(max_clearance, state["max_clearance_progress"]), max_clearance
        )
        b_seen |= active & state["b_raw"]
        first_seen |= active & state["first_step_success_latched"]
        miss_seen |= active & state["perception_miss_latched"]
        full_seen |= active & state["full_stair_success_latched"]

        actor_obs = policy_obs.clone()
        if not args.train_faithful:
            actor_obs[n_each:, 57:60] = 0.0
        with torch.inference_mode():
            action = actor(actor_obs)
        normal_timeseries.append({
            "step": step,
            "obs": actor_obs[0].detach().cpu().tolist(),
            "action": action[0].detach().cpu().tolist(),
        })
        if step == 0:
            observation_frames["INITIAL_COMMAND_FRAME"] = {
                "step": step,
                "obs": actor_obs[0].detach().cpu().tolist(),
                "action": action[0].detach().cpu().tolist(),
            }
        actor_b = float(actor_obs[0, 57].item())
        actor_d = float(actor_obs[0, 59].item())
        current_normal_frame = {
            "step": step,
            "obs": actor_obs[0].detach().cpu().tolist(),
            "action": action[0].detach().cpu().tolist(),
        }
        if actor_b > 0.5 and not first_b_seen:
            if previous_normal_frame is not None:
                observation_frames["PRE_B_ON_FRAME"] = previous_normal_frame
            observation_frames["FIRST_B_ON_FRAME"] = current_normal_frame
            first_b_seen = True
        previous_normal_frame = current_normal_frame
        d_error = abs(actor_d - 0.40)
        if actor_b > 0.5 and d_error < best_d_error:
            best_d_error = d_error
            observation_frames["B_D040_FRAME"] = {
                "step": step,
                "obs": policy_obs[0].detach().cpu().tolist(),
                "action": action[0].detach().cpu().tolist(),
            }
        max_fr_action = torch.where(
            active[:, None], torch.maximum(max_fr_action, action[:, 3:6].abs()), max_fr_action
        )
        next_obs, _, terminated, truncated, _ = env.step(action)
        policy_obs = next_obs["policy"]

        gravity = robot.data.projected_gravity_b
        stable = (gravity[:, 2] < -0.65) & (gravity[:, :2].abs().amax(dim=1) < 0.55)
        completed = full_seen
        stable_at_completion |= active & completed & stable
        ended = active & (terminated.bool() | truncated.bool())
        fell |= active & terminated.bool()
        end_step = torch.where(ended & (end_step < 0), torch.full_like(end_step, step), end_step)
        active &= ~ended
        if not active.any():
            break

    success = full_seen & stable_at_completion & ~fell
    rows = []
    for env_id in range(num_envs):
        condition = "PGTT_NORMAL" if env_id < n_each else "PGTT_MASKED"
        rows.append({
            "condition": condition,
            "trial": env_id if env_id < n_each else env_id - n_each,
            "success": int(success[env_id].item()),
            "fell": int(fell[env_id].item()),
            "max_base_x": float(max_base_x[env_id].item()),
            "max_hr_x": float(max_hr_x[env_id].item()),
            "b_seen_raw": int(b_seen[env_id].item()),
            "actor_pgtt_masked": int(env_id >= n_each),
            "first_success": int(first_seen[env_id].item()),
            "full_success_latched": int(full_seen[env_id].item()),
            "perception_miss": int(miss_seen[env_id].item()),
            "max_clearance_progress": float(max_clearance[env_id].item()),
            "max_active_fr_lift_m": float(max_fr_active_lift[env_id].item()),
            "max_fr_hipx_action": float(max_fr_action[env_id, 0].item()),
            "max_fr_hipy_action": float(max_fr_action[env_id, 1].item()),
            "max_fr_knee_action": float(max_fr_action[env_id, 2].item()),
            "end_step": int(end_step[env_id].item()),
        })

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    observation_path = output_dir / "isaac_observation_frames.csv"
    observation_fields = ["frame", "step"] + [f"obs_{i}" for i in range(60)] + [
        f"action_{i}" for i in range(16)
    ]
    with observation_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=observation_fields)
        writer.writeheader()
        for frame_name, frame in observation_frames.items():
            row = {"frame": frame_name, "step": frame["step"]}
            row.update({f"obs_{i}": value for i, value in enumerate(frame["obs"])})
            row.update({f"action_{i}": value for i, value in enumerate(frame["action"])})
            writer.writerow(row)

    timeseries_path = output_dir / "isaac_normal_timeseries.csv"
    timeseries_fields = ["step"] + [f"obs_{i}" for i in range(60)] + [f"action_{i}" for i in range(16)]
    with timeseries_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=timeseries_fields)
        writer.writeheader()
        for sample in normal_timeseries:
            row = {"step": sample["step"]}
            row.update({f"obs_{i}": value for i, value in enumerate(sample["obs"])})
            row.update({f"action_{i}": value for i, value in enumerate(sample["action"])})
            writer.writerow(row)

    lines = [
        "ISAAC_V10_FIXED14_AB=PASS",
        f"CHECKPOINT_ITERATION={checkpoint_iteration}",
        "ACTOR_OBS=60",
        "ACTION=16",
        f"NUM_ENVS_PER_CONDITION={n_each}",
        f"TERRAIN=official MeshInvertedPyramidStairsTerrainCfg fixed at {HEIGHT:.3f}m",
        "SUCCESS_DEFINITION=current V10 FULL_STAIR_SUCCESS latch AND stable AND no fall",
    ]
    conditions = ("PGTT_NORMAL",) if args.train_faithful else ("PGTT_NORMAL", "PGTT_MASKED")
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        count = len(selected)
        lines.extend((
            f"{condition}_SUCCESS_RATE={sum(row['success'] for row in selected) / count:.6f}",
            f"{condition}_FIRST_RATE={sum(row['first_success'] for row in selected) / count:.6f}",
            f"{condition}_MISS_RATE={sum(row['perception_miss'] for row in selected) / count:.6f}",
            f"{condition}_MEAN_MAX_BASE_X={sum(row['max_base_x'] for row in selected) / count:.6f}",
            f"{condition}_MEAN_MAX_HR_X={sum(row['max_hr_x'] for row in selected) / count:.6f}",
            f"{condition}_MEAN_MAX_CLEARANCE={sum(row['max_clearance_progress'] for row in selected) / count:.6f}",
            f"{condition}_MEAN_MAX_ACTIVE_FR_LIFT_M={sum(row['max_active_fr_lift_m'] for row in selected) / count:.6f}",
        ))
    report = "\n".join(lines) + "\n"
    (output_dir / "DIAGNOSIS.txt").write_text(report)
    print(report, end="", flush=True)
    print(f"SUMMARY_CSV={csv_path}", flush=True)
    print(f"ISAAC_OBSERVATION_CSV={observation_path}", flush=True)
    print(f"ISAAC_NORMAL_TIMESERIES={timeseries_path}", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
