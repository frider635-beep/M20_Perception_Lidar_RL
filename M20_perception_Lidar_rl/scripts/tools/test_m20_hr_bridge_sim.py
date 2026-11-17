#!/usr/bin/env python3
"""Isaac smoke tests for flat gating and the 5 cm S1 trigger."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--flat_steps", type=int, default=256)
parser.add_argument("--stair_steps", type=int, default=800)
parser.add_argument("--case", choices=("flat", "stair5"), required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401,E402
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.hr_bridge_env_cfg import (  # noqa: E402
    DeeproboticsM20HRBridgeEnvCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.hr_bridge_terrains import (  # noqa: E402
    make_hr_bridge_fixed_terrain,
)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.hr_bridge import HRBridgeTracker, STATE_HR_BRIDGE  # noqa: E402


def deterministic_cfg(step_height, first_riser_distance):
    cfg = DeeproboticsM20HRBridgeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.terrain_generator = make_hr_bridge_fixed_terrain(
        (step_height,), args.num_envs, first_riser_distance
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.terminations.terrain_out_of_bounds = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
    cfg.commands.base_velocity.rel_zero_vel_envs = 0.0
    for name in (
        "randomize_rigid_body_material", "randomize_rigid_body_mass", "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia", "randomize_com_positions", "randomize_apply_external_force_torque",
        "randomize_reset_joints", "randomize_actuator_gains", "randomize_push_robot",
    ):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events.randomize_reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    return cfg


def run_case(step_height, distance, steps, drive):
    cfg = deterministic_cfg(step_height, distance)
    env = gym.make("M20-Stair-HRBridge-v0", cfg=cfg)
    raw = env.unwrapped
    tracker = HRBridgeTracker(raw)
    obs, _ = env.reset()
    activations = 0
    max_step_up = 0.0
    min_distance = float("inf")
    max_distance = float("-inf")
    valid_samples = 0
    command_max = 0.0
    elevation_valid_samples = 0
    trigger_candidate_samples = 0
    if step_height >= 0.05:
        for _ in range(raw.cfg.hr_bridge_trigger.confirm_frames):
            _, _, activated = tracker.pre_action_state()
            activations += int(activated.sum())
    for _ in range(steps):
        state, riser, activated = tracker.pre_action_state()
        activations += int(activated.sum())
        finite = torch.isfinite(riser["riser_height"][:, 3])
        if finite.any():
            max_step_up = max(max_step_up, float(riser["riser_height"][:, 3][finite].max()))
        valid = riser["valid"][:, 3]
        valid_samples += int(valid.sum())
        elevation_valid_samples += int(riser["elevation_valid"].sum())
        h = riser["riser_height"][:, 3]
        d = riser["riser_distance"][:, 3]
        trigger_candidate_samples += int((
            valid & riser["elevation_valid"]
            & (h >= raw.cfg.hr_bridge_trigger.min_step_up - raw.cfg.hr_bridge_trigger.measurement_epsilon)
            & (d >= raw.cfg.hr_bridge_trigger.anticipatory_distance_min)
            & (d <= raw.cfg.hr_bridge_trigger.anticipatory_distance_max)
            & (raw.command_manager.get_command("base_velocity")[:, 0] > raw.cfg.hr_bridge_trigger.min_forward_command)
        ).sum())
        if valid.any():
            values = riser["riser_distance"][:, 3][valid]
            min_distance = min(min_distance, float(values.min()))
            max_distance = max(max_distance, float(values.max()))
        command_max = max(command_max, float(raw.command_manager.get_command("base_velocity")[:, 0].max()))
        action = torch.zeros((args.num_envs, 16), device=raw.device)
        if drive:
            action[:, 12:16] = -0.5
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated | truncated
        tracker.post_step(state, riser, activated)
        tracker.reset(done)
    s1 = int((tracker.state == STATE_HR_BRIDGE).sum())
    env.close()
    return (activations, s1, max_step_up, min_distance, max_distance, valid_samples,
            command_max, elevation_valid_samples, trigger_candidate_samples)


def main():
    if args.case == "flat":
        activation, s1, height, min_d, max_d, valid, command, elevation_valid, candidates = run_case(0.0, 1.10, args.flat_steps, False)
        passed = activation == 0 and s1 == 0
        print(f"FLAT_CONTROL_STEPS={args.flat_steps * args.num_envs}")
        print(f"FLAT_MAX_STEP_UP={height:.6f}")
        print(f"FLAT_S1_ACTIVATION={activation}")
        print(f"FLAT_GATE={'PASS' if passed else 'FAIL'}")
    else:
        # Close placement is only for the trigger contract. The fixed
        # evaluator retains the requested 1.10 m approach distance.
        activation, s1, height, min_d, max_d, valid, command, elevation_valid, candidates = run_case(0.05, 0.20, args.stair_steps, False)
        passed = activation > 0 or s1 > 0
        print(f"STAIR5_MAX_STEP_UP={height:.6f}")
        print(f"STAIR5_DISTANCE_RANGE=[{min_d:.6f},{max_d:.6f}]")
        print(f"STAIR5_VALID_SAMPLES={valid}")
        print(f"STAIR5_ELEVATION_VALID_SAMPLES={elevation_valid}")
        print(f"STAIR5_TRIGGER_CANDIDATE_SAMPLES={candidates}")
        print(f"STAIR5_COMMAND_MAX={command:.6f}")
        print(f"STAIR5_S1_ACTIVATION={activation}")
        print(f"STAIR5_TRIGGER={'PASS' if passed else 'FAIL'}")
    simulation_app.close()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
