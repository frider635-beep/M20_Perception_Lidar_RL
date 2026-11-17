#!/usr/bin/env python3
"""Measure per-encounter HR unload-to-lift delay without changing training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--unload_fz", type=float, default=50.0)
parser.add_argument("--confirm_steps", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rl_training.tasks.manager_based.locomotion.velocity import mdp  # noqa: E402
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    DeeproboticsM20PerceptiveV9Stage1PPORunnerCfg,
)
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.v9_stage1_env_cfg import (  # noqa: E402
    DeeproboticsM20PerceptiveV9Stage1EnvCfg,
)
from rl_training.v9_stage1_runner import V9Stage1Runner  # noqa: E402


WHEELS = ("FL", "FR", "HL", "HR")
LEG_NAMES = ("hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint")
DISTANCE_BINS = (
    ("30_20cm", 0.20, 0.30), ("20_10cm", 0.10, 0.20),
    ("10_05cm", 0.05, 0.10), ("05_00cm", 0.00, 0.05),
)


class Means:
    def __init__(self):
        self.count = 0
        self.sums: dict[str, float] = {}

    def add(self, mask: torch.Tensor, values: dict[str, torch.Tensor]):
        count = int(mask.sum().item())
        if count == 0:
            return
        self.count += count
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value[mask].sum().item())

    def result(self):
        return {"sample_count": self.count, **{
            key: value / max(self.count, 1) for key, value in self.sums.items()
        }}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, q).item())


def summarize_events(events: list[dict], step_dt: float):
    completed = [row for row in events if row["outcome"] == "LIFT_AFTER_UNLOAD"]
    delays = [float(row["delay_steps"]) for row in completed]
    unload_d = [float(row["unload_distance_m"]) for row in completed]
    lift_d = [float(row["lift_onset_distance_m"]) for row in completed]
    travelled = [float(row["unload_to_lift_travel_m"]) for row in completed]
    return {
        "encounters_finalized": len(events),
        "unload_confirmed_count": sum(row["unload_step"] is not None for row in events),
        "lift_after_unload_count": len(completed),
        "unloaded_but_never_lifted_count": sum(row["outcome"] == "UNLOADED_NO_LIFT" for row in events),
        "never_unloaded_count": sum(row["outcome"] == "NO_UNLOAD" for row in events),
        "lift_after_unload_fraction": len(completed) / max(len(events), 1),
        "delay_steps_mean": sum(delays) / max(len(delays), 1),
        "delay_steps_median": percentile(delays, 0.5),
        "delay_steps_p90": percentile(delays, 0.9),
        "delay_seconds_mean": step_dt * sum(delays) / max(len(delays), 1),
        "delay_seconds_median": None if not delays else step_dt * percentile(delays, 0.5),
        "unload_distance_cm_mean": 100.0 * sum(unload_d) / max(len(unload_d), 1),
        "lift_onset_distance_cm_mean": 100.0 * sum(lift_d) / max(len(lift_d), 1),
        "unload_to_lift_travel_cm_mean": 100.0 * sum(travelled) / max(len(travelled), 1),
        "lift_onset_actual_lift_cm_mean": (
            100.0 * sum(float(row["lift_onset_actual_lift_m"]) for row in completed) / max(len(completed), 1)
        ),
        "lift_onset_joint_delta_mean": (
            sum(float(row["lift_onset_joint_delta"]) for row in completed) / max(len(completed), 1)
        ),
    }


def main():
    cfg = DeeproboticsM20PerceptiveV9Stage1EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = gym.make("M20-Perceptive-V9-Stage1-v0", cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    runner = V9Stage1Runner(
        wrapped, DeeproboticsM20PerceptiveV9Stage1PPORunnerCfg().to_dict(),
        log_dir=None, device=cfg.sim.device,
    )
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    sensor = raw.scene.sensors["contact_forces"]
    step_dt = float(raw.step_dt)

    contact_ids = []
    for wheel in mdp.WHEEL_NAMES:
        ids, names = sensor.find_bodies(wheel)
        if len(ids) != 1 or names != [wheel]:
            raise RuntimeError(f"contact order mismatch for {wheel}: {names}")
        contact_ids.append(ids[0])
    contact_ids = torch.tensor(contact_ids, device=raw.device, dtype=torch.long)
    hr_joint_ids, resolved = robot.find_joints(list(LEG_NAMES), preserve_order=True)
    if resolved != list(LEG_NAMES):
        raise RuntimeError(f"HR joint order mismatch: {resolved}")

    n = args.num_envs
    in_encounter = torch.zeros(n, dtype=torch.bool, device=raw.device)
    encounter_id = torch.zeros(n, dtype=torch.long, device=raw.device)
    unloaded = torch.zeros(n, dtype=torch.bool, device=raw.device)
    lifted = torch.zeros(n, dtype=torch.bool, device=raw.device)
    low_streak = torch.zeros(n, dtype=torch.long, device=raw.device)
    unload_step = torch.full((n,), -1, dtype=torch.long, device=raw.device)
    unload_distance = torch.zeros(n, device=raw.device)
    unload_fz = torch.zeros(n, device=raw.device)
    unload_contact = torch.zeros(n, device=raw.device)
    unload_candidate_step = torch.full((n,), -1, dtype=torch.long, device=raw.device)
    unload_candidate_distance = torch.zeros(n, device=raw.device)
    unload_candidate_fz = torch.zeros(n, device=raw.device)
    unload_candidate_contact = torch.zeros(n, device=raw.device)
    state_means = {name: Means() for name in ("SUPPORT", "UNLOADED_WAIT", "SWING")}
    bin_means = {name: Means() for name, _, _ in DISTANCE_BINS}
    events: list[dict] = []

    def finalize(env_ids: torch.Tensor, outcome_override: str | None = None):
        for env_id_t in env_ids:
            env_id = int(env_id_t.item())
            if not bool(in_encounter[env_id]):
                continue
            # A completed lift event is latched at onset with all onset fields;
            # do not emit a second, incomplete row when its encounter ends.
            if bool(lifted[env_id]):
                continue
            if outcome_override is not None:
                outcome = outcome_override
            elif bool(unloaded[env_id]):
                outcome = "UNLOADED_NO_LIFT"
            else:
                outcome = "NO_UNLOAD"
            events.append({
                "env_id": env_id,
                "encounter_id": int(encounter_id[env_id].item()),
                "outcome": outcome,
                "unload_step": None if unload_step[env_id] < 0 else int(unload_step[env_id].item()),
                "unload_distance_m": None if unload_step[env_id] < 0 else float(unload_distance[env_id].item()),
                "unload_fz_N": None if unload_step[env_id] < 0 else float(unload_fz[env_id].item()),
                "unload_contact": None if unload_step[env_id] < 0 else bool(unload_contact[env_id].item()),
                "lift_onset_step": None,
                "delay_steps": None,
                "delay_seconds": None,
                "lift_onset_distance_m": None,
                "unload_to_lift_travel_m": None,
                "lift_onset_actual_lift_m": None,
                "lift_onset_joint_delta": None,
                "lift_onset_hr_action_delta": None,
                "lift_onset_base_vx": None,
            })
        in_encounter[env_ids] = False
        unloaded[env_ids] = False
        lifted[env_ids] = False
        low_streak[env_ids] = 0
        unload_step[env_ids] = -1
        unload_candidate_step[env_ids] = -1

    obs, _ = env.reset()
    for global_step in range(args.steps):
        with torch.inference_mode():
            policy_obs = runner.alg.policy.get_actor_obs(obs)
            actions = policy(obs)
            base_actions = runner._v9_base_actor(policy_obs)
            obs, _, dones, _ = wrapped.step(actions)
            state = mdp.perceptive_spatial_state(raw)

        forces = sensor.data.net_forces_w[:, contact_ids]
        force_norm = forces.norm(dim=-1)
        force_z = forces[..., 2].clamp_min(0.0)
        contacts = force_norm >= 1.0
        vertical = mdp.classify_vertical_riser_contact(
            state["wheel_yaw"][..., :2], state["z_actual_world"], forces,
            state["riser_x"], state["z_lower"], state["z_upper"],
            wheel_radius=raw.cfg.spatial_guidance.wheel_collision_radius,
        ) & state["valid"]
        distance = state["riser_distance"][:, 3]
        active = state["valid"][:, 3] & state["active"][:, 3] & (distance > 0.0) & ~vertical[:, 3]

        ended = in_encounter & (~active | dones)
        finalize(ended.nonzero(as_tuple=False).flatten())
        started = active & ~in_encounter
        in_encounter[started] = True
        encounter_id[started] += 1

        low_load_now = active & ((force_z[:, 3] <= args.unload_fz) | ~contacts[:, 3])
        new_candidate = in_encounter & ~unloaded & low_load_now & (low_streak == 0)
        unload_candidate_step[new_candidate] = global_step
        unload_candidate_distance[new_candidate] = distance[new_candidate]
        unload_candidate_fz[new_candidate] = force_z[new_candidate, 3]
        unload_candidate_contact[new_candidate] = contacts[new_candidate, 3].float()
        low_streak = torch.where(in_encounter & ~unloaded & low_load_now, low_streak + 1, torch.zeros_like(low_streak))
        confirmed = in_encounter & ~unloaded & (low_streak >= args.confirm_steps)
        unloaded[confirmed] = True
        unload_step[confirmed] = unload_candidate_step[confirmed]
        unload_distance[confirmed] = unload_candidate_distance[confirmed]
        unload_fz[confirmed] = unload_candidate_fz[confirmed]
        unload_contact[confirmed] = unload_candidate_contact[confirmed]

        actual_lift = state["actual_lift"][:, 3]
        lift_started = in_encounter & unloaded & ~lifted & active & (actual_lift > 0.005)
        gravity = robot.data.projected_gravity_b
        roll = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
        pitch = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
        hr_joint_delta = (
            robot.data.joint_pos[:, hr_joint_ids] - robot.data.default_joint_pos[:, hr_joint_ids]
        ).norm(dim=1)
        hr_action_delta = (actions[:, 9:12] - base_actions[:, 9:12]).norm(dim=1)
        for env_id_t in lift_started.nonzero(as_tuple=False).flatten():
            env_id = int(env_id_t.item())
            delay_steps = global_step - int(unload_step[env_id].item())
            events.append({
                "env_id": env_id,
                "encounter_id": int(encounter_id[env_id].item()),
                "outcome": "LIFT_AFTER_UNLOAD",
                "unload_step": int(unload_step[env_id].item()),
                "unload_distance_m": float(unload_distance[env_id].item()),
                "unload_fz_N": float(unload_fz[env_id].item()),
                "unload_contact": bool(unload_contact[env_id].item()),
                "lift_onset_step": global_step,
                "delay_steps": delay_steps,
                "delay_seconds": delay_steps * step_dt,
                "lift_onset_distance_m": float(distance[env_id].item()),
                "unload_to_lift_travel_m": float((unload_distance[env_id] - distance[env_id]).item()),
                "lift_onset_actual_lift_m": float(actual_lift[env_id].item()),
                "lift_onset_joint_delta": float(hr_joint_delta[env_id].item()),
                "lift_onset_hr_action_delta": float(hr_action_delta[env_id].item()),
                "lift_onset_base_vx": float(robot.data.root_lin_vel_b[env_id, 0].item()),
            })
        lifted[lift_started] = True

        values = {
            "HR_contact": contacts[:, 3].float(),
            "HR_force_z_up_N": force_z[:, 3],
            "HR_force_norm_N": force_norm[:, 3],
            "HR_actual_lift_m": actual_lift,
            "HR_lift_ref_m": state["lift_ref"][:, 3],
            "HR_leg_action_delta": hr_action_delta,
            "HR_joint_position_delta": hr_joint_delta,
            "base_vx_mps": robot.data.root_lin_vel_b[:, 0],
            "base_roll_rad": roll,
            "base_pitch_rad": pitch,
            "vertical_contact": vertical[:, 3].float(),
        }
        for index, wheel in enumerate(WHEELS[:3]):
            values[f"{wheel}_force_z_up_N"] = force_z[:, index]
        support = in_encounter & ~unloaded
        wait = in_encounter & unloaded & ~lifted
        swing = in_encounter & lifted
        state_means["SUPPORT"].add(support, values)
        state_means["UNLOADED_WAIT"].add(wait, values)
        state_means["SWING"].add(swing, values)
        for label, lower, upper in DISTANCE_BINS:
            mask = active & (distance > lower) & (distance <= upper)
            values_with_state = {
                **values,
                "unloaded_fraction": unloaded.float(),
                "unloaded_but_not_lifting_fraction": (unloaded & ~lifted).float(),
                "swing_fraction": lifted.float(),
            }
            bin_means[label].add(mask, values_with_state)

    finalize(in_encounter.nonzero(as_tuple=False).flatten(), outcome_override=None)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_iter": int(checkpoint["iter"]),
        "num_envs": args.num_envs,
        "rollout_steps": args.steps,
        "control_dt_seconds": step_dt,
        "definitions": {
            "SUPPORT": "HR precontact encounter before confirmed unload",
            "HR_UNLOADED": (
                f"(world-up Fz <= {args.unload_fz}N or contact=False) for "
                f"{args.confirm_steps} consecutive control steps"
            ),
            "UNLOADED_WAIT": "confirmed unloaded but actual_lift<=0.005m",
            "SWING": "actual_lift_base_frame>0.005m before riser/vertical contact",
            "unload_onset": "first step of the subsequently confirmed low-load streak",
        },
        "events": summarize_events(events, step_dt),
        "states": {key: value.result() for key, value in state_means.items()},
        "distance_bins": {key: value.result() for key, value in bin_means.items()},
    }
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "summary.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    if events:
        with open(os.path.join(args.output, "encounters.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(events[0]))
            writer.writeheader()
            writer.writerows(events)
    print(json.dumps(result, indent=2))
    env.close()


try:
    main()
finally:
    simulation_app.close()
