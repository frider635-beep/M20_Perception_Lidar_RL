#!/usr/bin/env python3
"""Frozen V13 checkpoint sweep under the fixed V13.1 strict evaluator.

This program never calls ``learn`` or ``optimizer.step``.  It also verifies
that every loaded actor remains bitwise identical after its rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--iterations", type=str, default="2000,2500,3000,3500,4000,4500,5000")
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=720)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--probe-cap", type=int, default=120000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_training.v131_ordered_runner import V131OrderedRetentionRunner


TASK = "M20-Stair-V13.1-V11FD-OrderedAlternation-Retention-v0"
ORDERED_TERMS = {
    "v131_b1_progress", "v131_b1_velocity_coupled_progress", "v131_expected_progress",
    "v131_velocity_coupled_expected_progress", "v131_alternating_completion",
    "v131_nonordered_completion", "v131_simultaneous_b1_lift", "v131_skip_riser",
    "v131_same_side_progress", "v131_simultaneous_progress", "v131_top_follow_progress",
    "v131_top_follow_completion", "v131_post_top_front_lift",
}
VELOCITY_TERMS = {"track_lin_vel_xy_exp"}
FORWARD_TERMS = {"world_forward_progress"}


def checkpoint_iteration(path: Path) -> int | None:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def snapshot(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def same_snapshot(module, before: dict[str, torch.Tensor]) -> bool:
    now = module.state_dict()
    return now.keys() == before.keys() and all(
        torch.equal(now[key].detach().cpu(), before[key]) for key in now
    )


def classify(state: dict[str, torch.Tensor], done: torch.Tensor) -> list[str]:
    labels = np.full(done.numel(), "FAILED", dtype=object)
    strict = state["strict_task_complete"].detach().cpu().numpy()
    skip = state["episode_skip_latched"].detach().cpu().numpy()
    same = state["episode_same_side_latched"].detach().cpu().numpy()
    simult = state["episode_simultaneous_latched"].detach().cpu().numpy()
    labels[strict] = "STRICT_ORDERED"
    labels[simult & ~strict] = "SIMULTANEOUS"
    labels[same & ~simult & ~strict] = "SAME_SIDE"
    labels[skip & ~same & ~simult & ~strict] = "SKIP"
    return labels[done.detach().cpu().numpy()].tolist()


def evaluate(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    # Re-create the environment for every policy: same task, terrain and RNG seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(TASK, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    train_cfg = agent_cfg.to_dict()
    os.environ["V131_RETENTION_ACTOR_CHECKPOINT"] = str(path.resolve())
    runner = V131OrderedRetentionRunner(wrapped, train_cfg, log_dir=None, device=agent_cfg.device)
    actor_before = snapshot(runner.alg.actor)
    critic_before = snapshot(runner.alg.critic)
    optimizer_before = json.dumps(runner.alg.optimizer.state_dict(), default=str, sort_keys=True)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    obs, _ = wrapped.reset()
    raw = wrapped.unwrapped
    term_names = list(raw.reward_manager._term_names)
    dt = float(raw.step_dt)
    episode_terms = torch.zeros((args.num_envs, len(term_names)), device=raw.device)
    native_velocity_episode = torch.zeros(args.num_envs, device=raw.device)
    native_forward_episode = torch.zeros(args.num_envs, device=raw.device)
    category_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    probe_x: list[np.ndarray] = []
    probe_y: list[np.ndarray] = []
    probe_group: list[np.ndarray] = []
    probe_count = 0
    episode_id = torch.zeros(args.num_envs, device=raw.device, dtype=torch.long)
    eval_skip = torch.zeros(args.num_envs, device=raw.device, dtype=torch.bool)
    eval_same = torch.zeros_like(eval_skip)
    eval_simult = torch.zeros_like(eval_skip)
    eval_strict = torch.zeros_like(eval_skip)
    eval_completed = torch.zeros(args.num_envs, device=raw.device, dtype=torch.long)
    completed_risers: list[int] = []
    previous_dones = torch.zeros(args.num_envs, device=raw.device, dtype=torch.bool)
    velocity_cfg = raw.reward_manager.get_term_cfg("track_lin_vel_xy_exp")
    forward_cfg = raw.reward_manager.get_term_cfg("world_forward_progress")
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards import track_lin_vel_xy_exp
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.v11_continuous import v11_world_forward_progress_reward
    velocity_error_sum = 0.0
    velocity_samples = 0

    for _ in range(args.steps):
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs["policy"]
        state = raw._v131_ordered
        # X and y are from the same policy-decision instant.  Transition frames
        # are excluded because their ledger target changes during settlement.
        actionable = (
            state["alternating_progress_active"] & (state["teaching_expected_side"] >= 0)
            & ~state["ordered_event"] & ~state["teaching_resync_event"]
            & ~previous_dones
        )
        if actionable.any() and probe_count < args.probe_cap:
            ids = actionable.nonzero(as_tuple=False).flatten()[: args.probe_cap - probe_count]
            probe_x.append(policy_obs[ids].detach().cpu().numpy())
            probe_y.append(state["teaching_expected_side"][ids].detach().cpu().numpy())
            group = (int(checkpoint_iteration(path) or 0) * 10**12
                     + ids * 10**6 + episode_id[ids])
            probe_group.append(group.detach().cpu().numpy())
            probe_count += ids.numel()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = wrapped.step(actions)
        state = raw._v131_ordered
        episode_terms += raw.reward_manager._step_reward * dt
        native_velocity_episode += track_lin_vel_xy_exp(
            raw, std=velocity_cfg.params["std"], command_name=velocity_cfg.params["command_name"],
        ) * float(velocity_cfg.weight) * dt
        forward_kwargs = {
            "target_speed_fraction": forward_cfg.params["target_speed_fraction"],
        }
        if forward_cfg.params.get("asset_cfg") is not None:
            forward_kwargs["asset_cfg"] = forward_cfg.params["asset_cfg"]
        native_forward_episode += v11_world_forward_progress_reward(
            raw, **forward_kwargs,
        ) * float(forward_cfg.weight) * dt

        eval_skip |= state["skip_event"] | state["episode_skip_latched"]
        eval_same |= state["wrong_side_event"] | state["episode_same_side_latched"]
        eval_simult |= state["simultaneous_b1_lift_event"] | state["episode_simultaneous_latched"]
        eval_strict |= state["strict_task_complete"]
        eval_completed = torch.maximum(eval_completed, state["ordered_last_riser_index"])

        command = raw.command_manager.get_command("base_velocity")[:, :2]
        actual = raw.scene["robot"].data.root_lin_vel_b[:, :2]
        velocity_error_sum += torch.linalg.vector_norm(command - actual, dim=1).sum().item()
        velocity_samples += args.num_envs

        if dones.any():
            ids = dones.nonzero(as_tuple=False).flatten()
            labels_all = np.full(args.num_envs, "FAILED", dtype=object)
            strict_np, skip_np = eval_strict.cpu().numpy(), eval_skip.cpu().numpy()
            same_np, simult_np = eval_same.cpu().numpy(), eval_simult.cpu().numpy()
            labels_all[strict_np] = "STRICT_ORDERED"
            labels_all[simult_np & ~skip_np & ~same_np & ~strict_np] = "SIMULTANEOUS"
            labels_all[same_np & ~skip_np & ~strict_np] = "SAME_SIDE"
            labels_all[skip_np & ~strict_np] = "SKIP"
            labels = labels_all[dones.cpu().numpy()].tolist()
            values = episode_terms[ids].detach().cpu()
            completed_risers.extend(eval_completed[ids].detach().cpu().tolist())
            native_vel_values = native_velocity_episode[ids].detach().cpu()
            native_fwd_values = native_forward_episode[ids].detach().cpu()
            for row, native_vel, native_fwd, label in zip(values, native_vel_values, native_fwd_values, labels):
                by_name = dict(zip(term_names, row.tolist()))
                ordered = sum(by_name.get(name, 0.0) for name in ORDERED_TERMS)
                velocity = sum(by_name.get(name, 0.0) for name in VELOCITY_TERMS)
                forward = sum(by_name.get(name, 0.0) for name in FORWARD_TERMS)
                total = sum(by_name.values())
                category_rows[label].append({
                    "native_v11_return": total - ordered - velocity - forward + float(native_vel) + float(native_fwd),
                    "native_velocity_return": float(native_vel),
                    "native_forward_progress_return": float(native_fwd),
                    "ordered_progress_return": by_name.get("v131_b1_progress", 0.0)
                        + by_name.get("v131_expected_progress", 0.0),
                    "ordered_velocity_coupled_return": by_name.get("v131_b1_velocity_coupled_progress", 0.0)
                        + by_name.get("v131_velocity_coupled_expected_progress", 0.0),
                    "same_side_cost": by_name.get("v131_same_side_progress", 0.0),
                    "skip_cost": by_name.get("v131_skip_riser", 0.0),
                    "total_return": total,
                })
            episode_terms[ids] = 0.0
            native_velocity_episode[ids] = 0.0; native_forward_episode[ids] = 0.0
            eval_skip[ids] = False; eval_same[ids] = False; eval_simult[ids] = False
            eval_strict[ids] = False; eval_completed[ids] = 0; episode_id[ids] += 1
        previous_dones = dones.clone()

    state = raw._v131_ordered
    physical = state["physical_completion_count"].clamp_min(1).float()
    opportunities = (state["physical_completion_count"] - state["b1_completion_count"]).clamp_min(1).float()
    encounter = state["encounter_count"].clamp_min(1).float()
    metrics = {
        "checkpoint": str(path.resolve()),
        "iteration": checkpoint_iteration(path),
        "strict_adjacent_completion_rate": float((state["adjacent_completion_count"].float() / physical).item()),
        "strict_ordered_alternation_rate": float((state["ordered_alternating_count"].float() / opportunities).item()),
        "skip_rate": float((state["skip_count"].float() / encounter).item()),
        "same_side_rate": float((state["wrong_side_count"].float() / physical).item()),
        "simultaneous_rate": float((state["simultaneous_b1_lift_count"].float() / encounter).item()),
        "platform_success_rate": float((state["top_follow_success_count"].float() / state["top_follow_entry_count"].clamp_min(1).float()).item()),
        "completed_risers_mean": float(np.mean(completed_risers)) if completed_risers else None,
        "velocity_tracking": float(np.exp(-velocity_error_sum / max(velocity_samples, 1))),
        "episodes": int(sum(len(rows) for rows in category_rows.values())),
        "probe_frames": probe_count,
        "actor_bitwise_identical": same_snapshot(runner.alg.actor, actor_before),
        "critic_bitwise_identical": same_snapshot(runner.alg.critic, critic_before),
        "optimizer_bitwise_identical": optimizer_before == json.dumps(runner.alg.optimizer.state_dict(), default=str, sort_keys=True),
        "episode_return_ledger": {},
    }
    for label in ("STRICT_ORDERED", "SKIP", "SAME_SIDE", "SIMULTANEOUS", "FAILED"):
        rows = category_rows[label]
        metrics["episode_return_ledger"][label] = {"count": len(rows)}
        for key in ("native_v11_return", "native_velocity_return", "native_forward_progress_return",
                    "ordered_progress_return", "ordered_velocity_coupled_return", "same_side_cost",
                    "skip_cost", "total_return"):
            metrics["episode_return_ledger"][label][key] = (
                float(np.mean([row[key] for row in rows])) if rows else None
            )
    if not (metrics["actor_bitwise_identical"] and metrics["critic_bitwise_identical"]
            and metrics["optimizer_bitwise_identical"]):
        raise RuntimeError(f"Frozen evaluation mutated parameters for {path}")
    x = np.concatenate(probe_x) if probe_x else np.empty((0, 63), np.float32)
    y = np.concatenate(probe_y) if probe_y else np.empty((0,), np.int64)
    groups = np.concatenate(probe_group) if probe_group else np.empty((0,), np.int64)
    wrapped.close()
    return metrics, x, y, groups


def main() -> None:
    wanted = {int(value) for value in args.iterations.split(",") if value.strip()}
    paths = sorted(
        (path for path in args.checkpoint_dir.glob("model_*.pt") if checkpoint_iteration(path) in wanted),
        key=lambda path: checkpoint_iteration(path) or -1,
    )
    if not paths:
        raise FileNotFoundError("No requested checkpoint exists; files are scanned, never fabricated.")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        result, x, y, groups = evaluate(path)
        np.savez_compressed(args.output / f"probe_{result['iteration']}.npz", X=x, y=y, groups=groups)
        results.append(result)
        print("V131_FROZEN_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    (args.output / "frozen_sweep.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    with (args.output / "frozen_sweep.csv").open("w", newline="") as stream:
        keys = [key for key in results[0] if key != "episode_return_ledger"]
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in results)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
