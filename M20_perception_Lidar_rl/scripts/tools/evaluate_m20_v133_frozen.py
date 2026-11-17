#!/usr/bin/env python3
"""Frozen evaluation of every V13.3 checkpoint on one fixed stair setup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-manifest", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import rl_training.tasks  # noqa: F401
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_training.v133_transition_runner import V133TransitionRunner


TASK = "M20-Stair-V13.3-V11FD-FromScratch-v0"


def verify_source_manifest() -> None:
    """Refuse to evaluate checkpoints under code different from training start."""
    root = Path(__file__).resolve().parents[2]
    for line in args.source_manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative.lstrip("*")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen evaluator source drift: {path}")


def iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"Not an RSL checkpoint: {path}")
    return int(match.group(1))


def snapshot(module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def evaluate(path: Path, thresholds) -> dict[str, float | int | str | bool]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum = False
    generator.num_rows = 1
    cfg.scene.terrain.max_init_terrain_level = 0
    for terrain in generator.sub_terrains.values():
        terrain.proportion = 0.0
    ledger = cfg.v133_transition
    ascending = generator.sub_terrains["pyramid_stairs_inv"]
    descending = generator.sub_terrains["pyramid_stairs"]
    ascending.proportion = float(ledger.evaluation_ascending_fraction)
    descending.proportion = 1.0 - float(ledger.evaluation_ascending_fraction)
    fixed_height = float(ledger.evaluation_step_height)
    ascending.step_height_range = (fixed_height, fixed_height)
    descending.step_height_range = (fixed_height, fixed_height)
    cfg.curriculum.terrain_levels = None

    env = gym.make(TASK, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100.0)
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    runner = V133TransitionRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    if not checkpoint.get("v133_from_scratch") or not checkpoint.get("v132_dual_return"):
        raise RuntimeError(f"Checkpoint lacks V13.3 from-scratch dual-return identity: {path}")
    if "style_critic_state_dict" not in checkpoint:
        raise RuntimeError(f"Checkpoint lacks the V13.3 style critic: {path}")
    saved_iteration = int(checkpoint.get("iter", -1000))
    if abs(saved_iteration - iteration(path)) > 1:
        raise RuntimeError(f"Checkpoint iteration mismatch: filename={iteration(path)} payload={saved_iteration}")
    runner.alg.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    runner.alg.critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    runner.alg.style_critic.load_state_dict(checkpoint["style_critic_state_dict"], strict=True)
    for name in ("actor_state_dict", "critic_state_dict", "style_critic_state_dict"):
        if not all(torch.isfinite(value).all().item() for value in checkpoint[name].values()):
            raise RuntimeError(f"Checkpoint has non-finite {name}: {path}")
    actor_before = snapshot(runner.alg.actor)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    obs, _ = wrapped.reset()
    raw = wrapped.unwrapped
    velocity_error_sum = 0.0
    sample_count = 0
    episode_count = 0
    finite = True
    with torch.inference_mode():
        for _ in range(args.steps):
            policy_obs = obs["policy"]
            actions = policy(obs)
            obs, reward, dones, _ = wrapped.step(actions)
            command = raw.command_manager.get_command("base_velocity")[:, :2]
            actual = raw.scene["robot"].data.root_lin_vel_b[:, :2]
            finite = finite and all(torch.isfinite(value).all().item() for value in (
                policy_obs, obs["policy"], actions, reward, command, actual,
            ))
            if not finite:
                break
            velocity_error_sum += torch.linalg.vector_norm(command - actual, dim=1).sum().item()
            sample_count += args.num_envs
            episode_count += int(dones.sum())

    state = raw._v133_chain
    failures = state["repeat_count"] + state["skip_count"] + state["simultaneous_count"]
    chain_count = int(state["chain_count"])
    fl_count, fr_count = int(state["chain_fl_count"]), int(state["chain_fr_count"])
    chain_precision = chain_count / max(chain_count + int(failures), 1)
    velocity_tracking = float(np.exp(-velocity_error_sum / max(sample_count, 1)))
    qualified = mdp.v133_checkpoint_qualified(
        chain_count=chain_count, fl_count=fl_count, fr_count=fr_count,
        chain_precision=chain_precision, velocity_tracking=velocity_tracking, finite=finite,
        minimum_chain_count=thresholds.evaluation_min_chain_count,
        minimum_each_final_side=thresholds.evaluation_min_each_final_side,
        minimum_chain_precision=thresholds.evaluation_min_chain_precision,
        minimum_velocity_tracking=thresholds.evaluation_min_velocity_tracking,
    )
    result = {
        "checkpoint": str(path.resolve()),
        "iteration": iteration(path),
        "chain_count": chain_count,
        "adjacent_count": int(state["adjacent_count"]),
        "repeat_count": int(state["repeat_count"]),
        "follower_count": int(state["follower_count"]),
        "skip_count": int(state["skip_count"]),
        "simultaneous_count": int(state["simultaneous_count"]),
        "chain_fl_count": fl_count,
        "chain_fr_count": fr_count,
        "chain_precision": chain_precision,
        "chain_fl_fraction": fl_count / max(chain_count, 1),
        "velocity_tracking": velocity_tracking,
        "episodes": episode_count,
        "finite": finite,
        "qualified": qualified,
        "actor_bitwise_identical": all(
            torch.equal(value.detach().cpu(), actor_before[key])
            for key, value in runner.alg.actor.state_dict().items()
        ),
    }
    if not result["actor_bitwise_identical"]:
        raise RuntimeError(f"Frozen evaluation mutated actor parameters: {path}")
    wrapped.close()
    return result


def main() -> None:
    verify_source_manifest()
    thresholds = parse_env_cfg(TASK, device=args.device, num_envs=1).v133_transition
    paths = sorted(args.checkpoint_dir.glob("model_*.pt"), key=iteration)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        row = evaluate(path, thresholds)
        rows.append(row)
        print("V133_FROZEN_RESULT=" + json.dumps(row, sort_keys=True), flush=True)
    qualified_rows = [row for row in rows if row["qualified"]]
    selected = max(
        qualified_rows,
        key=lambda row: (row["chain_precision"], row["chain_count"], row["velocity_tracking"]),
    ) if qualified_rows else None
    payload = {
        "results": rows,
        "best_by_chain_precision": max(rows, key=lambda row: (row["chain_precision"], row["chain_count"])),
        "best_by_chain_count": max(rows, key=lambda row: (row["chain_count"], row["chain_precision"])),
        "best_by_velocity": max(rows, key=lambda row: row["velocity_tracking"]),
        "selected_checkpoint": selected,
    }
    (args.output / "frozen_sweep.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("V133_FROZEN_SUMMARY=" + json.dumps({key: value for key, value in payload.items() if key != "results"}, sort_keys=True), flush=True)
    if selected is None:
        # AppLauncher.close() terminates this process with status 0 in the
        # standalone Isaac runtime.  Reject before closing so the training
        # launcher cannot silently accept an all-red checkpoint sweep.
        raise RuntimeError(
            "No V13.3 checkpoint passed chain, mirror, velocity and finite-value gates."
        )
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Isaac's standalone exception hook can otherwise turn an uncaught
        # Python failure into process status 0.  The launcher relies on this
        # status, so every ordinary evaluation failure must remain red.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
