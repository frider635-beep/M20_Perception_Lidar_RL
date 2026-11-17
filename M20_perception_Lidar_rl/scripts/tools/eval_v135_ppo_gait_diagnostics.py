#!/usr/bin/env python3
"""V135 Gait-Stratified PPO Diagnostics Evaluation.

Purpose: Diagnose WHY PPO drifts away from alternation gait.

Usage:
    python scripts/tools/eval_v135_ppo_gait_diagnostics.py \
        --checkpoints logs/.../model_2800.pt logs/.../model_3000.pt \
        --seeds 42,123,456 \
        --num_episodes 100

Output:
    CSV table with gait-stratified PPO metrics for each checkpoint
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add source to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "rl_training"))

import gymnasium as gym
from rl_training.tasks.manager_based.locomotion.velocity.mdp import (
    classify_gait_v135,
    compute_ppo_diagnostics_by_gait,
)


def evaluate_checkpoint_gait_diagnostics(
    checkpoint_path: str,
    task_name: str = "M20-Stair-V13.5-V11FD-NativeAlternation-v0",
    num_episodes: int = 100,
    seeds: list[int] = None,
) -> dict:
    """Evaluate a checkpoint and return gait-stratified metrics.

    Args:
        checkpoint_path: Path to model checkpoint
        task_name: Gym task name
        num_episodes: Number of episodes to run
        seeds: List of random seeds for reproducibility

    Returns:
        dict of aggregated gait-stratified metrics
    """
    if seeds is None:
        seeds = [42]

    print(f"\n{'='*80}")
    print(f"Evaluating: {checkpoint_path}")
    print(f"Task: {task_name}")
    print(f"Episodes: {num_episodes}, Seeds: {seeds}")
    print(f"{'='*80}\n")

    # TODO: Implement actual rollout collection
    # This is a placeholder showing the structure

    # Collect rollouts
    all_gait_labels = []
    all_advantages = []
    all_values = []
    all_returns = []
    all_old_log_probs = []
    all_new_log_probs = []

    for seed in seeds:
        # env = gym.make(task_name, num_envs=16, headless=True, seed=seed)
        # Load checkpoint
        # policy, value = load_checkpoint(checkpoint_path)

        # Run episodes
        # for _ in range(num_episodes // len(seeds)):
        #     obs = env.reset()
        #     # Collect trajectory...
        #     gait_label = classify_gait_v135(env)
        #     all_gait_labels.append(gait_label)
        #     # ...

        pass

    # Placeholder: generate random data to show structure
    n_steps = 1000
    all_gait_labels = torch.randint(0, 4, (n_steps,))
    all_advantages = torch.randn(n_steps)
    all_values = torch.randn(n_steps) * 10
    all_returns = all_values + torch.randn(n_steps)
    all_old_log_probs = torch.randn(n_steps) * 0.1
    all_new_log_probs = all_old_log_probs + torch.randn(n_steps) * 0.05

    # Compute diagnostics
    metrics = compute_ppo_diagnostics_by_gait(
        gait_labels=all_gait_labels,
        advantages=all_advantages,
        values=all_values,
        returns=all_returns,
        old_log_probs=all_old_log_probs,
        new_log_probs=all_new_log_probs,
        clip_param=0.2,
    )

    return metrics


def print_diagnostics_table(results: dict[str, dict]):
    """Print formatted table of gait-stratified metrics across checkpoints.

    Args:
        results: {checkpoint_name: {metric_name: value}}
    """
    print(f"\n{'='*120}")
    print("GAIT-STRATIFIED PPO DIAGNOSTICS")
    print(f"{'='*120}\n")

    gait_types = ["ALT", "NORMAL", "SKIP", "DUAL_SKIP"]
    metrics_to_show = [
        "sample_frac",
        "advantage_mean",
        "return_mean",
        "surrogate_mean",
        "clip_frac",
    ]

    for gait in gait_types:
        print(f"\n{gait} Gait:")
        print("-" * 100)

        header = f"{'Checkpoint':<40}"
        for metric in metrics_to_show:
            header += f"{metric:>15}"
        print(header)
        print("-" * 100)

        for ckpt_name, metrics in results.items():
            row = f"{Path(ckpt_name).name:<40}"
            for metric in metrics_to_show:
                key = f"{gait}/{metric}"
                value = metrics.get(key, 0.0)
                if "frac" in metric:
                    row += f"{value:>15.4f}"
                else:
                    row += f"{value:>15.3f}"
            print(row)

    # Print comparisons
    if "comparison/ALT_vs_NORMAL_advantage" in list(results.values())[0]:
        print(f"\n{'='*120}")
        print("ALT vs NORMAL Comparisons:")
        print("-" * 100)
        print(f"{'Checkpoint':<40}{'Δ Advantage':>15}{'Δ Return':>15}{'Sample Ratio':>15}")
        print("-" * 100)

        for ckpt_name, metrics in results.items():
            print(f"{Path(ckpt_name).name:<40}"
                  f"{metrics.get('comparison/ALT_vs_NORMAL_advantage', 0.0):>15.3f}"
                  f"{metrics.get('comparison/ALT_vs_NORMAL_return', 0.0):>15.3f}"
                  f"{metrics.get('comparison/ALT_vs_NORMAL_sample_ratio', 0.0):>15.3f}")


def main():
    parser = argparse.ArgumentParser(description="V135 Gait-Stratified PPO Diagnostics")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to checkpoints to evaluate",
    )
    parser.add_argument(
        "--task",
        default="M20-Stair-V13.5-V11FD-NativeAlternation-v0",
        help="Gym task name",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of episodes per checkpoint",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,123,456",
        help="Comma-separated list of random seeds",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional CSV output file",
    )

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # Evaluate each checkpoint
    results = {}
    for ckpt_path in args.checkpoints:
        metrics = evaluate_checkpoint_gait_diagnostics(
            checkpoint_path=ckpt_path,
            task_name=args.task,
            num_episodes=args.num_episodes,
            seeds=seeds,
        )
        results[ckpt_path] = metrics

    # Print results
    print_diagnostics_table(results)

    # Save to CSV if requested
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            all_keys = set()
            for metrics in results.values():
                all_keys.update(metrics.keys())
            writer.writerow(["checkpoint"] + sorted(all_keys))

            # Write data
            for ckpt_path, metrics in results.items():
                row = [ckpt_path]
                for key in sorted(all_keys):
                    row.append(metrics.get(key, ""))
                writer.writerow(row)

        print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
