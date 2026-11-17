# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""V135 Gait-Stratified PPO Diagnostics.

Purpose: Understand WHY PPO updates drift away from alternation gait.

Three possible causes:
1. Sample imbalance: p_ALT << p_NORMAL, but A_ALT > A_NORMAL
   -> alternation is valuable but overwhelmed by normal gait samples

2. Reward ranking: A_ALT < A_NORMAL
   -> alternation is genuinely less rewarding

3. Gradient conflict: A_ALT > A_NORMAL but policy still drifts
   -> multi-objective gradient conflict

Reference: 2026-09-18 analysis of V13.1 retention test NO_GO_RETURN_ORDERING
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def classify_gait_v135(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Classify each environment's current gait type based on V135 riser identity state.

    Returns:
        gait_labels [N]: 0=NORMAL, 1=ALT, 2=SKIP, 3=DUAL_SKIP
    """
    from .v135_riser_identity import v135_riser_identity_state

    s = v135_riser_identity_state(env)

    # Initialize all as NORMAL (0)
    labels = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    # Ordered alternation (1)
    labels = torch.where(s["ordered"], torch.ones_like(labels), labels)

    # Skip (2) - overrides ordered if both somehow true
    labels = torch.where(s["skip"], 2 * torch.ones_like(labels), labels)

    # Dual skip (3) - highest priority
    labels = torch.where(s["dual_skip"], 3 * torch.ones_like(labels), labels)

    return labels


def compute_ppo_diagnostics_by_gait(
    gait_labels: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    clip_param: float = 0.2,
) -> dict:
    """Compute PPO metrics stratified by gait type.

    Args:
        gait_labels: [N] 0=NORMAL, 1=ALT, 2=SKIP, 3=DUAL_SKIP
        advantages: [N] advantage estimates
        values: [N] value predictions
        returns: [N] actual returns
        old_log_probs: [N] log prob under old policy
        new_log_probs: [N] log prob under new policy
        clip_param: PPO clip parameter

    Returns:
        dict of metrics per gait type
    """
    gait_names = ["NORMAL", "ALT", "SKIP", "DUAL_SKIP"]
    metrics = {}

    for gait_id, gait_name in enumerate(gait_names):
        mask = (gait_labels == gait_id)
        n_samples = mask.sum().item()

        if n_samples == 0:
            # No samples of this gait type
            metrics[f"{gait_name}/sample_count"] = 0
            metrics[f"{gait_name}/sample_frac"] = 0.0
            continue

        # Sample statistics
        metrics[f"{gait_name}/sample_count"] = n_samples
        metrics[f"{gait_name}/sample_frac"] = n_samples / len(gait_labels)

        # Advantage statistics
        adv_gait = advantages[mask]
        metrics[f"{gait_name}/advantage_mean"] = adv_gait.mean().item()
        metrics[f"{gait_name}/advantage_std"] = adv_gait.std().item()

        # Value error
        val_err = (values[mask] - returns[mask])
        metrics[f"{gait_name}/value_error_mean"] = val_err.mean().item()
        metrics[f"{gait_name}/value_error_std"] = val_err.std().item()

        # Return statistics
        ret_gait = returns[mask]
        metrics[f"{gait_name}/return_mean"] = ret_gait.mean().item()
        metrics[f"{gait_name}/return_std"] = ret_gait.std().item()

        # PPO surrogate
        ratio = torch.exp(new_log_probs[mask] - old_log_probs[mask])
        surr1 = ratio * adv_gait
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_gait
        surrogate = torch.min(surr1, surr2)

        metrics[f"{gait_name}/surrogate_mean"] = surrogate.mean().item()
        metrics[f"{gait_name}/clip_frac"] = ((ratio < 1.0 - clip_param) | (ratio > 1.0 + clip_param)).float().mean().item()
        metrics[f"{gait_name}/ratio_mean"] = ratio.mean().item()

    # Cross-gait comparisons
    if metrics.get("ALT/sample_count", 0) > 0 and metrics.get("NORMAL/sample_count", 0) > 0:
        metrics["comparison/ALT_vs_NORMAL_advantage"] = metrics["ALT/advantage_mean"] - metrics["NORMAL/advantage_mean"]
        metrics["comparison/ALT_vs_NORMAL_return"] = metrics["ALT/return_mean"] - metrics["NORMAL/return_mean"]
        metrics["comparison/ALT_vs_NORMAL_sample_ratio"] = metrics["ALT/sample_frac"] / max(metrics["NORMAL/sample_frac"], 1e-6)

    return metrics


def v135_ppo_gait_diagnostics_hook(
    env: ManagerBasedRLEnv,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    clip_param: float = 0.2,
) -> dict:
    """Hook to be called during PPO update to log gait-stratified metrics.

    This should be called after advantages are computed but before policy update.

    Returns:
        dict of metrics to log to TensorBoard
    """
    # Classify gait for each environment
    gait_labels = classify_gait_v135(env)

    # Compute diagnostics
    metrics = compute_ppo_diagnostics_by_gait(
        gait_labels=gait_labels,
        advantages=advantages,
        values=values,
        returns=returns,
        old_log_probs=old_log_probs,
        new_log_probs=new_log_probs,
        clip_param=clip_param,
    )

    return {"ppo_gait_diag/" + k: v for k, v in metrics.items()}


__all__ = [
    "classify_gait_v135",
    "compute_ppo_diagnostics_by_gait",
    "v135_ppo_gait_diagnostics_hook",
]
