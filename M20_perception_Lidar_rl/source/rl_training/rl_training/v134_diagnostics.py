"""Pure tensor helpers for V13.4 terrain-separated PPO."""

from __future__ import annotations

import torch


def v134_correlated_latent(
    previous: torch.Tensor,
    innovation: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Advance a unit-variance AR(1) exploration latent."""
    if previous.shape != innovation.shape:
        raise ValueError("V13.4 exploration latent tensors must match.")
    if not 0.0 <= float(rho) < 1.0:
        raise ValueError("V13.4 exploration correlation must lie in [0, 1).")
    return float(rho) * previous + (1.0 - float(rho) ** 2) ** 0.5 * innovation


def v134_priority_gradient_gain(
    priority: list[torch.Tensor],
    reference: list[torch.Tensor],
    *,
    base_gain: float,
    minimum_ratio: float,
    maximum_gain: float,
) -> tuple[float, float, float]:
    """Give a discovered ascent signal a minimum norm relative to native ascent."""
    if len(priority) != len(reference) or not priority:
        raise ValueError("V13.4 gradient lists must be non-empty and have equal length.")
    if not 0.0 < float(base_gain) <= float(maximum_gain) or float(minimum_ratio) < 0.0:
        raise ValueError("V13.4 priority-gradient gain bounds are invalid.")
    priority_norm = sum(value.square().sum() for value in priority).sqrt()
    reference_norm = sum(value.square().sum() for value in reference).sqrt()
    required = float(minimum_ratio) * float(reference_norm) / max(float(priority_norm), 1.0e-12)
    gain = min(float(maximum_gain), max(float(base_gain), required))
    return gain, float(priority_norm), float(reference_norm)


def v134_project_gradient_away(
    source: list[torch.Tensor],
    priority: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor, bool]:
    """Remove only the component of ``source`` that opposes ``priority``."""
    if len(source) != len(priority) or not source:
        raise ValueError("V13.4 gradient lists must be non-empty and have equal length.")
    if any(left.shape != right.shape for left, right in zip(source, priority)):
        raise ValueError("V13.4 paired gradients must have matching shapes.")
    dot = sum((left * right).sum() for left, right in zip(source, priority))
    norm = sum(right.square().sum() for right in priority).clamp_min(1.0e-12)
    conflict = bool(dot < 0.0)
    if conflict:
        source = [left - dot / norm * right for left, right in zip(source, priority)]
    return source, dot.detach(), conflict


def v134_masked_mean(value: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    """Return an explicit NaN when a rollout contains no samples of a behavior."""
    if value.shape != mask.shape:
        raise ValueError("V13.4 diagnostic value and mask tensors must have matching shapes.")
    count = int(mask.sum())
    if count == 0:
        return float("nan"), 0
    return float(value[mask].mean()), count


def v134_masked_normalize(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_signal_count: int | None = None,
    minimum_positive_signals: int = 1,
    clip: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Normalize one terrain channel without manufacturing an empty objective.

    When ``positive_signal_count`` is provided, negative-only rollouts train the
    critic but exert no actor pressure.  This prevents early skip penalties from
    teaching the policy to avoid stairs before it has observed a correct move.
    """
    if value.shape != mask.shape:
        raise ValueError("V13.4 normalized value and mask tensors must match.")
    if minimum_positive_signals < 1:
        raise ValueError("V13.4 minimum positive signal count must be positive.")
    output = torch.zeros_like(value)
    count = int(mask.sum())
    if count == 0:
        return output, 0.0
    scale = 1.0
    if positive_signal_count is not None:
        scale = min(1.0, int(positive_signal_count) / float(minimum_positive_signals))
        if scale <= 0.0:
            return output, 0.0
    selected = value[mask]
    normalized = (selected - selected.mean()) / (selected.std(unbiased=False) + 1.0e-8)
    if clip is not None:
        normalized = normalized.clamp(-float(clip), float(clip))
    output[mask] = normalized * scale
    return output, scale


def v134_terrain_normalize(
    value: torch.Tensor,
    ascent_mask: torch.Tensor,
    descent_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize native advantage independently on flat, ascent and descent."""
    if value.shape != ascent_mask.shape or value.shape != descent_mask.shape:
        raise ValueError("V13.4 terrain masks must match the advantage tensor.")
    if bool((ascent_mask & descent_mask).any()):
        raise ValueError("V13.4 ascent and descent masks must be disjoint.")
    output = torch.zeros_like(value)
    for mask in (ascent_mask, descent_mask, ~(ascent_mask | descent_mask)):
        if int(mask.sum()) == 0:
            continue
        selected = value[mask]
        output[mask] = (selected - selected.mean()) / (
            selected.std(unbiased=False) + 1.0e-8
        )
    return output


def v134_combine_terrain_advantages(
    native: torch.Tensor,
    ascent: torch.Tensor,
    descent: torch.Tensor,
    ascent_mask: torch.Tensor,
    descent_mask: torch.Tensor,
    *,
    ascent_weight: float,
    descent_weight: float,
) -> torch.Tensor:
    """Apply each auxiliary objective only on its own terrain column."""
    if any(t.shape != native.shape for t in (ascent, descent, ascent_mask, descent_mask)):
        raise ValueError("V13.4 terrain advantage tensors must have matching shapes.")
    if bool((ascent_mask & descent_mask).any()):
        raise ValueError("V13.4 ascent and descent objectives cannot overlap.")
    return (
        native
        + float(ascent_weight) * ascent * ascent_mask
        + float(descent_weight) * descent * descent_mask
    )


def v134_redistribute_chain_settlement(
    rewards: torch.Tensor,
    adjacent: torch.Tensor,
    chain_event: torch.Tensor,
    dones: torch.Tensor,
    settlement: float,
) -> int:
    """Move a chain dividend onto its in-rollout adjacent transitions."""
    if any(value.shape != rewards.shape for value in (adjacent, chain_event, dones)):
        raise ValueError("V13.4 chain redistribution tensors must have matching shapes.")
    moved = 0
    amount = float(settlement)
    if amount <= 0.0:
        return moved
    # Chain events are sparse.  Copy the boolean ledger once instead of forcing
    # a CUDA synchronization for every environment at every rollout step.
    adjacent_cpu = adjacent[..., 0].detach().cpu()
    dones_cpu = dones[..., 0].detach().cpu()
    events = chain_event[..., 0].nonzero(as_tuple=False).detach().cpu().tolist()
    for step, env_id in events:
        previous_resets = dones_cpu[:step, env_id].nonzero(as_tuple=False)
        segment_start = int(previous_resets[-1, 0]) + 1 if previous_resets.numel() else 0
        history = adjacent_cpu[segment_start:step + 1, env_id].nonzero(as_tuple=False)
        if not history.numel():
            continue
        history = (history[-3:, 0] + segment_start).tolist()
        rewards[step, env_id, 0] -= amount
        rewards[history, env_id, 0] += amount / len(history)
        moved += 1
    return moved


__all__ = [
    "v134_correlated_latent", "v134_priority_gradient_gain",
    "v134_project_gradient_away",
    "v134_masked_mean", "v134_masked_normalize", "v134_terrain_normalize",
    "v134_combine_terrain_advantages", "v134_redistribute_chain_settlement",
]
