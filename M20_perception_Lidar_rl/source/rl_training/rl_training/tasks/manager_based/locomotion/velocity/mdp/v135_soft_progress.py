"""V13.5-Soft: continuous single-leg progress shaping for strict validation.

Soft-learning / Strict-validation approach:
- Learning signal: continuous positive rewards for single-leg progress toward target riser
- Validation: strict target_id++ only on valid completion (correct side + physical riser + crossing + support)
- Chain failed: closes ordered_success metric, but NOT local gait-learning shaping

Key difference from V13.5:
- V13.5: no reward until complete valid completion → sparse signal
- V13.5-Soft: reward partial progress continuously → dense signal
- Both: same strict validation for target_id advancement
"""

from __future__ import annotations

import torch


def v135_soft_single_leg_potential(
    lift_height: torch.Tensor,
    clearance_above_target: torch.Tensor,
    forward_projection: torch.Tensor,
    crossing_score: torch.Tensor,
    support_weight: torch.Tensor,
    *,
    w_lift: float = 0.20,
    w_clearance: float = 0.20,
    w_advance: float = 0.25,
    w_crossing: float = 0.20,
    w_support: float = 0.15,
) -> torch.Tensor:
    """Compute single-leg progress potential toward target riser.

    Returns bounded potential in [0, 1] representing how far this leg has progressed
    toward completing a single-leg ascent of the target riser.

    All input tensors: [num_envs, 2] for FL/FR.
    """
    if not all(x.shape == lift_height.shape for x in (
        clearance_above_target, forward_projection, crossing_score, support_weight
    )):
        raise ValueError("V135-Soft: all per-leg tensors must have matching shapes [num_envs, 2]")

    # Normalize each component to [0, 1]
    P_lift = lift_height.clamp(0, 1)
    P_clearance = clearance_above_target.clamp(0, 1)
    P_advance = forward_projection.clamp(0, 1)
    P_crossing = crossing_score.clamp(0, 1)
    P_support = support_weight.clamp(0, 1)

    # Weighted sum
    potential = (
        float(w_lift) * P_lift +
        float(w_clearance) * P_clearance +
        float(w_advance) * P_advance +
        float(w_crossing) * P_crossing +
        float(w_support) * P_support
    )

    return potential.clamp(0, 1)


def v135_soft_historical_best_progress(
    previous_best: torch.Tensor,
    current_potential: torch.Tensor,
    active: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Track historical best and return positive increments.

    Args:
        previous_best: [num_envs, 2] FL/FR historical best
        current_potential: [num_envs, 2] FL/FR current potential
        active: [num_envs] whether this env is active for progress tracking

    Returns:
        dict with:
            - best: [num_envs, 2] updated historical best
            - delta: [num_envs, 2] positive increment (0 if no improvement)
    """
    if previous_best.shape != current_potential.shape or previous_best.ndim != 2 or previous_best.shape[1] != 2:
        raise ValueError("V135-Soft: potential tensors must be [num_envs, 2]")
    if active.shape != previous_best.shape[:1]:
        raise ValueError("V135-Soft: active mask must match num_envs")

    # Update best only when active
    best = torch.where(
        active[:, None],
        torch.maximum(previous_best, current_potential),
        previous_best,
    )

    # Positive increment
    delta = (best - previous_best).clamp_min(0)

    return {"best": best, "delta": delta}


def v135_soft_b1_any_leg_progress(
    per_leg_delta: torch.Tensor,
    active: torch.Tensor,
    *,
    w_progress: float = 1.0,
) -> torch.Tensor:
    """B1 stage: reward whichever leg makes better progress.

    Args:
        per_leg_delta: [num_envs, 2] FL/FR progress increments
        active: [num_envs] whether B1 learning is active

    Returns:
        [num_envs] reward for the leg that made most progress
    """
    if per_leg_delta.ndim != 2 or per_leg_delta.shape[1] != 2:
        raise ValueError("V135-Soft B1: per_leg_delta must be [num_envs, 2]")
    if active.shape != per_leg_delta.shape[:1]:
        raise ValueError("V135-Soft B1: active must match num_envs")

    # Reward the max progress between FL and FR
    max_progress = per_leg_delta.amax(dim=1)

    return float(w_progress) * max_progress * active.float()


def v135_soft_b1_overlap_cost(
    per_leg_delta: torch.Tensor,
    active: torch.Tensor,
    *,
    w_overlap: float = 0.3,
) -> torch.Tensor:
    """B1 stage: penalize both legs making progress simultaneously.

    Args:
        per_leg_delta: [num_envs, 2] FL/FR progress increments
        active: [num_envs] whether B1 learning is active

    Returns:
        [num_envs] penalty when both legs progress together
    """
    if per_leg_delta.ndim != 2 or per_leg_delta.shape[1] != 2:
        raise ValueError("V135-Soft overlap: per_leg_delta must be [num_envs, 2]")
    if active.shape != per_leg_delta.shape[:1]:
        raise ValueError("V135-Soft overlap: active must match num_envs")

    # Penalize the minimum progress (when both legs move together)
    overlap = per_leg_delta.amin(dim=1)

    return -float(w_overlap) * overlap * active.float()


def v135_soft_expected_leg_progress(
    per_leg_delta: torch.Tensor,
    expected_side: torch.Tensor,
    active: torch.Tensor,
    *,
    w_progress: float = 1.0,
) -> torch.Tensor:
    """B2/B3 stage: reward only the expected leg's progress.

    Args:
        per_leg_delta: [num_envs, 2] FL/FR progress increments
        expected_side: [num_envs] 0=FL, 1=FR
        active: [num_envs] whether this stage is active

    Returns:
        [num_envs] reward for expected leg's progress
    """
    if per_leg_delta.ndim != 2 or per_leg_delta.shape[1] != 2:
        raise ValueError("V135-Soft expected: per_leg_delta must be [num_envs, 2]")
    if expected_side.shape != per_leg_delta.shape[:1]:
        raise ValueError("V135-Soft expected: expected_side must match num_envs")
    if active.shape != per_leg_delta.shape[:1]:
        raise ValueError("V135-Soft expected: active must match num_envs")

    # Gather expected leg's delta
    side = expected_side.clamp(0, 1)
    expected_delta = per_leg_delta.gather(1, side[:, None]).squeeze(1)

    return float(w_progress) * expected_delta * active.float()


def _init_soft_state(env):
    """Initialize V135-Soft progress tracking state."""
    if hasattr(env, "_v135_soft_progress"):
        return

    n, d = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=d, dtype=torch.bool)

    env._v135_soft_progress = {
        "last_episode_length": env.episode_length_buf.clone(),
        # Historical best potentials for FL/FR
        "best_potential": torch.zeros((n, 2), device=d),
        # Current frame deltas
        "fl_delta": torch.zeros(n, device=d),
        "fr_delta": torch.zeros(n, device=d),
        # Accumulated metrics
        "progress_sum": torch.zeros((), device=d),
        "overlap_sum": torch.zeros((), device=d),
    }
    env._v135_soft_progress_last_step = -1


def v135_soft_progress_state(env, sensor_cfg=None):
    """Compute continuous single-leg progress shaping.

    Returns state dict with:
        - fl_delta, fr_delta: [num_envs] per-leg progress increments this step
        - best_potential: [num_envs, 2] historical best for FL/FR
    """
    from .v11_continuous import v11_state
    from .v135_riser_identity import v135_riser_identity_state

    _init_soft_state(env)
    step = int(getattr(env, "common_step_counter", 0))
    if env._v135_soft_progress_last_step == step:
        return env._v135_soft_progress

    s = env._v135_soft_progress
    reset = env.episode_length_buf < s["last_episode_length"]
    s["best_potential"][reset] = 0.0

    # Get base V11 and strict V135 states
    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)
    landing = base["landing"]

    # Compute current single-leg potential from V11 landing components
    # Use V13's potential calculation as base
    from .v13_alternation import v13_front_riser_potential
    potential = v13_front_riser_potential(
        landing["max_clearance_progress"],
        landing["crossing_clearance_ok"],
        landing["wheel_yaw"][..., 0],
        landing["target_riser_x"],
        landing["support"],
        crossing_span=float(env.cfg.v11_perception.wheel_radius) + float(env.cfg.v11_perception.front_margin),
    )

    # Active condition: on stairs, target valid, NOT terminated by irrecoverable failure
    terrain_types = env.scene.terrain.terrain_types
    ascending_ids = torch.tensor(
        tuple(int(v) for v in env.cfg.v11_perception.ascending_stair_terrain_type_ids),
        device=env.device, dtype=terrain_types.dtype,
    )
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    ascent = (terrain_types[:, None] == ascending_ids[None, :]).any(1) & (
        command_x > float(env.cfg.v11_perception.forward_command_threshold)
    )

    # Active for progress: on stairs + target valid
    # KEY CHANGE: NOT gated by ordered_chain_failed
    # chain_failed only affects ordered_success metric, not local shaping
    active_for_progress = ascent & strict["target_valid"] & landing["target_active"]

    # Historical best update
    history = v135_soft_historical_best_progress(
        s["best_potential"],
        potential,
        active_for_progress,
    )

    s["best_potential"] = history["best"]
    per_leg_delta = history["delta"]  # [num_envs, 2]

    # Reset best on valid completion (new target acquired)
    s["best_potential"] = torch.where(
        strict["valid"][:, None],
        torch.zeros_like(s["best_potential"]),
        s["best_potential"],
    )

    # Store per-leg deltas
    s["fl_delta"] = per_leg_delta[:, 0]
    s["fr_delta"] = per_leg_delta[:, 1]

    # Accumulate metrics
    s["progress_sum"] += per_leg_delta.sum()
    s["overlap_sum"] += per_leg_delta.amin(dim=1).sum()

    s["last_episode_length"] = env.episode_length_buf.clone()
    env._v135_soft_progress_last_step = step
    return s


def v135_soft_b1_progress_reward(env, sensor_cfg=None):
    """B1 stage: reward whichever leg makes better single-leg progress."""
    from .v135_riser_identity import v135_riser_identity_state

    soft = v135_soft_progress_state(env, sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)

    # B1 stage: target_id == 1, not yet seeded
    b1_active = (strict["target_id"] == 1) & strict["target_valid"]

    per_leg_delta = torch.stack((soft["fl_delta"], soft["fr_delta"]), dim=1)

    return v135_soft_b1_any_leg_progress(per_leg_delta, b1_active)


def v135_soft_b1_overlap_penalty(env, sensor_cfg=None):
    """B1 stage: penalize both legs progressing simultaneously."""
    from .v135_riser_identity import v135_riser_identity_state

    soft = v135_soft_progress_state(env, sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)

    b1_active = (strict["target_id"] == 1) & strict["target_valid"]

    per_leg_delta = torch.stack((soft["fl_delta"], soft["fr_delta"]), dim=1)

    return v135_soft_b1_overlap_cost(per_leg_delta, b1_active)


def v135_soft_expected_progress_reward(env, sensor_cfg=None):
    """B2+ stage: reward only the expected leg's progress."""
    from .v135_riser_identity import v135_riser_identity_state

    soft = v135_soft_progress_state(env, sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)

    # B2+ stage: target_id > 1, seeded, has expected side
    b2_plus_active = (strict["target_id"] > 1) & strict["seeded"] & strict["target_valid"]

    per_leg_delta = torch.stack((soft["fl_delta"], soft["fr_delta"]), dim=1)

    return v135_soft_expected_leg_progress(
        per_leg_delta,
        strict["expected_side"],
        b2_plus_active,
    )


def v135_soft_expected_overlap_penalty(env, sensor_cfg=None):
    """B2+ stage: penalize other leg also progressing."""
    from .v135_riser_identity import v135_riser_identity_state

    soft = v135_soft_progress_state(env, sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)

    b2_plus_active = (strict["target_id"] > 1) & strict["seeded"] & strict["target_valid"]

    per_leg_delta = torch.stack((soft["fl_delta"], soft["fr_delta"]), dim=1)

    return v135_soft_b1_overlap_cost(per_leg_delta, b2_plus_active)


def v135_soft_progress_metric(env, env_ids, key, sensor_cfg=None):
    """Report soft progress metrics."""
    soft = v135_soft_progress_state(env, sensor_cfg)
    from .v135_riser_identity import v135_riser_identity_state
    strict = v135_riser_identity_state(env, sensor_cfg)

    if key == "single_leg_progress":
        return soft["progress_sum"] / max(1, int(getattr(env, "common_step_counter", 1)))
    if key == "overlap_progress":
        return soft["overlap_sum"] / max(1, int(getattr(env, "common_step_counter", 1)))
    if key == "fl_mean_potential":
        return soft["best_potential"][:, 0].mean()
    if key == "fr_mean_potential":
        return soft["best_potential"][:, 1].mean()
    raise ValueError(f"Unknown V135-Soft metric: {key}")


__all__ = [name for name in globals() if name.startswith("v135_soft_")]
