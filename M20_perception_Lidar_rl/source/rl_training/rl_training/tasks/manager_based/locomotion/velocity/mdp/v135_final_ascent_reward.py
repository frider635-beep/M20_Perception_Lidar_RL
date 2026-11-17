"""V13.5-Final: Single-leg dominance-based ascent reward.

Core principle:
- B1: reward max(dL, dR) * dominance, where dominance = |dL-dR|/(dL+dR)
- B2+: reward expected_leg_delta * share, where share = expected/(expected+other)

This replaces all dense negative style penalties with positive single-leg progress shaping.
"""

from __future__ import annotations

import torch


def v135_final_progress_eligible(ascent, target_valid, target_height_valid):
    """Keep learning against the locked Bk after the live scanner moves on."""
    if not (ascent.shape == target_valid.shape == target_height_valid.shape):
        raise ValueError("V13.5 ascent eligibility tensors must have matching shapes.")
    return ascent & target_valid & target_height_valid


def v135_locked_front_potential(
    clearance: torch.Tensor,
    crossed: torch.Tensor,
    support_steps: torch.Tensor,
    wheel_forward: torch.Tensor,
    wheel_lateral: torch.Tensor,
    target_height_valid: torch.Tensor,
    *,
    crossing_span: float,
    step_width: float,
    wheel_radius: float,
    lateral_tolerance: float,
    confirmation_frames: int,
    swing_advance: torch.Tensor | None = None,
) -> torch.Tensor:
    """V13 potential evaluated on the strict physical riser, never on scanner slots."""
    if any(x.shape != clearance.shape for x in (crossed, support_steps, wheel_forward, wheel_lateral)):
        raise ValueError("V13.5 locked potential requires matching [N, 2] inputs.")
    if clearance.ndim != 2 or clearance.shape[1] != 2 or target_height_valid.shape != clearance.shape[:1]:
        raise ValueError("V13.5 locked potential expects two front wheels per environment.")
    in_target_region = (
        (wheel_forward < torch.as_tensor(step_width, device=wheel_forward.device).reshape(-1, 1) - float(wheel_radius))
        & (wheel_lateral.abs() <= lateral_tolerance)
    )
    on_target_tread = crossed & (wheel_forward >= 0.0) & in_target_region
    crossing = (wheel_forward / float(crossing_span)).clamp(0.0, 1.0) * on_target_tread.float()
    support = (support_steps.float() / float(confirmation_frames)).clamp(0.0, 1.0)
    # Height alone, especially far before the edge, is not target progress.
    approach = ((wheel_forward + float(crossing_span)) / float(crossing_span)).clamp(0, 1)
    if swing_advance is not None:
        approach *= (swing_advance / float(crossing_span)).clamp(0, 1)
    potential = (
        0.5 * clearance.clamp(0.0, 1.0) * approach * in_target_region.float()
        + 0.25 * crossing + 0.25 * support
    )
    return potential.clamp(0.0, 1.0) * target_height_valid[:, None].float()


def v135_final_leader_dominance_reward(
    per_leg_delta: torch.Tensor,
    expected_side: torch.Tensor,
    seeded: torch.Tensor,
    active: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    """Single-leg leader progress with dominance/share quality factor.

    Args:
        per_leg_delta: [num_envs, 2] FL/FR historical-best positive increments
        expected_side: [num_envs] 0=FL, 1=FR (only used when seeded)
        seeded: [num_envs] whether B1 has been seeded
        active: [num_envs] whether ascent learning is active
        step_dt: simulation step duration

    Returns:
        [num_envs] reward for single-leg dominant progress
    """
    if per_leg_delta.ndim != 2 or per_leg_delta.shape[1] != 2:
        raise ValueError("V135-Final: per_leg_delta must be [num_envs, 2]")

    eps = 1e-6
    dL = per_leg_delta[:, 0]
    dR = per_leg_delta[:, 1]

    # B1 stage: reward whichever leg dominates
    b1_active = active & ~seeded
    leader_b1 = torch.maximum(dL, dR)
    dominance = (dL - dR).abs() / (dL + dR + eps)
    reward_b1 = leader_b1 * dominance

    # B2+ stage: reward expected leg's progress weighted by its share
    b2_active = active & seeded
    expected_mask = expected_side.clamp(0, 1)

    # Gather expected and other leg deltas
    expected_delta = torch.where(expected_mask == 0, dL, dR)
    other_delta = torch.where(expected_mask == 0, dR, dL)

    share = expected_delta / (expected_delta + other_delta + eps)
    reward_b2 = expected_delta * share

    # Combine stages
    reward = torch.where(b1_active, reward_b1, torch.where(b2_active, reward_b2, torch.zeros_like(dL)))

    return reward / step_dt


def _init_final_ascent(env):
    """Initialize V135-Final ascent reward state."""
    if hasattr(env, "_v135_final_ascent"):
        return

    n, d = env.num_envs, env.device

    env._v135_final_ascent = {
        "last_episode_length": env.episode_length_buf.clone(),
        "best_potential": torch.zeros((n, 2), device=d),
        "current_potential": torch.zeros((n, 2), device=d),
        "target_id_last": torch.zeros(n, device=d, dtype=torch.long),
        "target_lower_z": torch.zeros(n, device=d),
        "target_lower_valid": torch.zeros(n, device=d, dtype=torch.bool),
        "clearance_best": torch.zeros((n, 2), device=d),
        "swing_origin_forward": torch.zeros((n, 2), device=d),
        "fl_delta": torch.zeros(n, device=d),
        "fr_delta": torch.zeros(n, device=d),
    }
    env._v135_final_ascent_last_step = -1


def v135_final_ascent_state(env, sensor_cfg=None):
    """Compute historical-best deltas for V135-Final leader reward."""
    from .v11_continuous import v11_state
    from .v135_riser_identity import v135_riser_identity_state

    _init_final_ascent(env)
    step = int(getattr(env, "common_step_counter", 0))
    if env._v135_final_ascent_last_step == step:
        return env._v135_final_ascent

    s = env._v135_final_ascent
    reset = env.episode_length_buf < s["last_episode_length"]
    s["best_potential"][reset] = 0.0
    s["current_potential"][reset] = 0.0
    s["clearance_best"][reset] = 0.0
    s["target_lower_valid"][reset] = False

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)["reward_snapshot"]
    landing = base["landing"]

    changed_target = reset | (strict["target_id"] != s["target_id_last"])
    s["best_potential"][changed_target] = 0.0
    s["clearance_best"][changed_target] = 0.0
    s["target_lower_valid"][changed_target] = False
    s["target_id_last"] = strict["target_id"].clone()
    s["target_lower_z"] = strict["target_lower_z"].clone()
    s["target_lower_valid"] = strict["target_height_valid"].clone()

    wheel_world = env.scene["robot"].data.body_pos_w[:, env._v11_front_body_ids]
    wheel_bottom = wheel_world[..., 2] - float(env.cfg.v11_perception.wheel_radius)
    lift_span = (
        strict["target_upper_z"] - s["target_lower_z"]
        + float(env.cfg.v11_perception.clearance_margin)
    ).clamp_min(1.0e-6)
    clearance = ((wheel_bottom - s["target_lower_z"][:, None]) / lift_span[:, None]).clamp(0.0, 1.0)
    height_valid = strict["target_height_valid"] & s["target_lower_valid"]
    wheel_delta = wheel_world[..., :2] - strict["target_world_xy"][:, None, :]
    forward = strict["forward_world_xy"]
    lateral_axis = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    wheel_forward = (wheel_delta * forward[:, None, :]).sum(dim=2)
    wheel_lateral = ((wheel_world[..., :2] - env.scene.env_origins[:, None, :2]) * lateral_axis[:, None, :]).sum(dim=2)
    in_target_region = (
        (wheel_forward < strict["step_width"][:, None] - float(env.cfg.v11_perception.wheel_radius))
        & (wheel_lateral.abs() <= strict["tread_half_width"])
    )
    s["swing_origin_forward"] = torch.where(
        changed_target[:, None], wheel_forward, s["swing_origin_forward"])
    force = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids, 2]
    on_lower = ((wheel_bottom - strict["target_lower_z"][:, None]).abs()
                <= float(env.cfg.v11_perception.upper_support_tolerance))
    grounded = on_lower & (force > float(env.cfg.v11_perception.contact_force_threshold))
    s["swing_origin_forward"] = torch.where(grounded, wheel_forward, s["swing_origin_forward"])
    advance = (wheel_forward - s["swing_origin_forward"]).clamp_min(0)
    s["clearance_best"] = clearance
    potential = v135_locked_front_potential(
        s["clearance_best"], strict["locked_crossed"], strict["locked_support_steps"],
        wheel_forward, wheel_lateral, height_valid,
        crossing_span=float(env.cfg.v11_perception.wheel_radius) + float(env.cfg.v11_perception.front_margin),
        step_width=strict["step_width"],
        wheel_radius=float(env.cfg.v11_perception.wheel_radius),
        lateral_tolerance=strict["tread_half_width"],
        confirmation_frames=int(env.cfg.v11_perception.front_support_confirmation_frames),
        swing_advance=advance,
    )
    s["current_potential"] = potential

    # Active: on ascending stairs with valid target
    terrain_types = env.scene.terrain.terrain_types
    ascending_ids = torch.tensor(
        tuple(int(v) for v in env.cfg.v11_perception.ascending_stair_terrain_type_ids),
        device=env.device, dtype=terrain_types.dtype,
    )
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    ascent = (terrain_types[:, None] == ascending_ids[None, :]).any(1) & (
        command_x > float(env.cfg.v11_perception.forward_command_threshold)
    )

    active = v135_final_progress_eligible(ascent, strict["target_valid"], height_valid)

    # Update historical best
    current_best = torch.where(
        active[:, None],
        torch.maximum(s["best_potential"], potential),
        s["best_potential"],
    )

    # Positive deltas
    per_leg_delta = (current_best - s["best_potential"]).clamp_min(0)
    s["best_potential"] = current_best

    # Store per-leg deltas
    s["fl_delta"] = per_leg_delta[:, 0]
    s["fr_delta"] = per_leg_delta[:, 1]

    s["last_episode_length"] = env.episode_length_buf.clone()
    env._v135_final_ascent_last_step = step
    return s


def v135_final_leader_progress_reward(env, sensor_cfg=None):
    """Bare single-leg leader progress reward (for 0.5x weight)."""
    from .v135_riser_identity import v135_riser_identity_state

    state = v135_final_ascent_state(env, sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)["reward_snapshot"]

    terrain_types = env.scene.terrain.terrain_types
    ascending_ids = torch.tensor(
        tuple(int(v) for v in env.cfg.v11_perception.ascending_stair_terrain_type_ids),
        device=env.device, dtype=terrain_types.dtype,
    )
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    ascent = (terrain_types[:, None] == ascending_ids[None, :]).any(1) & (
        command_x > float(env.cfg.v11_perception.forward_command_threshold)
    )

    active = ascent & strict["target_valid"]
    per_leg_delta = torch.stack((state["fl_delta"], state["fr_delta"]), dim=1)

    reward = v135_final_leader_dominance_reward(
        per_leg_delta,
        strict["expected_side"],
        strict["seeded"],
        active,
        float(env.step_dt),
    )
    from .v135_descent_stance import v135_post_stair_flat_state
    return reward * (1.0 - v135_post_stair_flat_state(env, sensor_cfg)["flat_blend"])


def v135_final_velocity_coupled_leader_reward(env, sensor_cfg=None):
    """Velocity-quality coupled leader progress (for 1.5x weight)."""
    from .v11_continuous import v11_state

    base_reward = v135_final_leader_progress_reward(env, sensor_cfg)

    # Get velocity quality from V11
    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    command = env.command_manager.get_command("base_velocity")
    lin_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]
    vel_error = (lin_vel - command[:, :2]).square().sum(1)

    # Gaussian quality: exp(-error^2 / sigma^2)
    sigma_sq = 0.25  # same as V13
    velocity_quality = torch.exp(-vel_error / sigma_sq)

    return base_reward * velocity_quality


__all__ = [name for name in globals() if name.startswith("v135_final_")]
