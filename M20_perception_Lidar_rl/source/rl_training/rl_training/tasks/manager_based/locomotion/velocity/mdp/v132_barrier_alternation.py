"""V13.2 recoverable barrier-style alternation on the native V13 task.

The V11-FD/V13 task rewards remain authoritative.  This module contributes one
separate style return and a deployable event-state observation.  The first
unique front-leg lift towards B1 fixes the FL/FR parity for the complete stair
flight.  A violation is charged once and then re-synchronised; it never enters
an absorbing failure state and never gates physical stair-completion credit.
"""

from __future__ import annotations

import math

import torch


APPROACH_B1 = 0
ALTERNATING_RISERS = 1
TOP_PLATFORM_FOLLOW = 2
PLATEAU_ROLL = 3


def v132_episode_reset_mask(
    current_length: torch.Tensor, previous_length: torch.Tensor, *, same_step: bool,
) -> torch.Tensor:
    """Detect resets both across steps and in post-reset observation recompute.

    Isaac computes reward before resetting terminated environments, then builds
    the next observation at the same common step.  The strict ``<`` branch
    catches that same-step reset; ``<=`` on a new step also catches an episode
    that terminated on its very first action (length one followed by length
    one), which a simple decreasing-counter test misses.
    """
    if current_length.shape != previous_length.shape:
        raise ValueError("V13.2 episode-length ledgers must have matching shapes.")
    return current_length < previous_length if same_step else current_length <= previous_length


def v132_relaxed_log_barrier(slack: torch.Tensor, delta: float) -> torch.Tensor:
    """C2 relaxed log barrier used by the style-band score.

    It equals ``log(slack)`` above ``delta`` and uses the tangent quadratic
    continuation below it, so violated constraints retain a finite, steep
    gradient instead of producing NaNs or a hard gate.
    """
    if float(delta) <= 0.0:
        raise ValueError("V13.2 barrier delta must be positive.")
    d = torch.as_tensor(float(delta), dtype=slack.dtype, device=slack.device)
    quadratic = math.log(float(delta)) - 0.5 * ((slack - 2.0 * d) / d).square() + 0.5
    return torch.where(slack > d, torch.log(slack.clamp_min(torch.finfo(slack.dtype).tiny)), quadratic)


def v132_barrier_band_score(
    value: torch.Tensor, lower: float, upper: float, delta: float,
) -> torch.Tensor:
    """Return a bounded ``[0, 1]`` relaxed-barrier score for an open band."""
    if not float(lower) < float(upper):
        raise ValueError("V13.2 barrier band must have lower < upper.")
    left = value - float(lower)
    right = float(upper) - value
    barrier = v132_relaxed_log_barrier(left, delta) + v132_relaxed_log_barrier(right, delta)
    half_width = 0.5 * (float(upper) - float(lower))
    maximum = 2.0 * math.log(half_width)
    return torch.exp((barrier - maximum).clamp(max=0.0)).clamp(0.0, 1.0)


def v132_target_index(
    reference_world_xy: torch.Tensor,
    b1_world_xy: torch.Tensor,
    forward_world_xy: torch.Tensor,
    valid: torch.Tensor,
    *,
    step_width: float,
) -> torch.Tensor:
    """Map a detected face to B1/B2/... without silently renaming a skip."""
    if reference_world_xy.shape != b1_world_xy.shape or reference_world_xy.shape[-1] != 2:
        raise ValueError("V13.2 target positions must be matching [num_envs, 2] tensors.")
    if forward_world_xy.shape != reference_world_xy.shape or valid.shape != reference_world_xy.shape[:1]:
        raise ValueError("V13.2 target-index tensors have incompatible shapes.")
    if float(step_width) <= 0.0:
        raise ValueError("V13.2 step width must be positive.")
    forward = forward_world_xy / forward_world_xy.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    projected = ((reference_world_xy - b1_world_xy) * forward).sum(dim=1)
    index = torch.round(projected / float(step_width)).long() + 1
    return torch.where(valid, index.clamp_min(1), torch.zeros_like(index))


def v132_expected_side(
    seed_side: torch.Tensor, target_index: torch.Tensor, parity_seeded: torch.Tensor,
) -> torch.Tensor:
    """Return the fixed B1-seeded side for an arbitrary riser index."""
    if seed_side.shape != target_index.shape or seed_side.shape != parity_seeded.shape:
        raise ValueError("V13.2 parity tensors must have matching [num_envs] shapes.")
    side = torch.where((target_index.remainder(2) == 1), seed_side, 1 - seed_side)
    return torch.where(parity_seeded & (target_index > 0), side, torch.full_like(side, -1))


def v132_latch_first_lift(
    seed_side: torch.Tensor,
    parity_seeded: torch.Tensor,
    lift_ready: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Latch exactly one B1 leader; a bilateral onset never wins a tie."""
    if seed_side.shape != parity_seeded.shape or lift_ready.shape != (seed_side.shape[0], 2):
        raise ValueError("V13.2 first-lift tensors have incompatible shapes.")
    count = lift_ready.long().sum(dim=1)
    unique = ~parity_seeded & (count == 1)
    simultaneous = ~parity_seeded & (count == 2)
    leader = lift_ready.long().argmax(dim=1)
    return {
        "seed_side": torch.where(unique, leader, seed_side),
        "parity_seeded": parity_seeded | unique,
        "unique_event": unique,
        "simultaneous_event": simultaneous,
    }


def v132_deploy_clearance_onset(
    max_progress: torch.Tensor,
    previous_forward: torch.Tensor,
    previous_valid: torch.Tensor,
    wheel_forward: torch.Tensor,
    wheel_bottom_z: torch.Tensor,
    target_lower_z: torch.Tensor,
    step_height: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    clearance_margin: float,
    onset_progress: float,
    forward_epsilon: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    """Contact-free counterpart of V11's forward-qualified clearance escrow."""
    shape = max_progress.shape
    if shape[-1:] != (2,) or any(
        value.shape != shape
        for value in (previous_forward, previous_valid, wheel_forward, wheel_bottom_z)
    ):
        raise ValueError("V13.2 deploy clearance expects matching [num_envs, 2] tensors.")
    if any(value.shape != shape[:1] for value in (target_lower_z, step_height, target_valid)):
        raise ValueError("V13.2 deploy target geometry must have shape [num_envs].")
    if clearance_margin <= 0.0 or not 0.0 < onset_progress < 1.0 or forward_epsilon <= 0.0:
        raise ValueError("V13.2 deploy clearance thresholds are invalid.")
    denominator = (step_height + float(clearance_margin)).clamp_min(1.0e-6)
    progress = (
        (wheel_bottom_z - target_lower_z[:, None]) / denominator[:, None]
    ).clamp(0.0, 1.0)
    before = wheel_forward < 0.0
    advancing = previous_valid & (wheel_forward > previous_forward + float(forward_epsilon))
    eligible = target_valid[:, None] & before & advancing
    best = torch.where(eligible, torch.maximum(max_progress, progress), max_progress)
    return {
        "best": best,
        "lift_ready": target_valid[:, None] & (best >= float(onset_progress)),
        "previous_forward": wheel_forward,
        "previous_valid": target_valid[:, None] & before,
    }


def v132_positive_plateau_evidence(
    points_yaw: torch.Tensor,
    last_face_xy_yaw: torch.Tensor,
    top_height: torch.Tensor,
    *,
    probe_start: float,
    probe_length: float,
    corridor_half_width: float,
    height_tolerance: float,
    min_points: int,
    min_forward_span: float,
) -> dict[str, torch.Tensor]:
    """Detect an actual same-height plane beyond the last face.

    Absence of a new riser is deliberately insufficient.  A candidate needs a
    finite, laterally useful patch near ``top_height``, enough longitudinal
    span, low relief, and no higher surface in the same forward corridor.
    """
    if points_yaw.ndim != 4 or points_yaw.shape[-1] != 3:
        raise ValueError("V13.2 plateau points must have shape [N, H, W, 3].")
    n = points_yaw.shape[0]
    if last_face_xy_yaw.shape != (n, 2) or top_height.shape != (n,):
        raise ValueError("V13.2 plateau face/height tensors have incompatible shapes.")
    if min_points < 1 or min_forward_span <= 0.0 or probe_length <= probe_start:
        raise ValueError("V13.2 plateau probe geometry is invalid.")

    x, y, z = points_yaw.unbind(dim=-1)
    finite = torch.isfinite(z)
    rel_x = x - last_face_xy_yaw[:, None, None, 0]
    rel_y = y - last_face_xy_yaw[:, None, None, 1]
    corridor = (
        finite
        & (rel_x >= float(probe_start))
        & (rel_x <= float(probe_length))
        & (rel_y.abs() <= float(corridor_half_width))
    )
    near = corridor & ((z - top_height[:, None, None]).abs() <= float(height_tolerance))
    count = near.reshape(n, -1).sum(dim=1)
    inf = torch.full_like(x, float("inf"))
    ninf = torch.full_like(x, float("-inf"))
    x_min = torch.where(near, rel_x, inf).amin(dim=(1, 2))
    x_max = torch.where(near, rel_x, ninf).amax(dim=(1, 2))
    z_min = torch.where(near, z, inf).amin(dim=(1, 2))
    z_max = torch.where(near, z, ninf).amax(dim=(1, 2))
    enough = count >= int(min_points)
    span = torch.where(enough, x_max - x_min, torch.zeros_like(x_min))
    relief = torch.where(enough, z_max - z_min, torch.full_like(z_min, float("inf")))
    higher = (corridor & (z > top_height[:, None, None] + float(height_tolerance))).any(dim=(1, 2))
    positive = enough & (span >= float(min_forward_span)) & (
        relief <= 2.0 * float(height_tolerance)
    ) & ~higher
    return {
        "positive": positive,
        "point_count": count,
        "forward_span": span,
        "relief": relief,
        "higher_surface": higher,
    }


def v132_top_follow_potential(
    wheel_bottom_z: torch.Tensor,
    forward_distance: torch.Tensor,
    support: torch.Tensor,
    top_height: torch.Tensor,
    step_height: torch.Tensor,
    *,
    forward_target: float,
) -> torch.Tensor:
    """Bounded follower progress over the final face and onto the top plane."""
    if wheel_bottom_z.shape != forward_distance.shape or wheel_bottom_z.shape != support.shape:
        raise ValueError("V13.2 top-follow inputs must be matching [num_envs, 2] tensors.")
    if top_height.shape != wheel_bottom_z.shape[:1] or step_height.shape != top_height.shape:
        raise ValueError("V13.2 top-follow height tensors must have shape [num_envs].")
    if float(forward_target) <= 0.0:
        raise ValueError("V13.2 top-follow forward target must be positive.")
    lower = top_height - step_height.clamp_min(1.0e-6)
    height = ((wheel_bottom_z - lower[:, None]) / step_height[:, None].clamp_min(1.0e-6)).clamp(0.0, 1.0)
    forward = (forward_distance / float(forward_target)).clamp(0.0, 1.0)
    return (0.40 * height + 0.30 * forward + 0.30 * support.float()).clamp(0.0, 1.0)


def v132_violation_buckets(
    wrong_completion: torch.Tensor,
    simultaneous: torch.Tensor,
    skip: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Settle each violation into one, and only one, cost bucket."""
    if wrong_completion.shape != simultaneous.shape or skip.shape != simultaneous.shape:
        raise ValueError("V13.2 violation tensors must have matching shapes.")
    skip_only = skip
    simultaneous_only = simultaneous & ~skip_only
    wrong_only = wrong_completion & ~skip_only & ~simultaneous_only
    return {"wrong": wrong_only, "simultaneous": simultaneous_only, "skip": skip_only}


def _v132_initialize_state(env) -> None:
    if hasattr(env, "_v132_alternation"):
        return
    n, device = env.num_envs, env.device
    env._v132_alternation = {
        "last_episode_length": env.episode_length_buf.clone(),
        "phase": torch.full((n,), APPROACH_B1, device=device, dtype=torch.long),
        "seed_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "parity_seeded": torch.zeros(n, device=device, dtype=torch.bool),
        "expected_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "b1_world_xy": torch.zeros((n, 2), device=device),
        "b1_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "flight_forward_world_xy": torch.zeros((n, 2), device=device),
        "target_index": torch.zeros(n, device=device, dtype=torch.long),
        "last_completed_index": torch.zeros(n, device=device, dtype=torch.long),
        "active_riser_world_xy": torch.zeros((n, 2), device=device),
        "active_riser_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "best_potential": torch.zeros((n, 2), device=device),
        "expected_progress_delta": torch.zeros(n, device=device),
        "same_progress_delta": torch.zeros(n, device=device),
        "simultaneous_progress_delta": torch.zeros(n, device=device),
        "barrier_score": torch.zeros(n, device=device),
        "target_active": torch.zeros(n, device=device, dtype=torch.bool),
        "deploy_target_agreement": torch.zeros(n, device=device, dtype=torch.bool),
        "encounter_event": torch.zeros(n, device=device, dtype=torch.bool),
        "physical_event": torch.zeros(n, device=device, dtype=torch.bool),
        "ordered_event": torch.zeros(n, device=device, dtype=torch.bool),
        "b1_event": torch.zeros(n, device=device, dtype=torch.bool),
        "compliant_event": torch.zeros(n, device=device, dtype=torch.bool),
        "wrong_event": torch.zeros(n, device=device, dtype=torch.bool),
        "simultaneous_event": torch.zeros(n, device=device, dtype=torch.bool),
        "simultaneous_violation_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "skip_event": torch.zeros(n, device=device, dtype=torch.bool),
        "skip_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "overshoot": torch.zeros(n, device=device),
        "last_completed_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "last_completed_riser_world_xy": torch.zeros((n, 2), device=device),
        "last_completed_upper_z": torch.zeros(n, device=device),
        "last_step_height": torch.zeros(n, device=device),
        "last_completion_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "plateau_positive": torch.zeros(n, device=device, dtype=torch.bool),
        "plateau_confirm_steps": torch.zeros(n, device=device, dtype=torch.long),
        "top_follower_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "top_follow_best": torch.zeros(n, device=device),
        "top_follow_progress_delta": torch.zeros(n, device=device),
        "top_follower_support_steps": torch.zeros(n, device=device, dtype=torch.long),
        "top_follow_event": torch.zeros(n, device=device, dtype=torch.bool),
        "top_entry_event": torch.zeros(n, device=device, dtype=torch.bool),
        "post_top_lift": torch.zeros(n, device=device),
        "encounter_count": torch.zeros((), device=device, dtype=torch.long),
        "physical_completion_count": torch.zeros((), device=device, dtype=torch.long),
        "b1_completion_count": torch.zeros((), device=device, dtype=torch.long),
        "ordered_completion_count": torch.zeros((), device=device, dtype=torch.long),
        "wrong_count": torch.zeros((), device=device, dtype=torch.long),
        "simultaneous_count": torch.zeros((), device=device, dtype=torch.long),
        "skip_count": torch.zeros((), device=device, dtype=torch.long),
        "top_entry_count": torch.zeros((), device=device, dtype=torch.long),
        "top_success_count": torch.zeros((), device=device, dtype=torch.long),
        "plateau_positive_count": torch.zeros((), device=device, dtype=torch.long),
        "plateau_probe_count": torch.zeros((), device=device, dtype=torch.long),
        "style_reward": torch.zeros(n, device=device),
    }
    env._v132_alternation_last_step = -1


def _v132_reset_environments(state: dict[str, torch.Tensor], reset: torch.Tensor) -> None:
    minus_one = ("seed_side", "expected_side", "last_completed_side", "top_follower_side")
    false_fields = (
        "parity_seeded", "b1_valid", "active_riser_valid", "skip_latched",
        "simultaneous_violation_latched",
        "last_completion_valid", "plateau_positive",
    )
    zero_fields = (
        "b1_world_xy", "flight_forward_world_xy", "target_index", "last_completed_index",
        "active_riser_world_xy", "best_potential", "expected_progress_delta",
        "same_progress_delta", "simultaneous_progress_delta", "barrier_score", "overshoot",
        "last_completed_riser_world_xy", "last_completed_upper_z", "last_step_height",
        "plateau_confirm_steps", "top_follow_best", "top_follow_progress_delta",
        "top_follower_support_steps", "post_top_lift", "style_reward",
    )
    for name in minus_one:
        state[name][reset] = -1
    for name in false_fields:
        state[name][reset] = False
    for name in zero_fields:
        state[name][reset] = 0
    state["phase"][reset] = APPROACH_B1


def _v132_front_geometry(env, state):
    asset = env.scene["robot"]
    wheel_world = asset.data.body_pos_w[:, env._v11_front_body_ids]
    forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids]
    forward = state["flight_forward_world_xy"]
    delta = wheel_world[..., :2] - state["last_completed_riser_world_xy"][:, None, :]
    forward_distance = (delta * forward[:, None, :]).sum(dim=2)
    wheel_bottom = wheel_world[..., 2] - float(env.cfg.v11_perception.wheel_radius)
    force_norm = forces.norm(dim=-1)
    support = (
        (forward_distance >= float(env.cfg.v11_perception.wheel_radius) + float(env.cfg.v11_perception.front_margin))
        & ((wheel_bottom - state["last_completed_upper_z"][:, None]).abs()
           <= float(env.cfg.v11_perception.upper_support_tolerance))
        & (force_norm >= float(env.cfg.v11_perception.contact_force_threshold))
        & (forces[..., 2] > float(env.cfg.v11_perception.contact_force_threshold))
    )
    return wheel_world, wheel_bottom, forward_distance, support


def v132_alternation_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Update the recoverable B1-parity/top-platform ledger once per step."""
    from .perceptive_spatial import _elevation_points_yaw
    from .v11_continuous import v11_state
    from .v13_alternation import (
        v13_front_riser_potential,
        v13_historical_best_progress,
        v13_split_alternation_progress,
    )

    if sensor_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        sensor_cfg = SceneEntityCfg("v10_pgtt_scanner")
    _v132_initialize_state(env)
    state = env._v132_alternation
    step = int(getattr(env, "common_step_counter", 0))
    same_step = env._v132_alternation_last_step == step
    reset = v132_episode_reset_mask(
        env.episode_length_buf, state["last_episode_length"], same_step=same_step,
    )
    if reset.any():
        _v132_reset_environments(state, reset)
        state["last_episode_length"] = env.episode_length_buf.clone()
    if same_step:
        return env._v132_alternation

    state["last_episode_length"] = env.episode_length_buf.clone()
    for name in (
        "expected_progress_delta", "same_progress_delta", "simultaneous_progress_delta",
        "top_follow_progress_delta", "post_top_lift", "overshoot", "style_reward",
    ):
        state[name].zero_()
    for name in (
        "target_active", "deploy_target_agreement", "encounter_event", "physical_event", "ordered_event",
        "b1_event", "compliant_event", "wrong_event", "simultaneous_event",
        "skip_event", "top_follow_event", "top_entry_event",
    ):
        state[name].zero_()

    base = v11_state(env, sensor_cfg=sensor_cfg)
    deploy = v132_deployable_phase_state(env, sensor_cfg=sensor_cfg)
    landing, risers = base["landing"], base["risers"]
    cfg = env.cfg.v132_style
    reference = landing["reference_riser_world_xy"]
    reference_active = landing["target_active"]
    asset = env.scene["robot"]
    root = asset.data.root_pos_w
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)

    # A later, genuinely distinct staircase may re-arm the symmetric B1 phase.
    distance_from_last = torch.linalg.norm(reference - state["last_completed_riser_world_xy"], dim=1)
    rearm = (
        (state["phase"] == PLATEAU_ROLL) & reference_active
        & (distance_from_last >= float(cfg.new_flight_separation))
    )
    if rearm.any():
        _v132_reset_environments(state, rearm)

    raw_positive = base["valid"] & (risers["riser_height"] > 0.0)
    local_xy = torch.stack((risers["riser_x"], risers["riser_y"]), dim=2)
    c, s = torch.cos(yaw), torch.sin(yaw)
    raw_world = torch.stack((
        root[:, None, 0] + c[:, None] * local_xy[..., 0] - s[:, None] * local_xy[..., 1],
        root[:, None, 1] + s[:, None] * local_xy[..., 0] + c[:, None] * local_xy[..., 1],
    ), dim=2)
    raw_axis = raw_world[:, 1] - raw_world[:, 0]
    raw_axis_valid = raw_positive.all(dim=1) & (raw_axis.norm(dim=1) > float(cfg.target_identity_tolerance))
    raw_axis = raw_axis / raw_axis.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    flight_axis = torch.where(raw_axis_valid[:, None], raw_axis, heading)
    lock_b1 = (state["phase"] == APPROACH_B1) & ~state["b1_valid"] & reference_active
    state["b1_world_xy"] = torch.where(lock_b1[:, None], reference, state["b1_world_xy"])
    state["flight_forward_world_xy"] = torch.where(
        lock_b1[:, None], flight_axis, state["flight_forward_world_xy"],
    )
    state["b1_valid"] |= lock_b1
    state["encounter_event"] = lock_b1
    state["encounter_count"] += lock_b1.sum()

    target_valid = state["b1_valid"] & reference_active
    state["target_active"] = target_valid
    target_index = v132_target_index(
        reference, state["b1_world_xy"], state["flight_forward_world_xy"], target_valid,
        step_width=float(cfg.step_width),
    )
    state["target_index"] = target_index
    target_agreement = (
        target_valid & deploy["settlement_target_valid"]
        & (target_index == deploy["settlement_target_index"])
    )
    state["deploy_target_agreement"] = target_agreement
    target_distance = torch.linalg.norm(reference - state["active_riser_world_xy"], dim=1)
    new_target = target_valid & (
        ~state["active_riser_valid"] | (target_distance >= float(cfg.target_identity_tolerance))
    )
    state["active_riser_world_xy"] = torch.where(
        new_target[:, None], reference, state["active_riser_world_xy"],
    )
    state["active_riser_valid"] |= new_target
    state["best_potential"][new_target] = 0.0
    state["skip_latched"][new_target] = False
    state["simultaneous_violation_latched"][new_target] = False

    # The contact-free actor ledger is authoritative for parity.  Reward-side
    # contact geometry may settle credit, but may not silently choose a leader
    # different from the one encoded in the actor's extra five dimensions.
    state["seed_side"] = deploy["seed_side"]
    state["parity_seeded"] = deploy["parity_seeded"]
    simultaneous_lift = deploy["simultaneous_lift_event"]

    physical_event = landing["landing_event"]
    state["physical_event"] = physical_event
    completed_side = landing["responsible"]
    simultaneous_completion = physical_event & landing["candidate_ready"].all(dim=1)
    single_completion = physical_event & ~simultaneous_completion
    # The actor's deployable ledger owns the instruction. Reward-side contact
    # may verify progress, but never invent a different expected leg.
    expected = deploy["settlement_expected_side"]
    state["expected_side"] = expected

    raw_potential = v13_front_riser_potential(
        landing["max_clearance_progress"], landing["crossing_clearance_ok"],
        landing["wheel_yaw"][..., 0], landing["target_riser_x"], landing["support"],
        crossing_span=float(env.cfg.v11_perception.wheel_radius) + float(env.cfg.v11_perception.front_margin),
    )
    progress_active = (
        target_agreement & state["parity_seeded"] & (expected >= 0)
        & ((deploy["settlement_phase"] == APPROACH_B1)
           | (deploy["settlement_phase"] == ALTERNATING_RISERS))
    )
    history = v13_historical_best_progress(state["best_potential"], raw_potential, progress_active)
    split = v13_split_alternation_progress(history["delta"], expected, progress_active)
    state["best_potential"] = history["best"]
    expected_idx = expected.clamp(0, 1)
    expected_potential = raw_potential.gather(1, expected_idx[:, None]).squeeze(1)
    other_potential = raw_potential.gather(1, (1 - expected_idx)[:, None]).squeeze(1)
    lead = expected_potential - other_potential
    barrier = v132_barrier_band_score(
        lead, float(cfg.lead_margin_lower), float(cfg.lead_margin_upper), float(cfg.barrier_delta),
    )
    state["barrier_score"] = barrier * progress_active.float()
    state["expected_progress_delta"] = split["opposite"] * barrier
    state["same_progress_delta"] = split["same"]
    state["simultaneous_progress_delta"] = split["simultaneous"]

    next_required = state["last_completed_index"] + 1
    index_jump = (
        new_target & target_agreement & (target_index > next_required)
        & state["parity_seeded"]
    )
    forward_overrun = (
        landing["wheel_yaw"][..., 0] - landing["target_riser_x"][:, None]
        - float(cfg.overshoot_free_distance)
    ).clamp_min(0.0) / float(cfg.overshoot_scale)
    state["overshoot"] = forward_overrun.amax(dim=1).clamp(max=1.0) * progress_active.float()
    overshot = progress_active & (state["overshoot"] >= 1.0) & ~physical_event
    skip_raw = index_jump | overshot
    skip_event = skip_raw & ~state["skip_latched"]
    state["skip_latched"] |= skip_raw

    ordered = (
        single_completion & target_agreement & state["parity_seeded"]
        & (completed_side == expected)
        & (target_index == next_required)
    )
    b1_event = ordered & (target_index == 1)
    compliant = ordered & (target_index > 1)
    simultaneous_raw = simultaneous_lift | (simultaneous_completion & target_agreement)
    simultaneous_event = simultaneous_raw & ~state["simultaneous_violation_latched"]
    state["simultaneous_violation_latched"] |= simultaneous_raw
    wrong_completion = (
        physical_event & target_agreement & ~ordered
        & ~state["simultaneous_violation_latched"]
    )
    buckets = v132_violation_buckets(
        wrong_completion, simultaneous_event, skip_event,
    )
    state["ordered_event"] = ordered
    state["b1_event"] = b1_event
    state["compliant_event"] = compliant
    state["wrong_event"] = buckets["wrong"]
    state["simultaneous_event"] = buckets["simultaneous"]
    state["skip_event"] = buckets["skip"]

    distinct_completion = physical_event & target_valid & (
        ~state["last_completion_valid"]
        | (torch.linalg.norm(reference - state["last_completed_riser_world_xy"], dim=1)
           >= float(cfg.target_identity_tolerance))
    )
    # V11 clears target_slot on the completion step but deliberately retains
    # target_riser_x/target_upper_z for reward settlement. Recover the physical
    # slot by geometry instead of accidentally defaulting every event to slot 0.
    slot_error = (risers["riser_x"] - landing["target_riser_x"][:, None]).abs()
    slot_error = torch.where(base["valid"], slot_error, torch.full_like(slot_error, float("inf")))
    chosen = slot_error.argmin(dim=1)
    step_height = risers["riser_height"].gather(1, chosen[:, None]).squeeze(1).abs()
    state["last_completed_index"] = torch.where(
        distinct_completion, torch.maximum(state["last_completed_index"], target_index),
        state["last_completed_index"],
    )
    state["last_completed_side"] = torch.where(
        distinct_completion & ~simultaneous_completion, completed_side, state["last_completed_side"],
    )
    state["last_completed_riser_world_xy"] = torch.where(
        distinct_completion[:, None], reference, state["last_completed_riser_world_xy"],
    )
    state["last_completed_upper_z"] = torch.where(
        distinct_completion, landing["target_upper_z"], state["last_completed_upper_z"],
    )
    state["last_step_height"] = torch.where(
        distinct_completion, step_height, state["last_step_height"],
    )
    state["last_completion_valid"] |= distinct_completion
    state["phase"] = torch.where(
        state["parity_seeded"] & (state["phase"] == APPROACH_B1),
        torch.full_like(state["phase"], ALTERNATING_RISERS), state["phase"],
    )
    state["best_potential"][physical_event | skip_event] = 0.0

    # Positive top-plane evidence: a real same-height patch, not merely an
    # absent next riser tuple.  This handles one-step and N-step stairs alike.
    points = _elevation_points_yaw(env, sensor_cfg, asset)
    last_delta = state["last_completed_riser_world_xy"] - root[:, :2]
    last_face_yaw = torch.stack((
        c * last_delta[:, 0] + s * last_delta[:, 1],
        -s * last_delta[:, 0] + c * last_delta[:, 1],
    ), dim=1)
    plateau = v132_positive_plateau_evidence(
        points, last_face_yaw, state["last_completed_upper_z"],
        probe_start=float(cfg.plateau_probe_start),
        probe_length=float(cfg.plateau_probe_length),
        corridor_half_width=float(cfg.plateau_corridor_half_width),
        height_tolerance=float(cfg.plateau_height_tolerance),
        min_points=int(cfg.plateau_min_points),
        min_forward_span=float(cfg.plateau_min_forward_span),
    )
    wheel_world, wheel_bottom, forward_distance, top_support = _v132_front_geometry(env, state)
    last_side = state["last_completed_side"].clamp(0, 1)
    known_leader = state["last_completed_side"] >= 0
    leader_supported = top_support.gather(1, last_side[:, None]).squeeze(1)
    probe_active = (
        ((state["phase"] == ALTERNATING_RISERS) | (state["phase"] == APPROACH_B1))
        & state["last_completion_valid"]
    )
    state["plateau_positive"] = plateau["positive"] & probe_active
    state["plateau_probe_count"] += probe_active.sum()
    state["plateau_positive_count"] += state["plateau_positive"].sum()
    top_candidate = (
        probe_active & state["plateau_positive"]
        & torch.where(known_leader, leader_supported, top_support.all(dim=1))
    )
    state["plateau_confirm_steps"] = torch.where(
        top_candidate, state["plateau_confirm_steps"] + 1,
        torch.zeros_like(state["plateau_confirm_steps"]),
    )
    enter_top = top_candidate & (
        state["plateau_confirm_steps"] >= int(cfg.plateau_confirmation_frames)
    )
    deploy_follow = deploy["phase"] == TOP_PLATFORM_FOLLOW
    follower_agreement = deploy["top_follower_side"] == (1 - state["last_completed_side"])
    direct_plateau = enter_top & ~known_leader & (deploy["phase"] == PLATEAU_ROLL)
    enter_follow = enter_top & known_leader & deploy_follow & follower_agreement
    state["phase"] = torch.where(
        enter_follow, torch.full_like(state["phase"], TOP_PLATFORM_FOLLOW),
        torch.where(
            direct_plateau, torch.full_like(state["phase"], PLATEAU_ROLL), state["phase"],
        ),
    )
    state["top_follower_side"] = torch.where(
        enter_follow, deploy["top_follower_side"], state["top_follower_side"],
    )
    state["top_follow_best"] = torch.where(enter_top, torch.zeros_like(state["top_follow_best"]), state["top_follow_best"])
    state["top_entry_count"] += enter_follow.sum()
    state["top_entry_event"] = enter_follow

    top_active = state["phase"] == TOP_PLATFORM_FOLLOW
    follower = state["top_follower_side"].clamp(0, 1)
    all_top_potential = v132_top_follow_potential(
        wheel_bottom, forward_distance, top_support,
        state["last_completed_upper_z"], state["last_step_height"],
        forward_target=float(env.cfg.v11_perception.wheel_radius) + float(env.cfg.v11_perception.front_margin),
    )
    follower_potential = all_top_potential.gather(1, follower[:, None]).squeeze(1)
    follower_potential = torch.where(top_active, follower_potential, torch.zeros_like(follower_potential))
    new_best = torch.where(top_active, torch.maximum(state["top_follow_best"], follower_potential), state["top_follow_best"])
    state["top_follow_progress_delta"] = (new_best - state["top_follow_best"]).clamp_min(0.0)
    state["top_follow_best"] = new_best
    follower_supported = top_support.gather(1, follower[:, None]).squeeze(1)
    state["top_follower_support_steps"] = torch.where(
        top_active & follower_supported,
        state["top_follower_support_steps"] + 1,
        torch.zeros_like(state["top_follower_support_steps"]),
    )
    top_follow_event = top_active & (
        state["top_follower_support_steps"] >= int(cfg.top_follow_support_confirmation_frames)
    )
    state["top_follow_event"] = top_follow_event
    state["phase"] = torch.where(
        top_follow_event, torch.full_like(state["phase"], PLATEAU_ROLL), state["phase"],
    )
    state["top_success_count"] += top_follow_event.sum()

    plateau_roll = state["phase"] == PLATEAU_ROLL
    lift_excess = (
        (wheel_bottom - state["last_completed_upper_z"][:, None] - float(cfg.plateau_lift_free_height))
        / float(cfg.plateau_lift_scale)
    ).clamp(0.0, 1.0).mean(dim=1)
    state["post_top_lift"] = lift_excess * plateau_roll.float()

    state["physical_completion_count"] += physical_event.sum()
    state["b1_completion_count"] += b1_event.sum()
    state["ordered_completion_count"] += ordered.sum()
    state["wrong_count"] += state["wrong_event"].sum()
    state["simultaneous_count"] += state["simultaneous_event"].sum()
    state["skip_count"] += state["skip_event"].sum()
    env._v132_alternation_last_step = step
    return state


def _v132_initialize_deploy_state(env) -> None:
    """Allocate the contact-free state that is part of the actor ABI.

    Keep this ledger separate from ``_v132_alternation``: the latter is a
    reward settlement ledger and intentionally uses simulated contact forces.
    Every input used here has a hardware counterpart (PGTT, odometry/IMU and
    joint-state forward kinematics).
    """
    if hasattr(env, "_v132_deploy"):
        return
    n, device = env.num_envs, env.device
    env._v132_deploy = {
        "last_episode_length": env.episode_length_buf.clone(),
        "phase": torch.full((n,), APPROACH_B1, device=device, dtype=torch.long),
        "seed_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "parity_seeded": torch.zeros(n, device=device, dtype=torch.bool),
        "expected_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "b1_world_xy": torch.zeros((n, 2), device=device),
        "b1_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "flight_forward_world_xy": torch.zeros((n, 2), device=device),
        "target_world_xy": torch.zeros((n, 2), device=device),
        "target_upper_z": torch.zeros(n, device=device),
        "target_step_height": torch.zeros(n, device=device),
        "target_index": torch.zeros(n, device=device, dtype=torch.long),
        "target_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "settlement_target_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "settlement_target_index": torch.zeros(n, device=device, dtype=torch.long),
        "settlement_expected_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "settlement_phase": torch.full((n,), APPROACH_B1, device=device, dtype=torch.long),
        "max_clearance_progress": torch.zeros((n, 2), device=device),
        "previous_target_forward": torch.zeros((n, 2), device=device),
        "previous_target_forward_valid": torch.zeros((n, 2), device=device, dtype=torch.bool),
        "unique_lift_event": torch.zeros(n, device=device, dtype=torch.bool),
        "simultaneous_lift_event": torch.zeros(n, device=device, dtype=torch.bool),
        "simultaneous_lift_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "support_steps": torch.zeros((n, 2), device=device, dtype=torch.long),
        "support_latched": torch.zeros((n, 2), device=device, dtype=torch.bool),
        "last_completed_index": torch.zeros(n, device=device, dtype=torch.long),
        "last_completed_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "last_face_world_xy": torch.zeros((n, 2), device=device),
        "last_upper_z": torch.zeros(n, device=device),
        "last_step_height": torch.zeros(n, device=device),
        "last_completion_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "plateau_confirm_steps": torch.zeros(n, device=device, dtype=torch.long),
        "top_follower_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "top_follower_support_steps": torch.zeros(n, device=device, dtype=torch.long),
    }
    env._v132_deploy_last_step = -1


def _v132_reset_deploy_state(state: dict[str, torch.Tensor], reset: torch.Tensor) -> None:
    for name in (
        "seed_side", "expected_side", "last_completed_side", "top_follower_side",
        "settlement_expected_side",
    ):
        state[name][reset] = -1
    for name in (
        "parity_seeded", "b1_valid", "target_valid", "settlement_target_valid",
        "last_completion_valid",
        "previous_target_forward_valid", "unique_lift_event", "simultaneous_lift_event",
        "simultaneous_lift_latched", "support_latched",
    ):
        state[name][reset] = False
    for name in (
        "b1_world_xy", "flight_forward_world_xy", "target_world_xy", "target_upper_z",
        "target_step_height", "target_index", "settlement_target_index",
        "settlement_phase", "max_clearance_progress",
        "previous_target_forward", "support_steps", "last_completed_index",
        "last_face_world_xy", "last_upper_z", "last_step_height",
        "plateau_confirm_steps", "top_follower_support_steps",
    ):
        state[name][reset] = 0
    state["phase"][reset] = APPROACH_B1


def _v132_choose_deploy_target(
    positive: torch.Tensor,
    world_xy: torch.Tensor,
    upper_z: torch.Tensor,
    step_height: torch.Tensor,
    distance: torch.Tensor,
    b1_world_xy: torch.Tensor,
    forward_world_xy: torch.Tensor,
    required_index: torch.Tensor,
    b1_valid: torch.Tensor,
    step_width: float,
) -> dict[str, torch.Tensor]:
    """Choose the next physical face using only PGTT-derived geometry."""
    n, slots = positive.shape
    flat_b1 = b1_world_xy[:, None, :].expand(-1, slots, -1).reshape(-1, 2)
    flat_forward = forward_world_xy[:, None, :].expand(-1, slots, -1).reshape(-1, 2)
    indices = v132_target_index(
        world_xy.reshape(-1, 2), flat_b1, flat_forward,
        (positive & b1_valid[:, None]).reshape(-1), step_width=float(step_width),
    ).reshape(n, slots)
    eligible = positive & b1_valid[:, None] & (indices >= required_index[:, None])
    # Select the smallest physical index first, then the nearest detection at
    # that index.  Avoid a magic scalar rank whose ordering could change with
    # sensor range.
    missing_index = torch.full_like(indices, torch.iinfo(indices.dtype).max)
    eligible_index = torch.where(eligible, indices, missing_index)
    first_index = eligible_index.amin(dim=1)
    nearest_at_first = eligible & (indices == first_index[:, None])
    ranked_distance = torch.where(
        nearest_at_first, distance, torch.full_like(distance, float("inf")),
    )
    slot = ranked_distance.argmin(dim=1)
    valid = eligible.gather(1, slot[:, None]).squeeze(1)
    gather_xy = slot[:, None, None].expand(-1, 1, 2)
    return {
        "valid": valid,
        "world_xy": world_xy.gather(1, gather_xy).squeeze(1),
        "upper_z": upper_z.gather(1, slot[:, None]).squeeze(1),
        "step_height": step_height.gather(1, slot[:, None]).squeeze(1),
        "index": indices.gather(1, slot[:, None]).squeeze(1),
    }


def v132_deployable_phase_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Compute the actor's state machine without reward/contact information.

    Deploy-time input contract:

    * the existing PGTT ray-hit grid and its two-riser extractor;
    * base pose/yaw from odometry/IMU;
    * front-wheel centres obtained from joint-state forward kinematics.

    In particular this function never reads ``landing``, a reward event,
    terrain metadata, or a contact sensor.  Stable geometric support is a
    wheel-bottom/upper-plane test held for multiple control frames.
    """
    from .perceptive_spatial import _elevation_points_yaw
    from .v11_continuous import v11_state

    if sensor_cfg is None:
        from isaaclab.managers import SceneEntityCfg
        sensor_cfg = SceneEntityCfg("v10_pgtt_scanner")
    _v132_initialize_deploy_state(env)
    state = env._v132_deploy
    step = int(getattr(env, "common_step_counter", 0))
    same_step = env._v132_deploy_last_step == step
    reset = v132_episode_reset_mask(
        env.episode_length_buf, state["last_episode_length"], same_step=same_step,
    )
    if reset.any():
        _v132_reset_deploy_state(state, reset)
        state["last_episode_length"] = env.episode_length_buf.clone()
    if same_step:
        return env._v132_deploy

    state["last_episode_length"] = env.episode_length_buf.clone()
    state["unique_lift_event"].zero_()
    state["simultaneous_lift_event"].zero_()
    state["settlement_target_valid"].zero_()
    base = v11_state(env, sensor_cfg=sensor_cfg)
    risers = base["risers"]
    cfg = env.cfg.v132_style
    perception_cfg = env.cfg.v11_perception
    asset = env.scene["robot"]
    root = asset.data.root_pos_w
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    c, s = torch.cos(yaw), torch.sin(yaw)
    heading = torch.stack((c, s), dim=1)

    positive = base["valid"] & (risers["riser_height"] > 0.0)
    local_xy = torch.stack((risers["riser_x"], risers["riser_y"]), dim=2)
    world_xy = torch.stack((
        root[:, None, 0] + c[:, None] * local_xy[..., 0] - s[:, None] * local_xy[..., 1],
        root[:, None, 1] + s[:, None] * local_xy[..., 0] + c[:, None] * local_xy[..., 1],
    ), dim=2)
    step_height = risers["riser_height"].abs()
    upper_z = risers["riser_lower_world"] + risers["riser_height"]
    distance = risers["riser_distance"]

    # A later flight re-arms the symmetric B1 choice.  Distance is measured
    # between PGTT-anchored faces rather than supplied by terrain metadata.
    nearest_slot = torch.where(
        positive, distance, torch.full_like(distance, float("inf")),
    ).argmin(dim=1)
    nearest_xy = world_xy.gather(1, nearest_slot[:, None, None].expand(-1, 1, 2)).squeeze(1)
    nearest_valid = positive.gather(1, nearest_slot[:, None]).squeeze(1)
    rearm = (
        (state["phase"] == PLATEAU_ROLL) & nearest_valid
        & (torch.linalg.norm(nearest_xy - state["last_face_world_xy"], dim=1)
           >= float(cfg.new_flight_separation))
    )
    if rearm.any():
        _v132_reset_deploy_state(state, rearm)

    # B1 and the flight direction are captured from the first confirmed PGTT
    # face(s).  No absolute simulator stair index exists in this calculation.
    lock_b1 = (state["phase"] == APPROACH_B1) & ~state["b1_valid"] & nearest_valid
    raw_axis = world_xy[:, 1] - world_xy[:, 0]
    raw_axis_ok = positive.all(dim=1) & (
        raw_axis.norm(dim=1) > float(cfg.target_identity_tolerance)
    )
    raw_axis = raw_axis / raw_axis.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    initial_axis = torch.where(raw_axis_ok[:, None], raw_axis, heading)
    state["b1_world_xy"] = torch.where(lock_b1[:, None], nearest_xy, state["b1_world_xy"])
    state["flight_forward_world_xy"] = torch.where(
        lock_b1[:, None], initial_axis, state["flight_forward_world_xy"],
    )
    state["b1_valid"] |= lock_b1

    required = torch.where(
        state["last_completed_index"] > 0,
        state["last_completed_index"] + 1,
        torch.ones_like(state["last_completed_index"]),
    )
    candidate = _v132_choose_deploy_target(
        positive, world_xy, upper_z, step_height, distance,
        state["b1_world_xy"], state["flight_forward_world_xy"], required,
        state["b1_valid"], float(cfg.step_width),
    )
    acquire = ~state["target_valid"] & candidate["valid"] & (
        (state["phase"] == APPROACH_B1) | (state["phase"] == ALTERNATING_RISERS)
    )
    for name, candidate_name in (
        ("target_world_xy", "world_xy"), ("target_upper_z", "upper_z"),
        ("target_step_height", "step_height"), ("target_index", "index"),
    ):
        value = candidate[candidate_name]
        state[name] = torch.where(acquire[:, None], value, state[name]) if value.ndim == 2 else torch.where(acquire, value, state[name])
    state["target_valid"] |= acquire
    state["max_clearance_progress"][acquire] = 0.0
    state["previous_target_forward_valid"][acquire] = False
    state["simultaneous_lift_latched"][acquire] = False
    state["support_steps"][acquire] = 0
    state["support_latched"][acquire] = False

    wheel_world = asset.data.body_pos_w[:, env._v11_front_body_ids]
    wheel_bottom = wheel_world[..., 2] - float(perception_cfg.wheel_radius)
    target_delta = wheel_world[..., :2] - state["target_world_xy"][:, None, :]
    target_forward = (
        target_delta * state["flight_forward_world_xy"][:, None, :]
    ).sum(dim=2)
    target_geom = state["target_valid"][:, None] & (
        target_forward >= float(perception_cfg.wheel_radius) + float(perception_cfg.front_margin)
    ) & (
        (wheel_bottom - state["target_upper_z"][:, None]).abs()
        <= float(perception_cfg.upper_support_tolerance)
    )
    state["support_steps"] = torch.where(
        target_geom, state["support_steps"] + 1, torch.zeros_like(state["support_steps"]),
    )
    confirmed = state["support_steps"] >= int(perception_cfg.front_support_confirmation_frames)
    new_support = confirmed & ~state["support_latched"]
    state["support_latched"] |= confirmed

    # Mirror V11's deployable clearance escrow: only forward-moving height
    # progress before the face may seed parity.  Using the same normalized
    # onset as the reward ledger prevents the 5-D cue and style target from
    # choosing opposite leaders.
    clearance = v132_deploy_clearance_onset(
        state["max_clearance_progress"], state["previous_target_forward"],
        state["previous_target_forward_valid"], target_forward, wheel_bottom,
        state["target_upper_z"] - state["target_step_height"],
        state["target_step_height"], state["target_valid"],
        clearance_margin=float(perception_cfg.clearance_margin),
        onset_progress=float(cfg.b1_lift_onset_progress),
    )
    state["max_clearance_progress"] = clearance["best"]
    # Normally parity is seeded by B1.  After a penalized simultaneous B1
    # completion, permit the first later unique onset to recover the same
    # absolute B1/B2/... parity instead of leaving the controller unseeded for
    # the rest of the episode.
    lift_ready = state["target_valid"][:, None] & clearance["lift_ready"]
    latch = v132_latch_first_lift(state["seed_side"], state["parity_seeded"], lift_ready)
    inferred_seed = torch.where(
        state["target_index"].remainder(2) == 1,
        latch["seed_side"], 1 - latch["seed_side"],
    )
    state["seed_side"] = torch.where(latch["unique_event"], inferred_seed, state["seed_side"])
    state["parity_seeded"] = latch["parity_seeded"]
    state["unique_lift_event"] = latch["unique_event"]
    state["simultaneous_lift_event"] = (
        latch["simultaneous_event"] & ~state["simultaneous_lift_latched"]
    )
    state["simultaneous_lift_latched"] |= latch["simultaneous_event"]
    state["previous_target_forward"] = clearance["previous_forward"]
    state["previous_target_forward_valid"] = clearance["previous_valid"]

    support_count = new_support.long().sum(dim=1)
    any_completion = state["target_valid"] & (support_count > 0)
    unique_completion = state["target_valid"] & (support_count == 1)
    completed_side = new_support.long().argmax(dim=1)
    fallback_seed = unique_completion & (state["target_index"] == 1) & ~state["parity_seeded"]
    state["seed_side"] = torch.where(fallback_seed, completed_side, state["seed_side"])
    state["parity_seeded"] |= fallback_seed
    settlement_expected = v132_expected_side(
        state["seed_side"], state["target_index"], state["parity_seeded"],
    )
    # Preserve the instruction/target that governed the just-finished action.
    # The persistent target may be cleared below on geometric completion, but
    # reward settlement later in this same control step must still see it.
    state["settlement_target_valid"] = state["target_valid"].clone()
    state["settlement_target_index"] = state["target_index"].clone()
    state["settlement_expected_side"] = settlement_expected
    state["settlement_phase"] = state["phase"].clone()
    # A simultaneous support event is a style violation, but it must still
    # advance the physical target. Otherwise both support latches stay true and
    # the deploy state becomes permanently stuck on B1.
    complete = any_completion
    state["last_completed_index"] = torch.where(
        complete, state["target_index"], state["last_completed_index"],
    )
    state["last_completed_side"] = torch.where(
        unique_completion,
        completed_side,
        torch.where(
            complete, torch.full_like(state["last_completed_side"], -1),
            state["last_completed_side"],
        ),
    )
    state["last_face_world_xy"] = torch.where(
        complete[:, None], state["target_world_xy"], state["last_face_world_xy"],
    )
    state["last_upper_z"] = torch.where(complete, state["target_upper_z"], state["last_upper_z"])
    state["last_step_height"] = torch.where(
        complete, state["target_step_height"], state["last_step_height"],
    )
    state["last_completion_valid"] |= complete
    state["target_valid"] &= ~complete
    state["phase"] = torch.where(
        complete & state["parity_seeded"] & (state["phase"] == APPROACH_B1),
        torch.full_like(state["phase"], ALTERNATING_RISERS), state["phase"],
    )
    state["expected_side"] = v132_expected_side(
        state["seed_side"],
        torch.where(state["target_valid"], state["target_index"], state["last_completed_index"] + 1),
        state["parity_seeded"],
    )

    # Last-riser recognition is positive PGTT plane evidence plus contact-free
    # geometric persistence of the leading wheel on the same upper plane.
    points = _elevation_points_yaw(env, sensor_cfg, asset)
    last_delta = state["last_face_world_xy"] - root[:, :2]
    last_face_yaw = torch.stack((
        c * last_delta[:, 0] + s * last_delta[:, 1],
        -s * last_delta[:, 0] + c * last_delta[:, 1],
    ), dim=1)
    plateau = v132_positive_plateau_evidence(
        points, last_face_yaw, state["last_upper_z"],
        probe_start=float(cfg.plateau_probe_start), probe_length=float(cfg.plateau_probe_length),
        corridor_half_width=float(cfg.plateau_corridor_half_width),
        height_tolerance=float(cfg.plateau_height_tolerance),
        min_points=int(cfg.plateau_min_points), min_forward_span=float(cfg.plateau_min_forward_span),
    )
    last_delta_wheel = wheel_world[..., :2] - state["last_face_world_xy"][:, None, :]
    last_forward = (
        last_delta_wheel * state["flight_forward_world_xy"][:, None, :]
    ).sum(dim=2)
    last_geom = (
        last_forward >= float(perception_cfg.wheel_radius) + float(perception_cfg.front_margin)
    ) & (
        (wheel_bottom - state["last_upper_z"][:, None]).abs()
        <= float(perception_cfg.upper_support_tolerance)
    )
    known_leader = state["last_completed_side"] >= 0
    leader = state["last_completed_side"].clamp(0, 1)
    leader_geom = last_geom.gather(1, leader[:, None]).squeeze(1)
    top_candidate = (
        ((state["phase"] == ALTERNATING_RISERS) | (state["phase"] == APPROACH_B1))
        & state["last_completion_valid"] & plateau["positive"]
        & torch.where(known_leader, leader_geom, last_geom.all(dim=1))
    )
    state["plateau_confirm_steps"] = torch.where(
        top_candidate, state["plateau_confirm_steps"] + 1,
        torch.zeros_like(state["plateau_confirm_steps"]),
    )
    enter_top = top_candidate & (
        state["plateau_confirm_steps"] >= int(cfg.plateau_confirmation_frames)
    )
    direct_plateau = enter_top & ~known_leader
    enter_follow = enter_top & known_leader
    state["phase"] = torch.where(
        enter_follow, torch.full_like(state["phase"], TOP_PLATFORM_FOLLOW),
        torch.where(direct_plateau, torch.full_like(state["phase"], PLATEAU_ROLL), state["phase"]),
    )
    state["top_follower_side"] = torch.where(
        enter_follow, 1 - state["last_completed_side"], state["top_follower_side"],
    )
    state["expected_side"] = torch.where(
        state["phase"] == TOP_PLATFORM_FOLLOW,
        state["top_follower_side"], state["expected_side"],
    )
    follower = state["top_follower_side"].clamp(0, 1)
    follower_geom = last_geom.gather(1, follower[:, None]).squeeze(1)
    top_active = state["phase"] == TOP_PLATFORM_FOLLOW
    state["top_follower_support_steps"] = torch.where(
        top_active & follower_geom, state["top_follower_support_steps"] + 1,
        torch.zeros_like(state["top_follower_support_steps"]),
    )
    plateau_roll = top_active & (
        state["top_follower_support_steps"] >= int(cfg.top_follow_support_confirmation_frames)
    )
    state["phase"] = torch.where(
        plateau_roll, torch.full_like(state["phase"], PLATEAU_ROLL), state["phase"],
    )
    state["expected_side"] = torch.where(
        state["phase"] == PLATEAU_ROLL,
        torch.full_like(state["expected_side"], -1), state["expected_side"],
    )
    env._v132_deploy_last_step = step
    return state


def v132_event_phase_observation(env, sensor_cfg=None) -> torch.Tensor:
    """Contact-free deployable 5-D state: expected side plus four phases."""
    state = v132_deployable_phase_state(env, sensor_cfg=sensor_cfg)
    expected = state["expected_side"]
    signed_side = torch.where(
        expected == 0, torch.full_like(expected, -1),
        torch.where(expected == 1, torch.ones_like(expected), torch.zeros_like(expected)),
    ).float()
    phases = torch.nn.functional.one_hot(state["phase"].clamp(0, 3), num_classes=4).float()
    return torch.cat((signed_side[:, None], phases), dim=1)


def v132_style_objective(env, sensor_cfg=None) -> torch.Tensor:
    """Return the complete style reward rate consumed by the style critic."""
    from .rewards import track_lin_vel_xy_exp

    state = v132_alternation_state(env, sensor_cfg=sensor_cfg)
    cfg = env.cfg.v132_style
    quality = track_lin_vel_xy_exp(
        env, std=float(cfg.velocity_tracking_std), command_name=str(cfg.command_name),
    ).clamp(0.0, 1.0)
    dt = float(env.step_dt)
    progress_gain = state["expected_progress_delta"] * (
        float(cfg.expected_progress_weight)
        + float(cfg.barrier_progress_weight) * state["barrier_score"]
        + float(cfg.velocity_coupling_weight) * quality
    )
    rate = (
        progress_gain
        - float(cfg.same_progress_cost) * state["same_progress_delta"]
        - float(cfg.simultaneous_progress_cost) * state["simultaneous_progress_delta"]
        + float(cfg.ordered_completion_weight) * state["ordered_event"].float()
        - float(cfg.wrong_completion_cost) * state["wrong_event"].float()
        - float(cfg.simultaneous_event_cost) * state["simultaneous_event"].float()
        - float(cfg.skip_event_cost) * state["skip_event"].float()
        + float(cfg.top_follow_progress_weight) * state["top_follow_progress_delta"]
        + float(cfg.top_follow_completion_weight) * state["top_follow_event"].float()
    ) / dt
    rate -= float(cfg.overshoot_cost) * state["overshoot"].square()
    rate -= float(cfg.post_top_lift_cost) * state["post_top_lift"].square()
    state["style_reward"] = rate
    return rate


def v132_alternation_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose reachability, violations, top recognition and every live phase."""
    del env_ids
    state = v132_alternation_state(env)
    zero = torch.zeros((), device=env.device)
    if key == "b1_completion_rate":
        return state["b1_completion_count"].float() / state["encounter_count"].clamp_min(1).float()
    if key == "ordered_completion_rate":
        return state["ordered_completion_count"].float() / state["physical_completion_count"].clamp_min(1).float()
    if key == "wrong_completion_rate":
        return state["wrong_count"].float() / state["physical_completion_count"].clamp_min(1).float()
    if key == "simultaneous_event_rate":
        return state["simultaneous_count"].float() / state["encounter_count"].clamp_min(1).float()
    if key == "skip_rate":
        return state["skip_count"].float() / state["encounter_count"].clamp_min(1).float()
    if key == "plateau_positive_rate":
        return state["plateau_positive_count"].float() / state["plateau_probe_count"].clamp_min(1).float()
    if key == "top_follow_success_rate":
        return state["top_success_count"].float() / state["top_entry_count"].clamp_min(1).float()
    if key == "expected_progress_delta_mean":
        return state["expected_progress_delta"].mean()
    if key == "same_progress_delta_mean":
        return state["same_progress_delta"].mean()
    if key == "simultaneous_progress_delta_mean":
        return state["simultaneous_progress_delta"].mean()
    if key == "barrier_score_mean":
        active = state["expected_side"] >= 0
        return torch.where(active.any(), state["barrier_score"][active].mean(), zero)
    if key == "style_reward_mean":
        return v132_style_objective(env).mean()
    phase_names = {
        "approach_b1_fraction": APPROACH_B1,
        "alternating_fraction": ALTERNATING_RISERS,
        "top_follow_fraction": TOP_PLATFORM_FOLLOW,
        "plateau_roll_fraction": PLATEAU_ROLL,
    }
    if key in phase_names:
        return (state["phase"] == phase_names[key]).float().mean()
    raise ValueError(f"Unknown V13.2 alternation metric: {key!r}")


__all__ = [
    "APPROACH_B1", "ALTERNATING_RISERS", "TOP_PLATFORM_FOLLOW", "PLATEAU_ROLL",
    "v132_episode_reset_mask",
    "v132_relaxed_log_barrier", "v132_barrier_band_score", "v132_target_index",
    "v132_expected_side", "v132_latch_first_lift", "v132_positive_plateau_evidence",
    "v132_deploy_clearance_onset",
    "v132_top_follow_potential", "v132_violation_buckets", "v132_alternation_state",
    "v132_deployable_phase_state", "v132_event_phase_observation", "v132_style_objective",
    "v132_alternation_metric",
]
