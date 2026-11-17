"""Pure V11 continuous-stair landing geometry.

All coordinates are yaw-aligned: +x is the commanded stair approach direction
and y is lateral.  This module deliberately has no simulator access so its
interval and responsibility semantics can be unit-tested before PPO is run.
"""

from __future__ import annotations

import torch


def v11_dt_normalized_event(event: torch.Tensor, step_dt: float) -> torch.Tensor:
    """Return a one-shot event whose configured weight is its final return."""
    if float(step_dt) <= 0.0:
        raise ValueError(f"V11 step_dt must be positive, got {step_dt}.")
    return event.float() / float(step_dt)


def v11_softalt_termination(
    confirmed_front_riser_collision: torch.Tensor,
    same_side_repeat: torch.Tensor,
) -> torch.Tensor:
    """Terminate on physical collision, never on a style-only repeat."""
    if confirmed_front_riser_collision.ndim != 2:
        raise ValueError("V11 front-riser collision state must be [num_envs, num_front_wheels].")
    if same_side_repeat.shape != confirmed_front_riser_collision.shape[:1]:
        raise ValueError("V11 repeat state must match the collision environment dimension.")
    del same_side_repeat
    return confirmed_front_riser_collision.any(dim=1)


def v11_softalt_candidate_ready(
    clearance: torch.Tensor,
    crossing: torch.Tensor,
    support: torch.Tensor,
    vertical_contact: torch.Tensor,
    eligible_sides: torch.Tensor,
    sequence_started: torch.Tensor,
    anchor_support_valid: torch.Tensor,
    *,
    soft_alternation: bool,
) -> torch.Tensor:
    """Apply physical completion gates without letting SoftAlt style gate physics."""
    shape = clearance.shape
    if any(x.shape != shape for x in (crossing, support, vertical_contact, eligible_sides)):
        raise ValueError("V11 per-side completion tensors must have matching shapes.")
    if sequence_started.shape != shape[:1] or anchor_support_valid.shape != shape[:1]:
        raise ValueError("V11 sequence and anchor tensors must match the environment dimension.")
    ready = clearance & crossing & support & ~vertical_contact
    if not soft_alternation:
        ready &= eligible_sides
        ready &= (~sequence_started | anchor_support_valid)[:, None]
    return ready


def v11_restore_stair_tracking_quality(
    scaled_score: torch.Tensor, is_stair: torch.Tensor, stair_scale: float,
) -> torch.Tensor:
    """Restore the curriculum's 0..1 tracking quality after reward-only scaling."""
    if scaled_score.shape != is_stair.shape:
        raise ValueError("V11 tracking score and stair mask must have matching shapes.")
    if not 0.0 < float(stair_scale) <= 1.0:
        raise ValueError(f"V11 stair tracking scale must be in (0, 1], got {stair_scale}.")
    return torch.where(is_stair, scaled_score / float(stair_scale), scaled_score)


def v11_softalt_quality_gate(
    platform_rate: torch.Tensor,
    compliance_rate: torch.Tensor,
    repeat_rate: torch.Tensor,
    collision_rate: torch.Tensor,
    transition_valid: torch.Tensor,
    *,
    platform_threshold: float,
    compliance_threshold: float,
    repeat_threshold: float,
    collision_threshold: float,
) -> torch.Tensor:
    """Pure ascending-stair promotion gate; invalid transition rates fail closed."""
    shape = platform_rate.shape
    if any(x.shape != shape for x in (compliance_rate, repeat_rate, collision_rate, transition_valid)):
        raise ValueError("V11 SoftAlt curriculum tensors must have matching shapes.")
    return (
        transition_valid
        & (platform_rate >= float(platform_threshold))
        & (compliance_rate >= float(compliance_threshold))
        & (repeat_rate <= float(repeat_threshold))
        & (collision_rate <= float(collision_threshold))
    )


def v11_safe_landing_region(
    riser1_x: torch.Tensor,
    tread_y_min: torch.Tensor, tread_y_max: torch.Tensor,
    wheel_radius: float, front_margin: float, lateral_margin: float,
) -> dict[str, torch.Tensor]:
    """Return the one-sided feasible wheel-centre region on the first tread.

    The longitudinal constraint is deliberately defined only by the first
    riser: the wheel centre must clear that vertical face.  The next riser is
    not used as a rear boundary, so it cannot create an artificial ``-5 cm``
    (or other) second-riser target.  The observed lateral tread corridor keeps
    its two-sided safety bounds.
    """
    x_min = riser1_x + float(wheel_radius) + float(front_margin)
    y_min = tread_y_min + float(wheel_radius) + float(lateral_margin)
    y_max = tread_y_max - float(wheel_radius) - float(lateral_margin)
    valid = y_max > y_min
    return {
        "x_min": x_min, "y_min": y_min, "y_max": y_max,
        "y_ref": 0.5 * (y_min + y_max), "valid": valid,
    }


def v11_landing_scores(
    wheel_xy: torch.Tensor, region: dict[str, torch.Tensor], active: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Score wheel centres against a one-sided X and bounded Y landing region.

    ``inside`` requires clearing the first riser and staying in the observed
    lateral corridor.  ``center`` shapes only lateral centring: a longitudinal
    centre would reintroduce the removed second-riser constraint.
    """
    x, y = wheel_xy[..., 0], wheel_xy[..., 1]
    x_min, y_min, y_max = (region[k].unsqueeze(-1) for k in ("x_min", "y_min", "y_max"))
    valid = region["valid"].unsqueeze(-1) & active
    inside = valid & (x >= x_min) & (y >= y_min) & (y <= y_max)
    dy = (y - region["y_ref"].unsqueeze(-1)).abs() / (0.5 * (y_max - y_min)).clamp_min(1.0e-6)
    center = (1.0 - dy).clamp(0.0, 1.0) * inside.float()
    edge_violation = (
        (x_min - x).clamp_min(0.0) + (y_min - y).clamp_min(0.0) + (y - y_max).clamp_min(0.0)
    ) * valid.float()
    return {"inside": inside, "center": center, "edge_violation": edge_violation, "valid": valid}


def v11_latch_responsible_front_wheel(
    responsible: torch.Tensor, candidate_ready: torch.Tensor,
) -> torch.Tensor:
    """Latch FL/FR responsibility once a candidate first earns safe support.

    ``responsible`` is -1 when unassigned and otherwise 0 (FL) or 1 (FR).
    A simultaneous FL/FR success is resolved by a uniform random choice.  A
    fixed lower-index tie-break would otherwise create a training-side bias in
    perfectly symmetric scenes before the policy has learned a preference.
    """
    if candidate_ready.shape[-1] != 2:
        raise ValueError("V11 responsibility accepts FL/FR candidates only")
    choose = candidate_ready & (responsible[:, None] < 0)
    index = choose.float().argmax(dim=1)
    simultaneous = choose.all(dim=1)
    random_side = torch.randint(0, 2, (responsible.shape[0],), device=responsible.device)
    index = torch.where(simultaneous, random_side, index)
    assigned = choose.any(dim=1)
    return torch.where(assigned, index.to(responsible.dtype), responsible)


def v11_expected_front_mask(expected_side: torch.Tensor) -> torch.Tensor:
    """Return eligible FL/FR sides for one alternating-riser phase.

    ``-1`` leaves the first riser open to either side.  Once one side safely
    completes that riser, callers store the opposite side (0=FL, 1=FR) and
    only that side is eligible for the next uncredited riser.
    """
    if expected_side.ndim != 1:
        raise ValueError("V11 expected-side state must be a [num_envs] tensor.")
    if ((expected_side < -1) | (expected_side > 1)).any():
        raise ValueError("V11 expected-side values must be -1, 0 or 1.")
    sides = torch.arange(2, device=expected_side.device, dtype=expected_side.dtype)
    return (expected_side[:, None] < 0) | (expected_side[:, None] == sides[None, :])


def v11_latch_alternating_front_completion(
    expected_side: torch.Tensor, candidate_ready: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Choose one eligible completion and toggle the next expected side.

    The first riser keeps V11's unbiased random tie-break.  Later risers can
    only complete on the explicitly expected opposite side.  Ready signals on
    the other side are returned as diagnostics but never receive credit.
    """
    if candidate_ready.ndim != 2 or candidate_ready.shape[1] != 2:
        raise ValueError("V11 alternating completion expects [num_envs, 2] candidates.")
    if expected_side.shape != candidate_ready.shape[:1]:
        raise ValueError("V11 expected-side and candidate tensors have incompatible shapes.")
    eligible_mask = v11_expected_front_mask(expected_side)
    eligible_ready = candidate_ready & eligible_mask
    completed_side = v11_latch_responsible_front_wheel(
        torch.full_like(expected_side, -1), eligible_ready,
    )
    event = completed_side >= 0
    next_expected = torch.where(event, 1 - completed_side, expected_side)
    return {
        "event": event,
        "completed_side": completed_side,
        "next_expected_side": next_expected,
        "eligible_mask": eligible_mask,
        "out_of_turn_ready": (candidate_ready & ~eligible_mask).any(dim=1),
    }


def v11_classify_soft_alternation(
    previous_completed_side: torch.Tensor,
    candidate_ready: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify a physical completion after geometry has accepted either side.

    ``previous_completed_side`` is episode-local (-1 before the first event,
    otherwise 0=FL or 1=FR).  It is deliberately not consulted until after a
    physical candidate has been selected.  Therefore changing the alternation
    ledger cannot suppress clearance, crossing, support, or completion.
    """
    if candidate_ready.ndim != 2 or candidate_ready.shape[1] != 2:
        raise ValueError("V11 soft alternation expects [num_envs, 2] candidates.")
    if previous_completed_side.shape != candidate_ready.shape[:1]:
        raise ValueError("V11 previous-side and candidate tensors have incompatible shapes.")
    if ((previous_completed_side < -1) | (previous_completed_side > 1)).any():
        raise ValueError("V11 previous-side values must be -1, 0 or 1.")

    simultaneous = candidate_ready.all(dim=1)
    current = v11_latch_responsible_front_wheel(
        torch.full_like(previous_completed_side, -1), candidate_ready,
    )
    event = current >= 0
    first = event & (previous_completed_side < 0)
    transition = event & (previous_completed_side >= 0)
    # A simultaneous FL/FR completion is real physical progress, but it is
    # neither evidence for alternation nor a same-side repeat.  Keep it in a
    # dedicated style bucket so random responsibility cannot inflate quality.
    alternating = transition & ~simultaneous & (current != previous_completed_side)
    repeat = transition & ~simultaneous & (current == previous_completed_side)
    return {
        "event": event,
        "completed_side": current,
        "first_completion": first,
        "alternating_completion": alternating,
        "same_side_repeat": repeat,
        "simultaneous_completion": simultaneous & event,
        # Diagnostics only.  SoftAlt never uses this mask to gate physics.
        "next_expected_side": torch.where(event, 1 - current, torch.full_like(current, -1)),
    }


def v11_alternating_reward_event(
    physical_completion: torch.Tensor, sequence_started: torch.Tensor,
) -> torch.Tensor:
    """Pay only completions after the first side has seeded the sequence.

    The first physical riser remains observable in diagnostics but cannot be
    farmed by repeatedly terminating and resetting immediately afterwards.
    """
    if physical_completion.shape != sequence_started.shape:
        raise ValueError("V11 alternating event tensors must have matching shapes.")
    return physical_completion & sequence_started


def v11_select_next_uncredited_riser(
    valid: torch.Tensor,
    forward_separation: torch.Tensor,
    sequence_started: torch.Tensor,
    minimum_separation: float,
) -> torch.Tensor:
    """Select the nearest visible riser strictly ahead of the last credited one.

    Before the sequence starts, raw slot 0 is the current riser.  Afterwards a
    reappearing/held copy of the credited face is rejected by the signed
    forward separation, and slot 1 is used when slot 0 is still that old face.
    ``-1`` means no valid next target and therefore fail-closed reward gating.
    """
    if valid.ndim != 2 or valid.shape[1] != 2 or forward_separation.shape != valid.shape:
        raise ValueError("V11 next-riser selection expects matching [num_envs, 2] tensors.")
    if sequence_started.shape != valid.shape[:1]:
        raise ValueError("V11 sequence state has an incompatible shape.")
    eligible = valid & (
        (~sequence_started)[:, None]
        | (forward_separation >= float(minimum_separation))
    )
    slot = eligible.float().argmax(dim=1).to(torch.long)
    return torch.where(eligible.any(dim=1), slot, torch.full_like(slot, -1))


def v11_update_clearance_escrow(
    max_progress: torch.Tensor,
    escrow: torch.Tensor,
    previous_x: torch.Tensor,
    previous_x_valid: torch.Tensor,
    wheel_x: torch.Tensor,
    wheel_bottom_z: torch.Tensor,
    lower_z: torch.Tensor,
    upper_z: torch.Tensor,
    active_before_riser: torch.Tensor,
    clearance_margin: float,
    forward_epsilon: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    """Accumulate clearance only while a wheel advances toward one locked riser.

    Raising a stationary wheel must never earn credit.  The caller owns the
    eventual payout: this function only records an escrow amount which is
    released after crossing and upper-tread support have both been latched.
    """
    denominator = (upper_z - lower_z + float(clearance_margin)).clamp_min(1.0e-6)
    progress = ((wheel_bottom_z - lower_z) / denominator).clamp(0.0, 1.0)
    advancing = previous_x_valid & (wheel_x > previous_x + float(forward_epsilon))
    eligible = active_before_riser & advancing
    new_progress = torch.where(eligible, torch.maximum(max_progress, progress), max_progress)
    escrow_delta = torch.where(eligible, (new_progress - max_progress).clamp_min(0.0), torch.zeros_like(progress))
    return {
        "max_progress": new_progress,
        "escrow": escrow + escrow_delta,
        "clearance": eligible & (new_progress >= 1.0),
        "previous_x": wheel_x,
        "previous_x_valid": active_before_riser,
    }


def v11_settle_clearance_escrow(
    escrow: torch.Tensor, responsible: torch.Tensor, landing_event: torch.Tensor,
) -> torch.Tensor:
    """Release one selected-wheel escrow exactly on a safe landing event."""
    selected = escrow.gather(1, responsible.clamp_min(0)[:, None]).squeeze(1)
    return selected * landing_event.float()


def v11_settle_bilateral_escrow(
    escrow: torch.Tensor,
    bilateral_landing_event: torch.Tensor,
    *,
    average: bool = False,
) -> torch.Tensor:
    """Release both front-wheel escrows only on bilateral safe completion."""
    if escrow.ndim != 2 or escrow.shape[1] != 2 or bilateral_landing_event.shape != escrow.shape[:1]:
        raise ValueError("V11 bilateral escrow expects [num_envs, 2] values and [num_envs] events.")
    value = escrow.mean(dim=1) if average else escrow.sum(dim=1)
    return value * bilateral_landing_event.float()


def v11_front_reward_event(
    first_landing_event: torch.Tensor,
    bilateral_landing_event: torch.Tensor,
    defer_until_bilateral: bool,
) -> torch.Tensor:
    """Select the one-shot event which is allowed to release V11 front rewards."""
    if first_landing_event.shape != bilateral_landing_event.shape:
        raise ValueError("V11 front reward events must have matching [num_envs] tensors.")
    return bilateral_landing_event if bool(defer_until_bilateral) else first_landing_event


def v11_confirm_vertical_contact(
    consecutive_steps: torch.Tensor,
    contact_now: torch.Tensor,
    confirmation_frames: int,
) -> dict[str, torch.Tensor]:
    """Debounce front-riser contact before declaring an encounter failure."""
    if consecutive_steps.shape != contact_now.shape:
        raise ValueError("V11 vertical-contact confirmation expects matching [num_envs, 2] tensors.")
    frames = int(confirmation_frames)
    if frames < 1:
        raise ValueError(f"confirmation_frames must be at least one, got {confirmation_frames}.")
    steps = torch.where(contact_now, consecutive_steps + 1, torch.zeros_like(consecutive_steps))
    return {"steps": steps, "confirmed": steps >= frames}


def v11_confirm_support(
    consecutive_steps: torch.Tensor,
    support_now: torch.Tensor,
    confirmation_frames: int,
) -> dict[str, torch.Tensor]:
    """Require persistent tread support instead of latching a one-frame touch."""
    if consecutive_steps.shape != support_now.shape:
        raise ValueError("V11 support confirmation expects matching tensors.")
    frames = int(confirmation_frames)
    if frames < 1:
        raise ValueError(f"confirmation_frames must be at least one, got {confirmation_frames}.")
    steps = torch.where(support_now, consecutive_steps + 1, torch.zeros_like(consecutive_steps))
    return {"steps": steps, "confirmed": steps >= frames}


def v11_anchor_support_valid(
    anchor_side: torch.Tensor,
    wheel_x: torch.Tensor,
    wheel_bottom_z: torch.Tensor,
    force_z: torch.Tensor,
    previous_riser_x: torch.Tensor,
    previous_upper_z: torch.Tensor,
    target_riser_x: torch.Tensor,
    *,
    wheel_radius: float,
    front_margin: float,
    support_tolerance: float,
    force_threshold: float,
) -> torch.Tensor:
    """Check that the prior-step owner remains supported between two faces."""
    side = anchor_side.clamp_min(0)[:, None]
    selected_x = wheel_x.gather(1, side).squeeze(1)
    selected_bottom = wheel_bottom_z.gather(1, side).squeeze(1)
    selected_force_z = force_z.gather(1, side).squeeze(1)
    lower = previous_riser_x + float(wheel_radius) + float(front_margin)
    upper = target_riser_x - float(wheel_radius) - float(front_margin)
    return (
        (anchor_side >= 0)
        & (upper > lower)
        & (selected_x >= lower)
        & (selected_x < upper)
        & ((selected_bottom - previous_upper_z).abs() <= float(support_tolerance))
        & (selected_force_z > float(force_threshold))
    )


def v11_clearance_surplus(
    wheel_bottom_z: torch.Tensor,
    upper_tread_z: torch.Tensor,
    clearance_margin: float,
    surplus_scale: float,
) -> torch.Tensor:
    """Bound target-relative clearance above the unchanged safety threshold."""
    if float(surplus_scale) <= 0.0:
        raise ValueError(f"surplus_scale must be positive, got {surplus_scale}.")
    return (
        (wheel_bottom_z - upper_tread_z - float(clearance_margin))
        / float(surplus_scale)
    ).clamp(0.0, 1.0)


def v11_crossing_clearance_gate(
    wheel_center_z: torch.Tensor,
    upper_tread_z: torch.Tensor,
    crossing_now: torch.Tensor,
    *,
    wheel_radius: float,
    clearance_margin: float,
) -> dict[str, torch.Tensor]:
    """Evaluate target-relative wheel clearance at the actual face crossing.

    The gate is deliberately sampled at crossing rather than latched from an
    earlier high pose.  This prevents a wheel from reaching the requested
    height before the face, dropping again, and still earning a landing event.
    """
    center_above_tread = wheel_center_z - upper_tread_z
    required_center = float(wheel_radius) + float(clearance_margin)
    margin = center_above_tread - required_center
    return {
        "event": crossing_now,
        "clear": crossing_now & (margin >= 0.0),
        "low": crossing_now & (margin < 0.0),
        "center_above_tread": center_above_tread,
        "margin": margin,
    }


def v11_safe_support_clearance_progress(
    center_above_tread: torch.Tensor,
    safe_support_event: torch.Tensor,
    *,
    required_center_above_tread: float,
) -> torch.Tensor:
    """Bound crossing-height shaping and release it only on safe support."""
    required = float(required_center_above_tread)
    if required <= 0.0:
        raise ValueError(
            f"required_center_above_tread must be positive, got {required_center_above_tread}."
        )
    progress = (center_above_tread / required).clamp(0.0, 1.0)
    return progress * safe_support_event.float()


def v11_front_step_mobility_gate(
    actor_b1: torch.Tensor,
    target_active: torch.Tensor,
    wheel_x: torch.Tensor,
    riser_x: torch.Tensor,
    candidate_ready: torch.Tensor,
    vertical_contact: torch.Tensor,
    wheel_radius: float,
) -> torch.Tensor:
    """Open per-front-leg mobility only near an actor-visible live riser.

    Each side closes independently after safe completion or a vertical-face
    collision.  The two-radius approach band gives hip-y enough room to start
    contributing without relaxing the complete actor-visible approach.
    """
    near_face = (riser_x[:, None] - wheel_x) <= 2.0 * float(wheel_radius)
    active = actor_b1 & target_active
    return active[:, None] & near_face & ~candidate_ready & ~vertical_contact


def v11_bounded_relative_hipy_participation_scores(
    hipy_deviation: torch.Tensor,
    knee_deviation: torch.Tensor,
    active: torch.Tensor,
    target_share: float,
    share_half_width: float,
    activation_scale: float,
) -> torch.Tensor:
    """Return a bounded per-leg hip-y share score without rewarding action speed.

    The score peaks when ``hip-y / (hip-y + knee)`` reaches the configured
    target and falls to zero outside its bounded neighbourhood.  A saturating
    pose-activation factor prevents a vanishingly small hip-y displacement
    from earning the relative-share score, while giving no incentive to grow
    the displacement without bound.  The output is in ``[0, 1]`` per leg.
    """
    if hipy_deviation.shape != knee_deviation.shape or active.shape != hipy_deviation.shape:
        raise ValueError(
            "V11 hip-y participation expects matching [num_envs, 2] hip-y, knee and gate tensors."
        )
    target = float(target_share)
    width = float(share_half_width)
    scale = float(activation_scale)
    if not 0.0 < target < 1.0:
        raise ValueError(f"target_share must be in (0, 1), got {target_share}.")
    if not 0.0 < width <= min(target, 1.0 - target):
        raise ValueError(
            f"share_half_width must keep the target band inside [0, 1], got {share_half_width}."
        )
    if scale <= 0.0:
        raise ValueError(f"activation_scale must be positive, got {activation_scale}.")

    share = hipy_deviation / (hipy_deviation + knee_deviation).clamp_min(1.0e-6)
    relative_score = (1.0 - (share - target).abs() / width).clamp(0.0, 1.0)
    pose_activation = torch.tanh(hipy_deviation / scale)
    return relative_score * pose_activation * active.float()


def v11_bounded_relative_hipy_participation(
    hipy_deviation: torch.Tensor,
    knee_deviation: torch.Tensor,
    active: torch.Tensor,
    target_share: float,
    share_half_width: float,
    activation_scale: float,
) -> torch.Tensor:
    """Average the bounded relative hip-y score over active front legs."""
    per_leg = v11_bounded_relative_hipy_participation_scores(
        hipy_deviation,
        knee_deviation,
        active,
        target_share,
        share_half_width,
        activation_scale,
    )
    return per_leg.sum(dim=1) / active.float().sum(dim=1).clamp_min(1.0)


def v11_update_latest_participation_escrow(
    escrow: torch.Tensor, score: torch.Tensor, active: torch.Tensor,
) -> torch.Tensor:
    """Keep the latest eligible pose score instead of an exploitable phase maximum."""
    if escrow.shape != score.shape or active.shape != score.shape:
        raise ValueError("V11 participation escrow expects matching [num_envs, 2] tensors.")
    return torch.where(active, score, escrow)


def v11_bounded_knee_excess(
    knee_deviation: torch.Tensor,
    active: torch.Tensor,
    free_band: float,
    excess_scale: float,
) -> dict[str, torch.Tensor]:
    """Return a soft, bounded knee cost only outside a necessary-flexion band."""
    if knee_deviation.shape != active.shape:
        raise ValueError("V11 knee excess expects matching [num_envs, 2] deviation and gate tensors.")
    band = float(free_band)
    scale = float(excess_scale)
    if band < 0.0:
        raise ValueError(f"free_band must be non-negative, got {free_band}.")
    if scale <= 0.0:
        raise ValueError(f"excess_scale must be positive, got {excess_scale}.")
    normalized = ((knee_deviation - band).clamp_min(0.0) / scale).clamp_max(1.0)
    per_leg = normalized.square() * active.float()
    active_count = active.float().sum(dim=1).clamp_min(1.0)
    return {
        "cost": per_leg.sum(dim=1) / active_count,
        "rate": ((normalized > 0.0) & active).float().sum(dim=1) / active_count,
        "normalized": normalized * active.float(),
    }


def v11_update_bilateral_support_window(
    latched: torch.Tensor,
    age: torch.Tensor,
    support_now: torch.Tensor,
    active: torch.Tensor,
    window_frames: int,
) -> dict[str, torch.Tensor]:
    """Latch two support contacts only when both occur in one bounded window."""
    updated = (latched | support_now) & active[:, None]
    partial = updated.any(dim=1) & ~updated.all(dim=1)
    updated_age = torch.where(partial, age + 1, torch.zeros_like(age))
    expired = updated_age > int(window_frames)
    updated = torch.where(expired[:, None], support_now & active[:, None], updated)
    restarted_partial = updated.any(dim=1) & ~updated.all(dim=1)
    updated_age = torch.where(
        expired & restarted_partial, torch.ones_like(updated_age),
        torch.where(updated.all(dim=1), torch.zeros_like(updated_age), updated_age),
    )
    return {"latched": updated, "age": updated_age, "ready": updated.all(dim=1) & active}


__all__ = [
    "v11_safe_landing_region", "v11_landing_scores", "v11_latch_responsible_front_wheel",
    "v11_expected_front_mask", "v11_latch_alternating_front_completion",
    "v11_alternating_reward_event",
    "v11_select_next_uncredited_riser",
    "v11_update_clearance_escrow", "v11_settle_clearance_escrow", "v11_settle_bilateral_escrow",
    "v11_front_reward_event", "v11_confirm_vertical_contact", "v11_confirm_support",
    "v11_anchor_support_valid", "v11_clearance_surplus", "v11_crossing_clearance_gate",
    "v11_safe_support_clearance_progress",
    "v11_update_bilateral_support_window",
]
