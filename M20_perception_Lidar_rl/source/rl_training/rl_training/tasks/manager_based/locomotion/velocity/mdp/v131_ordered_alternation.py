"""V13.1 ordered-riser soft gait shaping on the frozen V13/V11-FD task.

The policy and physical landing detector remain unchanged.  This module owns a
reward-side state machine that locks the first observed ascending face as B1,
predicts exactly one tread-width-ahead target after every correct completion,
and refuses to reinterpret a skipped face as strict success. A separate local
teaching ledger re-synchronizes from physical truth at full strength. Strict
failure affects only scoring and curriculum: native rewards remain untouched.
Nothing here masks actions or terminates an episode.
"""

from __future__ import annotations

import torch


APPROACH_B1 = 0
ALTERNATING_RISERS = 1
TOP_FOLLOW = 2
TOP_TRAILING_FOLLOW = 3
PLATEAU_ROLL = 4
SEQUENCE_FAILED = 5


def v131_latch_b1_leader(
    phase: torch.Tensor,
    expected_side: torch.Tensor,
    lift_ready: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Commit a unique first mover to B1 while leaving a tie unassigned.

    ``lift_ready`` must already be target-qualified: generic joint motion is
    not enough.  The latch is deliberately soft (reward-side only), but once a
    unique FL/FR mover appears that side owns B1 until physical settlement.
    """
    if phase.ndim != 1 or expected_side.shape != phase.shape:
        raise ValueError("V13.1 B1 phase/side tensors must have shape [num_envs].")
    if lift_ready.shape != (phase.shape[0], 2) or lift_ready.dtype != torch.bool:
        raise ValueError("V13.1 B1 lift readiness must be a bool [num_envs, 2] tensor.")

    unassigned = (phase == APPROACH_B1) & (expected_side < 0)
    lift_count = lift_ready.long().sum(dim=1)
    unique = unassigned & (lift_count == 1)
    simultaneous = unassigned & (lift_count == 2)
    unique_side = lift_ready.long().argmax(dim=1)
    next_expected = torch.where(unique, unique_side.to(expected_side.dtype), expected_side)
    return {
        "expected_side": next_expected,
        "unique_lift_event": unique,
        "simultaneous_lift_event": simultaneous,
    }


def v131_confirm_b1_first_mover(
    previous_side: torch.Tensor,
    previous_steps: torch.Tensor,
    previous_tie_steps: torch.Tensor,
    potential: torch.Tensor,
    unassigned: torch.Tensor,
    *,
    onset_threshold: float,
    release_threshold: float,
    separation: float,
    confirmation_frames: int,
) -> dict[str, torch.Tensor]:
    """Debounce B1 side commitment while retaining a candidate hysteretically."""
    n = previous_side.shape[0]
    if previous_steps.shape != (n,) or previous_tie_steps.shape != (n,):
        raise ValueError("V13.1 first-mover counters must have shape [num_envs].")
    if potential.shape != (n, 2) or unassigned.shape != (n,):
        raise ValueError("V13.1 first-mover potential must be [num_envs, 2].")
    if not 0.0 < float(release_threshold) < float(onset_threshold) < 1.0:
        raise ValueError("V13.1 first-mover thresholds must satisfy 0 < release < onset < 1.")
    if float(separation) <= 0.0 or int(confirmation_frames) < 2:
        raise ValueError("V13.1 first-mover separation/confirmation are invalid.")

    above_onset = potential >= float(onset_threshold)
    lead_side = potential.argmax(dim=1)
    lead_gap = (potential[:, 0] - potential[:, 1]).abs()
    clear_onset = above_onset.any(dim=1) & (
        (above_onset.long().sum(dim=1) == 1) | (lead_gap >= float(separation))
    )
    previous_valid = (previous_side >= 0) & unassigned
    previous_index = previous_side.clamp(0, 1)
    previous_value = potential.gather(1, previous_index[:, None]).squeeze(1)
    other_value = potential.gather(1, (1 - previous_index)[:, None]).squeeze(1)
    hold_previous = (
        previous_valid
        & (previous_value >= float(release_threshold))
        & (previous_value >= other_value)
    )
    candidate = torch.where(
        unassigned & clear_onset,
        lead_side.to(previous_side.dtype),
        torch.where(hold_previous, previous_side, torch.full_like(previous_side, -1)),
    )
    candidate_valid = candidate >= 0
    same_candidate = candidate_valid & (candidate == previous_side)
    candidate_steps = torch.where(
        candidate_valid,
        torch.where(same_candidate, previous_steps + 1, torch.ones_like(previous_steps)),
        torch.zeros_like(previous_steps),
    )
    confirmed = candidate_valid & (candidate_steps >= int(confirmation_frames))
    lift_ready = torch.zeros_like(potential, dtype=torch.bool)
    lift_ready.scatter_(1, candidate.clamp(0, 1)[:, None], True)
    lift_ready &= confirmed[:, None]

    unresolved_tie = (
        unassigned & ~candidate_valid & above_onset.all(dim=1)
        & (lead_gap < float(separation))
    )
    tie_steps = torch.where(
        unresolved_tie, previous_tie_steps + 1, torch.zeros_like(previous_tie_steps),
    )
    simultaneous_ready = unresolved_tie & (tie_steps >= int(confirmation_frames))
    return {
        "candidate_side": candidate,
        "candidate_steps": candidate_steps,
        "tie_steps": tie_steps,
        "lift_ready": lift_ready,
        "simultaneous_ready": simultaneous_ready,
    }


def v131_locked_target_potential(
    wheel_bottom_z: torch.Tensor,
    wheel_forward_distance: torch.Tensor,
    support: torch.Tensor,
    target_lower_z: torch.Tensor,
    target_upper_z: torch.Tensor,
    *,
    clearance_margin: float,
    approach_span: float,
    crossing_span: float,
) -> torch.Tensor:
    """Continuous per-wheel progress against a locked adjacent riser.

    The pre-face half is the minimum of normalized lift and approach, so
    lifting in place or merely rolling toward the face cannot mint the whole
    signal.  The remaining halves require a clearance-qualified crossing and
    stable upper-tread support.  No live scanner reference is consumed here.
    """
    if wheel_bottom_z.shape != wheel_forward_distance.shape or wheel_bottom_z.shape != support.shape:
        raise ValueError("V13.1 locked-target wheel tensors must have matching shapes.")
    if wheel_bottom_z.ndim != 2 or wheel_bottom_z.shape[1] != 2:
        raise ValueError("V13.1 locked-target potential expects FL/FR tensors.")
    if target_lower_z.shape != wheel_bottom_z.shape[:1] or target_upper_z.shape != target_lower_z.shape:
        raise ValueError("V13.1 locked-target heights must have shape [num_envs].")
    if min(float(clearance_margin), float(approach_span), float(crossing_span)) <= 0.0:
        raise ValueError("V13.1 locked-target potential scales must be positive.")

    lift_span = (target_upper_z - target_lower_z + float(clearance_margin)).clamp_min(1.0e-6)
    lift = ((wheel_bottom_z - target_lower_z[:, None]) / lift_span[:, None]).clamp(0.0, 1.0)
    approach = ((wheel_forward_distance + float(approach_span)) / float(approach_span)).clamp(0.0, 1.0)
    qualified_approach = torch.minimum(lift, approach)
    clearance_ok = lift >= 1.0 - 1.0e-6
    crossing = (wheel_forward_distance / float(crossing_span)).clamp(0.0, 1.0) * clearance_ok.float()
    supported = support.float() * clearance_ok.float()
    return (0.50 * qualified_approach + 0.25 * crossing + 0.25 * supported).clamp(0.0, 1.0)


def v131_settle_progress_escrow(
    previous_escrow: torch.Tensor,
    signed_delta: torch.Tensor,
    eligible: torch.Tensor,
    completion: torch.Tensor,
    cancel: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Accumulate a bounded telescoping potential and pay only at completion.

    Regression removes escrow, so lift/drop/lift cannot mint repeated credit.
    A wrong side, skip, or target loss cancels unpaid progress before it can be
    observed by PPO.
    """
    if previous_escrow.ndim != 1 or any(
        value.shape != previous_escrow.shape
        for value in (signed_delta, eligible, completion, cancel)
    ):
        raise ValueError("V13.1 escrow tensors must all have shape [num_envs].")
    accrued = torch.where(
        eligible,
        (previous_escrow + signed_delta).clamp(0.0, 1.0),
        previous_escrow,
    )
    forfeited = torch.where(cancel, accrued, torch.zeros_like(accrued))
    candidate = torch.where(cancel, torch.zeros_like(accrued), accrued)
    payout = torch.where(completion & ~cancel, candidate, torch.zeros_like(candidate))
    remaining = torch.where(completion, torch.zeros_like(candidate), candidate)
    return {"escrow": remaining, "payout": payout, "forfeited": forfeited}


def v131_expected_swing_relief(
    base_cost: torch.Tensor,
    per_side_reduction: torch.Tensor,
    expected_side: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Remove only the configured expected-leg contribution from a base cost."""
    if base_cost.ndim != 1 or expected_side.shape != base_cost.shape or active.shape != base_cost.shape:
        raise ValueError("V13.1 expected-swing relief scalars must have shape [num_envs].")
    if per_side_reduction.shape != (base_cost.shape[0], 2):
        raise ValueError("V13.1 expected-swing relief must provide FL/FR reductions.")
    selected = per_side_reduction.gather(1, expected_side.clamp(0, 1)[:, None]).squeeze(1)
    selected *= (active & (expected_side >= 0)).float()
    return (base_cost - selected).clamp_min(0.0)


def v131_ordered_speed_scale(
    is_ascending: torch.Tensor,
    chain_valid: torch.Tensor,
    target_active: torch.Tensor,
    ordered_gate: torch.Tensor,
    *,
    free_fraction: float,
) -> torch.Tensor:
    """Split, without increasing, the native speed reward into free/dividend."""
    if is_ascending.ndim != 1 or any(
        value.shape != is_ascending.shape for value in (chain_valid, target_active, ordered_gate)
    ):
        raise ValueError("V13.1 speed-gate tensors must all have shape [num_envs].")
    if not 0.0 < float(free_fraction) < 1.0:
        raise ValueError("V13.1 ordered-speed free fraction must lie strictly inside (0, 1).")
    gate = ordered_gate.clamp(0.0, 1.0)
    # Approach/transfer frames without a live face retain the full native speed
    # signal. Once a target is actionable, ordered potential unlocks the
    # dividend. A strict-invalid episode retains only the free fraction in an
    # actionable window even though its local teaching chain may re-synchronise.
    actionable_gate = torch.where(target_active, gate, torch.ones_like(gate))
    actionable_gate = actionable_gate * chain_valid.float()
    ascending_scale = float(free_fraction) + (1.0 - float(free_fraction)) * actionable_gate
    return torch.where(is_ascending, ascending_scale, torch.ones_like(ascending_scale))


def v131_update_riser_ledgers(
    physical_last_riser_index: torch.Tensor,
    ordered_last_riser_index: torch.Tensor,
    observed_riser_index: torch.Tensor,
    physical_completion: torch.Tensor,
    ordered_completion: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Advance physical truth independently from strict ordered progression."""
    if physical_last_riser_index.ndim != 1 or any(
        value.shape != physical_last_riser_index.shape
        for value in (
            ordered_last_riser_index,
            observed_riser_index,
            physical_completion,
            ordered_completion,
        )
    ):
        raise ValueError("V13.1 riser-ledger tensors must all have shape [num_envs].")
    expected = ordered_last_riser_index + 1
    physical = torch.where(
        physical_completion,
        torch.maximum(physical_last_riser_index, observed_riser_index),
        physical_last_riser_index,
    )
    adjacent_ordered = physical_completion & ordered_completion & (observed_riser_index == expected)
    ordered = ordered_last_riser_index + adjacent_ordered.long()
    skip_completion = physical_completion & (observed_riser_index > expected)
    return {
        "physical_last_riser_index": physical,
        "ordered_last_riser_index": ordered,
        "adjacent_ordered_completion": adjacent_ordered,
        "skip_completion": skip_completion,
        "physical_minus_ordered_gap": (physical - ordered).clamp_min(0),
    }


def v131_platform_phase_transition(
    phase: torch.Tensor,
    last_ordered_side: torch.Tensor,
    lead_supported: torch.Tensor,
    trailing_supported: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Require opposite-to-platform, then the last-riser leg, before flat mode."""
    if phase.ndim != 1 or any(
        value.shape != phase.shape
        for value in (last_ordered_side, lead_supported, trailing_supported)
    ):
        raise ValueError("V13.1 platform transition tensors must all have shape [num_envs].")
    lead_event = (phase == TOP_FOLLOW) & lead_supported
    trailing_event = (phase == TOP_TRAILING_FOLLOW) & trailing_supported
    next_phase = torch.where(
        lead_event,
        torch.full_like(phase, TOP_TRAILING_FOLLOW),
        torch.where(trailing_event, torch.full_like(phase, PLATEAU_ROLL), phase),
    )
    active_side = torch.where(
        phase == TOP_FOLLOW,
        1 - last_ordered_side.clamp(0, 1),
        last_ordered_side.clamp(0, 1),
    )
    return {
        "phase": next_phase,
        "active_side": active_side,
        "lead_event": lead_event,
        "trailing_event": trailing_event,
        "platform_event": trailing_event,
    }


def v131_target_relation(
    reference_world_xy: torch.Tensor,
    expected_world_xy: torch.Tensor,
    forward_world_xy: torch.Tensor,
    expected_valid: torch.Tensor,
    *,
    longitudinal_tolerance: float,
    lateral_tolerance: float,
) -> dict[str, torch.Tensor]:
    """Compare a live physical face with the locked ordered-riser target."""
    if reference_world_xy.shape != expected_world_xy.shape or reference_world_xy.shape[-1] != 2:
        raise ValueError("V13.1 target positions must be matching [num_envs, 2] tensors.")
    if forward_world_xy.shape != reference_world_xy.shape or expected_valid.shape != reference_world_xy.shape[:1]:
        raise ValueError("V13.1 target validity/heading tensors have incompatible shapes.")
    if float(longitudinal_tolerance) <= 0.0 or float(lateral_tolerance) <= 0.0:
        raise ValueError("V13.1 target tolerances must be positive.")

    forward = forward_world_xy / forward_world_xy.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    lateral_axis = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
    delta = reference_world_xy - expected_world_xy
    longitudinal = (delta * forward).sum(dim=1)
    lateral = (delta * lateral_axis).sum(dim=1)
    matches = expected_valid & (longitudinal.abs() <= float(longitudinal_tolerance)) & (
        lateral.abs() <= float(lateral_tolerance)
    )
    return {"matches": matches, "longitudinal_error": longitudinal, "lateral_error": lateral}


def v131_classify_completion(
    phase: torch.Tensor,
    expected_side: torch.Tensor,
    completed_side: torch.Tensor,
    physical_event: torch.Tensor,
    simultaneous: torch.Tensor,
    target_matches: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Classify one physical completion without silently consuming a violation."""
    shape = phase.shape
    if phase.ndim != 1 or any(
        value.shape != shape
        for value in (expected_side, completed_side, physical_event, simultaneous, target_matches)
    ):
        raise ValueError("V13.1 completion tensors must all have shape [num_envs].")

    approach = phase == APPROACH_B1
    alternating = phase == ALTERNATING_RISERS
    active = approach | alternating
    single = physical_event & ~simultaneous
    # Before a measurable B1 lift onset either side may seed. Once one side has
    # lifted first, expected_side is latched even while phase remains APPROACH_B1
    # and the other side may no longer steal the B1 completion.
    approach_side_ok = (expected_side < 0) | (completed_side == expected_side)
    correct_side = torch.where(approach, approach_side_ok, completed_side == expected_side)
    ordered = active & single & target_matches & correct_side
    seed = ordered & approach
    compliant = ordered & alternating
    wrong_side = active & physical_event & target_matches & (~single | ~correct_side)
    wrong_target = active & physical_event & ~target_matches
    violation = wrong_side | wrong_target
    return {
        "ordered_event": ordered,
        "seed_event": seed,
        "compliant_event": compliant,
        "wrong_side_event": wrong_side,
        "wrong_target_event": wrong_target,
        "violation_event": violation,
    }


def v131_signed_progress(
    previous_potential: torch.Tensor,
    current_potential: torch.Tensor,
    phase: torch.Tensor,
    expected_side: torch.Tensor,
    eligible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return target-local progress with regression clawback and no rocking profit.

    Before B1 either side may lead, so the larger per-side potential is the
    leader signal.  Afterwards only the scheduled opposite side earns positive
    credit.  Same-side and bilateral overlap settle only as costs.  The positive
    signal is a signed potential difference, so lift/drop/lift cycles have zero
    undiscounted net credit rather than repeatedly minting historical maxima.
    """
    if previous_potential.shape != current_potential.shape or previous_potential.ndim != 2:
        raise ValueError("V13.1 potentials must be matching [num_envs, 2] tensors.")
    if previous_potential.shape[1] != 2 or any(
        value.shape != previous_potential.shape[:1] for value in (phase, expected_side, eligible)
    ):
        raise ValueError("V13.1 progress expects FL/FR potentials and one phase per environment.")

    delta = current_potential - previous_potential
    approach = phase == APPROACH_B1
    approach_unassigned = approach & (expected_side < 0)
    alternating = phase == ALTERNATING_RISERS
    expected = expected_side.clamp(0, 1)
    expected_delta = delta.gather(1, expected[:, None]).squeeze(1)
    same_delta = delta.gather(1, (1 - expected)[:, None]).squeeze(1)
    leader_delta = current_potential.amax(dim=1) - previous_potential.amax(dim=1)
    expected_delta = torch.where(approach_unassigned, leader_delta, expected_delta)
    same_positive = torch.where(
        approach_unassigned, torch.zeros_like(same_delta), same_delta.clamp_min(0.0),
    )
    per_side_positive = delta.clamp_min(0.0)
    simultaneous = per_side_positive.amin(dim=1)
    expected_positive = expected_delta.clamp_min(0.0)
    overlap = torch.minimum(expected_positive, simultaneous)
    active = eligible & (approach | alternating)
    expected_only = torch.where(expected_delta >= 0.0, expected_delta - overlap, expected_delta)
    return {
        "expected": expected_only * active.float(),
        "same": (same_positive - torch.minimum(same_positive, overlap)).clamp_min(0.0) * active.float(),
        "simultaneous": overlap * active.float(),
    }


def v131_historical_max_delta(
    previous_best: torch.Tensor,
    current: torch.Tensor,
    eligible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return a telescoping side-neutral historical-best increment.

    This is the B1 bootstrap primitive: either front leg may create progress,
    but lowering and raising it again cannot mint the same reward twice.
    """
    if previous_best.ndim != 1 or any(
        value.shape != previous_best.shape for value in (current, eligible)
    ):
        raise ValueError("V13.1 historical-max inputs must have shape [num_envs].")
    best = torch.where(eligible, torch.maximum(previous_best, current), previous_best)
    return {"best": best, "delta": (best - previous_best).clamp_min(0.0)}


def v131_skip_progress_delta(
    previous_best: torch.Tensor,
    wheel_forward_distance: torch.Tensor,
    active: torch.Tensor,
    valid_adjacent_completion: torch.Tensor,
    *,
    step_width: float,
    wheel_radius: float,
    front_margin: float,
) -> dict[str, torch.Tensor]:
    """Dense, telescoping overshoot cost relative to the locked adjacent tread."""
    safe_end = float(step_width) - float(wheel_radius) - float(front_margin)
    span = float(step_width) - safe_end
    if safe_end <= 0.0 or span <= 0.0:
        raise ValueError("V13.1 skip geometry must leave a positive safe tread and overshoot span.")
    psi = ((wheel_forward_distance.amax(dim=1) - safe_end) / span).clamp(0.0, 1.0)
    eligible = active & ~valid_adjacent_completion
    result = v131_historical_max_delta(previous_best, psi, eligible)
    return {**result, "potential": psi, "safe_end": torch.full_like(psi, safe_end)}


def v131_physical_transition(
    previous_side: torch.Tensor,
    previous_riser: torch.Tensor,
    completed_side: torch.Tensor,
    observed_riser: torch.Tensor,
    distinct_completion: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Separate physical side switching from adjacent spatial ordering."""
    valid_previous = (previous_side >= 0) & (previous_riser > 0)
    side_switch = distinct_completion & valid_previous & (completed_side != previous_side)
    adjacent = distinct_completion & valid_previous & (observed_riser == previous_riser + 1)
    strict = side_switch & adjacent
    return {
        "side_switch": side_switch,
        "adjacent": adjacent,
        "strict": strict,
        "skip_given_switch": side_switch & (observed_riser > previous_riser + 1),
    }


def v131_teaching_completion_events(
    seeded: torch.Tensor, expected_side: torch.Tensor, completed_side: torch.Tensor,
    physical_completion: torch.Tensor, simultaneous: torch.Tensor, target_matches: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Local style settlement deliberately independent of strict validity."""
    unique = physical_completion & ~simultaneous & (completed_side >= 0)
    valid = unique & target_matches
    return {
        "compliant": seeded & valid & (completed_side == expected_side),
        "wrong_side": seeded & valid & (completed_side != expected_side),
        "wrong_target": unique & ~target_matches,
    }


def v131_one_shot_event(
    latched: torch.Tensor, candidate: torch.Tensor, clear: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Emit a candidate once per locked target and clear only on target advance."""
    event = candidate & ~latched
    next_latched = (latched | candidate) & ~clear
    return {"event": event, "latched": next_latched}


def v131_native_reward_scale(strict_chain_valid: torch.Tensor) -> torch.Tensor:
    """Contract helper: strict accounting never scales native task rewards."""
    return torch.ones_like(strict_chain_valid, dtype=torch.float)


def v131_update_teaching_ledger(
    target_world_xy: torch.Tensor,
    expected_side: torch.Tensor,
    seeded: torch.Tensor,
    resync_pending: torch.Tensor,
    reference_world_xy: torch.Tensor,
    forward_world_xy: torch.Tensor,
    physical_completion: torch.Tensor,
    completed_side: torch.Tensor,
    *,
    step_width: float,
) -> dict[str, torch.Tensor]:
    """Advance local teaching from physical truth without repairing strict state."""
    n = expected_side.shape[0]
    if target_world_xy.shape != (n, 2) or reference_world_xy.shape != (n, 2):
        raise ValueError("V13.1 teaching targets must have shape [num_envs, 2].")
    if forward_world_xy.shape != (n, 2) or any(
        value.shape != (n,)
        for value in (seeded, resync_pending, physical_completion, completed_side)
    ):
        raise ValueError("V13.1 teaching ledger inputs have incompatible shapes.")
    if float(step_width) <= 0.0:
        raise ValueError("V13.1 teaching step width must be positive.")
    completion = physical_completion & (completed_side >= 0) & (completed_side <= 1)
    resync = completion & resync_pending
    next_target = reference_world_xy + float(step_width) * forward_world_xy
    return {
        "target_world_xy": torch.where(completion[:, None], next_target, target_world_xy),
        "expected_side": torch.where(completion, 1 - completed_side, expected_side),
        "seeded": seeded | completion,
        "resync_pending": resync_pending & ~completion,
        "resync_event": resync,
    }


def v131_top_follow_potential(
    wheel_bottom_z: torch.Tensor,
    forward_distance: torch.Tensor,
    support: torch.Tensor,
    top_height: torch.Tensor,
    step_height: torch.Tensor,
    *,
    forward_target: float,
) -> torch.Tensor:
    """Bounded progress of the final follower from its lower tread to the top."""
    if wheel_bottom_z.shape != forward_distance.shape or wheel_bottom_z.shape != support.shape:
        raise ValueError("V13.1 top-follow inputs must be matching [num_envs, 2] tensors.")
    if top_height.shape != wheel_bottom_z.shape[:1] or step_height.shape != top_height.shape:
        raise ValueError("V13.1 top height tensors must have shape [num_envs].")
    if float(forward_target) <= 0.0:
        raise ValueError("V13.1 top-follow forward target must be positive.")
    lower = top_height - step_height.clamp_min(1.0e-6)
    height = ((wheel_bottom_z - lower[:, None]) / step_height[:, None].clamp_min(1.0e-6)).clamp(0.0, 1.0)
    forward = (forward_distance / float(forward_target)).clamp(0.0, 1.0)
    return (0.40 * height + 0.30 * forward + 0.30 * support.float()).clamp(0.0, 1.0)


def v131_violation_settlement(
    classified_violation: torch.Tensor,
    simultaneous_b1_lift: torch.Tensor,
    skip_event: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Partition violations into disjoint one-shot cost buckets."""
    if classified_violation.ndim != 1 or any(
        value.shape != classified_violation.shape
        for value in (simultaneous_b1_lift, skip_event)
    ):
        raise ValueError("V13.1 violation settlement expects matching [num_envs] tensors.")
    b1_simultaneous = simultaneous_b1_lift
    skip = skip_event & ~b1_simultaneous
    nonordered = classified_violation & ~b1_simultaneous & ~skip
    return {
        "nonordered_completion": nonordered,
        "simultaneous_b1_lift": b1_simultaneous,
        "skip": skip,
    }


def _v131_initialize_state(env) -> None:
    if hasattr(env, "_v131_ordered"):
        return
    n, device = env.num_envs, env.device
    env._v131_ordered = {
        "last_episode_length": env.episode_length_buf.clone(),
        "phase": torch.full((n,), APPROACH_B1, device=device, dtype=torch.long),
        "strict_chain_valid": torch.ones(n, device=device, dtype=torch.bool),
        # Compatibility alias for old logs/tests. The strict ledger is the
        # authority and this tensor is synchronized after every update.
        "ordered_chain_valid": torch.ones(n, device=device, dtype=torch.bool),
        "expected_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "first_mover_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "first_mover_event": torch.zeros(n, device=device, dtype=torch.bool),
        "first_mover_candidate_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "first_mover_candidate_steps": torch.zeros(n, device=device, dtype=torch.long),
        "first_mover_tie_steps": torch.zeros(n, device=device, dtype=torch.long),
        "last_completed_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "b1_riser_world_xy": torch.zeros((n, 2), device=device),
        "expected_riser_world_xy": torch.zeros((n, 2), device=device),
        "expected_riser_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "flight_forward_world_xy": torch.zeros((n, 2), device=device),
        "potential": torch.zeros((n, 2), device=device),
        "best_potential": torch.zeros((n, 2), device=device),
        "b1_best_potential": torch.zeros(n, device=device),
        "b1_progress_delta": torch.zeros(n, device=device),
        "b1_progress_active": torch.zeros(n, device=device, dtype=torch.bool),
        "b1_velocity_coupled_reward": torch.zeros(n, device=device),
        "b1_progress_escrow": torch.zeros(n, device=device),
        "b1_velocity_escrow": torch.zeros(n, device=device),
        "b1_progress_payout": torch.zeros(n, device=device),
        "b1_velocity_payout": torch.zeros(n, device=device),
        "b1_escrow_forfeited": torch.zeros(n, device=device),
        "alternating_progress_active": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_seeded": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_target_riser_world_xy": torch.zeros((n, 2), device=device),
        "teaching_target_valid": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_target_lower_z": torch.zeros(n, device=device),
        "teaching_target_upper_z": torch.zeros(n, device=device),
        "teaching_step_height": torch.zeros(n, device=device),
        "teaching_expected_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "teaching_resync_pending": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_resync_event": torch.zeros(n, device=device, dtype=torch.bool),
        "expected_progress_delta": torch.zeros(n, device=device),
        "locked_target_potential": torch.zeros((n, 2), device=device),
        "locked_target_active": torch.zeros(n, device=device, dtype=torch.bool),
        "early_opposite_progress_delta": torch.zeros(n, device=device),
        "handoff_active": torch.zeros(n, device=device, dtype=torch.bool),
        "velocity_escrow": torch.zeros(n, device=device),
        "velocity_escrow_payout": torch.zeros(n, device=device),
        "velocity_escrow_forfeited": torch.zeros(n, device=device),
        "expected_target_support": torch.zeros(n, device=device, dtype=torch.bool),
        "expected_swing_relief_active": torch.zeros(n, device=device, dtype=torch.bool),
        "ordered_target_active": torch.zeros(n, device=device, dtype=torch.bool),
        "ordered_speed_gate": torch.ones(n, device=device),
        "free_velocity_reward": torch.zeros(n, device=device),
        "ordered_speed_reward": torch.zeros(n, device=device),
        "full_velocity_eligible": torch.zeros(n, device=device, dtype=torch.bool),
        "same_side_progress_delta": torch.zeros(n, device=device),
        "simultaneous_progress_delta": torch.zeros(n, device=device),
        "skip_progress_best": torch.zeros(n, device=device),
        "skip_progress_delta": torch.zeros(n, device=device),
        "skip_progress_active": torch.zeros(n, device=device, dtype=torch.bool),
        "physical_last_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "physical_previous_completion_riser": torch.zeros(n, device=device, dtype=torch.long),
        "ordered_event": torch.zeros(n, device=device, dtype=torch.bool),
        "seed_event": torch.zeros(n, device=device, dtype=torch.bool),
        "compliant_event": torch.zeros(n, device=device, dtype=torch.bool),
        "wrong_side_event": torch.zeros(n, device=device, dtype=torch.bool),
        "wrong_target_event": torch.zeros(n, device=device, dtype=torch.bool),
        "simultaneous_b1_lift_event": torch.zeros(n, device=device, dtype=torch.bool),
        "simultaneous_b1_lift_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "skip_event": torch.zeros(n, device=device, dtype=torch.bool),
        "violation_event": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_compliant_event": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_wrong_side_event": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_wrong_target_event": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_skip_event": torch.zeros(n, device=device, dtype=torch.bool),
        "teaching_skip_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "physical_last_riser_index": torch.zeros(n, device=device, dtype=torch.long),
        "ordered_last_riser_index": torch.zeros(n, device=device, dtype=torch.long),
        "physical_minus_ordered_gap": torch.zeros(n, device=device, dtype=torch.long),
        "adjacent_completion_event": torch.zeros(n, device=device, dtype=torch.bool),
        "skip_completion_event": torch.zeros(n, device=device, dtype=torch.bool),
        "episode_skip_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "episode_same_side_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "episode_simultaneous_latched": torch.zeros(n, device=device, dtype=torch.bool),
        "strict_task_complete": torch.zeros(n, device=device, dtype=torch.bool),
        "strict_required_risers": torch.zeros(n, device=device, dtype=torch.long),
        "last_completed_riser_world_xy": torch.zeros((n, 2), device=device),
        "last_completed_upper_z": torch.zeros(n, device=device),
        "last_step_height": torch.zeros(n, device=device),
        "top_no_riser_steps": torch.zeros(n, device=device, dtype=torch.long),
        "top_support": torch.zeros((n, 2), device=device, dtype=torch.bool),
        "top_follower_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "top_follower_support_steps": torch.zeros(n, device=device, dtype=torch.long),
        "top_trailing_support_steps": torch.zeros(n, device=device, dtype=torch.long),
        "top_follow_age_steps": torch.zeros(n, device=device, dtype=torch.long),
        "top_follow_potential": torch.zeros(n, device=device),
        "top_follow_best": torch.zeros(n, device=device),
        "top_follow_progress_delta": torch.zeros(n, device=device),
        "top_lead_event": torch.zeros(n, device=device, dtype=torch.bool),
        "top_follow_event": torch.zeros(n, device=device, dtype=torch.bool),
        "post_top_lift": torch.zeros(n, device=device),
        # Global diagnostic ledger.  These zero-dimensional tensors deliberately
        # survive individual environment resets.
        "encounter_count": torch.zeros((), device=device, dtype=torch.long),
        "b1_completion_count": torch.zeros((), device=device, dtype=torch.long),
        "ordered_completion_count": torch.zeros((), device=device, dtype=torch.long),
        "ordered_alternating_count": torch.zeros((), device=device, dtype=torch.long),
        "physical_completion_count": torch.zeros((), device=device, dtype=torch.long),
        "adjacent_completion_count": torch.zeros((), device=device, dtype=torch.long),
        "skip_completion_count": torch.zeros((), device=device, dtype=torch.long),
        "wrong_side_count": torch.zeros((), device=device, dtype=torch.long),
        "simultaneous_b1_lift_count": torch.zeros((), device=device, dtype=torch.long),
        "skip_count": torch.zeros((), device=device, dtype=torch.long),
        "jump_sum": torch.zeros((), device=device),
        "jump_count": torch.zeros((), device=device, dtype=torch.long),
        "top_follow_entry_count": torch.zeros((), device=device, dtype=torch.long),
        "top_follow_success_count": torch.zeros((), device=device, dtype=torch.long),
        "top_follow_delay_sum": torch.zeros((), device=device),
        "post_top_lift_sum": torch.zeros((), device=device),
        "post_top_sample_count": torch.zeros((), device=device, dtype=torch.long),
        "seed_fl_count": torch.zeros((), device=device, dtype=torch.long),
        "seed_fr_count": torch.zeros((), device=device, dtype=torch.long),
        "first_mover_count": torch.zeros((), device=device, dtype=torch.long),
        "b1_escrow_paid_count": torch.zeros((), device=device, dtype=torch.long),
        "b1_escrow_forfeit_count": torch.zeros((), device=device, dtype=torch.long),
        "velocity_escrow_paid_count": torch.zeros((), device=device, dtype=torch.long),
        "velocity_escrow_forfeit_count": torch.zeros((), device=device, dtype=torch.long),
        "teaching_resync_count": torch.zeros((), device=device, dtype=torch.long),
        "teaching_target_count": torch.zeros((), device=device, dtype=torch.long),
        "physical_side_switch_count": torch.zeros((), device=device, dtype=torch.long),
        "physical_transition_count": torch.zeros((), device=device, dtype=torch.long),
        "physical_adjacent_count": torch.zeros((), device=device, dtype=torch.long),
        "strict_physical_transition_count": torch.zeros((), device=device, dtype=torch.long),
        "skip_given_switch_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_side_switch_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_transition_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_encounter_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_teaching_target_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_adjacent_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_strict_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_skip_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_skip_given_switch_count": torch.zeros((), device=device, dtype=torch.long),
        "rollout_window_id": torch.full((), -1, device=device, dtype=torch.long),
    }
    env._v131_ordered_last_step = -1


def _v131_front_geometry(env, state):
    """Return front-wheel world geometry and support on the locked top tread."""
    asset = env.scene["robot"]
    wheel_world = asset.data.body_pos_w[:, env._v11_front_body_ids]
    forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids]
    forward = state["flight_forward_world_xy"]
    delta = wheel_world[..., :2] - state["last_completed_riser_world_xy"][:, None, :]
    forward_distance = (delta * forward[:, None, :]).sum(dim=2)
    wheel_bottom = wheel_world[..., 2] - float(env.cfg.v11_perception.wheel_radius)
    force_norm = forces.norm(dim=-1)
    top_support = (
        (forward_distance >= float(env.cfg.v11_perception.wheel_radius) + float(env.cfg.v11_perception.front_margin))
        & ((wheel_bottom - state["last_completed_upper_z"][:, None]).abs()
           <= float(env.cfg.v11_perception.upper_support_tolerance))
        & (force_norm >= float(env.cfg.v11_perception.contact_force_threshold))
        & (forces[..., 2] > float(env.cfg.v11_perception.contact_force_threshold))
    )
    return wheel_world, wheel_bottom, forward_distance, top_support


def v131_ordered_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Update the ordered B1/B2/.../top reward state exactly once per step."""
    from .v11_continuous import v11_state
    from .v13_alternation import (
        v13_historical_best_progress,
        v13_split_alternation_progress,
    )

    _v131_initialize_state(env)
    step = int(getattr(env, "common_step_counter", 0))
    if env._v131_ordered_last_step == step:
        return env._v131_ordered

    state = env._v131_ordered
    reset = env.episode_length_buf < state["last_episode_length"]
    per_env_reset = {
        "phase": APPROACH_B1,
        "strict_chain_valid": True,
        "ordered_chain_valid": True,
        "expected_side": -1,
        "first_mover_latched": False,
        "first_mover_candidate_side": -1,
        "first_mover_candidate_steps": 0,
        "first_mover_tie_steps": 0,
        "last_completed_side": -1,
        "b1_riser_world_xy": 0.0,
        "expected_riser_world_xy": 0.0,
        "expected_riser_valid": False,
        "flight_forward_world_xy": 0.0,
        "potential": 0.0,
        "best_potential": 0.0,
        "locked_target_potential": 0.0,
        "b1_best_potential": 0.0,
        "b1_progress_active": False,
        "b1_velocity_coupled_reward": 0.0,
        "b1_progress_escrow": 0.0,
        "b1_velocity_escrow": 0.0,
        "alternating_progress_active": False,
        "teaching_seeded": False,
        "teaching_target_riser_world_xy": 0.0,
        "teaching_target_valid": False,
        "teaching_target_lower_z": 0.0,
        "teaching_target_upper_z": 0.0,
        "teaching_step_height": 0.0,
        "teaching_expected_side": -1,
        "teaching_resync_pending": False,
        "ordered_target_active": False,
        "locked_target_active": False,
        "handoff_active": False,
        "velocity_escrow": 0.0,
        "expected_target_support": False,
        "expected_swing_relief_active": False,
        "ordered_speed_gate": 1.0,
        "free_velocity_reward": 0.0,
        "ordered_speed_reward": 0.0,
        "full_velocity_eligible": False,
        "skip_progress_best": 0.0,
        "physical_last_side": -1,
        "teaching_skip_latched": False,
        "physical_previous_completion_riser": 0,
        "simultaneous_b1_lift_latched": False,
        "physical_last_riser_index": 0,
        "ordered_last_riser_index": 0,
        "physical_minus_ordered_gap": 0,
        "episode_skip_latched": False,
        "episode_same_side_latched": False,
        "episode_simultaneous_latched": False,
        "strict_task_complete": False,
        "strict_required_risers": 0,
        "last_completed_riser_world_xy": 0.0,
        "last_completed_upper_z": 0.0,
        "last_step_height": 0.0,
        "top_no_riser_steps": 0,
        "top_support": False,
        "top_follower_side": -1,
        "top_follower_support_steps": 0,
        "top_trailing_support_steps": 0,
        "top_follow_age_steps": 0,
        "top_follow_potential": 0.0,
        "top_follow_best": 0.0,
        "post_top_lift": 0.0,
    }
    for name, value in per_env_reset.items():
        state[name][reset] = value
    state["last_episode_length"] = env.episode_length_buf.clone()
    for name in (
        "b1_progress_delta", "b1_velocity_coupled_reward", "b1_progress_payout",
        "b1_velocity_payout", "b1_escrow_forfeited", "expected_progress_delta",
        "early_opposite_progress_delta", "velocity_escrow_payout", "velocity_escrow_forfeited",
        "same_side_progress_delta", "simultaneous_progress_delta",
        "top_follow_progress_delta", "post_top_lift", "skip_progress_delta",
    ):
        state[name].zero_()

    rollout_window = step // 24
    if int(state["rollout_window_id"].item()) != rollout_window:
        for name in (
            "rollout_side_switch_count", "rollout_transition_count", "rollout_encounter_count",
            "rollout_teaching_target_count", "rollout_adjacent_count",
            "rollout_strict_count", "rollout_skip_count", "rollout_skip_given_switch_count",
        ):
            state[name].zero_()
        state["rollout_window_id"].fill_(rollout_window)
    for name in (
        "ordered_event", "seed_event", "compliant_event", "wrong_side_event",
        "wrong_target_event", "simultaneous_b1_lift_event", "skip_event", "violation_event",
        "first_mover_event",
        "adjacent_completion_event", "skip_completion_event", "teaching_resync_event",
        "teaching_compliant_event", "teaching_wrong_side_event",
        "teaching_wrong_target_event", "teaching_skip_event",
        "top_lead_event", "top_follow_event",
    ):
        state[name].zero_()

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    landing = base["landing"]
    cfg = env.cfg.v131_ordering
    reference = landing["reference_riser_world_xy"]
    reference_active = landing["target_active"]

    # Lock B1 at the first live ascending target.  If the second face is visible,
    # its direction is a better staircase axis than the robot's yaw; otherwise
    # use the current yaw heading as the fail-safe axis.
    asset = env.scene["robot"]
    root = asset.data.root_pos_w
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    risers = base["risers"]
    raw_positive = base["valid"] & (risers["riser_height"] > 0.0)
    local_xy = torch.stack((risers["riser_x"], risers["riser_y"]), dim=2)
    c, s = torch.cos(yaw), torch.sin(yaw)
    raw_world = torch.stack((
        root[:, None, 0] + c[:, None] * local_xy[..., 0] - s[:, None] * local_xy[..., 1],
        root[:, None, 1] + s[:, None] * local_xy[..., 0] + c[:, None] * local_xy[..., 1],
    ), dim=2)
    raw_axis = raw_world[:, 1] - raw_world[:, 0]
    raw_axis_valid = raw_positive.all(dim=1) & (raw_axis.norm(dim=1) > float(cfg.target_match_tolerance))
    raw_axis = raw_axis / raw_axis.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    flight_axis = torch.where(raw_axis_valid[:, None], raw_axis, heading)
    lock_b1 = (
        (state["phase"] == APPROACH_B1) & ~state["expected_riser_valid"] & reference_active
    )
    state["expected_riser_world_xy"] = torch.where(
        lock_b1[:, None], reference, state["expected_riser_world_xy"],
    )
    state["b1_riser_world_xy"] = torch.where(
        lock_b1[:, None], reference, state["b1_riser_world_xy"],
    )
    state["flight_forward_world_xy"] = torch.where(
        lock_b1[:, None], flight_axis, state["flight_forward_world_xy"],
    )
    state["expected_riser_valid"] |= lock_b1
    state["teaching_target_riser_world_xy"] = torch.where(
        lock_b1[:, None], reference, state["teaching_target_riser_world_xy"],
    )
    state["teaching_target_valid"] |= lock_b1
    chosen_slot = landing["target_slot"].clamp_min(0)
    observed_step_height = risers["riser_height"].gather(1, chosen_slot[:, None]).squeeze(1).abs()
    observed_target_upper_z = landing["target_upper_z"]
    observed_target_lower_z = observed_target_upper_z - observed_step_height
    state["teaching_target_lower_z"] = torch.where(
        lock_b1, observed_target_lower_z, state["teaching_target_lower_z"],
    )
    state["teaching_target_upper_z"] = torch.where(
        lock_b1, observed_target_upper_z, state["teaching_target_upper_z"],
    )
    state["teaching_step_height"] = torch.where(
        lock_b1, observed_step_height, state["teaching_step_height"],
    )
    state["encounter_count"] += lock_b1.sum()
    state["rollout_encounter_count"] += lock_b1.sum()
    state["teaching_target_count"] += lock_b1.sum()
    state["rollout_teaching_target_count"] += lock_b1.sum()

    strict_relation = v131_target_relation(
        reference, state["expected_riser_world_xy"], state["flight_forward_world_xy"],
        state["expected_riser_valid"] & reference_active,
        longitudinal_tolerance=float(cfg.target_match_tolerance),
        lateral_tolerance=float(cfg.target_lateral_tolerance),
    )
    teaching_relation = v131_target_relation(
        reference, state["teaching_target_riser_world_xy"], state["flight_forward_world_xy"],
        state["teaching_target_valid"] & reference_active,
        longitudinal_tolerance=float(cfg.target_match_tolerance),
        lateral_tolerance=float(cfg.target_lateral_tolerance),
    )
    phase_before = state["phase"].clone()

    # Calculate progress against the locked physical B(n+1), not the scanner's
    # current slot.  This keeps the teaching channel alive through target
    # flicker and while the live detector is already looking farther ahead.
    wheel_world = asset.data.body_pos_w[:, env._v11_front_body_ids]
    forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids]
    wheel_bottom = wheel_world[..., 2] - float(env.cfg.v11_perception.wheel_radius)
    teaching_forward_distance = (
        (wheel_world[..., :2] - state["teaching_target_riser_world_xy"][:, None, :])
        * state["flight_forward_world_xy"][:, None, :]
    ).sum(dim=2)
    force_norm = forces.norm(dim=-1)
    locked_support = (
        (teaching_forward_distance >= float(env.cfg.v11_perception.wheel_radius)
         + float(env.cfg.v11_perception.front_margin))
        & ((wheel_bottom - state["teaching_target_upper_z"][:, None]).abs()
           <= float(env.cfg.v11_perception.upper_support_tolerance))
        & (force_norm >= float(env.cfg.v11_perception.contact_force_threshold))
        & (forces[..., 2] > float(env.cfg.v11_perception.contact_force_threshold))
    )
    locked_target_active = state["teaching_target_valid"] & (
        (phase_before == APPROACH_B1) | (phase_before == ALTERNATING_RISERS)
    )
    locked_potential = v131_locked_target_potential(
        wheel_bottom,
        teaching_forward_distance,
        locked_support,
        state["teaching_target_lower_z"],
        state["teaching_target_upper_z"],
        clearance_margin=float(env.cfg.v11_perception.clearance_margin),
        approach_span=float(cfg.step_width),
        crossing_span=float(env.cfg.v11_perception.wheel_radius)
        + float(env.cfg.v11_perception.front_margin),
    )
    locked_potential *= locked_target_active[:, None].float()
    state["locked_target_potential"] = locked_potential
    state["locked_target_active"] = locked_target_active

    # B1 is side-neutral only until a target-qualified first mover is visible.
    # A near-tie pays a one-shot style cost and remains unassigned; a clear
    # leader owns B1, so the other side cannot steal the seed completion.
    unassigned_b1 = (
        (phase_before == APPROACH_B1) & (state["expected_side"] < 0)
        & state["teaching_target_valid"]
    )
    first_mover_confirmation = v131_confirm_b1_first_mover(
        state["first_mover_candidate_side"],
        state["first_mover_candidate_steps"],
        state["first_mover_tie_steps"],
        locked_potential,
        unassigned_b1,
        onset_threshold=float(cfg.first_mover_potential_threshold),
        release_threshold=float(cfg.first_mover_release_threshold),
        separation=float(cfg.first_mover_separation),
        confirmation_frames=int(cfg.first_mover_confirmation_frames),
    )
    state["first_mover_candidate_side"] = first_mover_confirmation["candidate_side"]
    state["first_mover_candidate_steps"] = first_mover_confirmation["candidate_steps"]
    state["first_mover_tie_steps"] = first_mover_confirmation["tie_steps"]
    lift_ready = first_mover_confirmation["lift_ready"]
    lift_ready = torch.where(
        first_mover_confirmation["simultaneous_ready"][:, None],
        torch.ones_like(lift_ready), lift_ready,
    )
    first_mover = v131_latch_b1_leader(
        phase_before, state["expected_side"], lift_ready,
    )
    state["expected_side"] = first_mover["expected_side"]
    state["teaching_expected_side"] = torch.where(
        first_mover["unique_lift_event"], state["expected_side"], state["teaching_expected_side"],
    )
    state["first_mover_event"] = first_mover["unique_lift_event"]
    state["first_mover_latched"] |= first_mover["unique_lift_event"]
    state["first_mover_count"] += first_mover["unique_lift_event"].sum()
    simultaneous_b1_lift = (
        first_mover["simultaneous_lift_event"] & ~state["simultaneous_b1_lift_latched"]
    )
    state["simultaneous_b1_lift_event"] = simultaneous_b1_lift
    state["simultaneous_b1_lift_latched"] |= first_mover["simultaneous_lift_event"]
    state["simultaneous_b1_lift_count"] += simultaneous_b1_lift.sum()

    physical_event = landing["landing_event"]
    completed_side = landing["responsible"]
    simultaneous = physical_event & landing["candidate_ready"].all(dim=1)
    classified = v131_classify_completion(
        phase_before, state["expected_side"], completed_side, physical_event,
        simultaneous, strict_relation["matches"],
    )
    strict_before = state["strict_chain_valid"].clone()
    # Once the ordered sequence has failed, later physical contacts may still
    # earn native locomotion credit but cannot silently restart this ledger.
    for name in (
        "ordered_event", "seed_event", "compliant_event", "wrong_side_event",
        "wrong_target_event", "violation_event",
    ):
        classified[name] &= strict_before

    teaching_unique = physical_event & ~simultaneous & (completed_side >= 0)
    teaching_valid_completion = teaching_unique & teaching_relation["matches"]
    teaching_target_advance = teaching_unique & reference_active
    teaching_started = state["teaching_seeded"]
    teaching_events = v131_teaching_completion_events(
        teaching_started, state["teaching_expected_side"], completed_side,
        physical_event, simultaneous, teaching_relation["matches"],
    )
    state["teaching_compliant_event"] = teaching_events["compliant"]
    state["teaching_wrong_side_event"] = teaching_events["wrong_side"]
    state["teaching_wrong_target_event"] = (
        teaching_events["wrong_target"] & state["teaching_target_valid"]
    )
    teaching_overshot = teaching_forward_distance.amax(dim=1) > (
        float(cfg.step_width) + float(env.cfg.v11_perception.wheel_radius)
    )
    teaching_skip_candidate = state["teaching_target_valid"] & (
        (teaching_overshot & ~teaching_valid_completion)
        | (
            state["teaching_wrong_target_event"]
            & (teaching_relation["longitudinal_error"] > float(cfg.target_match_tolerance))
        )
    )
    teaching_skip = v131_one_shot_event(
        state["teaching_skip_latched"], teaching_skip_candidate, teaching_target_advance,
    )
    state["teaching_skip_event"] = teaching_skip["event"]
    state["teaching_skip_latched"] = teaching_skip["latched"]
    expected_delta = wheel_world[..., :2] - state["expected_riser_world_xy"][:, None, :]
    expected_forward_distance = (
        expected_delta * state["flight_forward_world_xy"][:, None, :]
    ).sum(dim=2)
    overshot = expected_forward_distance.amax(dim=1) > (
        float(cfg.step_width) + float(env.cfg.v11_perception.wheel_radius)
    )
    skip_without_completion = (
        ((phase_before == APPROACH_B1) | (phase_before == ALTERNATING_RISERS))
        & state["expected_riser_valid"] & overshot & ~classified["ordered_event"]
    )
    wrong_target_jump = classified["wrong_target_event"] & (
        strict_relation["longitudinal_error"] > float(cfg.target_match_tolerance)
    )
    skip_event = strict_before & (skip_without_completion | wrong_target_jump)
    chain_failure = classified["violation_event"] | skip_event | simultaneous_b1_lift
    violation = chain_failure
    state["strict_chain_valid"] &= ~chain_failure
    state["ordered_chain_valid"] = state["strict_chain_valid"].clone()
    state["episode_skip_latched"] |= skip_event
    state["episode_same_side_latched"] |= classified["wrong_side_event"]
    state["episode_simultaneous_latched"] |= simultaneous | simultaneous_b1_lift
    state["teaching_resync_pending"] |= chain_failure

    # B1 remains free until first-mover onset, then pays only the committed
    # side. B2+ is an always-live support-to-swing handoff against the locked
    # adjacent target. Both use historical-best increments to prevent rocking.
    b1_actionable = (
        ~state["teaching_seeded"]
        & state["teaching_target_valid"]
        & (phase_before == APPROACH_B1)
        & ~violation
    )
    previous_best = state["best_potential"]
    all_progress_active = b1_actionable | (
        state["teaching_seeded"] & state["teaching_target_valid"]
        & ((phase_before == ALTERNATING_RISERS) | ~state["strict_chain_valid"])
        & ~violation
    )
    history = v13_historical_best_progress(
        previous_best, locked_potential, all_progress_active,
    )
    per_side_delta = history["delta"]
    b1_overlap = per_side_delta.amin(dim=1)
    b1_neutral = (per_side_delta.amax(dim=1) - b1_overlap) * (
        b1_actionable & (state["expected_side"] < 0)
    ).float()
    b1_committed = b1_actionable & (state["expected_side"] >= 0)
    b1_split = v13_split_alternation_progress(
        per_side_delta, state["expected_side"], b1_committed,
    )
    teaching_scale = torch.ones_like(state["b1_best_potential"]) * float(cfg.strict_invalid_teaching_scale)
    state["b1_progress_delta"] = (b1_neutral + b1_split["opposite"]) * teaching_scale
    state["b1_progress_active"] = b1_actionable
    state["b1_best_potential"] = history["best"].amax(dim=1)

    progress_eligible = (
        state["teaching_seeded"]
        & state["teaching_target_valid"]
        & ~violation
        & ((phase_before == ALTERNATING_RISERS) | ~state["strict_chain_valid"])
    )
    progress = v13_split_alternation_progress(
        history["delta"], state["teaching_expected_side"], progress_eligible,
    )
    state["best_potential"] = history["best"]
    state["expected_progress_delta"] = progress["opposite"] * teaching_scale
    state["early_opposite_progress_delta"] = b1_split["same"] * teaching_scale
    state["same_side_progress_delta"] = (
        progress["same"] + b1_split["same"]
    ) * teaching_scale
    state["simultaneous_progress_delta"] = (
        progress["simultaneous"] + b1_split["simultaneous"]
        + b1_overlap * (b1_actionable & (state["expected_side"] < 0)).float()
    ) * teaching_scale
    state["alternating_progress_active"] = progress_eligible
    state["handoff_active"] = progress_eligible
    state["potential"] = locked_potential

    # Convert V11's strong velocity quality into a bounded per-riser style
    # dividend without scaling or withholding any native reward. Partial
    # progress pays only after the expected opposite leg completes the locked
    # adjacent target; wrong-side/simultaneous/skip behavior forfeits it.
    from .rewards import track_lin_vel_xy_exp

    velocity_quality = track_lin_vel_xy_exp(
        env, std=float(cfg.velocity_quality_std), command_name="base_velocity",
    ).clamp(0.0, 1.0)
    escrow_cancel = (
        state["teaching_wrong_side_event"]
        | state["teaching_wrong_target_event"]
        | state["teaching_skip_event"]
        | simultaneous
    )
    escrow = v131_settle_progress_escrow(
        state["velocity_escrow"],
        velocity_quality * state["expected_progress_delta"],
        progress_eligible,
        state["teaching_compliant_event"],
        escrow_cancel,
    )
    state["velocity_escrow"] = escrow["escrow"]
    state["velocity_escrow_payout"] = escrow["payout"]
    state["velocity_escrow_forfeited"] = escrow["forfeited"]
    state["velocity_escrow_paid_count"] += (escrow["payout"] > 0.0).sum()
    state["velocity_escrow_forfeit_count"] += (escrow["forfeited"] > 0.0).sum()

    teaching_side = state["teaching_expected_side"].clamp(0, 1)
    expected_potential = locked_potential.gather(1, teaching_side[:, None]).squeeze(1)
    b2_actionable = progress_eligible
    actionable = b1_actionable | b2_actionable
    b1_side = state["expected_side"].clamp(0, 1)
    b1_committed_potential = locked_potential.gather(1, b1_side[:, None]).squeeze(1)
    b1_gate = torch.where(
        state["expected_side"] >= 0, b1_committed_potential, locked_potential.amax(dim=1),
    )
    positive_gate = torch.where(b1_actionable, b1_gate, expected_potential).clamp(0.0, 1.0)
    state["ordered_speed_gate"] = torch.where(
        actionable, positive_gate, torch.ones_like(positive_gate),
    )
    state["ordered_target_active"] = actionable

    # Ease only the expected front swing leg while it is advancing toward the
    # locked adjacent tread.  The support leg keeps native V13 regularization,
    # and all relief ends as soon as the expected wheel has stable target
    # support.  This changes neither action scale nor physical joint limits.
    relief_side = torch.where(
        state["teaching_seeded"], state["teaching_expected_side"], state["expected_side"],
    )
    expected_support = locked_support.gather(1, relief_side.clamp(0, 1)[:, None]).squeeze(1)
    state["expected_target_support"] = expected_support & actionable & (relief_side >= 0)
    state["expected_swing_relief_active"] = (
        actionable & (relief_side >= 0) & ~state["expected_target_support"]
    )

    # Dense spatial warning uses the locked teaching target, never the scanner's
    # potentially advanced live reference. It is side-neutral and telescoping.
    valid_adjacent_completion = teaching_valid_completion
    skip_progress = v131_skip_progress_delta(
        state["skip_progress_best"], teaching_forward_distance,
        state["teaching_target_valid"] & locked_target_active,
        valid_adjacent_completion,
        step_width=float(cfg.step_width),
        wheel_radius=float(env.cfg.v11_perception.wheel_radius),
        front_margin=float(env.cfg.v11_perception.front_margin),
    )
    state["skip_progress_best"] = skip_progress["best"]
    state["skip_progress_delta"] = skip_progress["delta"]
    state["skip_progress_active"] = (
        state["teaching_target_valid"] & locked_target_active & ~valid_adjacent_completion
    )

    for name in (
        "ordered_event", "seed_event", "compliant_event", "wrong_side_event", "wrong_target_event",
    ):
        state[name] = classified[name]
    state["skip_event"] = skip_event
    state["violation_event"] = violation
    state["physical_completion_count"] += physical_event.sum()
    state["wrong_side_count"] += classified["wrong_side_event"].sum()
    state["skip_count"] += state["teaching_skip_event"].sum()
    state["rollout_skip_count"] += state["teaching_skip_event"].sum()
    jump = strict_relation["longitudinal_error"] / float(cfg.step_width)
    state["jump_sum"] += (jump * classified["wrong_target_event"].float()).sum()
    state["jump_count"] += classified["wrong_target_event"].sum()

    relative_riser = (
        (reference - state["b1_riser_world_xy"]) * state["flight_forward_world_xy"]
    ).sum(dim=1)
    observed_riser_index = (
        torch.round(relative_riser / float(cfg.step_width)).long() + 1
    ).clamp_min(1)
    distinct_physical = physical_event & ~simultaneous & (completed_side >= 0)
    transition = v131_physical_transition(
        state["physical_last_side"], state["physical_previous_completion_riser"],
        completed_side, observed_riser_index, distinct_physical,
    )
    state["physical_transition_count"] += (distinct_physical & (state["physical_last_side"] >= 0)).sum()
    state["physical_side_switch_count"] += transition["side_switch"].sum()
    state["physical_adjacent_count"] += transition["adjacent"].sum()
    state["strict_physical_transition_count"] += transition["strict"].sum()
    state["skip_given_switch_count"] += transition["skip_given_switch"].sum()
    state["rollout_transition_count"] += (distinct_physical & (state["physical_last_side"] >= 0)).sum()
    state["rollout_side_switch_count"] += transition["side_switch"].sum()
    state["rollout_adjacent_count"] += transition["adjacent"].sum()
    state["rollout_strict_count"] += transition["strict"].sum()
    state["rollout_skip_given_switch_count"] += transition["skip_given_switch"].sum()
    state["physical_last_side"] = torch.where(
        distinct_physical, completed_side, state["physical_last_side"],
    )
    state["physical_previous_completion_riser"] = torch.where(
        distinct_physical, observed_riser_index, state["physical_previous_completion_riser"],
    )
    ledgers = v131_update_riser_ledgers(
        state["physical_last_riser_index"],
        state["ordered_last_riser_index"],
        observed_riser_index,
        physical_event,
        classified["ordered_event"],
    )
    for name in (
        "physical_last_riser_index",
        "ordered_last_riser_index",
        "physical_minus_ordered_gap",
    ):
        state[name] = ledgers[name]
    state["adjacent_completion_event"] = ledgers["adjacent_ordered_completion"]
    state["skip_completion_event"] = ledgers["skip_completion"]

    ordered = ledgers["adjacent_ordered_completion"]
    seed = classified["seed_event"] & ordered
    later = classified["compliant_event"] & ordered
    state["ordered_event"] = ordered
    state["seed_event"] = seed
    state["compliant_event"] = later
    state["b1_completion_count"] += seed.sum()
    state["ordered_completion_count"] += ordered.sum()
    state["ordered_alternating_count"] += later.sum()
    state["adjacent_completion_count"] += state["adjacent_completion_event"].sum()
    state["skip_completion_count"] += state["skip_completion_event"].sum()
    state["seed_fl_count"] += (seed & (completed_side == 0)).sum()
    state["seed_fr_count"] += (seed & (completed_side == 1)).sum()

    # B1 shaping is deliberately escrowed: approach/lift potential is useful
    # exploration guidance, but PPO observes it only if that same motion ends
    # in a stable, target-qualified B1 completion. Passing B1, skipping, or a
    # simultaneous/wrong-side settlement burns the unpaid balance.
    b1_cancel = (
        state["teaching_wrong_side_event"]
        | state["teaching_wrong_target_event"]
        | state["teaching_skip_event"]
        | simultaneous_b1_lift
        | simultaneous
        | skip_event
    )
    b1_progress_settlement = v131_settle_progress_escrow(
        state["b1_progress_escrow"], state["b1_progress_delta"],
        b1_actionable, seed, b1_cancel,
    )
    b1_velocity_settlement = v131_settle_progress_escrow(
        state["b1_velocity_escrow"], velocity_quality * state["b1_progress_delta"],
        b1_actionable, seed, b1_cancel,
    )
    state["b1_progress_escrow"] = b1_progress_settlement["escrow"]
    state["b1_velocity_escrow"] = b1_velocity_settlement["escrow"]
    state["b1_progress_payout"] = b1_progress_settlement["payout"]
    state["b1_velocity_payout"] = b1_velocity_settlement["payout"]
    state["b1_velocity_coupled_reward"] = state["b1_velocity_payout"] / float(env.step_dt)
    state["b1_escrow_forfeited"] = torch.maximum(
        b1_progress_settlement["forfeited"], b1_velocity_settlement["forfeited"],
    )
    state["b1_escrow_paid_count"] += (
        (state["b1_progress_payout"] > 0.0) | (state["b1_velocity_payout"] > 0.0)
    ).sum()
    state["b1_escrow_forfeit_count"] += (state["b1_escrow_forfeited"] > 0.0).sum()
    state["expected_side"] = torch.where(
        seed, 1 - completed_side,
        torch.where(later, 1 - state["expected_side"], state["expected_side"]),
    )
    state["last_completed_side"] = torch.where(ordered, completed_side, state["last_completed_side"])
    state["last_completed_riser_world_xy"] = torch.where(
        ordered[:, None], state["expected_riser_world_xy"], state["last_completed_riser_world_xy"],
    )
    state["last_completed_upper_z"] = torch.where(
        ordered, landing["target_upper_z"], state["last_completed_upper_z"],
    )
    chosen = landing["target_slot"].clamp_min(0)
    step_height = risers["riser_height"].gather(1, chosen[:, None]).squeeze(1).abs()
    state["last_step_height"] = torch.where(ordered, step_height, state["last_step_height"])
    state["expected_riser_world_xy"] = torch.where(
        ordered[:, None],
        state["expected_riser_world_xy"] + float(cfg.step_width) * state["flight_forward_world_xy"],
        state["expected_riser_world_xy"],
    )
    state["phase"] = torch.where(seed, torch.full_like(state["phase"], ALTERNATING_RISERS), state["phase"])

    # The teaching ledger follows physical truth rather than claiming strict
    # success. Any unique stable physical landing defines the next adjacent
    # target and its opposite leg. This is the only resynchronization path.
    teaching_completion = teaching_target_advance
    state["teaching_target_count"] += teaching_completion.sum()
    state["rollout_teaching_target_count"] += teaching_completion.sum()
    teaching = v131_update_teaching_ledger(
        state["teaching_target_riser_world_xy"],
        state["teaching_expected_side"],
        state["teaching_seeded"],
        state["teaching_resync_pending"],
        reference,
        state["flight_forward_world_xy"],
        teaching_completion,
        completed_side,
        step_width=float(cfg.step_width),
    )
    teaching_resync = teaching["resync_event"]
    state["teaching_resync_event"] = teaching_resync
    state["teaching_resync_count"] += teaching_resync.sum()
    state["teaching_seeded"] = teaching["seeded"]
    state["teaching_expected_side"] = teaching["expected_side"]
    state["teaching_target_riser_world_xy"] = teaching["target_world_xy"]
    state["teaching_target_valid"] |= teaching_completion
    # The next locked face is exactly one tread and one observed step height
    # beyond the physical completion.  This geometry stays usable even while
    # the two-slot scanner is looking at a different face.
    state["teaching_target_lower_z"] = torch.where(
        teaching_completion, observed_target_upper_z, state["teaching_target_lower_z"],
    )
    state["teaching_target_upper_z"] = torch.where(
        teaching_completion,
        observed_target_upper_z + observed_step_height,
        state["teaching_target_upper_z"],
    )
    state["teaching_step_height"] = torch.where(
        teaching_completion, observed_step_height, state["teaching_step_height"],
    )
    state["teaching_resync_pending"] = teaching["resync_pending"]
    state["potential"][teaching_completion | chain_failure] = 0.0
    state["best_potential"][teaching_completion | chain_failure] = 0.0
    state["b1_best_potential"][teaching_completion] = 0.0
    state["skip_progress_best"][teaching_completion] = 0.0

    # Once the last correctly completed leg is supported on a top tread and no
    # adjacent higher face remains visible, switch to an explicit other-leg-
    # follow phase.  Matching the predicted B(n+1), rather than waiting for the
    # just-passed Bn face to disappear behind the scanner, makes this transition
    # early enough to shape the follower instead of observing it afterwards.
    wheel_world, wheel_bottom, forward_distance, top_support = _v131_front_geometry(env, state)
    next_delta = raw_world - state["expected_riser_world_xy"][:, None, :]
    next_forward_error = (next_delta * state["flight_forward_world_xy"][:, None, :]).sum(dim=2)
    next_lateral_axis = torch.stack(
        (-state["flight_forward_world_xy"][:, 1], state["flight_forward_world_xy"][:, 0]), dim=1,
    )
    next_lateral_error = (next_delta * next_lateral_axis[:, None, :]).sum(dim=2)
    next_riser_visible = (
        raw_positive
        & (next_forward_error.abs() <= float(cfg.target_match_tolerance))
        & (next_lateral_error.abs() <= float(cfg.target_lateral_tolerance))
    ).any(dim=1)
    last_side = state["last_completed_side"].clamp(0, 1)
    last_side_supported = top_support.gather(1, last_side[:, None]).squeeze(1)
    top_candidate = (
        (state["phase"] == ALTERNATING_RISERS) & (state["ordered_last_riser_index"] > 0)
        & state["strict_chain_valid"] & ~next_riser_visible & last_side_supported
    )
    state["top_no_riser_steps"] = torch.where(
        top_candidate, state["top_no_riser_steps"] + 1, torch.zeros_like(state["top_no_riser_steps"]),
    )
    enter_top = top_candidate & (
        state["top_no_riser_steps"] >= int(cfg.top_no_riser_confirmation_frames)
    )
    state["strict_required_risers"] = torch.where(
        enter_top, state["ordered_last_riser_index"], state["strict_required_risers"],
    )
    state["phase"] = torch.where(enter_top, torch.full_like(state["phase"], TOP_FOLLOW), state["phase"])
    state["top_follower_side"] = torch.where(
        enter_top, 1 - state["last_completed_side"], state["top_follower_side"],
    )
    state["top_follow_entry_count"] += enter_top.sum()

    lead_active = state["phase"] == TOP_FOLLOW
    trailing_active = state["phase"] == TOP_TRAILING_FOLLOW
    top_active = lead_active | trailing_active
    state["top_follow_age_steps"] = torch.where(
        top_active, state["top_follow_age_steps"] + 1, state["top_follow_age_steps"],
    )
    platform_forward_target = (
        float(cfg.step_width)
        + float(env.cfg.v11_perception.wheel_radius)
        + float(env.cfg.v11_perception.front_margin)
    )
    platform_support = top_support & (forward_distance >= platform_forward_target)
    lead_side = (1 - state["last_completed_side"]).clamp(0, 1)
    trailing_side = state["last_completed_side"].clamp(0, 1)
    lead_supported = platform_support.gather(1, lead_side[:, None]).squeeze(1)
    trailing_supported = platform_support.gather(1, trailing_side[:, None]).squeeze(1)
    state["top_follower_support_steps"] = torch.where(
        lead_active & lead_supported,
        state["top_follower_support_steps"] + 1,
        torch.zeros_like(state["top_follower_support_steps"]),
    )
    state["top_trailing_support_steps"] = torch.where(
        trailing_active & trailing_supported,
        state["top_trailing_support_steps"] + 1,
        torch.zeros_like(state["top_trailing_support_steps"]),
    )
    support_frames = int(env.cfg.v11_perception.front_support_confirmation_frames)
    platform_transition = v131_platform_phase_transition(
        state["phase"],
        state["last_completed_side"],
        state["top_follower_support_steps"] >= support_frames,
        state["top_trailing_support_steps"] >= support_frames,
    )
    active_side = platform_transition["active_side"]
    top_potential_all = v131_top_follow_potential(
        wheel_bottom, forward_distance, platform_support,
        state["last_completed_upper_z"], state["last_step_height"],
        forward_target=platform_forward_target,
    )
    follower_potential = top_potential_all.gather(1, active_side[:, None]).squeeze(1)
    follower_potential = torch.where(
        top_active, follower_potential, torch.zeros_like(follower_potential),
    )
    state["top_follow_potential"] = follower_potential
    top_best = torch.where(
        top_active,
        torch.maximum(state["top_follow_best"], follower_potential),
        state["top_follow_best"],
    )
    # Two platform legs together retain the previous total progress scale.
    state["top_follow_progress_delta"] = (
        top_best - state["top_follow_best"]
    ).clamp_min(0.0) * 0.5
    state["top_follow_best"] = torch.where(
        platform_transition["lead_event"],
        torch.zeros_like(top_best),
        top_best,
    )
    state["top_lead_event"] = platform_transition["lead_event"]
    state["top_follow_event"] = platform_transition["platform_event"]
    state["strict_task_complete"] |= (
        platform_transition["platform_event"] & state["strict_chain_valid"]
    )
    state["phase"] = platform_transition["phase"]
    state["top_follow_success_count"] += state["top_follow_event"].sum()
    state["top_follow_delay_sum"] += (
        state["top_follow_age_steps"].float() * state["top_follow_event"].float()
    ).sum()
    state["ordered_target_active"] |= top_active

    plateau = state["phase"] == PLATEAU_ROLL
    lift_excess = (
        (wheel_bottom - state["last_completed_upper_z"][:, None] - float(cfg.plateau_lift_free_height))
        / float(cfg.plateau_lift_scale)
    ).clamp(0.0, 1.0).mean(dim=1)
    state["post_top_lift"] = lift_excess * plateau.float()
    state["post_top_lift_sum"] += state["post_top_lift"].sum()
    state["post_top_sample_count"] += plateau.sum()
    state["top_support"] = top_support

    # Velocity dividend follows the currently ordered leg. The full V11 weight
    # remains reachable, but only an intact B1/B2/.../top sequence can unlock
    # its ordered half on ascending stairs.
    state["ordered_speed_gate"] = torch.where(
        top_active, follower_potential,
        torch.where(
            state["phase"] == PLATEAU_ROLL,
            torch.ones_like(follower_potential),
            state["ordered_speed_gate"],
        ),
    )
    env._v131_ordered_last_step = step
    return state


def _v131_expected_swing_action_cost(
    env,
    diff: torch.Tensor,
    hipy_active_scale: float,
    knee_active_scale: float,
    expected_hipy_scale: float,
    expected_knee_scale: float,
    sensor_cfg=None,
) -> torch.Tensor:
    """Apply extra mobility only to the live expected front swing leg."""
    from isaaclab.managers import SceneEntityCfg
    from .rewards import get_gait_level_tensor
    from .v11_continuous import (
        _v11_front_mobility_gate,
        _v11_front_sagittal_action_indices,
        _v11_phase_weighted_action_cost,
    )

    resolved_sensor_cfg = SceneEntityCfg("v10_pgtt_scanner") if sensor_cfg is None else sensor_cfg
    scales = (
        float(hipy_active_scale), float(knee_active_scale),
        float(expected_hipy_scale), float(expected_knee_scale),
    )
    if any(not 0.0 < scale <= 1.0 for scale in scales):
        raise ValueError(f"V13.1 action regularization scales must be in (0, 1], got {scales}.")

    base = _v11_phase_weighted_action_cost(
        env, diff, float(hipy_active_scale), float(knee_active_scale), resolved_sensor_cfg,
    )
    gate = _v11_front_mobility_gate(env, sensor_cfg=resolved_sensor_cfg)
    reduction = torch.zeros((env.num_envs, 2), device=env.device, dtype=diff.dtype)
    target_scales = (float(expected_hipy_scale), float(expected_knee_scale))
    native_scales = (float(hipy_active_scale), float(knee_active_scale))
    for side, indices in enumerate(_v11_front_sagittal_action_indices(env)):
        for action_index, native_scale, target_scale in zip(
            indices, native_scales, target_scales, strict=True,
        ):
            current_scale = torch.where(
                gate[:, side],
                torch.full_like(reduction[:, side], native_scale),
                torch.ones_like(reduction[:, side]),
            )
            reduction[:, side] += diff[:, action_index] * (
                current_scale - float(target_scale)
            ).clamp_min(0.0)
    reduction *= get_gait_level_tensor(env)[:, None]
    state = v131_ordered_state(env, sensor_cfg=resolved_sensor_cfg)
    expected_side = torch.where(
        state["teaching_seeded"], state["teaching_expected_side"], state["expected_side"],
    )
    return v131_expected_swing_relief(
        base, reduction, expected_side, state["expected_swing_relief_active"],
    )


def v131_expected_swing_action_rate_l2(
    env,
    hipy_active_scale: float,
    knee_active_scale: float,
    expected_hipy_scale: float,
    expected_knee_scale: float,
    sensor_cfg=None,
) -> torch.Tensor:
    """Native V13 rate cost with bounded expected-leg sagittal relief."""
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    return _v131_expected_swing_action_cost(
        env, diff, hipy_active_scale, knee_active_scale,
        expected_hipy_scale, expected_knee_scale, sensor_cfg,
    )


def v131_expected_swing_action_smooth_l2(
    env,
    hipy_active_scale: float,
    knee_active_scale: float,
    expected_hipy_scale: float,
    expected_knee_scale: float,
    sensor_cfg=None,
) -> torch.Tensor:
    """Native V13 second-order cost with bounded expected-leg relief."""
    cache_name = "_v11_action_smooth_prev_prev_action"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, torch.zeros_like(env.action_manager.action))
    prev_prev = getattr(env, cache_name)
    diff = torch.square(env.action_manager.action + prev_prev - 2.0 * env.action_manager.prev_action)
    diff *= env.action_manager.prev_action != 0
    diff *= prev_prev != 0
    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return _v131_expected_swing_action_cost(
        env, diff, hipy_active_scale, knee_active_scale,
        expected_hipy_scale, expected_knee_scale, sensor_cfg,
    )


def v131_expected_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["expected_progress_delta"] / float(env.step_dt)


def v131_b1_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["b1_progress_payout"] / float(env.step_dt)


def v131_b1_velocity_coupled_progress_reward(
    env, command_name: str, std: float, sensor_cfg=None,
) -> torch.Tensor:
    del command_name, std
    state = v131_ordered_state(env, sensor_cfg=sensor_cfg)
    return state["b1_velocity_payout"] / float(env.step_dt)


def v131_ordered_stair_track_lin_vel_xy_exp(
    env,
    std: float,
    command_name: str,
    free_fraction: float,
    sensor_cfg=None,
) -> torch.Tensor:
    """Keep V11's speed ceiling and make half an ordered-sequence dividend."""
    from .rewards import track_lin_vel_xy_exp
    from .v11_continuous import _v11_ascending_stair_forward_mask

    native = track_lin_vel_xy_exp(env, std=std, command_name=command_name)
    state = v131_ordered_state(env, sensor_cfg=sensor_cfg)
    ascending = _v11_ascending_stair_forward_mask(env)
    scale = v131_ordered_speed_scale(
        ascending,
        state["strict_chain_valid"],
        state["ordered_target_active"],
        state["ordered_speed_gate"],
        free_fraction=float(free_fraction),
    )
    actionable_gate = torch.where(
        state["ordered_target_active"],
        state["ordered_speed_gate"].clamp(0.0, 1.0),
        torch.ones_like(native),
    ) * state["strict_chain_valid"].float()
    state["free_velocity_reward"] = torch.where(
        ascending, native * float(free_fraction), torch.zeros_like(native),
    )
    state["ordered_speed_reward"] = torch.where(
        ascending,
        native * (1.0 - float(free_fraction)) * actionable_gate,
        torch.zeros_like(native),
    )
    state["full_velocity_eligible"] = ascending & (
        ~state["ordered_target_active"]
        | (
            state["strict_chain_valid"]
            & (state["ordered_speed_gate"] >= 1.0 - 1.0e-6)
        )
    )
    return native * scale


def v131_ordered_world_forward_progress_reward(
    env,
    target_speed_fraction: float,
    free_fraction: float,
    asset_cfg=None,
) -> torch.Tensor:
    """Split V11 forward credit without changing its maximum or deficit."""
    from isaaclab.managers import SceneEntityCfg
    from .v11_continuous import (
        _v11_ascending_stair_forward_mask,
        v11_world_forward_progress_reward,
    )

    resolved_asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    native = v11_world_forward_progress_reward(
        env, target_speed_fraction=target_speed_fraction, asset_cfg=resolved_asset_cfg,
    )
    state = v131_ordered_state(env)
    scale = v131_ordered_speed_scale(
        _v11_ascending_stair_forward_mask(env),
        state["strict_chain_valid"],
        state["ordered_target_active"],
        state["ordered_speed_gate"],
        free_fraction=float(free_fraction),
    )
    return native * scale


def v131_velocity_coupled_expected_progress_reward(
    env, command_name: str, std: float, sensor_cfg=None,
) -> torch.Tensor:
    from .rewards import track_lin_vel_xy_exp

    state = v131_ordered_state(env, sensor_cfg=sensor_cfg)
    quality = track_lin_vel_xy_exp(env, std=std, command_name=command_name).clamp(0.0, 1.0)
    return quality * state["expected_progress_delta"] / float(env.step_dt)


def v131_same_side_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["same_side_progress_delta"] / float(env.step_dt)


def v131_simultaneous_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["simultaneous_progress_delta"] / float(env.step_dt)


def v131_ordered_physical_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Replacement for V11's generic landing pulse: only the ordered face pays."""
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["ordered_event"].float()


def v131_alternating_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Pay the bounded velocity-quality escrow on a correct B2+ handoff.

    B1 remains an unpaid style seed.  A completion without preceding expected-
    side progress cannot mint a pulse merely by crossing the event detector.
    """
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["velocity_escrow_payout"] / float(env.step_dt)


def v131_nonordered_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    state = v131_ordered_state(env, sensor_cfg=sensor_cfg)
    buckets = v131_violation_settlement(
        state["teaching_wrong_side_event"] | state["teaching_wrong_target_event"],
        state["simultaneous_b1_lift_event"], state["teaching_skip_event"],
    )
    return buckets["nonordered_completion"].float() / float(env.step_dt)


def v131_simultaneous_b1_lift_reward(env, sensor_cfg=None) -> torch.Tensor:
    state = v131_ordered_state(env, sensor_cfg=sensor_cfg)
    buckets = v131_violation_settlement(
        state["teaching_wrong_side_event"] | state["teaching_wrong_target_event"],
        state["simultaneous_b1_lift_event"], state["teaching_skip_event"],
    )
    return buckets["simultaneous_b1_lift"].float() / float(env.step_dt)


def v131_skip_riser_reward(env, sensor_cfg=None) -> torch.Tensor:
    state = v131_ordered_state(env, sensor_cfg=sensor_cfg)
    buckets = v131_violation_settlement(
        state["teaching_wrong_side_event"] | state["teaching_wrong_target_event"],
        state["simultaneous_b1_lift_event"], state["teaching_skip_event"],
    )
    return buckets["skip"].float() / float(env.step_dt)


def v131_skip_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["skip_progress_delta"] / float(env.step_dt)


def v131_top_follow_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["top_follow_progress_delta"] / float(env.step_dt)


def v131_top_follow_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["top_follow_event"].float() / float(env.step_dt)


def v131_post_top_front_lift_penalty(env, sensor_cfg=None) -> torch.Tensor:
    return v131_ordered_state(env, sensor_cfg=sensor_cfg)["post_top_lift"]


def v131_ordered_clearance_settlement_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Release inherited V11 clearance escrow only for the ordered target."""
    from .v11_continuous import v11_state

    state = v131_ordered_state(env, sensor_cfg=sensor_cfg)
    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    settlement = base["landing"]["clearance_settlement"]
    return settlement * state["ordered_event"].float() / float(env.step_dt)


def v131_ordered_hipy_participation_settlement_reward(
    env,
    front_hipy_joint_names: tuple[str, str],
    front_knee_joint_names: tuple[str, str],
    target_share: float,
    share_half_width: float,
    activation_scale: float,
    sensor_cfg=None,
) -> torch.Tensor:
    """Gate the inherited pose escrow with the same ordered completion event."""
    from .v11_continuous import v11_front_hipy_participation_settlement_reward

    value = v11_front_hipy_participation_settlement_reward(
        env,
        front_hipy_joint_names=front_hipy_joint_names,
        front_knee_joint_names=front_knee_joint_names,
        target_share=target_share,
        share_half_width=share_half_width,
        activation_scale=activation_scale,
        **({} if sensor_cfg is None else {"sensor_cfg": sensor_cfg}),
    )
    ordered = v131_ordered_state(env, sensor_cfg=sensor_cfg)["ordered_event"].float()
    return value * ordered


def v131_ordered_landing_center_reward(env, sensor_cfg=None) -> torch.Tensor:
    from .v11_continuous import v11_state

    state = v131_ordered_state(env, sensor_cfg=sensor_cfg)
    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    landing = base["landing"]
    chosen = landing["responsible"].clamp_min(0)[:, None]
    center = landing["scores"]["center"].gather(1, chosen).squeeze(1)
    return center * state["ordered_event"].float()


def v131_ordered_metric(env, env_ids, key: str) -> torch.Tensor:
    del env_ids
    state = v131_ordered_state(env)
    if key == "b1_first_completion_rate":
        return state["b1_completion_count"].float() / state["encounter_count"].clamp_min(1).float()
    if key == "ordered_completion_rate":
        return state["ordered_completion_count"].float() / state["physical_completion_count"].clamp_min(1).float()
    if key == "ordered_alternation_rate":
        opportunities = (state["physical_completion_count"] - state["b1_completion_count"]).clamp_min(1)
        return state["ordered_alternating_count"].float() / opportunities.float()
    if key == "adjacent_completion_count":
        return state["adjacent_completion_count"].float()
    if key == "skip_completion_count":
        return state["skip_completion_count"].float()
    if key == "physical_last_riser_mean":
        return state["physical_last_riser_index"].float().mean()
    if key == "ordered_last_riser_mean":
        return state["ordered_last_riser_index"].float().mean()
    if key == "physical_minus_ordered_gap":
        return state["physical_minus_ordered_gap"].float().mean()
    if key == "ordered_progress_reward":
        return state["expected_progress_delta"].mean() / float(env.step_dt)
    if key == "b1_progress_active_rate":
        return state["b1_progress_active"].float().mean()
    if key == "b1_progress_reward":
        return state["b1_progress_payout"].mean() / float(env.step_dt)
    if key == "b1_velocity_coupled_reward":
        return state["b1_velocity_coupled_reward"].mean()
    if key == "latch_to_b1_completion_rate":
        return state["b1_completion_count"].float() / state["first_mover_count"].clamp_min(1).float()
    if key == "b1_progress_escrow_mean":
        return state["b1_progress_escrow"].mean()
    if key == "b1_velocity_escrow_mean":
        return state["b1_velocity_escrow"].mean()
    if key == "b1_escrow_paid_rate":
        return state["b1_escrow_paid_count"].float() / state["encounter_count"].clamp_min(1).float()
    if key == "b1_escrow_forfeit_rate":
        return state["b1_escrow_forfeit_count"].float() / state["encounter_count"].clamp_min(1).float()
    if key == "expected_swing_relief_active_rate":
        return state["expected_swing_relief_active"].float().mean()
    if key == "alternating_progress_active_rate":
        return state["alternating_progress_active"].float().mean()
    if key == "strict_chain_valid_fraction":
        return state["strict_chain_valid"].float().mean()
    if key == "teaching_chain_active_fraction":
        active = state["b1_progress_active"] | state["alternating_progress_active"]
        return active.float().mean()
    if key == "teaching_resync_count":
        return state["teaching_resync_count"].float()
    if key == "ordered_speed_gate_on_actionable_target_mean":
        active = state["ordered_target_active"]
        return (
            (state["ordered_speed_gate"] * active.float()).sum()
            / active.sum().clamp_min(1).float()
        )
    if key == "ordered_speed_reward":
        return state["ordered_speed_reward"].mean()
    if key == "free_velocity_reward":
        return state["free_velocity_reward"].mean()
    if key == "full_velocity_eligibility_rate":
        return state["full_velocity_eligible"].float().mean()
    if key == "same_side_progress_rate":
        return state["same_side_progress_delta"].mean() / float(env.step_dt)
    if key == "simultaneous_progress_rate":
        return state["simultaneous_progress_delta"].mean() / float(env.step_dt)
    if key == "skip_rate":
        return state["skip_count"].float() / state["teaching_target_count"].clamp_min(1).float()
    if key == "platform_success_rate":
        return state["top_follow_success_count"].float() / state["top_follow_entry_count"].clamp_min(1).float()
    if key == "completed_risers_mean":
        return state["ordered_last_riser_index"].float().mean()
    if key == "skip_progress_active_rate":
        return state["skip_progress_active"].float().mean()
    if key == "skip_progress_reward":
        return state["skip_progress_delta"].mean() / float(env.step_dt)
    if key == "physical_side_switch_rate":
        return state["physical_side_switch_count"].float() / state["physical_transition_count"].clamp_min(1).float()
    if key == "physical_adjacent_riser_rate":
        return state["physical_adjacent_count"].float() / state["physical_transition_count"].clamp_min(1).float()
    if key == "strict_adjacent_alternating_rate":
        return state["strict_physical_transition_count"].float() / state["physical_transition_count"].clamp_min(1).float()
    if key == "skip_given_side_switch_rate":
        return state["skip_given_switch_count"].float() / state["physical_side_switch_count"].clamp_min(1).float()
    if key == "rollout_side_switch_rate":
        return state["rollout_side_switch_count"].float() / state["rollout_transition_count"].clamp_min(1).float()
    if key == "rollout_adjacent_rate":
        return state["rollout_adjacent_count"].float() / state["rollout_transition_count"].clamp_min(1).float()
    if key == "rollout_strict_alt_rate":
        return state["rollout_strict_count"].float() / state["rollout_transition_count"].clamp_min(1).float()
    if key == "rollout_skip_rate":
        return state["rollout_skip_count"].float() / state["rollout_teaching_target_count"].clamp_min(1).float()
    if key == "rollout_skip_given_switch_rate":
        return state["rollout_skip_given_switch_count"].float() / state["rollout_side_switch_count"].clamp_min(1).float()
    if key == "first_mover_latch_rate":
        return state["first_mover_count"].float() / state["encounter_count"].clamp_min(1).float()
    if key == "b1_commit_active_rate":
        active = state["b1_progress_active"] & state["first_mover_latched"]
        return active.float().mean()
    if key == "early_opposite_progress_rate":
        return state["early_opposite_progress_delta"].mean() / float(env.step_dt)
    if key == "handoff_active_rate":
        return state["handoff_active"].float().mean()
    if key == "locked_target_progress_active_rate":
        return state["locked_target_active"].float().mean()
    if key == "velocity_escrow_mean":
        return state["velocity_escrow"].mean()
    if key == "velocity_escrow_paid_rate":
        return state["velocity_escrow_paid_count"].float() / state["teaching_target_count"].clamp_min(1).float()
    if key == "velocity_escrow_forfeit_rate":
        return state["velocity_escrow_forfeit_count"].float() / state["teaching_target_count"].clamp_min(1).float()
    if key == "riser_skip_rate":
        return state["skip_count"].float() / state["teaching_target_count"].clamp_min(1).float()
    if key == "wrong_side_rate":
        return state["wrong_side_count"].float() / state["physical_completion_count"].clamp_min(1).float()
    if key == "simultaneous_b1_lift_rate":
        return state["simultaneous_b1_lift_count"].float() / state["encounter_count"].clamp_min(1).float()
    if key == "riser_index_jump_mean":
        return state["jump_sum"] / state["jump_count"].clamp_min(1).float()
    if key == "top_follow_success_rate":
        return state["top_follow_success_count"].float() / state["top_follow_entry_count"].clamp_min(1).float()
    if key == "top_follow_delay_steps":
        return state["top_follow_delay_sum"] / state["top_follow_success_count"].clamp_min(1).float()
    if key == "post_top_front_lift_rate":
        return state["post_top_lift_sum"] / state["post_top_sample_count"].clamp_min(1).float()
    if key == "seed_fl_fraction":
        total = (state["seed_fl_count"] + state["seed_fr_count"]).clamp_min(1)
        return state["seed_fl_count"].float() / total.float()
    if key == "approach_b1_fraction":
        return (state["phase"] == APPROACH_B1).float().mean()
    if key == "alternating_fraction":
        return (state["phase"] == ALTERNATING_RISERS).float().mean()
    if key == "top_follow_fraction":
        return (state["phase"] == TOP_FOLLOW).float().mean()
    if key == "top_trailing_follow_fraction":
        return (state["phase"] == TOP_TRAILING_FOLLOW).float().mean()
    if key == "plateau_roll_fraction":
        return (state["phase"] == PLATEAU_ROLL).float().mean()
    if key == "failed_fraction":
        return (~state["ordered_chain_valid"]).float().mean()
    if key == "ordered_chain_valid_fraction":
        return state["ordered_chain_valid"].float().mean()
    if key == "ordered_speed_gate_mean":
        return state["ordered_speed_gate"].mean()
    if key == "ordered_event_rate":
        return state["ordered_event"].float().mean()
    if key == "skip_event_rate":
        return state["skip_event"].float().mean()
    if key == "top_follow_event_rate":
        return state["top_follow_event"].float().mean()
    raise ValueError(f"Unknown V13.1 ordered metric: {key!r}")


__all__ = [
    "APPROACH_B1", "ALTERNATING_RISERS", "TOP_FOLLOW", "TOP_TRAILING_FOLLOW",
    "PLATEAU_ROLL", "SEQUENCE_FAILED",
    "v131_latch_b1_leader", "v131_confirm_b1_first_mover",
    "v131_settle_progress_escrow", "v131_expected_swing_relief", "v131_ordered_speed_scale",
    "v131_update_riser_ledgers", "v131_platform_phase_transition",
    "v131_target_relation", "v131_classify_completion", "v131_signed_progress",
    "v131_locked_target_potential",
    "v131_historical_max_delta", "v131_update_teaching_ledger",
    "v131_top_follow_potential", "v131_violation_settlement", "v131_ordered_state",
    "v131_expected_progress_reward", "v131_b1_progress_reward",
    "v131_b1_velocity_coupled_progress_reward",
    "v131_expected_swing_action_rate_l2", "v131_expected_swing_action_smooth_l2",
    "v131_ordered_stair_track_lin_vel_xy_exp",
    "v131_ordered_world_forward_progress_reward",
    "v131_velocity_coupled_expected_progress_reward", "v131_same_side_progress_reward",
    "v131_simultaneous_progress_reward", "v131_ordered_physical_completion_reward",
    "v131_alternating_completion_reward", "v131_nonordered_completion_reward",
    "v131_simultaneous_b1_lift_reward",
    "v131_skip_riser_reward", "v131_skip_progress_reward", "v131_top_follow_progress_reward",
    "v131_top_follow_completion_reward", "v131_post_top_front_lift_penalty",
    "v131_ordered_clearance_settlement_reward", "v131_ordered_hipy_participation_settlement_reward",
    "v131_ordered_landing_center_reward", "v131_ordered_metric",
]
