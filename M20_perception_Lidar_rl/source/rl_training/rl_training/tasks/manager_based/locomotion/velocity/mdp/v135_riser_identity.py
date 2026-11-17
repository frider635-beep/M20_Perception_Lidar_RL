"""Physical-riser identity settlement for V13.5 ascent."""

from __future__ import annotations

import torch


def v135_event_rate(event: torch.Tensor, step_dt: float) -> torch.Tensor:
    """Convert a one-frame event pulse into a dt-invariant reward rate."""
    if step_dt <= 0.0:
        raise ValueError("V13.5 event normalization requires a positive step_dt.")
    return event.float() / float(step_dt)


def v135_classify_riser_completion(
    seeded, expected_side, completed_side, completion, simultaneous,
    longitudinal_error, lateral_error, *, longitudinal_tolerance, lateral_tolerance,
):
    """Classify a completion without allowing an error to advance the target."""
    target = completion & (longitudinal_error.abs() <= longitudinal_tolerance) & (
        lateral_error.abs() <= lateral_tolerance
    )
    single = completion & ~simultaneous
    side_ok = ~seeded | (completed_side == expected_side)
    valid = target & single & side_ok
    wrong_side = target & completion & ~valid
    ahead = completion & (longitudinal_error > longitudinal_tolerance)
    behind = completion & (longitudinal_error < -longitudinal_tolerance)
    return {
        "valid": valid,
        "seed": valid & ~seeded,
        "ordered": valid & seeded,
        "wrong_side": wrong_side,
        "duplicate": behind,
        "skip": ahead & ~simultaneous,
        "dual_skip": ahead & simultaneous,
    }


def v135_update_episode_validity(previous, wrong_side, duplicate, skip, dual_skip, irrecoverable_behind):
    """Mark episode as invalid for strict success, but do NOT stop learning."""
    return previous & ~(wrong_side | duplicate | skip | dual_skip | irrecoverable_behind)


def v135_irrecoverable_behind(target_root_distance, valid_completion, active, *, threshold):
    """A valid completion wins over the old target becoming rearward that frame."""
    return active & ~valid_completion & (target_root_distance < float(threshold))


def v135_update_adjacent_chain(previous, ordered, error):
    """Track the current adjacent alternating segment after any resynchronization."""
    if not (previous.shape == ordered.shape == error.shape):
        raise ValueError("V13.5 adjacent-chain tensors must have matching shapes.")
    return torch.where(error, torch.zeros_like(previous), torch.where(ordered, previous + 1, previous))


def v135_select_gait_speed_factor(chain_length, recent_skip, *, ordered, ordinary, skip):
    """Pay local ordered movement again once the live ledger recovers from a skip."""
    if chain_length.shape != recent_skip.shape:
        raise ValueError("V13.5 speed-factor tensors must have matching shapes.")
    return torch.where(
        chain_length > 0,
        torch.full_like(chain_length, float(ordered), dtype=torch.float),
        torch.where(
            recent_skip,
            torch.full_like(chain_length, float(skip), dtype=torch.float),
            torch.full_like(chain_length, float(ordinary), dtype=torch.float),
        ),
    )


def v135_apply_gait_speed_factor(ascent, target_valid, target_actionable, target_match,
                                 recent_skip, held_factor, *, skip):
    """Keep a confirmed skip discount even while the scanner sees a later riser."""
    if not (ascent.shape == target_valid.shape == target_actionable.shape ==
            target_match.shape == recent_skip.shape == held_factor.shape):
        raise ValueError("V13.5 gait-speed masks and factors must have matching shapes.")
    skip_active = ascent & target_valid & recent_skip
    quality_active = ascent & target_valid & target_actionable & target_match
    return torch.where(
        skip_active, torch.full_like(held_factor, float(skip)),
        torch.where(quality_active, held_factor, torch.ones_like(held_factor)),
    )


def v135_match_locked_target_edges(
    edge_world_xy, edge_valid, root_world_xy, target_world_xy, locked_forward_world_xy,
    *, step_width, longitudinal_tolerance, lateral_tolerance,
    face_origin_xy=None, face_half_width=None,
):
    """Match P1/P2 in the locked stair frame, independent of current body yaw."""
    forward = locked_forward_world_xy / locked_forward_world_xy.norm(dim=1, keepdim=True).clamp_min(1e-6)
    lateral_axis = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    desired_world = torch.stack((target_world_xy, target_world_xy + torch.as_tensor(step_width, device=forward.device).reshape(-1, 1) * forward), 1)
    delta = edge_world_xy[:, :, None, :] - desired_world[:, None, :, :]
    longitudinal_error = (delta * forward[:, None, None, :]).sum(-1).abs()
    lateral_error = (delta * lateral_axis[:, None, None, :]).sum(-1).abs()
    lateral_ok = lateral_error <= float(lateral_tolerance)
    if face_origin_xy is not None:
        if face_half_width is None:
            raise ValueError("Physical edge matching requires face widths")
        edge_lateral = ((edge_world_xy - face_origin_xy[:, None, :]) * lateral_axis[:, None, :]).sum(-1).abs()
        lateral_ok = edge_lateral[:, :, None] <= face_half_width[:, None, :]
    eligible = edge_valid[:, :, None] & (longitudinal_error <= float(longitudinal_tolerance)) & lateral_ok
    score = torch.where(eligible, longitudinal_error, torch.full_like(longitudinal_error, float("inf")))
    slot = score.argmin(1)
    valid = eligible.gather(1, slot[:, None, :]).squeeze(1)
    root_delta = desired_world - root_world_xy[:, None, :]
    distance = (root_delta * forward[:, None, :]).sum(-1)
    return {"valid": valid, "slot": slot, "distance": distance}


def v135_locked_target_completion_step(
    crossed, support_steps, previous_forward, wheel_forward, wheel_lateral,
    wheel_bottom, vertical_force, upper_z, active, *, wheel_radius,
    front_margin, step_width, clearance_margin, support_tolerance,
    contact_threshold, lateral_tolerance, confirmation_frames,
):
    """Settle the *existing* locked riser after a safe crossing and stable support.

    V11 may switch its scanner reference to B(k+1) before its landing event.
    This keeps the crossing and support evidence attached to B(k), without
    creating a second target or granting credit for a wheel on B(k+1).
    """
    before = previous_forward < 0.0
    crossing = active[:, None] & before & (wheel_forward >= 0.0)
    crossing &= wheel_bottom >= upper_z[:, None] + float(clearance_margin)
    crossing &= wheel_lateral.abs() <= lateral_tolerance
    crossed = crossed | crossing
    width = torch.as_tensor(step_width, device=wheel_forward.device).reshape(-1, 1)
    lower = float(wheel_radius) + ((width - 2 * float(wheel_radius)) * .5).clamp(max=float(front_margin))
    upper = torch.as_tensor(step_width, device=wheel_forward.device).reshape(-1, 1) - float(wheel_radius)
    support = active[:, None] & crossed
    support &= (wheel_forward >= lower) & (wheel_forward < upper)
    support &= wheel_lateral.abs() <= lateral_tolerance
    support &= (wheel_bottom - upper_z[:, None]).abs() <= float(support_tolerance)
    support &= vertical_force > float(contact_threshold)
    support_steps = torch.where(support, support_steps + 1, torch.zeros_like(support_steps))
    ready = support_steps >= int(confirmation_frames)
    return crossed, support_steps, ready


def _init(env):
    if hasattr(env, "_v135_riser_identity"):
        return
    n, d = env.num_envs, env.device
    z = lambda: torch.zeros(n, device=d, dtype=torch.bool)
    env._v135_riser_identity = {
        "last_episode_length": env.episode_length_buf.clone(),
        "target_valid": z(), "seeded": z(),
        "episode_ordered_valid": torch.ones(n, device=d, dtype=torch.bool),
        "episode_skip_failed": z(),
        "recent_skip": z(),
        "b1_candidate_valid": z(),
        "b1_candidate_world_xy": torch.zeros((n, 2), device=d),
        "b1_candidate_frames": torch.zeros(n, device=d, dtype=torch.long),
        "seeded_before": z(),
        "target_world_xy": torch.zeros((n, 2), device=d),
        "target_upper_z": torch.zeros(n, device=d),
        "target_lower_z": torch.zeros(n, device=d),
        "flight_complete": z(),
        "strict_prefix_depth": torch.zeros(n, device=d, dtype=torch.long),
        "first_swing_side": torch.full((n,), -1, device=d, dtype=torch.long),
        "first_swing_finished": z(), "first_swing_failed": z(),
        "first_swing_grounded": torch.zeros((n, 2), device=d, dtype=torch.bool),
        "first_swing_anchor": torch.zeros((n, 2), device=d),
        "first_swing_contact_steps": torch.zeros(n, device=d, dtype=torch.long),
        **{k: z() for k in ("first_lift_event", "first_lift_empty_event", "first_lift_skip_event", "first_lift_b1_event", "b1_identity_rejected", "native_unconfirmed_event")},
        "target_height_valid": z(),
        "locked_crossed": torch.zeros((n, 2), device=d, dtype=torch.bool),
        "locked_support_steps": torch.zeros((n, 2), device=d, dtype=torch.long),
        "locked_previous_forward": torch.zeros((n, 2), device=d),
        "locked_previous_valid": z(),
        "settled_contact_held": torch.zeros((n, 2), device=d, dtype=torch.bool),
        "settled_world_xy": torch.zeros((n, 2, 2), device=d),
        "forward_world_xy": torch.zeros((n, 2), device=d),
        "target_id": torch.ones(n, device=d, dtype=torch.long),
        "adjacent_chain_length": torch.zeros(n, device=d, dtype=torch.long),
        "gait_speed_factor": torch.ones(n, device=d),
        "expected_side": torch.full((n,), -1, device=d, dtype=torch.long),
        "expected_side_before": torch.full((n,), -1, device=d, dtype=torch.long),
        "target_match": z(),
        "overstep_best": torch.zeros(n, device=d),
        "overstep_delta": torch.zeros(n, device=d),
        **{k: z() for k in ("valid", "seed", "ordered", "wrong_side", "duplicate", "skip", "dual_skip")},
        **{f"{k}_count": torch.zeros((), device=d, dtype=torch.long) for k in
           ("valid", "ordered", "wrong_side", "duplicate", "skip", "dual_skip")},
        "episode_invalid_count": torch.zeros((), device=d, dtype=torch.long),
    }
    env._v135_riser_identity_last_step = -1


def v135_riser_identity_state(env, sensor_cfg=None):
    from .v11_continuous import v11_state

    _init(env)
    step = int(getattr(env, "common_step_counter", 0))
    if env._v135_riser_identity_last_step == step:
        return env._v135_riser_identity
    s = env._v135_riser_identity
    reset = env.episode_length_buf < s["last_episode_length"]
    for k, value in (("target_valid", False), ("seeded", False), ("flight_complete", False), ("strict_prefix_depth", 0),
                     ("first_swing_side", -1), ("first_swing_finished", False),
                     ("first_swing_failed", False), ("first_swing_contact_steps", 0), ("episode_ordered_valid", True),
                     ("episode_skip_failed", False),
                     ("recent_skip", False),
                     ("target_height_valid", False), ("locked_previous_valid", False),
                     ("b1_candidate_valid", False), ("b1_candidate_frames", 0), ("target_id", 1),
                     ("expected_side", -1), ("adjacent_chain_length", 0),
                     ("gait_speed_factor", 1.0), ("overstep_best", 0.0)):
        s[k][reset] = value
    s["target_world_xy"][reset] = 0.0
    s["target_upper_z"][reset] = 0.0
    s["target_lower_z"][reset] = 0.0
    s["first_swing_grounded"][reset] = False
    s["first_swing_anchor"][reset] = 0.0
    s["settled_contact_held"][reset] = False
    s["locked_crossed"][reset] = False
    s["locked_support_steps"][reset] = 0
    s["locked_previous_forward"][reset] = 0.0
    s["b1_candidate_world_xy"][reset] = 0.0
    s["forward_world_xy"][reset] = 0.0
    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    landing = base["landing"]
    terrain_types = env.scene.terrain.terrain_types
    ascending_ids = torch.tensor(
        tuple(int(value) for value in env.cfg.v11_perception.ascending_stair_terrain_type_ids),
        device=env.device, dtype=terrain_types.dtype,
    )
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    ascent = (terrain_types[:, None] == ascending_ids[None, :]).any(1) & (
        command_x > float(env.cfg.v11_perception.forward_command_threshold)
    )

    asset = env.scene["robot"]
    from .v135_ascent_contract import inverted_flight_layout, flight_anchor, first_swing_step, tread_lateral_geometry
    from .v135_terrain_geometry import v135_terrain_geometry
    geometry = v135_terrain_geometry(env)
    radii, total_risers = geometry["radii"], geometry["flight_riser_count"]
    step_width = geometry["step_width"]
    s["step_width"] = step_width
    s["flight_riser_count"] = total_risers
    s["max_flight_risers"] = geometry["max_risers"]
    heading = env.command_manager.get_term("base_velocity").v11_path_heading_w
    true_b1, forward = flight_anchor(asset.data.root_pos_w[:, :2], env.scene.env_origins[:, :2], heading, radii)
    # Layout is privileged reward/verification information; Actor remains native.
    candidate_delta = landing["reference_riser_world_xy"] - true_b1
    b1_agrees = (candidate_delta * forward).sum(1).abs() <= float(env.cfg.v135_riser_identity.longitudinal_tolerance)
    s["b1_identity_rejected"] = ascent & landing["target_active"] & ~s["target_valid"] & ~b1_agrees
    candidate = ascent & landing["target_active"] & ~s["target_valid"] & ~s["flight_complete"] & b1_agrees
    candidate_error = torch.linalg.norm(
        landing["reference_riser_world_xy"] - s["b1_candidate_world_xy"], dim=1)
    stable = candidate & s["b1_candidate_valid"] & (
        candidate_error <= float(env.cfg.v135_riser_identity.b1_world_stability_tolerance))
    s["b1_candidate_frames"] = torch.where(
        candidate, torch.where(stable, s["b1_candidate_frames"] + 1, torch.ones_like(s["b1_candidate_frames"])),
        torch.zeros_like(s["b1_candidate_frames"]),
    )
    replace_candidate = candidate & ~stable
    s["b1_candidate_world_xy"] = torch.where(
        replace_candidate[:, None], landing["reference_riser_world_xy"], s["b1_candidate_world_xy"])
    s["b1_candidate_valid"] = candidate
    acquire = candidate & (s["b1_candidate_frames"] >= int(env.cfg.v135_riser_identity.b1_confirmation_frames))
    s["target_world_xy"] = torch.where(acquire[:, None], true_b1, s["target_world_xy"])
    s["forward_world_xy"] = torch.where(acquire[:, None], forward, s["forward_world_xy"])
    s["target_valid"] |= acquire

    delta = landing["reference_riser_world_xy"] - s["target_world_xy"]
    lateral_axis = torch.stack((-s["forward_world_xy"][:, 1], s["forward_world_xy"][:, 0]), dim=1)
    longitudinal = (delta * s["forward_world_xy"]).sum(1)
    lateral = (delta * lateral_axis).sum(1)
    s["seeded_before"] = s["seeded"].clone()
    s["expected_side_before"] = s["expected_side"].clone()
    # A riser is a finite face, not the first scanner ray's frozen lane.
    transverse_radius = (lateral_axis.abs() * radii).sum(1)
    face_half_width = transverse_radius[:, None] + (
        s["target_id"][:, None] - 1 + torch.arange(2, device=env.device)[None, :]
    ) * step_width[:, None]
    reference_lateral = ((landing["reference_riser_world_xy"] - env.scene.env_origins[:, :2]) * lateral_axis).sum(1).abs()
    s["target_match"] = s["target_valid"] & (longitudinal.abs() <= float(env.cfg.v135_riser_identity.longitudinal_tolerance)) & (reference_lateral <= face_half_width[:, 0])
    # Match the physical target in the three-edge pool even after V11 changes slots.
    pool = base["riser_pool"]
    match = v135_match_locked_target_edges(
        base["riser_pool_world_xy"], pool["valid"] & (pool["riser_height"] > 0),
        asset.data.root_pos_w[:, :2], s["target_world_xy"], s["forward_world_xy"],
        step_width=step_width,
        longitudinal_tolerance=float(env.cfg.v135_riser_identity.longitudinal_tolerance),
        lateral_tolerance=float(env.cfg.v135_riser_identity.lateral_tolerance),
        face_origin_xy=env.scene.env_origins[:, :2], face_half_width=face_half_width)
    s["height_match_valid"] = match["valid"][:, 0]
    s["scanner_lateral_residual"] = lateral.abs()
    slot = match["slot"][:, :1]
    lower_z = pool["riser_lower_world"].gather(1, slot).squeeze(1)
    upper_z = lower_z + pool["riser_height"].gather(1, slot).squeeze(1)
    latch_height = ascent & s["target_valid"] & match["valid"][:, 0] & ~s["target_height_valid"]
    s["target_lower_z"] = torch.where(latch_height, lower_z, s["target_lower_z"])
    s["target_upper_z"] = torch.where(latch_height, upper_z, s["target_upper_z"])
    s["target_height_valid"] |= latch_height
    wheel_world = asset.data.body_pos_w[:, env._v11_front_body_ids]
    wheel_delta = wheel_world[..., :2] - s["target_world_xy"][:, None, :]
    wheel_forward = (wheel_delta * s["forward_world_xy"][:, None, :]).sum(2)
    wheel_lateral, tread_half_width = tread_lateral_geometry(
        wheel_world[..., :2], env.scene.env_origins[:, :2], s["forward_world_xy"],
        radii, s["target_id"], step_width,
        float(env.cfg.v11_perception.wheel_radius))
    s["tread_half_width"] = tread_half_width
    force_z = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids, 2]
    locked_active = ascent & s["target_valid"] & s["target_height_valid"]
    previous = torch.where(s["locked_previous_valid"][:, None], s["locked_previous_forward"], wheel_forward)
    s["locked_crossed"], s["locked_support_steps"], locked_ready = v135_locked_target_completion_step(
        s["locked_crossed"], s["locked_support_steps"], previous, wheel_forward, wheel_lateral,
        wheel_world[..., 2] - float(env.cfg.v11_perception.wheel_radius), force_z,
        s["target_upper_z"], locked_active,
        wheel_radius=env.cfg.v11_perception.wheel_radius,
        front_margin=env.cfg.v11_perception.front_margin,
        step_width=step_width,
        clearance_margin=env.cfg.v11_perception.clearance_margin,
        support_tolerance=env.cfg.v11_perception.upper_support_tolerance,
        contact_threshold=env.cfg.v11_perception.contact_force_threshold,
        lateral_tolerance=tread_half_width,
        confirmation_frames=env.cfg.v11_perception.front_support_confirmation_frames,
    )
    # Two detectors can report the same touchdown on different frames.
    # Retain idempotency only while that foot never left contact; a new swing
    # re-arms duplicate detection, including a return to an already used tread.
    s["settled_contact_held"] &= force_z > float(env.cfg.v11_perception.contact_force_threshold)
    native_side = landing["responsible"].clamp(0, 1)
    settled_edge = s["settled_world_xy"].gather(1, native_side[:, None, None].expand(-1, 1, 2)).squeeze(1)
    same_edge = ((landing["reference_riser_world_xy"] - settled_edge) * s["forward_world_xy"]).sum(1).abs() <= float(env.cfg.v135_riser_identity.longitudinal_tolerance)
    delayed_native = s["settled_contact_held"].gather(1, native_side[:, None]).squeeze(1) & same_edge
    native_foreign = landing["landing_event"] & ~s["target_match"] & ~delayed_native
    fallback = locked_active & ~native_foreign & locked_ready.any(1)
    s["native_unconfirmed_event"] = locked_active & landing["landing_event"] & s["target_match"] & ~locked_ready.any(1)
    completion = ascent & s["target_valid"] & (native_foreign | fallback)
    simultaneous = torch.where(fallback, locked_ready.all(1), completion & landing["candidate_ready"].all(1))
    fallback_side = locked_ready.long().argmax(1)
    completed_side = torch.where(fallback, fallback_side, landing["responsible"])
    event_longitudinal = torch.where(fallback, torch.zeros_like(longitudinal), longitudinal)
    event_lateral = torch.where(fallback, torch.zeros_like(lateral), lateral)
    events = v135_classify_riser_completion(
        s["seeded"], s["expected_side"], completed_side, completion, simultaneous,
        event_longitudinal, event_lateral,
        longitudinal_tolerance=float(env.cfg.v135_riser_identity.longitudinal_tolerance),
        lateral_tolerance=float(env.cfg.v135_riser_identity.lateral_tolerance),
    )
    settle_mask = fallback[:, None] & locked_ready
    s["settled_contact_held"] |= settle_mask
    s["settled_world_xy"] = torch.where(settle_mask[..., None], s["target_world_xy"][:, None, :], s["settled_world_xy"])
    identity_cfg = env.cfg.v135_riser_identity
    swing = first_swing_step(
        s["first_swing_side"], s["first_swing_finished"], s["first_swing_failed"],
        s["first_swing_grounded"], s["first_swing_anchor"], s["first_swing_contact_steps"],
        wheel_forward, wheel_world[..., 2] - float(env.cfg.v11_perception.wheel_radius),
        force_z > float(env.cfg.v11_perception.contact_force_threshold),
        s["target_lower_z"], s["target_upper_z"], locked_active, s["target_id"],
        completed_side, events["valid"], lift_min=float(identity_cfg.first_swing_lift_min),
        support_tolerance=float(env.cfg.v11_perception.upper_support_tolerance),
        tread_end=step_width - float(env.cfg.v11_perception.wheel_radius),
        confirmation_frames=int(env.cfg.v11_perception.front_support_confirmation_frames))
    for field in ("side", "finished", "failed", "grounded", "anchor", "contact_steps"):
        s["first_swing_" + field] = swing[field]
    for field, event in (("first_lift_event", "started"), ("first_lift_empty_event", "empty"),
                         ("first_lift_skip_event", "skip"), ("first_lift_b1_event", "success")):
        s[field] = swing[event]
    # A later successful foot cannot rewrite which foot made the first attempt.
    first_mismatch = events["seed"] & (s["target_id"] == 1) & (
        (s["first_swing_side"] < 0) | (completed_side != s["first_swing_side"]))
    events["valid"] &= ~first_mismatch
    events["seed"] &= ~first_mismatch
    events["wrong_side"] |= first_mismatch
    s["first_swing_failed"] |= first_mismatch
    s["first_swing_finished"] |= first_mismatch
    # Immutable reward snapshot: settle old Bk before the live ledger advances.
    s["reward_snapshot"] = {key: s[key].clone() for key in (
        "target_id", "step_width", "first_swing_side", "first_swing_finished", "target_valid", "target_height_valid", "target_world_xy", "target_lower_z",
        "target_upper_z", "tread_half_width", "forward_world_xy", "locked_crossed", "locked_support_steps", "seeded", "expected_side")}
    s["reward_snapshot"].update({key: events[key].clone() for key in ("wrong_side", "duplicate", "skip", "dual_skip")})
    for k, value in events.items():
        s[k] = value
    target_delta = s["target_world_xy"] - asset.data.root_pos_w[:, :2]
    target_root_distance = (target_delta * s["forward_world_xy"]).sum(1)
    irrecoverable = v135_irrecoverable_behind(
        target_root_distance, events["valid"], ascent & s["target_valid"],
        threshold=float(env.cfg.v135_riser_identity.irrecoverable_behind_distance))

    # Update episode validity (for metrics only, does NOT stop learning)
    valid_before = s["episode_ordered_valid"].clone()
    s["episode_ordered_valid"] = v135_update_episode_validity(
        valid_before & ~s["first_swing_failed"], events["wrong_side"], events["duplicate"],
        events["skip"], events["dual_skip"], irrecoverable)
    s["strict_prefix_depth"] += (events["valid"] & s["episode_ordered_valid"]).long()
    s["episode_invalid_count"] += (valid_before & ~s["episode_ordered_valid"]).sum()
    s["episode_skip_failed"] |= events["skip"] | events["dual_skip"] | irrecoverable

    # Target resynchronization after errors
    valid = events["valid"]
    error_event = events["wrong_side"] | events["duplicate"] | events["skip"] | events["dual_skip"] | irrecoverable
    # A chain is the number of consecutive, expected-side, physically adjacent
    # completions after B1 establishes the phase. Any ledger error breaks it,
    # while the live target is still allowed to resynchronize below.
    s["adjacent_chain_length"] = v135_update_adjacent_chain(
        s["adjacent_chain_length"], events["ordered"], error_event,
    )
    s["recent_skip"] = torch.where(
        events["skip"] | events["dual_skip"] | irrecoverable,
        torch.ones_like(s["recent_skip"]),
        torch.where(valid | events["wrong_side"], torch.zeros_like(s["recent_skip"]), s["recent_skip"]),
    )
    identity_cfg = env.cfg.v135_riser_identity
    s["gait_speed_factor"] = v135_select_gait_speed_factor(
        s["adjacent_chain_length"], s["recent_skip"],
        ordered=identity_cfg.ordered_speed_multiplier,
        ordinary=identity_cfg.ordinary_speed_multiplier,
        skip=identity_cfg.skip_speed_multiplier,
    )

    # Wrong-side: advance target, flip expected side based on actual completed side
    wrong_side_advance = events["wrong_side"]
    s["expected_side"] = torch.where(
        wrong_side_advance,
        1 - completed_side,  # opposite of actual completed side
        s["expected_side"]
    )

    # Valid completion: standard advance and flip
    s["expected_side"] = torch.where(events["seed"], 1 - completed_side, s["expected_side"])
    s["expected_side"] = torch.where(events["ordered"], 1 - s["expected_side"], s["expected_side"])
    s["seeded"] |= events["seed"] | wrong_side_advance

    # Get step_width for all advance calculations

    # Skip/dual-skip: resynchronize target to actual physical position
    skip_event = events["skip"] | events["dual_skip"]
    if skip_event.any():
        # Calculate how many risers were actually completed based on longitudinal distance
        # Note: if robot completed B2 (starting from B1), it needs to advance to B3
        # longitudinal ≈ 0.30 means completed 1 riser → advance to next (2 steps forward from B1)
        risers_completed = (longitudinal / step_width).round().long().clamp(min=1)
        skip_advance_count = risers_completed + 1  # Advance to next after completed riser

        # Resync target forward
        skip_advance_distance = skip_advance_count.float() * step_width
        resync_target = s["target_world_xy"] + skip_advance_distance[:, None] * s["forward_world_xy"]
        s["target_world_xy"] = torch.where(skip_event[:, None], resync_target, s["target_world_xy"])
        s["target_id"] = torch.where(skip_event, s["target_id"] + skip_advance_count, s["target_id"])

        # Set expected side based on actual responsible leg
        s["expected_side"] = torch.where(
            skip_event,
            1 - completed_side,  # opposite of actual completed side
            s["expected_side"]
        )
        # A stable future-riser completion is also a valid live-phase seed,
        # even though it permanently invalidates strict episode success.
        s["seeded"] |= skip_event

    # Advance target for valid and wrong-side completions
    advance_event = valid | wrong_side_advance
    s["target_world_xy"] = torch.where(
        advance_event[:, None],
        s["target_world_xy"] + step_width[:, None] * s["forward_world_xy"],
        s["target_world_xy"],
    )
    s["target_id"] = torch.where(advance_event, s["target_id"] + 1, s["target_id"])

    s["flight_complete"] |= (advance_event | skip_event) & (s["target_id"] > total_risers)
    s["target_valid"] &= ~s["flight_complete"]
    changed_target = advance_event | skip_event
    s["target_height_valid"][changed_target] = False
    s["locked_crossed"][changed_target] = False
    s["locked_support_steps"][changed_target] = 0
    s["locked_previous_forward"] = wheel_forward
    s["locked_previous_valid"] = s["target_valid"].clone()
    s["locked_previous_valid"][changed_target] = False

    wheel_world = asset.data.body_pos_w[:, env._v11_front_body_ids, :2]
    wheel_forward = ((wheel_world - s["target_world_xy"][:, None, :]) * s["forward_world_xy"][:, None, :]).sum(2)
    onset = step_width - float(env.cfg.v135_riser_identity.overstep_margin)
    potential = ((wheel_forward.amax(1) - onset) / float(env.cfg.v135_riser_identity.overstep_margin)).clamp(0, 1)
    active = ascent & s["target_valid"] & ~(valid | wrong_side_advance | skip_event)
    best = torch.where(active, torch.maximum(s["overstep_best"], potential), s["overstep_best"])
    s["overstep_delta"] = (best - s["overstep_best"]).clamp_min(0)
    s["overstep_best"] = torch.where(valid | wrong_side_advance | skip_event, torch.zeros_like(best), best)
    for k in ("valid", "ordered", "wrong_side", "duplicate", "skip", "dual_skip"):
        s[f"{k}_count"] += s[k].sum()
    s["last_episode_length"] = env.episode_length_buf.clone()
    env._v135_riser_identity_last_step = step
    return s


def v135_valid_riser_completion_reward(env, sensor_cfg=None):
    event = v135_riser_identity_state(env, sensor_cfg)["valid"]
    from .v135_ascent_support import v135_ascent_support_state
    support = v135_ascent_support_state(env, sensor_cfg)
    # Spend existing completion credit once; no new reward for waiting, repeated
    # contact or rocking. Native velocity/landing rewards remain untouched.
    return v135_event_rate(event, float(env.step_dt)) * (0.5 + 0.5 * support["settlement_quality"])


def v135_no_skip_native_landing_reward(env, sensor_cfg=None):
    """Preserve V11 landing credit except for a physical-riser skip.

    This removes the main discrete payoff from Bk->B(k+2) without turning the
    skip into a large prohibitive penalty. Adjacent wrong-side landings retain
    native capability credit while losing strict adjacent/cycle credit.
    """
    from .v11_continuous import v11_landing_event_reward

    native = v11_landing_event_reward(env, sensor_cfg=sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)
    skip = strict["skip"] | strict["dual_skip"]
    return native * (~skip).float()


def v135_strict_chain_platform_exit_reward(env, sensor_cfg=None):
    """Pay ascent platform completion only for an unbroken physical chain.

    The number of risers is not configured here. The live physical ledger
    advances B1->B2->... until V11 detects platform exit. Any earlier ledger
    error permanently removes this terminal chain payoff for that episode.
    """
    from .v11_continuous import v11_platform_exit_success_reward

    native = v11_platform_exit_success_reward(env, sensor_cfg=sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)
    eligible = strict["seeded"] & strict["episode_ordered_valid"] & strict["flight_complete"]
    return native * eligible.float()


def v135_ascent_gait_speed_factor(env, sensor_cfg=None):
    """Bounded current-riser credit; scanner slot and old chain are not gates."""
    from .v135_forward_credit import v135_current_riser_speed_factor
    return v135_current_riser_speed_factor(env, sensor_cfg)


def v135_adjacent_chain_reward(env, sensor_cfg=None):
    """Reward each expected-side completion of the adjacent physical riser.

    B1 only seeds the phase and receives no chain credit. Errors break the
    chain, but do not disable subsequent learning or live-target resync.
    """
    s = v135_riser_identity_state(env, sensor_cfg)
    event = s["ordered"]
    return v135_event_rate(event, float(env.step_dt))


def v135_alternation_cycle_reward(env, sensor_cfg=None):
    """Reward a demonstrated FL-FR-FL or FR-FL-FR strict sequence.

    One opposite-side transition (B1->B2) earns adjacent credit. Cycle credit
    starts at chain depth two (B1->B2->B3) and remains available while the
    strict adjacent alternating chain continues.
    """
    s = v135_riser_identity_state(env, sensor_cfg)
    cycle_event = (
        s["ordered"]
        & (s["adjacent_chain_length"] >= 2)
    )
    return v135_event_rate(cycle_event, float(env.step_dt))


def v135_wrong_side_riser_reward(env, sensor_cfg=None):
    from .v135_longtrain_curriculum import v135_ascent_error_scale
    s = v135_riser_identity_state(env, sensor_cfg)
    event = s["wrong_side"] & s["seeded_before"]
    return float(v135_ascent_error_scale(env)) * v135_event_rate(event, float(env.step_dt))


def v135_duplicate_riser_reward(env, sensor_cfg=None):
    from .v135_longtrain_curriculum import v135_ascent_error_scale
    s = v135_riser_identity_state(env, sensor_cfg)
    event = s["duplicate"] & s["seeded_before"]
    return float(v135_ascent_error_scale(env)) * v135_event_rate(event, float(env.step_dt))


def v135_skip_riser_reward(env, sensor_cfg=None):
    from .v135_longtrain_curriculum import v135_ascent_error_scale
    s = v135_riser_identity_state(env, sensor_cfg)
    event = (s["skip"] | s["dual_skip"]) & s["seeded_before"]
    return float(v135_ascent_error_scale(env)) * v135_event_rate(event, float(env.step_dt))


def v135_dual_skip_riser_reward(env, sensor_cfg=None):
    from .v135_longtrain_curriculum import v135_ascent_error_scale
    s = v135_riser_identity_state(env, sensor_cfg)
    event = s["dual_skip"] & s["seeded_before"]
    return float(v135_ascent_error_scale(env)) * v135_event_rate(event, float(env.step_dt))


def v135_overstep_progress_reward(env, sensor_cfg=None):
    return v135_riser_identity_state(env, sensor_cfg)["overstep_delta"]


def v135_target_actor_perception(env, sensor_cfg=None):
    """Project the strict target and its adjacent successor into the unchanged 6D ABI."""
    from .v11_continuous import v11_state
    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    strict = v135_riser_identity_state(env, sensor_cfg)
    pool = base["riser_pool"]
    asset = env.scene["robot"]
    match = v135_match_locked_target_edges(
        base["riser_pool_world_xy"], pool["valid"], asset.data.root_pos_w[:, :2],
        strict["target_world_xy"], strict["forward_world_xy"],
        step_width=strict["step_width"],
        longitudinal_tolerance=float(env.cfg.v135_riser_identity.longitudinal_tolerance),
        lateral_tolerance=float(env.cfg.v135_riser_identity.lateral_tolerance))
    slot = match["slot"]
    height = pool["riser_height"].gather(1, slot)
    valid = strict["target_valid"][:, None] & match["valid"]
    desired = match["distance"]
    zero = torch.zeros_like(desired)
    return torch.stack((
        valid[:, 0].float(), torch.where(valid[:, 0], height[:, 0], zero[:, 0]),
        torch.where(valid[:, 0], desired[:, 0], zero[:, 0]),
        valid[:, 1].float(), torch.where(valid[:, 1], height[:, 1], zero[:, 1]),
        torch.where(valid[:, 1], desired[:, 1], zero[:, 1]),
    ), 1)


def v135_cancel_invalid_v13_completion(env, sensor_cfg=None):
    from .v13_alternation import v13_alternation_state
    strict = v135_riser_identity_state(env, sensor_cfg)
    native = v13_alternation_state(env, sensor_cfg)
    invalid = strict["wrong_side"] | strict["duplicate"] | strict["skip"] | strict["dual_skip"]
    return (invalid & native["compliant_event"]).float()


def v135_riser_identity_metric(env, env_ids, key, sensor_cfg=None):
    s = v135_riser_identity_state(env, sensor_cfg)
    if key == "target_id_mean": return s["target_id"].float().mean()
    if key == "adjacent_chain_length_mean": return s["adjacent_chain_length"].float().mean()
    if key == "gait_speed_factor_mean": return v135_ascent_gait_speed_factor(env, sensor_cfg).mean()
    if key == "target_valid_rate": return s["target_valid"].float().mean()
    if key == "episode_valid_rate": return s["episode_ordered_valid"].float().mean()
    if key == "ordered_alternation_rate": return s["ordered_count"].float() / s["valid_count"].clamp_min(1).float()
    if key.endswith("_rate"):
        name = key[:-5]
        return s[f"{name}_count"].float() / s["valid_count"].clamp_min(1).float()
    raise ValueError(key)


__all__ = [name for name in globals() if name.startswith("v135_")]
