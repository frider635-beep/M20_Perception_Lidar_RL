"""V13 fast-alternation style layer on the saved native V11-FD task.

V11-FD remains authoritative for locomotion, speed and safety. V13 changes the
front completion unit from bilateral-on-one-riser to one safe front side per
successive riser, then adds training-time style rewards: after a free first
seed, historical-best progress of the opposite front wheel earns credit, and
maintaining V11's velocity quality while making it earns coupled credit. No
V13 state gates a side, masks actions, terminates an episode or enters actor.
"""

from __future__ import annotations

import torch


def v13_route_engagement_quality(
    path_speed: torch.Tensor,
    command_x: torch.Tensor,
    heading_error: torch.Tensor,
    lateral_error: torch.Tensor,
    *,
    target_speed_fraction: float,
    heading_scale: float,
    lateral_tolerance: float,
    lateral_scale: float,
) -> torch.Tensor:
    """Measure real progress along the stair route without prescribing a gait."""
    if not 0.0 < float(target_speed_fraction) <= 1.0:
        raise ValueError("V13 route target-speed fraction must lie in (0, 1].")
    if float(heading_scale) <= 0.0 or float(lateral_scale) <= 0.0:
        raise ValueError("V13 route quality scales must be positive.")
    if float(lateral_tolerance) < 0.0:
        raise ValueError("V13 route lateral tolerance must be non-negative.")
    shape = path_speed.shape
    if any(value.shape != shape for value in (command_x, heading_error, lateral_error)):
        raise ValueError("V13 route-quality tensors must be aligned one-dimensional values.")
    target_speed = (command_x * float(target_speed_fraction)).clamp_min(1.0e-3)
    progress = (path_speed / target_speed).clamp(0.0, 1.0)
    heading = torch.exp(-torch.square(heading_error / float(heading_scale)))
    lateral_violation = (lateral_error - float(lateral_tolerance)).clamp_min(0.0)
    lateral = torch.exp(-torch.square(lateral_violation / float(lateral_scale)))
    return progress * heading * lateral


def v13_encode_critic_reward_state(
    expected_side: torch.Tensor,
    sequence_started: torch.Tensor,
    target_active: torch.Tensor,
    potential: torch.Tensor,
    best_potential: torch.Tensor,
) -> torch.Tensor:
    """Encode the six hidden variables needed to value V13's soft reward.

    The actor deliberately does not consume this state.  The asymmetric critic
    uses it to distinguish otherwise identical 63D observations whose expected
    side or remaining historical-best progress differs.
    """
    n = expected_side.shape[0]
    if expected_side.shape != (n,) or sequence_started.shape != (n,) or target_active.shape != (n,):
        raise ValueError("V13 critic reward-state flags must be one-dimensional and aligned.")
    if potential.shape != (n, 2) or best_potential.shape != (n, 2):
        raise ValueError("V13 critic reward-state potentials must have shape [N, 2].")
    valid_expected = sequence_started & (expected_side >= 0)
    side = expected_side.clamp(0, 1)
    current = potential.gather(1, side[:, None]).squeeze(1)
    historical = best_potential.gather(1, side[:, None]).squeeze(1)
    zeros = torch.zeros_like(current)
    return torch.stack((
        (valid_expected & (expected_side == 0)).float(),
        (valid_expected & (expected_side == 1)).float(),
        sequence_started.float(),
        target_active.float(),
        torch.where(valid_expected, current, zeros),
        torch.where(valid_expected, historical, zeros),
    ), dim=1)


def v13_encode_critic_descent_state(
    course_active: torch.Tensor,
    live_edge_active: torch.Tensor,
    target_edge_x: torch.Tensor,
    target_drop_height: torch.Tensor,
    rear_extension: torch.Tensor,
    rear_contact_fraction: torch.Tensor,
) -> torch.Tensor:
    """Encode descent context that remains useful after the front edge disappears."""
    n = course_active.shape[0]
    if any(x.shape != (n,) for x in (
        live_edge_active, target_edge_x, target_drop_height,
        rear_extension, rear_contact_fraction,
    )):
        raise ValueError("V13 critic descent scalars must be one-dimensional and aligned.")
    course = course_active.float()
    live = live_edge_active.float() * course
    return torch.stack((
        course,
        live,
        target_edge_x * live,
        target_drop_height * live,
        rear_extension * course,
        rear_contact_fraction * course,
    ), dim=1)


def v13_front_riser_potential(
    max_clearance_progress: torch.Tensor,
    crossing_clearance_ok: torch.Tensor,
    wheel_x: torch.Tensor,
    target_riser_x: torch.Tensor,
    support: torch.Tensor,
    *,
    crossing_span: float,
) -> torch.Tensor:
    """Return bounded per-side effective riser progress in ``[0, 1]``.

    Half of the potential is V11's forward-qualified clearance progress.  A
    clean post-face traversal contributes the next quarter continuously, and
    valid upper-tread support closes the final quarter.  Merely moving a
    support leg, lifting in place, or crossing with insufficient clearance
    cannot increase the latter two components.
    """
    shape = max_clearance_progress.shape
    if any(x.shape != shape for x in (crossing_clearance_ok, wheel_x, support)):
        raise ValueError("V13 per-side potential tensors must have matching shapes.")
    if target_riser_x.shape != shape[:1] or shape[-1] != 2:
        raise ValueError("V13 potential expects FL/FR tensors and one target x per environment.")
    if float(crossing_span) <= 0.0:
        raise ValueError("V13 crossing_span must be positive.")

    clearance = max_clearance_progress.clamp(0.0, 1.0)
    clean_crossing = crossing_clearance_ok.float()
    crossing = (
        (wheel_x - target_riser_x[:, None]) / float(crossing_span)
    ).clamp(0.0, 1.0) * clean_crossing
    supported = (support & crossing_clearance_ok).float()
    return (0.50 * clearance + 0.25 * crossing + 0.25 * supported).clamp(0.0, 1.0)


def v13_historical_best_progress(
    previous_best: torch.Tensor,
    current_potential: torch.Tensor,
    eligible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Credit only a positive historical-best increment on one locked riser."""
    if previous_best.shape != current_potential.shape or previous_best.shape[-1] != 2:
        raise ValueError("V13 historical potential expects matching FL/FR tensors.")
    if eligible.shape != previous_best.shape[:1]:
        raise ValueError("V13 potential eligibility must match the environment dimension.")
    best = torch.where(
        eligible[:, None], torch.maximum(previous_best, current_potential), previous_best,
    )
    return {"best": best, "delta": (best - previous_best).clamp_min(0.0)}


def v13_split_alternation_progress(
    per_side_delta: torch.Tensor,
    expected_side: torch.Tensor,
    eligible: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Split effective progress into opposite, repeated-side and overlap terms."""
    if per_side_delta.ndim != 2 or per_side_delta.shape[1] != 2:
        raise ValueError("V13 progress split expects [num_envs, 2] FL/FR deltas.")
    if expected_side.shape != per_side_delta.shape[:1] or eligible.shape != expected_side.shape:
        raise ValueError("V13 progress split tensors have incompatible shapes.")
    # Values come from the internal sequence ledger. Avoid Tensor.any() value
    # assertions here: this function runs on CUDA every control step, where a
    # Python branch would force a device synchronization.
    expected = expected_side.clamp(0, 1)
    opposite = per_side_delta.gather(1, expected[:, None]).squeeze(1)
    same = per_side_delta.gather(1, (1 - expected)[:, None]).squeeze(1)
    simultaneous = torch.minimum(opposite, same)
    active = eligible.float()
    return {
        # Mutually exclusive settlement buckets: overlapping progress cannot
        # collect opposite-side credit before paying only a smaller penalty.
        "opposite": (opposite - simultaneous) * active,
        "same": (same - simultaneous) * active,
        # Penalize only simultaneous *effective riser progress*, never generic
        # bilateral leg motion or support adjustment.
        "simultaneous": simultaneous * active,
    }


def v13_distinct_riser_event(
    first_completion_event: torch.Tensor,
    reference_world_xy: torch.Tensor,
    last_riser_world_xy: torch.Tensor,
    last_riser_valid: torch.Tensor,
    *,
    minimum_separation: float,
) -> torch.Tensor:
    """Reject a held/reacquired copy of the same physical riser."""
    if reference_world_xy.shape != last_riser_world_xy.shape or reference_world_xy.shape[-1] != 2:
        raise ValueError("V13 riser references must be matching [num_envs, 2] tensors.")
    if first_completion_event.shape != last_riser_valid.shape or first_completion_event.ndim != 1:
        raise ValueError("V13 riser event and validity state must be matching [num_envs] tensors.")
    if float(minimum_separation) <= 0.0:
        raise ValueError("V13 minimum riser separation must be positive.")
    moved = torch.linalg.norm(reference_world_xy - last_riser_world_xy, dim=1)
    return first_completion_event & (~last_riser_valid | (moved >= float(minimum_separation)))


def v13_update_alternating_sequence(
    expected_side: torch.Tensor,
    sequence_started: torch.Tensor,
    completed_side: torch.Tensor,
    distinct_riser_event: torch.Tensor,
    simultaneous_completion: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Update a symmetric FL/FR alternating sequence.

    Side 0 is FL and side 1 is FR.  The first distinct-riser completion seeds
    the sequence without reward. Every later distinct riser advances the
    seeded FL/FR phase even when the wrong side completed it, so one mistake
    cannot shift the fixed B1/B2/B3 parity. A simultaneous first riser cannot
    seed a side; a later simultaneous riser advances the already-seeded phase.
    """
    shape = expected_side.shape
    if expected_side.ndim != 1 or any(
        x.shape != shape
        for x in (sequence_started, completed_side, distinct_riser_event, simultaneous_completion)
    ):
        raise ValueError("V13 sequence tensors must all have shape [num_envs].")
    classified_event = distinct_riser_event & ~simultaneous_completion
    seed = classified_event & ~sequence_started
    later = distinct_riser_event & sequence_started
    compliant = later & ~simultaneous_completion & (completed_side == expected_side)
    violation = later & ~simultaneous_completion & (completed_side != expected_side)
    simultaneous = distinct_riser_event & simultaneous_completion
    next_expected = torch.where(
        seed,
        1 - completed_side,
        torch.where(later, 1 - expected_side, expected_side),
    )
    return {
        "seed_event": seed,
        "compliant_event": compliant,
        "violation_event": violation,
        "simultaneous_completion_event": simultaneous,
        "sequence_started": sequence_started | seed,
        "expected_side": next_expected,
    }


def _v13_initialize_state(env) -> None:
    if hasattr(env, "_v13_alternation"):
        return
    n = env.num_envs
    env._v13_alternation = {
        "last_episode_length": env.episode_length_buf.clone(),
        "expected_side": torch.full((n,), -1, device=env.device, dtype=torch.long),
        "sequence_started": torch.zeros(n, device=env.device, dtype=torch.bool),
        "last_riser_world_xy": torch.zeros((n, 2), device=env.device),
        "last_riser_valid": torch.zeros(n, device=env.device, dtype=torch.bool),
        "active_riser_world_xy": torch.zeros((n, 2), device=env.device),
        "active_riser_valid": torch.zeros(n, device=env.device, dtype=torch.bool),
        "potential": torch.zeros((n, 2), device=env.device),
        "best_potential": torch.zeros((n, 2), device=env.device),
        "opposite_progress_delta": torch.zeros(n, device=env.device),
        "same_side_progress_delta": torch.zeros(n, device=env.device),
        "simultaneous_progress_delta": torch.zeros(n, device=env.device),
        "ascent_target_active": torch.zeros(n, device=env.device, dtype=torch.bool),
        "seed_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "compliant_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "violation_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "simultaneous_completion_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "seed_fl_count": torch.zeros((), device=env.device, dtype=torch.long),
        "seed_fr_count": torch.zeros((), device=env.device, dtype=torch.long),
        "natural_compliant_count": torch.zeros((), device=env.device, dtype=torch.long),
        "natural_violation_count": torch.zeros((), device=env.device, dtype=torch.long),
        "natural_simultaneous_count": torch.zeros((), device=env.device, dtype=torch.long),
        "opposite_progress_sum": torch.zeros((), device=env.device),
        "same_side_progress_sum": torch.zeros((), device=env.device),
        "simultaneous_progress_sum": torch.zeros((), device=env.device),
    }
    env._v13_alternation_last_step = -1


def v13_alternation_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Consume V11-FD's strict first-safe-front event exactly once per step."""
    from .v11_continuous import v11_state

    _v13_initialize_state(env)
    step = int(getattr(env, "common_step_counter", 0))
    if env._v13_alternation_last_step == step:
        return env._v13_alternation

    state = env._v13_alternation
    reset = env.episode_length_buf < state["last_episode_length"]
    for name in ("expected_side",):
        state[name][reset] = -1
    for name in ("sequence_started", "last_riser_valid", "active_riser_valid"):
        state[name][reset] = False
    state["last_riser_world_xy"][reset] = 0.0
    state["active_riser_world_xy"][reset] = 0.0
    state["potential"][reset] = 0.0
    state["best_potential"][reset] = 0.0
    state["last_episode_length"] = env.episode_length_buf.clone()

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    landing = base["landing"]
    strict = None
    if hasattr(env.cfg, "v135_riser_identity"):
        from .v135_riser_identity import v135_riser_identity_state
        strict = v135_riser_identity_state(env, sensor_cfg)
    first_event = landing["landing_event"]
    side = landing["responsible"]
    reference = landing["reference_riser_world_xy"]
    # target_active was computed against the locked reference before V11
    # releases that reference on a completion step. Keep the final support
    # increment eligible on that same step.
    reference_active = landing["target_active"]
    active_distance = torch.linalg.norm(reference - state["active_riser_world_xy"], dim=1)
    new_active_riser = reference_active & (
        ~state["active_riser_valid"]
        | (active_distance >= float(env.cfg.v13_alternation.minimum_riser_separation))
    )
    state["potential"][new_active_riser] = 0.0
    state["best_potential"][new_active_riser] = 0.0
    state["active_riser_world_xy"] = torch.where(
        new_active_riser[:, None], reference, state["active_riser_world_xy"],
    )
    state["active_riser_valid"] |= new_active_riser

    sequence_started_before = state["sequence_started"].clone()
    expected_side_before = state["expected_side"].clone()
    if strict is not None:
        sequence_started_before = strict["seeded_before"]
        expected_side_before = strict["expected_side_before"]
    next_riser_distance = torch.linalg.norm(reference - state["last_riser_world_xy"], dim=1)
    eligible_progress = (
        sequence_started_before
        & state["last_riser_valid"]
        & reference_active
        & (next_riser_distance >= float(env.cfg.v13_alternation.minimum_riser_separation))
    )
    if strict is not None:
        eligible_progress = sequence_started_before & ~strict["ordered_chain_failed"] & reference_active & strict["target_match"]
    potential = v13_front_riser_potential(
        landing["max_clearance_progress"], landing["crossing_clearance_ok"],
        landing["wheel_yaw"][..., 0], landing["target_riser_x"], landing["support"],
        crossing_span=float(env.cfg.v11_perception.wheel_radius)
        + float(env.cfg.v11_perception.front_margin),
    )
    history = v13_historical_best_progress(
        state["best_potential"], potential, eligible_progress,
    )
    progress = v13_split_alternation_progress(
        history["delta"], expected_side_before, eligible_progress,
    )
    state["potential"] = potential
    state["best_potential"] = history["best"]
    state["opposite_progress_delta"] = progress["opposite"]
    state["same_side_progress_delta"] = progress["same"]
    state["simultaneous_progress_delta"] = progress["simultaneous"]
    state["ascent_target_active"] = eligible_progress
    distinct = v13_distinct_riser_event(
        first_event,
        reference,
        state["last_riser_world_xy"],
        state["last_riser_valid"],
        minimum_separation=float(env.cfg.v13_alternation.minimum_riser_separation),
    )
    simultaneous_completion = first_event & landing["candidate_ready"].all(dim=1)
    transition = v13_update_alternating_sequence(
        state["expected_side"], state["sequence_started"], side, distinct,
        simultaneous_completion,
    ) if strict is None else {
        "expected_side": strict["expected_side"],
        "sequence_started": strict["seeded"],
        "seed_event": strict["seed"],
        "compliant_event": strict["ordered"],
        "violation_event": strict["wrong_side"] | strict["duplicate"] | strict["skip"] | strict["dual_skip"],
        "simultaneous_completion_event": strict["dual_skip"] | (strict["wrong_side"] & simultaneous_completion),
    }
    for name in (
        "expected_side", "sequence_started", "seed_event", "compliant_event",
        "violation_event", "simultaneous_completion_event",
    ):
        state[name] = transition[name]
    advance = distinct if strict is None else strict["valid"]
    state["last_riser_world_xy"] = torch.where(
        advance[:, None], reference, state["last_riser_world_xy"],
    )
    state["last_riser_valid"] |= advance

    seed = state["seed_event"]
    state["seed_fl_count"] += (seed & (side == 0)).sum()
    state["seed_fr_count"] += (seed & (side == 1)).sum()
    state["natural_compliant_count"] += state["compliant_event"].sum()
    state["natural_violation_count"] += state["violation_event"].sum()
    state["natural_simultaneous_count"] += state["simultaneous_completion_event"].sum()
    state["opposite_progress_sum"] += state["opposite_progress_delta"].sum()
    state["same_side_progress_sum"] += state["same_side_progress_delta"].sum()
    state["simultaneous_progress_sum"] += state["simultaneous_progress_delta"].sum()

    env._v13_alternation_last_step = step
    return state


def v13_critic_reward_state(env, sensor_cfg=None) -> torch.Tensor:
    """Training-only 6D V13 lifecycle context for the asymmetric critic."""
    state = v13_alternation_state(env, sensor_cfg=sensor_cfg)
    expected = state["expected_side"]
    started = state["sequence_started"]
    target_active = state["ascent_target_active"]
    if hasattr(env.cfg, "v135_riser_identity"):
        from .v135_riser_identity import v135_riser_identity_state
        strict = v135_riser_identity_state(env, sensor_cfg)
        started = strict["seeded"] & ~strict["ordered_chain_failed"]
        expected = torch.where(started, strict["expected_side"], torch.full_like(strict["expected_side"], -1))
        target_active = target_active & ~strict["ordered_chain_failed"]
    return v13_encode_critic_reward_state(
        expected, started, target_active,
        state["potential"], state["best_potential"],
    )


def v13_descending_course_mask(env) -> torch.Tensor:
    """Select descending stair columns under a positive forward command."""
    cfg = env.cfg.v11_perception
    stair_ids = set(int(value) for value in cfg.stair_terrain_type_ids)
    ascending_ids = set(int(value) for value in cfg.ascending_stair_terrain_type_ids)
    descending_ids = tuple(sorted(stair_ids - ascending_ids))
    if not descending_ids:
        raise RuntimeError("V13 descent shaping requires a descending stair terrain column.")
    terrain_types = env.scene.terrain.terrain_types
    ids = torch.tensor(descending_ids, device=env.device, dtype=terrain_types.dtype)
    descending = (terrain_types[:, None] == ids[None, :]).any(dim=1)
    command = env.command_manager.get_command("base_velocity")[:, 0]
    return descending & (command > float(cfg.forward_command_threshold))


def v13_descent_rear_motion_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Measure rearward reach, contact and rolling without retaining a tread."""
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_v13_descent_motion_last_step", None) == step:
        return env._v13_descent_motion

    from .v11_continuous import v11_state

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    asset = env.scene["robot"]
    root = asset.data.root_pos_w
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    rear_world = asset.data.body_pos_w[:, env._v11_rear_body_ids]
    dx = rear_world[..., 0] - root[:, None, 0]
    dy = rear_world[..., 1] - root[:, None, 1]
    rear_x = cy[:, None] * dx + sy[:, None] * dy

    state = getattr(env, "_v13_descent_motion", None)
    if state is None:
        state = {
            "reference_rear_x": rear_x.detach().clone(),
            "last_episode_length": env.episode_length_buf.clone(),
        }
        env._v13_descent_motion = state
    reset = env.episode_length_buf < state["last_episode_length"]
    state["reference_rear_x"][reset] = rear_x[reset].detach()
    state["last_episode_length"] = env.episode_length_buf.clone()

    # Positive only when a rear wheel retreats farther behind the body than it
    # did at the episode's neutral start. This directly measures the reported
    # failure and is independent of mirrored joint sign conventions.
    rearward_extension = (state["reference_rear_x"] - rear_x).clamp_min(0.0)
    rear_forces = env.scene.sensors["contact_forces"].data.net_forces_w[
        :, env._v11_rear_contact_ids
    ]
    rear_contact = rear_forces[..., 2] > float(env.cfg.v11_perception.contact_force_threshold)

    if not hasattr(env, "_v13_rear_wheel_joint_ids"):
        ids, names = asset.find_joints("(hl|hr)_wheel_joint", preserve_order=True)
        if names != ["hl_wheel_joint", "hr_wheel_joint"]:
            raise RuntimeError(f"V13 rear-wheel joint mapping failed: {names}")
        env._v13_rear_wheel_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    rear_omega = asset.data.joint_vel[:, env._v13_rear_wheel_joint_ids].abs()
    rolling_speed = rear_omega * float(env.cfg.v11_perception.wheel_radius)
    forward_speed = asset.data.root_lin_vel_b[:, 0].abs()[:, None]
    roll_error_each = (
        (forward_speed - rolling_speed).abs()
        / float(env.cfg.v11_perception.descent_rear_roll_speed_scale)
    ).clamp(0.0, 1.0)
    roll_error = (
        (roll_error_each * rear_contact.float()).sum(dim=1)
        / rear_contact.float().sum(dim=1).clamp_min(1.0)
    )
    course_active = v13_descending_course_mask(env)
    state.update({
        "course_active": course_active,
        "live_edge_active": base["descent"]["target_active"],
        "rearward_extension": rearward_extension,
        "rear_contact_fraction": rear_contact.float().mean(dim=1),
        "rear_roll_error": roll_error,
    })
    env._v13_descent_motion_last_step = step
    return state


def v13_critic_descent_state(env, sensor_cfg=None) -> torch.Tensor:
    """Training-only 6D course/live-edge/rear-motion asymmetric context."""
    from .v11_continuous import v11_state

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    descent = base["descent"]
    motion = v13_descent_rear_motion_state(env, sensor_cfg=sensor_cfg)
    live = descent["target_active"]
    slot = descent["target_slot"].clamp_min(0)
    height = base["risers"]["riser_height"].gather(1, slot[:, None]).squeeze(1)
    drop = torch.where(live, (-height).clamp_min(0.0), torch.zeros_like(height))
    return v13_encode_critic_descent_state(
        motion["course_active"], live, descent["target_edge_x"], drop,
        motion["rearward_extension"].amax(dim=1), motion["rear_contact_fraction"],
    )


def v13_critic_descent_metric(env, env_ids, key: str) -> torch.Tensor:
    """Report the critic-only descent context without changing behavior."""
    del env_ids
    value = v13_critic_descent_state(env)
    course = value[:, 0] > 0.5
    live = value[:, 1] > 0.5
    if key == "course_active_rate":
        return course.float().mean()
    if key == "live_edge_rate":
        return live.float().mean()
    if key == "edge_distance_mean":
        return value[:, 2].sum() / live.float().sum().clamp_min(1.0)
    if key == "drop_height_mean":
        return value[:, 3].sum() / live.float().sum().clamp_min(1.0)
    if key == "rear_extension_max_mean":
        return value[:, 4].sum() / course.float().sum().clamp_min(1.0)
    if key == "rear_contact_fraction":
        return value[:, 5].sum() / course.float().sum().clamp_min(1.0)
    raise ValueError(f"Unknown V13 critic descent metric: {key!r}")


def v13_rear_extension_cost(
    rearward_extension: torch.Tensor,
    *,
    free_reach: float,
    reach_scale: float,
    max_penalty: float,
) -> torch.Tensor:
    """Return a bounded cost after each rear wheel's free reach is exhausted."""
    if rearward_extension.ndim != 2 or rearward_extension.shape[1] != 2:
        raise ValueError("V13 rearward extension must have shape [N, 2].")
    if free_reach < 0.0 or reach_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("V13 descent rear-extension bounds must be positive.")
    excess = ((rearward_extension - free_reach).clamp_min(0.0) / reach_scale).square()
    return excess.mean(dim=1).clamp_max(max_penalty)


def v13_descent_rear_extension_penalty(
    env, free_reach: float, reach_scale: float, max_penalty: float, sensor_cfg=None,
) -> torch.Tensor:
    """Bound only excessive rearward reach during descent; keep a free zone."""
    state = v13_descent_rear_motion_state(env, sensor_cfg=sensor_cfg)
    cost = v13_rear_extension_cost(
        state["rearward_extension"], free_reach=free_reach,
        reach_scale=reach_scale, max_penalty=max_penalty,
    )
    return -cost * state["course_active"].float()


def v13_descent_rear_roll_penalty(env, sensor_cfg=None) -> torch.Tensor:
    """Encourage contacted rear wheels to roll naturally throughout descent."""
    state = v13_descent_rear_motion_state(env, sensor_cfg=sensor_cfg)
    return -state["rear_roll_error"] * state["course_active"].float()


def v13_opposite_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Dt-normalized positive historical progress of the expected opposite side."""
    state = v13_alternation_state(env, sensor_cfg=sensor_cfg)
    return state["opposite_progress_delta"] / float(env.step_dt)


def v13_any_front_riser_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Preserve V11's generic +5 task credit for any safe single-riser completion."""
    from .v11_continuous import v11_state

    landing = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    return landing["landing"]["landing_event"].float()


def v13_any_front_riser_center_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Keep V11's weak landing-centre reward physical, not secretly style-gated."""
    from .v11_continuous import v11_state

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    landing = base["landing"]
    chosen = landing["responsible"].clamp_min(0)[:, None]
    center = landing["scores"]["center"].gather(1, chosen).squeeze(1)
    return center * landing["landing_event"].float()


def v13_velocity_coupled_opposite_progress_reward(
    env, command_name: str, std: float, sensor_cfg=None,
) -> torch.Tensor:
    """Reward expected-side progress while preserving native V11 velocity quality."""
    from .rewards import track_lin_vel_xy_exp

    state = v13_alternation_state(env, sensor_cfg=sensor_cfg)
    velocity_quality = track_lin_vel_xy_exp(env, std=std, command_name=command_name).clamp(0.0, 1.0)
    return velocity_quality * state["opposite_progress_delta"] / float(env.step_dt)


def v13_same_side_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Dt-normalized repeated-side riser progress; configure with a negative weight."""
    state = v13_alternation_state(env, sensor_cfg=sensor_cfg)
    return state["same_side_progress_delta"] / float(env.step_dt)


def v13_simultaneous_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Dt-normalized overlap of effective FL/FR riser progress, not generic motion."""
    state = v13_alternation_state(env, sensor_cfg=sensor_cfg)
    return state["simultaneous_progress_delta"] / float(env.step_dt)


def v13_alternating_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Small dt-normalized confirmation that opposite-side progress completed safely."""
    state = v13_alternation_state(env, sensor_cfg=sensor_cfg)
    return state["compliant_event"].float() / float(env.step_dt)


def v13_same_side_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Penalize repeated or simultaneous completion; never terminate on style."""
    state = v13_alternation_state(env, sensor_cfg=sensor_cfg)
    non_alternating = state["violation_event"] | state["simultaneous_completion_event"]
    return non_alternating.float() / float(env.step_dt)


def v13_alternation_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose potential reachability, side bias and physical completion compliance."""
    del env_ids
    state = v13_alternation_state(env)
    if key == "potential_active_rate":
        return state["ascent_target_active"].float().mean()
    if key == "opposite_progress_delta_mean":
        return state["opposite_progress_delta"].mean()
    if key == "same_side_progress_delta_mean":
        return state["same_side_progress_delta"].mean()
    if key == "simultaneous_progress_delta_mean":
        return state["simultaneous_progress_delta"].mean()
    if key == "opposite_potential_mean":
        expected = state["expected_side"].clamp(0, 1)
        value = state["potential"].gather(1, expected[:, None]).squeeze(1)
        active = state["ascent_target_active"]
        return torch.where(
            active.any(), (value * active.float()).sum() / active.sum().float(),
            torch.zeros((), device=env.device),
        )
    if key == "seed_event_rate":
        return state["seed_event"].float().mean()
    if key == "compliant_event_rate":
        return state["compliant_event"].float().mean()
    if key == "violation_event_rate":
        return state["violation_event"].float().mean()
    if key == "simultaneous_completion_event_rate":
        return state["simultaneous_completion_event"].float().mean()
    if key == "seed_fl_fraction":
        total = (state["seed_fl_count"] + state["seed_fr_count"]).clamp_min(1)
        return state["seed_fl_count"].float() / total.float()
    if key == "natural_compliance":
        total = (
            state["natural_compliant_count"] + state["natural_violation_count"]
            + state["natural_simultaneous_count"]
        ).clamp_min(1)
        return state["natural_compliant_count"].float() / total.float()
    if key == "progress_precision":
        total = (state["opposite_progress_sum"] + state["same_side_progress_sum"]).clamp_min(1.0e-6)
        return state["opposite_progress_sum"] / total
    if key == "simultaneous_progress_fraction":
        total = (state["opposite_progress_sum"] + state["same_side_progress_sum"]).clamp_min(1.0e-6)
        return state["simultaneous_progress_sum"] / total
    raise ValueError(f"Unknown V13 alternation metric: {key!r}")


__all__ = [
    "v13_route_engagement_quality",
    "v13_encode_critic_reward_state",
    "v13_encode_critic_descent_state",
    "v13_front_riser_potential",
    "v13_historical_best_progress",
    "v13_split_alternation_progress",
    "v13_distinct_riser_event",
    "v13_update_alternating_sequence",
    "v13_alternation_state",
    "v13_critic_reward_state",
    "v13_descending_course_mask",
    "v13_descent_rear_motion_state",
    "v13_critic_descent_state",
    "v13_critic_descent_metric",
    "v13_rear_extension_cost",
    "v13_descent_rear_extension_penalty",
    "v13_descent_rear_roll_penalty",
    "v13_any_front_riser_completion_reward",
    "v13_any_front_riser_center_reward",
    "v13_opposite_progress_reward",
    "v13_velocity_coupled_opposite_progress_reward",
    "v13_same_side_progress_reward",
    "v13_simultaneous_progress_reward",
    "v13_alternating_completion_reward",
    "v13_same_side_completion_reward",
    "v13_alternation_metric",
]
