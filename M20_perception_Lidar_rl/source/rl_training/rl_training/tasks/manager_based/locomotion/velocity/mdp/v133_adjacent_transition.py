"""V13.3 training-only settlement for three adjacent alternating risers."""

from __future__ import annotations

import torch


def v133_descending_stair_mask(env) -> torch.Tensor:
    """Identify the complete descending-stair terrain column."""
    cfg = env.cfg.v11_perception
    stair_ids = set(int(value) for value in cfg.stair_terrain_type_ids)
    ascending_ids = set(int(value) for value in cfg.ascending_stair_terrain_type_ids)
    descending_ids = tuple(sorted(stair_ids - ascending_ids))
    if not descending_ids:
        raise RuntimeError("V13.3 official descent routing has no descending terrain type IDs.")
    terrain_types = env.scene.terrain.terrain_types
    ids = torch.tensor(descending_ids, device=env.device, dtype=terrain_types.dtype)
    return (terrain_types[:, None] == ids[None, :]).any(dim=1)


def v133_ascending_stair_mask(env) -> torch.Tensor:
    """Identify only the ascending-stair terrain column."""
    cfg = env.cfg.v11_perception
    ascending_ids = tuple(sorted(int(value) for value in cfg.ascending_stair_terrain_type_ids))
    if not ascending_ids:
        raise RuntimeError("V13.3 chain settlement has no ascending terrain type IDs.")
    terrain_types = env.scene.terrain.terrain_types
    ids = torch.tensor(ascending_ids, device=env.device, dtype=terrain_types.dtype)
    return (terrain_types[:, None] == ids[None, :]).any(dim=1)


def v133_route_official_descent_curriculum(
    move_up: torch.Tensor, move_down: torch.Tensor, is_descending: torch.Tensor,
    official_up: torch.Tensor, official_down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace curriculum decisions only for descending terrain samples."""
    if any(value.shape != move_up.shape for value in (
        move_down, is_descending, official_up, official_down,
    )):
        raise ValueError("V13.3 curriculum routing tensors must have matching shapes.")
    return (
        torch.where(is_descending, official_up, move_up),
        torch.where(is_descending, official_down, move_down),
    )


def _v133_keep_outside_official_descent(env, value: torch.Tensor) -> torch.Tensor:
    return value * (~v133_descending_stair_mask(env)).float()


def v133_ascent_world_forward_deficit(env, min_speed_fraction, speed_scale, asset_cfg=None):
    from .v11_continuous import v11_world_forward_speed_deficit
    kwargs = {} if asset_cfg is None else {"asset_cfg": asset_cfg}
    return _v133_keep_outside_official_descent(
        env, v11_world_forward_speed_deficit(env, min_speed_fraction, speed_scale, **kwargs),
    )


def v133_ascent_world_forward_progress(env, target_speed_fraction, asset_cfg=None):
    from .v11_continuous import v11_world_forward_progress_reward
    kwargs = {} if asset_cfg is None else {"asset_cfg": asset_cfg}
    return _v133_keep_outside_official_descent(
        env, v11_world_forward_progress_reward(env, target_speed_fraction, **kwargs),
    )


def v133_ascent_path_lateral_deviation(env, tolerance, deviation_scale, asset_cfg=None):
    from .v11_continuous import v11_path_lateral_deviation_penalty
    kwargs = {} if asset_cfg is None else {"asset_cfg": asset_cfg}
    return _v133_keep_outside_official_descent(
        env, v11_path_lateral_deviation_penalty(env, tolerance, deviation_scale, **kwargs),
    )


def v133_ascent_path_heading_error(env, heading_scale, asset_cfg=None):
    from .v11_continuous import v11_path_heading_error_penalty
    kwargs = {} if asset_cfg is None else {"asset_cfg": asset_cfg}
    return _v133_keep_outside_official_descent(
        env, v11_path_heading_error_penalty(env, heading_scale, **kwargs),
    )


def v133_ascent_non_wheel_contact(
    env, sensor_cfg, force_threshold, force_scale, max_penalty,
):
    from .v11_continuous import v11_non_wheel_contact_force_penalty
    return _v133_keep_outside_official_descent(
        env, v11_non_wheel_contact_force_penalty(
            env, sensor_cfg, force_threshold, force_scale, max_penalty,
        ),
    )


def v133_ascent_non_wheel_scrape(
    env, sensor_cfg, force_threshold, speed_threshold, speed_scale, max_penalty, asset_cfg=None,
):
    from .v11_continuous import v11_non_wheel_scrape_penalty
    kwargs = {} if asset_cfg is None else {"asset_cfg": asset_cfg}
    return _v133_keep_outside_official_descent(
        env, v11_non_wheel_scrape_penalty(
            env, sensor_cfg, force_threshold, speed_threshold, speed_scale, max_penalty, **kwargs,
        ),
    )


def v133_ascent_wheel_impact(
    env, sensor_cfg, speed_threshold, max_penalty, asset_cfg=None,
):
    from .v11_continuous import v11_wheel_impact_penalty
    kwargs = {} if asset_cfg is None else {"asset_cfg": asset_cfg}
    return _v133_keep_outside_official_descent(
        env, v11_wheel_impact_penalty(env, sensor_cfg, speed_threshold, max_penalty, **kwargs),
    )


def v133_ascent_stair_straightness(env, asset_cfg=None):
    from .v11_continuous import v11_stair_straightness_penalty
    kwargs = {} if asset_cfg is None else {"asset_cfg": asset_cfg}
    return _v133_keep_outside_official_descent(
        env, v11_stair_straightness_penalty(env, **kwargs),
    )


def v133_ascent_front_riser_contact(env, sensor_cfg=None):
    from .v11_continuous import v11_front_riser_contact_penalty
    kwargs = {} if sensor_cfg is None else {"sensor_cfg": sensor_cfg}
    return _v133_keep_outside_official_descent(
        env, v11_front_riser_contact_penalty(env, **kwargs),
    )


def v133_official_descent_flat_orientation_correction(env) -> torch.Tensor:
    """Convert V11's -0.5 phase posture term to official M20's -50 term."""
    from .rewards import flat_orientation_l2
    from .v11_continuous import v11_phase_flat_orientation_l2

    official = flat_orientation_l2(env)
    inherited = v11_phase_flat_orientation_l2(env)
    return (50.0 * official - 0.5 * inherited) * v133_descending_stair_mask(env).float()


def v133_official_descent_lin_vel_z_correction(env) -> torch.Tensor:
    """Undo V11's live-edge vertical-velocity relief on descending columns."""
    from .rewards import lin_vel_z_l2
    from .v11_continuous import v11_phase_lin_vel_z_l2

    official = lin_vel_z_l2(env)
    inherited = v11_phase_lin_vel_z_l2(env)
    return (2.0 * official - 2.0 * inherited) * v133_descending_stair_mask(env).float()


def v133_official_descent_action_rate_correction(env) -> torch.Tensor:
    """Correct V11's -0.014 action-rate weight to official M20's -0.01."""
    from .rewards import action_rate_l2

    return action_rate_l2(env) * v133_descending_stair_mask(env).float()


def v133_route_action_rate(
    env, hipy_active_scale, knee_active_scale,
    descent_rear_hipy_scale=1.0, descent_rear_knee_scale=1.0, sensor_cfg=None,
) -> torch.Tensor:
    """Use official action rate on descent and V11 phase relief elsewhere."""
    from .rewards import action_rate_l2
    from .v11_continuous import v11_phase_relaxed_action_rate_l2

    kwargs = {} if sensor_cfg is None else {"sensor_cfg": sensor_cfg}
    perceptive = v11_phase_relaxed_action_rate_l2(
        env, hipy_active_scale, knee_active_scale,
        descent_rear_hipy_scale, descent_rear_knee_scale, **kwargs,
    )
    official = action_rate_l2(env)
    return torch.where(v133_descending_stair_mask(env), official, perceptive)


def v133_route_action_smooth(
    env, hipy_active_scale, knee_active_scale,
    descent_rear_hipy_scale=1.0, descent_rear_knee_scale=1.0, sensor_cfg=None,
) -> torch.Tensor:
    """Use official second-order action smoothing on descent."""
    from .rewards import action_smooth_l2
    from .v11_continuous import v11_phase_relaxed_action_smooth_l2

    kwargs = {} if sensor_cfg is None else {"sensor_cfg": sensor_cfg}
    perceptive = v11_phase_relaxed_action_smooth_l2(
        env, hipy_active_scale, knee_active_scale,
        descent_rear_hipy_scale, descent_rear_knee_scale, **kwargs,
    )
    official = action_smooth_l2(env)
    return torch.where(v133_descending_stair_mask(env), official, perceptive)


def v133_route_front_joint_pos_penalty(
    env, command_name, asset_cfg, front_joint_names, active_scale,
    stand_still_scale, velocity_threshold, command_threshold,
    ang_cmd_threshold=0.1, y_cmd_threshold=0.1, xy_norm_max=0.1, xz_norm_max=0.1,
    sensor_cfg=None, terrain_height_threshold=0.05, high_terrain_penalty_scale=0.5,
    mobility_sensor_cfg=None, descent_rear_joint_names=None, descent_rear_active_scale=1.0,
) -> torch.Tensor:
    """Use official M20 joint posture on descent and perceptive relief elsewhere."""
    from .rewards import joint_pos_penalty_except_turn_side_cmd
    from .v11_continuous import v11_phase_relaxed_front_joint_pos_penalty

    perceptive_kwargs = dict(
        command_name=command_name, asset_cfg=asset_cfg, front_joint_names=front_joint_names,
        active_scale=active_scale, stand_still_scale=stand_still_scale,
        velocity_threshold=velocity_threshold, command_threshold=command_threshold,
        ang_cmd_threshold=ang_cmd_threshold, y_cmd_threshold=y_cmd_threshold,
        xy_norm_max=xy_norm_max, xz_norm_max=xz_norm_max, sensor_cfg=sensor_cfg,
        terrain_height_threshold=terrain_height_threshold,
        high_terrain_penalty_scale=high_terrain_penalty_scale,
        descent_rear_joint_names=descent_rear_joint_names,
        descent_rear_active_scale=descent_rear_active_scale,
    )
    if mobility_sensor_cfg is not None:
        perceptive_kwargs["mobility_sensor_cfg"] = mobility_sensor_cfg
    perceptive = v11_phase_relaxed_front_joint_pos_penalty(env, **perceptive_kwargs)
    official = joint_pos_penalty_except_turn_side_cmd(
        env, command_name, asset_cfg, stand_still_scale, velocity_threshold,
        command_threshold, ang_cmd_threshold, y_cmd_threshold, xy_norm_max,
        xz_norm_max, sensor_cfg, terrain_height_threshold, high_terrain_penalty_scale,
    )
    return torch.where(v133_descending_stair_mask(env), official, perceptive)


def v133_official_descent_ang_vel_xy_correction(env) -> torch.Tensor:
    """Correct V11's -0.06 angular-velocity cost to official M20's -0.02."""
    from .rewards import ang_vel_xy_l2

    return ang_vel_xy_l2(env) * v133_descending_stair_mask(env).float()


def v133_official_descent_joint_pos_cost(
    env, command_name, asset_cfg, stand_still_scale, velocity_threshold,
    command_threshold, sensor_cfg=None, *, official_kind: str,
) -> torch.Tensor:
    """Return the official joint-position cost on descending columns only."""
    from .rewards import joint_pos_penalty, joint_pos_penalty_except_turn_side_cmd

    if official_kind == "hipx":
        value = joint_pos_penalty(
            env, command_name, asset_cfg, stand_still_scale,
            velocity_threshold, command_threshold,
        )
    elif official_kind in {"hipy", "knee"}:
        value = joint_pos_penalty_except_turn_side_cmd(
            env, command_name, asset_cfg, stand_still_scale,
            velocity_threshold, command_threshold,
            sensor_cfg=sensor_cfg, terrain_height_threshold=0.06,
            high_terrain_penalty_scale=0.1,
        )
    else:
        raise ValueError(f"Unknown official V13.3 joint cost kind: {official_kind!r}")
    return value * v133_descending_stair_mask(env).float()


def v133_official_descent_joint_limit_cancellation(env, asset_cfg) -> torch.Tensor:
    """Cancel V11's added joint-limit cost; official M20 config weights it zero."""
    from isaaclab.envs.mdp.rewards import joint_pos_limits

    return joint_pos_limits(env, asset_cfg=asset_cfg) * v133_descending_stair_mask(env).float()


def v133_cancel_nonofficial_descent_reward(env, source_term_name: str) -> torch.Tensor:
    """Re-evaluate a cached/state-safe source term for exact descent cancellation."""
    source = env.reward_manager.get_term_cfg(source_term_name)
    return source.func(env, **source.params) * v133_descending_stair_mask(env).float()


def v133_nonofficial_illegal_contact_termination(env, sensor_cfg, threshold) -> torch.Tensor:
    """Keep V11's added base-contact termination off official descent."""
    from isaaclab.envs.mdp import illegal_contact

    terminated = illegal_contact(env, sensor_cfg=sensor_cfg, threshold=threshold)
    return terminated & ~v133_descending_stair_mask(env)


def v133_ascent_front_riser_collision_termination(env, sensor_cfg=None) -> torch.Tensor:
    """Limit the V11 front-riser termination to perceptive ascent."""
    from .v11_continuous import v11_confirmed_front_riser_contact

    kwargs = {} if sensor_cfg is None else {"sensor_cfg": sensor_cfg}
    terminated = v11_confirmed_front_riser_contact(env, **kwargs)
    return terminated & ~v133_descending_stair_mask(env)


def v133_episode_chain_quality(
    successes: torch.Tensor, failures: torch.Tensor, *, minimum_precision: float,
) -> torch.Tensor:
    """Require observed chain success and a bounded episode precision."""
    if successes.shape != failures.shape:
        raise ValueError("V13.3 episode chain counts must have matching shapes.")
    if not 0.0 <= float(minimum_precision) <= 1.0:
        raise ValueError("V13.3 minimum precision must lie in [0, 1].")
    precision = successes.float() / (successes + failures).clamp_min(1).float()
    return (successes > 0) & (precision >= float(minimum_precision))


def v133_episode_reset_mask(
    current_length: torch.Tensor, previous_length: torch.Tensor, *, same_step: bool,
) -> torch.Tensor:
    """Catch same-step resets and one-action episodes without leaking chain state."""
    if current_length.shape != previous_length.shape:
        raise ValueError("V13.3 episode-length ledgers must have matching shapes.")
    return current_length < previous_length if same_step else current_length <= previous_length


def v133_update_adjacent_progress(
    previous_best: torch.Tensor, previous_escrow: torch.Tensor,
    current_potential: torch.Tensor, target_valid: torch.Tensor, *,
    correct_completion: torch.Tensor, failed_completion: torch.Tensor,
    terminal: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pay monotonic adjacent progress now and claw it back if the attempt fails."""
    if any(value.shape != previous_best.shape for value in (
        previous_escrow, current_potential, target_valid, correct_completion,
        failed_completion, terminal,
    )):
        raise ValueError("V13.3 adjacent progress tensors must have matching shapes.")
    best = torch.where(target_valid, torch.maximum(previous_best, current_potential), previous_best)
    delta = (best - previous_best).clamp_min(0.0)
    pending = previous_escrow + delta
    settle_failure = failed_completion | terminal
    clawback = torch.where(settle_failure, pending, torch.zeros_like(pending))
    settled = correct_completion | settle_failure
    return {
        "best": torch.where(settled, torch.zeros_like(best), best),
        "escrow": torch.where(settled, torch.zeros_like(pending), pending),
        "delta": delta,
        "clawback": clawback,
    }


def v133_confirm_distinct_front_support(
    support_steps: torch.Tensor, support_now: torch.Tensor,
    riser_world_xy: torch.Tensor, riser_forward: torch.Tensor,
    tracked_world_xy: torch.Tensor, tracked_valid: torch.Tensor,
    credited_forward: torch.Tensor, credited_valid: torch.Tensor, *, confirmation_frames: int,
    distinct_tolerance: float,
) -> dict[str, torch.Tensor]:
    """Turn identity-stable, monotonically advancing support into side events."""
    if support_steps.shape != support_now.shape or support_steps.ndim != 3:
        raise ValueError("V13.3 support ledgers must be matching [N, side, slot] tensors.")
    n, sides, slots = support_steps.shape
    if sides != 2 or riser_world_xy.shape != (n, slots, 2) or riser_forward.shape != (n, slots):
        raise ValueError("V13.3 live support expects two front sides and [N, slot, 2] risers.")
    if tracked_world_xy.shape != (n, sides, slots, 2) or tracked_valid.shape != (n, sides, slots):
        raise ValueError("V13.3 tracked support identities have incompatible shapes.")
    if credited_forward.shape != (n, sides) or credited_valid.shape != (n, sides):
        raise ValueError("V13.3 credited support tensors have incompatible shapes.")
    if int(confirmation_frames) < 2 or float(distinct_tolerance) <= 0.0:
        raise ValueError("V13.3 support confirmation and distinct tolerance must be positive.")
    current_xy = riser_world_xy[:, None, :, :].expand(-1, sides, -1, -1)
    same_identity = tracked_valid & (
        torch.linalg.norm(current_xy - tracked_world_xy, dim=-1) < float(distinct_tolerance)
    )
    next_steps = torch.where(
        support_now,
        torch.where(same_identity, support_steps + 1, torch.ones_like(support_steps)),
        torch.zeros_like(support_steps),
    )
    next_tracked_xy = torch.where(support_now[:, :, :, None], current_xy, tracked_world_xy)
    next_tracked_valid = torch.where(support_now, torch.ones_like(tracked_valid), torch.zeros_like(tracked_valid))
    # Per-side forward progress is monotonic within an episode.  This allows the
    # opposite side to follow onto the same tread, while preventing B1->B2->B1
    # replay from minting another event for the returning side.
    uncredited = ~credited_valid[:, :, None] | (
        riser_forward[:, None, :] >= credited_forward[:, :, None] + float(distinct_tolerance)
    )
    candidates = (next_steps >= int(confirmation_frames)) & uncredited
    side_event = candidates.any(dim=2)
    slot = candidates.long().argmax(dim=2)
    chosen_xy = riser_world_xy[:, None, :, :].expand(-1, 2, -1, -1).gather(
        2, slot[:, :, None, None].expand(-1, -1, 1, 2),
    ).squeeze(2)
    chosen_forward = riser_forward[:, None, :].expand(-1, sides, -1).gather(
        2, slot[:, :, None],
    ).squeeze(2)
    next_credited_forward = torch.where(side_event, chosen_forward, credited_forward)
    return {
        "steps": next_steps,
        "tracked_world_xy": next_tracked_xy,
        "tracked_valid": next_tracked_valid,
        "side_event": side_event,
        "slot": slot,
        "chosen_world_xy": chosen_xy,
        "credited_forward": next_credited_forward,
        "credited_valid": credited_valid | side_event,
    }


def v133_checkpoint_qualified(
    *, chain_count: int, fl_count: int, fr_count: int,
    chain_precision: float, velocity_tracking: float, finite: bool,
    minimum_chain_count: int, minimum_each_final_side: int,
    minimum_chain_precision: float, minimum_velocity_tracking: float,
) -> bool:
    """Apply the frozen learned-policy gate without a last-checkpoint bias."""
    return bool(
        finite
        and int(chain_count) >= int(minimum_chain_count)
        and int(fl_count) >= int(minimum_each_final_side)
        and int(fr_count) >= int(minimum_each_final_side)
        and float(chain_precision) >= float(minimum_chain_precision)
        and float(velocity_tracking) >= float(minimum_velocity_tracking)
    )


def v133_update_three_step_chain(
    chain_length: torch.Tensor, last_side: torch.Tensor,
    last_riser_world_xy: torch.Tensor, last_upper_z: torch.Tensor,
    forward_world_xy: torch.Tensor, chain_valid: torch.Tensor, *,
    completion_event: torch.Tensor, completed_side: torch.Tensor,
    riser_world_xy: torch.Tensor, upper_z: torch.Tensor,
    simultaneous_completion: torch.Tensor, heading_world_xy: torch.Tensor,
    step_width: float, longitudinal_tolerance: float,
    lateral_tolerance: float, minimum_height_gain: float,
) -> dict[str, torch.Tensor]:
    """Advance a mirror-symmetric chain and settle on B3 and later steps."""
    n = chain_length.shape[0]
    if any(value.shape != (n,) for value in (
        chain_length, last_side, last_upper_z, chain_valid, completion_event,
        completed_side, upper_z, simultaneous_completion,
    )):
        raise ValueError("V13.3 scalar chain tensors must have shape [num_envs].")
    if any(value.shape != (n, 2) for value in (
        last_riser_world_xy, forward_world_xy, riser_world_xy, heading_world_xy,
    )):
        raise ValueError("V13.3 geometric chain tensors must have shape [num_envs, 2].")
    if not 0.0 < float(longitudinal_tolerance) < 0.5 * float(step_width):
        raise ValueError("V13.3 tolerance must distinguish adjacent treads.")

    single = completion_event & ~simultaneous_completion & (completed_side >= 0)
    seed_event = single & ~chain_valid
    later = single & chain_valid
    delta = riser_world_xy - last_riser_world_xy
    forward_delta = (delta * forward_world_xy).sum(dim=1)
    lateral_axis = torch.stack((-forward_world_xy[:, 1], forward_world_xy[:, 0]), dim=1)
    lateral_delta = (delta * lateral_axis).sum(dim=1).abs()
    height_gain = upper_z - last_upper_z
    same_tread_geometry = (
        (forward_delta.abs() <= float(longitudinal_tolerance))
        & (lateral_delta <= float(lateral_tolerance))
        & (height_gain.abs() <= float(minimum_height_gain))
    )
    adjacent_geometry = (
        ((forward_delta - float(step_width)).abs() <= float(longitudinal_tolerance))
        & (lateral_delta <= float(lateral_tolerance))
        & (height_gain >= float(minimum_height_gain))
    )
    adjacent = later & adjacent_geometry
    correct_event = adjacent & (completed_side != last_side)
    repeat_event = adjacent & (completed_side == last_side)
    follower_event = later & same_tread_geometry & (completed_side != last_side)
    skip_event = later & ~adjacent_geometry & ~follower_event
    simultaneous_event = completion_event & simultaneous_completion
    chain_candidate_event = correct_event & (chain_length >= 2)

    next_length = chain_length.clone()
    next_length = torch.where(seed_event, torch.ones_like(next_length), next_length)
    next_length = torch.where(correct_event, (chain_length + 1).clamp_max(3), next_length)
    next_length = torch.where(repeat_event | skip_event, torch.ones_like(next_length), next_length)
    next_length = torch.where(simultaneous_event, torch.zeros_like(next_length), next_length)
    next_valid = (
        (chain_valid | seed_event | repeat_event | skip_event)
        & ~simultaneous_event
    )

    accepted_distinct = seed_event | correct_event | repeat_event | skip_event
    next_side = torch.where(accepted_distinct, completed_side, last_side)
    next_xy = torch.where(accepted_distinct[:, None], riser_world_xy, last_riser_world_xy)
    next_upper_z = torch.where(accepted_distinct, upper_z, last_upper_z)
    reseed = seed_event | repeat_event | skip_event
    next_forward = torch.where(reseed[:, None], heading_world_xy, forward_world_xy)
    return {
        "chain_length": next_length, "last_side": next_side,
        "last_riser_world_xy": next_xy, "last_upper_z": next_upper_z,
        "forward_world_xy": next_forward, "chain_valid": next_valid,
        "seed_event": seed_event, "adjacent_event": correct_event,
        "chain_candidate_event": chain_candidate_event, "repeat_event": repeat_event,
        "follower_event": follower_event, "skip_event": skip_event,
        "simultaneous_event": simultaneous_event,
    }


def v133_match_locked_first_riser(
    completion_event: torch.Tensor,
    completed_world_xy: torch.Tensor,
    locked_valid: torch.Tensor,
    locked_world_xy: torch.Tensor,
    locked_forward_world_xy: torch.Tensor,
    *, longitudinal_tolerance: float, lateral_tolerance: float,
) -> torch.Tensor:
    """Qualify B1 against the first world-locked ascending face."""
    n = completion_event.shape[0]
    if locked_valid.shape != (n,) or completed_world_xy.shape != (n, 2):
        raise ValueError("V13.4 locked-B1 scalar/vector shapes are incompatible.")
    if locked_world_xy.shape != (n, 2) or locked_forward_world_xy.shape != (n, 2):
        raise ValueError("V13.4 locked-B1 world geometry must have shape [num_envs, 2].")
    delta = completed_world_xy - locked_world_xy
    longitudinal = (delta * locked_forward_world_xy).sum(dim=1).abs()
    lateral_axis = torch.stack(
        (-locked_forward_world_xy[:, 1], locked_forward_world_xy[:, 0]), dim=1,
    )
    lateral = (delta * lateral_axis).sum(dim=1).abs()
    return (
        completion_event & locked_valid
        & (longitudinal <= float(longitudinal_tolerance))
        & (lateral <= float(lateral_tolerance))
    )


def _v133_initialize_state(env) -> None:
    if hasattr(env, "_v133_chain"):
        return
    n, device = env.num_envs, env.device
    zeros_bool = lambda: torch.zeros(n, device=device, dtype=torch.bool)
    zeros_long = lambda: torch.zeros(n, device=device, dtype=torch.long)
    env._v133_chain = {
        "last_episode_length": env.episode_length_buf.clone(),
        "chain_length": zeros_long(),
        "last_side": torch.full((n,), -1, device=device, dtype=torch.long),
        "last_riser_world_xy": torch.zeros((n, 2), device=device),
        "last_upper_z": torch.zeros(n, device=device),
        "forward_world_xy": torch.zeros((n, 2), device=device),
        "chain_valid": zeros_bool(),
        "b1_locked": zeros_bool(),
        "b1_locked_world_xy": torch.zeros((n, 2), device=device),
        "b1_locked_forward_world_xy": torch.zeros((n, 2), device=device),
        "b1_seed_miss_event": zeros_bool(),
        "live_support_steps": torch.zeros((n, 2, 2), device=device, dtype=torch.long),
        "support_tracked_world_xy": torch.zeros((n, 2, 2, 2), device=device),
        "support_tracked_valid": torch.zeros((n, 2, 2), device=device, dtype=torch.bool),
        "support_credited_forward": torch.zeros((n, 2), device=device),
        "support_credited_valid": torch.zeros((n, 2), device=device, dtype=torch.bool),
        "adjacent_progress_best": torch.zeros(n, device=device),
        "adjacent_progress_escrow": torch.zeros(n, device=device),
        "adjacent_progress_delta": torch.zeros(n, device=device),
        "adjacent_progress_clawback": torch.zeros(n, device=device),
        "adjacent_progress_active": zeros_bool(),
        "adjacent_progress_delta_sum": torch.zeros((), device=device),
        "adjacent_progress_clawback_sum": torch.zeros((), device=device),
        "adjacent_progress_active_samples": torch.zeros((), device=device, dtype=torch.long),
        "seed_event": zeros_bool(), "adjacent_event": zeros_bool(),
        "chain_candidate_event": zeros_bool(), "chain_event": zeros_bool(),
        "repeat_event": zeros_bool(),
        "follower_event": zeros_bool(), "skip_event": zeros_bool(),
        "simultaneous_event": zeros_bool(),
        "episode_chain_success_count": zeros_long(),
        "episode_chain_failure_count": zeros_long(),
        "seed_count": torch.zeros((), device=device, dtype=torch.long),
        "adjacent_count": torch.zeros((), device=device, dtype=torch.long),
        "chain_count": torch.zeros((), device=device, dtype=torch.long),
        "repeat_count": torch.zeros((), device=device, dtype=torch.long),
        "follower_count": torch.zeros((), device=device, dtype=torch.long),
        "skip_count": torch.zeros((), device=device, dtype=torch.long),
        "simultaneous_count": torch.zeros((), device=device, dtype=torch.long),
        "b1_lock_count": torch.zeros((), device=device, dtype=torch.long),
        "b1_seed_miss_count": torch.zeros((), device=device, dtype=torch.long),
        "chain_fl_count": torch.zeros((), device=device, dtype=torch.long),
        "chain_fr_count": torch.zeros((), device=device, dtype=torch.long),
    }
    env._v133_chain_last_step = -1


def v133_three_step_chain_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Convert stable live front-wheel support into ordered riser events."""
    from .v11_continuous import v11_state

    _v133_initialize_state(env)
    state = env._v133_chain
    step = int(getattr(env, "common_step_counter", 0))
    same_step = env._v133_chain_last_step == step
    reset = v133_episode_reset_mask(
        env.episode_length_buf, state["last_episode_length"], same_step=same_step,
    )
    if reset.any():
        for name in (
            "chain_length", "last_upper_z",
            "episode_chain_success_count", "episode_chain_failure_count",
            "adjacent_progress_best", "adjacent_progress_escrow",
            "adjacent_progress_delta", "adjacent_progress_clawback",
        ):
            state[name][reset] = 0
        state["live_support_steps"][reset] = 0
        state["support_tracked_world_xy"][reset] = 0.0
        state["support_tracked_valid"][reset] = False
        state["support_credited_forward"][reset] = 0.0
        state["support_credited_valid"][reset] = False
        state["last_side"][reset] = -1
        state["last_riser_world_xy"][reset] = 0.0
        state["forward_world_xy"][reset] = 0.0
        state["chain_valid"][reset] = False
        state["b1_locked"][reset] = False
        state["b1_locked_world_xy"][reset] = 0.0
        state["b1_locked_forward_world_xy"][reset] = 0.0
        state["b1_seed_miss_event"][reset] = False
        state["adjacent_progress_active"][reset] = False
        for name in (
            "seed_event", "adjacent_event", "chain_candidate_event", "chain_event", "repeat_event",
            "follower_event", "skip_event", "simultaneous_event",
        ):
            state[name][reset] = False
        state["last_episode_length"] = env.episode_length_buf.clone()
    if same_step:
        return state
    state["last_episode_length"] = env.episode_length_buf.clone()

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    landing = base["landing"]
    risers = base["risers"]
    asset = env.scene["robot"]
    root = asset.data.root_pos_w
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw_body = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    cy, sy = torch.cos(yaw_body), torch.sin(yaw_body)
    riser_world_xy = torch.stack((
        root[:, None, 0] + cy[:, None] * risers["riser_x"] - sy[:, None] * risers["riser_y"],
        root[:, None, 1] + sy[:, None] * risers["riser_x"] + cy[:, None] * risers["riser_y"],
    ), dim=-1)
    wheel_yaw = landing["wheel_yaw"]
    wheel_bottom = landing["wheel_bottom"]
    forces_z = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids, 2]
    upper_z = risers["riser_lower_world"] + risers["riser_height"]
    cfg11 = env.cfg.v11_perception
    positive = risers["valid"] & (risers["riser_height"] >= float(env.cfg.v133_transition.minimum_height_gain))
    x_min = risers["riser_x"] + float(cfg11.wheel_radius) + float(cfg11.front_margin)
    x_max = risers["riser_x"] + float(env.cfg.v133_transition.step_width) - float(cfg11.wheel_radius)
    y_min = risers["riser_y"] - float(cfg11.tread_corridor_half_width) + float(cfg11.lateral_margin)
    y_max = risers["riser_y"] + float(cfg11.tread_corridor_half_width) - float(cfg11.lateral_margin)
    support_now = (
        positive[:, None, :]
        & (wheel_yaw[:, :, None, 0] >= x_min[:, None, :])
        & (wheel_yaw[:, :, None, 0] <= x_max[:, None, :])
        & (wheel_yaw[:, :, None, 1] >= y_min[:, None, :])
        & (wheel_yaw[:, :, None, 1] <= y_max[:, None, :])
        & ((wheel_bottom[:, :, None] - upper_z[:, None, :]).abs() <= float(cfg11.upper_support_tolerance))
        & (forces_z[:, :, None] > float(cfg11.contact_force_threshold))
    )
    command_term = env.command_manager.get_term("base_velocity")
    if not hasattr(command_term, "v11_path_heading_w"):
        raise RuntimeError("V13.3 requires V11's fixed world route heading.")
    yaw = command_term.v11_path_heading_w
    heading = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
    riser_forward = (riser_world_xy * heading[:, None, :]).sum(dim=2)
    require_b1 = bool(getattr(env.cfg.v133_transition, "require_first_riser_seed", False))
    if require_b1:
        # Lock the nearest visible ascending face once per episode.  Scanner
        # reacquisition may update live slots, but it cannot rename B2 as B1.
        visible_positive = positive & (risers["riser_x"] >= 0.0)
        lock_candidate = (
            v133_ascending_stair_mask(env) & visible_positive.any(dim=1)
            & ~state["b1_locked"]
        )
        forward_score = torch.where(
            visible_positive, risers["riser_x"],
            torch.full_like(risers["riser_x"], torch.inf),
        )
        lock_slot = forward_score.argmin(dim=1)
        lock_xy = riser_world_xy.gather(
            1, lock_slot[:, None, None].expand(-1, 1, 2),
        ).squeeze(1)
        state["b1_locked_world_xy"] = torch.where(
            lock_candidate[:, None], lock_xy, state["b1_locked_world_xy"],
        )
        state["b1_locked_forward_world_xy"] = torch.where(
            lock_candidate[:, None], heading, state["b1_locked_forward_world_xy"],
        )
        state["b1_locked"] |= lock_candidate
        state["b1_lock_count"] += lock_candidate.sum()
    support = v133_confirm_distinct_front_support(
        state["live_support_steps"], support_now, riser_world_xy, riser_forward,
        state["support_tracked_world_xy"], state["support_tracked_valid"],
        state["support_credited_forward"], state["support_credited_valid"],
        confirmation_frames=int(cfg11.front_support_confirmation_frames),
        distinct_tolerance=float(env.cfg.v133_transition.longitudinal_tolerance),
    )
    state["live_support_steps"] = support["steps"]
    state["support_tracked_world_xy"] = support["tracked_world_xy"]
    state["support_tracked_valid"] = support["tracked_valid"]
    state["support_credited_forward"] = support["credited_forward"]
    state["support_credited_valid"] = support["credited_valid"]
    # Rough/boxes/slope edges can satisfy the same local geometry.  They are
    # outside the ordered-ascent task and must not contaminate event counters,
    # rollout precision or the ascent curriculum ledger.
    side_event = support["side_event"] & v133_ascending_stair_mask(env)[:, None]
    completion_event = side_event.any(dim=1)
    completed_side = side_event.long().argmax(dim=1)
    chosen_xy = support["chosen_world_xy"].gather(
        1, completed_side[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    chosen_slot = support["slot"].gather(1, completed_side[:, None]).squeeze(1)
    chosen_upper_z = upper_z.gather(1, chosen_slot[:, None]).squeeze(1)

    if require_b1:
        b1_match = v133_match_locked_first_riser(
            completion_event, chosen_xy, state["b1_locked"],
            state["b1_locked_world_xy"], state["b1_locked_forward_world_xy"],
            longitudinal_tolerance=float(env.cfg.v133_transition.longitudinal_tolerance),
            lateral_tolerance=float(env.cfg.v133_transition.lateral_tolerance),
        )
        state["b1_seed_miss_event"] = (
            completion_event & ~state["chain_valid"] & ~b1_match
        )
        chain_completion_event = completion_event & (state["chain_valid"] | b1_match)
    else:
        state["b1_seed_miss_event"].zero_()
        chain_completion_event = completion_event

    # Build one exact next-tread target from the last confirmed support.  This
    # is reward-side bookkeeping only; none of it enters the 63-D actor input.
    predicted_xy = state["last_riser_world_xy"] + state["forward_world_xy"] * float(
        env.cfg.v133_transition.step_width
    )
    target_delta = riser_world_xy - predicted_xy[:, None, :]
    target_long = (target_delta * state["forward_world_xy"][:, None, :]).sum(dim=2).abs()
    target_lateral_axis = torch.stack(
        (-state["forward_world_xy"][:, 1], state["forward_world_xy"][:, 0]), dim=1,
    )
    target_lateral = (target_delta * target_lateral_axis[:, None, :]).sum(dim=2).abs()
    target_match = (
        state["chain_valid"][:, None]
        & positive
        & (target_long <= float(env.cfg.v133_transition.longitudinal_tolerance))
        & (target_lateral <= float(env.cfg.v133_transition.lateral_tolerance))
        & (upper_z >= state["last_upper_z"][:, None] + float(env.cfg.v133_transition.minimum_height_gain))
    )
    target_score = target_long + target_lateral
    target_slot = torch.where(
        target_match, target_score, torch.full_like(target_score, torch.inf),
    ).argmin(dim=1)
    target_valid = target_match.any(dim=1) & v133_ascending_stair_mask(env)
    expected_side = (1 - state["last_side"]).clamp(0, 1)
    wheel = wheel_yaw.gather(1, expected_side[:, None, None].expand(-1, 1, 2)).squeeze(1)
    wheel_z = wheel_bottom.gather(1, expected_side[:, None]).squeeze(1)
    face_x = risers["riser_x"].gather(1, target_slot[:, None]).squeeze(1)
    lower_z = risers["riser_lower_world"].gather(1, target_slot[:, None]).squeeze(1)
    target_upper_z = upper_z.gather(1, target_slot[:, None]).squeeze(1)
    approach = ((wheel[:, 0] - (face_x - float(env.cfg.v133_transition.step_width))) /
                float(env.cfg.v133_transition.step_width)).clamp(0.0, 1.0)
    lift = ((wheel_z - lower_z) /
            (target_upper_z - lower_z + float(cfg11.clearance_margin)).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    crossing = ((wheel[:, 0] - face_x) /
                (float(cfg11.wheel_radius) + float(cfg11.front_margin))).clamp(0.0, 1.0)
    target_support = support_now[
        torch.arange(env.num_envs, device=env.device), expected_side, target_slot,
    ].float()
    adjacent_potential = (
        0.50 * torch.minimum(lift, approach) + 0.25 * crossing + 0.25 * target_support
    ) * target_valid.float()
    transition = v133_update_three_step_chain(
        state["chain_length"], state["last_side"], state["last_riser_world_xy"],
        state["last_upper_z"], state["forward_world_xy"], state["chain_valid"],
        completion_event=chain_completion_event, completed_side=completed_side,
        riser_world_xy=chosen_xy, upper_z=chosen_upper_z,
        simultaneous_completion=side_event.all(dim=1),
        heading_world_xy=heading, step_width=float(env.cfg.v133_transition.step_width),
        longitudinal_tolerance=float(env.cfg.v133_transition.longitudinal_tolerance),
        lateral_tolerance=float(env.cfg.v133_transition.lateral_tolerance),
        minimum_height_gain=float(env.cfg.v133_transition.minimum_height_gain),
    )
    state.update(transition)
    failure = state["repeat_event"] | state["skip_event"] | state["simultaneous_event"]
    # ObservationManager calls critic terms once for shape inference before
    # TerminationManager exists. That bootstrap call cannot be terminal.
    terminal = (
        env.termination_manager.terminated | env.termination_manager.time_outs
        if hasattr(env, "termination_manager")
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    progress = v133_update_adjacent_progress(
        state["adjacent_progress_best"], state["adjacent_progress_escrow"],
        adjacent_potential, target_valid,
        correct_completion=state["adjacent_event"], failed_completion=failure,
        terminal=terminal,
    )
    for name in ("best", "escrow", "delta", "clawback"):
        state[f"adjacent_progress_{name}"] = progress[name]
    state["adjacent_progress_active"] = target_valid
    state["adjacent_progress_delta_sum"] += progress["delta"].sum()
    state["adjacent_progress_clawback_sum"] += progress["clawback"].sum()
    state["adjacent_progress_active_samples"] += target_valid.sum()
    # Support itself is already contact-backed and multi-frame confirmed.  Each
    # B3+ candidate therefore settles independently in its own control step;
    # there is no single pending slot that could merge overlapping windows.
    state["chain_event"] = state["chain_candidate_event"]
    state["episode_chain_success_count"] += state["chain_event"].long()
    state["episode_chain_failure_count"] += failure.long()
    state["b1_seed_miss_count"] += state["b1_seed_miss_event"].sum()
    for event, count in (
        ("seed_event", "seed_count"), ("adjacent_event", "adjacent_count"),
        ("chain_event", "chain_count"), ("repeat_event", "repeat_count"),
        ("follower_event", "follower_count"), ("skip_event", "skip_count"),
        ("simultaneous_event", "simultaneous_count"),
    ):
        state[count] += state[event].sum()
    state["chain_fl_count"] += (state["chain_event"] & (state["last_side"] == 0)).sum()
    state["chain_fr_count"] += (state["chain_event"] & (state["last_side"] == 1)).sum()
    env._v133_chain_last_step = step
    return state


def v133_three_step_chain_reward(env, sensor_cfg=None) -> torch.Tensor:
    state = v133_three_step_chain_state(env, sensor_cfg=sensor_cfg)
    cfg = env.cfg.v133_transition
    settlement = (
        float(cfg.adjacent_progress_weight) * (
            state["adjacent_progress_delta"] - state["adjacent_progress_clawback"]
        )
        + float(cfg.adjacent_transition_settlement) * state["adjacent_event"].float()
        + float(cfg.chain_settlement) * state["chain_event"].float()
        - float(cfg.repeat_transition_cost) * state["repeat_event"].float()
        - float(cfg.skip_transition_cost) * state["skip_event"].float()
        - float(cfg.simultaneous_transition_cost) * state["simultaneous_event"].float()
    )
    # Positive edges can also occur on boxes, rough ground and slopes.  Keep the
    # chain objective exclusive to the actual ascending-stair terrain column.
    return settlement * v133_ascending_stair_mask(env).float() / float(env.step_dt)


def v133_chain_metric(env, env_ids, key: str) -> torch.Tensor:
    del env_ids
    state = v133_three_step_chain_state(env)
    if key.endswith("_count") and key in state:
        return state[key].float()
    if key == "chain_precision":
        failures = (
            state["repeat_count"] + state["skip_count"] + state["simultaneous_count"]
        )
        return state["chain_count"].float() / (state["chain_count"] + failures).clamp_min(1).float()
    if key == "chain_fl_fraction":
        return state["chain_fl_count"].float() / state["chain_count"].clamp_min(1).float()
    if key == "chain_active_fraction":
        return state["chain_valid"].float().mean()
    if key == "b1_locked_rate":
        return state["b1_locked"].float().mean()
    if key == "b1_seed_miss_rate":
        misses = state["b1_seed_miss_count"].float()
        return misses / (state["seed_count"] + state["b1_seed_miss_count"]).clamp_min(1).float()
    if key == "chain_length_mean":
        return state["chain_length"].float().mean()
    if key == "promotion_eligible_rate":
        debug = getattr(env, "_v11_terrain_curriculum_debug", {})
        return debug.get("v133_chain_promotion_eligible_rate", torch.zeros((), device=env.device))
    if key == "adjacent_progress_active_rate":
        return state["adjacent_progress_active"].float().mean()
    if key in {
        "adjacent_progress_delta_mean", "adjacent_progress_escrow_mean",
        "adjacent_progress_clawback_mean",
    }:
        return state[key.removesuffix("_mean")].float().mean()
    raise KeyError(f"Unknown V13.3 chain metric: {key}")


__all__ = [
    "v133_descending_stair_mask", "v133_ascending_stair_mask",
    "v133_route_official_descent_curriculum", "v133_ascent_world_forward_deficit",
    "v133_ascent_world_forward_progress", "v133_ascent_path_lateral_deviation",
    "v133_ascent_path_heading_error", "v133_ascent_non_wheel_contact",
    "v133_ascent_non_wheel_scrape", "v133_ascent_wheel_impact",
    "v133_ascent_stair_straightness", "v133_ascent_front_riser_contact",
    "v133_official_descent_flat_orientation_correction",
    "v133_official_descent_lin_vel_z_correction",
    "v133_official_descent_action_rate_correction",
    "v133_route_action_rate", "v133_route_action_smooth",
    "v133_route_front_joint_pos_penalty",
    "v133_official_descent_ang_vel_xy_correction",
    "v133_official_descent_joint_pos_cost",
    "v133_official_descent_joint_limit_cancellation",
    "v133_cancel_nonofficial_descent_reward",
    "v133_nonofficial_illegal_contact_termination",
    "v133_ascent_front_riser_collision_termination",
    "v133_confirm_distinct_front_support", "v133_episode_chain_quality",
    "v133_episode_reset_mask", "v133_update_adjacent_progress", "v133_checkpoint_qualified",
    "v133_update_three_step_chain", "v133_match_locked_first_riser",
    "v133_three_step_chain_state", "v133_three_step_chain_reward", "v133_chain_metric",
]
