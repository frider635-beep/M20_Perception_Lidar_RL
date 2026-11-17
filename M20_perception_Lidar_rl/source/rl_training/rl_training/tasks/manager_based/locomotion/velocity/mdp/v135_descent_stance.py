"""V13.5 post-stair flat recovery without prescribing a stair gait.

The native V13 objective remains authoritative while a stair is being crossed.
This module only settles a bounded outcome after local geometry confirms that a
real stair traversal has ended and all four wheels have returned to one plane.
No state from this module enters the actor observation or gates an action.
"""

from __future__ import annotations

import torch


def _scale(env) -> float:
    """The restored descent objective is active at full strength from step zero."""
    del env
    return 1.0


def _zeros(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def v135_historical_descent_follow_reward(env, sensor_cfg=None) -> torch.Tensor:
    scale = _scale(env)
    if scale == 0.0:
        return _zeros(env)
    from .v134_perceptive_follow import v134_descent_follow_reward
    return float(scale) * v134_descent_follow_reward(env, sensor_cfg=sensor_cfg)


def v135_historical_descent_follow_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    scale = _scale(env)
    if scale == 0.0:
        return _zeros(env)
    from .v134_perceptive_follow import v134_descent_follow_completion_reward
    return float(scale) * v134_descent_follow_completion_reward(env, sensor_cfg=sensor_cfg)


def v135_historical_descent_active_extension_penalty(
    env, free_reach: float, reach_scale: float, max_penalty: float, sensor_cfg=None,
) -> torch.Tensor:
    scale = _scale(env)
    if scale == 0.0:
        return _zeros(env)
    from .v134_perceptive_follow import v134_descent_active_extension_penalty
    return float(scale) * v134_descent_active_extension_penalty(
        env, free_reach=free_reach, reach_scale=reach_scale,
        max_penalty=max_penalty, sensor_cfg=sensor_cfg,
    )


def v135_historical_descent_active_roll_penalty(env, sensor_cfg=None) -> torch.Tensor:
    scale = _scale(env)
    if scale == 0.0:
        return _zeros(env)
    from .v134_perceptive_follow import v134_descent_active_roll_penalty
    return float(scale) * v134_descent_active_roll_penalty(env, sensor_cfg=sensor_cfg)


def v135_cancel_native_descent_step_completion(env, sensor_cfg=None) -> torch.Tensor:
    scale = _scale(env)
    if scale == 0.0:
        return _zeros(env)
    from .v11_continuous import v11_descent_step_complete_reward
    return float(scale) * v11_descent_step_complete_reward(env, sensor_cfg=sensor_cfg)


def v135_descent_rear_knee_folding_penalty(env, sensor_cfg=None) -> torch.Tensor:
    """V13.5-Final: prevent rear knee folding only during active descent follow.

    This replaces the global v111 descent knee penalty with a targeted constraint
    that only activates during the rear-follow window, preventing the extreme
    folding behavior without restricting natural rear-leg motion at other times.
    """
    from .v134_perceptive_follow import v134_descent_follow_state
    from .v111_rear_posture import v111_rear_knee_flexion_cost

    follow_state = v134_descent_follow_state(env, sensor_cfg=sensor_cfg)

    asset = env.scene["robot"]
    ids = getattr(env, "_v111_rear_knee_joint_ids", None)
    if ids is None:
        ids, names = asset.find_joints(
            ["hl_knee_joint", "hr_knee_joint"], preserve_order=True,
        )
        if names != ["hl_knee_joint", "hr_knee_joint"]:
            raise RuntimeError(f"V135-Final could not resolve rear knee joints: {names}.")
        env._v111_rear_knee_joint_ids = ids

    deviation = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]

    cfg = env.cfg.v135_descent_stance
    cost = v111_rear_knee_flexion_cost(
        deviation,
        free_flexion=float(cfg.rear_knee_flexion_free),
        flexion_scale=float(cfg.rear_knee_flexion_scale),
        max_penalty=float(cfg.rear_knee_flexion_max_penalty),
    )

    # Only active during descent follow window
    return -cost * follow_state["active"].float()


def v135_descent_rear_extension_penalty(env, sensor_cfg=None) -> torch.Tensor:
    """V13.5-Final: prevent excessive rearward reach during descent follow."""
    from .v134_perceptive_follow import v134_descent_follow_state
    from .v13_alternation import v13_rear_extension_cost

    follow_state = v134_descent_follow_state(env, sensor_cfg=sensor_cfg)

    cfg = env.cfg.v135_descent_stance
    cost = v13_rear_extension_cost(
        follow_state["rearward_extension"],
        free_reach=float(cfg.rear_extension_free_reach),
        reach_scale=float(cfg.rear_extension_scale),
        max_penalty=float(cfg.rear_extension_max_penalty),
    )

    return -cost * follow_state["active"].float()


def v135_descent_rear_roll_penalty(env, sensor_cfg=None) -> torch.Tensor:
    """V13.5-Final: encourage rear wheels to roll naturally during descent."""
    from .v134_perceptive_follow import v134_descent_follow_state

    follow_state = v134_descent_follow_state(env, sensor_cfg=sensor_cfg)

    return -follow_state["rear_roll_error"] * follow_state["active"].float()


def v135_cancel_native_descent_follow_delay(env, sensor_cfg=None) -> torch.Tensor:
    scale = _scale(env)
    if scale == 0.0:
        return _zeros(env)
    from .v11_continuous import v11_descent_rear_follow_delay_penalty
    return float(scale) * v11_descent_rear_follow_delay_penalty(env, sensor_cfg=sensor_cfg)


def v135_cancel_native_descent_roll_consistency(env, sensor_cfg=None) -> torch.Tensor:
    scale = _scale(env)
    if scale == 0.0:
        return _zeros(env)
    from .v11_continuous import v11_descent_rear_roll_consistency_penalty
    return float(scale) * v11_descent_rear_roll_consistency_penalty(env, sensor_cfg=sensor_cfg)


def v135_historical_follow_metric(env, env_ids, key: str) -> torch.Tensor:
    del env_ids
    scale = _scale(env)
    if key == "schedule_scale":
        return torch.tensor(scale, device=env.device)
    if scale == 0.0:
        return torch.zeros((), device=env.device)
    from .v134_perceptive_follow import v134_metric
    return v134_metric(env, None, key)


def v135_post_stair_quality(
    clearance_ratio: torch.Tensor,
    gravity_xy_norm: torch.Tensor,
    speed_error: torch.Tensor,
    all_contact: torch.Tensor,
    coplanar: torch.Tensor,
    *,
    clearance_free_ratio: float,
    clearance_scale_ratio: float,
    upright_std: float,
    velocity_std: float,
) -> torch.Tensor:
    """Score a normal flat-running outcome using body-level quantities only."""
    shape = clearance_ratio.shape
    if any(value.shape != shape for value in (gravity_xy_norm, speed_error, all_contact, coplanar)):
        raise ValueError("V13.5 post-stair quality tensors must be aligned one-dimensional values.")
    if not 0.0 < float(clearance_free_ratio) <= 1.0:
        raise ValueError("V13.5 free clearance ratio must lie in (0, 1].")
    if min(float(clearance_scale_ratio), float(upright_std), float(velocity_std)) <= 0.0:
        raise ValueError("V13.5 recovery scales must be positive.")
    clearance = (
        (clearance_ratio - (float(clearance_free_ratio) - float(clearance_scale_ratio)))
        / float(clearance_scale_ratio)
    ).clamp(0.0, 1.0)
    upright = torch.exp(-torch.square(gravity_xy_norm / float(upright_std)))
    velocity = torch.exp(-torch.square(speed_error / float(velocity_std)))
    eligible = all_contact & coplanar
    return eligible.float() * (0.50 * clearance + 0.25 * upright + 0.25 * velocity)


def _initialize(env, asset) -> None:
    if hasattr(env, "_v135_post_stair"):
        return
    sensor = env.scene.sensors["contact_forces"]
    body_ids, contact_ids = [], []
    for name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"):
        found, names = asset.find_bodies(name)
        contact_found, contact_names = sensor.find_bodies(name)
        if len(found) != 1 or names != [name]:
            raise RuntimeError(f"V13.5 expected one {name!r} body, got {names}")
        if len(contact_found) != 1 or contact_names != [name]:
            raise RuntimeError(f"V13.5 expected one {name!r} contact body, got {contact_names}")
        body_ids.append(found[0])
        contact_ids.append(contact_found[0])
    env._v135_wheel_body_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)
    env._v135_wheel_contact_ids = torch.tensor(contact_ids, device=env.device, dtype=torch.long)
    wheel_bottom = asset.data.body_pos_w[:, env._v135_wheel_body_ids, 2] - float(
        env.cfg.v11_perception.wheel_radius
    )
    clearance = asset.data.root_pos_w[:, 2] - wheel_bottom.mean(dim=1)
    n = env.num_envs
    env._v135_post_stair = {
        "last_episode_length": env.episode_length_buf.clone(),
        "neutral_clearance": clearance.detach().clamp_min(1.0e-3),
        "traversal_seen": torch.zeros(n, device=env.device, dtype=torch.bool),
        "exit_edge": torch.zeros((n, 2), device=env.device),
        "exit_edge_seen": torch.zeros(n, device=env.device, dtype=torch.bool),
        "recovery_started": torch.zeros(n, device=env.device, dtype=torch.bool),
        "no_riser_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "quality_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "best_quality": torch.zeros(n, device=env.device),
        "quality_delta": torch.zeros(n, device=env.device),
        "settled": torch.zeros(n, device=env.device, dtype=torch.bool),
        "flat_mode": torch.zeros(n, device=env.device, dtype=torch.bool),
        "flat_blend_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "flat_blend": torch.zeros(n, device=env.device),
        "stair_reentry_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "completion_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "completion_count": torch.zeros((), device=env.device, dtype=torch.long),
    }


def v135_post_stair_flat_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Track one fail-closed flat recovery settlement per traversed stair flight."""
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_v135_post_stair_last_step", None) == step:
        return env._v135_post_stair

    from .v11_continuous import v11_state

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    asset = env.scene["robot"]
    _initialize(env, asset)
    state = env._v135_post_stair
    cfg = env.cfg.v135_post_stair_flat

    wheel_bottom = asset.data.body_pos_w[:, env._v135_wheel_body_ids, 2] - float(
        env.cfg.v11_perception.wheel_radius
    )
    current_clearance = asset.data.root_pos_w[:, 2] - wheel_bottom.mean(dim=1)
    reset = env.episode_length_buf < state["last_episode_length"]
    state["neutral_clearance"][reset] = current_clearance[reset].detach().clamp_min(1.0e-3)
    for name in (
        "traversal_seen", "no_riser_steps", "quality_steps", "best_quality",
        "quality_delta", "settled", "flat_mode", "flat_blend_steps", "flat_blend",
        "stair_reentry_steps", "completion_event", "exit_edge_seen", "recovery_started",
    ):
        state[name][reset] = 0
    state["last_episode_length"] = env.episode_length_buf.clone()

    landing = base["landing"]
    descent = base["descent"]

    from .v135_descent_support_transfer import v135_descent_rear_support_transfer_state
    rear_transfer = v135_descent_rear_support_transfer_state(env, sensor_cfg=sensor_cfg)

    # FLAT persists across isolated range-sensor glitches.  A new stair flight
    # clears it only after a real ascent/descent target remains active for three
    # consecutive frames, at which point recovery is armed for the next exit.
    directional_exit = bool(getattr(getattr(env.cfg, "v111_rear_posture", None),
                                    "directional_exit_evidence", False))
    stair_target_active = landing["target_active"] | descent["target_active"]
    if directional_exit:
        from .v111_exit_evidence import directional_riser_masks
        no_forward_riser, new_forward_riser = directional_riser_masks(env, base)
        # A retained target behind the robot is not a new stair flight.
        stair_target_active &= new_forward_riser
    state["stair_reentry_steps"] = torch.where(
        (state["flat_mode"] | state["recovery_started"]) & stair_target_active,
        state["stair_reentry_steps"] + 1,
        torch.zeros_like(state["stair_reentry_steps"]),
    )
    new_flight = state["stair_reentry_steps"] >= 3
    state["flat_mode"][new_flight] = False
    state["recovery_started"][new_flight] = False
    state["exit_edge_seen"][new_flight] = False
    state["traversal_seen"][new_flight] = False
    state["settled"][new_flight] = False
    state["no_riser_steps"][new_flight] = 0
    state["quality_steps"][new_flight] = 0
    state["best_quality"][new_flight] = 0.0
    state["stair_reentry_steps"][new_flight] = 0
    state["flat_blend_steps"][new_flight] = 0
    state["flat_blend"][new_flight] = 0.0

    # Refresh the physical edge while visible. An old intermediate completion
    # cannot authorize recovery after an unrelated scanner dropout.
    heading = env.command_manager.get_term("base_velocity").v11_path_heading_w
    forward = torch.stack((heading.cos(), heading.sin()), 1)
    observed = landing["target_active"] | descent["target_active"]
    reference = torch.where(descent["target_active"][:, None],
                            descent["reference_world_xy"], landing["reference_riser_world_xy"])
    state["exit_edge"] = torch.where(observed[:, None], reference, state["exit_edge"])
    state["exit_edge_seen"] |= observed
    wheel_xy = asset.data.body_pos_w[:, env._v135_wheel_body_ids, :2]
    from .v135_exit_geometry import platform_exit_evidence
    raw_no_riser = (~base["valid"].any(dim=1) if not directional_exit else no_forward_riser)
    geometric_exit = platform_exit_evidence(
        wheel_xy, state["exit_edge"], state["exit_edge_seen"], asset.data.root_pos_w[:, :2],
        forward, env.scene.sensors["v10_pgtt_scanner"].data.ray_hits_w, raw_no_riser,
        radius=float(env.cfg.v11_perception.wheel_radius), plane_tolerance=float(cfg.wheel_plane_tolerance),
        lookahead=float(cfg.platform_lookahead), half_width=float(cfg.platform_half_width))
    state["no_riser_steps"] = torch.where(
        geometric_exit, state["no_riser_steps"] + 1, torch.zeros_like(state["no_riser_steps"]))
    state["recovery_started"] |= state["no_riser_steps"] >= int(cfg.no_riser_confirmation_frames)
    state["traversal_seen"] |= state["recovery_started"]

    forces_z = env.scene.sensors["contact_forces"].data.net_forces_w[
        :, env._v135_wheel_contact_ids, 2
    ]
    all_contact = (forces_z > float(env.cfg.v11_perception.contact_force_threshold)).all(dim=1)
    wheel_plane_span = wheel_bottom.amax(dim=1) - wheel_bottom.amin(dim=1)
    coplanar = wheel_plane_span <= float(cfg.wheel_plane_tolerance)
    candidate = (
        ~state["settled"]
        & state["recovery_started"] & raw_no_riser
    )
    clearance_ratio = current_clearance / state["neutral_clearance"].clamp_min(1.0e-3)
    gravity_xy_norm = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    speed_error = (asset.data.root_lin_vel_b[:, 0] - command_x).abs()
    quality = v135_post_stair_quality(
        clearance_ratio, gravity_xy_norm, speed_error, all_contact, coplanar,
        clearance_free_ratio=cfg.clearance_free_ratio,
        clearance_scale_ratio=cfg.clearance_scale_ratio,
        upright_std=cfg.upright_std,
        velocity_std=cfg.velocity_std,
    )
    new_best = torch.where(candidate, torch.maximum(state["best_quality"], quality), state["best_quality"])
    state["quality_delta"] = (new_best - state["best_quality"]).clamp_min(0.0)
    state["best_quality"] = new_best
    stable = candidate & all_contact & coplanar & (quality >= float(cfg.minimum_quality))
    state["quality_steps"] = torch.where(
        stable, state["quality_steps"] + 1, torch.zeros_like(state["quality_steps"])
    )
    event = (~state["settled"]) & (
        state["quality_steps"] >= int(cfg.support_confirmation_frames)
    )
    state["completion_event"] = event
    state["settled"] |= event
    # V13.5-Final: Enter persistent flat_mode after settlement confirmation
    state["flat_mode"] |= event
    state["flat_blend_steps"] = torch.where(
        state["recovery_started"],
        state["flat_blend_steps"] + 1,
        torch.zeros_like(state["flat_blend_steps"]),
    )
    blend_horizon = max(1, int(env.cfg.v135_longtrain.flat_blend_steps))
    state["flat_blend"] = (state["flat_blend_steps"].float() / float(blend_horizon)).clamp(0.0, 1.0)
    state["completion_count"] += event.sum()
    state.update({
        "active": candidate,
        "raw_no_riser": raw_no_riser,
        "all_contact": all_contact,
        "coplanar": coplanar,
        "wheel_plane_span": wheel_plane_span,
        "clearance_ratio": clearance_ratio,
        "quality": quality,
    })
    env._v135_post_stair_last_step = step
    return state


def v135_post_stair_flat_progress_reward(env, sensor_cfg=None) -> torch.Tensor:
    return v135_post_stair_flat_state(env, sensor_cfg=sensor_cfg)["quality_delta"] / float(env.step_dt)


def v135_post_stair_flat_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    event = v135_post_stair_flat_state(env, sensor_cfg=sensor_cfg)["completion_event"]
    return event.float() / float(env.step_dt)


def v135_post_stair_flat_crawl_penalty(env, sensor_cfg=None) -> torch.Tensor:
    """V13.5-Final: Persistent low-clearance penalty during flat_mode.

    This prevents the policy from crouching or crawling after stair exit,
    but only activates after confirmed flat settlement, so it never interferes
    with natural stair gait (pitch, crouch, or leg extension during ascent/descent).
    """
    state = v135_post_stair_flat_state(env, sensor_cfg=sensor_cfg)
    cfg = env.cfg.v135_post_stair_flat

    # Only active in persistent flat mode
    if not state["recovery_started"].any():
        return torch.zeros(env.num_envs, device=env.device)

    clearance_ratio = state["clearance_ratio"]
    # deficit = clamp((0.90 - clearance_ratio) / 0.12, min=0)
    # cost = clamp(deficit^2, max=1)
    deficit = ((float(cfg.crawl_clearance_threshold) - clearance_ratio)
               / float(cfg.crawl_clearance_scale)).clamp_min(0.0)
    cost = torch.square(deficit).clamp_max(1.0)

    # Only penalize in flat_mode & no_riser
    active = state["recovery_started"] & state["raw_no_riser"]
    return -cost * active.float() * state["flat_blend"]


def v135_post_stair_flat_metric(env, env_ids, key: str) -> torch.Tensor:
    del env_ids
    state = v135_post_stair_flat_state(env)
    active = state["active"]
    if key == "traversal_seen_rate":
        return state["traversal_seen"].float().mean()
    if key == "no_riser_after_stair_rate":
        return (state["traversal_seen"] & state["raw_no_riser"]).float().mean()
    if key == "active_rate":
        return active.float().mean()
    if key == "contact_coplanar_rate":
        return (state["all_contact"] & state["coplanar"]).float().mean()
    if key in ("clearance_ratio", "quality"):
        value = state[key]
        return (value * active.float()).sum() / active.float().sum().clamp_min(1.0)
    if key == "flat_blend":
        return state["flat_blend"].mean()
    if key == "completion_event_rate":
        return state["completion_event"].float().mean()
    if key == "completion_count":
        return state["completion_count"].float()
    raise ValueError(f"Unknown V13.5 post-stair metric: {key!r}")


__all__ = [
    "v135_historical_descent_follow_reward",
    "v135_historical_descent_follow_completion_reward",
    "v135_descent_rear_knee_folding_penalty",
    "v135_descent_rear_extension_penalty",
    "v135_descent_rear_roll_penalty",
    "v135_historical_descent_active_extension_penalty",
    "v135_historical_descent_active_roll_penalty",
    "v135_cancel_native_descent_step_completion",
    "v135_cancel_native_descent_follow_delay",
    "v135_cancel_native_descent_roll_consistency",
    "v135_historical_follow_metric",
    "v135_post_stair_quality",
    "v135_post_stair_flat_state",
    "v135_post_stair_flat_progress_reward",
    "v135_post_stair_flat_completion_reward",
    "v135_post_stair_flat_crawl_penalty",
    "v135_post_stair_flat_metric",
]
