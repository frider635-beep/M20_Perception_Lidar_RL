"""V11.1 bounded rear-leg posture repair for descending stairs."""

from __future__ import annotations

import torch


def v111_track_lin_vel_with_descent_support(env, std: float, command_name: str):
    from .rewards import track_lin_vel_xy_exp
    from .v135_descent_support_transfer import v135_descent_velocity_modulation
    return track_lin_vel_xy_exp(env, std=std, command_name=command_name) * v135_descent_velocity_modulation(env)


def v111_world_forward_with_descent_support(env, target_speed_fraction: float = 0.5):
    from .v11_continuous import v11_world_forward_progress_reward
    from .v135_descent_support_transfer import v135_descent_velocity_modulation
    return v11_world_forward_progress_reward(env, target_speed_fraction=target_speed_fraction) * v135_descent_velocity_modulation(env)


def v111_world_forward_deficit_with_descent_support(env, min_speed_fraction: float = 0.5,
                                                    speed_scale: float = 0.3):
    from .v11_continuous import v11_world_forward_speed_deficit
    from .v135_descent_support_transfer import v135_descent_rear_support_transfer_state
    native = v11_world_forward_speed_deficit(env, min_speed_fraction=min_speed_fraction,
                                             speed_scale=speed_scale)
    transfer = v135_descent_rear_support_transfer_state(env)["active"]
    return torch.where(transfer, torch.zeros_like(native), native)


def v111_low_body_clearance_cost(
    body_clearance: torch.Tensor,
    *,
    free_height: float,
    height_scale: float,
    max_penalty: float,
) -> torch.Tensor:
    """Bound only the collapsed-body extreme after reaching level ground."""
    if body_clearance.ndim != 1:
        raise ValueError("V11.1 body clearance must be one-dimensional.")
    if free_height <= 0.0 or height_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("V11.1 body-clearance bounds must be positive.")
    deficit = (float(free_height) - body_clearance).clamp_min(0.0)
    return (deficit / float(height_scale)).square().clamp_max(float(max_penalty))


def v111_confirm_level_exit(
    previous_steps: torch.Tensor,
    stair_started: torch.Tensor,
    no_visible_riser: torch.Tensor,
    all_wheels_supported: torch.Tensor,
    support_height_spread: torch.Tensor,
    *,
    support_height_tolerance: float,
    confirmation_frames: int,
) -> dict[str, torch.Tensor]:
    """Confirm a level stair exit without retaining or naming a stair tread."""
    shape = previous_steps.shape
    if previous_steps.ndim != 1 or any(value.shape != shape for value in (
        stair_started, no_visible_riser, all_wheels_supported, support_height_spread,
    )):
        raise ValueError("V11.1 level-exit tensors must be aligned one-dimensional values.")
    if support_height_tolerance <= 0.0 or confirmation_frames < 2:
        raise ValueError("V11.1 level-exit confirmation bounds are invalid.")
    candidate = (
        stair_started & no_visible_riser & all_wheels_supported
        & (support_height_spread <= float(support_height_tolerance))
    )
    steps = torch.where(candidate, previous_steps + 1, torch.zeros_like(previous_steps))
    return {"candidate": candidate, "steps": steps, "confirmed": steps >= int(confirmation_frames)}


def v111_rear_knee_flexion_cost(
    signed_deviation: torch.Tensor,
    free_flexion: float,
    flexion_scale: float,
    max_penalty: float,
) -> torch.Tensor:
    """Bound only rear-knee folding beyond the configured free region."""
    if signed_deviation.ndim != 2 or signed_deviation.shape[1] != 2:
        raise ValueError("V11.1 rear-knee deviation must have shape [num_envs, 2].")
    if free_flexion < 0.0 or flexion_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("V11.1 rear-knee flexion bounds must be non-negative and bounded.")
    excess = (-signed_deviation - float(free_flexion)).clamp_min(0.0)
    return ((excess / float(flexion_scale)).square().clamp_max(float(max_penalty))).mean(dim=1)


def v111_descent_rear_extension_penalty(
    env, free_reach: float, reach_scale: float, max_penalty: float, sensor_cfg=None,
) -> torch.Tensor:
    """Penalize only excessive rearward reach on descending stair columns."""
    from .v13_alternation import v13_descent_rear_motion_state, v13_rear_extension_cost

    state = v13_descent_rear_motion_state(env, sensor_cfg=sensor_cfg)
    cost = v13_rear_extension_cost(
        state["rearward_extension"], free_reach=free_reach,
        reach_scale=reach_scale, max_penalty=max_penalty,
    )
    return -cost * state["course_active"].float()


def v111_descent_rear_knee_flexion_penalty(
    env, free_flexion: float, flexion_scale: float, max_penalty: float,
) -> torch.Tensor:
    """Prevent the rear knee/wheel folding shortcut without prescribing steps."""
    from .v13_alternation import v13_descending_course_mask

    asset = env.scene["robot"]
    ids = getattr(env, "_v111_rear_knee_joint_ids", None)
    if ids is None:
        ids, names = asset.find_joints(
            ["hl_knee_joint", "hr_knee_joint"], preserve_order=True,
        )
        if names != ["hl_knee_joint", "hr_knee_joint"]:
            raise RuntimeError(f"V11.1 could not resolve rear knee joints: {names}.")
        env._v111_rear_knee_joint_ids = ids
    deviation = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    cost = v111_rear_knee_flexion_cost(
        deviation, free_flexion=free_flexion,
        flexion_scale=flexion_scale, max_penalty=max_penalty,
    )
    return -cost * v13_descending_course_mask(env).float()


def v111_descent_rear_roll_penalty(env, sensor_cfg=None) -> torch.Tensor:
    """Encourage contacted rear wheels to roll with the body during descent."""
    from .v13_alternation import v13_descent_rear_motion_state

    state = v13_descent_rear_motion_state(env, sensor_cfg=sensor_cfg)
    return -state["rear_roll_error"] * state["course_active"].float()


def v111_level_exit_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Latch level ground after ascent or descent using contact-backed evidence."""
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_v111_level_exit_last_step", None) == step:
        return env._v111_level_exit

    from .v11_continuous import v11_state

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    cfg = env.cfg.v111_rear_posture
    asset = env.scene["robot"]
    terrain_types = env.scene.terrain.terrain_types
    perception = env.cfg.v11_perception
    ascending_values = tuple(int(value) for value in perception.ascending_stair_terrain_type_ids)
    stair_values = set(int(value) for value in perception.stair_terrain_type_ids)
    descending_values = tuple(sorted(stair_values - set(ascending_values)))
    ascending_ids = torch.tensor(ascending_values, device=env.device, dtype=terrain_types.dtype)
    descending_ids = torch.tensor(descending_values, device=env.device, dtype=terrain_types.dtype)
    ascending = (terrain_types[:, None] == ascending_ids[None, :]).any(dim=1)
    descending = (terrain_types[:, None] == descending_ids[None, :]).any(dim=1)
    commanded_forward = env.command_manager.get_command("base_velocity")[:, 0] > float(
        perception.forward_command_threshold
    )

    wheel_ids = torch.cat((env._v11_front_body_ids, env._v11_rear_body_ids))
    contact_ids = torch.cat((env._v11_front_contact_ids, env._v11_rear_contact_ids))
    wheel_bottom = asset.data.body_pos_w[:, wheel_ids, 2] - float(perception.wheel_radius)
    forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, contact_ids, 2]
    all_supported = (forces > float(perception.contact_force_threshold)).all(dim=1)
    support_spread = wheel_bottom.amax(dim=1) - wheel_bottom.amin(dim=1)
    support_z = wheel_bottom.mean(dim=1)
    no_riser = ~base["risers"]["valid"].any(dim=1)

    state = getattr(env, "_v111_level_exit", None)
    if state is None:
        state = {
            "last_episode_length": env.episode_length_buf.clone(),
            "top_steps": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
            "bottom_steps": torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
            "top_active": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            "bottom_active": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        }
        env._v111_level_exit = state
    reset = env.episode_length_buf < state["last_episode_length"]
    for name in ("top_steps", "bottom_steps", "top_active", "bottom_active"):
        state[name][reset] = 0
    state["last_episode_length"] = env.episode_length_buf.clone()

    top = v111_confirm_level_exit(
        state["top_steps"],
        ascending & commanded_forward & base["landing"]["sequence_started"],
        no_riser, all_supported, support_spread,
        support_height_tolerance=float(cfg.exit_support_height_tolerance),
        confirmation_frames=int(cfg.exit_confirmation_frames),
    )
    bottom = v111_confirm_level_exit(
        state["bottom_steps"],
        descending & commanded_forward & base["descent"]["sequence_started"],
        no_riser, all_supported, support_spread,
        support_height_tolerance=float(cfg.exit_support_height_tolerance),
        confirmation_frames=int(cfg.exit_confirmation_frames),
    )
    state["top_steps"] = top["steps"]
    state["bottom_steps"] = bottom["steps"]
    state["top_active"] |= top["confirmed"]
    state["bottom_active"] |= bottom["confirmed"]
    state.update({
        "active": state["top_active"] | state["bottom_active"],
        "body_clearance": asset.data.root_pos_w[:, 2] - support_z,
        "support_height_spread": support_spread,
        "all_wheels_supported": all_supported,
    })
    env._v111_level_exit_last_step = step
    return state


def v111_level_exit_body_clearance_penalty(
    env, free_height: float, height_scale: float, max_penalty: float, sensor_cfg=None,
) -> torch.Tensor:
    """Prevent crawling only after a contact-confirmed top or bottom exit."""
    state = v111_level_exit_state(env, sensor_cfg=sensor_cfg)
    cost = v111_low_body_clearance_cost(
        state["body_clearance"], free_height=free_height,
        height_scale=height_scale, max_penalty=max_penalty,
    )
    return -cost * state["active"].float()


def v111_rear_posture_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose the retained pre-thigh-alignment descent envelope."""
    del env_ids
    from .v13_alternation import v13_descent_rear_motion_state

    state = v13_descent_rear_motion_state(env)
    course = state["course_active"]
    count = course.float().sum().clamp_min(1.0)
    if key == "course_active_rate":
        return course.float().mean()
    if key == "rear_extension_max_mean":
        return (state["rearward_extension"].amax(dim=1) * course.float()).sum() / count
    if key == "rear_contact_fraction":
        return (state["rear_contact_fraction"] * course.float()).sum() / count
    if key == "rear_roll_error_mean":
        return (state["rear_roll_error"] * course.float()).sum() / count
    if key in {"rear_knee_flexion_mean", "rear_knee_flexion_rate"}:
        asset = env.scene["robot"]
        ids = getattr(env, "_v111_rear_knee_joint_ids", None)
        if ids is None:
            ids, names = asset.find_joints(
                ["hl_knee_joint", "hr_knee_joint"], preserve_order=True,
            )
            if names != ["hl_knee_joint", "hr_knee_joint"]:
                raise RuntimeError(f"V11.1 could not resolve rear knee joints: {names}.")
            env._v111_rear_knee_joint_ids = ids
        deviation = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
        flexion = (-deviation).clamp_min(0.0).amax(dim=1)
        if key == "rear_knee_flexion_mean":
            return (flexion * course.float()).sum() / count
        threshold = float(env.cfg.v111_rear_posture.rear_knee_flexion_free)
        return ((flexion > threshold) & course).float().sum() / count
    if key in {
        "top_exit_active_rate", "bottom_exit_active_rate", "exit_body_clearance_mean",
        "exit_low_body_rate",
    }:
        exit_state = v111_level_exit_state(env)
        if key == "top_exit_active_rate":
            return exit_state["top_active"].float().mean()
        if key == "bottom_exit_active_rate":
            return exit_state["bottom_active"].float().mean()
        active = exit_state["active"]
        active_count = active.float().sum().clamp_min(1.0)
        if key == "exit_body_clearance_mean":
            return (exit_state["body_clearance"] * active.float()).sum() / active_count
        threshold = float(env.cfg.v111_rear_posture.exit_body_clearance_free_height)
        return ((exit_state["body_clearance"] < threshold) & active).float().sum() / active_count
    raise KeyError(key)


def v111_handoff_metric(env, env_ids, key):
    from .v11_continuous import v11_state
    landing = v11_state(env)["landing"]
    if key not in ("bilateral_confirmed", "unsupported_handoff", "bilateral_landing_event"):
        raise KeyError(key)
    return landing[key][env_ids].float().mean()


__all__ = [
    "v111_handoff_metric",
    "v111_track_lin_vel_with_descent_support",
    "v111_world_forward_with_descent_support",
    "v111_world_forward_deficit_with_descent_support",
    "v111_low_body_clearance_cost", "v111_confirm_level_exit",
    "v111_rear_knee_flexion_cost", "v111_descent_rear_extension_penalty",
    "v111_descent_rear_knee_flexion_penalty", "v111_descent_rear_roll_penalty",
    "v111_level_exit_state", "v111_level_exit_body_clearance_penalty",
    "v111_rear_posture_metric",
]
