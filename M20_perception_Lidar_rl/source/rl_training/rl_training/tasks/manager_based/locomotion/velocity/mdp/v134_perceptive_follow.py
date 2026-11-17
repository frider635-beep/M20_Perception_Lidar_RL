"""V13.4 critic context and perception-triggered natural rear follow."""

from __future__ import annotations

import torch


def v134_rear_knee_flexion_cost(
    signed_deviation: torch.Tensor,
    free_flexion: float,
    flexion_scale: float,
    max_penalty: float,
) -> torch.Tensor:
    """Return a bounded one-sided cost for knees folding past their default pose."""
    if signed_deviation.ndim != 2 or signed_deviation.shape[1] != 2:
        raise ValueError("V13.4 rear-knee deviation must have shape [num_envs, 2].")
    if free_flexion < 0.0 or flexion_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("V13.4 rear-knee flexion bounds must be non-negative and bounded.")
    excess = (-signed_deviation - float(free_flexion)).clamp_min(0.0)
    return ((excess / float(flexion_scale)).square().clamp_max(float(max_penalty))).mean(dim=1)


def v134_rear_thigh_parallel_cost(
    parallel_cosine: torch.Tensor,
    free_cosine: float,
    cosine_scale: float,
    max_penalty: float,
) -> torch.Tensor:
    """Bound only the extreme where a rear thigh lies along the torso axis."""
    if parallel_cosine.ndim != 2 or parallel_cosine.shape[1] != 2:
        raise ValueError("V13.4 rear-thigh cosine must have shape [num_envs, 2].")
    if not 0.0 <= free_cosine < 1.0 or cosine_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("V13.4 rear-thigh parallel bounds are invalid.")
    excess = (parallel_cosine.abs() - float(free_cosine)).clamp_min(0.0)
    return ((excess / float(cosine_scale)).square().clamp_max(float(max_penalty))).mean(dim=1)


def v134_low_body_clearance_cost(
    body_clearance: torch.Tensor,
    free_height: float,
    height_scale: float,
    max_penalty: float,
) -> torch.Tensor:
    """Return a bounded cost only when the torso collapses near the platform."""
    if body_clearance.ndim != 1:
        raise ValueError("V13.4 platform body clearance must be one-dimensional.")
    if free_height <= 0.0 or height_scale <= 0.0 or max_penalty <= 0.0:
        raise ValueError("V13.4 platform height bounds must be positive.")
    deficit = (float(free_height) - body_clearance).clamp_min(0.0)
    return (deficit / float(height_scale)).square().clamp_max(float(max_penalty))


def v134_platform_clearance_active(
    ascending: torch.Tensor, sequence_started: torch.Tensor,
    no_visible_riser: torch.Tensor, front_on_last_tread: torch.Tensor,
    platform_confirmed: torch.Tensor,
) -> torch.Tensor:
    """Protect the last-riser transition before rear-wheel platform confirmation."""
    if any(value.shape != ascending.shape for value in (
        sequence_started, no_visible_riser, front_on_last_tread, platform_confirmed,
    )):
        raise ValueError("V13.4 platform clearance masks must have matching shapes.")
    return ascending & (
        platform_confirmed
        | (sequence_started & no_visible_riser & front_on_last_tread)
    )


def v134_update_follow_progress(
    previous_best: torch.Tensor,
    previous_escrow: torch.Tensor,
    potential: torch.Tensor,
    active: torch.Tensor,
    completion: torch.Tensor,
    failure: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Accumulate contact-backed progress and claw it back on failed follow."""
    shape = previous_best.shape
    if any(value.shape != shape for value in (
        previous_escrow, potential, active, completion, failure,
    )):
        raise ValueError("V13.4 rear-follow progress tensors must have matching shapes.")
    best = torch.where(active, torch.maximum(previous_best, potential), previous_best)
    delta = (best - previous_best).clamp_min(0.0)
    escrow = previous_escrow + delta
    clawback = torch.where(failure, escrow, torch.zeros_like(escrow))
    settled = completion | failure
    return {
        "best": torch.where(settled, torch.zeros_like(best), best),
        "escrow": torch.where(settled, torch.zeros_like(escrow), escrow),
        "delta": delta,
        "clawback": clawback,
    }


def _v134_initialize_follow(env) -> None:
    if hasattr(env, "_v134_follow"):
        return
    n, device = env.num_envs, env.device
    env._v134_follow = {
        "last_episode_length": env.episode_length_buf.clone(),
        "active": torch.zeros(n, device=device, dtype=torch.bool),
        "age": torch.zeros(n, device=device, dtype=torch.long),
        "baseline_rear_bottom": torch.zeros((n, 2), device=device),
        "expected_drop": torch.zeros(n, device=device),
        "support_steps": torch.zeros((n, 2), device=device, dtype=torch.long),
        "best": torch.zeros(n, device=device),
        "escrow": torch.zeros(n, device=device),
        "potential": torch.zeros(n, device=device),
        "delta": torch.zeros(n, device=device),
        "clawback": torch.zeros(n, device=device),
        "trigger_event": torch.zeros(n, device=device, dtype=torch.bool),
        "completion_event": torch.zeros(n, device=device, dtype=torch.bool),
        "timeout_event": torch.zeros(n, device=device, dtype=torch.bool),
        "previous_front_complete": torch.zeros(n, device=device, dtype=torch.bool),
        "trigger_count": torch.zeros((), device=device, dtype=torch.long),
        "completion_count": torch.zeros((), device=device, dtype=torch.long),
        "timeout_count": torch.zeros((), device=device, dtype=torch.long),
    }
    env._v134_follow_last_step = -1


def v134_descent_follow_state(env, sensor_cfg=None) -> dict[str, torch.Tensor]:
    """Follow a perceived front descent without retaining a world tread target."""
    from .v11_continuous import v11_state
    from .v133_adjacent_transition import v133_descending_stair_mask

    _v134_initialize_follow(env)
    state = env._v134_follow
    step = int(getattr(env, "common_step_counter", 0))
    if env._v134_follow_last_step == step:
        return state

    reset = env.episode_length_buf < state["last_episode_length"]
    for name in (
        "active", "age", "baseline_rear_bottom", "expected_drop", "support_steps",
        "best", "escrow", "potential", "delta", "clawback", "trigger_event",
        "completion_event", "timeout_event", "previous_front_complete",
    ):
        state[name][reset] = 0
    state["last_episode_length"] = env.episode_length_buf.clone()
    state["trigger_event"].zero_()
    state["completion_event"].zero_()
    state["timeout_event"].zero_()
    state["delta"].zero_()
    state["clawback"].zero_()

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    descent = base["descent"]
    asset = env.scene["robot"]
    cfg = env.cfg.v134_follow
    course = v133_descending_stair_mask(env) & (
        env.command_manager.get_command("base_velocity")[:, 0]
        > float(env.cfg.v11_perception.forward_command_threshold)
    )
    rear_world = asset.data.body_pos_w[:, env._v11_rear_body_ids]
    rear_bottom = rear_world[..., 2] - float(env.cfg.v11_perception.wheel_radius)
    rear_forces = env.scene.sensors["contact_forces"].data.net_forces_w[
        :, env._v11_rear_contact_ids
    ]
    rear_contact = rear_forces[..., 2] > float(env.cfg.v11_perception.contact_force_threshold)

    front_complete = descent["front_complete"] & course
    trigger = front_complete & ~state["previous_front_complete"] & ~state["active"]
    valid_trigger = trigger & descent["target_active"]
    state["trigger_event"] = valid_trigger
    state["baseline_rear_bottom"] = torch.where(
        valid_trigger[:, None], rear_bottom, state["baseline_rear_bottom"],
    )
    observed_drop = (
        rear_bottom.mean(dim=1) - descent["target_lower_z"]
    ).clamp(float(cfg.minimum_drop), float(cfg.maximum_drop))
    state["expected_drop"] = torch.where(
        valid_trigger, observed_drop, state["expected_drop"],
    )
    state["active"] |= valid_trigger
    state["age"] = torch.where(
        state["active"], state["age"] + 1, torch.zeros_like(state["age"]),
    )

    drop = state["expected_drop"].clamp_min(float(cfg.minimum_drop))
    drop_progress = (
        (state["baseline_rear_bottom"] - rear_bottom) / drop[:, None]
    ).clamp(0.0, 1.0)
    lower_contact = (
        state["active"][:, None] & rear_contact
        & (drop_progress >= float(cfg.completion_drop_fraction))
    )
    next_steps = torch.where(
        lower_contact, state["support_steps"] + 1,
        torch.zeros_like(state["support_steps"]),
    )
    confirmed = next_steps >= int(cfg.support_confirmation_frames)
    completion = state["active"] & confirmed.all(dim=1)
    timeout = state["active"] & (state["age"] > int(cfg.follow_timeout_frames))
    terminal = (
        env.termination_manager.terminated | env.termination_manager.time_outs
        if hasattr(env, "termination_manager")
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    # A stable support reached on the terminal frame is still a completion.
    # Keep success and failure disjoint so the same escrow cannot settle twice.
    failure = (timeout | (state["active"] & terminal)) & ~completion
    potential = (
        float(cfg.contact_progress_fraction)
        * (drop_progress * rear_contact.float()).mean(dim=1)
        + (1.0 - float(cfg.contact_progress_fraction)) * confirmed.float().mean(dim=1)
    ) * state["active"].float()
    progress = v134_update_follow_progress(
        state["best"], state["escrow"], potential,
        state["active"], completion, failure,
    )
    state["support_steps"] = next_steps
    state["potential"] = potential
    state.update(progress)
    state["completion_event"] = completion
    state["timeout_event"] = timeout
    state["trigger_count"] += valid_trigger.sum()
    state["completion_count"] += completion.sum()
    state["timeout_count"] += timeout.sum()
    state["active"] &= ~(completion | failure)
    state["previous_front_complete"] = front_complete
    env._v134_follow_last_step = step
    return state


def v134_descent_follow_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Dense contact-backed rear descent progress with failed-attempt clawback."""
    state = v134_descent_follow_state(env, sensor_cfg=sensor_cfg)
    return (state["delta"] - state["clawback"]) / float(env.step_dt)


def v134_descent_follow_completion_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Settle only when both rear wheels regain stable lower support."""
    state = v134_descent_follow_state(env, sensor_cfg=sensor_cfg)
    return state["completion_event"].float() / float(env.step_dt)


def v134_descent_active_extension_penalty(
    env, free_reach: float, reach_scale: float, max_penalty: float, sensor_cfg=None,
) -> torch.Tensor:
    """Bound rearward reach only while a perceived front descent awaits follow."""
    from .v13_alternation import v13_descent_rear_motion_state, v13_rear_extension_cost

    follow = v134_descent_follow_state(env, sensor_cfg=sensor_cfg)
    motion = v13_descent_rear_motion_state(env, sensor_cfg=sensor_cfg)
    cost = v13_rear_extension_cost(
        motion["rearward_extension"], free_reach=free_reach,
        reach_scale=reach_scale, max_penalty=max_penalty,
    )
    return -cost * follow["active"].float()


def v134_descent_active_roll_penalty(env, sensor_cfg=None) -> torch.Tensor:
    """Penalize rear-wheel rolling mismatch only in the active follow window."""
    from .v13_alternation import v13_descent_rear_motion_state

    follow = v134_descent_follow_state(env, sensor_cfg=sensor_cfg)
    motion = v13_descent_rear_motion_state(env, sensor_cfg=sensor_cfg)
    return -motion["rear_roll_error"] * follow["active"].float()


def v134_descent_rear_knee_flexion_penalty(
    env, free_flexion: float, flexion_scale: float, max_penalty: float,
) -> torch.Tensor:
    """Discourage folded rear knees on descent without prescribing a gait or tread."""
    from .v133_adjacent_transition import v133_descending_stair_mask

    asset = env.scene["robot"]
    ids = getattr(env, "_v134_rear_knee_joint_ids", None)
    if ids is None:
        ids, names = asset.find_joints(
            ["hl_knee_joint", "hr_knee_joint"], preserve_order=True,
        )
        if names != ["hl_knee_joint", "hr_knee_joint"]:
            raise RuntimeError(f"V13.4 could not resolve rear knee joints: {names}.")
        env._v134_rear_knee_joint_ids = ids
    signed_deviation = (
        asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    )
    cost = v134_rear_knee_flexion_cost(
        signed_deviation, free_flexion, flexion_scale, max_penalty,
    )
    commanded_forward = (
        env.command_manager.get_command("base_velocity")[:, 0]
        > float(env.cfg.v11_perception.forward_command_threshold)
    )
    active = v133_descending_stair_mask(env) & commanded_forward
    return -cost * active.float()


def v134_descent_rear_thigh_parallel_penalty(
    env, free_cosine: float, cosine_scale: float, max_penalty: float,
) -> torch.Tensor:
    """Reject the rear-thigh/torso parallel extreme without choosing footfalls."""
    from .v133_adjacent_transition import v133_descending_stair_mask

    parallel = _v134_rear_thigh_parallel_cosine(env)
    cost = v134_rear_thigh_parallel_cost(
        parallel, free_cosine=free_cosine,
        cosine_scale=cosine_scale, max_penalty=max_penalty,
    )
    commanded_forward = (
        env.command_manager.get_command("base_velocity")[:, 0]
        > float(env.cfg.v11_perception.forward_command_threshold)
    )
    active = v133_descending_stair_mask(env) & commanded_forward
    return -cost * active.float()


def _v134_rear_thigh_parallel_cosine(env) -> torch.Tensor:
    """Measure rear upper-leg alignment with the torso forward axis."""
    asset = env.scene["robot"]
    hip_ids = getattr(env, "_v134_rear_hipy_body_ids", None)
    knee_ids = getattr(env, "_v134_rear_knee_body_ids", None)
    if hip_ids is None or knee_ids is None:
        hip_ids, hip_names = asset.find_bodies(["hl_hipy", "hr_hipy"], preserve_order=True)
        knee_ids, knee_names = asset.find_bodies(["hl_knee", "hr_knee"], preserve_order=True)
        if hip_names != ["hl_hipy", "hr_hipy"] or knee_names != ["hl_knee", "hr_knee"]:
            raise RuntimeError(
                f"V13.4 could not resolve rear thigh bodies: {hip_names}, {knee_names}."
            )
        env._v134_rear_hipy_body_ids = hip_ids
        env._v134_rear_knee_body_ids = knee_ids
    upper_leg = asset.data.body_pos_w[:, knee_ids] - asset.data.body_pos_w[:, hip_ids]
    upper_leg /= torch.linalg.norm(upper_leg, dim=-1, keepdim=True).clamp_min(1.0e-6)
    w, x, y, z = asset.data.root_quat_w.unbind(dim=-1)
    base_forward = torch.stack((
        1.0 - 2.0 * (y.square() + z.square()),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    ), dim=1)
    return (upper_leg * base_forward[:, None, :]).sum(dim=-1).abs().clamp(0.0, 1.0)


def v134_platform_body_clearance_penalty(
    env, free_height: float, height_scale: float, max_penalty: float,
    sensor_cfg=None,
) -> torch.Tensor:
    """Keep the torso off the top tread after contact-backed platform confirmation."""
    active, clearance = _v134_platform_clearance_state(env, sensor_cfg=sensor_cfg)
    cost = v134_low_body_clearance_cost(
        clearance, free_height=free_height,
        height_scale=height_scale, max_penalty=max_penalty,
    )
    return -cost * active.float()


def _v134_platform_clearance_state(env, sensor_cfg=None):
    from .v11_continuous import v11_state
    from .v133_adjacent_transition import v133_ascending_stair_mask

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    landing = base["landing"]
    front_forces = env.scene.sensors["contact_forces"].data.net_forces_w[
        :, env._v11_front_contact_ids
    ]
    front_on_last_tread = (
        (landing["wheel_bottom"] - landing["last_completed_upper_z"][:, None]).abs()
        <= float(env.cfg.v11_perception.upper_support_tolerance)
    ) & (
        front_forces[..., 2] > float(env.cfg.v11_perception.contact_force_threshold)
    )
    active = v134_platform_clearance_active(
        v133_ascending_stair_mask(env),
        landing["sequence_started"],
        ~base["risers"]["valid"].any(dim=1),
        front_on_last_tread.any(dim=1),
        landing["platform_confirmed"],
    )
    clearance = env.scene["robot"].data.root_pos_w[:, 2] - landing["last_completed_upper_z"]
    return active, clearance


def v134_critic_state(env, sensor_cfg=None) -> torch.Tensor:
    """Expose ascent reward history and descent follow physics only to critics."""
    from .v13_alternation import v13_descent_rear_motion_state
    from .v133_adjacent_transition import v133_three_step_chain_state

    ascent = v133_three_step_chain_state(env, sensor_cfg=sensor_cfg)
    descent = v134_descent_follow_state(env, sensor_cfg=sensor_cfg)
    motion = v13_descent_rear_motion_state(env, sensor_cfg=sensor_cfg)
    valid = ascent["chain_valid"]
    expected = (1 - ascent["last_side"]).clamp(0, 1)
    timeout = max(int(env.cfg.v134_follow.follow_timeout_frames), 1)
    return torch.stack((
        (valid & (expected == 0)).float(),
        (valid & (expected == 1)).float(),
        valid.float(),
        ascent["adjacent_progress_active"].float(),
        ascent["adjacent_progress_best"],
        ascent["adjacent_progress_escrow"],
        descent["active"].float(),
        (descent["age"].float() / float(timeout)).clamp(0.0, 1.0),
        descent["expected_drop"],
        descent["potential"],
        motion["rearward_extension"].amax(dim=1),
        motion["rear_contact_fraction"],
    ), dim=1)


def v134_positive_ascent_chain_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Pay observed adjacent ascent success without penalizing exploration."""
    from .v133_adjacent_transition import (
        v133_ascending_stair_mask,
        v133_three_step_chain_state,
    )

    state = v133_three_step_chain_state(env, sensor_cfg=sensor_cfg)
    cfg = env.cfg.v133_transition
    reward = (
        float(cfg.b1_settlement) * state["seed_event"].float()
        + float(cfg.adjacent_progress_weight) * state["adjacent_progress_delta"]
        + float(cfg.adjacent_transition_settlement) * state["adjacent_event"].float()
        + float(cfg.chain_settlement) * state["chain_event"].float()
    )
    return reward * v133_ascending_stair_mask(env).float() / float(env.step_dt)


def v134_native_chain_critic_state(env, sensor_cfg=None) -> torch.Tensor:
    """Expose strict ascent history and tread-free descent facts to the critic."""
    from .v13_alternation import v13_critic_descent_state
    from .v133_adjacent_transition import v133_three_step_chain_state

    ascent = v133_three_step_chain_state(env, sensor_cfg=sensor_cfg)
    valid = ascent["chain_valid"]
    expected = (1 - ascent["last_side"]).clamp(0, 1)
    ascent_state = torch.stack((
        (valid & (expected == 0)).float(),
        (valid & (expected == 1)).float(),
        valid.float(),
        (ascent["chain_length"].float() / 3.0).clamp(0.0, 1.0),
        ascent["adjacent_progress_active"].float(),
        ascent["adjacent_progress_best"],
    ), dim=1)
    return torch.cat(
        (ascent_state, v13_critic_descent_state(env, sensor_cfg=sensor_cfg)), dim=1,
    )


def v134_posture_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose the two newly bounded posture failures for checkpoint review."""
    del env_ids
    from .v11_continuous import v11_state
    from .v133_adjacent_transition import v133_descending_stair_mask

    if key in {"rear_thigh_parallel_mean", "rear_thigh_parallel_rate"}:
        parallel = _v134_rear_thigh_parallel_cosine(env).amax(dim=1)
        active = v133_descending_stair_mask(env)
        count = active.float().sum().clamp_min(1.0)
        if key == "rear_thigh_parallel_mean":
            return (parallel * active.float()).sum() / count
        threshold = float(env.cfg.v133_transition.rear_thigh_parallel_free_cosine)
        return ((parallel > threshold) & active).float().sum() / count
    landing = v11_state(env)["landing"]
    active, clearance = _v134_platform_clearance_state(env)
    if key == "platform_confirmed_rate":
        return landing["platform_confirmed"].float().mean()
    if key == "platform_clearance_active_rate":
        return active.float().mean()
    count = active.float().sum().clamp_min(1.0)
    if key == "platform_body_clearance_mean":
        return (clearance * active.float()).sum() / count
    if key == "platform_low_body_rate":
        threshold = float(env.cfg.v133_transition.platform_body_clearance_free_height)
        return ((clearance < threshold) & active).float().sum() / count
    raise KeyError(f"Unknown V13.4 posture metric: {key}")


def v134_ascent_style_anchor_mask(
    ascending_terrain: torch.Tensor,
    riser_valid: torch.Tensor,
    riser_height: torch.Tensor,
    minimum_height_gain: float,
    forward_command: torch.Tensor,
) -> torch.Tensor:
    """Select ascent-with-visible-riser states; exclude descent and the top."""
    if riser_valid.shape != riser_height.shape or riser_valid.ndim != 2:
        raise ValueError("V13.4 style-anchor risers must be matching [N, slots] tensors.")
    if ascending_terrain.shape != (riser_valid.shape[0],):
        raise ValueError("V13.4 style-anchor terrain mask has the wrong shape.")
    if forward_command.shape != ascending_terrain.shape:
        raise ValueError("V13.4 style-anchor command mask has the wrong shape.")
    visible_ascent = (
        riser_valid & (riser_height >= float(minimum_height_gain))
    ).any(dim=1)
    return ascending_terrain & visible_ascent & forward_command


def v134_metric(env, env_ids, key: str) -> torch.Tensor:
    del env_ids
    state = v134_descent_follow_state(env)
    if key == "active_rate":
        return state["active"].float().mean()
    if key == "trigger_count":
        return state["trigger_count"].float()
    if key == "completion_count":
        return state["completion_count"].float()
    if key == "timeout_count":
        return state["timeout_count"].float()
    if key == "potential_mean":
        active = state["active"].float()
        return (state["potential"] * active).sum() / active.sum().clamp_min(1.0)
    if key == "completion_rate":
        return state["completion_count"].float() / state["trigger_count"].clamp_min(1).float()
    raise KeyError(f"Unknown V13.4 metric: {key}")


__all__ = [
    "v134_update_follow_progress", "v134_descent_follow_state",
    "v134_descent_follow_reward", "v134_descent_follow_completion_reward",
    "v134_descent_active_extension_penalty", "v134_descent_active_roll_penalty",
    "v134_rear_knee_flexion_cost", "v134_descent_rear_knee_flexion_penalty",
    "v134_rear_thigh_parallel_cost", "v134_descent_rear_thigh_parallel_penalty",
    "v134_low_body_clearance_cost", "v134_platform_body_clearance_penalty",
    "v134_posture_metric",
    "v134_critic_state", "v134_metric", "v134_positive_ascent_chain_reward",
    "v134_native_chain_critic_state", "v134_ascent_style_anchor_mask",
]
