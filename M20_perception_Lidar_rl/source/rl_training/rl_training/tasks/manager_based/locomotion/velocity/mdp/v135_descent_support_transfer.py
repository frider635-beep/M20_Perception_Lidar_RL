"""V13.5-Final rear support transfer using V11's locked physical reference."""
from __future__ import annotations

import torch


def v135_rear_early_extension_cost(
    extension_ratio: torch.Tensor,
    horizontal_progress: torch.Tensor,
    *,
    free_ratio: float,
    ratio_scale: float,
) -> torch.Tensor:
    """Charge rear extension only before that rear wheel has crossed the edge."""
    if extension_ratio.shape != horizontal_progress.shape or extension_ratio.ndim != 2:
        raise ValueError("V13.5 rear extension/progress tensors must have matching [N, 2] shapes.")
    if free_ratio <= 0.0 or ratio_scale <= 0.0:
        raise ValueError("V13.5 rear extension bounds must be positive.")
    extension = ((extension_ratio - float(free_ratio)).clamp_min(0.0) / float(ratio_scale)).square()
    progress = horizontal_progress.clamp(0.0, 1.0)
    phase_gate = (1.0 - progress) * (progress < 0.3).float()
    return (extension * phase_gate).mean(dim=1).clamp_max(1.0)


def v135_rear_support_quality(
    stable_support: torch.Tensor,
    extension_ratio: torch.Tensor,
    *,
    extension_free_ratio: float,
    recovery_free_ratio: float,
) -> torch.Tensor:
    """Quality of lower support including recovery from temporary extension."""
    if stable_support.shape != extension_ratio.shape or stable_support.ndim != 2:
        raise ValueError("V13.5 rear support/extension tensors must have matching [N, 2] shapes.")
    span = float(extension_free_ratio) - float(recovery_free_ratio)
    if span <= 0.0:
        raise ValueError("V13.5 extension-free ratio must exceed the recovery ratio.")
    support_fraction = stable_support.float().mean(dim=1)
    # A folded rear leg is as poor a final stance as an overextended one.
    # Use the same neutral-length tolerance in both directions, and take the
    # worse of the two rear legs before restoring full speed value.
    worst_deviation = (extension_ratio - 1.0).abs().amax(dim=1)
    recovery_quality = (
        (float(extension_free_ratio) - 1.0 - worst_deviation) / span
    ).clamp(0.0, 1.0)
    return support_fraction * recovery_quality


def v135_full_body_completion_step(
    steps: torch.Tensor,
    completed: torch.Tensor,
    front_phase_complete: torch.Tensor,
    target_active: torch.Tensor,
    rear_support: torch.Tensor,
    front_contact: torch.Tensor,
    pitch_abs: torch.Tensor,
    *,
    confirmation_frames: int,
    pitch_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Settle once after sustained rear support, current front load, and safe pitch."""
    if rear_support.shape != front_contact.shape or rear_support.ndim != 2 or rear_support.shape[1] != 2:
        raise ValueError("V13.5 full-body support expects [N, 2] front/rear masks.")
    if any(x.shape != steps.shape for x in (completed, front_phase_complete, target_active, pitch_abs)):
        raise ValueError("V13.5 full-body completion masks must match environment count.")
    if confirmation_frames < 1:
        raise ValueError("V13.5 full-body confirmation_frames must be positive.")
    stable = (
        front_phase_complete & target_active & rear_support.all(dim=1)
        & front_contact.all(dim=1) & (pitch_abs <= float(pitch_tolerance))
    )
    new_steps = torch.where(stable, steps + 1, torch.zeros_like(steps))
    event = ~completed & (new_steps >= int(confirmation_frames))
    return new_steps, completed | event, event


def v135_descent_velocity_factor(
    active: torch.Tensor,
    rear_progress: torch.Tensor,
    recovery_armed: torch.Tensor,
    support_quality: torch.Tensor,
    *,
    baseline: float,
) -> torch.Tensor:
    """Withhold full speed credit whenever post-transfer support degrades."""
    if not (active.shape == rear_progress.shape == recovery_armed.shape == support_quality.shape):
        raise ValueError("V13.5 descent velocity inputs must have matching shapes.")
    transfer_gate = float(baseline) + (1.0 - float(baseline)) * rear_progress.clamp(0, 1)
    recovery_gate = 0.70 + 0.30 * support_quality.clamp(0, 1)
    gate = torch.where(active, transfer_gate, torch.ones_like(transfer_gate))
    # recovery_done is only an event latch. A brief normal posture must not
    # restore full velocity credit if the rear legs extend again afterward.
    return torch.where(recovery_armed, recovery_gate, gate)


def _initialize_rear_transfer(env):
    if hasattr(env, "_v135_rear_transfer"):
        return
    n, d = env.num_envs, env.device
    env._v135_rear_transfer = {
        "last_episode_length": env.episode_length_buf.clone(),
        "reference_seen": torch.zeros(n, device=d, dtype=torch.bool),
        "last_reference_world_xy": torch.zeros((n, 2), device=d),
        "front_complete": torch.zeros(n, device=d, dtype=torch.bool),
        "target_active_now": torch.zeros(n, device=d, dtype=torch.bool),
        "edge_world_xy": torch.zeros((n, 2), device=d),
        "forward_world_xy": torch.zeros((n, 2), device=d),
        "z_lower": torch.zeros(n, device=d),
        "rear_init_world_xy": torch.zeros((n, 2, 2), device=d),
        "rear_init_z": torch.zeros((n, 2), device=d),
        "body_init_forward": torch.zeros(n, device=d),
        "rear_init_forward": torch.zeros((n, 2), device=d),
        "p_x": torch.zeros((n, 2), device=d),
        "p_z": torch.zeros((n, 2), device=d),
        "p_xz": torch.zeros((n, 2), device=d),
        "stable_support": torch.zeros((n, 2), device=d, dtype=torch.bool),
        "stable_frames": torch.zeros((n, 2), device=d, dtype=torch.long),
        "front_support_now": torch.zeros((n, 2), device=d, dtype=torch.bool),
        "whole_body_support_steps": torch.zeros(n, device=d, dtype=torch.long),
        "phi": torch.zeros((n, 2), device=d),
        "phi_best": torch.zeros((n, 2), device=d),
        "P_rear": torch.zeros(n, device=d),
        "P_rear_best": torch.zeros(n, device=d),
        "P_rear_best_prev": torch.zeros(n, device=d),
        "active": torch.zeros(n, device=d, dtype=torch.bool),
        "completion": torch.zeros(n, device=d, dtype=torch.bool),
        "completion_event": torch.zeros(n, device=d, dtype=torch.bool),
        "body_rear_lag": torch.zeros(n, device=d),
        "rear_extension_ratio": torch.ones((n, 2), device=d),
        "neutral_rear_leg_length": torch.ones((n, 2), device=d),
        "recovery_armed": torch.zeros(n, device=d, dtype=torch.bool),
        "recovery_steps": torch.zeros(n, device=d, dtype=torch.long),
        "recovery_done": torch.zeros(n, device=d, dtype=torch.bool),
        "recovery_event": torch.zeros(n, device=d, dtype=torch.bool),
        "support_quality": torch.zeros(n, device=d),
        "post_support_extension_steps": torch.zeros(n, device=d, dtype=torch.long),
        "persistent_extension_event": torch.zeros(n, device=d, dtype=torch.bool),
        "persistent_extension_latched": torch.zeros(n, device=d, dtype=torch.bool),
        "body_pitch_abs": torch.zeros(n, device=d),
        "rear_load_share": torch.zeros(n, device=d),
        "four_wheel_support": torch.zeros(n, device=d, dtype=torch.bool),
        "body_path_speed": torch.zeros(n, device=d),
        "body_pitch_rate_abs": torch.zeros(n, device=d),
    }
    env._v135_rear_transfer_last_step = -1
    asset, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
    body_ids, contact_ids = [], []
    for name in ("hl_wheel", "hr_wheel"):
        body, _ = asset.find_bodies(name)
        contact, _ = sensor.find_bodies(name)
        if len(body) != 1 or len(contact) != 1:
            raise RuntimeError(f"V135 rear transfer expected one body/contact for {name}")
        body_ids.append(body[0]); contact_ids.append(contact[0])
    env._v135_rear_body_ids = torch.tensor(body_ids, device=d, dtype=torch.long)
    env._v135_rear_contact_ids = torch.tensor(contact_ids, device=d, dtype=torch.long)
    hip_ids, hip_names = asset.find_bodies(["hl_hipy", "hr_hipy"], preserve_order=True)
    if hip_names != ["hl_hipy", "hr_hipy"]:
        raise RuntimeError(f"V135 rear transfer could not resolve rear hip bodies: {hip_names}")
    env._v135_rear_hip_body_ids = torch.tensor(hip_ids, device=d, dtype=torch.long)
    rear_pos = asset.data.body_pos_w[:, env._v135_rear_body_ids]
    hip_pos = asset.data.body_pos_w[:, env._v135_rear_hip_body_ids]
    env._v135_rear_transfer["neutral_rear_leg_length"] = torch.linalg.norm(
        rear_pos - hip_pos, dim=2
    ).detach().clamp_min(1.0e-3)


def _clear_transfer(s, mask):
    one_dim = ("reference_seen", "front_complete", "target_active_now", "z_lower", "P_rear", "P_rear_best",
               "P_rear_best_prev", "active", "completion", "whole_body_support_steps",
               "completion_event", "body_init_forward", "body_rear_lag", "support_quality",
               "recovery_armed", "recovery_steps", "recovery_done", "recovery_event",
               "support_quality", "post_support_extension_steps",
               "persistent_extension_event", "persistent_extension_latched", "body_pitch_abs")
    one_dim += ("rear_load_share", "four_wheel_support", "body_path_speed", "body_pitch_rate_abs")
    two_dim = ("last_reference_world_xy", "edge_world_xy", "forward_world_xy", "rear_init_z",
               "rear_init_forward", "p_x", "p_z", "p_xz", "stable_support",
               "stable_frames", "front_support_now", "phi", "phi_best", "rear_extension_ratio")
    for key in one_dim:
        s[key][mask] = 0
    for key in two_dim:
        s[key][mask] = 0
    s["rear_init_world_xy"][mask] = 0


def v135_descent_rear_support_transfer_state(env, sensor_cfg=None):
    from .v11_continuous import v11_state

    _initialize_rear_transfer(env)
    step = int(getattr(env, "common_step_counter", 0))
    if env._v135_rear_transfer_last_step == step:
        return env._v135_rear_transfer
    s = env._v135_rear_transfer
    reset = env.episode_length_buf < s["last_episode_length"]
    _clear_transfer(s, reset)

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    descent, asset = base["descent"], env.scene["robot"]
    reference = descent["reference_world_xy"]
    target_active = descent["target_active"]
    reference_changed = target_active & s["reference_seen"] & (
        torch.linalg.norm(reference - s["last_reference_world_xy"], dim=1) > 0.10
    )
    _clear_transfer(s, reference_changed)
    s["target_active_now"] = target_active
    new_reference = target_active & ~s["reference_seen"]
    remember_reference = new_reference | reference_changed
    s["last_reference_world_xy"] = torch.where(
        remember_reference[:, None], reference, s["last_reference_world_xy"]
    )
    s["reference_seen"] |= target_active

    front_event = target_active & descent["front_complete"] & ~s["front_complete"]
    rear_pos = asset.data.body_pos_w[:, env._v135_rear_body_ids, :]
    rear_hip_pos = asset.data.body_pos_w[:, env._v135_rear_hip_body_ids, :]
    rear_xy, rear_z = rear_pos[..., :2], rear_pos[..., 2]
    body_xy = asset.data.root_pos_w[:, :2]
    rear_leg_length = torch.linalg.norm(rear_pos - rear_hip_pos, dim=2)
    s["neutral_rear_leg_length"] = torch.where(
        reset[:, None],
        rear_leg_length.detach().clamp_min(1.0e-3),
        s["neutral_rear_leg_length"],
    )
    s["rear_extension_ratio"] = rear_leg_length / s["neutral_rear_leg_length"].clamp_min(1.0e-3)
    if front_event.any():
        command_term = env.command_manager.get_term("base_velocity")
        if not hasattr(command_term, "v11_path_heading_w"):
            raise RuntimeError("V135 rear transfer requires v11_path_heading_w")
        heading = command_term.v11_path_heading_w
        forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
        s["edge_world_xy"] = torch.where(front_event[:, None], reference, s["edge_world_xy"])
        s["forward_world_xy"] = torch.where(front_event[:, None], forward, s["forward_world_xy"])
        s["z_lower"] = torch.where(front_event, descent["target_lower_z"], s["z_lower"])
        s["rear_init_world_xy"] = torch.where(front_event[:, None, None], rear_xy, s["rear_init_world_xy"])
        s["rear_init_z"] = torch.where(front_event[:, None], rear_z, s["rear_init_z"])
        body_init = ((body_xy - reference) * forward).sum(dim=1)
        rear_init = ((rear_xy - reference[:, None, :]) * forward[:, None, :]).sum(dim=2)
        s["body_init_forward"] = torch.where(front_event, body_init, s["body_init_forward"])
        s["rear_init_forward"] = torch.where(front_event[:, None], rear_init, s["rear_init_forward"])
    s["front_complete"] |= front_event

    cfg = env.cfg.v135_descent_support_transfer
    wheel_radius = float(cfg.wheel_radius)
    front_margin = float(cfg.front_margin)
    rear_forward = ((rear_xy - s["edge_world_xy"][:, None, :]) * s["forward_world_xy"][:, None, :]).sum(dim=2)
    target_forward = wheel_radius + front_margin
    span = (target_forward - s["rear_init_forward"]).clamp_min(0.03)
    # A PhysX body/sensor sample can be non-finite while root_state_w remains
    # finite. Missing rear-wheel evidence must earn zero progress, never poison
    # the historical best (which otherwise persists until episode reset).
    s["p_x"] = torch.nan_to_num(
        ((rear_forward - s["rear_init_forward"]) / span).clamp(0, 1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    expected_drop = (s["rear_init_z"] - wheel_radius - s["z_lower"][:, None]).clamp(0.04, 0.30)
    s["p_z"] = torch.nan_to_num(
        ((s["rear_init_z"] - rear_z) / expected_drop).clamp(0, 1),
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    s["p_xz"] = s["p_x"] * s["p_z"]

    rear_bottom_z = rear_z - wheel_radius
    crossed = rear_forward > target_forward
    on_lower = (rear_bottom_z - s["z_lower"][:, None]).abs() < 0.05
    forces_z = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v135_rear_contact_ids, 2]
    front_forces_z = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids, 2]
    front_load = front_forces_z.clamp_min(0).sum(dim=1)
    rear_load = forces_z.clamp_min(0).sum(dim=1)
    s["rear_load_share"] = rear_load / (front_load + rear_load).clamp_min(1.0e-6)
    contact_threshold = float(cfg.contact_force_threshold)
    s["front_support_now"] = front_forces_z > contact_threshold
    s["four_wheel_support"] = (
        s["front_support_now"].all(dim=1)
        & (forces_z > contact_threshold).all(dim=1)
    )
    s["body_path_speed"] = (
        asset.data.root_lin_vel_w[:, :2] * s["forward_world_xy"]
    ).sum(dim=1)
    lateral_axis = torch.stack((-s["forward_world_xy"][:, 1], s["forward_world_xy"][:, 0]), dim=1)
    s["body_pitch_rate_abs"] = (
        asset.data.root_ang_vel_w[:, :2] * lateral_axis
    ).sum(dim=1).abs()
    contact = forces_z > float(cfg.contact_force_threshold)
    stable = crossed & on_lower & contact & s["front_complete"][:, None] & target_active[:, None]
    s["stable_frames"] = torch.where(stable, s["stable_frames"] + 1, torch.zeros_like(s["stable_frames"]))
    s["stable_support"] = s["stable_frames"] >= int(cfg.stable_frames)
    s["phi"] = 0.45 * s["p_x"] + 0.35 * s["p_xz"] + 0.20 * s["stable_support"].float()
    s["phi_best"] = torch.maximum(s["phi_best"], s["phi"])
    left, right = s["phi"][:, 0], s["phi"][:, 1]
    mean_phi = 0.5 * (left + right)
    s["P_rear"] = (0.5 * mean_phi + 0.5 * torch.minimum(left, right)).clamp(0, 1)
    s["P_rear_best"] = torch.maximum(s["P_rear_best"], s["P_rear"])

    from .v135_whole_body_descent import v135_whole_body_descent_state
    whole = v135_whole_body_descent_state(env, sensor_cfg)
    w, x, y, z = asset.data.root_quat_w.unbind(-1)
    sin_pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    s["body_pitch_abs"] = torch.asin(sin_pitch).abs()
    s["whole_body_support_steps"], s["completion"], s["completion_event"] = v135_full_body_completion_step(
        s["whole_body_support_steps"], s["completion"], s["front_complete"], target_active,
        s["stable_support"] & (whole["workspace"][:, 2:] >= 0.5) & whole["ground_valid"][:, None],
        s["front_support_now"].any(1)[:, None].expand(-1, 2), whole["pitch_error"],
        confirmation_frames=int(cfg.stable_frames), pitch_tolerance=float(cfg.recovery_pitch_tolerance),
    )
    s["active"] = target_active & s["front_complete"] & ~s["completion"]
    s["recovery_armed"] |= s["completion_event"]
    recovered = (
        s["recovery_armed"]
        & ~s["recovery_done"]
        & s["stable_support"].all(dim=1)
        & s["front_support_now"].all(dim=1)
        & ((s["rear_extension_ratio"] - 1.0).abs() <=
           float(cfg.recovery_free_ratio) - 1.0).all(dim=1)
        & (whole["pitch_error"] <= float(cfg.recovery_pitch_tolerance))
        & (whole["workspace"].amin(1) >= 0.5)
        & whole["ground_valid"]
    )
    s["recovery_steps"] = torch.where(
        recovered, s["recovery_steps"] + 1, torch.zeros_like(s["recovery_steps"])
    )
    s["recovery_event"] = (
        ~s["recovery_done"]
        & (s["recovery_steps"] >= int(cfg.recovery_confirmation_frames))
    )
    s["recovery_done"] |= s["recovery_event"]
    persistent_extension = (
        s["recovery_armed"]
        & ~s["recovery_done"]
        & s["stable_support"].all(dim=1)
        & (s["rear_extension_ratio"].amax(dim=1) > float(cfg.persistent_extension_ratio))
    )
    s["post_support_extension_steps"] = torch.where(
        persistent_extension,
        s["post_support_extension_steps"] + 1,
        torch.zeros_like(s["post_support_extension_steps"]),
    )
    extension_now = s["post_support_extension_steps"] >= int(cfg.persistent_extension_frames)
    s["persistent_extension_event"] = extension_now & ~s["persistent_extension_latched"]
    s["persistent_extension_latched"] |= extension_now
    # Support quality is deliberately separate from transfer progress. Stable
    # lower support alone is insufficient: an extremely extended rear leg must
    # recover toward its neutral length before full velocity value returns.
    from .v135_whole_body_descent import v135_whole_body_descent_state
    whole = v135_whole_body_descent_state(env, sensor_cfg)
    s["support_quality"] = torch.nan_to_num(
        whole["quality"].clamp(0, 1), nan=0.0, posinf=0.0, neginf=0.0,
    )
    s["monitor_active"] = whole["active"]

    body_forward = ((body_xy - s["edge_world_xy"]) * s["forward_world_xy"]).sum(dim=1)
    body_delta = body_forward - s["body_init_forward"]
    rear_delta = rear_forward - s["rear_init_forward"]
    s["body_rear_lag"] = body_delta - rear_delta.min(dim=1).values
    s["last_episode_length"] = env.episode_length_buf.clone()
    env._v135_rear_transfer_last_step = step
    return s


def v135_descent_rear_transfer_progress_reward(env, sensor_cfg=None):
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    delta = (s["P_rear_best"] - s["P_rear_best_prev"]).clamp_min(0)
    s["P_rear_best_prev"] = s["P_rear_best"].clone()
    # Keep the transition that establishes the second stable rear support.
    # `active` becomes false on that exact frame, but the final support credit
    # is part of the transfer and must not be discarded.
    eligible = s["front_complete"] & (s["active"] | s["completion_event"])
    from .v135_descent_stance import v135_post_stair_flat_state
    blend = v135_post_stair_flat_state(env, sensor_cfg)["flat_blend"]
    return delta / float(env.step_dt) * eligible.float() * s["support_quality"] * (1.0 - blend)


def v135_descent_rear_transfer_completion_reward(env, sensor_cfg=None):
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    return s["completion_event"].float() / float(env.step_dt)


def v135_descent_body_rear_lag_penalty(env, sensor_cfg=None):
    from .v135_longtrain_curriculum import v135_descent_error_scale
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    cfg = env.cfg.v135_descent_support_transfer
    cost = (
        (s["body_rear_lag"] - float(cfg.free_lag)).clamp_min(0) / float(cfg.lag_scale)
    ).square().clamp_max(1)
    return -float(v135_descent_error_scale(env)) * cost * s["active"].float()


def v135_descent_premature_collapse_penalty(env, sensor_cfg=None):
    from .v135_longtrain_curriculum import v135_descent_error_scale
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    cfg = env.cfg.v135_descent_support_transfer
    rear_z = env.scene["robot"].data.body_pos_w[:, env._v135_rear_body_ids, 2]
    drop = (s["rear_init_z"] - rear_z).clamp_min(0)
    cost = (
        (1 - s["p_x"])
        * ((drop - float(cfg.collapse_threshold)) / float(cfg.collapse_scale)).clamp(0, 1).square()
    ).mean(dim=1).clamp_max(1)
    return -float(v135_descent_error_scale(env)) * cost * s["active"].float()


def v135_descent_rear_early_extension_penalty(env, sensor_cfg=None):
    from .v135_longtrain_curriculum import v135_descent_error_scale
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    cfg = env.cfg.v135_descent_support_transfer
    cost = v135_rear_early_extension_cost(
        s["rear_extension_ratio"], s["p_x"],
        free_ratio=float(cfg.rear_extension_free_ratio),
        ratio_scale=float(cfg.rear_extension_ratio_scale),
    )
    return -float(v135_descent_error_scale(env)) * cost * s["active"].float()


def v135_descent_rear_recovery_reward(env, sensor_cfg=None):
    """One-shot credit after lower support and rear-leg length recovery."""
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    return s["recovery_event"].float() / float(env.step_dt)


def v135_descent_persistent_extension_penalty(env, sensor_cfg=None):
    """One-shot cost when excessive extension persists after stable support."""
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    return -s["persistent_extension_event"].float() / float(env.step_dt)


def v135_descent_support_transfer_metric(env, env_ids, key: str, sensor_cfg=None):
    del env_ids
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    active = s["active"].float()
    count = active.sum().clamp_min(1.0)
    if key in ("whole_body_slip", "whole_body_workspace", "terrain_pitch_error", "ground_evidence_rate"):
        from .v135_whole_body_descent import v135_whole_body_descent_state
        whole = v135_whole_body_descent_state(env, sensor_cfg)
        values = {"whole_body_slip": whole["slip"], "whole_body_workspace": whole["workspace"].amin(1),
                  "terrain_pitch_error": whole["pitch_error"], "ground_evidence_rate": whole["ground_valid"].float()}
        mask = whole["active"].float()
        return (values[key] * mask).sum() / mask.sum().clamp_min(1)
    if key == "active_rate":
        return active.mean()
    if key == "rear_extension_ratio":
        return (s[key].amax(dim=1) * active).sum() / count
    if key == "rear_length_min_ratio":
        return (s["rear_extension_ratio"].amin(dim=1) * active).sum() / count
    if key == "early_extension_rate":
        cfg = env.cfg.v135_descent_support_transfer
        early = (
            (s["p_x"] < 0.3)
            & (s["rear_extension_ratio"] > float(cfg.rear_extension_free_ratio))
        ).any(dim=1)
        return (early.float() * active).sum() / count
    if key == "recovery_event_rate":
        return s["recovery_event"].float().mean()
    if key == "support_quality":
        envelope = s["monitor_active"].float()
        return (s["support_quality"] * envelope).sum() / envelope.sum().clamp_min(1.0)
    if key == "persistent_extension_event_rate":
        return s["persistent_extension_event"].float().mean()
    if key == "recovery_pitch_abs":
        envelope = (s["recovery_armed"] & ~s["recovery_done"]).float()
        return (s["body_pitch_abs"] * envelope).sum() / envelope.sum().clamp_min(1.0)
    if key in ("rear_load_share", "four_wheel_support", "body_path_speed", "body_pitch_rate_abs", "body_rear_lag"):
        envelope = (s["active"] | s["recovery_armed"]).float()
        return (s[key].float() * envelope).sum() / envelope.sum().clamp_min(1.0)
    raise ValueError(f"Unknown V13.5 rear-transfer metric: {key!r}")


def v135_descent_velocity_modulation(env, sensor_cfg=None):
    from .v135_longtrain_curriculum import v135_descent_velocity_baseline
    s = v135_descent_rear_support_transfer_state(env, sensor_cfg)
    baseline = float(v135_descent_velocity_baseline(env))
    from .v135_descent_stance import v135_post_stair_flat_state
    flat = v135_post_stair_flat_state(env, sensor_cfg)["flat_blend"]
    gate = baseline + (1.0 - baseline) * s["support_quality"]
    return torch.where(s["monitor_active"], 1.0 - (1.0 - gate) * (1.0 - flat), torch.ones_like(gate))



__all__ = [name for name in globals() if name.startswith("v135_descent_")]
