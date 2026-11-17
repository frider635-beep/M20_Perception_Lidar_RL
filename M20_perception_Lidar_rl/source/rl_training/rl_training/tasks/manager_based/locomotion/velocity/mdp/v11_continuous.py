"""V11 deployable continuous two-riser observation state."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .observations import perceptive_elevation_clean
from .perceptive_spatial import (
    _elevation_points_yaw,
    classify_vertical_riser_contact,
    extract_two_risers_points, extract_three_risers_points,
    wheel_positions_yaw,
)
from .v11_landing import (
    v11_bounded_knee_excess,
    v11_bounded_relative_hipy_participation_scores,
    v11_alternating_reward_event,
    v11_anchor_support_valid,
    v11_clearance_surplus,
    v11_crossing_clearance_gate,
    v11_classify_soft_alternation,
    v11_dt_normalized_event,
    v11_confirm_support,
    v11_confirm_vertical_contact,
    v11_expected_front_mask,
    v11_front_step_mobility_gate,
    v11_front_reward_event,
    v11_latch_alternating_front_completion,
    v11_landing_scores,
    v11_latch_responsible_front_wheel,
    v11_safe_landing_region,
    v11_safe_support_clearance_progress,
    v11_softalt_candidate_ready,
    v11_softalt_termination,
    v11_select_next_uncredited_riser,
    v11_settle_clearance_escrow,
    v11_settle_bilateral_escrow,
    v11_update_bilateral_support_window,
    v11_update_clearance_escrow,
    v11_update_latest_participation_escrow,
)
from .rewards import get_gait_level_tensor, joint_mirror


def _v11_initialize_landing(env, asset: Articulation) -> None:
    """Create simulator-only reward state for the symmetric FL/FR encounter.

    The actor never observes this state.  In particular, responsibility is an
    outcome of the two symmetric candidates, not a leg-selection command.
    """
    if hasattr(env, "_v11_landing"):
        return
    body_ids, contact_ids = [], []
    sensor = env.scene.sensors["contact_forces"]
    for name in ("fl_wheel", "fr_wheel"):
        found, names = asset.find_bodies(name)
        if len(found) != 1 or names != [name]:
            raise RuntimeError(f"V11 expected one {name!r} body, got {names}")
        found_contact, contact_names = sensor.find_bodies(name)
        if len(found_contact) != 1 or contact_names != [name]:
            raise RuntimeError(f"V11 expected one {name!r} contact body, got {contact_names}")
        body_ids.append(found[0])
        contact_ids.append(found_contact[0])
    env._v11_front_body_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)
    env._v11_front_contact_ids = torch.tensor(contact_ids, device=env.device, dtype=torch.long)
    front_hipy_body_ids, front_knee_body_ids = [], []
    for hipy_name, knee_name in (("fl_hipy", "fl_knee"), ("fr_hipy", "fr_knee")):
        hipy_found, hipy_names = asset.find_bodies(hipy_name)
        knee_found, knee_names = asset.find_bodies(knee_name)
        if len(hipy_found) != 1 or hipy_names != [hipy_name]:
            raise RuntimeError(f"V11 expected one {hipy_name!r} body, got {hipy_names}")
        if len(knee_found) != 1 or knee_names != [knee_name]:
            raise RuntimeError(f"V11 expected one {knee_name!r} body, got {knee_names}")
        front_hipy_body_ids.append(hipy_found[0])
        front_knee_body_ids.append(knee_found[0])
    env._v11_front_hipy_body_ids = torch.tensor(front_hipy_body_ids, device=env.device, dtype=torch.long)
    env._v11_front_knee_body_ids = torch.tensor(front_knee_body_ids, device=env.device, dtype=torch.long)
    rear_body_ids, rear_contact_ids = [], []
    for name in ("hl_wheel", "hr_wheel"):
        found, names = asset.find_bodies(name)
        if len(found) != 1 or names != [name]:
            raise RuntimeError(f"V11 expected one {name!r} body, got {names}")
        found_contact, contact_names = sensor.find_bodies(name)
        if len(found_contact) != 1 or contact_names != [name]:
            raise RuntimeError(f"V11 expected one {name!r} contact body, got {contact_names}")
        rear_body_ids.append(found[0])
        rear_contact_ids.append(found_contact[0])
    env._v11_rear_body_ids = torch.tensor(rear_body_ids, device=env.device, dtype=torch.long)
    env._v11_rear_contact_ids = torch.tensor(rear_contact_ids, device=env.device, dtype=torch.long)
    n = env.num_envs
    env._v11_landing = {
        "last_episode_length": env.episode_length_buf.clone(),
        # -1: no wheel has achieved valid first-tread support; 0/1: FL/FR.
        "responsible": torch.full((n,), -1, device=env.device, dtype=torch.long),
        # Alternating V11: -1 lets either side own the first riser; every
        # completion toggles this to the opposite side for the next riser.
        "expected_side": torch.full((n,), -1, device=env.device, dtype=torch.long),
        # SoftAlt evaluates these only after a side-neutral physical completion.
        "previous_completed_side": torch.full((n,), -1, device=env.device, dtype=torch.long),
        "current_completed_side": torch.full((n,), -1, device=env.device, dtype=torch.long),
        "anchor_side": torch.full((n,), -1, device=env.device, dtype=torch.long),
        "anchor_support_valid": torch.zeros(n, device=env.device, dtype=torch.bool),
        "anchor_support_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "completed_side": torch.full((n,), -1, device=env.device, dtype=torch.long),
        "sequence_started": torch.zeros(n, device=env.device, dtype=torch.bool),
        "completed_riser_count": torch.zeros(n, device=env.device, dtype=torch.long),
        "last_completed_riser_world_xy": torch.zeros((n, 2), device=env.device),
        "last_completed_upper_z": torch.zeros(n, device=env.device),
        "target_slot": torch.full((n,), -1, device=env.device, dtype=torch.long),
        "target_match_valid": torch.zeros(n, device=env.device, dtype=torch.bool),
        "clearance": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "crossing": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "front_crossing_event": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "crossing_clearance_ok": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "low_clearance_crossing_event": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "crossing_wheel_center_above_tread": torch.zeros((n, 2), device=env.device),
        "crossing_sample_valid": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "support": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "front_support_event": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "front_support_clearance_settlement": torch.zeros((n, 2), device=env.device),
        "front_support_credited_valid": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "front_support_credited_world_xy": torch.zeros((n, 2, 2), device=env.device),
        "support_steps": torch.zeros((n, 2), device=env.device, dtype=torch.long),
        "candidate_ready": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "front_candidate_event": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "crossing_gap_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "support_gap_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "vertical_contact": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "vertical_contact_now": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "vertical_contact_steps": torch.zeros((n, 2), device=env.device, dtype=torch.long),
        "reference_valid": torch.zeros(n, device=env.device, dtype=torch.bool),
        "reference_riser_world_xy": torch.zeros((n, 2), device=env.device),
        "target_lost_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "max_clearance_progress": torch.zeros((n, 2), device=env.device),
        "clearance_escrow": torch.zeros((n, 2), device=env.device),
        "previous_wheel_x": torch.zeros((n, 2), device=env.device),
        "previous_wheel_x_valid": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "clearance_settlement": torch.zeros(n, device=env.device),
        "clearance_surplus_escrow": torch.zeros((n, 2), device=env.device),
        "clearance_surplus_settlement": torch.zeros(n, device=env.device),
        "pending_first_clearance": torch.zeros(n, device=env.device),
        "pending_first_hipy": torch.zeros(n, device=env.device),
        "second_clearance_settlement": torch.zeros(n, device=env.device),
        "hipy_participation_escrow": torch.zeros((n, 2), device=env.device),
        "hipy_participation_settlement": torch.zeros(n, device=env.device),
        "landing_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "first_completion_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "alternating_completion_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "same_side_repeat_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "simultaneous_completion_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "alternating_reward_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "bilateral_landing_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "front_collision_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "alternation_violation_now": torch.zeros(n, device=env.device, dtype=torch.bool),
        "alternation_violation_latched": torch.zeros(n, device=env.device, dtype=torch.bool),
        "alternation_violation_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "alternation_violation_confirmed": torch.zeros(n, device=env.device, dtype=torch.bool),
        "crossing_clearance_margin": torch.zeros(n, device=env.device),
        "crossing_clearance_margin_per_side": torch.zeros((n, 2), device=env.device),
        "platform_no_riser_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "platform_support_steps": torch.zeros(n, device=env.device, dtype=torch.long),
        "platform_confirmed": torch.zeros(n, device=env.device, dtype=torch.bool),
        "platform_exit_event": torch.zeros(n, device=env.device, dtype=torch.bool),
        "rear_support_latched": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
        "rear_support_window_age": torch.zeros(n, device=env.device, dtype=torch.long),
        # Persist across a reference release within an episode.  The same
        # physical riser must not become a second clearance-reward encounter
        # merely because it disappears from the scan and then reappears.
        "credited_reference_valid": torch.zeros(n, device=env.device, dtype=torch.bool),
        "credited_reference_world_xy": torch.zeros((n, 2), device=env.device),
        # Global training diagnostics; 0-D tensors intentionally survive
        # per-environment episode resets.
        "responsibility_fl_count": torch.zeros((), device=env.device, dtype=torch.long),
        "responsibility_fr_count": torch.zeros((), device=env.device, dtype=torch.long),
        "responsibility_tie_count": torch.zeros((), device=env.device, dtype=torch.long),
        "alternating_completion_count": torch.zeros((), device=env.device, dtype=torch.long),
        "alternation_violation_count": torch.zeros((), device=env.device, dtype=torch.long),
        "first_completion_count": torch.zeros((), device=env.device, dtype=torch.long),
        "fl_completion_count": torch.zeros((), device=env.device, dtype=torch.long),
        "fr_completion_count": torch.zeros((), device=env.device, dtype=torch.long),
        "same_side_repeat_count": torch.zeros((), device=env.device, dtype=torch.long),
        "simultaneous_completion_count": torch.zeros((), device=env.device, dtype=torch.long),
        "simultaneous_transition_count": torch.zeros((), device=env.device, dtype=torch.long),
        "fl_to_fr_count": torch.zeros((), device=env.device, dtype=torch.long),
        "fr_to_fl_count": torch.zeros((), device=env.device, dtype=torch.long),
        "fl_to_fl_repeat_count": torch.zeros((), device=env.device, dtype=torch.long),
        "fr_to_fr_repeat_count": torch.zeros((), device=env.device, dtype=torch.long),
        "platform_success_count": torch.zeros((), device=env.device, dtype=torch.long),
        "stair_episode_count": torch.zeros((), device=env.device, dtype=torch.long),
        "ascending_episode_count": torch.zeros((), device=env.device, dtype=torch.long),
        "front_collision_episode_count": torch.zeros((), device=env.device, dtype=torch.long),
        "stair_forward_velocity_sum": torch.zeros((), device=env.device),
        "stair_forward_velocity_samples": torch.zeros((), device=env.device, dtype=torch.long),
        "episode_alternating_count": torch.zeros(n, device=env.device, dtype=torch.long),
        "episode_repeat_count": torch.zeros(n, device=env.device, dtype=torch.long),
        "episode_simultaneous_count": torch.zeros(n, device=env.device, dtype=torch.long),
        "episode_front_collision": torch.zeros(n, device=env.device, dtype=torch.bool),
        "post_first_collision_count": torch.zeros((), device=env.device, dtype=torch.long),
        "crossing_clearance_margin_sum": torch.zeros((), device=env.device),
        "crossing_clearance_margin_count": torch.zeros((), device=env.device, dtype=torch.long),
        "front_crossing_count": torch.zeros((), device=env.device, dtype=torch.long),
        "low_clearance_crossing_count": torch.zeros((), device=env.device, dtype=torch.long),
        "front_crossing_center_sum": torch.zeros((), device=env.device),
        "front_crossing_margin_sum": torch.zeros((), device=env.device),
        "front_crossing_min_margin": torch.full((), float("inf"), device=env.device),
    }


def _v11_reset_landing(
    state: dict[str, torch.Tensor], reset: torch.Tensor, *, reset_credit_ledger: bool = True,
) -> None:
    for name, value in state.items():
        if name in (
            "last_episode_length", "credited_reference_valid", "credited_reference_world_xy",
            "cycle_paid_position", "cycle_paid_best",
            "front_support_credited_valid", "front_support_credited_world_xy",
        ):
            continue
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            value[reset] = 0
    if "cycle_leader" in state:
        state["cycle_leader"][reset] = -1
    if reset_credit_ledger and "cycle_paid_position" in state:
        state["cycle_paid_position"][reset] = -torch.inf
        state["cycle_paid_best"][reset] = 0
    if "handoff_leader" in state:
        state["handoff_leader"][reset] = -1
    state["responsible"][reset] = -1
    state["expected_side"][reset] = -1
    state["previous_completed_side"][reset] = -1
    state["current_completed_side"][reset] = -1
    state["anchor_side"][reset] = -1
    state["completed_side"][reset] = -1
    state["target_slot"][reset] = -1
    if reset_credit_ledger:
        state["credited_reference_valid"][reset] = False
        state["credited_reference_world_xy"][reset] = 0.0
        state["front_support_credited_valid"][reset] = False
        state["front_support_credited_world_xy"][reset] = 0.0


def _v11_update_platform_exit_state(
    state: dict[str, torch.Tensor], cfg, raw_no_riser: torch.Tensor,
    front_complete: torch.Tensor, rear_crossed: torch.Tensor, rear_support: torch.Tensor,
) -> None:
    """Advance the fail-closed platform detector after a safe front-wheel landing."""
    platform_candidate = (
        state["reference_valid"] & front_complete & raw_no_riser & rear_crossed
    )
    state["platform_no_riser_steps"] = torch.where(
        platform_candidate,
        state["platform_no_riser_steps"] + 1,
        torch.zeros_like(state["platform_no_riser_steps"]),
    )
    stable_candidate = platform_candidate & rear_support
    state["platform_support_steps"] = torch.where(
        stable_candidate,
        state["platform_support_steps"] + 1,
        torch.zeros_like(state["platform_support_steps"]),
    )
    platform_now = (~state["platform_confirmed"]) & (
        state["platform_no_riser_steps"] >= int(cfg.platform_no_riser_confirmation_frames)
    ) & (
        state["platform_support_steps"] >= int(cfg.platform_rear_support_confirmation_frames)
    )
    state["platform_exit_event"] = platform_now
    state["platform_confirmed"] |= platform_now


def _v11_clear_alternating_target_phase(
    state: dict[str, torch.Tensor], clear: torch.Tensor,
) -> None:
    """Clear one riser phase without erasing the flight alternation ledger."""
    for name in (
        "clearance", "crossing", "front_crossing_event", "crossing_clearance_ok", "support", "candidate_ready",
        "front_support_event", "front_support_clearance_settlement", "front_candidate_event",
        "support_steps", "clearance_surplus_escrow",
        "vertical_contact", "vertical_contact_now", "vertical_contact_steps",
        "max_clearance_progress", "clearance_escrow", "previous_wheel_x",
        "previous_wheel_x_valid", "hipy_participation_escrow",
        "crossing_clearance_margin_per_side",
        "crossing_wheel_center_above_tread", "crossing_sample_valid", "low_clearance_crossing_event",
    ):
        state[name][clear] = 0
    for name in ("crossing_gap_steps", "support_gap_steps", "target_lost_steps"):
        state[name][clear] = 0
    state["reference_valid"][clear] = False
    state["reference_riser_world_xy"][clear] = 0.0
    state["target_slot"][clear] = -1
    state["target_match_valid"][clear] = False
    state["completed_side"][clear] = -1
    state["alternation_violation_latched"][clear] = False
    state["alternation_violation_steps"][clear] = 0
    state["alternation_violation_confirmed"][clear] = False


def _v11_softalt_sequence_diagnostic(env, reset: torch.Tensor, event: torch.Tensor,
                                     completed_side: torch.Tensor, repeat: torch.Tensor) -> None:
    """Emit a bounded, opt-in physical-completion trace for selected envs."""
    configured = tuple(getattr(env.cfg.v11_perception, "soft_alternation_debug_env_ids", ()))
    if not configured:
        return
    histories = getattr(env, "_v11_softalt_sequence_history", {})
    emitted = int(getattr(env, "_v11_softalt_sequence_emitted", 0))
    limit = int(getattr(env.cfg.v11_perception, "soft_alternation_debug_max_events", 12))
    for env_id in configured:
        if env_id < 0 or env_id >= env.num_envs:
            continue
        if bool(reset[env_id]):
            histories[env_id] = []
        if bool(event[env_id]) and emitted < limit:
            side = "FL" if int(completed_side[env_id]) == 0 else "FR"
            histories.setdefault(env_id, []).append(side)
            suffix = "  ^ repeat" if bool(repeat[env_id]) else ""
            print(f"V11_SOFTALT_SEQUENCE env {env_id}: {' -> '.join(histories[env_id])}{suffix}", flush=True)
            emitted += 1
    env._v11_softalt_sequence_history = histories
    env._v11_softalt_sequence_emitted = emitted


def _v11_alternating_landing_state(
    env, asset: Articulation, risers: dict[str, torch.Tensor], active: torch.Tensor,
):
    """Credit one alternating front wheel on each successive physical riser."""
    _v11_initialize_landing(env, asset)
    state, cfg = env._v11_landing, env.cfg.v11_perception
    reset = env.episode_length_buf < state["last_episode_length"]
    _v11_reset_landing(state, reset, reset_credit_ledger=True)
    state["last_episode_length"] = env.episode_length_buf.clone()
    for name in (
        "landing_event", "first_completion_event", "alternating_completion_event",
        "same_side_repeat_event", "simultaneous_completion_event", "alternating_reward_event",
        "bilateral_landing_event", "front_collision_event",
        "platform_exit_event", "alternation_violation_now", "front_support_event", "front_candidate_event",
    ):
        state[name].zero_()
    state["current_completed_side"].fill_(-1)
    for name in (
        "clearance_settlement", "second_clearance_settlement",
        "hipy_participation_settlement", "clearance_surplus_settlement", "crossing_clearance_margin",
    ):
        state[name].zero_()
    state["front_support_clearance_settlement"].zero_()

    root = asset.data.root_pos_w
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    riser_x, riser_y = risers["riser_x"], risers["riser_y"]
    detected_world_xy = torch.stack((
        root[:, None, 0] + cy[:, None] * riser_x - sy[:, None] * riser_y,
        root[:, None, 1] + sy[:, None] * riser_x + cy[:, None] * riser_y,
    ), dim=-1)

    # Select the closest visible face before the first completion.  Afterwards
    # only a face strictly ahead of the last credited world-space face can
    # become the next target, so detector hold/reappearance cannot pay twice.
    delta_from_last = detected_world_xy - state["last_completed_riser_world_xy"][:, None, :]
    forward_separation = (
        cy[:, None] * delta_from_last[..., 0] + sy[:, None] * delta_from_last[..., 1]
    )
    new_slot = v11_select_next_uncredited_riser(
        risers["valid"], forward_separation, state["sequence_started"],
        float(cfg.encounter_riser_switch_distance),
    )
    assign = active & ~state["reference_valid"] & (new_slot >= 0)
    if assign.any():
        _v11_clear_alternating_target_phase(state, assign)
        chosen = new_slot.clamp_min(0)
        chosen_world = detected_world_xy.gather(
            1, chosen[:, None, None].expand(-1, 1, 2),
        ).squeeze(1)
        state["reference_valid"][assign] = True
        state["reference_riser_world_xy"][assign] = chosen_world[assign]
        state["target_slot"][assign] = chosen[assign]

    distance_to_reference = torch.linalg.norm(
        detected_world_xy - state["reference_riser_world_xy"][:, None, :], dim=-1,
    )
    matching = risers["valid"] & (
        distance_to_reference < float(cfg.encounter_riser_switch_distance)
    )
    masked_distance = torch.where(
        matching, distance_to_reference, torch.full_like(distance_to_reference, torch.inf),
    )
    live_slot = masked_distance.argmin(dim=1)
    match_valid = state["reference_valid"] & matching.any(dim=1)
    state["target_match_valid"] = match_valid
    state["target_slot"] = torch.where(match_valid, live_slot, state["target_slot"])
    state["target_lost_steps"] = torch.where(
        state["reference_valid"] & ~match_valid,
        state["target_lost_steps"] + 1,
        torch.zeros_like(state["target_lost_steps"]),
    )
    release_lost = state["reference_valid"] & (
        state["target_lost_steps"] > int(cfg.target_loss_release_steps)
    )
    if release_lost.any():
        _v11_clear_alternating_target_phase(state, release_lost)

    chosen = state["target_slot"].clamp_min(0)
    target_riser_x = riser_x.gather(1, chosen[:, None]).squeeze(1)
    target_riser_y = riser_y.gather(1, chosen[:, None]).squeeze(1)
    target_height = risers["riser_height"].gather(1, chosen[:, None]).squeeze(1)
    target_lower_z = risers["riser_lower_world"].gather(1, chosen[:, None]).squeeze(1)
    target_upper_z = target_lower_z + target_height
    tread_y_min = target_riser_y - float(cfg.tread_corridor_half_width)
    tread_y_max = target_riser_y + float(cfg.tread_corridor_half_width)
    region = v11_safe_landing_region(
        target_riser_x, tread_y_min, tread_y_max,
        cfg.wheel_radius, cfg.front_margin, cfg.lateral_margin,
    )
    target_active = active & state["reference_valid"] & state["target_match_valid"] & region["valid"]
    soft_alternation = bool(getattr(cfg, "soft_alternation", False))
    eligible_sides = v11_expected_front_mask(state["expected_side"])
    # SoftAlt keeps expected_side as a diagnostic ledger only.  Both front
    # wheels see identical physical clearance/crossing/support gates.
    phase_active = target_active[:, None]
    if not soft_alternation:
        phase_active &= eligible_sides

    wheel_yaw = wheel_positions_yaw(asset, env._v11_front_body_ids)
    wheel_world = asset.data.body_pos_w[:, env._v11_front_body_ids]
    forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids]
    wheel_bottom = wheel_world[..., 2] - float(cfg.wheel_radius)
    before = wheel_yaw[..., 0] < target_riser_x[:, None]
    escrow = v11_update_clearance_escrow(
        state["max_clearance_progress"], state["clearance_escrow"],
        state["previous_wheel_x"], state["previous_wheel_x_valid"],
        wheel_yaw[..., 0], wheel_bottom, target_lower_z[:, None], target_upper_z[:, None],
        phase_active & before, float(cfg.clearance_margin),
    )
    state["max_clearance_progress"] = escrow["max_progress"]
    state["clearance_escrow"] = escrow["escrow"]
    state["previous_wheel_x"] = escrow["previous_x"]
    state["previous_wheel_x_valid"] = escrow["previous_x_valid"]
    state["clearance"] |= escrow["clearance"]

    crossing_now = phase_active & (wheel_yaw[..., 0] >= target_riser_x[:, None])
    new_crossing = crossing_now & ~state["crossing"]
    enforce_crossing_clearance = bool(cfg.require_clearance_at_crossing)
    if enforce_crossing_clearance:
        crossing_clearance = v11_crossing_clearance_gate(
            wheel_world[..., 2], target_upper_z[:, None], new_crossing,
            wheel_radius=cfg.wheel_radius, clearance_margin=cfg.clearance_margin,
        )
        state["front_crossing_event"] = new_crossing
        state["crossing_clearance_ok"] |= crossing_clearance["clear"]
        state["low_clearance_crossing_event"] = crossing_clearance["low"]
        state["crossing_wheel_center_above_tread"] = torch.where(
            new_crossing,
            crossing_clearance["center_above_tread"],
            state["crossing_wheel_center_above_tread"],
        )
        state["crossing_sample_valid"] |= new_crossing
        state["crossing_clearance_margin_per_side"] = torch.where(
            new_crossing, crossing_clearance["margin"], state["crossing_clearance_margin_per_side"],
        )
        state["front_crossing_count"] += new_crossing.sum()
        state["low_clearance_crossing_count"] += crossing_clearance["low"].sum()
        state["front_crossing_center_sum"] += (
            crossing_clearance["center_above_tread"] * new_crossing.float()
        ).sum()
        state["front_crossing_margin_sum"] += (
            crossing_clearance["margin"] * new_crossing.float()
        ).sum()
        event_margin = torch.where(
            new_crossing,
            crossing_clearance["margin"],
            torch.full_like(crossing_clearance["margin"], float("inf")),
        ).amin()
        state["front_crossing_min_margin"] = torch.minimum(
            state["front_crossing_min_margin"], event_margin,
        )
    else:
        crossing_margin = wheel_bottom - target_upper_z[:, None] - float(cfg.clearance_margin)
        state["crossing_clearance_margin_per_side"] = torch.where(
            new_crossing, crossing_margin, state["crossing_clearance_margin_per_side"],
        )
    surplus_now = v11_clearance_surplus(
        wheel_bottom, target_upper_z[:, None], cfg.clearance_margin,
        cfg.clearance_surplus_scale,
    )
    post_first_crossing = new_crossing & state["sequence_started"][:, None]
    if not soft_alternation:
        post_first_crossing &= eligible_sides
    state["clearance_surplus_escrow"] = torch.where(
        post_first_crossing, surplus_now, state["clearance_surplus_escrow"],
    )
    # A low actual face crossing invalidates any height accumulated earlier.
    if enforce_crossing_clearance:
        state["clearance_escrow"] = torch.where(
            crossing_clearance["low"], torch.zeros_like(state["clearance_escrow"]),
            state["clearance_escrow"],
        )
    state["crossing"] |= crossing_now
    force_norm = forces.norm(dim=-1)
    support_now = (
        phase_active
        & (wheel_yaw[..., 0] >= region["x_min"][:, None])
        & (wheel_yaw[..., 1] >= region["y_min"][:, None])
        & (wheel_yaw[..., 1] <= region["y_max"][:, None])
        & ((wheel_bottom - target_upper_z[:, None]).abs() <= float(cfg.upper_support_tolerance))
        & (force_norm >= float(cfg.contact_force_threshold))
        & (forces[..., 2] > float(cfg.contact_force_threshold))
    )
    support_confirmation = v11_confirm_support(
        state["support_steps"], support_now, int(cfg.front_support_confirmation_frames),
    )
    state["support_steps"] = support_confirmation["steps"]
    state["support"] = support_confirmation["confirmed"]

    # Safety observes both live faces, independent of which one is currently
    # credited.  A leg aiming for the second riser therefore cannot collide
    # with either the first or second face without the existing debounce gate.
    vertical_contact_now = torch.zeros_like(state["vertical_contact_now"])
    for slot in range(2):
        slot_lower = risers["riser_lower_world"][:, slot]
        slot_upper = slot_lower + risers["riser_height"][:, slot]
        slot_contact = classify_vertical_riser_contact(
            wheel_yaw[..., :2], wheel_world[..., 2], forces,
            riser_x[:, slot:slot + 1], slot_lower[:, None], slot_upper[:, None],
            wheel_radius=cfg.wheel_radius,
            force_threshold=cfg.contact_force_threshold,
            normal_ratio_threshold=cfg.vertical_force_ratio,
        )
        vertical_contact_now |= slot_contact & (active & risers["valid"][:, slot])[:, None]
    contact_confirmation = v11_confirm_vertical_contact(
        state["vertical_contact_steps"], vertical_contact_now,
        int(cfg.vertical_contact_confirmation_frames),
    )
    state["vertical_contact_now"] = vertical_contact_now
    state["vertical_contact_steps"] = contact_confirmation["steps"]
    confirmed_contact = contact_confirmation["confirmed"]
    new_contact = confirmed_contact & ~state["vertical_contact"]
    state["vertical_contact"] |= confirmed_contact
    state["front_collision_event"] = new_contact.any(dim=1)
    if soft_alternation:
        state["episode_front_collision"] |= state["front_collision_event"]
    state["post_first_collision_count"] += (
        state["front_collision_event"] & state["sequence_started"]
    ).sum()

    last_delta_for_anchor = state["last_completed_riser_world_xy"] - root[:, :2]
    last_riser_x_for_anchor = (
        cy * last_delta_for_anchor[:, 0] + sy * last_delta_for_anchor[:, 1]
    )
    anchor_now = v11_anchor_support_valid(
        state["anchor_side"], wheel_yaw[..., 0], wheel_bottom, forces[..., 2],
        last_riser_x_for_anchor, state["last_completed_upper_z"], target_riser_x,
        wheel_radius=cfg.wheel_radius, front_margin=cfg.front_margin,
        support_tolerance=cfg.upper_support_tolerance,
        force_threshold=cfg.contact_force_threshold,
    )
    state["anchor_support_steps"] = torch.where(
        anchor_now, state["anchor_support_steps"] + 1,
        torch.zeros_like(state["anchor_support_steps"]),
    )
    anchor_confirmed = state["anchor_support_steps"] >= int(cfg.front_support_confirmation_frames)
    state["anchor_support_valid"] = anchor_confirmed & state["sequence_started"]

    candidate_ready = v11_softalt_candidate_ready(
        state["clearance"], state["crossing"], state["support"], state["vertical_contact"],
        eligible_sides, state["sequence_started"], state["anchor_support_valid"],
        soft_alternation=soft_alternation,
    )
    if enforce_crossing_clearance:
        candidate_ready &= state["crossing_clearance_ok"]
    state["candidate_ready"] = candidate_ready
    completion = (
        v11_classify_soft_alternation(state["previous_completed_side"], candidate_ready)
        if soft_alternation else
        v11_latch_alternating_front_completion(state["expected_side"], candidate_ready)
    )
    event = completion["event"]
    completed_side = completion["completed_side"]
    first_event = (
        completion["first_completion"] if soft_alternation
        else event & ~state["sequence_started"]
    )
    alternating_event = (
        completion["alternating_completion"] if soft_alternation
        else v11_alternating_reward_event(event, state["sequence_started"])
    )
    repeat_event = (
        completion["same_side_repeat"] if soft_alternation
        else torch.zeros_like(event)
    )
    simultaneous_event = (
        completion["simultaneous_completion"] if soft_alternation
        else torch.zeros_like(event)
    )
    state["landing_event"] = event
    state["first_completion_event"] = first_event
    state["alternating_completion_event"] = alternating_event
    state["same_side_repeat_event"] = repeat_event
    state["simultaneous_completion_event"] = simultaneous_event
    # Legacy V11 still rewards every hard-gated post-first completion.  SoftAlt
    # rewards only actual opposite-side transitions through its dedicated term.
    reward_event = alternating_event
    state["alternating_reward_event"] = alternating_event
    state["completed_side"] = torch.where(event, completed_side, state["completed_side"])
    state["current_completed_side"] = torch.where(event, completed_side, state["current_completed_side"])
    state["responsible"] = torch.where(event, completed_side, state["responsible"])
    state["bilateral_landing_event"].zero_()

    selected_current = state["clearance_escrow"].gather(
        1, completed_side.clamp_min(0)[:, None],
    ).squeeze(1)
    if soft_alternation:
        state["clearance_settlement"] = selected_current * event.float()
        second_event = torch.zeros_like(event)
    else:
        state["pending_first_clearance"] = torch.where(
            first_event, selected_current, state["pending_first_clearance"],
        )
        second_event = reward_event & (state["completed_riser_count"] == 1)
        later_settlement = selected_current * float(cfg.alternating_clearance_settlement_scale)
        first_pair_settlement = state["pending_first_clearance"] + selected_current
        state["clearance_settlement"] = torch.where(
            second_event, first_pair_settlement,
            torch.where(reward_event, later_settlement, torch.zeros_like(selected_current)),
        )
    selected_surplus = state["clearance_surplus_escrow"].gather(
        1, completed_side.clamp_min(0)[:, None],
    ).squeeze(1)
    state["clearance_surplus_settlement"] = selected_surplus * (
        event if soft_alternation else reward_event
    ).float()
    state["pending_first_clearance"] = torch.where(
        second_event, torch.zeros_like(state["pending_first_clearance"]),
        state["pending_first_clearance"],
    )
    selected_margin = state["crossing_clearance_margin_per_side"].gather(
        1, completed_side.clamp_min(0)[:, None],
    ).squeeze(1)
    state["crossing_clearance_margin"] = selected_margin * event.float()
    state["crossing_clearance_margin_sum"] += (selected_margin * event.float()).sum()
    state["crossing_clearance_margin_count"] += event.sum()

    state["responsibility_fl_count"] += (first_event & (completed_side == 0)).sum()
    state["responsibility_fr_count"] += (first_event & (completed_side == 1)).sum()
    state["responsibility_tie_count"] += (first_event & candidate_ready.all(dim=1)).sum()
    state["alternating_completion_count"] += event.sum()

    if soft_alternation:
        previous = state["previous_completed_side"]
        state["first_completion_count"] += first_event.sum()
        state["fl_completion_count"] += (event & (completed_side == 0)).sum()
        state["fr_completion_count"] += (event & (completed_side == 1)).sum()
        state["same_side_repeat_count"] += repeat_event.sum()
        state["simultaneous_completion_count"] += simultaneous_event.sum()
        state["simultaneous_transition_count"] += (simultaneous_event & ~first_event).sum()
        state["fl_to_fr_count"] += (alternating_event & (previous == 0) & (completed_side == 1)).sum()
        state["fr_to_fl_count"] += (alternating_event & (previous == 1) & (completed_side == 0)).sum()
        state["fl_to_fl_repeat_count"] += (repeat_event & (completed_side == 0)).sum()
        state["fr_to_fr_repeat_count"] += (repeat_event & (completed_side == 1)).sum()
        state["episode_alternating_count"] += alternating_event.long()
        state["episode_repeat_count"] += repeat_event.long()
        state["episode_simultaneous_count"] += simultaneous_event.long()
        _v11_softalt_sequence_diagnostic(env, reset, event, completed_side, repeat_event)

    wrong_side_beyond = (
        target_active[:, None] & ~eligible_sides
        & (wheel_yaw[..., 0] >= target_riser_x[:, None])
    ).any(dim=1)
    if soft_alternation:
        wrong_side_beyond = torch.zeros_like(wrong_side_beyond)
    violation_now = wrong_side_beyond & ~state["alternation_violation_latched"]
    state["alternation_violation_now"] = violation_now
    state["alternation_violation_latched"] |= wrong_side_beyond
    state["alternation_violation_steps"] = torch.where(
        wrong_side_beyond, state["alternation_violation_steps"] + 1,
        torch.zeros_like(state["alternation_violation_steps"]),
    )
    state["alternation_violation_confirmed"] |= (
        state["alternation_violation_steps"] >= int(cfg.alternation_violation_confirmation_frames)
    )
    if wrong_side_beyond.any():
        state["clearance_escrow"][wrong_side_beyond] = 0.0
        state["clearance_surplus_escrow"][wrong_side_beyond] = 0.0
        state["hipy_participation_escrow"][wrong_side_beyond] = 0.0
    state["alternation_violation_count"] += violation_now.sum()

    state["last_completed_riser_world_xy"] = torch.where(
        event[:, None], state["reference_riser_world_xy"], state["last_completed_riser_world_xy"],
    )
    state["last_completed_upper_z"] = torch.where(
        event, target_upper_z, state["last_completed_upper_z"],
    )
    state["credited_reference_valid"] |= event
    state["credited_reference_world_xy"] = torch.where(
        event[:, None], state["reference_riser_world_xy"], state["credited_reference_world_xy"],
    )
    state["completed_riser_count"] += event.long()
    state["sequence_started"] |= event
    state["expected_side"] = torch.where(
        event, completion["next_expected_side"], state["expected_side"],
    )
    if soft_alternation:
        state["previous_completed_side"] = torch.where(
            event, completed_side, state["previous_completed_side"],
        )
    state["anchor_side"] = torch.where(event, completed_side, state["anchor_side"])
    state["anchor_support_steps"] = torch.where(
        event, torch.full_like(state["anchor_support_steps"], int(cfg.front_support_confirmation_frames)),
        state["anchor_support_steps"],
    )
    state["anchor_support_valid"] = torch.where(
        event, torch.ones_like(state["anchor_support_valid"]), state["anchor_support_valid"],
    )

    scores = v11_landing_scores(
        wheel_yaw[..., :2], region, phase_active,
    )

    # Final-flight closure: after the last visible face, both front wheels must
    # support the final top tread before the unchanged bilateral rear-support
    # platform confirmation can complete the episode milestone.
    last_delta = state["last_completed_riser_world_xy"] - root[:, :2]
    last_riser_x = cy * last_delta[:, 0] + sy * last_delta[:, 1]
    front_top_support = (
        (wheel_yaw[..., 0] >= (
            last_riser_x + float(cfg.wheel_radius) + float(cfg.front_margin)
        )[:, None])
        & ((wheel_bottom - state["last_completed_upper_z"][:, None]).abs()
           <= float(cfg.upper_support_tolerance))
        & (force_norm >= float(cfg.contact_force_threshold))
        & (forces[..., 2] > float(cfg.contact_force_threshold))
    ).all(dim=1)
    rear_yaw = wheel_positions_yaw(asset, env._v11_rear_body_ids)
    rear_forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_rear_contact_ids]
    rear_crossed = (rear_yaw[..., 0] >= last_riser_x[:, None]).all(dim=1)
    rear_support = (rear_forces.norm(dim=-1) >= float(cfg.contact_force_threshold)) & (
        rear_forces[..., 2] > float(cfg.contact_force_threshold)
    )
    raw_no_riser = ~risers["valid"].any(dim=1)
    platform_candidate = (
        state["sequence_started"] & raw_no_riser & front_top_support & rear_crossed
    )
    rear_window = v11_update_bilateral_support_window(
        state["rear_support_latched"], state["rear_support_window_age"],
        rear_support, platform_candidate, int(cfg.bilateral_rear_support_window_frames),
    )
    state["rear_support_latched"] = rear_window["latched"]
    state["rear_support_window_age"] = rear_window["age"]
    state["platform_no_riser_steps"] = torch.where(
        platform_candidate,
        state["platform_no_riser_steps"] + 1,
        torch.zeros_like(state["platform_no_riser_steps"]),
    )
    stable_platform = platform_candidate & rear_window["ready"]
    state["platform_support_steps"] = torch.where(
        stable_platform,
        state["platform_support_steps"] + 1,
        torch.zeros_like(state["platform_support_steps"]),
    )
    platform_now = (~state["platform_confirmed"]) & (
        state["platform_no_riser_steps"] >= int(cfg.platform_no_riser_confirmation_frames)
    ) & (
        state["platform_support_steps"] >= int(cfg.platform_rear_support_confirmation_frames)
    )
    state["platform_exit_event"] = platform_now
    state["platform_confirmed"] |= platform_now
    if soft_alternation:
        state["platform_success_count"] += platform_now.sum()

    # Keep event data available to all RewardManager terms in this control
    # step; the next step will select and clear the next target phase.
    state["reference_valid"] &= ~event
    state["target_match_valid"] &= ~event
    state["target_slot"] = torch.where(event, torch.full_like(state["target_slot"], -1), state["target_slot"])
    state.update({
        "region": region, "scores": scores, "target_active": target_active,
        "target_riser_x": target_riser_x, "target_upper_z": target_upper_z,
        "eligible_sides": eligible_sides, "wheel_yaw": wheel_yaw,
        "wheel_bottom": wheel_bottom, "upper_z": target_upper_z,
    })
    return state


def _v11_bilateral_landing_state(env, asset: Articulation, risers: dict[str, torch.Tensor], active: torch.Tensor):
    """Update continuous first-tread geometry and a one-time FL/FR outcome."""
    _v11_initialize_landing(env, asset)
    state, cfg = env._v11_landing, env.cfg.v11_perception
    adjacent_credit = bool(getattr(getattr(env.cfg, "v111_rear_posture", None),
                                   "adjacent_landing_credit", False))
    if adjacent_credit and "adjacent_credit_blocked" not in state:
        state["adjacent_credit_blocked"] = torch.zeros_like(state["reference_valid"])
        state["approach_seen"] = torch.zeros_like(state["crossing"])
    handoff = bool(getattr(getattr(env.cfg, "v111_rear_posture", None), "same_tread_handoff", False))
    if handoff and "bilateral_confirmed" not in state:
        state["handoff_leader"] = torch.full_like(state["responsible"], -1)
        state["bilateral_confirmed"] = torch.zeros_like(state["reference_valid"])
        state["front_airborne_steps"] = torch.zeros_like(state["target_lost_steps"])
        state["unsupported_handoff"] = torch.zeros_like(state["reference_valid"])
    timed_cycle = handoff and bool(getattr(env.cfg.v111_rear_posture, "timed_support_cycle", False))
    reset = env.episode_length_buf < state["last_episode_length"]
    _v11_reset_landing(state, reset, reset_credit_ledger=True)
    state["last_episode_length"] = env.episode_length_buf.clone()
    first_tread = bool(getattr(getattr(env.cfg, "v111_rear_posture", None), "physical_first_tread", False))
    if first_tread:
        from .v111_first_tread import prepare_first_tread
        risers = prepare_first_tread(env, asset, risers)
    if timed_cycle:
        from .v111_support_cycle import prepare_cycle_risers
        risers = prepare_cycle_risers(env, asset, risers, state, reset)
    state["landing_event"].zero_()
    state["bilateral_landing_event"].zero_()
    state["front_collision_event"].zero_()
    state["front_crossing_event"].zero_()
    state["low_clearance_crossing_event"].zero_()
    state["front_support_event"].zero_()
    state["front_support_clearance_settlement"].zero_()
    state["front_candidate_event"].zero_()
    state["platform_exit_event"].zero_()
    state["clearance_settlement"].zero_()
    state["second_clearance_settlement"].zero_()
    state["hipy_participation_settlement"].zero_()

    # Keep an encounter identity in world coordinates.  The detected local x
    # naturally moves as the robot advances, so it cannot identify whether the
    # detector has promoted the second riser to the current one.
    root = asset.data.root_pos_w
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    detected_world_xy = torch.stack((
        root[:, 0] + cy * risers["riser_x"][:, 0] - sy * risers["riser_y"][:, 0],
        root[:, 1] + sy * risers["riser_x"][:, 0] + cy * risers["riser_y"][:, 0],
    ), dim=1)
    # The reward-side encounter is anchored by the current riser.  A final
    # stair often has no second riser in the scan, which must not discard a
    # valid landing target or prevent subsequent platform confirmation.
    stair_forward = None
    if hasattr(env.cfg, "v112_handoff"):
        from .v112_handoff import stair_coordinates, wheels_in_stair_frame
        risers, stair_forward = stair_coordinates(env, asset, risers)
        cy, sy = stair_forward[:, 0], stair_forward[:, 1]
    detected_current = risers["valid"][:, 0]
    state["target_lost_steps"] = torch.where(
        state["reference_valid"] & ~detected_current,
        state["target_lost_steps"] + 1,
        torch.zeros_like(state["target_lost_steps"]),
    )
    riser_switch = state["reference_valid"] & detected_current & (
        (detected_world_xy - state["reference_riser_world_xy"]).norm(dim=1)
        >= float(cfg.encounter_riser_switch_distance)
    )
    if adjacent_credit:
        heading = env.command_manager.get_term("base_velocity").v11_path_heading_w
        route = torch.stack((heading.cos(), heading.sin()), 1)
        if stair_forward is not None:
            route = stair_forward
        displacement = ((detected_world_xy - state["reference_riser_world_xy"]) * route).sum(1)
        riser_switch = state["reference_valid"] & detected_current & (
            displacement.abs() >= float(cfg.encounter_riser_switch_distance))
    # A zero B1/B2 tuple alone is not a platform: the raw map must remain
    # riser-free, both rear wheels must have passed the locked riser, and a
    # rear support contact must persist.  This state is reward-side only; the
    # actor continues to receive the unchanged delayed 6-D input.
    rear_yaw = wheel_positions_yaw(asset, env._v11_rear_body_ids)
    if stair_forward is not None:
        rear_yaw = wheels_in_stair_frame(asset, env._v11_rear_body_ids, stair_forward)
    reference_delta_w = state["reference_riser_world_xy"] - root[:, :2]
    reference_x_yaw = cy * reference_delta_w[:, 0] + sy * reference_delta_w[:, 1]
    rear_crossed = (rear_yaw[..., 0] >= reference_x_yaw[:, None]).all(dim=1)
    rear_forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_rear_contact_ids]
    rear_support = (rear_forces.norm(dim=-1) >= float(cfg.contact_force_threshold)) & (
        rear_forces[..., 2] > float(cfg.contact_force_threshold)
    )
    raw_no_riser = ~risers["valid"].any(dim=1)
    front_complete = (
        state["candidate_ready"].all(dim=1)
        if bool(cfg.require_bilateral_front_completion)
        else state["responsible"] >= 0
    )
    if handoff:
        front_complete = state["bilateral_confirmed"]
    platform_candidate = state["reference_valid"] & front_complete & raw_no_riser & rear_crossed
    if bool(cfg.require_bilateral_rear_support):
        rear_window = v11_update_bilateral_support_window(
            state["rear_support_latched"], state["rear_support_window_age"],
            rear_support, platform_candidate, int(cfg.bilateral_rear_support_window_frames),
        )
        state["rear_support_latched"] = rear_window["latched"]
        state["rear_support_window_age"] = rear_window["age"]
        rear_support_ready = rear_window["ready"]
    else:
        rear_support_ready = rear_support.any(dim=1)
    _v11_update_platform_exit_state(
        state, cfg, raw_no_riser, front_complete, rear_crossed, rear_support_ready
    )

    platform_release_steps = getattr(cfg, "platform_target_loss_release_steps", None)
    release_steps = (
        int(cfg.target_loss_release_steps)
        if platform_release_steps is None else int(platform_release_steps)
    )
    release = (state["responsible"] >= 0) & (
        riser_switch
        | (~state["platform_confirmed"] & (state["target_lost_steps"] > release_steps))
    )
    if adjacent_credit:
        # Scanner promotion must never merge one tread's crossing/support
        # history with the next tread. Keep physical completion for resync,
        # but withhold the promoted encounter's credit after a missed tread.
        previous_complete = state["bilateral_confirmed"] if handoff else state["candidate_ready"].all(dim=1)
        missed_previous = riser_switch & ~previous_complete
        if hasattr(env.cfg, "v112_handoff"):
            from .v135_terrain_geometry import v135_terrain_geometry
            expected_width = v135_terrain_geometry(env)["step_width"]
            missed_previous |= riser_switch & (
                (displacement - expected_width).abs() >= float(cfg.encounter_riser_switch_distance))
        blocked = state["adjacent_credit_blocked"].clone()
        blocked = torch.where(riser_switch, missed_previous, blocked)
        release |= riser_switch
    _v11_reset_landing(state, release, reset_credit_ledger=False)
    if adjacent_credit:
        state["adjacent_credit_blocked"] = blocked
    assign_reference = detected_current & ~state["reference_valid"]
    state["reference_valid"] |= assign_reference
    state["reference_riser_world_xy"] = torch.where(
        assign_reference[:, None], detected_world_xy, state["reference_riser_world_xy"]
    )

    # The ray detector returns yaw-local coordinates about the base.  The
    # lateral interval is deliberately a deployable corridor contract rather
    # than an unobservable terrain-sidewall oracle.
    tread_y_min = risers["riser_y"][:, 0] - float(cfg.tread_corridor_half_width)
    tread_y_max = risers["riser_y"][:, 0] + float(cfg.tread_corridor_half_width)
    region = v11_safe_landing_region(
        risers["riser_x"][:, 0], tread_y_min, tread_y_max,
        cfg.wheel_radius, cfg.front_margin, cfg.lateral_margin,
    )
    v112_bound = None
    if hasattr(env.cfg, "v112_handoff"):
        from .v112_handoff import support_bounds
        from .v135_terrain_geometry import v135_terrain_geometry
        width = v135_terrain_geometry(env)["step_width"]
        region["x_min"], v112_bound, valid_width = support_bounds(
            risers["riser_x"][:, 0], width, env.cfg.v112_handoff)
        region["valid"] &= valid_width
    same_credited_riser = state["credited_reference_valid"] & detected_current & (
        (detected_world_xy - state["credited_reference_world_xy"]).norm(dim=1)
        < float(cfg.encounter_riser_switch_distance)
    )
    finish_bilateral = bool(cfg.require_bilateral_front_completion) & state["reference_valid"] & (
        ~state["candidate_ready"].all(dim=1)
    )
    if handoff:
        finish_bilateral = state["reference_valid"] & ~state["bilateral_confirmed"]
    if timed_cycle:
        finish_bilateral |= state["reference_valid"] & ~state.get("cycle_done", torch.zeros_like(active))
    target_active = active & detected_current & region["valid"] & (
        ~same_credited_riser | finish_bilateral
    )
    wheel_yaw = wheel_positions_yaw(asset, env._v11_front_body_ids)
    if stair_forward is not None:
        wheel_yaw = wheels_in_stair_frame(asset, env._v11_front_body_ids, stair_forward)
    wheel_world = asset.data.body_pos_w[:, env._v11_front_body_ids]
    forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids]
    wheel_bottom = wheel_world[..., 2] - float(cfg.wheel_radius)
    upper_z = risers["riser_lower_world"][:, 0] + risers["riser_height"][:, 0]
    lower_z = risers["riser_lower_world"][:, 0]
    before = wheel_yaw[..., 0] < risers["riser_x"][:, :1]
    escrow = v11_update_clearance_escrow(
        state["max_clearance_progress"], state["clearance_escrow"],
        state["previous_wheel_x"], state["previous_wheel_x_valid"],
        wheel_yaw[..., 0], wheel_bottom, lower_z[:, None], upper_z[:, None],
        target_active[:, None] & before, float(cfg.clearance_margin),
    )
    state["max_clearance_progress"] = escrow["max_progress"]
    state["clearance_escrow"] = escrow["escrow"]
    state["previous_wheel_x"] = escrow["previous_x"]
    state["previous_wheel_x_valid"] = escrow["previous_x_valid"]
    state["clearance"] |= escrow["clearance"]

    crossing_now = target_active[:, None] & (wheel_yaw[..., 0] >= risers["riser_x"][:, :1])
    if adjacent_credit:
        state["approach_seen"] |= target_active[:, None] & before
        crossing_now &= state["approach_seen"]
    new_crossing = crossing_now & ~state["crossing"]
    crossing_clearance = v11_crossing_clearance_gate(
        wheel_world[..., 2], upper_z[:, None], new_crossing,
        wheel_radius=cfg.wheel_radius, clearance_margin=cfg.clearance_margin,
    )
    state["front_crossing_event"] = new_crossing
    state["crossing_clearance_ok"] |= crossing_clearance["clear"]
    state["low_clearance_crossing_event"] = crossing_clearance["low"]
    state["crossing_wheel_center_above_tread"] = torch.where(
        new_crossing, crossing_clearance["center_above_tread"],
        state["crossing_wheel_center_above_tread"],
    )
    state["crossing_sample_valid"] |= new_crossing
    state["crossing_clearance_margin_per_side"] = torch.where(
        new_crossing, crossing_clearance["margin"],
        state["crossing_clearance_margin_per_side"],
    )
    state["front_crossing_count"] += new_crossing.sum()
    state["low_clearance_crossing_count"] += crossing_clearance["low"].sum()
    state["front_crossing_center_sum"] += (
        crossing_clearance["center_above_tread"] * new_crossing.float()
    ).sum()
    state["front_crossing_margin_sum"] += (
        crossing_clearance["margin"] * new_crossing.float()
    ).sum()
    event_margin = torch.where(
        new_crossing, crossing_clearance["margin"],
        torch.full_like(crossing_clearance["margin"], float("inf")),
    ).amin()
    state["front_crossing_min_margin"] = torch.minimum(
        state["front_crossing_min_margin"], event_margin,
    )
    # A low crossing cannot retain dense credit accumulated before the face.
    state["clearance_escrow"] = torch.where(
        crossing_clearance["low"], torch.zeros_like(state["clearance_escrow"]),
        state["clearance_escrow"],
    )
    state["crossing"] |= crossing_now
    one_crossed = state["crossing"].any(dim=1) & ~state["crossing"].all(dim=1)
    state["crossing_gap_steps"] = torch.where(
        one_crossed, state["crossing_gap_steps"] + 1, torch.zeros_like(state["crossing_gap_steps"])
    )
    force_norm = forces.norm(dim=-1)
    support_now = (
        target_active[:, None]
        & (wheel_yaw[..., 0] >= region["x_min"][:, None])
        & (wheel_yaw[..., 1] >= region["y_min"][:, None])
        & (wheel_yaw[..., 1] <= region["y_max"][:, None])
        & ((wheel_bottom - upper_z[:, None]).abs() <= float(cfg.upper_support_tolerance))
        & (force_norm >= float(cfg.contact_force_threshold))
        & (forces[..., 2] > float(cfg.contact_force_threshold))
    )
    if adjacent_credit:
        # The next riser is the upper bound of this tread, not another valid
        # landing surface for the current target.
        next_bound = torch.where(risers["valid"][:, 1],
                                 risers["riser_x"][:, 1] - float(cfg.wheel_radius),
                                 torch.full_like(upper_z, float("inf")))
        if v112_bound is not None:
            next_bound = v112_bound
        support_now &= wheel_yaw[..., 0] <= next_bound[:, None]
        state["support_steps"] = torch.where(support_now, state["support_steps"] + 1,
                                             torch.zeros_like(state["support_steps"]))
        support_now &= state["support_steps"] >= 3
    if handoff:
        contact = forces[..., 2] > float(cfg.contact_force_threshold)
        lifted = wheel_bottom > lower_z[:, None] + env.cfg.v111_rear_posture.front_airborne_min_lift
        airborne = target_active & (~contact & lifted).all(1)
        state["front_airborne_steps"] = torch.where(
            airborne, state["front_airborne_steps"] + 1, torch.zeros_like(state["front_airborne_steps"]))
        state["unsupported_handoff"] |= state["front_airborne_steps"] >= env.cfg.v111_rear_posture.front_airborne_confirmation_frames
    previous_support = state["support"].clone()
    state["support"] |= support_now
    one_supported = state["support"].any(dim=1) & ~state["support"].all(dim=1)
    state["support_gap_steps"] = torch.where(
        one_supported, state["support_gap_steps"] + 1, torch.zeros_like(state["support_gap_steps"])
    )
    vertical_contact_now = classify_vertical_riser_contact(
        wheel_yaw[..., :2], wheel_world[..., 2], forces, risers["riser_x"][:, :1],
        lower_z[:, None], upper_z[:, None], wheel_radius=cfg.wheel_radius,
        force_threshold=cfg.contact_force_threshold,
        normal_ratio_threshold=cfg.vertical_force_ratio,
    ) & target_active[:, None]
    contact_confirmation = v11_confirm_vertical_contact(
        state["vertical_contact_steps"], vertical_contact_now,
        int(cfg.vertical_contact_confirmation_frames),
    )
    state["vertical_contact_now"] = vertical_contact_now
    state["vertical_contact_steps"] = contact_confirmation["steps"]
    confirmed_vertical_contact = contact_confirmation["confirmed"]
    new_vertical_contact = confirmed_vertical_contact & ~state["vertical_contact"]
    state["vertical_contact"] |= confirmed_vertical_contact
    # Cold-start shaping remains an escrowed landing signal: sample the
    # target-relative height at the actual face crossing, but release it only
    # after that same wheel reaches upper-tread support without a confirmed
    # vertical collision.  Crossing or lifting alone can never farm reward.
    support_credit_distance = torch.linalg.norm(
        state["front_support_credited_world_xy"]
        - state["reference_riser_world_xy"][:, None, :],
        dim=-1,
    )
    already_support_credited = state["front_support_credited_valid"] & (
        support_credit_distance < float(cfg.encounter_riser_switch_distance)
    )
    safe_support_event = (
        ~previous_support
        & state["support"]
        & state["crossing"]
        & state["crossing_sample_valid"]
        & ~state["vertical_contact"]
        & ~already_support_credited
    )
    required_center_above_tread = float(cfg.wheel_radius) + float(cfg.clearance_margin)
    state["front_support_event"] = safe_support_event
    state["front_support_clearance_settlement"] = v11_safe_support_clearance_progress(
        state["crossing_wheel_center_above_tread"], safe_support_event,
        required_center_above_tread=required_center_above_tread,
    )
    state["front_support_credited_valid"] |= safe_support_event
    state["front_support_credited_world_xy"] = torch.where(
        safe_support_event[..., None],
        state["reference_riser_world_xy"][:, None, :],
        state["front_support_credited_world_xy"],
    )
    previous_candidate_ready = state["candidate_ready"].clone()
    candidate_ready = state["clearance"] & state["crossing"] & state["support"] & ~state["vertical_contact"]
    if bool(cfg.require_clearance_at_crossing):
        candidate_ready &= state["crossing_clearance_ok"]
    state["candidate_ready"] = candidate_ready
    state["front_candidate_event"] = (
        ~previous_candidate_ready & candidate_ready & ~already_support_credited
    )
    previous = state["responsible"].clone()
    state["responsible"] = v11_latch_responsible_front_wheel(previous, candidate_ready)
    state["landing_event"] = (previous < 0) & (state["responsible"] >= 0)
    state["bilateral_landing_event"] = (
        ~previous_candidate_ready.all(dim=1) & candidate_ready.all(dim=1) & (state["responsible"] >= 0)
    )
    if handoff:
        # History is useful for geometry/resync, but cannot substitute for
        # both front wheels CURRENTLY bearing weight on this same tread.
        single = support_now.sum(1) == 1
        leader_now = support_now.long().argmax(1)
        state["handoff_leader"] = torch.where(
            (state["handoff_leader"] < 0) & single, leader_now, state["handoff_leader"])
        together = candidate_ready.all(1) & support_now.all(1)
        event = together & ~state["bilateral_confirmed"]
        state["bilateral_confirmed"] |= together
        state["bilateral_landing_event"] = event
        if timed_cycle:
            from .v111_support_cycle import update_cycle
            state["target_active"] = target_active
            update_cycle(env, asset, state, support_now, forces, lower_z, before)
            if hasattr(env.cfg, "v112_handoff"):
                from .v112_handoff import certify_cycle
                certify_cycle(state)
            if not hasattr(env.cfg, "v112_handoff"):
                state["bilateral_landing_event"] = state["cycle_event"]
    assigned_now = state["landing_event"]
    first_settlement_event = v11_front_reward_event(
        assigned_now, state["bilateral_landing_event"],
        bool(cfg.defer_first_front_rewards_until_bilateral),
    )
    if bool(cfg.defer_first_front_rewards_until_bilateral):
        state["clearance_settlement"] = v11_settle_bilateral_escrow(
            state["clearance_escrow"], first_settlement_event,
        )
    else:
        state["clearance_settlement"] = v11_settle_clearance_escrow(
            state["clearance_escrow"], state["responsible"], first_settlement_event,
        )
    second_front = 1 - state["responsible"].clamp_min(0)
    state["second_clearance_settlement"] = v11_settle_clearance_escrow(
        state["clearance_escrow"], second_front, state["bilateral_landing_event"],
    )
    state["credited_reference_valid"] |= assigned_now
    state["credited_reference_world_xy"] = torch.where(
        assigned_now[:, None], state["reference_riser_world_xy"], state["credited_reference_world_xy"]
    )
    state["responsibility_fl_count"] += (assigned_now & (state["responsible"] == 0)).sum()
    state["responsibility_fr_count"] += (assigned_now & (state["responsible"] == 1)).sum()
    state["responsibility_tie_count"] += (assigned_now & candidate_ready.all(dim=1)).sum()
    # Diagnostic semantics are now identical to the hard failure: every newly
    # confirmed front-riser collision is counted, regardless of whether a
    # first responsible wheel had already been selected.
    state["front_collision_event"] = new_vertical_contact.any(dim=1)
    scores = v11_landing_scores(wheel_yaw[..., :2], region, target_active[:, None].expand(-1, 2))
    state.update({
        "region": region, "scores": scores, "target_active": target_active,
        "wheel_yaw": wheel_yaw, "wheel_bottom": wheel_bottom, "upper_z": upper_z,
    })
    if adjacent_credit:
        credit = ~state["adjacent_credit_blocked"]
        if handoff and not hasattr(env.cfg, "v112_handoff"):
            credit &= state["bilateral_confirmed"] & ~state["unsupported_handoff"] & (state["handoff_leader"] >= 0)
            if timed_cycle:
                credit &= state["cycle_done"]
        for name in ("landing_event", "bilateral_landing_event", "front_support_event",
                     "front_candidate_event", "front_support_clearance_settlement",
                     "clearance_settlement", "second_clearance_settlement",
                     "hipy_participation_settlement"):
            mask = credit if state[name].ndim == 1 else credit[:, None]
            state[name] *= mask
    if first_tread:
        from .v111_first_tread import finish_first_tread
        finish_first_tread(env, asset, state)
    return state


def _v11_landing_state(env, asset: Articulation, risers: dict[str, torch.Tensor], active: torch.Tensor):
    """Dispatch the explicit V11/V12 front-completion contract."""
    mode = env.cfg.v11_perception.front_completion_mode
    if mode == "alternating_risers":
        return _v11_alternating_landing_state(env, asset, risers, active)
    if mode == "bilateral_same_riser":
        return _v11_bilateral_landing_state(env, asset, risers, active)
    raise RuntimeError(f"Unsupported V11 front completion mode: {mode!r}")


def v11_descent_phase_active(
    active: torch.Tensor,
    reference_valid: torch.Tensor,
    match_valid: torch.Tensor,
    front_complete: torch.Tensor,
    rear_complete: torch.Tensor,
    retain_after_front: bool = True,
) -> torch.Tensor:
    """Keep a locked descent edge active while the rear wheels close the loop.

    Before front support is confirmed, the edge must remain visible.  After
    that point its world pose and lower-tread height have already been locked
    from a valid observation, so it may leave the forward scan while the rear
    wheels still traverse it.
    """
    following = front_complete & ~rear_complete & bool(retain_after_front)
    return active & reference_valid & (match_valid | following)


def _v11_descent_state(
    env, asset: Articulation, risers: dict[str, torch.Tensor], active: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Track bilateral front-to-rear descent completion without a gait order."""
    _v11_initialize_landing(env, asset)
    cfg = env.cfg.v11_perception
    n = env.num_envs
    if not hasattr(env, "_v11_descent"):
        env._v11_descent = {
            "last_episode_length": env.episode_length_buf.clone(),
            "reference_valid": torch.zeros(n, device=env.device, dtype=torch.bool),
            "reference_world_xy": torch.zeros((n, 2), device=env.device),
            "reference_lower_z": torch.zeros(n, device=env.device),
            "last_completed_world_xy": torch.zeros((n, 2), device=env.device),
            "sequence_started": torch.zeros(n, device=env.device, dtype=torch.bool),
            "target_slot": torch.full((n,), -1, device=env.device, dtype=torch.long),
            "target_active": torch.zeros(n, device=env.device, dtype=torch.bool),
            "target_lost_steps": torch.zeros(n, device=env.device, dtype=torch.long),
            "front_support_steps": torch.zeros((n, 2), device=env.device, dtype=torch.long),
            "rear_support_steps": torch.zeros((n, 2), device=env.device, dtype=torch.long),
            "rear_support_credited": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
            "rear_support_event": torch.zeros((n, 2), device=env.device, dtype=torch.bool),
            "front_complete": torch.zeros(n, device=env.device, dtype=torch.bool),
            "rear_complete": torch.zeros(n, device=env.device, dtype=torch.bool),
            "completion_event": torch.zeros(n, device=env.device, dtype=torch.bool),
            "rear_follow_gap_steps": torch.zeros(n, device=env.device, dtype=torch.long),
            "rear_roll_error": torch.zeros(n, device=env.device),
            "contact_count": torch.zeros(n, device=env.device),
            "target_edge_x": torch.zeros(n, device=env.device),
            "target_lower_z": torch.zeros(n, device=env.device),
        }
    state = env._v11_descent
    reset = env.episode_length_buf < state["last_episode_length"]
    for name, value in state.items():
        if name != "last_episode_length" and isinstance(value, torch.Tensor) and value.ndim >= 1:
            value[reset] = 0
    state["target_slot"][reset] = -1
    state["last_episode_length"] = env.episode_length_buf.clone()
    state["completion_event"].zero_()
    state["rear_roll_error"].zero_()
    state["rear_support_event"].zero_()

    root = asset.data.root_pos_w
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    detected_world_xy = torch.stack((
        root[:, None, 0] + cy[:, None] * risers["riser_x"] - sy[:, None] * risers["riser_y"],
        root[:, None, 1] + sy[:, None] * risers["riser_x"] + cy[:, None] * risers["riser_y"],
    ), dim=-1)
    delta = detected_world_xy - state["last_completed_world_xy"][:, None, :]
    separation = cy[:, None] * delta[..., 0] + sy[:, None] * delta[..., 1]
    slot = v11_select_next_uncredited_riser(
        risers["valid"], separation, state["sequence_started"],
        float(cfg.encounter_riser_switch_distance),
    )
    assign = active & ~state["reference_valid"] & (slot >= 0)
    if assign.any():
        chosen = slot.clamp_min(0)
        world = detected_world_xy.gather(1, chosen[:, None, None].expand(-1, 1, 2)).squeeze(1)
        lower = risers["riser_lower_world"].gather(1, chosen[:, None]).squeeze(1)
        state["reference_valid"][assign] = True
        state["reference_world_xy"][assign] = world[assign]
        state["reference_lower_z"][assign] = lower[assign]
        state["target_slot"][assign] = chosen[assign]
        state["front_support_steps"][assign] = 0
        state["rear_support_steps"][assign] = 0
        state["rear_support_credited"][assign] = False
        state["front_complete"][assign] = False
        state["rear_complete"][assign] = False
        state["rear_follow_gap_steps"][assign] = 0

    ref_distance = torch.linalg.norm(
        detected_world_xy - state["reference_world_xy"][:, None, :], dim=-1,
    )
    matching = risers["valid"] & (
        ref_distance < float(cfg.encounter_riser_switch_distance)
    )
    masked = torch.where(matching, ref_distance, torch.full_like(ref_distance, torch.inf))
    live_slot = masked.argmin(dim=1)
    match_valid = state["reference_valid"] & matching.any(dim=1)
    state["target_slot"] = torch.where(match_valid, live_slot, state["target_slot"])
    state["target_lost_steps"] = torch.where(
        state["reference_valid"] & ~match_valid, state["target_lost_steps"] + 1,
        torch.zeros_like(state["target_lost_steps"]),
    )
    # A live edge is mandatory until both front wheels have safely descended.
    # Afterwards, retain the locked world geometry for the bounded rear-follow
    # phase: the edge naturally disappears from the forward two-riser scan
    # before the rear wheels can reach it.
    retain_rear_phase = bool(cfg.enable_persistent_descent_rear_phase)
    lost = (
        (state["target_lost_steps"] > int(cfg.target_loss_release_steps))
        & ~(state["front_complete"] & retain_rear_phase)
    )
    state["reference_valid"] &= ~lost

    reference_delta = state["reference_world_xy"] - root[:, :2]
    edge_x = cy * reference_delta[:, 0] + sy * reference_delta[:, 1]
    lower_z = state["reference_lower_z"]
    target_active = v11_descent_phase_active(
        active, state["reference_valid"], match_valid,
        state["front_complete"], state["rear_complete"], retain_rear_phase,
    )
    state["target_active"] = target_active
    state["target_edge_x"] = edge_x
    state["target_lower_z"] = lower_z

    front_yaw = wheel_positions_yaw(asset, env._v11_front_body_ids)
    rear_yaw = wheel_positions_yaw(asset, env._v11_rear_body_ids)
    front_world = asset.data.body_pos_w[:, env._v11_front_body_ids]
    rear_world = asset.data.body_pos_w[:, env._v11_rear_body_ids]
    front_forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_front_contact_ids]
    rear_forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v11_rear_contact_ids]
    x_min = edge_x[:, None] + float(cfg.wheel_radius) + float(cfg.front_margin)

    def lower_support(wheel_yaw, wheel_world, forces):
        bottom = wheel_world[..., 2] - float(cfg.wheel_radius)
        return (
            target_active[:, None] & (wheel_yaw[..., 0] >= x_min)
            & ((bottom - lower_z[:, None]).abs() <= float(cfg.upper_support_tolerance))
            & (forces[..., 2] > float(cfg.contact_force_threshold))
        )

    front_now = lower_support(front_yaw, front_world, front_forces)
    front_confirmation = v11_confirm_support(
        state["front_support_steps"], front_now,
        int(cfg.descent_front_support_confirmation_frames),
    )
    state["front_support_steps"] = front_confirmation["steps"]
    state["front_complete"] |= front_confirmation["confirmed"].all(dim=1) & target_active
    rear_now = lower_support(rear_yaw, rear_world, rear_forces) & state["front_complete"][:, None]
    rear_confirmation = v11_confirm_support(
        state["rear_support_steps"], rear_now,
        int(cfg.descent_rear_support_confirmation_frames),
    )
    state["rear_support_steps"] = rear_confirmation["steps"]
    state["rear_complete"] |= rear_confirmation["confirmed"].all(dim=1) & target_active
    following = target_active & state["front_complete"] & ~state["rear_complete"]
    state["rear_support_event"] = (
        rear_confirmation["confirmed"] & ~state["rear_support_credited"]
        & state["front_complete"][:, None] & target_active[:, None]
    )
    state["rear_support_credited"] |= state["rear_support_event"]
    state["rear_follow_gap_steps"] = torch.where(
        following, state["rear_follow_gap_steps"] + 1,
        torch.zeros_like(state["rear_follow_gap_steps"]),
    )

    follow_timeout = retain_rear_phase & (
        state["rear_follow_gap_steps"] > int(cfg.descent_rear_follow_timeout_frames)
    )
    state["reference_valid"] &= ~follow_timeout
    target_active &= ~follow_timeout
    state["target_active"] = target_active

    if not hasattr(env, "_v11_rear_wheel_joint_ids"):
        ids, names = asset.find_joints("(hl|hr)_wheel_joint", preserve_order=True)
        if names != ["hl_wheel_joint", "hr_wheel_joint"]:
            raise RuntimeError(f"V11 rear wheel joint mapping failed: {names}")
        env._v11_rear_wheel_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    rear_omega = asset.data.joint_vel[:, env._v11_rear_wheel_joint_ids].abs()
    rolling_speed = rear_omega * float(cfg.wheel_radius)
    forward_speed = asset.data.root_lin_vel_b[:, 0].abs()[:, None]
    rear_contact = rear_forces[..., 2] > float(cfg.contact_force_threshold)
    roll_error = ((forward_speed - rolling_speed).abs() / float(cfg.descent_rear_roll_speed_scale)).clamp(0.0, 1.0)
    roll_active = following[:, None] & rear_contact
    state["rear_roll_error"] = (
        (roll_error * roll_active.float()).sum(dim=1)
        / roll_active.float().sum(dim=1).clamp_min(1.0)
    )
    all_contact = torch.cat((front_forces[..., 2], rear_forces[..., 2]), dim=1) > float(cfg.contact_force_threshold)
    state["contact_count"] = all_contact.float().sum(dim=1)

    event = state["rear_complete"] & target_active
    state["completion_event"] = event
    state["last_completed_world_xy"] = torch.where(
        event[:, None], state["reference_world_xy"], state["last_completed_world_xy"],
    )
    state["sequence_started"] |= event
    state["reference_valid"] &= ~event
    state["target_slot"] = torch.where(event, torch.full_like(state["target_slot"], -1), state["target_slot"])
    return state


def v11_state(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"), asset_cfg=SceneEntityCfg("robot")):
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_v11_last_step", None) == step:
        return env._v11_state
    cfg = env.cfg.v11_perception
    asset: Articulation = env.scene[asset_cfg.name]
    points = _elevation_points_yaw(env, sensor_cfg, asset)
    riser_pool = extract_three_risers_points(
        points, corridor_half_width=cfg.corridor_half_width,
        h_min=cfg.min_step_height - cfg.height_valid_tolerance,
        d_end=cfg.passed_distance, min_separation=cfg.min_riser_separation,
        signed_edges=cfg.step_feature_mode == "signed_edges",
    )
    wq, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    pool_yaw = torch.atan2(2.0 * (wq * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    pool_c, pool_s = torch.cos(pool_yaw)[:, None], torch.sin(pool_yaw)[:, None]
    riser_pool_world_xy = torch.stack((
        asset.data.root_pos_w[:, 0:1] + pool_c * riser_pool["riser_x"] - pool_s * riser_pool["riser_y"],
        asset.data.root_pos_w[:, 1:2] + pool_s * riser_pool["riser_x"] + pool_c * riser_pool["riser_y"],
    ), dim=-1)
    risers = {key: value[:, :2] for key, value in riser_pool.items()}
    valid = risers["valid"] & (
        risers["riser_height"].abs() <= cfg.max_step_height + cfg.height_valid_tolerance
    )
    positive_valid = valid & (risers["riser_height"] > 0.0)
    negative_valid = valid & (risers["riser_height"] < 0.0)
    distance = risers["riser_distance"]
    command_forward = env.command_manager.get_command("base_velocity")[:, 0] > cfg.forward_command_threshold
    in_window = (distance >= cfg.anticipation_distance_min) & (distance <= cfg.anticipation_distance_max)

    # The detector keeps the current and next raw risers for reward geometry.
    # Once the current one has left the actor anticipation window, however,
    # promote the next riser to slot 1 instead of exposing an all-zero feature.
    # This is the continuous perceptive-stair handoff: the policy keeps its
    # V11 6-D terrain input while traversing a flight, rather than falling back
    # to the 57-D locomotion observation after each first tread.
    promote_next = ~valid[:, 0] | (distance[:, 0] < cfg.anticipation_distance_min)
    primary_valid = torch.where(promote_next, valid[:, 1], valid[:, 0])
    primary_height = torch.where(promote_next, risers["riser_height"][:, 1], risers["riser_height"][:, 0])
    primary_distance = torch.where(promote_next, distance[:, 1], distance[:, 0])
    primary_in_window = (primary_distance >= cfg.anticipation_distance_min) & (
        primary_distance <= cfg.anticipation_distance_max
    )
    b1 = primary_valid & primary_in_window & command_forward
    # Slot 2 remains the raw next riser only before promotion.  After promotion
    # no third riser is available in this two-riser representation.
    b2 = ~promote_next & valid[:, 1] & b1
    zero = torch.zeros(env.num_envs, device=env.device)
    detected_feature = torch.stack((
        b1.float(), torch.where(b1, primary_height, zero), torch.where(b1, primary_distance, zero),
        b2.float(), torch.where(b2, risers["riser_height"][:, 1], zero), torch.where(b2, distance[:, 1], zero),
    ), dim=1)

    if not hasattr(env, "_v11_feature_ring"):
        env._v11_feature_ring = torch.zeros(
            (env.num_envs, cfg.perception_delay_steps + 1, 6), device=env.device
        )
        env._v11_ring_index = 0
        env._v11_last_feature = torch.zeros((env.num_envs, 6), device=env.device)
        env._v11_last_episode_length = env.episode_length_buf.clone()
        env._v11_confirm_count = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.long)
        env._v11_invalid_count = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.long)
        env._v11_tracked_feature = torch.zeros((env.num_envs, 6), device=env.device)
    reset = env.episode_length_buf < env._v11_last_episode_length
    env._v11_feature_ring[reset] = 0.0
    env._v11_last_feature[reset] = 0.0
    env._v11_confirm_count[reset] = 0
    env._v11_invalid_count[reset] = 0
    env._v11_tracked_feature[reset] = 0.0
    env._v11_last_episode_length = env.episode_length_buf.clone()

    # A feature needs consecutive confirmation before it reaches the actor.
    # Once confirmed, only a bounded number of missing frames may retain the
    # last detected tuple.  Reward geometry still uses the live map and never
    # treats this actor-side hold as ground-truth geometry.
    detected_bits = torch.stack((b1, b2), dim=1)
    prior_bits = torch.stack((
        env._v11_tracked_feature[:, 0] > 0.5,
        env._v11_tracked_feature[:, 3] > 0.5,
    ), dim=1)
    env._v11_confirm_count = torch.where(
        detected_bits, env._v11_confirm_count + 1, torch.zeros_like(env._v11_confirm_count)
    )
    env._v11_invalid_count = torch.where(
        detected_bits, torch.zeros_like(env._v11_invalid_count), env._v11_invalid_count + 1
    )
    confirmed = env._v11_confirm_count >= int(cfg.confirmation_frames)
    held = prior_bits & (env._v11_invalid_count <= int(cfg.hysteresis_frames))
    actor_bits = (detected_bits & confirmed) | held
    for slot, offset in enumerate((0, 3)):
        update = detected_bits[:, slot]
        env._v11_tracked_feature[:, offset + 1:offset + 3] = torch.where(
            update[:, None],
            detected_feature[:, offset + 1:offset + 3],
            env._v11_tracked_feature[:, offset + 1:offset + 3],
        )
        env._v11_tracked_feature[:, offset] = actor_bits[:, slot].float()
        env._v11_tracked_feature[:, offset + 1:offset + 3] *= actor_bits[:, slot:slot + 1].float()
    raw = env._v11_tracked_feature.clone()
    hold = torch.rand(env.num_envs, device=env.device) < cfg.frame_hold_probability
    noisy = torch.where(hold[:, None], env._v11_last_feature, raw)
    noisy[:, (1, 4)] += torch.randn((env.num_envs, 2), device=env.device) * cfg.height_noise_std
    noisy[:, (2, 5)] += torch.randn((env.num_envs, 2), device=env.device) * cfg.distance_noise_std
    noisy[:, (0, 2, 3, 5)].clamp_(min=0.0)
    noisy[:, (1, 4)].clamp_(
        min=-float(cfg.max_step_height), max=float(cfg.max_step_height),
    )
    write = env._v11_ring_index
    env._v11_feature_ring[:, write] = noisy
    read = (write - cfg.perception_delay_steps) % env._v11_feature_ring.shape[1]
    actor = env._v11_feature_ring[:, read].clone()
    actor[:, 1:3] *= actor[:, 0:1]
    actor[:, 4:6] *= actor[:, 3:4]
    env._v11_ring_index = (write + 1) % env._v11_feature_ring.shape[1]
    env._v11_last_feature = actor
    # Reward-side geometry uses unpromoted live risers through the first tread;
    # actor-target promotion must not alter crossing/support or landing credit.
    # Each completion mode applies its own live-riser validity gate.  Passing
    # the command gate alone lets alternating V11 acquire raw slot 1 after the
    # just-credited slot 0 remains visible through the tread handoff.
    geometry_active = command_forward
    ascent_risers = {key: value for key, value in risers.items()}
    ascent_risers["valid"] = positive_valid
    descent_risers = {key: value for key, value in risers.items()}
    descent_risers["valid"] = negative_valid
    landing = _v11_landing_state(env, asset, ascent_risers, geometry_active)
    descent = _v11_descent_state(env, asset, descent_risers, geometry_active)
    env._v11_state = {
        "actor_feature": actor, "raw_feature": raw, "detected_feature": detected_feature,
        "risers": risers, "riser_pool": riser_pool, "riser_pool_world_xy": riser_pool_world_xy,
        "valid": valid, "actor_target_is_next": promote_next,
        "landing": landing, "descent": descent,
    }
    env._v11_last_step = step
    return env._v11_state


def v11_actor_perception(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return v11_state(env, sensor_cfg=sensor_cfg)["actor_feature"]


def v11_critic_pgtt(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return perceptive_elevation_clean(env, sensor_cfg)


def v11_critic_geometry(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return v11_state(env, sensor_cfg=sensor_cfg)["detected_feature"]


def v11_landing_event_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Pay the main front-wheel pulse only after the configured completion contract."""
    landing = v11_state(env, sensor_cfg=sensor_cfg)["landing"]
    if env.cfg.v11_perception.front_completion_mode == "alternating_risers":
        return landing["alternating_reward_event"].float()
    event = v11_front_reward_event(
        landing["landing_event"], landing["bilateral_landing_event"],
        bool(env.cfg.v11_perception.defer_first_front_rewards_until_bilateral),
    )
    return event.float()


def v11_first_completion_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Dt-normalized one-shot credit for either side's first completion."""
    event = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["first_completion_event"]
    return v11_dt_normalized_event(event, env.step_dt)


def v11_alternating_completion_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Dt-normalized one-shot credit for an opposite-side transition."""
    event = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["alternating_completion_event"]
    return v11_dt_normalized_event(event, env.step_dt)


def v11_same_side_repeat_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Dt-normalized one-shot indicator; the negative weight is configured outside."""
    event = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["same_side_repeat_event"]
    return v11_dt_normalized_event(event, env.step_dt)


def v11_simultaneous_completion_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Dt-normalized style penalty event for a simultaneous FL/FR completion."""
    event = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["simultaneous_completion_event"]
    return v11_dt_normalized_event(event, env.step_dt)


def v11_softalt_platform_success_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Dt-normalized safe platform event for the SoftAlt reward ledger."""
    event = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["platform_exit_event"]
    return v11_dt_normalized_event(event, env.step_dt)


def v11_platform_exit_success_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """One pulse after a safely crossed final riser is confirmed as a platform."""
    return v11_state(env, sensor_cfg=sensor_cfg)["landing"]["platform_exit_event"].float()


def v11_descent_step_complete_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """One pulse only after both front and both rear wheels support the lower tread."""
    event = v11_state(env, sensor_cfg=sensor_cfg)["descent"]["completion_event"]
    return event.float() / float(env.step_dt)


def v11_descent_rear_support_progress_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Give a small one-shot credit for each rear wheel that safely supports.

    This is deliberately subordinate to the full bilateral completion event:
    it only opens a reachable gradient during the front-complete/rear-follow
    phase and cannot prescribe a left/right descent order.
    """
    event = v11_state(env, sensor_cfg=sensor_cfg)["descent"]["rear_support_event"]
    return event.float().mean(dim=1) / float(env.step_dt)


def v11_descent_rear_follow_delay_penalty(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    state = v11_state(env, sensor_cfg=sensor_cfg)["descent"]
    grace = max(int(env.cfg.v11_perception.descent_rear_follow_grace_frames), 1)
    return ((state["rear_follow_gap_steps"].float() - grace).clamp_min(0.0) / grace).clamp_max(1.0)


def v11_descent_rear_roll_consistency_penalty(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return v11_state(env, sensor_cfg=sensor_cfg)["descent"]["rear_roll_error"]


def v11_phase_flat_orientation_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """Keep the normal posture cost, but permit natural descent pitch."""
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_xy = asset.data.projected_gravity_b[:, :2].square()
    normal = gravity_xy.sum(dim=1)
    roll_only = gravity_xy[:, 1]
    descent = v11_state(env)["descent"]["target_active"]
    return torch.where(descent, roll_only, normal)


def v11_phase_lin_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")):
    """Relax, rather than remove, necessary vertical motion on a visible descent."""
    asset: Articulation = env.scene[asset_cfg.name]
    cost = asset.data.root_lin_vel_b[:, 2].square()
    descent = v11_state(env)["descent"]["target_active"]
    scale = float(env.cfg.v11_perception.descent_vertical_velocity_penalty_scale)
    return cost * torch.where(descent, torch.full_like(cost, scale), torch.ones_like(cost))


def v11_pure_yaw_xy_drift_penalty(
    env, cmd_lin_threshold: float, cmd_ang_threshold: float, speed_scale: float,
    asset_cfg=SceneEntityCfg("robot"),
):
    """Bound translation during a pure-yaw command without prescribing contacts."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    gate = (command[:, :2].norm(dim=1) < float(cmd_lin_threshold)) & (
        command[:, 2].abs() > float(cmd_ang_threshold)
    )
    drift = (asset.data.root_lin_vel_b[:, :2].norm(dim=1) / float(speed_scale)).square().clamp_max(1.0)
    return drift * gate.float()


def v11_clearance_progress_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Pay escrowed clearance only after clearance -> crossing -> support."""
    settlement = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["clearance_settlement"]
    return settlement / float(env.step_dt)


def v11_fd_safe_front_landing_progress_reward(
    env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"),
):
    """Pay graded crossing-height credit only after safe upper-tread support."""
    settlement = v11_state(env, sensor_cfg=sensor_cfg)["landing"][
        "front_support_clearance_settlement"
    ]
    return settlement.sum(dim=1) / float(env.step_dt)


def v11_fd_single_front_landing_reward(
    env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"),
):
    """Reward each wheel once after the full 14 cm crossing-and-support gate."""
    event = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["front_candidate_event"]
    return event.float().sum(dim=1) / float(env.step_dt)


def v11_post_first_clearance_surplus_reward(
    env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"),
):
    """Release the bounded expected-side surplus only after a valid paired landing."""
    settlement = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["clearance_surplus_settlement"]
    return settlement / float(env.step_dt)


def v11_landing_center_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Weak, post-selection shaping inside the observed safety rectangle."""
    landing = v11_state(env, sensor_cfg=sensor_cfg)["landing"]
    responsible = landing["responsible"]
    chosen = responsible.clamp_min(0)[:, None]
    center = landing["scores"]["center"].gather(1, chosen).squeeze(1)
    if env.cfg.v11_perception.front_completion_mode == "alternating_risers":
        ready = landing["alternating_reward_event"]
    else:
        ready = (
        landing["candidate_ready"].all(dim=1)
        if bool(env.cfg.v11_perception.require_bilateral_front_completion)
        else responsible >= 0
        )
    return center * ready.float()


def v11_landing_edge_penalty(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Penalize a selected wheel violating first-riser or lateral bounds."""
    landing = v11_state(env, sensor_cfg=sensor_cfg)["landing"]
    responsible = landing["responsible"]
    chosen = responsible.clamp_min(0)[:, None]
    if env.cfg.v11_perception.front_completion_mode == "alternating_risers":
        violation = landing["scores"]["edge_violation"].gather(1, chosen).squeeze(1)
        active = landing["landing_event"]
    elif bool(env.cfg.v11_perception.require_bilateral_front_completion):
        violation = landing["scores"]["edge_violation"].amax(dim=1)
        active = responsible >= 0
    else:
        violation = landing["scores"]["edge_violation"].gather(1, chosen).squeeze(1)
        active = responsible >= 0
    return -violation * active.float()


def v11_front_riser_contact_penalty(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Bounded per-step cost while either front wheel presses a live riser face."""
    contact = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["vertical_contact_now"]
    return contact.any(dim=1).float()


def v11_confirmed_front_riser_contact(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Terminate after debounced front-wheel contact with a vertical riser face."""
    contact = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["vertical_contact"]
    if bool(getattr(env.cfg.v11_perception, "soft_alternation", False)):
        repeat = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["same_side_repeat_event"]
        return v11_softalt_termination(contact, repeat)
    violation = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["alternation_violation_confirmed"]
    return contact.any(dim=1) | violation


def _v11_straight_command_mask(env, tolerance: float) -> torch.Tensor:
    command = env.command_manager.get_command("base_velocity")
    return (command[:, 1].abs() <= tolerance) & (command[:, 2].abs() <= tolerance)


def v11_action_term_mirror(
    env,
    action_term_name: str,
    mirror_joint_names: list[list[str]],
    phase: str = "always_straight",
    straight_command_tolerance: float = 0.05,
    sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """V9-style bilateral action-magnitude cost with V11 crossing-phase gates.

    ``front_approach_or_complete`` disables the front-leg mirror only while a
    live stair target is being crossed and no platform is confirmed.  That is
    the one phase in which V11 deliberately permits FL and FR to differ so
    the policy can select a responsible first wheel.  Rear legs and wheels use
    ``always_straight`` and remain bilaterally regularized during traversal.
    """
    action_term = env.action_manager.get_term(action_term_name)
    cache_name = f"_v11_{action_term_name}_{phase}_mirror_indices"
    mirror_indices = getattr(env, cache_name, None)
    if mirror_indices is None:
        joint_names = list(action_term._joint_names)
        mirror_indices = []
        for first_pattern, second_pattern in mirror_joint_names:
            _, first_names = action_term._asset.find_joints(first_pattern, preserve_order=True)
            _, second_names = action_term._asset.find_joints(second_pattern, preserve_order=True)
            if len(first_names) != len(second_names):
                raise ValueError(
                    f"V11 mirror patterns {first_pattern!r}/{second_pattern!r} resolve to different sizes."
                )
            mirror_indices.append(
                ([joint_names.index(name) for name in first_names], [joint_names.index(name) for name in second_names])
            )
        setattr(env, cache_name, mirror_indices)

    costs = [
        torch.mean(
            torch.square(torch.abs(action_term.raw_actions[:, first_ids]) - torch.abs(action_term.raw_actions[:, second_ids])),
            dim=1,
        )
        for first_ids, second_ids in mirror_indices
    ]
    cost = torch.stack(costs, dim=1).mean(dim=1) if costs else torch.zeros(env.num_envs, device=env.device)
    gate = _v11_straight_command_mask(env, straight_command_tolerance)
    if phase == "front_approach_or_complete":
        landing = v11_state(env, sensor_cfg=sensor_cfg)["landing"]
        crossing = _v11_stair_forward_mask(env) & landing["target_active"] & ~landing["platform_confirmed"]
        gate &= ~crossing
    elif phase != "always_straight":
        raise ValueError(f"Unknown V11 mirror phase: {phase!r}")
    return cost * gate.float()


def v11_descent_phase_joint_mirror(
    env,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    descent_rear_follow_scale: float = 1.0,
) -> torch.Tensor:
    """Keep normal diagonal pose symmetry except during rear descent follow."""
    if not 0.0 <= float(descent_rear_follow_scale) <= 1.0:
        raise ValueError(
            "V11 descent mirror scale must be in [0, 1], "
            f"got {descent_rear_follow_scale}."
        )
    cost = joint_mirror(env, asset_cfg, mirror_joints)
    descent = v11_state(env)["descent"]
    rear_following = descent["target_active"] & descent["front_complete"]
    scale = torch.where(
        rear_following,
        torch.full_like(cost, float(descent_rear_follow_scale)),
        torch.ones_like(cost),
    )
    return cost * scale


def _v11_front_mobility_gate(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")) -> torch.Tensor:
    """Resolve the actor-visible, per-front-leg stair mobility phase."""
    state = v11_state(env, sensor_cfg=sensor_cfg)
    landing = state["landing"]
    alternating = env.cfg.v11_perception.front_completion_mode == "alternating_risers"
    actor_visible = (
        (state["actor_feature"][:, 0] > 0.5) | (state["actor_feature"][:, 3] > 0.5)
        if alternating else state["actor_feature"][:, 0] > 0.5
    )
    target_riser_x = (
        landing["target_riser_x"] if alternating else state["risers"]["riser_x"][:, 0]
    )
    gate = v11_front_step_mobility_gate(
        actor_visible,
        landing["target_active"],
        landing["wheel_yaw"][..., 0],
        target_riser_x,
        landing["candidate_ready"],
        landing["vertical_contact"],
        env.cfg.v11_perception.wheel_radius,
    )
    if alternating and not bool(getattr(env.cfg.v11_perception, "soft_alternation", False)):
        gate &= landing["eligible_sides"]
    return gate


def _v11_front_sagittal_action_indices(env) -> tuple[list[int], list[int]]:
    cache_name = "_v11_front_sagittal_action_indices"
    cached = getattr(env, cache_name, None)
    if cached is not None:
        return cached
    start = 0
    by_side = [[], []]
    expected = (("fl_hipy_joint", "fl_knee_joint"), ("fr_hipy_joint", "fr_knee_joint"))
    for term_name, term_dim in zip(
        env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True,
    ):
        term = env.action_manager.get_term(term_name)
        names = list(getattr(term, "_joint_names", ()))
        for side, side_names in enumerate(expected):
            by_side[side].extend(start + names.index(name) for name in side_names if name in names)
        start += int(term_dim)
    if any(len(indices) != 2 for indices in by_side):
        raise RuntimeError(f"V11 expected two front sagittal actions per side, resolved {by_side}.")
    cached = (by_side[0], by_side[1])
    setattr(env, cache_name, cached)
    return cached


def _v11_rear_sagittal_action_indices(env) -> tuple[list[int], list[int]]:
    """Resolve HL/HR hip-y and knee action slots without changing the ABI."""
    cache_name = "_v11_rear_sagittal_action_indices"
    cached = getattr(env, cache_name, None)
    if cached is not None:
        return cached
    start = 0
    by_side = [[], []]
    expected = (("hl_hipy_joint", "hl_knee_joint"), ("hr_hipy_joint", "hr_knee_joint"))
    for term_name, term_dim in zip(
        env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True,
    ):
        term = env.action_manager.get_term(term_name)
        names = list(getattr(term, "_joint_names", ()))
        for side, side_names in enumerate(expected):
            by_side[side].extend(start + names.index(name) for name in side_names if name in names)
        start += int(term_dim)
    if any(len(indices) != 2 for indices in by_side):
        raise RuntimeError(f"V11 expected two rear sagittal actions per side, resolved {by_side}.")
    cached = (by_side[0], by_side[1])
    setattr(env, cache_name, cached)
    return cached


def _v11_phase_weighted_action_cost(
    env,
    diff: torch.Tensor,
    hipy_active_scale: float,
    knee_active_scale: float,
    sensor_cfg: SceneEntityCfg,
    descent_rear_hipy_scale: float = 1.0,
    descent_rear_knee_scale: float = 1.0,
) -> torch.Tensor:
    component_scales = (
        float(hipy_active_scale), float(knee_active_scale),
        float(descent_rear_hipy_scale), float(descent_rear_knee_scale),
    )
    if any(not 0.0 < scale <= 1.0 for scale in component_scales):
        raise ValueError(f"V11 mobility component scales must be in (0, 1], got {component_scales}.")
    weights = torch.ones_like(diff)
    gate = _v11_front_mobility_gate(env, sensor_cfg=sensor_cfg)
    for side, indices in enumerate(_v11_front_sagittal_action_indices(env)):
        for action_index, component_scale in zip(indices, component_scales[:2], strict=True):
            weights[:, action_index] = torch.where(
                gate[:, side],
                torch.full_like(weights[:, action_index], component_scale),
                weights[:, action_index],
            )
    descent = v11_state(env, sensor_cfg=sensor_cfg)["descent"]
    rear_following = descent["target_active"] & descent["front_complete"]
    for indices in _v11_rear_sagittal_action_indices(env):
        for action_index, component_scale in zip(
            indices, (float(descent_rear_hipy_scale), float(descent_rear_knee_scale)), strict=True,
        ):
            weights[:, action_index] = torch.where(
                rear_following,
                torch.full_like(weights[:, action_index], component_scale),
                weights[:, action_index],
            )
    return torch.sum(diff * weights, dim=1) * get_gait_level_tensor(env)


def v11_phase_relaxed_action_rate_l2(
    env,
    hipy_active_scale: float,
    knee_active_scale: float,
    descent_rear_hipy_scale: float = 1.0,
    descent_rear_knee_scale: float = 1.0,
    sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """Keep action-rate cost while independently scaling active hip-y and knee."""
    diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
    return _v11_phase_weighted_action_cost(
        env, diff, hipy_active_scale, knee_active_scale, sensor_cfg,
        descent_rear_hipy_scale, descent_rear_knee_scale,
    )


def v11_phase_relaxed_action_smooth_l2(
    env,
    hipy_active_scale: float,
    knee_active_scale: float,
    descent_rear_hipy_scale: float = 1.0,
    descent_rear_knee_scale: float = 1.0,
    sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """Keep second-order smoothing with bounded front-sagittal stair relief."""
    cache_name = "_v11_action_smooth_prev_prev_action"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, torch.zeros_like(env.action_manager.action))
    prev_prev = getattr(env, cache_name)
    diff = torch.square(env.action_manager.action + prev_prev - 2.0 * env.action_manager.prev_action)
    diff *= env.action_manager.prev_action != 0
    diff *= prev_prev != 0
    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return _v11_phase_weighted_action_cost(
        env, diff, hipy_active_scale, knee_active_scale, sensor_cfg,
        descent_rear_hipy_scale, descent_rear_knee_scale,
    )


def _v11_front_joint_deviations(
    env, front_joint_names: tuple[str, str],
) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    names = list(asset.joint_names)
    try:
        joint_ids = [names.index(name) for name in front_joint_names]
    except ValueError as exc:
        raise RuntimeError(f"V11 could not resolve all configured front joints {front_joint_names}.") from exc
    return (asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]).abs()


def v11_front_hipy_participation_settlement_reward(
    env,
    front_hipy_joint_names: tuple[str, str],
    front_knee_joint_names: tuple[str, str],
    target_share: float,
    share_half_width: float,
    activation_scale: float,
    sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """Settle the latest useful hip-y/knee ratio once crossing and support succeed."""
    landing = v11_state(env, sensor_cfg=sensor_cfg)["landing"]
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_v11_hipy_escrow_last_step", None) != step:
        hipy = _v11_front_joint_deviations(env, front_hipy_joint_names)
        knee = _v11_front_joint_deviations(env, front_knee_joint_names)
        gate = _v11_front_mobility_gate(env, sensor_cfg=sensor_cfg)
        per_leg_score = v11_bounded_relative_hipy_participation_scores(
            hipy,
            knee,
            gate,
            target_share,
            share_half_width,
            activation_scale,
        )
        # Store the latest eligible score, not the maximum: a brief good pose
        # followed by a knee-only landing must not retain full credit.
        landing["hipy_participation_escrow"] = v11_update_latest_participation_escrow(
            landing["hipy_participation_escrow"], per_leg_score, gate,
        )
        alternating = env.cfg.v11_perception.front_completion_mode == "alternating_risers"
        settlement_event = (
            landing["alternating_reward_event"]
            if alternating else v11_front_reward_event(
                landing["landing_event"], landing["bilateral_landing_event"],
                bool(env.cfg.v11_perception.defer_first_front_rewards_until_bilateral),
            )
        )
        soft_alternation = bool(getattr(env.cfg.v11_perception, "soft_alternation", False))
        if alternating:
            settlement_event = (
                landing["landing_event"] if soft_alternation
                else landing["alternating_reward_event"]
            )
        if alternating:
            selected = landing["hipy_participation_escrow"].gather(
                1, landing["responsible"].clamp_min(0)[:, None],
            ).squeeze(1)
            if soft_alternation:
                landing["hipy_participation_settlement"] = selected * settlement_event.float()
            else:
                landing["pending_first_hipy"] = torch.where(
                    landing["first_completion_event"], selected, landing["pending_first_hipy"],
                )
                second_event = settlement_event & (landing["completed_riser_count"] == 2)
                landing["hipy_participation_settlement"] = torch.where(
                    second_event, 0.5 * (landing["pending_first_hipy"] + selected),
                    selected * settlement_event.float(),
                )
                landing["pending_first_hipy"] = torch.where(
                    second_event, torch.zeros_like(landing["pending_first_hipy"]),
                    landing["pending_first_hipy"],
                )
        elif bool(env.cfg.v11_perception.defer_first_front_rewards_until_bilateral):
            landing["hipy_participation_settlement"] = v11_settle_bilateral_escrow(
                landing["hipy_participation_escrow"], settlement_event, average=True,
            )
        else:
            landing["hipy_participation_settlement"] = v11_settle_clearance_escrow(
                landing["hipy_participation_escrow"], landing["responsible"], settlement_event,
            )
        env._v11_hipy_escrow_last_step = step
    # RewardManager multiplies by step_dt; cancel it so this remains a one-shot
    # per-riser settlement, like the existing clearance escrow.
    return landing["hipy_participation_settlement"] / float(env.step_dt)


def v11_front_knee_excess_penalty(
    env,
    front_knee_joint_names: tuple[str, str],
    free_band: float,
    excess_scale: float,
    sensor_cfg=SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """Bound excessive active-side knee folding without high-terrain rescaling."""
    result = v11_bounded_knee_excess(
        _v11_front_joint_deviations(env, front_knee_joint_names),
        _v11_front_mobility_gate(env, sensor_cfg=sensor_cfg),
        free_band,
        excess_scale,
    )
    return result["cost"]


def v11_phase_relaxed_front_joint_pos_penalty(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    front_joint_names: tuple[str, str],
    active_scale: float,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    ang_cmd_threshold: float = 0.1,
    y_cmd_threshold: float = 0.1,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
    sensor_cfg: SceneEntityCfg | None = None,
    terrain_height_threshold: float = 0.05,
    high_terrain_penalty_scale: float = 0.5,
    mobility_sensor_cfg: SceneEntityCfg = SceneEntityCfg("v10_pgtt_scanner"),
    descent_rear_joint_names: tuple[str, str] | None = None,
    descent_rear_active_scale: float = 1.0,
) -> torch.Tensor:
    """Preserve nominal pose cost, with phase-local front/rear mobility relief."""
    if not 0.0 < float(active_scale) <= 1.0:
        raise ValueError(f"V11 posture active_scale must be in (0, 1], got {active_scale}.")
    if not 0.0 < float(descent_rear_active_scale) <= 1.0:
        raise ValueError(
            "V11 descent rear posture scale must be in (0, 1], "
            f"got {descent_rear_active_scale}."
        )
    asset: Articulation = env.scene[asset_cfg.name]
    resolved_joint_ids = asset_cfg.joint_ids
    if resolved_joint_ids is None:
        joint_ids = list(range(asset.data.joint_pos.shape[1]))
    elif isinstance(resolved_joint_ids, slice):
        joint_ids = list(range(asset.data.joint_pos.shape[1]))[resolved_joint_ids]
    else:
        joint_ids = list(resolved_joint_ids)
    selected_names = [asset.joint_names[index] for index in joint_ids]
    try:
        front_local = [selected_names.index(name) for name in front_joint_names]
    except ValueError as exc:
        raise RuntimeError(
            f"V11 front posture joints {front_joint_names} are not all in {selected_names}."
        ) from exc

    delta = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    scales = torch.ones_like(delta)
    mobility_gate = _v11_front_mobility_gate(env, sensor_cfg=mobility_sensor_cfg)
    for side, local_index in enumerate(front_local):
        scales[:, local_index] = torch.where(
            mobility_gate[:, side],
            torch.full_like(scales[:, local_index], float(active_scale)),
                scales[:, local_index],
            )
    if descent_rear_joint_names is not None:
        try:
            rear_local = [selected_names.index(name) for name in descent_rear_joint_names]
        except ValueError as exc:
            raise RuntimeError(
                f"V11 descent rear posture joints {descent_rear_joint_names} are not all in {selected_names}."
            ) from exc
        descent = v11_state(env, sensor_cfg=mobility_sensor_cfg)["descent"]
        rear_following = descent["target_active"] & descent["front_complete"]
        for local_index in rear_local:
            scales[:, local_index] = torch.where(
                rear_following,
                torch.full_like(scales[:, local_index], float(descent_rear_active_scale)),
                scales[:, local_index],
            )
    running_cost = torch.linalg.norm(delta * scales, dim=1)
    command = env.command_manager.get_command(command_name)
    command_norm = torch.linalg.norm(command, dim=1)
    body_speed = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    cost = torch.where(
        (command_norm > command_threshold) | (body_speed > velocity_threshold),
        running_cost,
        float(stand_still_scale) * running_cost,
    )
    turn = (command[:, 2].abs() > ang_cmd_threshold) & (
        torch.linalg.norm(command[:, :2], dim=1) < xy_norm_max
    )
    side = (command[:, 1].abs() > y_cmd_threshold) & (
        torch.linalg.norm(torch.stack((command[:, 0], command[:, 2]), dim=1), dim=1) < xz_norm_max
    )
    cost *= (~(turn | side)).float()

    if sensor_cfg is not None:
        height_sensor = env.scene[sensor_cfg.name]
        ray_hits = height_sensor.data.ray_hits_w[..., 2]
        valid_hits = torch.isfinite(ray_hits) & (ray_hits.abs() <= 1.0e6)
        valid_count = valid_hits.sum(dim=1)
        terrain_height = torch.where(valid_hits, ray_hits, torch.zeros_like(ray_hits)).sum(dim=1)
        terrain_height /= valid_count.clamp_min(1)
        terrain_height -= env.scene.env_origins[:, 2]
        high_terrain = (terrain_height > terrain_height_threshold) & (valid_count > 0)
        cost *= torch.where(
            high_terrain,
            torch.full_like(cost, float(high_terrain_penalty_scale)),
            torch.ones_like(cost),
        )
    return cost


def v11_front_mobility_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose whether phase relief changes hip-y/knee participation as intended."""
    del env_ids
    gate = _v11_front_mobility_gate(env)
    asset: Articulation = env.scene["robot"]
    names = list(asset.joint_names)
    hipy_ids = [names.index("fl_hipy_joint"), names.index("fr_hipy_joint")]
    knee_ids = [names.index("fl_knee_joint"), names.index("fr_knee_joint")]
    hipy = (asset.data.joint_pos[:, hipy_ids] - asset.data.default_joint_pos[:, hipy_ids]).abs()
    knee = (asset.data.joint_pos[:, knee_ids] - asset.data.default_joint_pos[:, knee_ids]).abs()
    count = gate.float().sum().clamp_min(1.0)
    if key == "gate_rate":
        return gate.float().mean()
    if key == "hipy_deviation":
        return (hipy * gate.float()).sum() / count
    if key == "knee_deviation":
        return (knee * gate.float()).sum() / count
    if key == "hipy_share":
        return ((hipy / (hipy + knee).clamp_min(1.0e-6)) * gate.float()).sum() / count
    if key in ("knee_excess", "knee_excess_rate"):
        term_cfg = env.reward_manager.get_term_cfg("front_knee_excess")
        bounded = v11_bounded_knee_excess(
            knee,
            gate,
            term_cfg.params["free_band"],
            term_cfg.params["excess_scale"],
        )
        return bounded["cost"].mean() if key == "knee_excess" else bounded["rate"].mean()
    if key == "hipy_escrow":
        return v11_state(env)["landing"]["hipy_participation_escrow"].mean()
    if key in ("upperleg_torso_angle", "upperleg_parallel_cosine"):
        hip_pos = asset.data.body_pos_w[:, env._v11_front_hipy_body_ids]
        knee_pos = asset.data.body_pos_w[:, env._v11_front_knee_body_ids]
        upper_leg = knee_pos - hip_pos
        upper_leg /= torch.linalg.norm(upper_leg, dim=-1, keepdim=True).clamp_min(1.0e-6)
        w, x, y, z = asset.data.root_quat_w.unbind(dim=-1)
        base_forward = torch.stack((
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ), dim=1)
        parallel = (upper_leg * base_forward[:, None, :]).sum(dim=-1).abs().clamp(0.0, 1.0)
        value = torch.acos(parallel) if key == "upperleg_torso_angle" else parallel
        return (value * gate.float()).sum() / count
    raise ValueError(f"Unknown V11 front mobility metric: {key!r}")


def v11_front_collision_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose current, newly confirmed and latched front-riser collisions."""
    del env_ids
    landing = v11_state(env)["landing"]
    if key == "contact_rate":
        return landing["vertical_contact_now"].any(dim=1).float().mean()
    if key == "event_rate":
        return landing["front_collision_event"].float().mean()
    if key == "latched_rate":
        return landing["vertical_contact"].any(dim=1).float().mean()
    raise ValueError(f"Unknown V11 front collision metric: {key!r}")


def v11_responsibility_metric(env, env_ids, side: str = "fl_rate") -> torch.Tensor:
    """Return global FL/FR responsibility statistics for TensorBoard.

    CurriculumManager invokes all terms as ``func(env, env_ids, **params)``.
    ``env_ids`` is deliberately unused because these counters are global for the
    complete training run rather than reset-local statistics.
    """
    state = v11_state(env)["landing"]
    fl = state["responsibility_fl_count"].float()
    fr = state["responsibility_fr_count"].float()
    total = (fl + fr).clamp_min(1.0)
    if side == "fl_rate":
        return fl / total
    if side == "fr_rate":
        return fr / total
    if side == "tie_rate":
        return state["responsibility_tie_count"].float() / total
    raise ValueError(f"Unknown V11 responsibility metric: {side!r}")


def v11_alternation_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose the deploy-ABI-neutral alternating-riser lifecycle."""
    del env_ids
    state = v11_state(env)["landing"]
    expected = state["expected_side"]
    if key == "expected_fl_rate":
        return (expected == 0).float().mean()
    if key == "expected_fr_rate":
        return (expected == 1).float().mean()
    if key == "completed_risers_mean":
        return state["completed_riser_count"].float().mean()
    if key == "compliance_rate":
        first = state["responsibility_fl_count"] + state["responsibility_fr_count"]
        alternating = (state["alternating_completion_count"] - first).clamp_min(0)
        denominator = (alternating + state["alternation_violation_count"]).clamp_min(1)
        return alternating.float() / denominator.float()
    if key == "violation_rate":
        denominator = state["alternating_completion_count"].clamp_min(1)
        return state["alternation_violation_count"].float() / denominator.float()
    if key == "crossing_clearance_margin":
        count = state["crossing_clearance_margin_count"].clamp_min(1)
        return state["crossing_clearance_margin_sum"] / count.float()
    if key == "post_first_collision_rate":
        denominator = state["alternating_completion_count"].clamp_min(1)
        return state["post_first_collision_count"].float() / denominator.float()
    if key == "anchor_valid_rate":
        return state["anchor_support_valid"].float().mean()
    if key == "anchor_support_steps":
        return state["anchor_support_steps"].float().mean()
    if key == "clearance_surplus":
        return state["clearance_surplus_settlement"].mean()
    if key == "confirmed_violation_rate":
        return state["alternation_violation_confirmed"].float().mean()
    raise ValueError(f"Unknown V11 alternation metric: {key!r}")


def v11_softalt_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose SoftAlt counters and rates without fabricating zero compliance."""
    del env_ids
    state = v11_state(env)["landing"]
    scalar_keys = {
        "first_completion_count": "first_completion_count",
        "FL_completion_count": "fl_completion_count",
        "FR_completion_count": "fr_completion_count",
        "same_side_repeat_count": "same_side_repeat_count",
        "simultaneous_completion_count": "simultaneous_completion_count",
        "FL_to_FR_count": "fl_to_fr_count",
        "FR_to_FL_count": "fr_to_fl_count",
        "FL_to_FL_repeat_count": "fl_to_fl_repeat_count",
        "FR_to_FR_repeat_count": "fr_to_fr_repeat_count",
    }
    if key in scalar_keys:
        return state[scalar_keys[key]].float()
    alternating = state["fl_to_fr_count"] + state["fr_to_fl_count"]
    transitions = alternating + state["same_side_repeat_count"] + state["simultaneous_transition_count"]
    if key == "alternating_completion_count":
        return alternating.float()
    if key == "alternation_compliance_rate":
        return torch.where(
            transitions > 0,
            alternating.float() / transitions.clamp_min(1).float(),
            torch.full((), float("nan"), device=env.device),
        )
    if key == "simultaneous_completion_rate":
        return torch.where(
            transitions > 0,
            state["simultaneous_transition_count"].float() / transitions.clamp_min(1).float(),
            torch.full((), float("nan"), device=env.device),
        )
    if key == "front_riser_collision_rate":
        episodes = state["stair_episode_count"]
        return torch.where(
            episodes > 0,
            state["front_collision_episode_count"].float() / episodes.clamp_min(1).float(),
            torch.full((), float("nan"), device=env.device),
        )
    if key == "platform_success_rate":
        episodes = state["ascending_episode_count"]
        return torch.where(
            episodes > 0,
            state["platform_success_count"].float() / episodes.clamp_min(1).float(),
            torch.full((), float("nan"), device=env.device),
        )
    if key == "stair_forward_velocity_mean":
        mask = _v11_stair_forward_mask(env)
        if not mask.any():
            return torch.full((), float("nan"), device=env.device)
        asset: Articulation = env.scene["robot"]
        forward, _, _ = _v11_path_state(env, asset)
        path_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * forward, dim=1)
        return path_speed[mask].mean()
    raise ValueError(f"Unknown V11 SoftAlt metric: {key!r}")


def v11_event_reachability_metric(env, env_ids, key: str) -> torch.Tensor:
    """Report each physical gate so a zero final event is diagnosable."""
    del env_ids
    state = v11_state(env)["landing"]
    if key == "target_active":
        return state["target_active"].float().mean()
    if key == "eligible_wheel":
        eligible = (
            torch.ones_like(state["candidate_ready"], dtype=torch.bool)
            if bool(getattr(env.cfg.v11_perception, "soft_alternation", False))
            else state.get(
                "eligible_sides",
                torch.ones_like(state["candidate_ready"], dtype=torch.bool),
            )
        )
        return (state["target_active"][:, None] & eligible).float().mean()
    if key == "clearance":
        return state["clearance"].any(dim=1).float().mean()
    if key == "crossing":
        return state["crossing"].any(dim=1).float().mean()
    if key == "support":
        return state["support"].any(dim=1).float().mean()
    if key == "physical_completion":
        return state["landing_event"].float().mean()
    if key == "rewarded_alternation":
        return state["alternating_reward_event"].float().mean()
    raise ValueError(f"Unknown V11 reachability metric: {key!r}")


def v11_descent_metric(env, env_ids, key: str) -> torch.Tensor:
    """Expose the bilateral descent lifecycle without changing actor observations."""
    del env_ids
    state = v11_state(env)["descent"]
    if key in {"target_active", "front_complete", "rear_complete", "completion_event"}:
        return state[key].float().mean()
    if key == "rear_support_event_rate":
        return state["rear_support_event"].float().mean()
    if key == "rear_follow_gap_steps":
        return state["rear_follow_gap_steps"].float().mean()
    if key == "rear_roll_error":
        return state["rear_roll_error"].mean()
    if key == "three_plus_contact_rate":
        return (state["contact_count"] >= 3.0).float().mean()
    raise ValueError(f"Unknown V11 descent metric: {key!r}")


def v11_fd_clearance_metric(env, env_ids, key: str) -> torch.Tensor:
    """Global per-wheel crossing diagnostics for the high-clearance task."""
    del env_ids
    state = v11_state(env)["landing"]
    if key == "safe_support_event_rate":
        return state["front_support_event"].float().mean()
    if key == "hard_candidate_event_rate":
        return state["front_candidate_event"].float().mean()
    count = state["front_crossing_count"].float().clamp_min(1.0)
    if key == "crossing_center_mean":
        return state["front_crossing_center_sum"] / count
    if key == "crossing_margin_mean":
        return state["front_crossing_margin_sum"] / count
    if key == "crossing_min_margin":
        value = state["front_crossing_min_margin"]
        return torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    if key == "crossing_p10_margin":
        sample = (
            state["crossing_wheel_center_above_tread"]
            - float(env.cfg.v11_perception.wheel_radius)
            - float(env.cfg.v11_perception.clearance_margin)
        )
        valid = state["crossing_sample_valid"]
        if valid.any():
            return torch.quantile(sample[valid], 0.10)
        return torch.zeros((), device=env.device)
    if key == "low_crossing_rate":
        return state["low_clearance_crossing_count"].float() / count
    raise ValueError(f"Unknown V11-FD clearance metric: {key!r}")


def v11_turn_metric(env, env_ids, key: str, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """Report pure-yaw tracking and drift without prescribing a contact gait."""
    del env_ids
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    gate = (command[:, :2].norm(dim=1) < 0.1) & (command[:, 2].abs() > 0.1)
    denominator = gate.float().sum().clamp_min(1.0)
    if key == "yaw_error":
        error = (command[:, 2] - asset.data.root_ang_vel_b[:, 2]).abs()
        return (error * gate.float()).sum() / denominator
    if key == "xy_speed":
        speed = asset.data.root_lin_vel_b[:, :2].norm(dim=1)
        return (speed * gate.float()).sum() / denominator
    if key == "action_delta":
        delta = (env.action_manager.action - env.action_manager.prev_action).abs().mean(dim=1)
        return (delta * gate.float()).sum() / denominator
    raise ValueError(f"Unknown V11 turn metric: {key!r}")


def _v11_stair_forward_mask(env) -> torch.Tensor:
    """Return the V11 stair-only forward-command gate used by all dense terms."""
    cfg = env.cfg.v11_perception
    stair_types = tuple(cfg.stair_terrain_type_ids)
    if not stair_types:
        raise RuntimeError("V11 stair reward has no resolved stair terrain type IDs.")
    terrain_types = env.scene.terrain.terrain_types
    stair_type_tensor = torch.tensor(stair_types, device=env.device, dtype=terrain_types.dtype)
    return (terrain_types[:, None] == stair_type_tensor[None, :]).any(dim=1) & (
        env.command_manager.get_command("base_velocity")[:, 0] > cfg.forward_command_threshold
    )


def _v11_ascending_stair_forward_mask(env) -> torch.Tensor:
    """Return only ascending V11 stairs under a positive forward command."""
    cfg = env.cfg.v11_perception
    terrain_types = env.scene.terrain.terrain_types
    ascending = torch.tensor(
        tuple(cfg.ascending_stair_terrain_type_ids), device=env.device, dtype=terrain_types.dtype,
    )
    return (terrain_types[:, None] == ascending[None, :]).any(dim=1) & (
        env.command_manager.get_command("base_velocity")[:, 0] > cfg.forward_command_threshold
    )


def v11_stair_masked_track_lin_vel_xy_exp(
    env, std: float, command_name: str, stair_scale: float, ascending_only: bool = False,
) -> torch.Tensor:
    """Preserve flat tracking and reduce only the V11 stair contribution."""
    if not 0.0 < float(stair_scale) <= 1.0:
        raise ValueError(f"V11 stair tracking scale must be in (0, 1], got {stair_scale}.")
    from .rewards import track_lin_vel_xy_exp

    reward = track_lin_vel_xy_exp(env, std=std, command_name=command_name)
    scale_mask = _v11_ascending_stair_forward_mask(env) if ascending_only else _v11_stair_forward_mask(env)
    scale = torch.where(
        scale_mask,
        torch.full_like(reward, float(stair_scale)),
        torch.ones_like(reward),
    )
    return reward * scale


def _v11_route_engagement_quality(
    env,
    target_speed_fraction: float,
    heading_scale: float,
    lateral_tolerance: float,
    lateral_scale: float,
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute route engagement from world motion and the immutable command route."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    forward, left, origin = _v11_path_state(env, asset)
    path_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * forward, dim=1)
    lateral_error = torch.abs(torch.sum((asset.data.root_pos_w[:, :2] - origin) * left, dim=1))
    command_term = env.command_manager.get_term("base_velocity")
    w, x, y, z = asset.data.root_quat_w.unbind(dim=-1)
    heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    heading_error = torch.atan2(
        torch.sin(heading - command_term.v11_path_heading_w),
        torch.cos(heading - command_term.v11_path_heading_w),
    )
    from .v13_alternation import v13_route_engagement_quality

    return v13_route_engagement_quality(
        path_speed, command_x, heading_error, lateral_error,
        target_speed_fraction=target_speed_fraction,
        heading_scale=heading_scale,
        lateral_tolerance=lateral_tolerance,
        lateral_scale=lateral_scale,
    )


def v11_ascent_route_coupled_track_lin_vel_xy_exp(
    env, std: float, command_name: str, stair_scale: float,
    target_speed_fraction: float, heading_scale: float,
    lateral_tolerance: float, lateral_scale: float,
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pay ascent linear tracking only while advancing along the stair route."""
    if not 0.0 < float(stair_scale) <= 1.0:
        raise ValueError("V11 ascent linear tracking scale must lie in (0, 1].")
    from .rewards import track_lin_vel_xy_exp

    native = track_lin_vel_xy_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    mask = _v11_ascending_stair_forward_mask(env)
    quality = _v11_route_engagement_quality(
        env, target_speed_fraction, heading_scale, lateral_tolerance, lateral_scale, asset_cfg,
    )
    ascent = native * float(stair_scale) * quality
    return torch.where(mask, ascent, native)


def v11_ascent_route_coupled_track_ang_vel_z_exp(
    env, std: float, command_name: str, stair_scale: float,
    target_speed_fraction: float, heading_scale: float,
    lateral_tolerance: float, lateral_scale: float,
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Prevent stationary or off-route ascent from collecting yaw tracking reward."""
    if not 0.0 < float(stair_scale) <= 1.0:
        raise ValueError("V11 ascent yaw tracking scale must lie in (0, 1].")
    from .rewards import track_ang_vel_z_exp

    native = track_ang_vel_z_exp(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    mask = _v11_ascending_stair_forward_mask(env)
    quality = _v11_route_engagement_quality(
        env, target_speed_fraction, heading_scale, lateral_tolerance, lateral_scale, asset_cfg,
    )
    ascent = native * float(stair_scale) * quality
    return torch.where(mask, ascent, native)


def v11_ascent_route_engagement_metric(env, env_ids) -> torch.Tensor:
    """Report useful ascent-route engagement without changing curriculum."""
    del env_ids
    cfg = env.cfg.v13_critic12d_balance
    value = _v11_route_engagement_quality(
        env, cfg.route_target_speed_fraction, cfg.route_heading_scale,
        cfg.route_lateral_tolerance, cfg.route_lateral_scale,
    )
    mask = _v11_ascending_stair_forward_mask(env)
    return (value * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def v11_softalt_world_forward_speed_deficit(
    env, min_speed_fraction: float, speed_scale: float, ascending_scale: float,
    asset_cfg=SceneEntityCfg("robot"),
):
    """Keep original V11 descent cost while reducing authority on ascending stairs."""
    base = v11_world_forward_speed_deficit(env, min_speed_fraction, speed_scale, asset_cfg)
    scale = torch.where(
        _v11_ascending_stair_forward_mask(env),
        torch.full_like(base, float(ascending_scale)), torch.ones_like(base),
    )
    return base * scale


def v11_softalt_world_forward_progress_reward(
    env, target_speed_fraction: float, ascending_scale: float,
    asset_cfg=SceneEntityCfg("robot"),
):
    """Keep original V11 descent reward while reducing authority on ascending stairs."""
    base = v11_world_forward_progress_reward(env, target_speed_fraction, asset_cfg)
    scale = torch.where(
        _v11_ascending_stair_forward_mask(env),
        torch.full_like(base, float(ascending_scale)), torch.ones_like(base),
    )
    return base * scale


def _v11_path_state(env, asset: Articulation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get immutable world-path vectors captured by the V11 command term."""
    command_term = env.command_manager.get_term("base_velocity")
    if not hasattr(command_term, "v11_path_heading_w") or not hasattr(command_term, "v11_path_origin_w"):
        raise RuntimeError("V11 path rewards require UniformThresholdVelocityCommand route state.")
    heading = command_term.v11_path_heading_w
    forward = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
    left = torch.stack((-torch.sin(heading), torch.cos(heading)), dim=1)
    return forward, left, command_term.v11_path_origin_w


def v11_world_forward_speed_deficit(
    env, min_speed_fraction: float, speed_scale: float, asset_cfg=SceneEntityCfg("robot")
):
    """Bounded cost for failing to advance along the sampled world stair route."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    forward, _, _ = _v11_path_state(env, asset)
    path_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * forward, dim=1)
    deficit = (command_x * min_speed_fraction - path_speed).clamp_min(0.0) / speed_scale
    mask = _v11_stair_forward_mask(env)
    landing = v11_state(env)["landing"]
    confirming_anchor = (
        landing["target_active"] & landing["sequence_started"]
        & (landing["anchor_support_steps"] < int(env.cfg.v11_perception.front_support_confirmation_frames))
    )
    return -(1.0 - torch.exp(-deficit.square())) * (mask & ~confirming_anchor).float()


def v11_world_forward_progress_reward(
    env, target_speed_fraction: float, asset_cfg=SceneEntityCfg("robot")
):
    """Reward bounded world-route progress; zero outside V11 stair traversal."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    forward, _, _ = _v11_path_state(env, asset)
    path_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * forward, dim=1)
    target_speed = (command_x * target_speed_fraction).clamp_min(1.0e-3)
    mask = _v11_stair_forward_mask(env)
    landing = v11_state(env)["landing"]
    anchor_required = landing["target_active"] & landing["sequence_started"]
    contract_valid = ~anchor_required | landing["anchor_support_valid"]
    return (path_speed / target_speed).clamp(0.0, 1.0) * (mask & contract_valid).float()


def v11_path_lateral_deviation_penalty(
    env, tolerance: float, deviation_scale: float, asset_cfg=SceneEntityCfg("robot")
):
    """Bounded accumulated lateral-route penalty, not merely instantaneous vy."""
    asset: Articulation = env.scene[asset_cfg.name]
    _, left, origin = _v11_path_state(env, asset)
    lateral_error = torch.abs(torch.sum((asset.data.root_pos_w[:, :2] - origin) * left, dim=1))
    violation = (lateral_error - tolerance).clamp_min(0.0) / deviation_scale
    return -(1.0 - torch.exp(-violation.square())) * _v11_stair_forward_mask(env).float()


def v11_path_heading_error_penalty(env, heading_scale: float, asset_cfg=SceneEntityCfg("robot")):
    """Bounded accumulated yaw error from the sampled world stair route."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term("base_velocity")
    if not hasattr(command_term, "v11_path_heading_w"):
        raise RuntimeError("V11 path rewards require UniformThresholdVelocityCommand route state.")
    w, x, y, z = asset.data.root_quat_w.unbind(dim=-1)
    heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    error = torch.atan2(
        torch.sin(heading - command_term.v11_path_heading_w),
        torch.cos(heading - command_term.v11_path_heading_w),
    ) / heading_scale
    return -(1.0 - torch.exp(-error.square())) * _v11_stair_forward_mask(env).float()


def v11_non_wheel_contact_force_penalty(
    env, sensor_cfg: SceneEntityCfg, force_threshold: float, force_scale: float, max_penalty: float
):
    """Penalize forceful shank/leg contacts during a forward stair traverse."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1)
    excess = (force.amax(dim=1) - force_threshold).clamp_min(0.0) / force_scale
    return -torch.square(excess).sum(dim=1).clamp_max(max_penalty) * _v11_stair_forward_mask(env).float()


def v11_non_wheel_scrape_penalty(
    env, sensor_cfg: SceneEntityCfg, force_threshold: float, speed_threshold: float,
    speed_scale: float, max_penalty: float, asset_cfg=SceneEntityCfg("robot"),
):
    """Penalize non-wheel links sliding while contacting a stair."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1)
    contact = force.amax(dim=1) > force_threshold
    speed_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    scrape = ((speed_xy - speed_threshold).clamp_min(0.0) / speed_scale).square()
    return -(scrape * contact.float()).sum(dim=1).clamp_max(max_penalty) * _v11_stair_forward_mask(env).float()


def v11_wheel_impact_penalty(
    env, sensor_cfg: SceneEntityCfg, speed_threshold: float, max_penalty: float,
    asset_cfg=SceneEntityCfg("robot"),
):
    """Penalize excessive downward wheel speed at initial stair contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids].float()
    downward = (-asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2] - speed_threshold).clamp_min(0.0)
    return -(first_contact * downward.square()).sum(dim=1).clamp_max(max_penalty) * _v11_stair_forward_mask(env).float()


def v11_stair_straightness_penalty(env, asset_cfg=SceneEntityCfg("robot")):
    """Small, stair-only cost for lateral drift and yaw rate under forward command."""
    asset: Articulation = env.scene[asset_cfg.name]
    drift_and_yaw = (asset.data.root_lin_vel_b[:, 1].square() + asset.data.root_ang_vel_b[:, 2].square()).clamp_max(1.0)
    return -drift_and_yaw * _v11_stair_forward_mask(env).float()


__all__ = [
    "v11_state", "v11_actor_perception", "v11_critic_pgtt", "v11_critic_geometry",
    "v11_landing_event_reward", "v11_platform_exit_success_reward", "v11_clearance_progress_reward",
    "v11_first_completion_reward", "v11_alternating_completion_reward",
    "v11_same_side_repeat_reward", "v11_simultaneous_completion_reward",
    "v11_softalt_platform_success_reward",
    "v11_post_first_clearance_surplus_reward", "v11_landing_center_reward",
    "v11_descent_step_complete_reward", "v11_descent_rear_support_progress_reward",
    "v11_descent_rear_follow_delay_penalty",
    "v11_descent_rear_roll_consistency_penalty", "v11_phase_flat_orientation_l2",
    "v11_phase_lin_vel_z_l2", "v11_pure_yaw_xy_drift_penalty",
    "v11_fd_safe_front_landing_progress_reward", "v11_fd_single_front_landing_reward",
    "v11_landing_edge_penalty", "v11_front_riser_contact_penalty", "v11_confirmed_front_riser_contact",
    "v11_world_forward_speed_deficit", "v11_world_forward_progress_reward",
    "v11_softalt_world_forward_speed_deficit", "v11_softalt_world_forward_progress_reward",
    "v11_action_term_mirror", "v11_descent_phase_joint_mirror", "v11_responsibility_metric", "v11_front_mobility_metric",
    "v11_front_collision_metric",
    "v11_alternation_metric",
    "v11_softalt_metric", "v11_stair_masked_track_lin_vel_xy_exp",
    "v11_ascent_route_coupled_track_lin_vel_xy_exp",
    "v11_ascent_route_coupled_track_ang_vel_z_exp",
    "v11_ascent_route_engagement_metric",
    "v11_event_reachability_metric",
    "v11_descent_metric", "v11_fd_clearance_metric", "v11_turn_metric",
    "v11_phase_relaxed_action_rate_l2", "v11_phase_relaxed_action_smooth_l2",
    "v11_phase_relaxed_front_joint_pos_penalty", "v11_front_hipy_participation_settlement_reward",
    "v11_front_knee_excess_penalty",
    "v11_path_lateral_deviation_penalty", "v11_path_heading_error_penalty",
    "v11_non_wheel_contact_force_penalty", "v11_non_wheel_scrape_penalty", "v11_wheel_impact_penalty",
    "v11_stair_straightness_penalty",
]
