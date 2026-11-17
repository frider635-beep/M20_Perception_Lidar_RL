"""V10 first-riser perception, privileged labels, events, and curriculum."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

from .observations import perceptive_elevation_clean
from .perceptive_spatial import _elevation_points_yaw, extract_nearest_riser_points
from .rewards import track_lin_vel_xy_exp


V10_TOPOLOGY_FLAT = 0
V10_TOPOLOGY_ROUGH = 1
V10_TOPOLOGY_SINGLE = 2
V10_TOPOLOGY_FULL = 3
V10_RECENT_CAPACITY = 500
V10_RECENT_FIELDS = (
    "first_success", "miss", "single_complete", "full_success", "joint_success", "blind_only_success",
)
V10_GATED_OUTCOME_NAMES = (
    "vertical_contact_before_clearance",
    "clearance_not_reached",
    "clearance_reached_but_no_crossing",
    "crossing_reached_but_no_upper_support",
    "first_success",
)


def _cfg(env):
    return env.cfg.v10_perception


def _zeros(env, *shape, dtype=torch.float32):
    return torch.zeros((env.num_envs, *shape), device=env.device, dtype=dtype)


def v10_topology_ids(terrain_type, single_start, single_end, full_start, full_end):
    """Map exact official terrain-column ranges to V10 metric topologies."""
    topology = torch.full_like(terrain_type, V10_TOPOLOGY_ROUGH, dtype=torch.long)
    single = (terrain_type >= int(single_start)) & (terrain_type < int(single_end))
    full = (terrain_type >= int(full_start)) & (terrain_type < int(full_end))
    topology = torch.where(single, torch.full_like(topology, V10_TOPOLOGY_SINGLE), topology)
    topology = torch.where(full, torch.full_like(topology, V10_TOPOLOGY_FULL), topology)
    return topology


def v10_official_promotion_diagnostics(
    official_up, official_down, is_stair, terrain_level, first_step_success
):
    """Preserve official promotion while labeling former level>=2 gate opportunities."""
    diagnostic_scope = is_stair & (terrain_level >= 2)
    move_up = official_up
    move_down = official_down
    blind_shortcut = official_up & is_stair & ~first_step_success
    stair_up_opportunity = official_up & is_stair
    gated_stair_up_opportunity = official_up & diagnostic_scope
    would_fail_first = gated_stair_up_opportunity & ~first_step_success
    return (
        move_up, move_down, blind_shortcut, stair_up_opportunity,
        gated_stair_up_opportunity, would_fail_first,
    )


def v10_gated_outcome_masks(
    gated_opportunity,
    vertical_contact_before_clearance,
    clearance_reached,
    crossing_reached,
    upper_support_reached,
    first_success,
):
    """Return mutually exclusive first-step outcomes for gated promotion opportunities."""
    success = gated_opportunity & first_success
    remaining = gated_opportunity & ~success
    vertical_contact = remaining & vertical_contact_before_clearance
    remaining &= ~vertical_contact
    no_clearance = remaining & ~clearance_reached
    remaining &= clearance_reached
    no_crossing = remaining & ~crossing_reached
    remaining &= crossing_reached
    no_upper_support = remaining & ~upper_support_reached
    classified = success | vertical_contact | no_clearance | no_crossing | no_upper_support
    unclassified = gated_opportunity & ~classified
    return (
        vertical_contact,
        no_clearance,
        no_crossing,
        no_upper_support,
        success,
        unclassified,
    )


def v10_clearance_progress_step(
    wheel_bottom_z,
    lower_ground_z,
    upper_z,
    clearance_margin,
    active,
    max_clearance_progress,
):
    """Compute one monotonic first-riser clearance-progress shaping step."""
    target_z = upper_z + float(clearance_margin)
    target_range = (target_z - lower_ground_z).clamp_min(1.0e-6)
    progress = ((wheel_bottom_z - lower_ground_z) / target_range).clamp(0.0, 1.0)
    progress = torch.where(active, progress, torch.zeros_like(progress))
    delta_progress = torch.where(
        active,
        torch.relu(progress - max_clearance_progress),
        torch.zeros_like(progress),
    )
    next_max = torch.where(
        active,
        torch.maximum(max_clearance_progress, progress),
        max_clearance_progress,
    )
    return progress, delta_progress, next_max


def v10_first_wheel_geometry(
    wheel_pos,
    forces,
    target_x,
    target_y,
    target_lower_z,
    target_upper_z,
    target_dir_x,
    target_dir_y,
    target_valid,
    vertical_contact_latched,
    wheel_radius,
    contact_force_threshold,
    vertical_force_ratio,
    clearance_margin,
    upper_support_tolerance,
):
    """Pure first-wheel geometry shared by FR runtime logic and unit tests."""
    force_norm = forces.norm(dim=-1)
    contact = force_norm >= float(contact_force_threshold)
    dx, dy = wheel_pos[:, 0] - target_x, wheel_pos[:, 1] - target_y
    progress = dx * target_dir_x + dy * target_dir_y
    horizontal_force = forces[:, :2].norm(dim=-1)
    radius = float(wheel_radius)
    wheel_bottom = wheel_pos[:, 2] - radius
    vertical_overlap = (
        (wheel_pos[:, 2] + radius > target_lower_z)
        & (wheel_bottom < target_upper_z)
    )
    vertical_contact = (
        target_valid & contact & (progress.abs() <= radius) & vertical_overlap
        & (horizontal_force >= float(vertical_force_ratio) * force_norm.clamp_min(1.0e-6))
    )
    clearance = (
        target_valid & ~vertical_contact_latched & (progress < 0.0)
        & (wheel_bottom >= target_upper_z + float(clearance_margin))
    )
    crossing = target_valid & (progress >= 0.0)
    upper_support = (
        target_valid & crossing & contact
        & ((wheel_bottom - target_upper_z).abs() <= float(upper_support_tolerance))
        & (forces[:, 2] > float(contact_force_threshold))
    )
    return {
        "contact": contact,
        "progress": progress,
        "wheel_bottom": wheel_bottom,
        "vertical_contact": vertical_contact,
        "clearance": clearance,
        "crossing": crossing,
        "upper_support": upper_support,
    }


def v10_height_bucket(height):
    """Return aligned 3-cm bucket ids for the official 5-23 cm stair range."""
    boundaries = torch.tensor((0.08, 0.11, 0.14, 0.17, 0.20), device=height.device, dtype=height.dtype)
    return torch.bucketize(height, boundaries, right=True).clamp(0, 5)


def v10_route_events(
    single_mask,
    full_mask,
    target_valid,
    base_progress,
    hr_progress,
    stable,
    first_step_success_latched,
    perception_miss_latched,
    single_complete_latched,
    full_success_latched,
    joint_success_latched,
    step_depth,
    completion_margin,
):
    """Pure topology-strict single/full completion and joint event logic."""
    single_required = float(completion_margin)
    full_required = 5.0 * float(step_depth) + float(completion_margin)
    single_now = (
        single_mask & target_valid & (base_progress >= single_required)
        & (hr_progress >= single_required - 0.2) & stable
    )
    single_event = single_now & ~single_complete_latched
    full_now = (
        full_mask & target_valid & (base_progress >= full_required)
        & (hr_progress >= full_required - 0.2) & stable
    )
    full_event = full_now & ~full_success_latched
    full_latched_next = full_success_latched | full_now
    joint_now = full_mask & first_step_success_latched & full_latched_next
    joint_event = joint_now & ~joint_success_latched
    blind_only_event = full_event & perception_miss_latched
    return {
        "single_now": single_now,
        "single_event": single_event,
        "full_now": full_now,
        "full_event": full_event,
        "joint_now": joint_now,
        "joint_event": joint_event,
        "blind_only_event": blind_only_event,
    }


def v10_metric_event_masks(
    topology,
    lock_event,
    first_step_event,
    miss_event,
    single_event,
    full_event,
    joint_event,
    blind_only_event,
):
    """Pure masks used by lifetime/recent counters and unit tests."""
    single = topology == V10_TOPOLOGY_SINGLE
    full = topology == V10_TOPOLOGY_FULL
    valid_stair = single | full
    return {
        "single_encounter": lock_event & single,
        "full_encounter": lock_event & full,
        "all_stair_encounter": lock_event & valid_stair,
        "single_first_success": first_step_event & single,
        "full_first_success": first_step_event & full,
        "single_miss": miss_event & single,
        "full_miss": miss_event & full,
        "single_complete": single_event & single,
        "full_success": full_event & full,
        "joint_success": joint_event & full,
        "blind_only_success": blind_only_event & full,
        "rough_false_trigger": lock_event & (topology == V10_TOPOLOGY_ROUGH),
        "flat_false_trigger": lock_event & (topology == V10_TOPOLOGY_FLAT),
    }


def v10_recent_window_counts(ring, next_sequence, window, field=None, bucket=None):
    """Count the last N encounters in one topology ring; active encounters are included."""
    sequence = ring["sequence"]
    lower = max(int(next_sequence) - int(window), 0)
    valid = (sequence >= lower) & (sequence < int(next_sequence))
    if bucket is not None:
        valid &= ring["bucket"] == int(bucket)
    denominator = valid.sum()
    if field is None:
        return denominator, denominator
    numerator = (valid & ring[field]).sum()
    return numerator, denominator


def v10_recent_window_mean(ring, next_sequence, window, field, bucket=None):
    """Mean a continuous encounter field over a recent window and optional height bucket."""
    sequence = ring["sequence"]
    lower = max(int(next_sequence) - int(window), 0)
    valid = (sequence >= lower) & (sequence < int(next_sequence))
    if bucket is not None:
        valid &= ring["bucket"] == int(bucket)
    values = ring[field]
    numerator = torch.where(valid, values, torch.zeros_like(values)).sum()
    return numerator / valid.sum().clamp_min(1)


def v10_recent_gated_outcome_counts(ring, next_sequence, window, outcome=None):
    """Count one outcome in the last N finalized gated promotion opportunities."""
    sequence = ring["sequence"]
    lower = max(int(next_sequence) - int(window), 0)
    valid = (sequence >= lower) & (sequence < int(next_sequence))
    denominator = valid.sum()
    if outcome is None:
        return denominator, denominator
    numerator = (valid & (ring["outcome"] == int(outcome))).sum()
    return numerator, denominator


def v10_recent_gated_clearance_mean(ring, next_sequence, window):
    """Mean finalized max-clearance progress over the last N gated opportunities."""
    sequence = ring["sequence"]
    lower = max(int(next_sequence) - int(window), 0)
    valid = (sequence >= lower) & (sequence < int(next_sequence))
    denominator = valid.sum()
    numerator = torch.where(
        valid, ring["max_clearance_progress"], torch.zeros_like(ring["max_clearance_progress"])
    ).sum()
    return numerator / denominator.clamp_min(1)


def _append_gated_recent(env, outcome, max_clearance_progress, mask):
    ids = torch.where(mask)[0]
    count = int(ids.numel())
    if count == 0:
        return
    start = env._v10_gated_recent_next
    sequence = torch.arange(start, start + count, device=env.device, dtype=torch.long)
    slots = sequence.remainder(V10_RECENT_CAPACITY)
    keep = min(count, V10_RECENT_CAPACITY)
    keep_ids = ids[-keep:]
    keep_sequence = sequence[-keep:]
    keep_slots = slots[-keep:]
    env._v10_gated_recent["sequence"][keep_slots] = keep_sequence
    env._v10_gated_recent["outcome"][keep_slots] = outcome[keep_ids]
    env._v10_gated_recent["max_clearance_progress"][keep_slots] = max_clearance_progress[keep_ids]
    env._v10_gated_recent_next = start + count


def _new_recent_ring(env):
    ring = {
        "sequence": torch.full((V10_RECENT_CAPACITY,), -1, device=env.device, dtype=torch.long),
        "bucket": torch.full((V10_RECENT_CAPACITY,), -1, device=env.device, dtype=torch.long),
    }
    for field in V10_RECENT_FIELDS:
        ring[field] = torch.zeros(V10_RECENT_CAPACITY, device=env.device, dtype=torch.bool)
    for field in (
        "height", "fr_action_amplitude_mean", "fr_action_amplitude_max",
        "absolute_fr_max_lift", "max_clearance_progress",
        "fr_hipx_action", "fr_hipy_action", "fr_knee_action",
        "abs_fr_hipx_action", "abs_fr_hipy_action", "abs_fr_knee_action",
        "max_active_fr_lift", "active_lift_ratio",
    ):
        ring[field] = torch.zeros(V10_RECENT_CAPACITY, device=env.device)
    return ring


def _append_recent_ring(env, state, key, mask, bucket, slot_name, sequence_name):
    ids = torch.where(mask)[0]
    count = int(ids.numel())
    if count == 0:
        return
    start = env._v10_recent_next[key]
    sequence = torch.arange(start, start + count, device=env.device, dtype=torch.long)
    slots = sequence.remainder(V10_RECENT_CAPACITY)
    state[slot_name][ids] = slots
    state[sequence_name][ids] = sequence
    keep = min(count, V10_RECENT_CAPACITY)
    keep_ids = ids[-keep:]
    keep_sequence = sequence[-keep:]
    keep_slots = slots[-keep:]
    ring = env._v10_recent_metrics[key]
    ring["sequence"][keep_slots] = keep_sequence
    ring["bucket"][keep_slots] = bucket[keep_ids]
    ring["height"][keep_slots] = state["locked_h"][keep_ids]
    for field in V10_RECENT_FIELDS:
        ring[field][keep_slots] = False
    for field in (
        "fr_action_amplitude_mean", "fr_action_amplitude_max",
        "absolute_fr_max_lift", "max_clearance_progress",
        "fr_hipx_action", "fr_hipy_action", "fr_knee_action",
        "abs_fr_hipx_action", "abs_fr_hipy_action", "abs_fr_knee_action",
        "max_active_fr_lift", "active_lift_ratio",
    ):
        ring[field][keep_slots] = 0.0
    env._v10_recent_next[key] = start + count


def _update_recent_ring(state, ring, event, slot_name, sequence_name, field):
    ids = torch.where(event & (state[sequence_name] >= 0))[0]
    if ids.numel() == 0:
        return
    slots = state[slot_name][ids]
    sequence = state[sequence_name][ids]
    current = ring["sequence"][slots] == sequence
    ring[field][slots[current]] = True


def _update_recent_ring_values(state, ring, mask, slot_name, sequence_name):
    """Copy current encounter-level continuous diagnostics into their ring slot."""
    ids = torch.where(mask & (state[sequence_name] >= 0))[0]
    if ids.numel() == 0:
        return
    slots = state[slot_name][ids]
    sequence = state[sequence_name][ids]
    current = ring["sequence"][slots] == sequence
    ids = ids[current]
    slots = slots[current]
    if ids.numel() == 0:
        return
    action_count = state["fr_action_amplitude_count"][ids].clamp_min(1).float()
    ring["fr_action_amplitude_mean"][slots] = (
        state["fr_action_amplitude_sum"][ids] / action_count
    )
    ring["fr_action_amplitude_max"][slots] = state["max_fr_action_amplitude"][ids]
    ring["absolute_fr_max_lift"][slots] = state["absolute_fr_max_lift"][ids]
    ring["max_clearance_progress"][slots] = state["max_clearance_progress"][ids]
    joint_mean = state["fr_joint_action_sum"][ids] / action_count[:, None]
    joint_abs_mean = state["fr_joint_action_abs_sum"][ids] / action_count[:, None]
    for index, name in enumerate(("fr_hipx_action", "fr_hipy_action", "fr_knee_action")):
        ring[name][slots] = joint_mean[:, index]
    for index, name in enumerate(
        ("abs_fr_hipx_action", "abs_fr_hipy_action", "abs_fr_knee_action")
    ):
        ring[name][slots] = joint_abs_mean[:, index]
    ring["max_active_fr_lift"][slots] = state["max_active_fr_lift"][ids]
    ring["active_lift_ratio"][slots] = (
        state["max_active_fr_lift"][ids] / state["locked_h"][ids].clamp_min(1.0e-6)
    )


def _initialize(env, asset: Articulation):
    if hasattr(env, "_v10_state"):
        return
    contact = env.scene.sensors["contact_forces"]
    fr_body_ids, fr_body_names = asset.find_bodies("fr_wheel")
    fr_contact_ids, fr_contact_names = contact.find_bodies("fr_wheel")
    rear_body_ids, rear_body_names = asset.find_bodies("hr_wheel")
    if len(fr_body_ids) != 1 or fr_body_names != ["fr_wheel"]:
        raise RuntimeError(f"V10 requires one FR wheel body, got {fr_body_names}")
    if len(fr_contact_ids) != 1 or fr_contact_names != ["fr_wheel"]:
        raise RuntimeError(f"V10 requires one FR contact body, got {fr_contact_names}")
    if len(rear_body_ids) != 1 or rear_body_names != ["hr_wheel"]:
        raise RuntimeError(f"V10 FULL completion requires one HR rear wheel body, got {rear_body_names}")
    env._v10_fr_body_id = fr_body_ids[0]
    env._v10_fr_contact_id = fr_contact_ids[0]
    env._v10_full_rear_body_id = rear_body_ids[0]
    cfg = _cfg(env)
    delay = int(cfg.perception_delay_steps)
    env._v10_feature_ring = _zeros(env, delay + 1, 3)
    env._v10_ring_index = 0
    env._v10_state = {
        "valid_count": _zeros(env, dtype=torch.long),
        "invalid_count": _zeros(env, dtype=torch.long),
        "no_stair_count": _zeros(env, dtype=torch.long),
        "first_riser_consumed": _zeros(env, dtype=torch.bool),
        "first_riser_locked": _zeros(env, dtype=torch.bool),
        "first_valid_seen": _zeros(env, dtype=torch.bool),
        "b_on_latched": _zeros(env, dtype=torch.bool),
        "d_first_valid": _zeros(env),
        "d_first_lock": _zeros(env),
        "d_b_on": _zeros(env),
        "locked_h": _zeros(env),
        "locked_d_raw": _zeros(env),
        "locked_d_filtered": _zeros(env),
        "first_riser_visible": _zeros(env, dtype=torch.bool),
        "pass_count": _zeros(env, dtype=torch.long),
        "b_prev_raw": _zeros(env, dtype=torch.bool),
        "actor_prev_feature": _zeros(env, 3),
        "b_duration_steps": _zeros(env, dtype=torch.long),
        "first_riser_triggered": _zeros(env, dtype=torch.bool),
        "target_valid": _zeros(env, dtype=torch.bool),
        "target_x": _zeros(env),
        "target_y": _zeros(env),
        "target_lower_z": _zeros(env),
        "target_upper_z": _zeros(env),
        "target_dir_x": _zeros(env),
        "target_dir_y": _zeros(env),
        "precontact_clearance_latched": _zeros(env, dtype=torch.bool),
        "clearance_progress": _zeros(env),
        "clearance_progress_delta": _zeros(env),
        "max_clearance_progress": _zeros(env),
        "fr_action_amplitude_sum": _zeros(env),
        "fr_action_amplitude_count": _zeros(env, dtype=torch.long),
        "max_fr_action_amplitude": _zeros(env),
        "absolute_fr_max_lift": _zeros(env),
        "fr_joint_action_sum": _zeros(env, 3),
        "fr_joint_action_abs_sum": _zeros(env, 3),
        "fr_base_z_at_b_on": _zeros(env),
        "fr_base_z_reference_valid": _zeros(env, dtype=torch.bool),
        "max_active_fr_lift": _zeros(env),
        "vertical_contact_before_clearance_latched": _zeros(env, dtype=torch.bool),
        "vertical_contact_latched": _zeros(env, dtype=torch.bool),
        "crossing_latched": _zeros(env, dtype=torch.bool),
        "upper_support_latched": _zeros(env, dtype=torch.bool),
        "first_step_success_latched": _zeros(env, dtype=torch.bool),
        # Episode-level handoff latch. Unlike encounter state, this deliberately
        # survives first-riser rearm and is cleared only by episode reset.
        "post_first_velocity_latched": _zeros(env, dtype=torch.bool),
        "perception_miss_latched": _zeros(env, dtype=torch.bool),
        "single_step_complete_latched": _zeros(env, dtype=torch.bool),
        "full_stair_success_latched": _zeros(env, dtype=torch.bool),
        "joint_success_latched": _zeros(env, dtype=torch.bool),
        # Cache-safe episode reset detection. Observation evaluation can reuse
        # the previous v10_state on the reset step, so checking only == 0 can
        # miss reset. A buffer decrease is observable on the following update.
        "last_episode_length_buf": env.episode_length_buf.clone(),
        "encounter_topology": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
        "encounter_bucket": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
        "recent_all_slot": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
        "recent_all_sequence": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
        "recent_topology_slot": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
        "recent_topology_sequence": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
    }
    env._v10_last_step = -1
    env._v10_last_actor_feature = _zeros(env, 3)
    env._v10_metrics_total = {
        # Topology axis: 0=single-step, 1=full staircase.
        "stair_encounters": _zeros(env, 2, dtype=torch.long),
        "first_success": _zeros(env, 2, dtype=torch.long),
        "miss": _zeros(env, 2, dtype=torch.long),
        "single_complete": _zeros(env, 2, dtype=torch.long),
        "full_success": _zeros(env, 2, dtype=torch.long),
        "joint_success": _zeros(env, 2, dtype=torch.long),
        "blind_only_success": _zeros(env, 2, dtype=torch.long),
        "b_duration_steps": _zeros(env, dtype=torch.long),
        "b_duration_encounters": _zeros(env, dtype=torch.long),
        "d_at_b_on_sum": _zeros(env), "d_at_b_on_count": _zeros(env, dtype=torch.long),
        "d_first_valid_sum": _zeros(env), "d_first_valid_count": _zeros(env, dtype=torch.long),
        "d_first_lock_sum": _zeros(env), "d_first_lock_count": _zeros(env, dtype=torch.long),
        "d_first_b_on_sum": _zeros(env), "d_first_b_on_count": _zeros(env, dtype=torch.long),
        # [no_valid_riser, not_locked, D_out_of_range, command_not_forward]
        "pre_b_on_blocker_steps": _zeros(env, 4, dtype=torch.long),
        "d_at_b_raw_off_sum": _zeros(env), "d_at_b_raw_off_count": _zeros(env, dtype=torch.long),
        "d_at_b_actor_off_sum": _zeros(env), "d_at_b_actor_off_count": _zeros(env, dtype=torch.long),
        "teacher_success_while_b": _zeros(env, dtype=torch.long),
        "teacher_miss_while_b": _zeros(env, dtype=torch.long),
        "terrain_steps": _zeros(env, 4, dtype=torch.long),
        "terrain_b_steps": _zeros(env, 4, dtype=torch.long),
        "false_trigger_events": _zeros(env, 2, dtype=torch.long),
        # Minimal FIRST -> command-following handoff diagnostics. These are
        # simulator-only accumulators and never enter policy observations.
        "post_first_full_steps": _zeros(env, dtype=torch.long),
        "post_first_motion_steps": _zeros(env, dtype=torch.long),
        "post_first_reverse_steps": _zeros(env, dtype=torch.long),
        "post_first_strong_reverse_steps": _zeros(env, dtype=torch.long),
        "post_first_v_along_command_sum": _zeros(env),
        "post_first_command_speed_sum": _zeros(env),
        "post_first_velocity_error_sum": _zeros(env),
        "post_first_xy_reward_sum": _zeros(env),
        "post_first_event_count": _zeros(env, dtype=torch.long),
        # Axes: topology(single/full), height bucket.
        "bucket_encounters": _zeros(env, 2, 6, dtype=torch.long),
        "bucket_first_success": _zeros(env, 2, 6, dtype=torch.long),
        "bucket_miss": _zeros(env, 2, 6, dtype=torch.long),
        "bucket_single_complete": _zeros(env, 2, 6, dtype=torch.long),
        "bucket_full_success": _zeros(env, 2, 6, dtype=torch.long),
        "bucket_joint_success": _zeros(env, 2, 6, dtype=torch.long),
        "bucket_blind_only_success": _zeros(env, 2, 6, dtype=torch.long),
    }
    env._v10_recent_metrics = {key: _new_recent_ring(env) for key in ("all", "single", "full")}
    env._v10_recent_next = {key: 0 for key in ("all", "single", "full")}


def _clear_encounter(state, mask):
    for name in (
        "first_riser_consumed", "first_riser_locked", "first_riser_visible",
        "first_valid_seen", "b_on_latched",
        "fr_base_z_reference_valid",
        "first_riser_triggered", "target_valid", "b_prev_raw",
        "precontact_clearance_latched", "vertical_contact_before_clearance_latched",
        "vertical_contact_latched", "crossing_latched",
        "upper_support_latched", "first_step_success_latched", "perception_miss_latched",
        "single_step_complete_latched", "full_stair_success_latched", "joint_success_latched",
    ):
        state[name][mask] = False
    for name in (
        "locked_h", "locked_d_raw", "locked_d_filtered", "d_first_valid", "d_first_lock", "d_b_on",
        "clearance_progress", "clearance_progress_delta", "max_clearance_progress",
        "fr_action_amplitude_sum", "max_fr_action_amplitude", "absolute_fr_max_lift",
        "fr_joint_action_sum", "fr_joint_action_abs_sum",
        "fr_base_z_at_b_on", "max_active_fr_lift",
        "target_x", "target_y",
        "target_lower_z", "target_upper_z", "target_dir_x", "target_dir_y",
    ):
        state[name][mask] = 0.0
    state["pass_count"][mask] = 0
    state["b_duration_steps"][mask] = 0
    state["fr_action_amplitude_count"][mask] = 0
    state["actor_prev_feature"][mask] = 0.0
    for name in (
        "encounter_topology", "encounter_bucket", "recent_all_slot", "recent_all_sequence",
        "recent_topology_slot", "recent_topology_sequence",
    ):
        state[name][mask] = -1


def _reset_episode(env, state, reset):
    if not reset.any():
        return
    _clear_encounter(state, reset)
    state["post_first_velocity_latched"][reset] = False
    state["valid_count"][reset] = 0
    state["invalid_count"][reset] = 0
    state["no_stair_count"][reset] = 0
    env._v10_feature_ring[reset] = 0.0
    env._v10_last_actor_feature[reset] = 0.0


def v10_event_pulses(
    triggered,
    clearance_latched,
    crossing_latched,
    support_latched,
    vertical_contact_latched,
    success_latched,
    miss_latched,
    vertical_contact,
):
    """Pure one-shot first-step event logic used by runtime and unit tests."""
    success_now = (
        triggered & clearance_latched & crossing_latched & support_latched
        & ~vertical_contact_latched & ~miss_latched
    )
    success_event = success_now & ~success_latched
    miss_now = triggered & vertical_contact & ~success_latched
    miss_event = miss_now & ~miss_latched
    return success_now, success_event, miss_now, miss_event


def v10_deployable_lifecycle_step(state, cfg, detected_valid, detected_h, detected_d, command):
    """Update first-riser lock/consume using PGTT and deployable command only."""
    state["valid_count"] = torch.where(
        detected_valid, state["valid_count"] + 1, torch.zeros_like(state["valid_count"])
    )
    confirmed = state["valid_count"] >= int(cfg.confirmation_frames)
    lock_event = confirmed & ~state["first_riser_locked"] & ~state["first_riser_consumed"]
    state["first_riser_locked"] |= lock_event
    state["locked_h"] = torch.where(lock_event, detected_h, state["locked_h"])
    state["locked_d_raw"] = torch.where(lock_event, detected_d, state["locked_d_raw"])
    state["locked_d_filtered"] = torch.where(lock_event, detected_d, state["locked_d_filtered"])

    height_match = (detected_h - state["locked_h"]).abs() <= float(cfg.track_height_tolerance)
    distance_continuity = detected_d <= state["locked_d_raw"] + float(cfg.track_max_forward_jump)
    track_visible = state["first_riser_locked"] & detected_valid & height_match & distance_continuity
    state["invalid_count"] = torch.where(
        track_visible, torch.zeros_like(state["invalid_count"]), state["invalid_count"] + 1
    )
    held_visible = track_visible | (
        state["first_riser_locked"] & (state["invalid_count"] <= int(cfg.hysteresis_frames))
    )
    alpha = float(cfg.distance_filter_alpha)
    filtered_d = alpha * detected_d + (1.0 - alpha) * state["locked_d_filtered"]
    filtered_h = alpha * detected_h + (1.0 - alpha) * state["locked_h"]
    update = track_visible & ~lock_event
    state["locked_d_raw"] = torch.where(update, detected_d, state["locked_d_raw"])
    state["locked_d_filtered"] = torch.where(update, filtered_d, state["locked_d_filtered"])
    state["locked_h"] = torch.where(update, filtered_h, state["locked_h"])
    state["first_riser_visible"] = held_visible & ~state["first_riser_consumed"]

    passed_signal = state["first_riser_locked"] & (
        (track_visible & (state["locked_d_filtered"] <= float(cfg.pgtt_pass_exit_distance)))
        | (
            ~held_visible
            & (state["locked_d_filtered"] <= float(cfg.pgtt_pass_lost_max_distance))
        )
    )
    state["pass_count"] = torch.where(
        passed_signal, state["pass_count"] + 1, torch.zeros_like(state["pass_count"])
    )
    passed_event = (
        (state["pass_count"] >= int(cfg.pass_confirmation_frames))
        & ~state["first_riser_consumed"]
    )
    state["first_riser_consumed"] |= passed_event
    state["first_riser_visible"] &= ~state["first_riser_consumed"]

    in_window = (
        (state["locked_d_filtered"] >= float(cfg.anticipation_distance_min))
        & (state["locked_d_filtered"] <= float(cfg.anticipation_distance_max))
    )
    b_raw = (
        state["first_riser_locked"] & ~state["first_riser_consumed"]
        & state["first_riser_visible"] & in_window
        & (command > float(cfg.forward_command_threshold))
    )
    h_raw = torch.where(b_raw, state["locked_h"], torch.zeros_like(state["locked_h"]))
    d_raw = torch.where(b_raw, state["locked_d_filtered"], torch.zeros_like(state["locked_d_filtered"]))
    return {
        "lock_event": lock_event, "passed_event": passed_event, "b_raw": b_raw,
        "h_raw": h_raw, "d_raw": d_raw, "track_visible": track_visible,
    }


def v10_delay_pipeline_step(feature_ring, write_index, feature_input, delay_steps):
    """Pure ring-buffer update; lifecycle transitions never flush this buffer."""
    feature_ring[:, write_index] = feature_input
    read_index = (write_index - int(delay_steps)) % feature_ring.shape[1]
    actor_feature = feature_ring[:, read_index].clone()
    next_index = (write_index + 1) % feature_ring.shape[1]
    actor_feature[:, 1:] *= actor_feature[:, :1]
    return actor_feature, next_index


def v10_state(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("v10_pgtt_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Update the deployable first-riser gate and simulator-only teacher labels."""
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_v10_last_step", None) == step:
        return env._v10_state
    asset: Articulation = env.scene[asset_cfg.name]
    _initialize(env, asset)
    state, cfg = env._v10_state, _cfg(env)
    episode_length = getattr(
        env, "episode_length_buf", torch.ones(env.num_envs, device=env.device)
    )
    reset = episode_length < state["last_episode_length_buf"]
    _reset_episode(env, state, reset)
    state["last_episode_length_buf"] = episode_length.clone()

    points = _elevation_points_yaw(env, sensor_cfg, asset)
    virtual_base = _zeros(env, 4, 2)
    riser = extract_nearest_riser_points(
        points,
        virtual_base,
        corridor_half_width=float(cfg.corridor_half_width),
        h_min=float(cfg.min_step_height) - float(cfg.height_valid_tolerance),
        d_end=-0.30,
    )
    detected_valid = riser["valid"][:, 0]
    detected_h = torch.where(
        detected_valid, riser["riser_height"][:, 0], torch.zeros(env.num_envs, device=env.device)
    )
    detected_d = torch.where(
        detected_valid, riser["riser_distance"][:, 0], torch.zeros(env.num_envs, device=env.device)
    )
    height_tolerance = float(cfg.height_valid_tolerance)
    detected_valid &= (
        (detected_h >= float(cfg.min_step_height) - height_tolerance)
        & (detected_h <= float(cfg.max_step_height) + height_tolerance)
    )
    first_valid_event = detected_valid & ~state["first_valid_seen"]
    state["d_first_valid"] = torch.where(
        first_valid_event, detected_d, state["d_first_valid"]
    )
    state["first_valid_seen"] |= first_valid_event
    command = env.command_manager.get_command("base_velocity")[:, 0]
    lifecycle = v10_deployable_lifecycle_step(
        state, cfg, detected_valid, detected_h, detected_d, command
    )
    lock_event = lifecycle["lock_event"]
    pgtt_passed_event = lifecycle["passed_event"]
    b_raw, h_raw, d_raw = lifecycle["b_raw"], lifecycle["h_raw"], lifecycle["d_raw"]
    state["d_first_lock"] = torch.where(lock_event, detected_d, state["d_first_lock"])
    first_b_on_event = b_raw & ~state["b_on_latched"]
    state["d_b_on"] = torch.where(
        first_b_on_event, state["locked_d_filtered"], state["d_b_on"]
    )
    activate = b_raw & ~state["first_riser_triggered"]
    terrain_type = env.scene.terrain.terrain_types
    topology = v10_topology_ids(
        terrain_type,
        int(cfg.single_stair_column_start), int(cfg.single_stair_column_end),
        int(cfg.full_stair_column_start), int(cfg.full_stair_column_end),
    )
    full_topology = topology == V10_TOPOLOGY_FULL
    in_window_now = (
        (state["locked_d_filtered"] >= float(cfg.anticipation_distance_min))
        & (state["locked_d_filtered"] <= float(cfg.anticipation_distance_max))
    )
    pre_b_on = (
        full_topology & ~state["b_on_latched"] & ~b_raw & ~state["first_riser_consumed"]
    )
    no_valid_riser = pre_b_on & (
        ~detected_valid | (state["first_riser_locked"] & ~state["first_riser_visible"])
    )
    not_locked = pre_b_on & detected_valid & ~state["first_riser_locked"]
    d_out_of_range = (
        pre_b_on & detected_valid & state["first_riser_locked"]
        & state["first_riser_visible"] & ~in_window_now
    )
    command_not_forward = (
        pre_b_on & detected_valid & state["first_riser_locked"]
        & state["first_riser_visible"] & in_window_now
        & (command <= float(cfg.forward_command_threshold))
    )
    pre_b_on_blockers = torch.stack(
        (no_valid_riser, not_locked, d_out_of_range, command_not_forward), dim=1
    )
    state["b_on_latched"] |= first_b_on_event
    state["encounter_topology"] = torch.where(lock_event, topology, state["encounter_topology"])
    locked_bucket = v10_height_bucket(state["locked_h"])
    state["encounter_bucket"] = torch.where(lock_event, locked_bucket, state["encounter_bucket"])

    root = asset.data.root_pos_w
    quat = asset.data.root_quat_w
    w, xq, yq, zq = quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    local_x = riser["riser_x"][:, 0]
    local_y = riser["riser_y"][:, 0]
    target_x = root[:, 0] + cy * local_x - sy * local_y
    target_y = root[:, 1] + sy * local_x + cy * local_y
    lower_z = riser["riser_lower_world"][:, 0]
    upper_z = lower_z + state["locked_h"]
    for name, value in (
        ("target_x", target_x), ("target_y", target_y),
        ("target_lower_z", lower_z), ("target_upper_z", upper_z),
        ("target_dir_x", cy), ("target_dir_y", sy),
    ):
        state[name] = torch.where(activate, value, state[name])
    state["target_valid"] |= activate
    state["first_riser_triggered"] |= activate

    fr_pos = asset.data.body_pos_w[:, env._v10_fr_body_id]
    fr_pos_base = quat_apply_inverse(asset.data.root_quat_w, fr_pos - root)
    fr_base_z = fr_pos_base[:, 2]
    state["fr_base_z_at_b_on"] = torch.where(
        first_b_on_event, fr_base_z, state["fr_base_z_at_b_on"]
    )
    state["fr_base_z_reference_valid"] |= first_b_on_event
    forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, env._v10_fr_contact_id]
    wheel_radius = float(cfg.wheel_radius)
    geometry = v10_first_wheel_geometry(
        fr_pos, forces,
        state["target_x"], state["target_y"],
        state["target_lower_z"], state["target_upper_z"],
        state["target_dir_x"], state["target_dir_y"],
        state["target_valid"], state["vertical_contact_latched"],
        wheel_radius, float(cfg.contact_force_threshold), float(cfg.vertical_force_ratio),
        float(cfg.clearance_margin), float(cfg.upper_support_tolerance),
    )
    contact = geometry["contact"]
    fr_progress = geometry["progress"]
    wheel_bottom = geometry["wheel_bottom"]
    vertical_contact = geometry["vertical_contact"]
    state["vertical_contact_before_clearance_latched"] |= (
        vertical_contact & ~state["precontact_clearance_latched"]
    )
    clearance_active = (
        full_topology & state["target_valid"] & state["first_riser_locked"] & b_raw
        & state["fr_base_z_reference_valid"]
        & ~state["crossing_latched"] & (fr_progress < 0.0)
    )
    fr_joint_action = env.action_manager.action[:, 3:6]
    fr_action_amplitude = fr_joint_action.abs().mean(dim=1)
    state["fr_action_amplitude_sum"] += torch.where(
        clearance_active, fr_action_amplitude, torch.zeros_like(fr_action_amplitude)
    )
    state["fr_action_amplitude_count"] += clearance_active.long()
    state["fr_joint_action_sum"] += torch.where(
        clearance_active[:, None], fr_joint_action, torch.zeros_like(fr_joint_action)
    )
    state["fr_joint_action_abs_sum"] += torch.where(
        clearance_active[:, None], fr_joint_action.abs(), torch.zeros_like(fr_joint_action)
    )
    state["max_fr_action_amplitude"] = torch.where(
        clearance_active,
        torch.maximum(state["max_fr_action_amplitude"], fr_action_amplitude),
        state["max_fr_action_amplitude"],
    )
    absolute_fr_lift = (wheel_bottom - state["target_lower_z"]).clamp_min(0.0)
    state["absolute_fr_max_lift"] = torch.where(
        clearance_active,
        torch.maximum(state["absolute_fr_max_lift"], absolute_fr_lift),
        state["absolute_fr_max_lift"],
    )
    # Base-frame z points upward: retracting the FR wheel toward the base makes
    # z less negative and therefore yields positive active_fr_lift.
    active_fr_lift = fr_base_z - state["fr_base_z_at_b_on"]
    state["max_active_fr_lift"] = torch.where(
        clearance_active,
        torch.maximum(state["max_active_fr_lift"], active_fr_lift.clamp_min(0.0)),
        state["max_active_fr_lift"],
    )
    (
        state["clearance_progress"],
        state["clearance_progress_delta"],
        state["max_clearance_progress"],
    ) = v10_clearance_progress_step(
        wheel_bottom,
        state["target_lower_z"],
        state["target_upper_z"],
        float(cfg.clearance_margin),
        clearance_active,
        state["max_clearance_progress"],
    )
    clearance = geometry["clearance"]
    state["precontact_clearance_latched"] |= clearance
    crossing = geometry["crossing"]
    state["crossing_latched"] |= crossing
    upper_support = geometry["upper_support"]
    state["upper_support_latched"] |= upper_support

    success_now, first_step_event, miss_now, perception_miss_event = v10_event_pulses(
        state["first_riser_triggered"], state["precontact_clearance_latched"],
        state["crossing_latched"], state["upper_support_latched"],
        state["vertical_contact_latched"], state["first_step_success_latched"],
        state["perception_miss_latched"], vertical_contact,
    )
    state["first_step_success_latched"] |= success_now
    state["post_first_velocity_latched"] |= first_step_event
    state["perception_miss_latched"] |= miss_now
    state["vertical_contact_latched"] |= vertical_contact

    single_mask = topology == V10_TOPOLOGY_SINGLE
    full_mask = topology == V10_TOPOLOGY_FULL
    base_progress = (
        (root[:, 0] - state["target_x"]) * state["target_dir_x"]
        + (root[:, 1] - state["target_y"]) * state["target_dir_y"]
    )
    # FULL/SINGLE completion intentionally keeps the rear HR wheel as the last
    # wheel that must clear the route. This is separate from FR first-step skill.
    full_rear_pos = asset.data.body_pos_w[:, env._v10_full_rear_body_id]
    full_rear_progress = (
        (full_rear_pos[:, 0] - state["target_x"]) * state["target_dir_x"]
        + (full_rear_pos[:, 1] - state["target_y"]) * state["target_dir_y"]
    )
    stable = (asset.data.projected_gravity_b[:, 2] < -0.65) & (
        asset.data.projected_gravity_b[:, :2].abs().amax(dim=1) < 0.55
    )
    route = v10_route_events(
        single_mask, full_mask, state["target_valid"], base_progress, full_rear_progress, stable,
        state["first_step_success_latched"], state["perception_miss_latched"],
        state["single_step_complete_latched"], state["full_stair_success_latched"],
        state["joint_success_latched"], float(cfg.step_depth), float(cfg.full_success_margin),
    )
    single_step_event = route["single_event"]
    full_event = route["full_event"]
    joint_event = route["joint_event"]
    blind_only_event = route["blind_only_event"]
    state["single_step_complete_latched"] |= route["single_now"]
    state["full_stair_success_latched"] |= route["full_now"]
    state["joint_success_latched"] |= route["joint_now"]

    feature_raw = torch.stack((b_raw.float(), h_raw, d_raw), dim=1)
    hold = torch.rand(env.num_envs, device=env.device) < float(cfg.frame_hold_probability)
    feature_input = torch.where(hold[:, None], env._v10_last_actor_feature, feature_raw)
    feature_input[:, 1] += torch.randn(env.num_envs, device=env.device) * float(cfg.height_noise_std)
    feature_input[:, 2] += torch.randn(env.num_envs, device=env.device) * float(cfg.distance_noise_std)
    feature_input[:, 1:].clamp_(min=0.0)
    actor_feature, env._v10_ring_index = v10_delay_pipeline_step(
        env._v10_feature_ring, int(env._v10_ring_index), feature_input, int(cfg.perception_delay_steps)
    )
    env._v10_last_actor_feature = actor_feature

    # Cumulative denominators are updated at control rate, so sparse events
    # cannot be missed by the once-per-iteration TensorBoard logger.
    totals = env._v10_metrics_total
    one_hot_topology = torch.nn.functional.one_hot(topology, num_classes=4).to(torch.long)
    totals["terrain_steps"] += one_hot_topology
    totals["terrain_b_steps"] += one_hot_topology * b_raw[:, None].long()
    totals["pre_b_on_blocker_steps"] += pre_b_on_blockers.long()
    post_first = full_mask & state["post_first_velocity_latched"]
    command_xy = env.command_manager.get_command("base_velocity")[:, :2]
    command_speed = torch.linalg.norm(command_xy, dim=1)
    commanded_motion = post_first & (command_speed > 1.0e-6)
    command_unit = command_xy / command_speed.clamp_min(1.0e-6)[:, None]
    base_velocity_xy = asset.data.root_lin_vel_b[:, :2]
    velocity_along_command = (base_velocity_xy * command_unit).sum(dim=1)
    velocity_error = torch.linalg.norm(command_xy - base_velocity_xy, dim=1)
    official_xy_reward = torch.exp(
        -torch.sum(torch.square(command_xy - base_velocity_xy), dim=1) / 0.5
    ) * 5.0 * float(env.step_dt)
    totals["post_first_full_steps"] += post_first.long()
    totals["post_first_motion_steps"] += commanded_motion.long()
    totals["post_first_reverse_steps"] += (commanded_motion & (velocity_along_command < 0.0)).long()
    totals["post_first_strong_reverse_steps"] += (
        commanded_motion & (velocity_along_command < -0.2)
    ).long()
    totals["post_first_v_along_command_sum"] += torch.where(
        commanded_motion, velocity_along_command, torch.zeros_like(velocity_along_command)
    )
    totals["post_first_command_speed_sum"] += torch.where(
        commanded_motion, command_speed, torch.zeros_like(command_speed)
    )
    totals["post_first_velocity_error_sum"] += torch.where(
        commanded_motion, velocity_error, torch.zeros_like(velocity_error)
    )
    totals["post_first_xy_reward_sum"] += torch.where(
        post_first, official_xy_reward, torch.zeros_like(official_xy_reward)
    )
    totals["post_first_event_count"] += (first_step_event & full_mask).long()
    for event, value, sum_name, count_name in (
        (first_valid_event & full_topology, state["d_first_valid"], "d_first_valid_sum", "d_first_valid_count"),
        (lock_event & full_topology, state["d_first_lock"], "d_first_lock_sum", "d_first_lock_count"),
        (first_b_on_event & full_topology, state["d_b_on"], "d_first_b_on_sum", "d_first_b_on_count"),
    ):
        totals[sum_name] += torch.where(event, value, torch.zeros_like(value))
        totals[count_name] += event.long()
    b_on_event = b_raw & ~state["b_prev_raw"]
    b_raw_off_event = ~b_raw & state["b_prev_raw"]
    actor_off_event = (state["actor_prev_feature"][:, 0] > 0.5) & (actor_feature[:, 0] <= 0.5)
    state["b_duration_steps"] += b_raw.long()
    completed_duration = pgtt_passed_event & (state["b_duration_steps"] > 0)
    totals["b_duration_steps"] += torch.where(
        completed_duration, state["b_duration_steps"], torch.zeros_like(state["b_duration_steps"])
    )
    totals["b_duration_encounters"] += completed_duration.long()
    for event, value, sum_name, count_name in (
        (b_on_event, state["locked_d_filtered"], "d_at_b_on_sum", "d_at_b_on_count"),
        (b_raw_off_event, state["locked_d_filtered"], "d_at_b_raw_off_sum", "d_at_b_raw_off_count"),
        (actor_off_event, state["actor_prev_feature"][:, 2], "d_at_b_actor_off_sum", "d_at_b_actor_off_count"),
    ):
        totals[sum_name] += torch.where(event, value, torch.zeros_like(value))
        totals[count_name] += event.long()

    metric_masks = v10_metric_event_masks(
        state["encounter_topology"], lock_event, first_step_event, perception_miss_event,
        single_step_event, full_event, joint_event, blind_only_event,
    )
    valid_stair_lock = metric_masks["all_stair_encounter"]
    stair_index = (state["encounter_topology"] - V10_TOPOLOGY_SINGLE).clamp(0, 1)
    stair_one_hot = torch.nn.functional.one_hot(stair_index, num_classes=2).to(torch.long)
    stair_one_hot *= valid_stair_lock[:, None].long()
    totals["stair_encounters"] += stair_one_hot
    for name, event in (
        ("first_success", metric_masks["single_first_success"] | metric_masks["full_first_success"]),
        ("miss", metric_masks["single_miss"] | metric_masks["full_miss"]),
        ("single_complete", metric_masks["single_complete"]),
        ("full_success", metric_masks["full_success"]),
        ("joint_success", metric_masks["joint_success"]),
        ("blind_only_success", metric_masks["blind_only_success"]),
    ):
        totals[name] += torch.nn.functional.one_hot(stair_index, num_classes=2).to(torch.long) * event[:, None].long()

    totals["teacher_success_while_b"] += (first_step_event & b_raw).long()
    totals["teacher_miss_while_b"] += (perception_miss_event & b_raw).long()
    totals["false_trigger_events"][:, 0] += metric_masks["flat_false_trigger"].long()
    totals["false_trigger_events"][:, 1] += metric_masks["rough_false_trigger"].long()

    bucket = state["encounter_bucket"].clamp(0, 5)
    bucket_one_hot = torch.nn.functional.one_hot(bucket, num_classes=6).to(torch.long)
    topology_bucket = stair_one_hot[:, :, None] * bucket_one_hot[:, None, :]
    totals["bucket_encounters"] += topology_bucket
    for name, event in (
        ("bucket_first_success", metric_masks["single_first_success"] | metric_masks["full_first_success"]),
        ("bucket_miss", metric_masks["single_miss"] | metric_masks["full_miss"]),
        ("bucket_single_complete", metric_masks["single_complete"]),
        ("bucket_full_success", metric_masks["full_success"]),
        ("bucket_joint_success", metric_masks["joint_success"]),
        ("bucket_blind_only_success", metric_masks["blind_only_success"]),
    ):
        event_topology = torch.nn.functional.one_hot(stair_index, num_classes=2).to(torch.long)
        totals[name] += event_topology[:, :, None] * bucket_one_hot[:, None, :] * event[:, None, None].long()

    # Append each valid stair lock to both the all-stair and its topology ring.
    _append_recent_ring(
        env, state, "all", valid_stair_lock, bucket, "recent_all_slot", "recent_all_sequence"
    )
    _append_recent_ring(
        env, state, "single", metric_masks["single_encounter"], bucket,
        "recent_topology_slot", "recent_topology_sequence",
    )
    _append_recent_ring(
        env, state, "full", metric_masks["full_encounter"], bucket,
        "recent_topology_slot", "recent_topology_sequence",
    )
    recent_events = {
        "first_success": metric_masks["single_first_success"] | metric_masks["full_first_success"],
        "miss": metric_masks["single_miss"] | metric_masks["full_miss"],
        "single_complete": metric_masks["single_complete"],
        "full_success": metric_masks["full_success"],
        "joint_success": metric_masks["joint_success"],
        "blind_only_success": metric_masks["blind_only_success"],
    }
    for field, event in recent_events.items():
        _update_recent_ring(
            state, env._v10_recent_metrics["all"], event,
            "recent_all_slot", "recent_all_sequence", field,
        )
        _update_recent_ring(
            state, env._v10_recent_metrics["single"], event & (state["encounter_topology"] == V10_TOPOLOGY_SINGLE),
            "recent_topology_slot", "recent_topology_sequence", field,
        )
        _update_recent_ring(
            state, env._v10_recent_metrics["full"], event & (state["encounter_topology"] == V10_TOPOLOGY_FULL),
            "recent_topology_slot", "recent_topology_sequence", field,
        )
    active_stair_encounter = (
        (state["encounter_topology"] == V10_TOPOLOGY_SINGLE)
        | (state["encounter_topology"] == V10_TOPOLOGY_FULL)
    )
    _update_recent_ring_values(
        state, env._v10_recent_metrics["all"], active_stair_encounter,
        "recent_all_slot", "recent_all_sequence",
    )
    _update_recent_ring_values(
        state, env._v10_recent_metrics["single"],
        state["encounter_topology"] == V10_TOPOLOGY_SINGLE,
        "recent_topology_slot", "recent_topology_sequence",
    )
    _update_recent_ring_values(
        state, env._v10_recent_metrics["full"],
        state["encounter_topology"] == V10_TOPOLOGY_FULL,
        "recent_topology_slot", "recent_topology_sequence",
    )
    state["b_prev_raw"] = b_raw.clone()
    state["actor_prev_feature"] = actor_feature.clone()

    state["no_stair_count"] = torch.where(
        detected_valid, torch.zeros_like(state["no_stair_count"]), state["no_stair_count"] + 1
    )
    rearm = state["first_riser_consumed"] & (state["no_stair_count"] >= int(cfg.rearm_no_stair_steps))

    state.update({
        "valid": detected_valid, "detected_h": detected_h, "detected_d": detected_d,
        "h_raw": h_raw, "d_raw": d_raw, "b_raw": b_raw,
        "actor_feature": actor_feature, "vertical_contact": vertical_contact,
        "contact": contact, "fr_z": fr_pos[:, 2], "fr_progress": fr_progress,
        "full_rear_hr_progress": full_rear_progress,
        "first_step_event": first_step_event, "perception_miss_event": perception_miss_event,
        "single_step_complete_event": single_step_event,
        "full_stair_event": full_event, "joint_event": joint_event,
        "blind_only_event": blind_only_event, "topology": topology,
        "pgtt_passed_event": pgtt_passed_event, "lock_event": lock_event,
        "base_vx": asset.data.root_lin_vel_b[:, 0],
    })
    _clear_encounter(state, rearm)
    env._v10_last_step = step
    return state


def v10_actor_perception(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return v10_state(env, sensor_cfg=sensor_cfg)["actor_feature"]


def v10_critic_pgtt(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return perceptive_elevation_clean(env, sensor_cfg)


def v10_critic_geometry(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    state = v10_state(env, sensor_cfg=sensor_cfg)
    return torch.stack((state["b_raw"].float(), state["h_raw"], state["d_raw"]), dim=1)


def v10_critic_teacher(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    state = v10_state(env, sensor_cfg=sensor_cfg)
    return torch.stack(
        (state["fr_z"], state["contact"].float(), state["vertical_contact"].float(), state["fr_progress"]),
        dim=1,
    )


def v10_first_step_success_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return v10_state(env, sensor_cfg=sensor_cfg)["first_step_event"].float()


def v10_post_first_track_lin_vel_xy_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("v10_pgtt_scanner"),
):
    """Repeat the official XY command-tracking reward after FIRST until episode reset."""
    active = v10_state(env, sensor_cfg=sensor_cfg)["post_first_velocity_latched"]
    official_reward = track_lin_vel_xy_exp(
        env, std=std, command_name=command_name, asset_cfg=asset_cfg
    )
    return official_reward * active.float()


def v10_fr_clearance_progress_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    """Return dt-normalized delta so RewardManager contributes 2 * delta at weight 2."""
    delta = v10_state(env, sensor_cfg=sensor_cfg)["clearance_progress_delta"]
    return delta / float(env.step_dt)


def v10_perception_miss_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return v10_state(env, sensor_cfg=sensor_cfg)["perception_miss_event"].float()


def v10_joint_success_reward(env, sensor_cfg=SceneEntityCfg("v10_pgtt_scanner")):
    return v10_state(env, sensor_cfg=sensor_cfg)["joint_event"].float()


def v10_official_terrain_levels(
    env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Official M20 distance-based terrain curriculum with perception diagnostics only."""
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    terrain = env.scene.terrain
    if ids.numel() == 0:
        return terrain.terrain_levels.float().mean()
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(
        asset.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1
    )
    official_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    official_down = (
        distance < torch.norm(command[ids, :2], dim=1) * env.max_episode_length_s * 0.5
    ) & ~official_up

    cfg = _cfg(env)
    terrain_type = terrain.terrain_types[ids]
    is_stair = (
        (terrain_type >= int(cfg.full_stair_column_start))
        & (terrain_type < int(cfg.full_stair_column_end))
    )
    if hasattr(env, "_v10_state"):
        first_step_success = env._v10_state["first_step_success_latched"][ids]
    else:
        first_step_success = torch.zeros_like(official_up)
    terrain_level = terrain.terrain_levels[ids]
    (
        move_up, move_down, blind_shortcut, stair_up_opportunity,
        gated_stair_up_opportunity, would_fail_first,
    ) = v10_official_promotion_diagnostics(
        official_up, official_down, is_stair, terrain_level, first_step_success,
    )

    if not hasattr(env, "_v10_blind_shortcut_count"):
        env._v10_blind_shortcut_count = torch.zeros((), device=env.device, dtype=torch.long)
        env._v10_stair_official_up_count = torch.zeros((), device=env.device, dtype=torch.long)
        env._v10_stair_curriculum_count = torch.zeros((), device=env.device, dtype=torch.long)
        env._v10_stair_promotion_count = torch.zeros((), device=env.device, dtype=torch.long)
        env._v10_stair_gated_official_up_count = torch.zeros((), device=env.device, dtype=torch.long)
        env._v10_stair_would_fail_first_count = torch.zeros(
            (), device=env.device, dtype=torch.long
        )
        env._v10_gated_outcome_counts = torch.zeros(
            len(V10_GATED_OUTCOME_NAMES), device=env.device, dtype=torch.long
        )
        env._v10_gated_unclassified_count = torch.zeros((), device=env.device, dtype=torch.long)
        env._v10_gated_recent = {
            "sequence": torch.full(
                (V10_RECENT_CAPACITY,), -1, device=env.device, dtype=torch.long
            ),
            "outcome": torch.full(
                (V10_RECENT_CAPACITY,), -1, device=env.device, dtype=torch.long
            ),
            "max_clearance_progress": torch.zeros(
                V10_RECENT_CAPACITY, device=env.device
            ),
        }
        env._v10_gated_recent_next = 0
    env._v10_blind_shortcut_count += blind_shortcut.sum()
    env._v10_stair_official_up_count += stair_up_opportunity.sum()
    env._v10_stair_curriculum_count += is_stair.sum()
    env._v10_stair_promotion_count += (move_up & is_stair).sum()
    env._v10_stair_gated_official_up_count += gated_stair_up_opportunity.sum()
    env._v10_stair_would_fail_first_count += would_fail_first.sum()

    if hasattr(env, "_v10_state"):
        state = env._v10_state
        outcomes = v10_gated_outcome_masks(
            gated_stair_up_opportunity,
            state["vertical_contact_before_clearance_latched"][ids],
            state["precontact_clearance_latched"][ids],
            state["crossing_latched"][ids],
            state["upper_support_latched"][ids],
            state["first_step_success_latched"][ids],
        )
        outcome_code = torch.full_like(terrain_level, -1)
        for index, outcome_mask in enumerate(outcomes[:-1]):
            env._v10_gated_outcome_counts[index] += outcome_mask.sum()
            outcome_code[outcome_mask] = index
        env._v10_gated_unclassified_count += outcomes[-1].sum()
        _append_gated_recent(
            env,
            outcome_code,
            state["max_clearance_progress"][ids],
            gated_stair_up_opportunity,
        )

    terrain.update_env_origins(ids, move_up, move_down)
    mean_level = terrain.terrain_levels.float().mean()
    from .rewards import update_gait_level_from_terrain_mean

    update_gait_level_from_terrain_mean(mean_level)
    return mean_level


def v10_two_dimensional_curriculum(env, env_ids):
    """Backward-compatible alias for the restored official terrain curriculum."""
    return v10_official_terrain_levels(env, env_ids)


__all__ = [name for name in globals() if name.startswith("v10_")]
