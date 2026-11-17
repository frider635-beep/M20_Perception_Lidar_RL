"""Training-only physical flight geometry and first-swing evidence.

No policy observations, actions or descent state are changed here.
"""
from __future__ import annotations

import torch


def inverted_flight_layout(size, border, platform, tread):
    """Mirror the installed Isaac mesh generator's ring boundaries (not heights)."""
    rings = int(min((float(v) - 2 * border - platform) // (2 * tread) + 1 for v in size))
    if rings < 1 or tread <= 0:
        raise ValueError("V135 requires a nonempty inverted-pyramid stair flight")
    radii = tuple((float(v) - 2 * border) / 2 - rings * tread for v in size)
    # The border adds the last rise from the outermost ring to z=0.
    return radii, rings + int(border > 0)


def flight_anchor(root_xy, origin_xy, heading, radii):
    """Use the generated stair normal, not the robot's jittered yaw."""
    side = torch.round(heading / (0.5 * torch.pi)).long() % 4
    axes = root_xy.new_tensor(((1, 0), (0, 1), (-1, 0), (0, -1)))
    forward = axes[side]
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    lane = ((root_xy - origin_xy) * lateral).sum(1)
    radii = torch.as_tensor(radii, device=root_xy.device, dtype=root_xy.dtype)
    radius = radii[side % 2] if radii.ndim == 1 else radii.gather(1, (side % 2)[:, None]).squeeze(1)
    return origin_xy + radius[:, None] * forward + lane[:, None] * lateral, forward


def first_swing_step(side, finished, failed, grounded, anchor, contact_steps,
                     wheel_forward, wheel_bottom, contact, lower_z, upper_z,
                     active, target_id, completed_side, valid_completion, *,
                     lift_min, support_tolerance, tread_end,
                     confirmation_frames):
    """One immutable first attempt per episode; restarts cannot erase failure.

    A meaningful swing requires prior ground contact and a meaningful lift.
    Touchdown is debounced. Landing back on the lower tread is an empty swing;
    landing beyond B1 is a skip. FL/FR are symmetric; simultaneous onset fails.
    """
    can_start = active & (target_id == 1) & (side == -1)
    onset = (can_start[:, None] & grounded & ~contact
             & (wheel_bottom - lower_z[:, None] > lift_min))
    started = onset.any(1)
    simultaneous = onset.all(1)
    side = torch.where(started, onset.long().argmax(1), side)
    failed = failed | simultaneous
    tracked = side.clamp(0, 1)
    def chosen(value):
        return value.gather(1, tracked[:, None]).squeeze(1)
    pending = active & (side >= 0) & ~finished
    contact_steps = torch.where(pending & chosen(contact), contact_steps + 1,
                                torch.zeros_like(contact_steps))
    touchdown = pending & (contact_steps >= confirmation_frames)
    selected_x, selected_z = chosen(wheel_forward), chosen(wheel_bottom)
    skip = touchdown & ((selected_x >= tread_end) | (selected_z > upper_z + support_tolerance))
    empty = touchdown & ~skip & ((selected_x < 0) | (selected_z < upper_z - support_tolerance))
    success = pending & ~failed & (target_id == 1) & valid_completion & (completed_side == side)
    failed = failed | empty | skip
    finished = finished | empty | skip | success | simultaneous
    anchor = torch.where(can_start[:, None] & contact, wheel_forward, anchor)
    grounded = grounded | (can_start[:, None] & contact)
    return dict(side=side, finished=finished, failed=failed, grounded=grounded,
                anchor=anchor, contact_steps=contact_steps, started=started,
                empty=empty, skip=skip, success=success, simultaneous=simultaneous)


def tread_lateral_geometry(wheel_xy, origin_xy, forward, radii, target_id, tread, wheel_radius):
    """Physical ring face bounds; scanner ray matching tolerance is unrelated."""
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    wheel_lateral = ((wheel_xy - origin_xy[:, None, :]) * lateral[:, None, :]).sum(-1)
    transverse_radius = (lateral.abs() * torch.as_tensor(radii, device=wheel_xy.device, dtype=wheel_xy.dtype)).sum(-1)
    half_width = (transverse_radius + (target_id - 1).clamp_min(0) * tread - wheel_radius).clamp_min(0)
    return wheel_lateral, half_width[:, None]
