"""Continuous, wheel-centric spatial guidance for M20 perceptive stairs.

This module has no clock, prescribed gait grouping, or contact-driven control
transition.  Terrain geometry is used only to construct dense reward targets;
none of the tensors produced here are actor observations.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .observations import perceptive_elevation_clean_map, perceptive_elevation_world_map


WHEEL_NAMES = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


def smoothstep01(u: torch.Tensor) -> torch.Tensor:
    """Cubic Hermite interpolation on [0, 1]."""
    u = u.clamp(0.0, 1.0)
    return u.square() * (3.0 - 2.0 * u)


def spatial_guidance_curve(
    distance: torch.Tensor,
    d_start: float,
    d_peak: float,
    d_hold: float,
    d_end: float,
) -> torch.Tensor:
    """Continuous clearance request ``g(d)`` for signed wheel/riser distance.

    ``distance = x_riser - x_wheel`` in the robot yaw frame.  It is positive
    before the wheel reaches a riser and negative after passing it.
    """
    if not d_start > d_peak > d_hold > d_end:
        raise ValueError("guidance distances must satisfy d_start > d_peak > d_hold > d_end")
    rise_u = (d_start - distance) / (d_start - d_peak)
    fall_u = (d_hold - distance) / (d_hold - d_end)
    rising = smoothstep01(rise_u)
    falling = 1.0 - smoothstep01(fall_u)
    return torch.where(distance > d_peak, rising, torch.where(distance >= d_hold, 1.0, falling))


def distance_to_vertical_segment(
    distance: torch.Tensor, wheel_z: torch.Tensor, z_lower: torch.Tensor, z_upper: torch.Tensor
) -> torch.Tensor:
    """Euclidean distance from a wheel centre to the finite riser segment."""
    closest_z = wheel_z.clamp(min=z_lower, max=z_upper)
    return torch.sqrt(distance.square() + (wheel_z - closest_z).square())


def minimum_safe_wheel_center_height(
    distance: torch.Tensor,
    z_lower: torch.Tensor,
    z_upper: torch.Tensor,
    radius_clear: float,
) -> torch.Tensor:
    """Minimum world-Z satisfying riser-segment and upper-tread clearance.

    Before the wheel reaches the riser, only centres within ``R_clear`` of the
    vertical line require the top-corner arc. At/after crossing, upper-tread
    clearance is mandatory.
    """
    r = torch.as_tensor(radius_clear, device=distance.device, dtype=distance.dtype)
    corner = z_upper + torch.sqrt(torch.clamp(r.square() - distance.square(), min=0.0))
    # Outside the collision band the riser imposes no extra world-Z lower
    # bound; z_preview still starts from the nominal lower-tread centre
    # z_lower+R. Returning z_lower here avoids incorrectly adding margin in
    # the far field.
    before = torch.where((distance > 0.0) & (distance < r), corner, z_lower)
    after = z_upper + r
    return torch.where(distance <= 0.0, after, before)


def classify_vertical_riser_contact(
    wheel_xy: torch.Tensor,
    wheel_z: torch.Tensor,
    contact_force_w: torch.Tensor,
    riser_x: torch.Tensor,
    z_lower: torch.Tensor,
    z_upper: torch.Tensor,
    *,
    wheel_radius: float,
    force_threshold: float = 5.0,
    normal_ratio_threshold: float = 0.55,
) -> torch.Tensor:
    """Classify BAD vertical-face contact; upper-tread support is not BAD.

    This is a metric classifier, never a policy observation or gait state.
    A candidate must be within one wheel radius of the riser plane, vertically
    overlap the finite riser segment, and have a sufficiently horizontal
    contact-force component. Normal upper-tread contact is predominantly Z.
    """
    d = riser_x - wheel_xy[..., 0]
    force_norm = contact_force_w.norm(dim=-1)
    horizontal = contact_force_w[..., :2].norm(dim=-1)
    vertical_overlap = (wheel_z + wheel_radius > z_lower) & (wheel_z - wheel_radius < z_upper)
    near_face = d.abs() <= wheel_radius
    face_dominant = horizontal >= normal_ratio_threshold * force_norm.clamp_min(1.0e-6)
    return (force_norm >= force_threshold) & near_face & vertical_overlap & face_dominant


def _median3(profile: torch.Tensor) -> torch.Tensor:
    padded = torch.cat((profile[:, :1], profile, profile[:, -1:]), dim=1)
    return padded.unfold(1, 3, 1).median(dim=-1).values


def extract_nearest_riser(
    elevation_map: torch.Tensor,
    wheel_xy: torch.Tensor,
    *,
    corridor_half_width: float,
    h_min: float,
    d_end: float,
) -> dict[str, torch.Tensor]:
    """Detect the closest still-relevant positive riser for every wheel.

    A lateral corridor is reduced by a median at each X row, followed by a
    three-cell median filter. Candidate height is the positive adjacent-cell
    jump in that robust profile. Among candidates with ``d > d_end``, minimum
    absolute signed distance retains a just-passed riser through the smooth
    descent window without any discrete gait state.
    """
    if elevation_map.ndim != 3 or elevation_map.shape[1:] != (21, 15):
        raise ValueError(f"expected elevation [N,21,15], got {tuple(elevation_map.shape)}")
    if wheel_xy.shape[1:] != (4, 2):
        raise ValueError(f"expected wheel_xy [N,4,2], got {tuple(wheel_xy.shape)}")
    device, dtype = elevation_map.device, elevation_map.dtype
    x_desc = torch.linspace(1.25, -0.75, 21, device=device, dtype=dtype)
    y_desc = torch.linspace(0.70, -0.70, 15, device=device, dtype=dtype)
    x_asc = x_desc.flip(0)
    n = elevation_map.shape[0]
    batch = torch.arange(n, device=device)

    all_height, all_lower, all_distance, all_x, all_valid, all_relief = [], [], [], [], [], []
    for wheel_idx in range(4):
        wheel_y = wheel_xy[:, wheel_idx, 1]
        corridor = torch.abs(y_desc[None, :] - wheel_y[:, None]) <= corridor_half_width
        # Always include the nearest column at grid edges.
        nearest_col = torch.argmin(torch.abs(y_desc[None, :] - wheel_y[:, None]), dim=1)
        corridor[batch, nearest_col] = True
        values = elevation_map.masked_fill(~corridor[:, None, :], float("nan"))
        profile = torch.nanmedian(values, dim=2).values.flip(1)  # X ascending
        profile = _median3(profile)

        # Riser j lies between X[j] (lower) and X[j+1] (upper). A half-cell XY
        # shift splits one edge into two adjacent positive gradients. Aggregate
        # only adjacent pairs (not a three-cell lower/upper plateau), retain a
        # single rightmost local maximum, and use the gradient centroid for X.
        gradient = torch.relu(profile[:, 1:] - profile[:, :-1])
        x_grid = 0.5 * (x_asc[:-1] + x_asc[1:])
        gradient_next = torch.cat((gradient[:, 1:], torch.zeros_like(gradient[:, :1])), dim=1)
        x_next = torch.cat((x_grid[1:], x_grid[-1:]))
        height = gradient + gradient_next
        riser_x = (gradient * x_grid[None, :] + gradient_next * x_next[None, :]) / height.clamp_min(
            torch.finfo(dtype).eps
        )
        left = torch.cat((torch.full_like(height[:, :1], float("-inf")), height[:, :-1]), dim=1)
        right = torch.cat((height[:, 1:], torch.full_like(height[:, :1], float("-inf"))), dim=1)
        local_peak = (height >= left) & (height > right)
        distance = riser_x - wheel_xy[:, wheel_idx, 0, None]
        candidates = local_peak & (height >= h_min) & (distance > d_end)
        score = torch.where(candidates, distance.abs(), torch.full_like(distance, float("inf")))
        index = score.argmin(dim=1)
        valid = candidates.any(dim=1)
        picked_height = height[batch, index]
        picked_lower = profile[batch, index]
        picked_distance = distance[batch, index]
        picked_x = riser_x[batch, index]
        all_height.append(torch.where(valid, picked_height, torch.zeros_like(picked_height)))
        all_lower.append(torch.where(valid, picked_lower, torch.zeros_like(picked_lower)))
        all_distance.append(torch.where(valid, picked_distance, torch.full_like(picked_distance, float("inf"))))
        all_x.append(torch.where(valid, picked_x, torch.zeros_like(picked_x)))
        all_valid.append(valid)
        all_relief.append(profile.amax(dim=1) - profile.amin(dim=1))

    return {
        "riser_height": torch.stack(all_height, dim=1),
        "riser_lower_local": torch.stack(all_lower, dim=1),
        "riser_distance": torch.stack(all_distance, dim=1),
        "riser_x": torch.stack(all_x, dim=1),
        "valid": torch.stack(all_valid, dim=1),
        "corridor_relief": torch.stack(all_relief, dim=1),
    }


def _elevation_points_yaw(env, sensor_cfg: SceneEntityCfg, asset: Articulation) -> torch.Tensor:
    """Return actual ray-hit points in the robot yaw-aligned frame [N,21,15,3]."""
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w
    if hits.shape[1] != 21 * 15:
        raise RuntimeError(f"expected 315 ray hits, got {tuple(hits.shape)}")
    delta = hits - asset.data.root_pos_w[:, None, :]
    w, xq, yq, zq = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq.square() + zq.square()))
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    local = torch.stack(
        (c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1], hits[..., 2]), dim=-1
    )
    # GridPatternCfg(ordering="xy") stores outer-y/inner-x.  Semantic map row
    # zero is front (+x), column zero is left (+y).
    return local.reshape(-1, 15, 21, 3).transpose(1, 2).flip(dims=(1, 2))


def extract_nearest_riser_points(
    points_yaw: torch.Tensor,
    wheel_xy: torch.Tensor,
    *,
    corridor_half_width: float,
    h_min: float,
    d_end: float,
) -> dict[str, torch.Tensor]:
    """Detect risers from actual yaw-frame ray points, preserving diagonal edges."""
    if points_yaw.shape[1:] != (21, 15, 3):
        raise ValueError(f"expected points [N,21,15,3], got {tuple(points_yaw.shape)}")
    n, _, _, _ = points_yaw.shape
    batch = torch.arange(n, device=points_yaw.device)
    out = {key: [] for key in (
        "riser_height", "riser_lower_world", "riser_distance", "riser_x", "riser_y", "valid", "corridor_relief"
    )}
    for wi in range(4):
        p = points_yaw
        valid = torch.isfinite(p[..., 2])
        # Convert front-to-rear semantic rows to increasing local x.
        p = p.flip(1)
        valid = valid.flip(1)
        low, up = p[:, :-1], p[:, 1:]
        pair_valid = valid[:, :-1] & valid[:, 1:]
        height = torch.relu(up[..., 2] - low[..., 2])
        edge_x = 0.5 * (low[..., 0] + up[..., 0])
        edge_y = 0.5 * (low[..., 1] + up[..., 1])
        lower_z = low[..., 2]
        d = edge_x - wheel_xy[:, wi, 0, None, None]
        corridor = torch.abs(edge_y - wheel_xy[:, wi, 1, None, None]) <= corridor_half_width
        candidates = pair_valid & corridor & (height >= h_min) & (d > d_end)
        # Keep the nearest edge even when a diagonal riser falls between two
        # adjacent scanner rows and the finite-difference height is numerically
        # split below h_min at one yaw sign.  The height is still required to
        # be positive; this is only a sampling-robustness fallback.
        fallback = pair_valid & corridor & (height > 0.005) & (d > d_end)
        candidates = candidates | fallback
        no_candidate = ~candidates.reshape(n, -1).any(dim=1)
        wide_corridor = torch.abs(edge_y - wheel_xy[:, wi, 1, None, None]) <= 2.0 * corridor_half_width
        wide_fallback = pair_valid & wide_corridor & (height >= h_min) & (d > d_end)
        candidates = torch.where(no_candidate[:, None, None], wide_fallback, candidates)
        score = torch.where(candidates, d.abs(), torch.full_like(d, float("inf")))
        flat = score.reshape(n, -1)
        index = flat.argmin(dim=1)
        flat_height = height.reshape(n, -1)
        flat_lower = lower_z.reshape(n, -1)
        flat_x = edge_x.reshape(n, -1)
        flat_y = edge_y.reshape(n, -1)
        flat_d = d.reshape(n, -1)
        any_valid = candidates.reshape(n, -1).any(dim=1)
        for key, value, fill in (
            ("riser_height", flat_height[batch, index], 0.0),
            ("riser_lower_world", flat_lower[batch, index], 0.0),
            ("riser_distance", flat_d[batch, index], float("inf")),
            ("riser_x", flat_x[batch, index], 0.0),
            ("riser_y", flat_y[batch, index], 0.0),
        ):
            out[key].append(torch.where(any_valid, value, torch.full_like(value, fill)))
        out["valid"].append(any_valid)
        corridor_points = valid & (torch.abs(p[..., 1] - wheel_xy[:, wi, 1, None, None]) <= corridor_half_width)
        z_max = torch.where(corridor_points, p[..., 2], torch.full_like(p[..., 2], float("-inf"))).amax(dim=(1, 2))
        z_min = torch.where(corridor_points, p[..., 2], torch.full_like(p[..., 2], float("inf"))).amin(dim=(1, 2))
        relief = torch.where(torch.isfinite(z_max) & torch.isfinite(z_min), z_max - z_min, torch.zeros_like(z_max))
        out["corridor_relief"].append(relief)
    return {key: torch.stack(values, dim=1) for key, values in out.items()}


def extract_two_risers_points(
    points_yaw: torch.Tensor,
    *,
    corridor_half_width: float,
    h_min: float,
    d_end: float,
    min_separation: float,
    signed_edges: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the current and next same-sign edge in the approach corridor.

    Outputs have shape ``[N, 2]`` ordered along +x.  The first slot retains a
    just-passed edge down to ``d_end``; the second slot must be farther forward
    by ``min_separation`` and have the same sign.  With ``signed_edges=False``
    this preserves the historical positive-riser contract.  With it enabled,
    positive heights denote an ascent and negative heights a descent.
    """
    if points_yaw.shape[1:] != (21, 15, 3):
        raise ValueError(f"expected points [N,21,15,3], got {tuple(points_yaw.shape)}")
    p = points_yaw.flip(1)
    finite = torch.isfinite(p[..., 2])
    low, up = p[:, :-1], p[:, 1:]
    pair_valid = finite[:, :-1] & finite[:, 1:]
    signed_height = up[..., 2] - low[..., 2]
    height = signed_height if signed_edges else torch.relu(signed_height)
    magnitude = height.abs() if signed_edges else height
    edge_x = 0.5 * (low[..., 0] + up[..., 0])
    edge_y = 0.5 * (low[..., 1] + up[..., 1])
    lower_z = torch.minimum(low[..., 2], up[..., 2])
    candidate = pair_valid & (edge_y.abs() <= float(corridor_half_width)) & (magnitude >= float(h_min))

    # Collapse each x row to one robust lateral peak so one physical riser cannot
    # occupy both output slots merely because several y rays hit it.
    masked_height = torch.where(candidate, magnitude, torch.full_like(magnitude, float("-inf")))
    y_index = masked_height.argmax(dim=2)
    row_valid = candidate.any(dim=2)
    gather = y_index.unsqueeze(-1)
    row_height = height.gather(2, gather).squeeze(-1)
    row_x = edge_x.gather(2, gather).squeeze(-1)
    row_y = edge_y.gather(2, gather).squeeze(-1)
    row_lower = lower_z.gather(2, gather).squeeze(-1)
    row_valid &= row_x > float(d_end)

    n = points_yaw.shape[0]
    batch = torch.arange(n, device=points_yaw.device)
    first_score = torch.where(row_valid, row_x.abs(), torch.full_like(row_x, float("inf")))
    first_index = first_score.argmin(dim=1)
    first_valid = row_valid.any(dim=1)
    first_x = row_x[batch, first_index]
    first_height = row_height[batch, first_index]
    same_sign = (row_height * first_height[:, None]) > 0.0 if signed_edges else torch.ones_like(row_valid)
    second_candidate = (
        row_valid & same_sign
        & (row_x > first_x[:, None] + float(min_separation))
        & first_valid[:, None]
    )
    second_score = torch.where(second_candidate, row_x, torch.full_like(row_x, float("inf")))
    second_index = second_score.argmin(dim=1)
    second_valid = second_candidate.any(dim=1)

    def pair(values: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        a = torch.where(first_valid, values[batch, first_index], torch.full_like(first_x, fill))
        b = torch.where(second_valid, values[batch, second_index], torch.full_like(first_x, fill))
        return torch.stack((a, b), dim=1)

    return {
        "valid": torch.stack((first_valid, second_valid), dim=1),
        "riser_height": pair(row_height),
        "riser_distance": pair(row_x),
        "riser_x": pair(row_x),
        "riser_y": pair(row_y),
        "riser_lower_world": pair(row_lower),
    }


def extract_three_risers_points(points_yaw: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
    """Return three ordered same-sign edges for task tracking and two-edge projection."""
    first_two = extract_two_risers_points(points_yaw, **kwargs)
    p = points_yaw.flip(1)
    finite = torch.isfinite(p[..., 2])
    low, up = p[:, :-1], p[:, 1:]
    signed = up[..., 2] - low[..., 2]
    signed_edges = bool(kwargs.get("signed_edges", False))
    height = signed if signed_edges else torch.relu(signed)
    magnitude = height.abs() if signed_edges else height
    edge_x = 0.5 * (low[..., 0] + up[..., 0])
    edge_y = 0.5 * (low[..., 1] + up[..., 1])
    lower = torch.minimum(low[..., 2], up[..., 2])
    candidate = finite[:, :-1] & finite[:, 1:] & (edge_y.abs() <= float(kwargs["corridor_half_width"])) & (
        magnitude >= float(kwargs["h_min"]))
    masked = torch.where(candidate, magnitude, torch.full_like(magnitude, float("-inf")))
    yi = masked.argmax(2); gather = yi.unsqueeze(-1)
    row_valid = candidate.any(2)
    row_h = height.gather(2, gather).squeeze(-1)
    row_x = edge_x.gather(2, gather).squeeze(-1)
    row_y = edge_y.gather(2, gather).squeeze(-1)
    row_l = lower.gather(2, gather).squeeze(-1)
    row_valid &= row_x > float(kwargs["d_end"])
    second_x = first_two["riser_x"][:, 1]
    same_sign = (row_h * first_two["riser_height"][:, 0:1]) > 0 if signed_edges else torch.ones_like(row_valid)
    third_candidate = row_valid & same_sign & first_two["valid"][:, 1:2] & (
        row_x > second_x[:, None] + float(kwargs["min_separation"]))
    score = torch.where(third_candidate, row_x, torch.full_like(row_x, float("inf")))
    idx = score.argmin(1); batch = torch.arange(points_yaw.shape[0], device=points_yaw.device)
    third_valid = third_candidate.any(1)
    def third(value, fill=0.0):
        picked = value[batch, idx]
        return torch.where(third_valid, picked, torch.full_like(picked, fill))[:, None]
    return {
        "valid": torch.cat((first_two["valid"], third_valid[:, None]), 1),
        "riser_height": torch.cat((first_two["riser_height"], third(row_h)), 1),
        "riser_distance": torch.cat((first_two["riser_distance"], third(row_x, float("inf"))), 1),
        "riser_x": torch.cat((first_two["riser_x"], third(row_x)), 1),
        "riser_y": torch.cat((first_two["riser_y"], third(row_y)), 1),
        "riser_lower_world": torch.cat((first_two["riser_lower_world"], third(row_l)), 1),
    }


def _semantic_body_ids(asset: Articulation) -> torch.Tensor:
    ids = []
    for name in WHEEL_NAMES:
        found, names = asset.find_bodies(name)
        if len(found) != 1 or names[0] != name:
            raise RuntimeError(f"expected exactly one body {name!r}, got {names}")
        ids.append(found[0])
    return torch.tensor(ids, device=asset.device, dtype=torch.long)


def wheel_positions_base(asset: Articulation, body_ids: torch.Tensor) -> torch.Tensor:
    """Wheel centres in full base frame, shape [N,4,3], FL/FR/HL/HR."""
    delta_w = asset.data.body_pos_w[:, body_ids] - asset.data.root_pos_w[:, None, :]
    quat = asset.data.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4)
    return math_utils.quat_apply_inverse(quat, delta_w.reshape(-1, 3)).reshape(-1, 4, 3)


def wheel_positions_yaw(asset: Articulation, body_ids: torch.Tensor) -> torch.Tensor:
    """Wheel centres in yaw-aligned local frame, shape [N,4,3]."""
    delta = asset.data.body_pos_w[:, body_ids] - asset.data.root_pos_w[:, None, :]
    w, x, y, z = asset.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    return torch.stack(
        (c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]),
        dim=-1,
    )


def perceptive_spatial_state(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Compute and cache all reward-side geometry once per control step."""
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_perceptive_spatial_step", None) == step:
        return env._perceptive_spatial_cache
    cfg = env.cfg.spatial_guidance
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "_perceptive_wheel_body_ids"):
        env._perceptive_wheel_body_ids = _semantic_body_ids(asset)
    body_ids = env._perceptive_wheel_body_ids
    wheel_b = wheel_positions_base(asset, body_ids)
    wheel_yaw = wheel_positions_yaw(asset, body_ids)
    if not hasattr(env, "_perceptive_z_nom"):
        env._perceptive_z_nom = wheel_b[..., 2].detach().clone()
    reset = getattr(env, "episode_length_buf", torch.ones(env.num_envs, device=env.device)) == 0
    env._perceptive_z_nom = torch.where(reset[:, None], wheel_b[..., 2].detach(), env._perceptive_z_nom)
    actual_lift = wheel_b[..., 2] - env._perceptive_z_nom

    elevation = perceptive_elevation_clean_map(env, sensor_cfg)
    elevation_world = perceptive_elevation_world_map(env, sensor_cfg)
    riser = extract_nearest_riser_points(
        _elevation_points_yaw(env, sensor_cfg, asset), wheel_yaw[..., :2],
        corridor_half_width=cfg.corridor_half_width, h_min=cfg.h_min, d_end=cfg.d_end,
    )
    guide = spatial_guidance_curve(riser["riser_distance"], cfg.d_start, cfg.d_peak, cfg.d_hold, cfg.d_end)
    guide = torch.where(riser["valid"], guide, torch.zeros_like(guide))
    radius = cfg.wheel_collision_radius
    radius_clear = radius + cfg.clearance_margin
    # The robust lower/upper corridor samples already used by the detector are
    # represented by local minimum + detected rise in world-Z. This preserves
    # terrain origin/level and does not equate stair height with leg shortening.
    valid_world = torch.isfinite(elevation_world)
    safe_world = torch.where(valid_world, elevation_world, torch.full_like(elevation_world, float("inf")))
    local_world_min = safe_world.flatten(1).amin(dim=1).view(-1, 1)
    local_world_min = torch.where(torch.isfinite(local_world_min), local_world_min, asset.data.root_pos_w[:, 2:3] - 0.58)
    if "riser_lower_world" in riser:
        z_lower = riser["riser_lower_world"]
        riser["riser_lower_local"] = z_lower - local_world_min.expand_as(z_lower)
    else:
        z_lower = local_world_min.expand_as(riser["riser_height"]) + riser["riser_lower_local"]
    z_upper = z_lower + riser["riser_height"]
    z_nominal_world = z_lower + radius
    z_cross = z_upper + radius_clear
    # PGTT geometry contract: height determines the final amplitude; distance
    # determines only the timing of the request/reward.  In particular, the
    # distance-dependent desired lift must never be used as the denominator of
    # clearance completion.
    final_required_lift = riser["riser_height"] + cfg.clearance_margin
    desired_lift_now = guide * final_required_lift
    z_preview = z_nominal_world + desired_lift_now
    z_geom_required = minimum_safe_wheel_center_height(
        riser["riser_distance"], z_lower, z_upper, radius_clear
    )
    z_ref_world = torch.maximum(z_preview, z_geom_required)
    z_actual_world = asset.data.body_pos_w[:, body_ids, 2]
    z_actual_relative = z_actual_world - z_lower
    z_ref_relative = z_ref_world - z_lower
    active = riser["valid"] & (guide > 0.0)
    target_range = z_ref_world - z_nominal_world
    wheel_bottom_z = z_actual_world - radius
    actual_geometric_lift = wheel_bottom_z - z_lower
    absolute_progress = (
        actual_geometric_lift / final_required_lift.clamp_min(1.0e-6)
    ).clamp(0.0, 1.0)

    # Historical metric only: this is the superseded distance-dependent
    # completion definition and must not affect reward or the new true metric.
    old_progress = (
        actual_geometric_lift / target_range.clamp_min(1.0e-6)
    ).clamp(0.0, 1.0)

    # Latch bad vertical-face contact for the episode.  V9 Stage-1 contains a
    # single riser, so an episode latch is exactly a current-riser latch.  A
    # wheel cannot collide first, lift later, and regain spatial reward.
    contact_sensor = env.scene.sensors["contact_forces"]
    if not hasattr(env, "_perceptive_contact_body_ids"):
        contact_ids = []
        for name in WHEEL_NAMES:
            found, names = contact_sensor.find_bodies(name)
            if len(found) != 1 or names != [name]:
                raise RuntimeError(f"expected contact body {name!r}, got {names}")
            contact_ids.append(found[0])
        env._perceptive_contact_body_ids = torch.tensor(
            contact_ids, device=asset.device, dtype=torch.long
        )
    wheel_forces_w = contact_sensor.data.net_forces_w[:, env._perceptive_contact_body_ids]
    vertical_contact = classify_vertical_riser_contact(
        wheel_yaw[..., :2], z_actual_world, wheel_forces_w,
        riser["riser_x"], z_lower, z_upper, wheel_radius=radius,
    ) & riser["valid"]
    if not hasattr(env, "_perceptive_vertical_contact_latched"):
        env._perceptive_vertical_contact_latched = torch.zeros_like(vertical_contact)
    env._perceptive_vertical_contact_latched = torch.where(
        reset[:, None], torch.zeros_like(vertical_contact), env._perceptive_vertical_contact_latched
    )
    env._perceptive_vertical_contact_latched |= vertical_contact
    vertical_contact_latched = env._perceptive_vertical_contact_latched

    reward_active = active & ~vertical_contact_latched
    per_wheel_reward = guide * (2.0 * absolute_progress - absolute_progress.square())
    spatial_raw = torch.where(
        reward_active, per_wheel_reward, torch.zeros_like(per_wheel_reward)
    ).sum(1)
    spatial_raw /= reward_active.sum(1).clamp_min(1)
    spatial_raw = torch.where(reward_active.any(1), spatial_raw, torch.zeros_like(spatial_raw))

    precontact_window = riser["valid"] & (riser["riser_distance"] > 0.0) & (
        riser["riser_distance"] <= cfg.d_start
    ) & ~vertical_contact_latched
    true_precontact_clearance = precontact_window & (
        wheel_bottom_z >= z_upper + cfg.clearance_margin
    )
    riser_plane_crossing = riser["valid"] & (riser["riser_distance"] <= 0.0)
    upper_tread_support = riser_plane_crossing & (
        wheel_bottom_z >= z_upper - cfg.clearance_margin
    ) & (wheel_forces_w.norm(dim=-1) >= 1.0)

    for name in (
        "true_precontact_clearance", "riser_plane_crossing", "upper_tread_support"
    ):
        latch_name = f"_perceptive_{name}_latched"
        value = locals()[name]
        if not hasattr(env, latch_name):
            setattr(env, latch_name, torch.zeros_like(value))
        latched = torch.where(reset[:, None], torch.zeros_like(value), getattr(env, latch_name))
        latched |= value
        setattr(env, latch_name, latched)

    flat_mask = (~riser["valid"]) & (riser["corridor_relief"] < cfg.flat_relief_threshold)
    false_excess = torch.relu(actual_lift - cfg.l_flat_tol)
    false_cost_wheel = (false_excess / cfg.false_lift_scale).square()
    false_cost = torch.where(flat_mask, false_cost_wheel, torch.zeros_like(false_cost_wheel)).mean(1)

    state = {
        **riser,
        "elevation_clean": elevation,
        "elevation_world": elevation_world,
        "wheel_base": wheel_b,
        "wheel_yaw": wheel_yaw,
        "z_nom": env._perceptive_z_nom,
        "actual_lift": actual_lift,
        "guide": guide,
        "active": active,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "z_nominal_world": z_nominal_world,
        "z_cross": z_cross,
        "z_preview": z_preview,
        "z_geom_required": z_geom_required,
        "z_ref_world": z_ref_world,
        "z_actual_world": z_actual_world,
        "z_actual_relative": z_actual_relative,
        "z_ref_relative": z_ref_relative,
        "wheel_bottom_z": wheel_bottom_z,
        "actual_geometric_lift": actual_geometric_lift,
        "final_required_lift": final_required_lift,
        "timing_progress": guide,
        "desired_lift_now": desired_lift_now,
        "progress": absolute_progress,
        "absolute_progress": absolute_progress,
        "old_progress": old_progress,
        "vertical_contact": vertical_contact,
        "vertical_contact_latched": vertical_contact_latched,
        "reward_active": reward_active,
        "true_precontact_clearance": true_precontact_clearance,
        "true_precontact_clearance_latched": env._perceptive_true_precontact_clearance_latched,
        "riser_plane_crossing": riser_plane_crossing,
        "riser_plane_crossing_latched": env._perceptive_riser_plane_crossing_latched,
        "upper_tread_support": upper_tread_support,
        "upper_tread_support_latched": env._perceptive_upper_tread_support_latched,
        "clamp_warning": torch.zeros_like(active),
        # Compatibility names for existing scalar logging. Semantics are now
        # world-Z clearance above theoretical lower-tread wheel-centre height.
        "lift_ref": desired_lift_now,
        "deficit": torch.relu(desired_lift_now - actual_geometric_lift),
        "spatial_raw": spatial_raw,
        # Spatial reward is gated only by detected riser geometry and the
        # continuous preview window. It is independent of terrain row index;
        # the former 8-entry lookup assumed a different legacy curriculum.
        "spatial_effective": spatial_raw,
        "false_excess": false_excess,
        "false_cost": false_cost,
    }
    # Metric tensors are simulation-only and never registered observations.
    env.perceptive_spatial_state = state
    env._perceptive_spatial_step = step
    env._perceptive_spatial_cache = state
    return state


def perceptive_spatial_lift(env, sensor_cfg=SceneEntityCfg("height_scanner"), asset_cfg=SceneEntityCfg("robot")):
    """Dense one-sided clearance reward from geometry, without row scaling."""
    return perceptive_spatial_state(env, sensor_cfg, asset_cfg)["spatial_effective"]


def perceptive_false_lift(env, sensor_cfg=SceneEntityCfg("height_scanner"), asset_cfg=SceneEntityCfg("robot")):
    """Non-negative flat-terrain excess-lift cost (configured with negative weight)."""
    return perceptive_spatial_state(env, sensor_cfg, asset_cfg)["false_cost"]
