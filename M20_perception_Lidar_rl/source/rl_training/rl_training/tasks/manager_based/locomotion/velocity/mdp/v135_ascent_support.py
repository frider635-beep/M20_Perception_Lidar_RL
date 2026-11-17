"""Ascent-only three-wheel support quality; no action gates or new policy inputs."""
from __future__ import annotations
import torch


def three_wheel_quality(force, workspace, swing_side, slip, pitch_error, roll_error,
                        angular_rate, ground_valid):
    """Expected stance = opposite front plus both rears, independent of swing contact.

    Continuous load scoring avoids a hard contact switch. Missing ground cannot
    certify posture. Quality only scales finite *extra* forward credit downstream.
    """
    stance = torch.ones_like(force, dtype=torch.bool)
    stance.scatter_(1, swing_side.clamp(0, 1)[:, None], False)
    load = (force.clamp_min(0) / 20.0).clamp(0, 1)
    support = load.masked_fill(~stance, 1).amin(1)
    work = workspace.clamp(0, 1).masked_fill(~stance, 1).amin(1)
    posture_error = torch.maximum(pitch_error, roll_error)
    posture = torch.exp(-((posture_error - 0.2617994).clamp_min(0) / 0.2617994).square())
    motion = torch.exp(-(slip / 0.5).square()
                       - ((angular_rate - 1.0).clamp_min(0) / 2.0).square())
    quality = support * work * posture * motion * ground_valid.float()
    return torch.nan_to_num(quality, nan=0., posinf=0., neginf=0.).clamp(0, 1), torch.nan_to_num(support, nan=0., posinf=0., neginf=0.)


def support_margin(points, contact, com_xy):
    """Signed distance to the measured XY support hull, not contact count.

    Only actual contact pairs bounding all remaining contacts form hull edges.
    Fewer than three non-collinear supports cannot certify a polygon.
    """
    margins = []
    for i in range(4):
        for j in range(i + 1, 4):
            edge = points[:, j] - points[:, i]
            norm = edge.norm(dim=-1).clamp_min(1e-6)
            offset = points - points[:, i:i+1]
            signed = (edge[:, None, 0] * offset[..., 1] - edge[:, None, 1] * offset[..., 0]) / norm[:, None]
            left = ((signed >= -1e-5) | ~contact).all(1)
            right = ((signed <= 1e-5) | ~contact).all(1)
            edge_valid = contact[:, i] & contact[:, j] & (left ^ right)
            offset_com = com_xy - points[:, i]
            d = (edge[:, 0] * offset_com[:, 1] - edge[:, 1] * offset_com[:, 0]) / norm
            d = torch.where(left, d, -d)
            margins.append(torch.where(edge_valid, d, torch.full_like(d, torch.inf)))
    margin = torch.stack(margins, 1).amin(1)
    valid = (contact.sum(1) >= 3) & torch.isfinite(margin)
    return torch.where(valid, margin, torch.zeros_like(margin)), valid


def measured_swing_side(contact, bottom, lower_z, first_side, seeded, *, lift_min):
    """No expected-side/argmax fallback: only an actual single lifted front."""
    lifted = ~contact[:, :2] & (bottom[:, :2] > lower_z[:, None] + lift_min)
    single = lifted.sum(1) == 1
    side = lifted.long().argmax(1)
    # The B1 first-attempt identity is immutable, including an invalid attempt.
    side_ok = seeded | (first_side < 0) | (side == first_side)
    return torch.where(single & side_ok, side, torch.full_like(side, -1))


def v135_ascent_support_state(env, sensor_cfg=None):
    from .v135_whole_body_descent import v135_whole_body_descent_state
    from .v135_riser_identity import v135_riser_identity_state
    whole = v135_whole_body_descent_state(env, sensor_cfg)
    ledger = v135_riser_identity_state(env, sensor_cfg)
    strict = ledger["reward_snapshot"]
    step = int(env.common_step_counter)
    cfg = env.cfg.v135_riser_identity
    if not hasattr(env, "_v135_ascent_support"):
        n, d = env.num_envs, env.device
        env._v135_ascent_support = dict(step=-1, episode=env.episode_length_buf.clone(),
            target=torch.full((n,), -1, device=d, dtype=torch.long),
            side=torch.full((n,), -1, device=d, dtype=torch.long),
            prep=torch.zeros((n, int(cfg.support_preparation_frames), 2), device=d),
            prep_count=torch.zeros(n, device=d, dtype=torch.long),
            prep_quality=torch.zeros(n, device=d),
            swing_sum=torch.zeros(n, device=d), swing_count=torch.zeros(n, device=d),
            accept_sum=torch.zeros(n, device=d), accept_count=torch.zeros(n, device=d),
            started=torch.zeros(n, device=d, dtype=torch.bool))
    s = env._v135_ascent_support
    if s["step"] == step:
        return s
    episode_reset = env.episode_length_buf < s["episode"]
    target_changed = s["target"] != strict["target_id"]
    reset = episode_reset | target_changed
    # Retain recent pre-acquisition evidence only for the first target. Never
    # carry an earlier riser's preparation into a new seeded target.
    prep_reset = episode_reset | (target_changed & strict["seeded"])
    s["prep"][prep_reset] = 0
    s["prep_count"][prep_reset] = 0
    for key in ("prep_quality", "swing_sum", "swing_count", "accept_sum", "accept_count", "started"):
        s[key][reset] = 0
    s["side"][reset] = -1
    asset = env.scene["robot"]
    force = env.scene.sensors["contact_forces"].data.net_forces_w[:, whole["contacts"], 2]
    contact = force > float(env.cfg.v11_perception.contact_force_threshold)
    wheel = asset.data.body_pos_w[:, whole["bodies"]]
    velocity = asset.data.body_lin_vel_w[:, whole["bodies"]]
    omega = asset.data.body_ang_vel_w[:, whole["bodies"]]
    radius = torch.zeros_like(velocity)
    radius[..., 2] = -float(env.cfg.v11_perception.wheel_radius)
    contact_velocity = velocity + torch.linalg.cross(omega, radius, dim=-1)
    bottom = wheel[..., 2] - float(env.cfg.v11_perception.wheel_radius)
    actual = measured_swing_side(contact, bottom, strict["target_lower_z"],
        strict["first_swing_side"], strict["seeded"], lift_min=float(cfg.first_swing_lift_min))
    active = strict["target_valid"] & strict["target_height_valid"]
    from .v135_terrain_geometry import v135_terrain_geometry
    ascent = v135_terrain_geometry(env)["ascent"]
    active &= ascent
    start = active & ~s["started"] & (actual >= 0)
    s["side"] = torch.where(start, actual, s["side"])
    # Known generated stairs have no cross-slope. Split tread heights must not
    # manufacture a desired body roll. This is ascent-only.
    forward = strict["forward_world_xy"]
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    w,x,y,z = asset.data.root_quat_w.unbind(-1)
    body_lateral = torch.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)),1)
    roll_error = torch.atan2(body_lateral[:,2], (body_lateral[:,:2]*lateral).sum(1)).abs()
    angular = torch.linalg.vector_norm(asset.data.root_ang_vel_w[:,:2],dim=1)
    if "masses" not in s:
        s["masses"] = asset.root_physx_view.get_masses().to(env.device)
    mass = s["masses"]
    # Use inertial COMs of all links, not root position or wheel centroid.
    com = (asset.data.body_com_pos_w * mass[..., None]).sum(1) / mass.sum(1, keepdim=True)
    candidates, supports, slips, margins = [], [], [], []
    for candidate in range(2):
        side = torch.full_like(actual, candidate)
        stance = contact.clone(); stance[:, candidate] = False
        slip = (contact_velocity[..., :2].norm(dim=-1) * stance).sum(1) / stance.sum(1).clamp_min(1)
        margin, polygon_valid = support_margin(wheel[..., :2], stance, com[:, :2])
        q, load = three_wheel_quality(force, whole["workspace"], side, slip,
            whole["pitch_error"], roll_error, angular, whole["ground_valid"])
        q *= torch.exp(-(margin.clamp_max(0) / float(cfg.support_margin_scale)).square()) * polygon_valid
        candidates.append(q); supports.append(load); slips.append(slip); margins.append(margin)
    candidate_q = torch.stack(candidates, 1)
    selected = s["side"].clamp_min(0)[:, None]
    prep = s["prep"].sum(1) / s["prep_count"].clamp_min(1)[:, None]
    pre_q = prep.gather(1, selected).squeeze(1)
    s["prep_quality"] = torch.where(start, pre_q, s["prep_quality"])
    s["started"] |= start
    swing = active & s["started"] & ~contact[:, :2].gather(1, selected).squeeze(1)
    # Include landing/load acceptance while the selected foot is supported;
    # no tripod demand during a rear swing or ordinary inter-riser rolling.
    acceptance = active & s["started"] & contact[:, :2].gather(1, selected).squeeze(1) & (strict["locked_support_steps"].gather(1, selected).squeeze(1) > 0)
    q = candidate_q.gather(1, selected).squeeze(1)
    all_margin, all_valid = support_margin(wheel[..., :2], contact, com[:, :2])
    acceptance_slip = (contact_velocity[..., :2].norm(dim=-1) * contact).sum(1) / contact.sum(1).clamp_min(1)
    accepted_q = (whole["workspace"].masked_fill(~contact, 1).amin(1) * whole["ground_valid"] * all_valid
        * torch.exp(-(all_margin.clamp_max(0) / float(cfg.support_margin_scale)).square())
        * torch.exp(-((roll_error - .2617994).clamp_min(0) / .2617994).square())
        * torch.exp(-((angular - 1.).clamp_min(0) / 2.).square())
        * torch.exp(-(acceptance_slip / .5).square())
        * torch.exp(-((whole["pitch_error"] - .2617994).clamp_min(0) / .2617994).square()))
    q = torch.where(acceptance, accepted_q, q)
    measured = swing | acceptance
    q = torch.nan_to_num(q, nan=0., posinf=0., neginf=0.).clamp(0,1)
    # Preparation, swing and acceptance each retain their own evidence. Waiting
    # on the upper tread cannot dilute a poor load-transfer/swing segment.
    s["swing_sum"] += q * swing
    s["swing_count"] += swing
    s["accept_sum"] = torch.where(acceptance, s["accept_sum"], torch.zeros_like(s["accept_sum"]))
    s["accept_count"] = torch.where(acceptance, s["accept_count"], torch.zeros_like(s["accept_count"]))
    accept_sample = acceptance & (s["accept_count"] < int(env.cfg.v11_perception.front_support_confirmation_frames))
    s["accept_sum"] += q * accept_sample
    s["accept_count"] += accept_sample
    settlement = torch.minimum(s["prep_quality"], torch.minimum(
        s["swing_sum"] / s["swing_count"].clamp_min(1),
        s["accept_sum"] / s["accept_count"].clamp_min(1)))
    # Before first lift, keep only the recent preparation window, no payoff.
    preparing = ascent & ~s["started"] & contact[:, :2].all(1)
    # A broken contact interval must not leave stale preparation in the buffer.
    clear_prep = ~preparing & ~s["started"]
    s["prep"][clear_prep] = 0
    s["prep_count"][clear_prep] = 0
    s["prep"] = torch.where(preparing[:, None, None], torch.cat((s["prep"][:, 1:], candidate_q[:, None]), 1), s["prep"])
    s["prep_count"] = torch.where(preparing, (s["prep_count"] + 1).clamp_max(s["prep"].shape[1]), s["prep_count"])
    s.update(quality=torch.where(measured, q, torch.zeros_like(q)),
        settlement_quality=settlement.clamp(0,1),
        settlement_evidence=(s["prep_count"] > 0) & (s["swing_count"] > 0) & (s["accept_count"] > 0),
        missing_preparation=s["prep_count"] == 0,
        missing_swing=s["swing_count"] == 0,
        missing_acceptance=s["accept_count"] == 0, measured=measured, swing=swing, acceptance=acceptance,
        preparation=preparing, support=torch.stack(supports,1).gather(1,selected).squeeze(1),
        slip=torch.stack(slips,1).gather(1,selected).squeeze(1),
        support_margin=torch.where(acceptance, all_margin, torch.stack(margins,1).gather(1,selected).squeeze(1)),
        roll_error=roll_error, pitch_error=whole["pitch_error"], ground_valid=whole["ground_valid"],
        step=step, episode=env.episode_length_buf.clone(), target=strict["target_id"].clone())
    return s
