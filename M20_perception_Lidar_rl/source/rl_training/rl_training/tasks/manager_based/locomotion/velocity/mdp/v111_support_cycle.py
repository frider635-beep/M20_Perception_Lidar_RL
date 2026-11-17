"""V11.1 reward-only support timing and persistent level-ground posture.

No action clipping or hidden controller: the exported actor remains authoritative.
"""
from __future__ import annotations

import math
import torch


def prepare_cycle_risers(env, asset, risers, landing, reset):
    """Keep the selected tread's geometry while its support dwell is pending."""
    q = asset.data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2].square() + q[:, 3].square()))
    c, s = yaw.cos()[:, None], yaw.sin()[:, None]
    root = asset.data.root_pos_w[:, :2]
    world = torch.stack((root[:, 0:1] + c * risers["riser_x"] - s * risers["riser_y"],
                         root[:, 1:2] + s * risers["riser_x"] + c * risers["riser_y"]), -1)
    cache = getattr(env, "_v111_cycle_risers", None)
    out = {k: v.clone() for k, v in risers.items()}
    if cache is not None and "cycle_leader" in landing:
        hold = ~reset & (landing["cycle_leader"] >= 0) & ~landing["cycle_done"] & ~landing["cycle_bad"]
        world = torch.where(hold[:, None, None], cache["world"], world)
        delta = world - root[:, None]
        out["riser_x"] = c * delta[..., 0] + s * delta[..., 1]
        out["riser_y"] = -s * delta[..., 0] + c * delta[..., 1]
        for key in ("valid", "riser_height", "riser_lower_world"):
            out[key] = torch.where(hold[:, None], cache[key], out[key])
    env._v111_cycle_risers = {"world": world.detach().clone(),
        **{k: out[k].clone() for k in ("valid", "riser_height", "riser_lower_world")}}
    return out


def cycle_step(state, upper, lower, active, dt, cfg):
    """Advance one same-tread handoff; upper/lower are current physical support."""
    n, device = active.shape[0], active.device
    if "cycle_leader" not in state:
        state.update(cycle_leader=torch.full((n,), -1, device=device, dtype=torch.long),
                     cycle_lead_steps=torch.zeros(n, device=device, dtype=torch.long),
                     cycle_pair_steps=torch.zeros(n, device=device, dtype=torch.long),
                     cycle_loss_steps=torch.zeros(n, device=device, dtype=torch.long),
                     cycle_age=torch.zeros(n, device=device, dtype=torch.long),
                     cycle_ready=torch.zeros(n, device=device, dtype=torch.bool),
                     cycle_bad=torch.zeros(n, device=device, dtype=torch.bool),
                     cycle_done=torch.zeros(n, device=device, dtype=torch.bool),
                     cycle_best=torch.zeros(n, device=device))
    lead_frames = max(1, math.ceil(cfg.leader_support_seconds / dt))
    pair_frames = max(1, math.ceil(cfg.pair_support_seconds / dt))
    loss_frames = max(1, math.ceil(cfg.support_loss_seconds / dt))
    # A leader can be acquired only with its mate still supported below.
    choices = upper & lower.flip(1) & active[:, None]
    acquire = (state["cycle_leader"] < 0) & (choices.sum(1) == 1)
    state["cycle_leader"] = torch.where(acquire, choices.long().argmax(1), state["cycle_leader"])
    index = state["cycle_leader"].clamp_min(0)[:, None]
    lead = upper.gather(1, index).squeeze(1)
    mate_lower = lower.gather(1, 1 - index).squeeze(1)
    selected = active & (state["cycle_leader"] >= 0) & ~state["cycle_done"]
    state["cycle_bad"] |= active & (state["cycle_leader"] < 0) & upper.all(1)
    was_ready = state["cycle_ready"].clone()
    state["cycle_age"] = torch.where(selected, state["cycle_age"] + 1, state["cycle_age"])
    state["cycle_loss_steps"] = torch.where(selected & ~lead, state["cycle_loss_steps"] + 1, 0)
    # Leaving the lower tread before the leader's dwell is over invalidates
    # this cycle. Physical bilateral completion still permits later resync.
    state["cycle_bad"] |= selected & ((~was_ready & ~mate_lower)
        | (state["cycle_loss_steps"] >= loss_frames))
    dwell = selected & ~state["cycle_bad"] & lead & mate_lower & ~was_ready
    state["cycle_lead_steps"] = torch.where(dwell, state["cycle_lead_steps"] + 1,
        torch.where(was_ready, state["cycle_lead_steps"], 0))
    state["cycle_ready"] |= state["cycle_lead_steps"] >= lead_frames
    together = selected & was_ready & ~state["cycle_bad"] & upper.all(1)
    state["cycle_pair_steps"] = torch.where(together, state["cycle_pair_steps"] + 1, 0)
    event = together & (state["cycle_pair_steps"] >= pair_frames) & ~state["cycle_done"]
    state["cycle_done"] |= event
    progress = 0.35 * (state["cycle_lead_steps"].float() / lead_frames).clamp(0, 1)
    progress += 0.65 * (state["cycle_pair_steps"].float() / pair_frames).clamp(0, 1)
    progress = torch.where(state["cycle_done"], torch.ones_like(progress), progress)
    progress *= (active & ~state["cycle_bad"]).float()
    best = torch.maximum(state["cycle_best"], progress)
    state["cycle_delta"] = best - state["cycle_best"]
    state["cycle_best"] = best
    state["cycle_event"] = event
    state["cycle_holding"] = selected & ~state["cycle_bad"] & lead & (
        ~was_ready | upper.all(1)) & ~state["cycle_done"] & (
        state["cycle_age"] * dt <= cfg.hold_grace_seconds)
    return state


def update_cycle(env, asset, landing, support_now, forces, lower_z, before):
    cfg = env.cfg.v111_rear_posture
    ids = env._v11_front_body_ids
    velocity = asset.data.body_lin_vel_w[:, ids]
    angular = asset.data.body_ang_vel_w[:, ids]
    radius = torch.zeros_like(velocity)
    radius[..., 2] = -float(env.cfg.v11_perception.wheel_radius)
    slip = torch.linalg.vector_norm((velocity + torch.linalg.cross(angular, radius, dim=-1))[..., :2], dim=-1)
    contact = forces[..., 2] >= cfg.cycle_contact_force
    quiet = slip <= cfg.cycle_slip_limit
    # Partial support feedback must precede the strict clearance candidate:
    # the resumed policy can reach the tread without yet clearing the full
    # safety margin. A riser collision still never earns support credit.
    upper = support_now & contact & quiet & landing["crossing"] & ~landing["vertical_contact"]
    bottom = asset.data.body_pos_w[:, ids, 2] - float(env.cfg.v11_perception.wheel_radius)
    lower = before & contact & quiet & ((bottom - lower_z[:, None]).abs() <= cfg.cycle_plane_tolerance)
    cycle_step(landing, upper, lower, landing["target_active"], float(env.step_dt), cfg)
    # Only upward progress to a new reference replenishes the bounded reward
    # budget. A scanner loss/reacquisition must not repay the same tread.
    heading = env.command_manager.get_term("base_velocity").v11_path_heading_w
    route = torch.stack((heading.cos(), heading.sin()), 1)
    position = (landing["reference_riser_world_xy"] * route).sum(1)
    if "cycle_paid_position" not in landing:
        landing["cycle_paid_position"] = torch.full_like(position, -torch.inf)
        landing["cycle_paid_best"] = torch.zeros_like(position)
    newer = landing["target_active"] & (position > landing["cycle_paid_position"] + float(env.cfg.v11_perception.encounter_riser_switch_distance))
    landing["cycle_paid_position"] = torch.where(newer, position, landing["cycle_paid_position"])
    landing["cycle_paid_best"] = torch.where(newer, torch.zeros_like(position), landing["cycle_paid_best"])
    same = (position - landing["cycle_paid_position"]).abs() < float(env.cfg.v11_perception.encounter_riser_switch_distance)
    clearance = ((landing["crossing_wheel_center_above_tread"] - float(env.cfg.v11_perception.wheel_radius))
                 / float(env.cfg.v11_perception.clearance_margin)).clamp(0, 1)
    lead_quality = clearance.gather(1, landing["cycle_leader"].clamp_min(0)[:, None]).squeeze(1)
    lead_fraction = (landing["cycle_lead_steps"].float() * float(env.step_dt) / cfg.leader_support_seconds).clamp(0, 1)
    pair_fraction = (landing["cycle_pair_steps"].float() * float(env.step_dt) / cfg.pair_support_seconds).clamp(0, 1)
    # Stable support itself supplies a small reachable recovery signal even
    # after an earlier airborne mistake; good clearance increases its value.
    # The follower/pair portion and full completion still require a clean cycle.
    progress = .35 * lead_fraction * (cfg.support_credit_floor + (1 - cfg.support_credit_floor) * lead_quality)
    progress += .65 * pair_fraction * clearance.amin(1) * (~landing["unsupported_handoff"]).float()
    progress *= (landing["target_active"] & ~landing["cycle_bad"]).float()
    delta = (progress - landing["cycle_paid_best"]).clamp_min(0)
    eligible = same & ~landing["adjacent_credit_blocked"]
    landing["cycle_delta"] = delta * eligible.float()
    landing["cycle_paid_best"] = torch.maximum(landing["cycle_paid_best"], torch.where(same, progress, 0))


def cycle_drive(env):
    from .v11_continuous import v11_state
    s = v11_state(env)["landing"]
    cfg = env.cfg.v111_rear_posture
    active = s["target_active"] & ~s.get("cycle_done", torch.zeros_like(s["target_active"]))
    holding = s.get("cycle_holding", torch.zeros_like(active))
    factor = torch.where(active, cfg.cycle_drive_floor, 1.0)
    factor = torch.where(holding, cfg.hold_drive_scale, factor)
    return factor, holding


def support_cycle_reward(env):
    from .v11_continuous import v11_state
    s = v11_state(env)["landing"]
    return s.get("cycle_delta", torch.zeros(env.num_envs, device=env.device)) / float(env.step_dt)


def landing_cycle_reward(env):
    from .v11_continuous import v11_state
    # Unlike the legacy pulse, this is an event budget (not multiplied by dt twice).
    return v11_state(env)["landing"]["bilateral_landing_event"].float() / float(env.step_dt)


def flat_geometry(wheel, root, hits, forward, no_riser, cfg):
    finite = torch.isfinite(hits).all(-1)
    distances = ((wheel[:, :, None, :2] - hits[:, None, :, :2]) ** 2).sum(-1)
    distances = distances.masked_fill(~finite[:, None], torch.inf)
    distance, nearest = distances.min(-1)
    ground = hits[..., 2].gather(1, nearest)
    near = (distance < cfg.flat_sample_radius ** 2).all(1) & torch.isfinite(ground).all(1)
    offset = hits[..., :2] - root[:, None, :2]
    x = (offset * forward[:, None]).sum(-1)
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    y = (offset * lateral[:, None]).sum(-1)
    corridor = finite & (x >= 0) & (x <= cfg.flat_lookahead) & (y.abs() <= cfg.flat_half_width)
    coverage = (corridor & (x < cfg.flat_lookahead * .25)).any(1) & (corridor & (x > cfg.flat_lookahead * .75)).any(1)
    zmax = torch.maximum(hits[..., 2].masked_fill(~corridor, -torch.inf).amax(1), ground.amax(1))
    zmin = torch.minimum(hits[..., 2].masked_fill(~corridor, torch.inf).amin(1), ground.amin(1))
    evidence = no_riser & near & coverage & ((zmax - zmin) <= cfg.flat_plane_tolerance)
    return evidence, torch.nan_to_num(ground.mean(1))


def flat_posture_cost(clearance, gravity_xy, rate_xy, cfg):
    height = ((cfg.flat_clearance_min - clearance).clamp_min(0) / cfg.flat_clearance_scale).square().clamp_max(1)
    tilt = ((gravity_xy - cfg.flat_tilt_free).clamp_min(0) / cfg.flat_tilt_scale).square().clamp_max(1)
    rocking = ((rate_xy - cfg.flat_rate_free).clamp_min(0) / cfg.flat_rate_scale).square().clamp_max(1)
    return height + tilt + .25 * rocking


def flat_state(env):
    from .v11_continuous import v11_state
    from .v111_exit_evidence import riser_masks
    base = v11_state(env)
    cfg, robot = env.cfg.v111_rear_posture, env.scene["robot"]
    step = int(env.common_step_counter)
    if not hasattr(env, "_v111_level_posture"):
        ids = [robot.find_bodies(name)[0][0] for name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")]
        env._v111_level_posture = dict(step=-1, ids=ids, last=env.episode_length_buf.clone(),
            frames=torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    s = env._v111_level_posture
    if s["step"] == step:
        return s
    wheel = robot.data.body_pos_w[:, s["ids"]]
    heading = env.command_manager.get_term("base_velocity").v11_path_heading_w
    forward = torch.stack((heading.cos(), heading.sin()), 1)
    no_riser, _ = riser_masks(base["riser_pool_world_xy"], base["riser_pool"]["valid"],
        wheel[..., :2], robot.data.root_pos_w[:, :2], forward, env.cfg.v11_perception.wheel_radius)
    evidence, ground = flat_geometry(wheel, robot.data.root_pos_w,
        env.scene.sensors["v10_pgtt_scanner"].data.ray_hits_w, forward, no_riser, cfg)
    reset = env.episode_length_buf < s["last"]
    s["frames"][reset] = 0
    s["frames"] = torch.where(evidence, s["frames"] + 1, 0)
    blend = (s["frames"].float() * float(env.step_dt) / cfg.flat_confirmation_seconds).clamp(0, 1)
    clearance = robot.data.root_pos_w[:, 2] - ground
    tilt = torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1)
    rate = torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=1)
    cost = torch.nan_to_num(flat_posture_cost(clearance, tilt, rate, cfg), nan=0., posinf=0.)
    s.update(step=step, last=env.episode_length_buf.clone(), blend=blend, cost=cost,
             clearance=clearance, tilt=tilt, active=evidence)
    return s


def level_posture_penalty(env):
    s = flat_state(env)
    return -s["cost"] * s["blend"]


def cycle_metric(env, env_ids, key):
    from .v11_continuous import v11_state
    s = v11_state(env)["landing"]
    value = s.get(key, torch.zeros(env.num_envs, device=env.device)).float()
    return value[env_ids].mean()


def level_metric(env, env_ids, key):
    s = flat_state(env)
    if key == "active":
        return s["active"][env_ids].float().mean()
    mask = s["active"][env_ids].float()
    return (s[key][env_ids] * mask).sum() / mask.sum().clamp_min(1)
