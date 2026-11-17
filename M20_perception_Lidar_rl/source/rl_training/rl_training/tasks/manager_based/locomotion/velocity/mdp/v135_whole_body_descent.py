"""Whole-body descent diagnostics and soft credit quality, before front touchdown."""
from __future__ import annotations
import torch


def workspace_quality(relative, neutral, *, free, scale):
    """Signed hip-to-wheel displacement catches both rear reach and folding."""
    length = torch.linalg.vector_norm(neutral, dim=-1).clamp_min(0.1)
    error = torch.linalg.vector_norm(relative - neutral, dim=-1) / length
    return torch.exp(-((error - free).clamp_min(0) / scale).square())


def v135_whole_body_descent_state(env, sensor_cfg=None):
    from isaaclab.utils.math import quat_apply_inverse
    from .v11_continuous import v11_state
    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)
    asset = env.scene["robot"]
    cfg = env.cfg.v135_descent_support_transfer
    step = int(env.common_step_counter)
    if not hasattr(env, "_v135_whole_body"):
        bodies, contacts, hips = [], [], []
        for leg in ("fl", "fr", "hl", "hr"):
            for obj, name, out in ((asset, leg + "_wheel", bodies),
                                   (env.scene.sensors["contact_forces"], leg + "_wheel", contacts),
                                   (asset, leg + "_hipy", hips)):
                ids, _ = obj.find_bodies(name)
                if len(ids) != 1:
                    raise RuntimeError(f"Expected one {name}, got {ids}")
                out.append(ids[0])
        env._v135_whole_body = dict(step=-1, bodies=bodies, contacts=contacts, hips=hips,
            episode=env.episode_length_buf.clone(),
            contact_age=torch.full((env.num_envs, 4), 100, device=env.device, dtype=torch.long))
    s = env._v135_whole_body
    if s["step"] == step:
        return s
    wheel = asset.data.body_pos_w[:, s["bodies"]]
    hip = asset.data.body_pos_w[:, s["hips"]]
    quat = asset.data.root_quat_w[:, None, :].expand(-1, 4, -1)
    relative = quat_apply_inverse(quat.reshape(-1, 4), (wheel - hip).reshape(-1, 3)).reshape(-1, 4, 3)
    reset = env.episode_length_buf < s["episode"]
    if "neutral" not in s:
        s["neutral"] = relative.detach().clone()
        s["neutral_clearance"] = (asset.data.root_pos_w[:, 2] - wheel[..., 2].mean(1)
                                  + float(cfg.wheel_radius)).detach().clamp_min(0.1)
    s["neutral"][reset] = relative[reset].detach()
    reset_clearance = asset.data.root_pos_w[:, 2] - wheel[..., 2].mean(1) + float(cfg.wheel_radius)
    s["neutral_clearance"][reset] = reset_clearance[reset].detach().clamp_min(0.1)
    s["contact_age"][reset] = 100
    force = env.scene.sensors["contact_forces"].data.net_forces_w[:, s["contacts"], 2]
    contact = force > float(cfg.contact_force_threshold)
    s["contact_age"] = torch.where(contact, torch.zeros_like(s["contact_age"]), s["contact_age"] + 1)
    recent = s["contact_age"] < int(cfg.stable_frames)
    # Two recent supports suffice during transfer; no prescribed front/rear load ratio.
    support = (recent.sum(1).float() / 2).clamp(0, 1)
    support *= contact.any(1).float()
    support *= (recent[:, :2].any(1) & recent[:, 2:].any(1)).float()
    scanner = env.scene.sensors[sensor_cfg.name if sensor_cfg is not None else "v10_pgtt_scanner"]
    hits = scanner.data.ray_hits_w
    finite = torch.isfinite(hits).all(-1)
    distances = (wheel[:, :, None, :2] - hits[:, None, :, :2]).square().sum(-1)
    distances = distances.masked_fill(~finite[:, None, :], float("inf"))
    distance, nearest = distances.min(-1)
    ground = hits[..., 2].gather(1, nearest)
    ground_valid = (distance < float(cfg.ground_sample_radius) ** 2).all(1) & torch.isfinite(ground).all(1)
    ground = torch.nan_to_num(ground)
    heading = env.command_manager.get_term("base_velocity").v11_path_heading_w
    forward = torch.stack((heading.cos(), heading.sin()), 1)
    local_x = ((wheel[..., :2] - asset.data.root_pos_w[:, None, :2]) * forward[:, None]).sum(-1)
    x_centered = local_x - local_x.mean(1, keepdim=True)
    slope = (x_centered * (ground - ground.mean(1, keepdim=True))).sum(1) / x_centered.square().sum(1).clamp_min(0.01)
    # Body's world forward axis projected into route vertical plane (yaw-invariant).
    w, x, y, z = asset.data.root_quat_w.unbind(-1)
    body_axis = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y)), 1)
    body_angle = torch.atan2(body_axis[:, 2], (body_axis[:, :2] * forward).sum(1))
    pitch_error = (body_angle - torch.atan(slope)).abs()
    pitch = torch.exp(-((pitch_error - float(cfg.terrain_pitch_free)).clamp_min(0) / float(cfg.terrain_pitch_scale)).square())
    clearance_ratio = (asset.data.root_pos_w[:, 2] - ground.mean(1)) / s["neutral_clearance"]
    height = torch.exp(-((float(cfg.clearance_floor) - clearance_ratio).clamp_min(0) / float(cfg.clearance_quality_scale)).square())
    work = workspace_quality(relative, s["neutral"], free=float(cfg.workspace_free), scale=float(cfg.workspace_scale))
    velocity = asset.data.body_lin_vel_w[:, s["bodies"]]
    angular = asset.data.body_ang_vel_w[:, s["bodies"]]
    radius = torch.zeros_like(velocity); radius[..., 2] = -float(cfg.wheel_radius)
    contact_velocity = velocity + torch.linalg.cross(angular, radius, dim=-1)
    slip = (torch.linalg.vector_norm(contact_velocity[..., :2], dim=-1) * contact).sum(1) / contact.sum(1).clamp_min(1)
    rolling = torch.exp(-(slip / float(cfg.rolling_slip_scale)).square())
    relative_speed = ((asset.data.root_lin_vel_w[:, None, :2] - velocity[..., :2]) * forward[:, None]).sum(-1)
    lagging = (relative_speed.clamp_min(0) * contact).amax(1)
    transport = torch.exp(-(lagging / float(cfg.relative_speed_scale)).square())
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    rate = (asset.data.root_ang_vel_w[:, :2] * lateral).sum(1).abs()
    angular_quality = torch.exp(-((rate - float(cfg.pitch_rate_free)).clamp_min(0) / float(cfg.pitch_rate_scale)).square())
    # Missing ground evidence cannot certify good posture; expose it separately.
    quality = torch.minimum(torch.minimum(pitch, height), work.amin(1))
    quality = quality * support * torch.sqrt(rolling * transport * angular_quality)
    quality *= ground_valid.float()
    negative_edge = ((base["raw_feature"][:, [0, 3]] > 0.5) & (base["raw_feature"][:, [1, 4]] < -0.03)).any(1)
    active = base["descent"]["target_active"] | negative_edge
    length = torch.linalg.vector_norm(s["neutral"], dim=-1).clamp_min(0.1)
    s["leg_length_ratio"] = torch.linalg.vector_norm(relative, dim=-1) / length
    s["signed_rear_reach"] = (s["neutral"][:, 2:, 0] - relative[:, 2:, 0]) / length[:, 2:]
    s.update(quality=quality, active=active, ground_valid=ground_valid, workspace=work,
             slip=slip, pitch_error=pitch_error, clearance_ratio=clearance_ratio,
             support=support, step=step, episode=env.episode_length_buf.clone())
    return s
