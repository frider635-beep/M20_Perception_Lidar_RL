"""V12 local port of the V9 teacher's map-based stair rewards.

These functions deliberately consume the training-only ``v10_pgtt_scanner``.
The scanner is never registered in the V12 Actor observation group: deployed
policies retain the V11 57D proprioception plus six geometry values.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import yaw_quat

from .v11_continuous import v11_state


def _bounded_squared_penalty(excess: torch.Tensor) -> torch.Tensor:
    """Map a non-negative normalized error smoothly to at most one."""
    return -torch.expm1(-excess.square())


def v12_v9_action_term_mirror(
    env,
    action_term_name: str,
    mirror_joint_names: list[list[str]],
) -> torch.Tensor:
    """V9 bilateral action-magnitude penalty using action-term-local indices."""
    action_term = env.action_manager.get_term(action_term_name)
    cache_name = f"_v12_v9_{action_term_name}_mirror_indices"
    mirror_indices = getattr(env, cache_name, None)
    if mirror_indices is None:
        action_joint_names = list(action_term._joint_names)
        mirror_indices = []
        for first_pattern, second_pattern in mirror_joint_names:
            _, first_names = action_term._asset.find_joints(first_pattern, preserve_order=True)
            _, second_names = action_term._asset.find_joints(second_pattern, preserve_order=True)
            if len(first_names) != len(second_names):
                raise ValueError(
                    f"V12 V9 mirror patterns {first_pattern!r}/{second_pattern!r} "
                    f"resolved to {len(first_names)}/{len(second_names)} joints."
                )
            mirror_indices.append(
                (
                    [action_joint_names.index(name) for name in first_names],
                    [action_joint_names.index(name) for name in second_names],
                )
            )
        setattr(env, cache_name, mirror_indices)

    costs = [
        torch.mean(
            torch.square(
                torch.abs(action_term.raw_actions[:, first_ids])
                - torch.abs(action_term.raw_actions[:, second_ids])
            ),
            dim=1,
        )
        for first_ids, second_ids in mirror_indices
    ]
    return torch.stack(costs, dim=1).mean(dim=1) if costs else torch.zeros(env.num_envs, device=env.device)


def v12_second_front_completion_reward(
    env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """One pulse when the second front wheel completes clearance/crossing/support."""
    return v11_state(env, sensor_cfg=sensor_cfg)["landing"]["bilateral_landing_event"].float()


def v12_second_front_clearance_settlement_reward(
    env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """Release the second front wheel's forward-only clearance escrow."""
    settlement = v11_state(env, sensor_cfg=sensor_cfg)["landing"]["second_clearance_settlement"]
    return settlement / float(env.step_dt)


def v12_bilateral_wheel_height_difference(
    env,
    tolerance: float = 0.04,
    height_scale: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """Bound FL/FR and HL/HR wheel-height mismatch throughout the stair encounter."""
    landing = v11_state(env, sensor_cfg=sensor_cfg)["landing"]
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    mismatch = torch.stack(((heights[:, 0] - heights[:, 1]).abs(), (heights[:, 2] - heights[:, 3]).abs()), dim=1)
    excess = (mismatch - float(tolerance)).clamp_min(0.0) / max(float(height_scale), 1.0e-6)
    active = landing["reference_valid"] & ~landing["platform_confirmed"]
    return excess.square().mean(dim=1) * active.float()


def v12_front_phase_time_gap(
    env,
    phase: str,
    tolerance_steps: int = 4,
    scale_steps: int = 10,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("v10_pgtt_scanner"),
) -> torch.Tensor:
    """Bound the penalty for elapsed steps between bilateral milestones."""
    if phase not in ("crossing", "support"):
        raise ValueError(f"Unknown V12 bilateral phase: {phase!r}")
    landing = v11_state(env, sensor_cfg=sensor_cfg)["landing"]
    gap = landing[f"{phase}_gap_steps"].float()
    excess = (gap - float(tolerance_steps)).clamp_min(0.0) / max(float(scale_steps), 1.0)
    return _bounded_squared_penalty(excess) * landing["target_active"].float()


def _height_scan_variation(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hit_z = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    return hit_z.max(dim=1).values - hit_z.min(dim=1).values


def _terrain_height_under_bodies(env, sensor_cfg: SceneEntityCfg, asset: Articulation, body_ids) -> torch.Tensor:
    """V9 nearest-ray terrain lookup, kept local for deterministic comparison."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_w = torch.nan_to_num(sensor.data.ray_hits_w, nan=0.0, posinf=0.0, neginf=0.0)
    yaw = yaw_quat(asset.data.root_quat_w)
    rel_hits_w = hits_w - asset.data.root_pos_w.unsqueeze(1)
    rel_hits_b = math_utils.quat_apply_inverse(
        yaw.unsqueeze(1).expand(-1, rel_hits_w.shape[1], -1).reshape(-1, 4), rel_hits_w.reshape(-1, 3)
    ).reshape(env.num_envs, rel_hits_w.shape[1], 3)
    rel_body_w = asset.data.body_pos_w[:, body_ids, :] - asset.data.root_pos_w.unsqueeze(1)
    rel_body_b = math_utils.quat_apply_inverse(
        yaw.unsqueeze(1).expand(-1, rel_body_w.shape[1], -1).reshape(-1, 4), rel_body_w.reshape(-1, 3)
    ).reshape(env.num_envs, rel_body_w.shape[1], 3)
    nearest = torch.square(rel_body_b[:, :, None, :2] - rel_hits_b[:, None, :, :2]).sum(dim=-1).argmin(dim=-1)
    return torch.gather(hits_w[..., 2], dim=1, index=nearest)


def _rough_gate(env, sensor_cfg: SceneEntityCfg, threshold: float, scale: float) -> torch.Tensor:
    return torch.sigmoid((_height_scan_variation(env, sensor_cfg) - threshold) / scale)


def v12_track_lin_vel_y_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """V9's independent lateral velocity tracking score."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = (env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1]).square()
    return torch.exp(-error / std**2)


def v12_base_lin_vel_xy_l2_without_cmd(
    env, command_name: str, command_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """V9 inherited stationary-body xy drift penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return asset.data.root_lin_vel_b[:, :2].square().sum(dim=1) * (command[:, :3].norm(dim=1) < command_threshold)


def v12_base_ang_vel_z_l2_without_cmd(
    env, command_name: str, command_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """V9 inherited stationary-body yaw drift penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return asset.data.root_ang_vel_b[:, 2].square() * (command[:, :3].norm(dim=1) < command_threshold)


def v12_lin_vel_xy_l2_with_ang_z_command(
    env, command_name: str, command_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """V9 inherited suppression of translation during a yaw-only command."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    yaw_only = (command[:, 2].square() > command_threshold) & (command[:, :2].square().sum(dim=1) < command_threshold)
    return asset.data.root_lin_vel_b[:, :2].square().sum(dim=1) * yaw_only


def v12_non_wheel_contact_force(env, sensor_cfg: SceneEntityCfg, force_threshold: float = 18.0, force_scale: float = 180.0) -> torch.Tensor:
    """V9 non-wheel-link contact cost."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).amax(dim=1)
    return ((force - force_threshold).clamp_min(0.0) / force_scale).square().sum(dim=1)


def v12_non_wheel_scrape(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 8.0, speed_threshold: float = 0.08, speed_scale: float = 0.55,
) -> torch.Tensor:
    """V9 non-wheel-link sliding-contact cost."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contact = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).amax(dim=1) > force_threshold
    speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
    return (((speed - speed_threshold).clamp_min(0.0) / speed_scale).square() * contact.float()).sum(dim=1)


def v12_front_step_approach_clearance(
    env, command_name: str, terrain_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_clearance: float = 0.14, wheel_radius: float = 0.09, top_clearance_margin: float = 0.015,
    step_height_threshold: float = 0.025, step_height_scale: float = 0.018, command_threshold: float = 0.12,
    near_x_range: tuple[float, float] = (-0.30, 0.18), front_x_range: tuple[float, float] = (0.28, 0.90),
    y_abs_max: float = 0.48,
) -> torch.Tensor:
    """Exact V9 front-riser clearance cost, evaluated from V11's PGTT map."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[terrain_sensor_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    forward_gate = torch.sigmoid((command_x - command_threshold) / 0.05)
    hits_w = torch.nan_to_num(sensor.data.ray_hits_w, nan=0.0, posinf=0.0, neginf=0.0)
    rel_w = hits_w - asset.data.root_pos_w.unsqueeze(1)
    yaw = yaw_quat(asset.data.root_quat_w)
    rel_b = math_utils.quat_apply_inverse(
        yaw.unsqueeze(1).expand(-1, rel_w.shape[1], -1).reshape(-1, 4), rel_w.reshape(-1, 3)
    ).reshape(env.num_envs, rel_w.shape[1], 3)
    x, y_abs, z = rel_b[..., 0], rel_b[..., 1].abs(), hits_w[..., 2]
    near = (x >= near_x_range[0]) & (x <= near_x_range[1]) & (y_abs <= y_abs_max)
    front = (x >= front_x_range[0]) & (x <= front_x_range[1]) & (y_abs <= y_abs_max)
    missing = torch.full_like(z, -1.0e6)
    near_z, front_z = torch.where(near, z, missing).max(dim=1).values, torch.where(front, z, missing).max(dim=1).values
    valid = (near_z > -1.0e5) & (front_z > -1.0e5)
    step_gate = torch.sigmoid(((front_z - near_z).clamp_min(0.0) - step_height_threshold) / step_height_scale) * valid.float()
    wheel_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    target_z = torch.maximum(near_z + target_clearance, front_z + wheel_radius + top_clearance_margin).unsqueeze(1)
    deficit = (target_z - wheel_z).clamp_min(0.0) / max(wheel_radius + top_clearance_margin, 1.0e-6)
    return deficit.square().mean(dim=1) * forward_gate * step_gate


def v12_rough_wheel_over_clearance(
    env, command_name: str, terrain_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    allowed_clearance: float = 0.22, clearance_scale: float = 0.07, command_threshold: float = 0.12,
    terrain_threshold: float = 0.04, terrain_scale: float = 0.02,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    clearance = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - _terrain_height_under_bodies(env, terrain_sensor_cfg, asset, asset_cfg.body_ids)
    excess = (clearance - allowed_clearance).clamp_min(0.0) / clearance_scale
    command_gate = torch.sigmoid((env.command_manager.get_command(command_name)[:, 0].abs() - command_threshold) / 0.05)
    return excess.square().mean(dim=1) * command_gate * _rough_gate(env, terrain_sensor_cfg, terrain_threshold, terrain_scale)


def v12_rough_wheel_height_symmetry(
    env, terrain_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pair_indices: tuple[tuple[int, int], ...] = ((0, 1), (2, 3), (0, 3), (1, 2)),
    diff_tolerance: float = 0.06, diff_scale: float = 0.09, terrain_threshold: float = 0.04, terrain_scale: float = 0.02,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    clearance = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - _terrain_height_under_bodies(env, terrain_sensor_cfg, asset, asset_cfg.body_ids)
    costs = [((clearance[:, a] - clearance[:, b]).abs() - diff_tolerance).clamp_min(0.0).square() / diff_scale**2 for a, b in pair_indices]
    return torch.stack(costs, dim=1).mean(dim=1) * _rough_gate(env, terrain_sensor_cfg, terrain_threshold, terrain_scale)


def v12_rough_wheel_impact(
    env, terrain_sensor_cfg: SceneEntityCfg, contact_sensor_cfg: SceneEntityCfg, force_threshold: float = 260.0,
    force_scale: float = 350.0, terrain_threshold: float = 0.045, terrain_scale: float = 0.025,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids].norm(dim=-1).amax(dim=1)
    return ((force - force_threshold).clamp_min(0.0) / force_scale).square().mean(dim=1) * _rough_gate(env, terrain_sensor_cfg, terrain_threshold, terrain_scale)


def v12_rough_wheel_blocked_drive(
    env, command_name: str, terrain_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_speed_threshold: float = 8.0, wheel_speed_scale: float = 10.0, base_speed_fraction: float = 0.35,
    terrain_threshold: float = 0.05, terrain_scale: float = 0.025,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    forward_gate = torch.sigmoid((command_x - 0.12) / 0.05)
    blocked_gate = torch.sigmoid((command_x * base_speed_fraction - asset.data.root_lin_vel_b[:, 0]) / 0.08)
    excess = (asset.data.joint_vel[:, asset_cfg.joint_ids].abs() - wheel_speed_threshold).clamp_min(0.0) / wheel_speed_scale
    return excess.square().mean(dim=1) * forward_gate * blocked_gate * _rough_gate(env, terrain_sensor_cfg, terrain_threshold, terrain_scale)


def v12_rough_forward_stall(
    env, command_name: str, terrain_sensor_cfg: SceneEntityCfg, min_speed_fraction: float = 0.45, speed_scale: float = 0.35,
    command_threshold: float = 0.12, terrain_threshold: float = 0.05, terrain_scale: float = 0.025,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    deficit = (command_x * min_speed_fraction - asset.data.root_lin_vel_b[:, 0]).clamp_min(0.0) / speed_scale
    return deficit.square() * torch.sigmoid((command_x - command_threshold) / 0.05) * _rough_gate(env, terrain_sensor_cfg, terrain_threshold, terrain_scale)


def v12_flat_leg_motion_l2(
    env, command_name: str, terrain_sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_threshold: float = 0.035, terrain_scale: float = 0.015, joint_vel_weight: float = 0.02, command_threshold: float = 0.08,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    flat_gate = torch.sigmoid((terrain_threshold - _height_scan_variation(env, terrain_sensor_cfg)) / terrain_scale)
    command_gate = torch.sigmoid((env.command_manager.get_command(command_name)[:, :2].norm(dim=1) - command_threshold) / 0.05)
    pos_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return (pos_error.square().mean(dim=1) + joint_vel_weight * asset.data.joint_vel[:, asset_cfg.joint_ids].square().mean(dim=1)) * flat_gate * command_gate


def v12_rough_base_speed_limit(
    env, command_name: str, terrain_sensor_cfg: SceneEntityCfg, max_abs_speed: float = 0.75,
    terrain_threshold: float = 0.055, terrain_scale: float = 0.025, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # V9 constrains only forward body speed here (not the xy norm).
    excess = (asset.data.root_lin_vel_b[:, 0].abs() - max_abs_speed).clamp_min(0.0)
    command_gate = torch.sigmoid((env.command_manager.get_command(command_name)[:, 0].abs() - 0.12) / 0.05)
    return excess.square() * command_gate * _rough_gate(env, terrain_sensor_cfg, terrain_threshold, terrain_scale)


def _v12_path_state(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    term = env.command_manager.get_term(command_name)
    if not hasattr(term, "v12_path_heading_w") or not hasattr(term, "v12_path_origin_w"):
        raise RuntimeError("V12 route rewards require the V12 command route state.")
    heading = term.v12_path_heading_w
    return torch.stack((torch.cos(heading), torch.sin(heading)), dim=1), torch.stack((-torch.sin(heading), torch.cos(heading)), dim=1), term.v12_path_origin_w


def v12_world_forward_speed_deficit(env, command_name: str, min_speed_fraction: float = 0.50, speed_scale: float = 0.30, command_threshold: float = 0.12, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    forward, _, _ = _v12_path_state(env, command_name)
    deficit = (command_x * min_speed_fraction - (asset.data.root_lin_vel_w[:, :2] * forward).sum(dim=1)).clamp_min(0.0) / speed_scale
    return (1.0 - torch.exp(-deficit.square())) * torch.sigmoid((command_x - command_threshold) / 0.05)


def v12_world_forward_progress_score(env, command_name: str, target_speed_fraction: float = 0.50, command_threshold: float = 0.12, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    forward, _, _ = _v12_path_state(env, command_name)
    speed = (asset.data.root_lin_vel_w[:, :2] * forward).sum(dim=1)
    score = (speed / (command_x * target_speed_fraction).clamp_min(1.0e-3)).clamp(0.0, 1.0)
    return torch.where(command_x > command_threshold, score, torch.ones_like(score))


def v12_path_lateral_deviation_l2(env, command_name: str, tolerance: float = 0.25, deviation_scale: float = 0.35, command_threshold: float = 0.12, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command(command_name)[:, 0]
    _, left, origin = _v12_path_state(env, command_name)
    violation = ((asset.data.root_pos_w[:, :2] - origin) * left).sum(dim=1).abs().sub(tolerance).clamp_min(0.0) / deviation_scale
    return (1.0 - torch.exp(-violation.square())) * torch.sigmoid((command_x - command_threshold) / 0.05)


def v12_path_heading_error_l2(env, command_name: str, heading_scale: float = 0.5, command_threshold: float = 0.12, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    command_x = env.command_manager.get_command(command_name)[:, 0]
    error = math_utils.wrap_to_pi(asset.data.heading_w - term.v12_path_heading_w) / heading_scale
    return (1.0 - torch.exp(-error.square())) * torch.sigmoid((command_x - command_threshold) / 0.05)
