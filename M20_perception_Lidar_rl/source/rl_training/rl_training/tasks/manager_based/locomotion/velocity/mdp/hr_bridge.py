"""Simulation teacher and three-state controller for M20 HR Bridge."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.managers import SceneEntityCfg

from .hr_bridge_contracts import HR_INDEX, STATE_BASE, STATE_BLIND_STAIR, STATE_HR_BRIDGE
from .perceptive_spatial import (
    _elevation_points_yaw,
    _semantic_body_ids,
    classify_vertical_riser_contact,
    extract_nearest_riser_points,
    wheel_positions_yaw,
)

@dataclass
class HRBridgeStep:
    state_before: torch.Tensor
    state_after: torch.Tensor
    activation: torch.Tensor
    clearance_event: torch.Tensor
    crossing_event: torch.Tensor
    landing_event: torch.Tensor
    support_event: torch.Tensor
    success_event: torch.Tensor
    miss_event: torch.Tensor
    joint_success_event: torch.Tensor
    riser: dict[str, torch.Tensor]


class HRBridgeTracker:
    """Per-environment first-riser state with simulator-only teacher labels."""

    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.trigger_cfg = env.cfg.hr_bridge_trigger
        self.teacher_cfg = env.cfg.hr_bridge_teacher
        self.state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.confirm_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.release_count = torch.zeros_like(self.confirm_count)
        self.elapsed_steps = torch.zeros_like(self.confirm_count)
        self.clearance_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.crossing_latched = torch.zeros_like(self.clearance_latched)
        self.landing_latched = torch.zeros_like(self.clearance_latched)
        self.support_latched = torch.zeros_like(self.clearance_latched)
        self.success_latched = torch.zeros_like(self.clearance_latched)
        self.miss_latched = torch.zeros_like(self.clearance_latched)
        self.joint_success_latched = torch.zeros_like(self.clearance_latched)
        self.support_count = torch.zeros_like(self.confirm_count)
        self.bridge_start_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.bridge_heading_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.full_stair_distance = torch.zeros(self.num_envs, device=self.device)
        self._resolve_ids()

    def _resolve_ids(self):
        robot = self.env.scene["robot"]
        self.wheel_body_ids = _semantic_body_ids(robot)
        contact = self.env.scene.sensors["contact_forces"]
        ids = []
        for name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"):
            found, names = contact.find_bodies(name)
            if len(found) != 1 or names != [name]:
                raise RuntimeError(f"HR Bridge contact body mismatch for {name}: {names}")
            ids.append(found[0])
        self.contact_body_ids = torch.tensor(ids, device=self.device, dtype=torch.long)

    def reset(self, dones: torch.Tensor):
        ids = dones.reshape(-1).bool()
        self.state[ids] = STATE_BASE
        self.confirm_count[ids] = 0
        self.release_count[ids] = 0
        self.elapsed_steps[ids] = 0
        self.clearance_latched[ids] = False
        self.crossing_latched[ids] = False
        self.landing_latched[ids] = False
        self.support_latched[ids] = False
        self.success_latched[ids] = False
        self.miss_latched[ids] = False
        self.joint_success_latched[ids] = False
        self.support_count[ids] = 0
        self.bridge_start_xy[ids] = 0.0
        self.bridge_heading_xy[ids] = 0.0
        self.full_stair_distance[ids] = 0.0

    def _geometry(self):
        robot = self.env.scene["robot"]
        wheel_yaw = wheel_positions_yaw(robot, self.wheel_body_ids)
        points = _elevation_points_yaw(self.env, SceneEntityCfg("height_scanner"), robot)
        riser = extract_nearest_riser_points(
            points,
            wheel_yaw[..., :2],
            corridor_half_width=self.trigger_cfg.corridor_half_width,
            h_min=self.trigger_cfg.min_step_up,
            d_end=-0.12,
        )
        finite_fraction = torch.isfinite(points[..., 2]).float().mean(dim=(1, 2))
        riser["elevation_valid"] = finite_fraction >= self.trigger_cfg.elevation_valid_fraction
        riser["wheel_yaw"] = wheel_yaw
        return riser

    def pre_action_state(self):
        riser = self._geometry()
        h = riser["riser_height"][:, HR_INDEX]
        d = riser["riser_distance"][:, HR_INDEX]
        cmd = self.env.command_manager.get_command("base_velocity")[:, 0]
        threshold = self.trigger_cfg.min_step_up
        enter_valid = (
            riser["valid"][:, HR_INDEX]
            & riser["elevation_valid"]
            & (h >= threshold - self.trigger_cfg.measurement_epsilon)
            & (d >= self.trigger_cfg.anticipatory_distance_min)
            & (d <= self.trigger_cfg.anticipatory_distance_max)
            & (cmd > self.trigger_cfg.min_forward_command)
            & (~self.success_latched)
            & (~self.miss_latched)
        )
        self.confirm_count = torch.where(enter_valid, self.confirm_count + 1, torch.zeros_like(self.confirm_count))
        activate = (self.state == STATE_BASE) & (self.confirm_count >= self.trigger_cfg.confirm_frames)
        if activate.any():
            robot = self.env.scene["robot"]
            quat = robot.data.root_quat_w[activate]
            w, x, y, z = quat.unbind(dim=-1)
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
            self.bridge_start_xy[activate] = robot.data.root_pos_w[activate, :2]
            self.bridge_heading_xy[activate] = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=1)
            terrain = self.env.cfg.scene.terrain.terrain_generator
            if "pyramid_stairs_inv" in terrain.sub_terrains:
                step_depth = terrain.sub_terrains["pyramid_stairs_inv"].step_width
            else:
                step_depth = terrain.sub_terrains["hr_bridge_fixed_stair"].step_depth
            first_riser_from_base = riser["riser_x"][activate, HR_INDEX].clamp_min(0.0)
            self.full_stair_distance[activate] = first_riser_from_base + 6.0 * step_depth
        self.state[activate] = STATE_HR_BRIDGE
        self.elapsed_steps = torch.where(
            self.state == STATE_HR_BRIDGE, self.elapsed_steps + 1, self.elapsed_steps
        )
        return self.state.clone(), riser, activate

    def post_step(self, state_before, riser, activation):
        robot = self.env.scene["robot"]
        hr = HR_INDEX
        force = self.env.scene.sensors["contact_forces"].data.net_forces_w[:, self.contact_body_ids]
        force_hr = force[:, hr]
        z = robot.data.body_pos_w[:, self.wheel_body_ids[hr], 2]
        radius = self.teacher_cfg.wheel_radius
        lower = riser["riser_lower_world"][:, hr]
        upper = lower + riser["riser_height"][:, hr]
        wheel_bottom = z - radius
        valid = riser["valid"][:, hr]
        vertical = classify_vertical_riser_contact(
            riser["wheel_yaw"][..., :2],
            robot.data.body_pos_w[:, self.wheel_body_ids, 2],
            force,
            riser["riser_x"],
            riser["riser_lower_world"],
            riser["riser_lower_world"] + riser["riser_height"],
            wheel_radius=radius,
        )[:, hr] & valid
        in_s1 = state_before == STATE_HR_BRIDGE
        clearance_now = in_s1 & valid & (~vertical) & (
            wheel_bottom >= upper + self.teacher_cfg.clearance_margin
        )
        clearance_event = clearance_now & (~self.clearance_latched)
        self.clearance_latched |= clearance_now
        crossed_now = in_s1 & self.clearance_latched & (
            riser["wheel_yaw"][:, hr, 0] > riser["riser_x"][:, hr] + self.teacher_cfg.cross_margin
        )
        crossing_event = crossed_now & (~self.crossing_latched)
        self.crossing_latched |= crossed_now
        landed_now = (
            in_s1
            & self.crossing_latched
            & ((wheel_bottom - upper).abs() <= self.teacher_cfg.landing_height_tolerance)
            & (force_hr[:, 2] > self.teacher_cfg.support_force_threshold)
        )
        landing_event = landed_now & (~self.landing_latched)
        self.landing_latched |= landed_now
        stable = (robot.data.projected_gravity_b[:, 2] < -0.65) & (
            robot.data.projected_gravity_b[:, :2].abs().amax(dim=1) < 0.55
        )
        support_now = landed_now & stable
        self.support_count = torch.where(support_now, self.support_count + 1, torch.zeros_like(self.support_count))
        support_ready = self.support_count >= self.teacher_cfg.support_frames
        support_event = support_ready & (~self.support_latched)
        self.support_latched |= support_ready
        success_now = (
            self.clearance_latched & self.crossing_latched & self.landing_latched
            & self.support_latched & (~self.miss_latched)
        )
        success_event = success_now & (~self.success_latched)
        self.success_latched |= success_now
        valid_stair = valid & (
            riser["riser_height"][:, hr]
            >= self.trigger_cfg.min_step_up - self.trigger_cfg.measurement_epsilon
        )
        miss_now = valid_stair & vertical & (~self.success_latched)
        miss_event = miss_now & (~self.miss_latched)
        self.miss_latched |= miss_now
        self.state[success_event | miss_event] = STATE_BLIND_STAIR

        root_delta = robot.data.root_pos_w[:, :2] - self.bridge_start_xy
        root_progress = (root_delta * self.bridge_heading_xy).sum(dim=1)
        full = (self.full_stair_distance > 0.0) & (root_progress > self.full_stair_distance)
        joint_success_event = full & self.success_latched & (~self.joint_success_latched)
        self.joint_success_latched |= joint_success_event
        return HRBridgeStep(
            state_before=state_before,
            state_after=self.state.clone(),
            activation=activation,
            clearance_event=clearance_event,
            crossing_event=crossing_event,
            landing_event=landing_event,
            support_event=support_event,
            success_event=success_event,
            miss_event=miss_event,
            joint_success_event=joint_success_event,
            riser=riser,
        )
