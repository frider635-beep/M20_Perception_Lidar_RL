# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


UPSTAIRS_HEIGHT_BUCKETS = (
    ("05_08cm", 0.05, 0.08),
    ("08_11cm", 0.08, 0.11),
    ("11_14cm", 0.11, 0.14),
    ("14_17cm", 0.14, 0.17),
    ("17_19cm", 0.17, 0.19),
    ("19_21cm", 0.19, 0.21),
    ("21_23cm", 0.21, 0.230001),
)


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    Supports fixed-proportion special-case samples that always use the original (full)
    velocity ranges, preventing catastrophic forgetting of basic skills as the curriculum
    narrows the command range.
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._v6_env = env
        # Additional metrics for TensorBoard.
        self.metrics["base_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["knee_pos"] = torch.zeros(self.num_envs, device=self.device)
        self._metric_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if cfg.v6_metrics:
            for name in (
                "v6/active_wheel_ratio", "v6/clearance_deficit_active", "v6/clearance_deficit_all_envs",
                "v6/mean_clearance_ratio", "v6/mean_actual_clearance", "v6/mean_required_clearance",
                "v6/stagnation_ratio", "v6/riser_collision_count", "v6/wheel_crossing_count",
                "v6/step_complete_count", "v6/stair17_full_success", "v6/stair20_full_success",
                "v6/stair08_15_full_success", "v6/stair_failure", "base_roll_abs", "base_pitch_abs",
                "joint_effort_rms",
            ):
                self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
            for level in range(6):
                self.metrics[f"v6/level_{level + 1}_success"] = torch.zeros(self.num_envs, device=self.device)
        if cfg.perceptive_metrics:
            names = (
                "curriculum/terrain_level_mean", "stair/height", "stair/L1_cross", "stair/L2_cross",
                "stair/L3_cross", "stair/L4_cross", "stair/L5_cross", "stair/L6_cross", "stair/full_success",
                "stair/sample_fraction",
                "guide/active_fraction", "guide/riser_height_mean", "guide/riser_distance_mean",
                "guide/lift_deficit_mean", "guide/false_lift", "reward/spatial_raw", "reward/spatial_weighted",
                "reward/false_lift", "body/roll_abs", "body/pitch_abs", "body/ang_vel_xy",
                "action/leg_norm", "action/wheel_norm",
            )
            for name in names:
                self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
            for wheel in ("FL", "FR", "HL", "HR"):
                self.metrics[f"guide/lift_ref_{wheel}"] = torch.zeros(self.num_envs, device=self.device)
                self.metrics[f"guide/lift_actual_{wheel}"] = torch.zeros(self.num_envs, device=self.device)
            # These metrics are reduced with explicit condition counts rather
            # than averaging zero-masked values over all 4096 environments.
            # They are simulation/TensorBoard diagnostics only: none is
            # registered with an observation or reward manager.
            self._upstairs_metric_denominators: dict[str, torch.Tensor] = {}
            self._upstairs_raw_count_metrics: set[str] = set()
            upstairs_names = (
                "upstairs/sample_fraction",
                "upstairs/terrain_level",
                "upstairs/actual_step_height",
                "upstairs/riser_valid",
                "upstairs/guide_active",
                "upstairs/spatial_raw",
                "upstairs/spatial_weighted",
                "upstairs/spatial_nonzero_fraction",
                "upstairs/lift_ref",
                "upstairs/lift_actual",
                "upstairs/lift_deficit",
                "upstairs/clearance_progress",
                "upstairs/vertical_riser_contact",
                "upstairs/precontact_clearance_pass",
                "upstairs/full_success",
                "upstairs/base_roll_abs",
                "upstairs/base_pitch_abs",
                "upstairs/base_lin_vel_z",
                "upstairs/base_ang_vel_xy",
                "upstairs/fall",
                "upstairs/timeout",
            )
            for name in upstairs_names:
                self._register_upstairs_metric(name)
            for stair_level in range(1, 7):
                self._register_upstairs_metric(f"upstairs/L{stair_level}_cross")
            for wheel in ("FL", "FR", "HL", "HR"):
                self._register_upstairs_metric(f"upstairs/{wheel}/active_fraction")
                self._register_upstairs_metric(f"upstairs/{wheel}/active_sample_count", raw_count=True)
                self._register_upstairs_metric(f"upstairs/{wheel}/lift_ref_active")
                self._register_upstairs_metric(f"upstairs/{wheel}/lift_actual")
                self._register_upstairs_metric(f"upstairs/{wheel}/deficit_active")
                self._register_upstairs_metric(f"upstairs/{wheel}/clearance_progress")
            for bucket, _, _ in UPSTAIRS_HEIGHT_BUCKETS:
                for suffix in (
                    "sample_fraction", "clearance_progress", "vertical_contact", "L1_cross", "full_success"
                ):
                    self._register_upstairs_metric(f"upstairs/height_{bucket}/{suffix}")
            # Resolve by exact semantic names and assert the explicit metric
            # order. This does not rely on either asset's native body order.
            robot_order = []
            contact_order = []
            contact_sensor = env.scene.sensors["contact_forces"]
            contact_ids = []
            for wheel_name in mdp.WHEEL_NAMES:
                robot_ids, robot_names = self.robot.find_bodies(wheel_name)
                sensor_ids, sensor_names = contact_sensor.find_bodies(wheel_name)
                if len(robot_ids) != 1 or robot_names != [wheel_name]:
                    raise RuntimeError(f"robot wheel mapping failed for {wheel_name!r}: {robot_names}")
                if len(sensor_ids) != 1 or sensor_names != [wheel_name]:
                    raise RuntimeError(f"contact wheel mapping failed for {wheel_name!r}: {sensor_names}")
                robot_order.append((wheel_name, robot_ids[0]))
                contact_order.append((wheel_name, sensor_ids[0]))
                contact_ids.append(sensor_ids[0])
            self._contact_wheel_body_ids = torch.tensor(contact_ids, device=self.device, dtype=torch.long)
            print(f"PERCEPTIVE_WHEEL_BODY_ORDER={list(mdp.WHEEL_NAMES)}", flush=True)
            print(f"PERCEPTIVE_ROBOT_WHEEL_BODY_IDS={robot_order}", flush=True)
            print(f"PERCEPTIVE_CONTACT_WHEEL_BODY_IDS={contact_order}", flush=True)
        self._stair_metric_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        knee_joint_ids = self.robot.find_joints(".*[Kk]nee.*")[0]
        self._knee_joint_ids = torch.tensor(knee_joint_ids, dtype=torch.long, device=self.device)

        # ---- Store original full ranges for special-case sampling ----
        # These are used by special-case envs so they always sample from the
        # full range regardless of curriculum changes.
        self._full_ranges_lin_vel_x = tuple(cfg.ranges.lin_vel_x)
        self._full_ranges_lin_vel_y = tuple(cfg.ranges.lin_vel_y)
        self._full_ranges_ang_vel_z = tuple(cfg.ranges.ang_vel_z)
        # V11 stair rewards use a world-fixed route captured when the command
        # is sampled.  This is reward-only state: it never changes the 63-D
        # actor ABI and prevents a body-frame forward command from being met
        # after turning away from an open-sided staircase.
        self.v11_path_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.v11_path_origin_w = torch.zeros((self.num_envs, 2), device=self.device)
        # V12 adopts V9's held-heading command semantics.  Keep independent
        # route state so V11's forward-only override remains byte-for-byte
        # unchanged for the currently running V11 task.
        self.v12_path_heading_w = torch.zeros(self.num_envs, device=self.device)
        self.v12_path_origin_w = torch.zeros((self.num_envs, 2), device=self.device)

    def _register_upstairs_metric(self, name: str, *, raw_count: bool = False):
        """Register a conditional TensorBoard numerator and denominator."""
        self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._upstairs_metric_denominators[name] = torch.zeros(self.num_envs, device=self.device)
        if raw_count:
            self._upstairs_raw_count_metrics.add(name)

    def _accumulate_upstairs_metric(
        self, name: str, numerator: torch.Tensor, denominator: torch.Tensor
    ):
        """Accumulate a conditional statistic without 4096-env zero dilution."""
        self.metrics[name] += numerator.float()
        self._upstairs_metric_denominators[name] += denominator.float()

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)

        extras = {}
        mean_metrics = {
            "base_z", "knee_pos", "v6/active_wheel_ratio", "v6/clearance_deficit_active",
            "v6/clearance_deficit_all_envs", "v6/mean_clearance_ratio", "v6/mean_actual_clearance",
            "v6/mean_required_clearance", "v6/stagnation_ratio", "base_roll_abs", "base_pitch_abs",
            "joint_effort_rms",
        }
        if self.cfg.perceptive_metrics:
            mean_metrics.update(name for name in self.metrics if not name.startswith("stair/L") and name != "stair/full_success")
        for metric_name, metric_value in self.metrics.items():
            if metric_name in getattr(self, "_upstairs_metric_denominators", {}):
                denominator = self._upstairs_metric_denominators[metric_name]
                denominator_sum = denominator[env_ids].sum()
                if metric_name in self._upstairs_raw_count_metrics:
                    extras[metric_name] = metric_value[env_ids].sum().item()
                elif denominator_sum > 0:
                    extras[metric_name] = (metric_value[env_ids].sum() / denominator_sum).item()
                else:
                    extras[metric_name] = 0.0
                denominator[env_ids] = 0.0
            elif metric_name == "stair/height":
                stair_count = self._stair_metric_step_counter[env_ids]
                valid = stair_count > 0
                if torch.any(valid):
                    extras[metric_name] = torch.mean(metric_value[env_ids][valid] / stair_count[valid].float()).item()
                else:
                    extras[metric_name] = 0.0
            elif metric_name in mean_metrics:
                step_count = torch.clamp(self._metric_step_counter[env_ids].float(), min=1.0)
                extras[metric_name] = torch.mean(metric_value[env_ids] / step_count).item()
            else:
                extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            metric_value[env_ids] = 0.0

        self._metric_step_counter[env_ids] = 0
        self._stair_metric_step_counter[env_ids] = 0
        self.command_counter[env_ids] = 0
        self._resample(env_ids)
        return extras

    def _update_metrics(self):
        super()._update_metrics()

        # 1) base_z metric: root_pos_w[:, 2]
        base_z = self.robot.data.root_pos_w[:, 2]

        # 2) knee_pos metric: same formulation as joint_pos_penalty for knee joints
        cmd = torch.linalg.norm(self.vel_command_b, dim=1)
        body_vel = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)

        if self._knee_joint_ids.numel() > 0:
            running_reward = torch.linalg.norm(
                self.robot.data.joint_pos[:, self._knee_joint_ids]
                - self.robot.data.default_joint_pos[:, self._knee_joint_ids],
                dim=1,
            )
        else:
            running_reward = torch.zeros(self.num_envs, device=self.device)

        knee_pos = torch.where(
            torch.logical_or(cmd > 0.1, body_vel > 0.5),
            running_reward,
            5.0 * running_reward,
        )

        self.metrics["base_z"] += base_z
        self.metrics["knee_pos"] += knee_pos
        if self.cfg.v6_metrics and hasattr(self._v6_env, "v6_current_level"):
            env = self._v6_env
            defaults = torch.zeros(self.num_envs, device=self.device)
            mapping = {
                "v6/active_wheel_ratio": "v6_active_wheel_ratio",
                "v6/clearance_deficit_active": "v6_mean_deficit_active",
                "v6/clearance_deficit_all_envs": "v6_mean_deficit_all",
                "v6/mean_clearance_ratio": "v6_mean_clearance_ratio",
                "v6/mean_actual_clearance": "v6_mean_actual_clearance",
                "v6/mean_required_clearance": "v6_mean_required_clearance",
                "v6/stagnation_ratio": "v6_stagnation",
                "v6/riser_collision_count": "v6_collision_count",
                "v6/wheel_crossing_count": "v6_crossing_count",
                "v6/step_complete_count": "v6_step_complete_event",
                "v6/stair_failure": "v6_failure_event",
            }
            for metric, attr in mapping.items():
                self.metrics[metric] += getattr(env, attr, defaults).float()
            level_events = getattr(env, "v6_level_complete_event", torch.zeros((self.num_envs, 6), device=self.device))
            for level in range(6):
                self.metrics[f"v6/level_{level + 1}_success"] += level_events[:, level]
            success = getattr(env, "v6_success_event", defaults)
            self.metrics["v6/stair17_full_success"] += success * ((env.v6_stair_height >= 0.16) & (env.v6_stair_height < 0.185))
            self.metrics["v6/stair20_full_success"] += success * (env.v6_stair_height >= 0.185)
            self.metrics["v6/stair08_15_full_success"] += success * (env.v6_stair_height < 0.16)
            gravity = self.robot.data.projected_gravity_b
            self.metrics["base_roll_abs"] += torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
            self.metrics["base_pitch_abs"] += torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
            self.metrics["joint_effort_rms"] += self.robot.data.applied_torque.square().mean(1).sqrt()
        if self.cfg.perceptive_metrics and hasattr(self._v6_env, "perceptive_spatial_state"):
            env = self._v6_env
            state = env.perceptive_spatial_state
            level = env.scene.terrain.terrain_levels
            terrain_cfg = env.cfg.scene.terrain.terrain_generator
            stair_cfg = terrain_cfg.sub_terrains["pyramid_stairs_inv"]
            sub_names = tuple(terrain_cfg.sub_terrains)
            proportions = torch.tensor(
                [terrain_cfg.sub_terrains[name].proportion for name in sub_names],
                device=self.device,
                dtype=torch.float32,
            )
            proportions /= proportions.sum()
            column_fraction = torch.arange(terrain_cfg.num_cols, device=self.device, dtype=torch.float32)
            column_fraction = column_fraction / terrain_cfg.num_cols + 0.001
            column_sub_index = torch.searchsorted(torch.cumsum(proportions, dim=0), column_fraction, right=False)
            env_sub_index = column_sub_index[env.scene.terrain.terrain_types]
            stair_indices = torch.tensor(
                [sub_names.index("pyramid_stairs"), sub_names.index("pyramid_stairs_inv")],
                device=self.device,
            )
            stair_mask = (env_sub_index[:, None] == stair_indices[None, :]).any(dim=1)
            # The official generator samples continuous difficulty within each
            # row. Recover the actual generated step height from the runtime
            # origin instead of treating a row index as one fixed height. Both
            # pyramid variants encode origin Z as +/-(num_steps+1)*height.
            stair_span_x = stair_cfg.size[0] - 2.0 * stair_cfg.border_width - stair_cfg.platform_width
            stair_span_y = stair_cfg.size[1] - 2.0 * stair_cfg.border_width - stair_cfg.platform_width
            num_steps_x = stair_span_x // (2.0 * stair_cfg.step_width) + 1.0
            num_steps_y = stair_span_y // (2.0 * stair_cfg.step_width) + 1.0
            num_steps = int(min(num_steps_x, num_steps_y))
            stair_heights = env.scene.env_origins[:, 2].abs() / float(num_steps + 1)
            first_riser = stair_cfg.platform_width / 2.0
            step_depth = stair_cfg.step_width
            if hasattr(env.cfg, "v135_riser_identity"):
                from .v135_terrain_geometry import v135_terrain_geometry
                geometry = v135_terrain_geometry(env)
                stair_mask |= geometry["ascent"]
                stair_heights = torch.where(geometry["ascent"], env.scene.env_origins[:, 2].abs()
                    / geometry["flight_riser_count"].clamp_min(1), stair_heights)
                first_riser = torch.where(geometry["ascent"], geometry["radii"][:, 0],
                                          torch.full_like(stair_heights, first_riser))
                step_depth = torch.where(geometry["ascent"], geometry["step_width"],
                                         torch.full_like(stair_heights, step_depth))
            self.metrics["curriculum/terrain_level_mean"] += level.float()
            self.metrics["stair/height"] += torch.where(stair_mask, stair_heights, 0.0)
            self.metrics["stair/sample_fraction"] += stair_mask.float()
            self._stair_metric_step_counter += stair_mask.long()
            progress = torch.linalg.norm(
                self.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1
            )
            for stair_level in range(6):
                crossed = stair_mask & (progress > (first_riser + (stair_level + 1) * step_depth))
                self.metrics[f"stair/L{stair_level + 1}_cross"] += crossed.float()
            self.metrics["stair/full_success"] += stair_mask.float() * getattr(
                env, "perceptive_curriculum_success", torch.zeros(self.num_envs, device=self.device)
            )
            active = state["active"]
            active_count = active.sum(1).clamp_min(1)
            self.metrics["guide/active_fraction"] += active.float().mean(1)
            self.metrics["guide/riser_height_mean"] += torch.where(active, state["riser_height"], 0.0).sum(1) / active_count
            finite_distance = torch.where(active, state["riser_distance"], 0.0)
            self.metrics["guide/riser_distance_mean"] += finite_distance.sum(1) / active_count
            self.metrics["guide/lift_deficit_mean"] += torch.where(active, state["deficit"], 0.0).sum(1) / active_count
            self.metrics["guide/false_lift"] += state["false_excess"].mean(1)
            self.metrics["reward/spatial_raw"] += state["spatial_raw"]
            self.metrics["reward/spatial_weighted"] += state["spatial_effective"] * 2.0
            self.metrics["reward/false_lift"] += state["false_cost"] * -0.2
            gravity = self.robot.data.projected_gravity_b
            self.metrics["body/roll_abs"] += torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
            self.metrics["body/pitch_abs"] += torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
            self.metrics["body/ang_vel_xy"] += self.robot.data.root_ang_vel_b[:, :2].norm(dim=1)
            action = env.action_manager.action
            self.metrics["action/leg_norm"] += action[:, :12].norm(dim=1)
            self.metrics["action/wheel_norm"] += action[:, 12:16].norm(dim=1)
            for index, wheel in enumerate(("FL", "FR", "HL", "HR")):
                self.metrics[f"guide/lift_ref_{wheel}"] += state["lift_ref"][:, index]
                self.metrics[f"guide/lift_actual_{wheel}"] += state["actual_lift"][:, index]
            self._update_upstairs_metrics(env, state, terrain_cfg, sub_names, env_sub_index, stair_heights)
        self._metric_step_counter += 1

    def _update_upstairs_metrics(self, env, state, terrain_cfg, sub_names, env_sub_index, stair_heights):
        """Update simulation-only metrics conditioned on ascending stair envs."""
        up_sub_index = sub_names.index("pyramid_stairs_inv")
        up_mask = env_sub_index == up_sub_index
        if hasattr(env.cfg, "v135_riser_identity"):
            from .v135_terrain_geometry import v135_terrain_geometry
            up_mask = v135_terrain_geometry(env)["ascent"]
        up = up_mask.float()
        all_env = torch.ones_like(up)
        level = env.scene.terrain.terrain_levels.float()

        self._accumulate_upstairs_metric("upstairs/sample_fraction", up, all_env)
        self._accumulate_upstairs_metric("upstairs/terrain_level", level * up, up)
        self._accumulate_upstairs_metric("upstairs/actual_step_height", stair_heights * up, up)

        valid = state["valid"] & up_mask[:, None]
        active = state["active"] & up_mask[:, None]
        valid_count = valid.sum(dim=1).float()
        active_count = active.sum(dim=1).float()
        self._accumulate_upstairs_metric("upstairs/riser_valid", valid_count, 4.0 * up)
        self._accumulate_upstairs_metric("upstairs/guide_active", active_count, 4.0 * up)
        self._accumulate_upstairs_metric("upstairs/spatial_raw", state["spatial_raw"] * up, up)
        self._accumulate_upstairs_metric(
            "upstairs/spatial_weighted", state["spatial_effective"] * 2.0 * up, up
        )
        self._accumulate_upstairs_metric(
            "upstairs/spatial_nonzero_fraction", (state["spatial_effective"] > 0.0).float() * up, up
        )

        active_float = active.float()
        self._accumulate_upstairs_metric(
            "upstairs/lift_ref", (state["lift_ref"] * active_float).sum(dim=1), active_count
        )
        self._accumulate_upstairs_metric(
            "upstairs/lift_actual", (state["actual_lift"] * active_float).sum(dim=1), active_count
        )
        self._accumulate_upstairs_metric(
            "upstairs/lift_deficit", (state["deficit"] * active_float).sum(dim=1), active_count
        )
        self._accumulate_upstairs_metric(
            "upstairs/clearance_progress", (state["progress"] * active_float).sum(dim=1), active_count
        )

        # BAD vertical-face contact uses the already accepted geometry/contact
        # classifier. Upper-tread support is deliberately excluded.
        contact_sensor = env.scene.sensors["contact_forces"]
        if not hasattr(self, "_contact_wheel_body_ids"):
            contact_ids = []
            for wheel_name in mdp.WHEEL_NAMES:
                found, names = contact_sensor.find_bodies(wheel_name)
                if len(found) != 1 or names[0] != wheel_name:
                    raise RuntimeError(f"expected one contact body {wheel_name!r}, got {names}")
                contact_ids.append(found[0])
            self._contact_wheel_body_ids = torch.tensor(contact_ids, device=self.device, dtype=torch.long)
        wheel_body_ids = self._contact_wheel_body_ids
        wheel_forces_w = contact_sensor.data.net_forces_w[:, wheel_body_ids]
        vertical_contact_wheel = mdp.classify_vertical_riser_contact(
            state["wheel_yaw"][..., :2],
            state["z_actual_world"],
            wheel_forces_w,
            state["riser_x"],
            state["z_lower"],
            state["z_upper"],
            wheel_radius=env.cfg.spatial_guidance.wheel_collision_radius,
        ) & valid
        vertical_contact = vertical_contact_wheel.any(dim=1).float()
        self._accumulate_upstairs_metric("upstairs/vertical_riser_contact", vertical_contact, up)

        precontact = active & (state["riser_distance"] >= 0.0)
        precontact_count = precontact.sum(dim=1).float()
        precontact_pass = (precontact & (state["progress"] >= 0.999)).sum(dim=1).float()
        self._accumulate_upstairs_metric(
            "upstairs/precontact_clearance_pass", precontact_pass, precontact_count
        )

        stair_cfg = terrain_cfg.sub_terrains["pyramid_stairs_inv"]
        first_riser = stair_cfg.platform_width / 2.0
        root_progress = self.robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        step_depth = stair_cfg.step_width
        if hasattr(env.cfg, "v135_riser_identity"):
            geometry = v135_terrain_geometry(env)
            first_riser = geometry["radii"][:, 0]
            step_depth = geometry["step_width"]
            # Route, not world x: ascent resets cover all four cardinal faces.
            direction = torch.stack((self.v11_path_heading_w.cos(), self.v11_path_heading_w.sin()), 1)
            root_progress = ((self.robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]) * direction).sum(1)
        crossings = []
        for stair_level in range(1, 7):
            crossed = up_mask & (root_progress > first_riser + stair_level * step_depth)
            crossings.append(crossed)
            self._accumulate_upstairs_metric(f"upstairs/L{stair_level}_cross", crossed.float(), up)
        gravity = self.robot.data.projected_gravity_b
        stable = (gravity[:, 2] < -0.65) & (gravity[:, :2].abs().amax(dim=1) < 0.55)
        full_success = crossings[-1] & stable
        self._accumulate_upstairs_metric("upstairs/full_success", full_success.float(), up)

        roll = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1.0e-6))
        pitch = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1.0e-6))
        self._accumulate_upstairs_metric("upstairs/base_roll_abs", roll * up, up)
        self._accumulate_upstairs_metric("upstairs/base_pitch_abs", pitch * up, up)
        self._accumulate_upstairs_metric(
            "upstairs/base_lin_vel_z", self.robot.data.root_lin_vel_b[:, 2] * up, up
        )
        self._accumulate_upstairs_metric(
            "upstairs/base_ang_vel_xy", self.robot.data.root_ang_vel_b[:, :2].norm(dim=1) * up, up
        )
        termination_manager = env.termination_manager
        fall = termination_manager.terminated.float() * up
        timeout = termination_manager.time_outs.float() * up
        self._accumulate_upstairs_metric("upstairs/fall", fall, up)
        self._accumulate_upstairs_metric("upstairs/timeout", timeout, up)

        for index, wheel in enumerate(("FL", "FR", "HL", "HR")):
            wheel_active = active[:, index].float()
            self._accumulate_upstairs_metric(
                f"upstairs/{wheel}/active_fraction", wheel_active, up
            )
            self._accumulate_upstairs_metric(
                f"upstairs/{wheel}/active_sample_count", wheel_active, up
            )
            self._accumulate_upstairs_metric(
                f"upstairs/{wheel}/lift_ref_active", state["lift_ref"][:, index] * wheel_active, wheel_active
            )
            self._accumulate_upstairs_metric(
                f"upstairs/{wheel}/lift_actual", state["actual_lift"][:, index] * wheel_active, wheel_active
            )
            self._accumulate_upstairs_metric(
                f"upstairs/{wheel}/deficit_active", state["deficit"][:, index] * wheel_active, wheel_active
            )
            self._accumulate_upstairs_metric(
                f"upstairs/{wheel}/clearance_progress", state["progress"][:, index] * wheel_active, wheel_active
            )

        env_clearance_num = (state["progress"] * active_float).sum(dim=1)
        env_clearance_den = active_count
        for bucket, lower, upper in UPSTAIRS_HEIGHT_BUCKETS:
            bucket_mask = up_mask & (stair_heights >= lower) & (stair_heights < upper)
            bucket_float = bucket_mask.float()
            self._accumulate_upstairs_metric(
                f"upstairs/height_{bucket}/sample_fraction", bucket_float, up
            )
            self._accumulate_upstairs_metric(
                f"upstairs/height_{bucket}/clearance_progress",
                env_clearance_num * bucket_float,
                env_clearance_den * bucket_float,
            )
            self._accumulate_upstairs_metric(
                f"upstairs/height_{bucket}/vertical_contact", vertical_contact * bucket_float, bucket_float
            )
            self._accumulate_upstairs_metric(
                f"upstairs/height_{bucket}/L1_cross", crossings[0].float() * bucket_float, bucket_float
            )
            self._accumulate_upstairs_metric(
                f"upstairs/height_{bucket}/full_success", full_success.float() * bucket_float, bucket_float
            )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # ---- Fixed-proportion special-case samples ----
        # These override the sampled command for a fixed fraction of envs,
        # always using the original (full) velocity ranges so they are not
        # affected by curriculum narrowing.  This prevents catastrophic
        # forgetting of basic skills.
        n = len(env_ids)
        if n == 0:
            return

        r = torch.empty(n, device=self.device)
        # Assign each env a slot in [0, 1).  Slots are laid out as:
        #   [0,                          rel_zero_vel          ) -> zero velocity
        #   [rel_zero_vel,              +rel_only_lin_y        ) -> only lin_y
        #   [rel_zero_vel+rel_only_lin_y, +rel_only_lin_x      ) -> only lin_x
        #   [...,                        +rel_only_ang_z        ) -> only ang_z
        #   [...,                        1.0                    ) -> normal (no override)
        slot = r.uniform_(0.0, 1.0)
        cum = 0.0

        # --- Zero velocity (standing) ---
        rel_zero_vel = self.cfg.rel_zero_vel_envs
        mask_zero = slot < cum + rel_zero_vel
        cum += rel_zero_vel

        # --- Only linear y ---
        rel_only_lin_y = self.cfg.rel_only_lin_y_envs
        mask_only_y = (slot >= cum) & (slot < cum + rel_only_lin_y)
        cum += rel_only_lin_y

        # --- Only linear x ---
        rel_only_lin_x = self.cfg.rel_only_lin_x_envs
        mask_only_x = (slot >= cum) & (slot < cum + rel_only_lin_x)
        cum += rel_only_lin_x

        # --- Only angular z ---
        rel_only_ang_z = self.cfg.rel_only_ang_z_envs
        mask_only_ang = (slot >= cum) & (slot < cum + rel_only_ang_z)
        cum += rel_only_ang_z

        ids_zero = env_ids[mask_zero]
        ids_only_y = env_ids[mask_only_y]
        ids_only_x = env_ids[mask_only_x]
        ids_only_ang = env_ids[mask_only_ang]

        # Apply zero velocity
        if len(ids_zero) > 0:
            self.vel_command_b[ids_zero, :] = 0.0
            self.is_standing_env[ids_zero] = True
            self.is_heading_env[ids_zero] = False

        # Apply only-lin_y: vx=0, vy from full range, wz=0
        if len(ids_only_y) > 0:
            r_y = torch.empty(len(ids_only_y), device=self.device)
            self.vel_command_b[ids_only_y, 0] = 0.0
            self.vel_command_b[ids_only_y, 1] = r_y.uniform_(*self._full_ranges_lin_vel_y)
            self.vel_command_b[ids_only_y, 2] = 0.0
            self.is_standing_env[ids_only_y] = False
            self.is_heading_env[ids_only_y] = False

        # Apply only-lin_x: vx from full range, vy=0, wz=0
        if len(ids_only_x) > 0:
            r_x = torch.empty(len(ids_only_x), device=self.device)
            self.vel_command_b[ids_only_x, 0] = r_x.uniform_(*self._full_ranges_lin_vel_x)
            self.vel_command_b[ids_only_x, 1] = 0.0
            self.vel_command_b[ids_only_x, 2] = 0.0
            self.is_standing_env[ids_only_x] = False
            self.is_heading_env[ids_only_x] = False

        # Apply only-ang_z: vx=0, vy=0, heading from full range
        if len(ids_only_ang) > 0:
            r_h = torch.empty(len(ids_only_ang), device=self.device)
            self.vel_command_b[ids_only_ang, 0] = 0.0
            self.vel_command_b[ids_only_ang, 1] = 0.0
            self.is_standing_env[ids_only_ang] = False
            self.is_heading_env[ids_only_ang] = True
            if self.cfg.heading_command:
                self.heading_target[ids_only_ang] = r_h.uniform_(*self.cfg.ranges.heading)

        # V12 captures a world route at every long-held command resample, as
        # in the V9 teacher.  The parent command term converts the heading
        # error to the normal 3-D yaw-rate command; no extra Actor input is
        # introduced.  This precedes V11's optional strict-forward override.
        if getattr(self._v6_env.cfg, "v12_teacher_route_commands", False):
            self.v12_path_heading_w[env_ids] = self.robot.data.heading_w[env_ids]
            self.v12_path_origin_w[env_ids] = self.robot.data.root_pos_w[env_ids, :2]
            if self.cfg.hold_heading:
                self.heading_target[env_ids] = self.v12_path_heading_w[env_ids]
                self.is_heading_env[env_ids] = True

        # V11 stair traversal is intentionally a one-dimensional task.  Apply
        # this last so the global standing/lateral/yaw special cases above are
        # retained exactly on non-stair terrain, but cannot leak into either
        # ascending or descending stair columns.
        v11_cfg = getattr(self._v6_env.cfg, "v11_perception", None)
        if v11_cfg is None or not v11_cfg.stair_forward_only_commands:
            return
        stair_types = tuple(v11_cfg.stair_terrain_type_ids)
        if not stair_types:
            raise RuntimeError("V11 stair command override has no resolved stair terrain type IDs.")
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        terrain_types = self._v6_env.scene.terrain.terrain_types[ids]
        stair_type_tensor = torch.tensor(stair_types, device=self.device, dtype=terrain_types.dtype)
        stair_mask = (terrain_types[:, None] == stair_type_tensor[None, :]).any(dim=1)
        stair_ids = ids[stair_mask]
        if stair_ids.numel() == 0:
            return

        def apply_stair_commands(target_ids, forward_min, configured_max):
            if target_ids.numel() == 0:
                return
            forward_min = float(forward_min)
            forward_max = (
                float(self._full_ranges_lin_vel_x[1])
                if configured_max is None else float(configured_max)
            )
            if forward_max > float(self._full_ranges_lin_vel_x[1]):
                raise RuntimeError("V11 stair command maximum exceeds the inherited command range.")
            if not 0.0 < forward_min <= forward_max:
                raise RuntimeError(
                    "V11 stair_forward_command_min must be positive and no larger than commands.ranges.lin_vel_x[1]."
                )
            self.vel_command_b[target_ids, 0] = torch.empty(
                target_ids.numel(), device=self.device,
            ).uniform_(forward_min, forward_max)
            self.vel_command_b[target_ids, 1:] = 0.0
            self.is_standing_env[target_ids] = False
            self.is_heading_env[target_ids] = False
            quat = self.robot.data.root_quat_w[target_ids]
            w, x, y, z = quat.unbind(dim=-1)
            self.v11_path_heading_w[target_ids] = torch.atan2(
                2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square())
            )
            self.v11_path_origin_w[target_ids] = self.robot.data.root_pos_w[target_ids, :2]

        ascending_types = tuple(v11_cfg.ascending_stair_terrain_type_ids)
        ascending_type_tensor = torch.tensor(
            ascending_types, device=self.device, dtype=terrain_types.dtype,
        )
        ascending_mask = (terrain_types[:, None] == ascending_type_tensor[None, :]).any(dim=1)

        if bool(getattr(v11_cfg, "soft_alternation_command_split", False)):
            ascent_ids = ids[stair_mask & ascending_mask]
            descent_ids = ids[stair_mask & ~ascending_mask]
            apply_stair_commands(
                ascent_ids, v11_cfg.stair_forward_command_min,
                getattr(v11_cfg, "stair_forward_command_max", None),
            )
            apply_stair_commands(
                descent_ids, v11_cfg.non_ascending_stair_forward_command_min,
                v11_cfg.non_ascending_stair_forward_command_max,
            )
            longtrain_cfg = getattr(self._v6_env.cfg, "v135_longtrain", None)
            if longtrain_cfg is not None:
                from .v135_longtrain_curriculum import (
                    v135_early_mixture_probability,
                    v135_linear_fraction,
                )

                fraction = v135_linear_fraction(self._v6_env)
                probability = v135_early_mixture_probability(self._v6_env)
                if ascent_ids.numel() > 0 and probability > 0.0:
                    biased = torch.rand(ascent_ids.numel(), device=self.device) < probability
                    biased_ids = ascent_ids[biased]
                    if biased_ids.numel() > 0:
                        self.vel_command_b[biased_ids, 0] = torch.empty(
                            biased_ids.numel(), device=self.device,
                        ).uniform_(*longtrain_cfg.early_ascent_command_range)
                if descent_ids.numel() > 0:
                    low = float(longtrain_cfg.early_descent_command_range[0]) + fraction * (
                        float(v11_cfg.non_ascending_stair_forward_command_min)
                        - float(longtrain_cfg.early_descent_command_range[0])
                    )
                    high = float(longtrain_cfg.early_descent_command_range[1]) + fraction * (
                        float(v11_cfg.non_ascending_stair_forward_command_max)
                        - float(longtrain_cfg.early_descent_command_range[1])
                    )
                    self.vel_command_b[descent_ids, 0] = torch.empty(
                        descent_ids.numel(), device=self.device,
                    ).uniform_(low, high)
        else:
            apply_stair_commands(
                stair_ids, v11_cfg.stair_forward_command_min,
                getattr(v11_cfg, "stair_forward_command_max", None),
            )


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand

    rel_zero_vel_envs: float = 0.07
    """Fraction of environments that always receive a zero-velocity command.

    These samples prevent forgetting of standing-still behavior and are not
    affected by the command-level curriculum.
    """

    rel_only_lin_y_envs: float = 0.07
    """Fraction of environments that receive only a lateral (y) velocity command.

    vx and wz are forced to zero; vy is sampled from the original full range.
    Not affected by the command-level curriculum.
    """

    rel_only_lin_x_envs: float = 0.07
    """Fraction of environments that receive only a forward/backward (x) velocity command.

    vy and wz are forced to zero; vx is sampled from the original full range.
    Not affected by the command-level curriculum.
    """

    rel_only_ang_z_envs: float = 0.07
    """Fraction of environments that receive only an angular-velocity (heading) command.

    vx and vy are forced to zero; heading is sampled from the full range.
    Not affected by the command-level curriculum.
    """

    v6_metrics: bool = False
    """Enable tensorized V6 staircase logging without adding policy observations."""

    perceptive_metrics: bool = False
    """Enable reward-side perceptive stair metrics without adding observations."""


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
