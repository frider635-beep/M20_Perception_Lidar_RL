"""V12: V9 stair teacher rewards with V11's deployable six-point perception."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v11_continuous_env_cfg import DeeproboticsM20V11ContinuousEnvCfg, M20V11RewardsCfg


@configclass
class M20V12TeacherRewardsCfg(M20V11RewardsCfg):
    """V9 dense teacher terms, with V11 landing terms intentionally retained.

    V11's parent initializer still needs its original terms to exist.  They are
    disconnected in the V12 initializer after that parent setup has completed.
    """

    second_front_completion = RewTerm(func=mdp.v12_second_front_completion_reward, weight=5.0)
    second_front_clearance_settlement = RewTerm(
        func=mdp.v12_second_front_clearance_settlement_reward, weight=0.5,
    )
    bilateral_wheel_height_difference = RewTerm(
        func=mdp.v12_bilateral_wheel_height_difference,
        weight=-0.5,
        params={
            "tolerance": 0.04,
            "height_scale": 0.08,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True,
            ),
        },
    )
    front_crossing_time_gap = RewTerm(
        func=mdp.v12_front_phase_time_gap,
        weight=-0.5,
        params={"phase": "crossing", "tolerance_steps": 4, "scale_steps": 10},
    )
    front_support_time_gap = RewTerm(
        func=mdp.v12_front_phase_time_gap,
        weight=-0.5,
        params={"phase": "support", "tolerance_steps": 4, "scale_steps": 10},
    )


@configclass
class DeeproboticsM20V12V9Teacher6DEnvCfg(DeeproboticsM20V11ContinuousEnvCfg):
    """Independent V12 task.  Actor ABI remains V11's 57D + [B1,H1,D1,B2,H2,D2]."""

    rewards: M20V12TeacherRewardsCfg = M20V12TeacherRewardsCfg()
    # Read by UniformThresholdVelocityCommand; never an observation.
    v12_teacher_route_commands: bool = True

    def _resolve_stair_columns(self) -> None:
        terrain_cfg = self.scene.terrain.terrain_generator
        names = list(terrain_cfg.sub_terrains)
        weights = [float(item.proportion) for item in terrain_cfg.sub_terrains.values()]
        total = sum(weights)
        if total <= 0.0:
            raise RuntimeError("V12 terrain generator has no positive terrain proportion.")
        cumulative, running = [], 0.0
        for weight in weights:
            running += weight / total
            cumulative.append(running)
        chosen = [names[next(i for i, upper in enumerate(cumulative) if col / float(terrain_cfg.num_cols) + 0.001 < upper)] for col in range(int(terrain_cfg.num_cols))]
        stairs = tuple(i for i, name in enumerate(chosen) if name in {"pyramid_stairs", "pyramid_stairs_inv"})
        ascending = tuple(i for i, name in enumerate(chosen) if name == "pyramid_stairs_inv")
        if not stairs or not ascending:
            raise RuntimeError("V12 requires both pyramid_stairs and pyramid_stairs_inv terrain columns.")
        if ascending != tuple(range(ascending[0], ascending[-1] + 1)):
            raise RuntimeError("V12 ascending-stair terrain columns must be contiguous for PGTT topology.")
        self.v11_perception.stair_terrain_type_ids = stairs
        self.v11_perception.ascending_stair_terrain_type_ids = ascending
        # V10 PGTT state is still the geometry backend used by V12.  Refresh
        # its topology ledger after V12 changes terrain proportions; otherwise
        # it keeps the parent configuration's stale column interval.
        self.v10_perception.single_stair_column_start = -1
        self.v10_perception.single_stair_column_end = -1
        self.v10_perception.full_stair_column_start = ascending[0]
        self.v10_perception.full_stair_column_end = ascending[-1] + 1
        self.v10_perception.num_terrain_columns = int(terrain_cfg.num_cols)

    def _validate_v9_reward_windows(self) -> None:
        """Fail fast if the V11 PGTT cannot support V9's map reward windows."""
        scanner = self.scene.v10_pgtt_scanner
        size_x, size_y = scanner.pattern_cfg.size
        offset_x = scanner.offset.pos[0]
        x_min, x_max = offset_x - size_x / 2.0, offset_x + size_x / 2.0
        y_half = size_y / 2.0
        if x_min > -0.30 or x_max < 0.90 or y_half < 0.48:
            raise RuntimeError(
                "V12 V9 reward windows require PGTT coverage x=[-0.30, 0.90], |y|<=0.48; "
                f"configured coverage is x=[{x_min:.2f}, {x_max:.2f}], |y|<={y_half:.2f}."
            )

    def __post_init__(self):
        # Set the inherited dispatch contract before V11's initializer runs so
        # V11-only signed descent/turn changes cannot leak into V12.
        self.v11_perception.front_completion_mode = "bilateral_same_riser"
        self.v11_perception.step_feature_mode = "positive_risers"
        self.v11_perception.feature_contract = "v11_positive_risers_v1"
        super().__post_init__()

        # V12 closes the single-side stair loophole without changing the 63-D
        # actor ABI.  Both front wheels must complete the locked riser before
        # its reward state closes; both rear wheels must establish support
        # within one bounded platform window.
        self.v11_perception.require_bilateral_front_completion = True
        # V12 has a dedicated second-front completion/clearance settlement,
        # so preserve its original first-front reward timing.
        self.v11_perception.defer_first_front_rewards_until_bilateral = False
        self.v11_perception.require_bilateral_rear_support = True
        self.v11_perception.bilateral_rear_support_window_frames = 15
        # Alternating-flight diagnostics belong to V11 only.  V12 retains its
        # existing bilateral phase/time-gap instrumentation without dead or
        # misleading expected-side metrics in CurriculumManager.
        self.curriculum.v11_expected_fl_rate = None
        self.curriculum.v11_expected_fr_rate = None
        self.curriculum.v11_completed_risers_mean = None
        self.curriculum.v11_alternation_compliance_rate = None
        self.curriculum.v11_alternation_violation_rate = None
        self.curriculum.v11_crossing_clearance_margin = None
        self.curriculum.v11_post_first_collision_rate = None
        self.curriculum.v11_target_active_rate = None
        self.curriculum.v11_eligible_wheel_rate = None
        self.curriculum.v11_clearance_rate = None
        self.curriculum.v11_crossing_rate = None
        self.curriculum.v11_support_rate = None
        self.curriculum.v11_physical_completion_rate = None
        self.curriculum.v11_rewarded_alternation_rate = None

        # Replace V11's route/contact variants with signed V9 reference terms.
        self.rewards.world_forward_deficit = None
        self.rewards.world_forward_progress = None
        self.rewards.path_lateral_deviation = None
        self.rewards.path_heading_error = None
        self.rewards.non_wheel_leg_contact = None
        self.rewards.non_wheel_leg_scrape = None
        self.rewards.wheel_impact = None
        self.rewards.stair_straightness = None
        # These are active in kunyu_ppo's Rough parent but do not exist in the
        # V9 teacher lineage.  V12 uses V9's terminal fall handling instead.
        self.rewards.rotation_gait_status = None
        self.rewards.rotation_gait_symmetry = None
        self.rewards.feet_air_time_ang_z_M20 = None
        self.rewards.feet_slide_ang_z_cmd = None
        self.rewards.action_smooth_l2 = None
        self.rewards.bad_orientation_penalty = None

        self.scene.num_envs = 4096
        self.scene.terrain.max_init_terrain_level = 0
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        for terrain in terrains.values():
            terrain.proportion = 0.0
        # V9 teacher distribution: 80% stairs, remaining transfer terrains.
        terrains["pyramid_stairs"].proportion = 0.40
        terrains["pyramid_stairs"].step_height_range = (0.04, 0.18)
        terrains["pyramid_stairs"].step_width = 0.35
        terrains["pyramid_stairs_inv"].proportion = 0.40
        terrains["pyramid_stairs_inv"].step_height_range = (0.04, 0.16)
        terrains["pyramid_stairs_inv"].step_width = 0.35
        terrains["random_rough"].proportion = 0.10
        terrains["hf_pyramid_slope"].proportion = 0.05
        terrains["hf_pyramid_slope_inv"].proportion = 0.05
        self._resolve_stair_columns()
        self._validate_v9_reward_windows()

        # V9 held-heading command.  The V11 command override is disabled so
        # heading error is converted into the observable yaw-rate command.
        self.v11_perception.stair_forward_only_commands = False
        command = self.commands.base_velocity
        command.resampling_time_range = (20.0, 20.0)
        command.heading_command = True
        command.rel_heading_envs = 1.0
        command.hold_heading = True
        command.rel_standing_envs = 0.0
        command.rel_yaw_only_envs = 0.0
        command.rel_lateral_only_envs = 0.0
        command.ranges.lin_vel_x = (0.25, 0.65)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.8, 0.8)

        # Stable-from-scratch V9 reset and domain randomization envelope.
        # V10 replaces this with an ascent-conditioned reset.  V9 uses Isaac
        # Lab's uniform root reset on every terrain, so restore both the
        # function and its exact parameter surface before applying V9 ranges.
        self.events.randomize_reset_base.func = mdp.reset_root_state_uniform
        reset = self.events.randomize_reset_base.params
        for stale_name in ("stair_distance_range", "stair_lateral_range", "stair_yaw_jitter_range"):
            reset.pop(stale_name, None)
        reset["pose_range"].update({"x": (-0.50, 0.50), "y": (-0.50, 0.50), "z": (0.0, 0.0), "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-3.14, 3.14)})
        reset["velocity_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05), "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.05, 0.05)}
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.95, 1.05)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.95, 1.05)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.5, 2.0)
        self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.randomize_rigid_body_inertia.params["inertia_distribution_params"] = (0.95, 1.05)
        self.events.randomize_com_positions.params["com_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.03)}
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.6, 1.2)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.5, 1.0)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.1)

        # V9 core tracking, posture, contact and smoothness weights.
        self.rewards.track_lin_vel_xy_exp.weight, self.rewards.track_lin_vel_xy_exp.params["std"] = 2.5, 0.45
        self.rewards.track_lin_vel_y_exp = RewTerm(
            func=mdp.v12_track_lin_vel_y_exp,
            weight=1.5,
            params={"std": 0.35, "command_name": "base_velocity"},
        )
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.is_terminated = RewTerm(func=mdp.is_terminated, weight=-5.0)
        self.rewards.action_rate_l2.weight = -0.014
        self.rewards.lin_vel_z_l2.weight = -3.0
        self.rewards.ang_vel_xy_l2.weight = -0.06
        self.rewards.flat_orientation_l2.weight = -0.35
        # V11's parent removes this zero-weight term; V9 subsequently enables
        # it, so recreate it rather than accessing an inherited ``None``.
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-0.8,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
                "target_height": 0.40,
            },
        )
        self.rewards.contact_forces.weight = -2.0e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # V9 disables ``upward``; V11's parent has already removed the
        # zero-weight term, so it intentionally remains absent in V12.
        # Explicit final V9 inherited regularization ledger (Rough ->
        # RoughBalancedLong -> PerceptiveMoE -> StairTraverse -> V9).
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_l2.weight = -2.0e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -1.0e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_pos_limits = RewTerm(
            func=mdp.joint_pos_limits, weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)},
        )
        self.rewards.joint_power = RewTerm(
            func=mdp.joint_power, weight=-2.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)},
        )
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1, weight=-3.0,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)},
        )
        self.rewards.wheel_vel_penalty = RewTerm(
            func=mdp.wheel_vel_penalty, weight=-0.04,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names), "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]), "command_name": "base_velocity", "velocity_threshold": 0.10, "command_threshold": 0.08},
        )
        self.rewards.joint_mirror = RewTerm(
            func=mdp.joint_mirror, weight=-0.03,
            params={"asset_cfg": SceneEntityCfg("robot"), "mirror_joints": [["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"], ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"]]},
        )
        self.rewards.feet_contact_without_cmd = RewTerm(
            func=mdp.feet_contact_without_cmd, weight=0.25,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]), "command_name": "base_velocity"},
        )
        self.rewards.v12_base_lin_vel_xy_l2_without_cmd = RewTerm(
            func=mdp.v12_base_lin_vel_xy_l2_without_cmd, weight=-2.0,
            params={"command_name": "base_velocity", "command_threshold": 0.08},
        )
        self.rewards.v12_base_ang_vel_z_l2_without_cmd = RewTerm(
            func=mdp.v12_base_ang_vel_z_l2_without_cmd, weight=-0.8,
            params={"command_name": "base_velocity", "command_threshold": 0.08},
        )
        self.rewards.v12_lin_vel_xy_l2_with_ang_z_command = RewTerm(
            func=mdp.v12_lin_vel_xy_l2_with_ang_z_command, weight=-1.0,
            params={"command_name": "base_velocity", "command_threshold": 0.04},
        )
        self.rewards.hipx_joint_pos_penalty.weight = -0.4
        self.rewards.hipx_joint_pos_penalty.params["asset_cfg"].joint_names = self.hipx_joint_names
        # Recreate V9's complete posture terms instead of relying on V11's
        # inherited objects: this locks both joint selection and command gates.
        posture_params = {
            "command_name": "base_velocity",
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        }
        self.rewards.hipy_joint_pos_penalty = RewTerm(
            func=mdp.joint_pos_penalty,
            weight=-0.1,
            params={**posture_params, "asset_cfg": SceneEntityCfg("robot", joint_names=self.hipy_joint_names)},
        )
        self.rewards.knee_joint_pos_penalty = RewTerm(
            func=mdp.joint_pos_penalty,
            weight=-0.1,
            params={**posture_params, "asset_cfg": SceneEntityCfg("robot", joint_names=self.knee_joint_names)},
        )
        self.rewards.feet_stumble = RewTerm(func=mdp.feet_stumble, weight=-0.35, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])})
        self.rewards.foot_impact_velocity = RewTerm(func=mdp.foot_impact_velocity, weight=-0.8, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]), "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]), "speed_threshold": 0.10})

        non_wheel = [f"^(?!({self.base_link_name}|.*_wheel)$).*"]
        self.rewards.v12_non_wheel_leg_contact_force = RewTerm(func=mdp.v12_non_wheel_contact_force, weight=-2.2, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=non_wheel), "force_threshold": 18.0, "force_scale": 180.0})
        self.rewards.v12_non_wheel_leg_scrape = RewTerm(func=mdp.v12_non_wheel_scrape, weight=-1.1, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=non_wheel), "asset_cfg": SceneEntityCfg("robot", body_names=non_wheel), "force_threshold": 8.0, "speed_threshold": 0.08, "speed_scale": 0.55})
        scanner = SceneEntityCfg("v10_pgtt_scanner")
        wheel_bodies = ["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"]
        self.rewards.v12_front_step_approach_clearance = RewTerm(func=mdp.v12_front_step_approach_clearance, weight=-1.0, params={"command_name": "base_velocity", "terrain_sensor_cfg": scanner, "asset_cfg": SceneEntityCfg("robot", body_names=["fl_wheel", "fr_wheel"]), "target_clearance": 0.14, "wheel_radius": 0.09, "top_clearance_margin": 0.015, "step_height_threshold": 0.025, "step_height_scale": 0.018, "command_threshold": 0.12, "near_x_range": (-0.30, 0.18), "front_x_range": (0.28, 0.90), "y_abs_max": 0.48})
        self.rewards.v12_rough_wheel_over_clearance = RewTerm(func=mdp.v12_rough_wheel_over_clearance, weight=-0.5, params={"command_name": "base_velocity", "terrain_sensor_cfg": scanner, "asset_cfg": SceneEntityCfg("robot", body_names=wheel_bodies, preserve_order=True), "allowed_clearance": 0.22, "clearance_scale": 0.07, "command_threshold": 0.12, "terrain_threshold": 0.04, "terrain_scale": 0.02})
        self.rewards.v12_rough_wheel_height_symmetry = RewTerm(func=mdp.v12_rough_wheel_height_symmetry, weight=-0.25, params={"terrain_sensor_cfg": scanner, "asset_cfg": SceneEntityCfg("robot", body_names=wheel_bodies, preserve_order=True), "pair_indices": ((0, 1), (2, 3), (0, 3), (1, 2)), "diff_tolerance": 0.06, "diff_scale": 0.09, "terrain_threshold": 0.04, "terrain_scale": 0.02})
        self.rewards.v12_rough_wheel_impact = RewTerm(func=mdp.v12_rough_wheel_impact, weight=-0.7, params={"terrain_sensor_cfg": scanner, "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]), "force_threshold": 260.0, "force_scale": 350.0, "terrain_threshold": 0.045, "terrain_scale": 0.025})
        self.rewards.v12_rough_wheel_blocked_drive = RewTerm(func=mdp.v12_rough_wheel_blocked_drive, weight=-0.25, params={"command_name": "base_velocity", "terrain_sensor_cfg": scanner, "asset_cfg": SceneEntityCfg("robot", joint_names=self.wheel_joint_names), "wheel_speed_threshold": 8.0, "wheel_speed_scale": 10.0, "base_speed_fraction": 0.35, "terrain_threshold": 0.05, "terrain_scale": 0.025})
        self.rewards.v12_rough_forward_stall = RewTerm(func=mdp.v12_rough_forward_stall, weight=-1.5, params={"command_name": "base_velocity", "terrain_sensor_cfg": scanner, "min_speed_fraction": 0.45, "speed_scale": 0.35, "command_threshold": 0.12, "terrain_threshold": 0.05, "terrain_scale": 0.025})
        self.rewards.v12_flat_leg_motion_l2 = RewTerm(func=mdp.v12_flat_leg_motion_l2, weight=-0.08, params={"command_name": "base_velocity", "terrain_sensor_cfg": scanner, "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names), "terrain_threshold": 0.035, "terrain_scale": 0.015, "joint_vel_weight": 0.02, "command_threshold": 0.08})
        self.rewards.v12_rough_base_speed_limit = RewTerm(func=mdp.v12_rough_base_speed_limit, weight=-0.7, params={"command_name": "base_velocity", "terrain_sensor_cfg": scanner, "max_abs_speed": 0.75, "terrain_threshold": 0.055, "terrain_scale": 0.025})
        self.rewards.world_forward_speed_deficit = RewTerm(func=mdp.v12_world_forward_speed_deficit, weight=-3.0, params={"command_name": "base_velocity", "min_speed_fraction": 0.50, "speed_scale": 0.30, "command_threshold": 0.12})
        self.rewards.world_forward_progress_score = RewTerm(func=mdp.v12_world_forward_progress_score, weight=2.0, params={"command_name": "base_velocity", "target_speed_fraction": 0.50, "command_threshold": 0.12})
        self.rewards.path_lateral_deviation_l2 = RewTerm(func=mdp.v12_path_lateral_deviation_l2, weight=-1.0, params={"command_name": "base_velocity", "tolerance": 0.25, "deviation_scale": 0.35, "command_threshold": 0.12})
        self.rewards.path_heading_error_l2 = RewTerm(func=mdp.v12_path_heading_error_l2, weight=-0.75, params={"command_name": "base_velocity", "heading_scale": 0.5, "command_threshold": 0.12})

        # V9 compares left/right raw action magnitudes throughout the episode;
        # it never releases the front pair during a stair crossing as V11 does.
        self.rewards.front_action_mirror = None
        self.rewards.rear_action_mirror = None
        self.rewards.action_mirror = RewTerm(
            func=mdp.v12_v9_action_term_mirror,
            weight=-0.15,
            params={
                "action_term_name": "joint_pos",
                "mirror_joint_names": [
                    ["fl_(hipx|hipy|knee).*", "fr_(hipx|hipy|knee).*"],
                    ["hl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
                ],
            },
        )
        self.rewards.wheel_action_mirror = RewTerm(
            func=mdp.v12_v9_action_term_mirror,
            weight=-0.05,
            params={
                "action_term_name": "joint_vel",
                "mirror_joint_names": [
                    ["fl_wheel_joint", "fr_wheel_joint"],
                    ["hl_wheel_joint", "hr_wheel_joint"],
                ],
            },
        )

        # Match V9's inherited ``terrain_levels_vel`` quality thresholds.  No
        # V11 platform-confirmation condition is allowed to alter promotion.
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.v12_v9_teacher_terrain_levels,
            params={
                "lin_vel_reward_term_name": "track_lin_vel_xy_exp",
                "ang_vel_reward_term_name": "track_ang_vel_z_exp",
                "forward_progress_reward_term_name": "world_forward_progress_score",
                "move_up_tracking_threshold": 0.50,
                "move_down_tracking_threshold": 0.32,
                "move_up_forward_progress_threshold": 0.45,
                "move_down_forward_progress_threshold": 0.20,
                "random_reset_at_max": True,
                "allow_move_down": True,
            },
        )
        # V9 fixes this at one: terrain level zero must not silently disable
        # posture/smoothness regularizers during from-scratch acquisition.
        self.curriculum.gait_level = CurrTerm(func=mdp.v12_fixed_gait_level, params={"level": 1.0})
        self.curriculum.terrain_tracking_score_mean = CurrTerm(func=mdp.v12_terrain_curriculum_debug, params={"key": "tracking_score_mean"})
        self.curriculum.terrain_tracking_score_min = CurrTerm(func=mdp.v12_terrain_curriculum_debug, params={"key": "tracking_score_min"})
        self.curriculum.terrain_forward_progress_score_mean = CurrTerm(func=mdp.v12_terrain_curriculum_debug, params={"key": "forward_progress_score_mean"})
        self.curriculum.terrain_forward_progress_score_min = CurrTerm(func=mdp.v12_terrain_curriculum_debug, params={"key": "forward_progress_score_min"})
        self.curriculum.terrain_timeout_rate = CurrTerm(func=mdp.v12_terrain_curriculum_debug, params={"key": "timeout_rate"})
        self.curriculum.terrain_platform_success_rate = None
        self.curriculum.terrain_move_up_rate = CurrTerm(func=mdp.v12_terrain_curriculum_debug, params={"key": "move_up_rate"})
        self.curriculum.terrain_move_down_rate = CurrTerm(func=mdp.v12_terrain_curriculum_debug, params={"key": "move_down_rate"})
        # The V11 parent already removed zero-weight terms.  Do not call its
        # legacy helper again: V12 explicitly uses ``None`` to disconnect
        # non-V9 kunyu reward terms, and that helper predates such entries.


__all__ = ["DeeproboticsM20V12V9Teacher6DEnvCfg", "M20V12TeacherRewardsCfg"]
