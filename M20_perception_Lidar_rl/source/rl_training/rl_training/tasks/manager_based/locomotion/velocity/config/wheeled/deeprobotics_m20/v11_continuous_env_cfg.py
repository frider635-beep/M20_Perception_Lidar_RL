"""M20 V11 continuous two-riser perception environment."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v10_unified_env_cfg import DeeproboticsM20V10UnifiedEnvCfg, M20V10EventRewardsCfg


@configclass
class M20V11PerceptionCfg:
    # V11 credits one front wheel per successive riser.  V12 explicitly
    # overrides this to ``bilateral_same_riser`` to preserve its teacher task.
    front_completion_mode: str = "alternating_risers"
    # Opt-in only: the legacy V11/V11-FD/V13 tasks retain their existing
    # lifecycle unless the dedicated V11 SoftAlt config enables this switch.
    soft_alternation: bool = False
    # Command redistribution is a separate opt-in from the physical
    # soft-candidate lifecycle.  V13 needs the latter but must retain the
    # reference V11-FD command distribution unchanged.
    soft_alternation_command_split: bool = False
    # Existing SoftAlt uses alternation as an ascending-terrain promotion
    # gate. Reward-only descendants may disable that gate without changing
    # physical candidate or collision semantics.
    soft_alternation_curriculum_gate: bool = True
    step_feature_mode: str = "signed_edges"
    feature_contract: str = "v11_signed_edges_v2"
    min_step_height: float = 0.05
    max_step_height: float = 0.23
    height_valid_tolerance: float = 0.002
    anticipation_distance_min: float = 0.02
    anticipation_distance_max: float = 1.20
    passed_distance: float = -0.30
    min_riser_separation: float = 0.15
    forward_command_threshold: float = 0.05
    # Keep the stair traversal task one-dimensional: both ascending and
    # descending stair columns receive a strictly forward command.  The upper
    # bound remains commands.ranges.lin_vel_x[1], so non-stair command ranges
    # and their sampling distribution remain unchanged.
    stair_forward_only_commands: bool = True
    stair_forward_command_min: float = 0.10
    stair_forward_command_max: float | None = None
    soft_alternation_debug_env_ids: tuple[int, ...] = ()
    soft_alternation_debug_max_events: int = 12
    # Resolved from the inherited terrain generator in __post_init__.  This is
    # deliberately a list of terrain-type IDs rather than a column interval:
    # the two stair variants need not be adjacent in future generators.
    stair_terrain_type_ids: tuple[int, ...] = ()
    ascending_stair_terrain_type_ids: tuple[int, ...] = ()
    corridor_half_width: float = 0.20
    confirmation_frames: int = 2
    hysteresis_frames: int = 2
    perception_delay_steps: int = 2
    height_noise_std: float = 0.005
    distance_noise_std: float = 0.010
    frame_hold_probability: float = 0.02
    front_margin: float = 0.01
    lateral_margin: float = 0.01
    # This is the observed high-map approach corridor, not a terrain-sidewall
    # oracle.  A target with no wheel-centre rectangle is invalid by design.
    tread_corridor_half_width: float = 0.20
    wheel_radius: float = 0.09
    clearance_margin: float = 0.01
    # Legacy V11 only required reaching clearance before the crossing.  New
    # tasks can additionally require the same target-relative clearance on
    # the exact control step where the wheel centre crosses the face.
    require_clearance_at_crossing: bool = False
    contact_force_threshold: float = 1.0
    vertical_force_ratio: float = 0.35
    vertical_contact_confirmation_frames: int = 3
    front_support_confirmation_frames: int = 3
    alternation_violation_confirmation_frames: int = 3
    upper_support_tolerance: float = 0.04
    encounter_riser_switch_distance: float = 0.15
    target_loss_release_steps: int = 3
    # Optional longer grace after a front completion while the rear pair is
    # still crossing onto the top platform.  None preserves every legacy task.
    platform_target_loss_release_steps: int | None = None
    # A platform is declared only after the raw map stays riser-free and rear
    # support persists; 6-D B1/B2=0 alone is deliberately insufficient.
    platform_no_riser_confirmation_frames: int = 3
    platform_rear_support_confirmation_frames: int = 2
    # V11 requires bilateral front completion and bilateral rear support inside
    # a bounded window; this safety contract does not change the actor ABI.
    require_bilateral_front_completion: bool = True
    # V11 has one main front-wheel reward contract.  Defer its first-wheel
    # payout until the second front wheel has also completed the same riser.
    defer_first_front_rewards_until_bilateral: bool = True
    require_bilateral_rear_support: bool = True
    bilateral_rear_support_window_frames: int = 15
    # The old V11 bilateral escrow summed two values in [0, 1].  Alternating
    # V11 settles one value per riser, so this keeps the maximum per-riser
    # contribution unchanged without changing the configured reward weight.
    alternating_clearance_settlement_scale: float = 2.0
    clearance_surplus_scale: float = 0.02
    enable_post_first_clearance_surplus: bool = False
    post_first_clearance_surplus_weight: float = 0.10
    descent_front_support_confirmation_frames: int = 3
    descent_rear_support_confirmation_frames: int = 3
    descent_vertical_velocity_penalty_scale: float = 0.25
    descent_rear_follow_grace_frames: int = 25
    # Off by default so legacy V11 variants retain their existing descent
    # lifecycle.  V11-FD opts in to the persistent front-to-rear closure.
    enable_persistent_descent_rear_phase: bool = False
    # Once both front wheels are confirmed on the lower tread, the locked edge
    # may leave the forward scan while the rear wheels follow.  This bounds
    # that retained phase so a failed descent cannot hold stale geometry.
    descent_rear_follow_timeout_frames: int = 200
    descent_rear_roll_speed_scale: float = 0.50


@configclass
class M20V11RewardsCfg(M20V10EventRewardsCfg):
    """Official M20 rewards plus V11 terms; V10 FR terms are disconnected."""

    # RewardManager multiplies continuous terms by step_dt.  Event terms stay
    # as one-shot pulses; the progress function is explicitly dt-normalized.
    landing_event = RewTerm(func=mdp.v11_landing_event_reward, weight=5.0)
    platform_exit_success = RewTerm(func=mdp.v11_platform_exit_success_reward, weight=5.0)
    # Clearance is escrowed and paid only after the same wheel has crossed the
    # locked riser and established upper-tread support.  Keep it subordinate
    # to the safe-landing event instead of rewarding a stationary high wheel.
    clearance_progress = RewTerm(func=mdp.v11_clearance_progress_reward, weight=0.5)
    # Installed by __post_init__ only for the second-stage A/B arm.
    post_first_clearance_surplus = None
    descent_step_complete = RewTerm(func=mdp.v11_descent_step_complete_reward, weight=3.0)
    descent_rear_follow_delay = RewTerm(func=mdp.v11_descent_rear_follow_delay_penalty, weight=-0.5)
    descent_rear_roll_consistency = RewTerm(
        func=mdp.v11_descent_rear_roll_consistency_penalty, weight=-0.2,
    )
    pure_yaw_xy_drift = RewTerm(
        func=mdp.v11_pure_yaw_xy_drift_penalty, weight=-0.5,
        params={"cmd_lin_threshold": 0.1, "cmd_ang_threshold": 0.1, "speed_scale": 0.35},
    )
    # Store the latest bounded pose contribution during the visible approach,
    # then pay it once only after clearance -> crossing -> support.  A transient
    # good pose followed by a knee-only landing overwrites, rather than keeps,
    # the earlier score.
    front_hipy_participation_settlement = RewTerm(
        func=mdp.v11_front_hipy_participation_settlement_reward,
        weight=0.10,
        params={
            "front_hipy_joint_names": ("fl_hipy_joint", "fr_hipy_joint"),
            "front_knee_joint_names": ("fl_knee_joint", "fr_knee_joint"),
            "target_share": 0.50,
            "share_half_width": 0.10,
            "activation_scale": 0.20,
        },
    )
    # Direct active-side knee barrier.  Unlike the inherited nominal-position
    # term, this is not multiplied by the generic high-terrain 0.1 relaxation.
    # The empirically successful early policy stays inside the 0.55-rad free
    # band; only excess folding is penalized, with a bounded maximum cost.
    front_knee_excess = RewTerm(
        func=mdp.v11_front_knee_excess_penalty,
        weight=-0.20,
        params={
            "front_knee_joint_names": ("fl_knee_joint", "fr_knee_joint"),
            "free_band": 0.55,
            "excess_scale": 0.25,
        },
    )
    landing_center = RewTerm(func=mdp.v11_landing_center_reward, weight=0.10)
    landing_edge = RewTerm(func=mdp.v11_landing_edge_penalty, weight=0.20)
    # V9-style world-route supervision.  These terms are gated to V11 stairs
    # and use command-side state only, so they add no actor observation.
    world_forward_deficit = RewTerm(
        func=mdp.v11_world_forward_speed_deficit, weight=3.0,
        params={"min_speed_fraction": 0.50, "speed_scale": 0.30},
    )
    world_forward_progress = RewTerm(
        func=mdp.v11_world_forward_progress_reward, weight=2.0,
        params={"target_speed_fraction": 0.50},
    )
    path_lateral_deviation = RewTerm(
        func=mdp.v11_path_lateral_deviation_penalty, weight=1.0,
        params={"tolerance": 0.25, "deviation_scale": 0.35},
    )
    path_heading_error = RewTerm(
        func=mdp.v11_path_heading_error_penalty, weight=0.75,
        params={"heading_scale": 0.50},
    )
    # Collision and hard-touchdown coverage complements the V11 front-riser
    # MISS event, which intentionally remains responsibility-wheel specific.
    non_wheel_leg_contact = RewTerm(
        func=mdp.v11_non_wheel_contact_force_penalty,
        weight=2.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!(base_link|.*_wheel)$).*"]),
            "force_threshold": 20.0, "force_scale": 180.0, "max_penalty": 1.0,
        },
    )
    non_wheel_leg_scrape = RewTerm(
        func=mdp.v11_non_wheel_scrape_penalty,
        weight=1.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!(base_link|.*_wheel)$).*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["^(?!(base_link|.*_wheel)$).*"]),
            "force_threshold": 8.0, "speed_threshold": 0.08, "speed_scale": 0.55, "max_penalty": 1.0,
        },
    )
    wheel_impact = RewTerm(
        func=mdp.v11_wheel_impact_penalty,
        weight=0.8,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_wheel"]),
            "speed_threshold": 0.10, "max_penalty": 1.0,
        },
    )
    # Small, bounded stair-only regularizer.  Main command tracking remains
    # dominant; this only suppresses lateral drift and yaw while climbing.
    # Restore the recorded V11 run's small straightness regularizer.  Its
    # reward set contained no split front/rear action-mirroring terms.
    stair_straightness = RewTerm(func=mdp.v11_stair_straightness_penalty, weight=0.05)
    front_riser_contact = RewTerm(func=mdp.v11_front_riser_contact_penalty, weight=-1.0)


@configclass
class DeeproboticsM20V11ContinuousEnvCfg(DeeproboticsM20V10UnifiedEnvCfg):
    rewards: M20V11RewardsCfg = M20V11RewardsCfg()
    v11_perception: M20V11PerceptionCfg = M20V11PerceptionCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.v11_perception.front_completion_mode not in {
            "alternating_risers", "bilateral_same_riser",
        }:
            raise RuntimeError(
                "V11 front_completion_mode must be 'alternating_risers' or "
                "'bilateral_same_riser'."
            )
        if self.v11_perception.step_feature_mode not in {"signed_edges", "positive_risers"}:
            raise RuntimeError("V11 step_feature_mode must be 'signed_edges' or 'positive_risers'.")
        alternating_v11 = self.v11_perception.front_completion_mode == "alternating_risers"
        terrain_cfg = self.scene.terrain.terrain_generator
        if alternating_v11:
            existing = terrain_cfg.sub_terrains
            required = {
                "pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough",
                "hf_pyramid_slope", "hf_pyramid_slope_inv",
            }
            if not required.issubset(existing):
                raise RuntimeError(f"V11 terrain generator is missing {sorted(required - set(existing))}.")
            terrain_cfg.sub_terrains = {
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
                "pyramid_stairs": existing["pyramid_stairs"],
                "pyramid_stairs_inv": existing["pyramid_stairs_inv"],
                "boxes": existing["boxes"],
                "random_rough": existing["random_rough"],
                "hf_pyramid_slope": existing["hf_pyramid_slope"],
                "hf_pyramid_slope_inv": existing["hf_pyramid_slope_inv"],
            }
            for name, proportion in {
                "pyramid_stairs": 0.20, "pyramid_stairs_inv": 0.20,
                "boxes": 0.15, "random_rough": 0.15,
                "hf_pyramid_slope": 0.10, "hf_pyramid_slope_inv": 0.10,
            }.items():
                terrain_cfg.sub_terrains[name].proportion = proportion
        sub_names = list(terrain_cfg.sub_terrains)
        proportions = [float(item.proportion) for item in terrain_cfg.sub_terrains.values()]
        total_proportion = sum(proportions)
        if total_proportion <= 0.0:
            raise RuntimeError("V11 terrain generator must assign positive terrain proportions.")
        cumulative = []
        running = 0.0
        for proportion in proportions:
            running += proportion / total_proportion
            cumulative.append(running)
        column_names = []
        for column in range(int(terrain_cfg.num_cols)):
            choice = column / float(terrain_cfg.num_cols) + 0.001
            sub_index = next(index for index, upper in enumerate(cumulative) if choice < upper)
            column_names.append(sub_names[sub_index])
        stair_columns = tuple(
            column
            for column, name in enumerate(column_names)
            if name in {"pyramid_stairs", "pyramid_stairs_inv"}
        )
        ascending_stair_columns = tuple(
            column for column, name in enumerate(column_names) if name == "pyramid_stairs_inv"
        )
        if not stair_columns or not ascending_stair_columns:
            raise RuntimeError("V11 terrain generator must contain at least one stair terrain column.")
        self.v11_perception.stair_terrain_type_ids = stair_columns
        self.v11_perception.ascending_stair_terrain_type_ids = ascending_stair_columns

        # Replace V10's distance-only curriculum.  The V9 teacher quality gate
        # protects all terrain columns; V11 adds stair-only forward progress
        # and confirmed top-platform completion for ascending stairs.
        self.curriculum.terrain_levels = CurrTerm(func=mdp.v11_teacher_gated_terrain_levels)
        self.curriculum.terrain_tracking_score_mean = CurrTerm(
            func=mdp.v11_terrain_curriculum_debug, params={"key": "tracking_score_mean"}
        )
        self.curriculum.terrain_forward_progress_score_mean = CurrTerm(
            func=mdp.v11_terrain_curriculum_debug, params={"key": "forward_progress_score_mean"}
        )
        self.curriculum.terrain_platform_success_rate = CurrTerm(
            func=mdp.v11_terrain_curriculum_debug, params={"key": "stair_platform_success_rate"}
        )
        self.curriculum.terrain_move_up_rate = CurrTerm(
            func=mdp.v11_terrain_curriculum_debug, params={"key": "move_up_rate"}
        )
        self.curriculum.terrain_move_down_rate = CurrTerm(
            func=mdp.v11_terrain_curriculum_debug, params={"key": "move_down_rate"}
        )
        self.curriculum.v11_responsible_fl_rate = CurrTerm(
            func=mdp.v11_responsibility_metric, params={"side": "fl_rate"}
        )
        self.curriculum.v11_responsible_fr_rate = CurrTerm(
            func=mdp.v11_responsibility_metric, params={"side": "fr_rate"}
        )
        self.curriculum.v11_responsibility_tie_rate = CurrTerm(
            func=mdp.v11_responsibility_metric, params={"side": "tie_rate"}
        )
        self.curriculum.v11_front_mobility_gate_rate = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "gate_rate"}
        )
        self.curriculum.v11_front_hipy_deviation = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "hipy_deviation"}
        )
        self.curriculum.v11_front_knee_deviation = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "knee_deviation"}
        )
        self.curriculum.v11_front_hipy_share = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "hipy_share"}
        )
        self.curriculum.v11_front_knee_excess = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "knee_excess"}
        )
        self.curriculum.v11_front_knee_excess_rate = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "knee_excess_rate"}
        )
        self.curriculum.v11_front_hipy_escrow = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "hipy_escrow"}
        )
        self.curriculum.v11_front_upperleg_torso_angle = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "upperleg_torso_angle"}
        )
        self.curriculum.v11_front_upperleg_parallel_cosine = CurrTerm(
            func=mdp.v11_front_mobility_metric, params={"key": "upperleg_parallel_cosine"}
        )
        self.curriculum.v11_front_riser_contact_rate = CurrTerm(
            func=mdp.v11_front_collision_metric, params={"key": "contact_rate"}
        )
        self.curriculum.v11_front_collision_event_rate = CurrTerm(
            func=mdp.v11_front_collision_metric, params={"key": "event_rate"}
        )
        self.curriculum.v11_front_collision_latched_rate = CurrTerm(
            func=mdp.v11_front_collision_metric, params={"key": "latched_rate"}
        )
        self.curriculum.v11_expected_fl_rate = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "expected_fl_rate"}
        )
        self.curriculum.v11_expected_fr_rate = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "expected_fr_rate"}
        )
        self.curriculum.v11_completed_risers_mean = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "completed_risers_mean"}
        )
        self.curriculum.v11_alternation_compliance_rate = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "compliance_rate"}
        )
        self.curriculum.v11_alternation_violation_rate = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "violation_rate"}
        )
        self.curriculum.v11_crossing_clearance_margin = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "crossing_clearance_margin"}
        )
        self.curriculum.v11_post_first_collision_rate = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "post_first_collision_rate"}
        )
        self.curriculum.v11_anchor_valid_rate = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "anchor_valid_rate"}
        )
        self.curriculum.v11_anchor_support_steps = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "anchor_support_steps"}
        )
        self.curriculum.v11_clearance_surplus = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "clearance_surplus"}
        )
        self.curriculum.v11_confirmed_violation_rate = CurrTerm(
            func=mdp.v11_alternation_metric, params={"key": "confirmed_violation_rate"}
        )
        self.curriculum.v11_target_active_rate = CurrTerm(
            func=mdp.v11_event_reachability_metric, params={"key": "target_active"}
        )
        self.curriculum.v11_eligible_wheel_rate = CurrTerm(
            func=mdp.v11_event_reachability_metric, params={"key": "eligible_wheel"}
        )
        self.curriculum.v11_clearance_rate = CurrTerm(
            func=mdp.v11_event_reachability_metric, params={"key": "clearance"}
        )
        self.curriculum.v11_crossing_rate = CurrTerm(
            func=mdp.v11_event_reachability_metric, params={"key": "crossing"}
        )
        self.curriculum.v11_support_rate = CurrTerm(
            func=mdp.v11_event_reachability_metric, params={"key": "support"}
        )
        self.curriculum.v11_physical_completion_rate = CurrTerm(
            func=mdp.v11_event_reachability_metric, params={"key": "physical_completion"}
        )
        self.curriculum.v11_rewarded_alternation_rate = CurrTerm(
            func=mdp.v11_event_reachability_metric, params={"key": "rewarded_alternation"}
        )
        for key in (
            "target_active", "front_complete", "rear_complete", "completion_event",
            "rear_support_event_rate", "rear_follow_gap_steps", "rear_roll_error", "three_plus_contact_rate",
        ):
            setattr(
                self.curriculum, f"v11_descent_{key}",
                CurrTerm(func=mdp.v11_descent_metric, params={"key": key}),
            )
        for key in ("yaw_error", "xy_speed", "action_delta"):
            setattr(
                self.curriculum, f"v11_turn_{key}",
                CurrTerm(func=mdp.v11_turn_metric, params={"key": key}),
            )

        # V11's deployable observation is official 57D + two [B,H,D] tuples.
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.v11_actor_perception,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None, clip=(-2.0, 2.0), scale=1.0,
        )
        self.observations.critic.v10_pgtt = ObsTerm(
            func=mdp.v11_critic_pgtt,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")}, noise=None,
        )
        self.observations.critic.v10_geometry = ObsTerm(
            func=mdp.v11_critic_geometry,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")}, noise=None,
        )
        self.observations.critic.v10_teacher = None

        # Fail closed: V10's FR-specific teaching terms cannot leak into V11.
        self.rewards.first_step_success = None
        self.rewards.perception_miss = None
        self.rewards.joint_success = None
        self.rewards.fr_clearance_progress = None
        self.rewards.post_first_track_lin_vel_xy_exp = None
        if alternating_v11:
            self.rewards.feet_air_time_ang_z_M20 = None
            self.rewards.rotation_gait_status = None
            self.rewards.rotation_gait_symmetry = None
        else:
            self.rewards.descent_step_complete = None
            self.rewards.descent_rear_follow_delay = None
            self.rewards.descent_rear_roll_consistency = None
            self.rewards.pure_yaw_xy_drift = None
        if alternating_v11 and self.v11_perception.enable_post_first_clearance_surplus:
            self.rewards.post_first_clearance_surplus = RewTerm(
                func=mdp.v11_post_first_clearance_surplus_reward,
                weight=self.v11_perception.post_first_clearance_surplus_weight,
            )
        # Stair pitch is necessary for climbing; use V9's bounded posture and
        # joint-limit regularization instead of inherited flat-locomotion costs.
        # Posture-stability fine-tune: retain enough pitch for a riser, while
        # making persistent nose-down/roll compensation less attractive.
        self.rewards.flat_orientation_l2.weight = -0.50
        self.rewards.flat_orientation_l2.func = mdp.v11_phase_flat_orientation_l2
        self.rewards.lin_vel_z_l2.func = mdp.v11_phase_lin_vel_z_l2
        self.rewards.ang_vel_xy_l2.weight = -0.06
        self.rewards.action_rate_l2 = RewTerm(
            func=mdp.v11_phase_relaxed_action_rate_l2,
            weight=-0.014,
            params={"hipy_active_scale": 0.35, "knee_active_scale": 1.00},
        )
        self.rewards.action_smooth_l2 = RewTerm(
            func=mdp.v11_phase_relaxed_action_smooth_l2,
            weight=-0.025,
            params={"hipy_active_scale": 0.50, "knee_active_scale": 0.80},
        )
        # These V11 terms return negative costs, so their positive weights are
        # penalty magnitudes.  Tighten the straight-line contract only on
        # forward stair traversal; normal non-stair commands are unaffected.
        self.rewards.path_lateral_deviation.weight = 1.35
        self.rewards.path_heading_error.weight = 1.00
        self.rewards.hipx_joint_pos_penalty.weight = -0.4
        self.rewards.hipy_joint_pos_penalty.weight = -0.1
        self.rewards.knee_joint_pos_penalty.weight = -0.1
        self.rewards.hipy_joint_pos_penalty.func = mdp.v11_phase_relaxed_front_joint_pos_penalty
        self.rewards.hipy_joint_pos_penalty.params.update({
            "front_joint_names": ("fl_hipy_joint", "fr_hipy_joint"),
            "active_scale": 0.10,
        })
        self.rewards.knee_joint_pos_penalty.func = mdp.v11_phase_relaxed_front_joint_pos_penalty
        self.rewards.knee_joint_pos_penalty.params.update({
            "front_joint_names": ("fl_knee_joint", "fr_knee_joint"),
            "active_scale": 0.80,
        })
        # ``disable_zero_weight_rewards`` runs in the parent configuration and
        # removes this inherited term because its default weight is zero.  V11
        # enables it again, so recreate the complete term rather than assuming
        # the inherited object still exists.
        self.rewards.joint_pos_limits = RewTerm(
            func=mdp.joint_pos_limits,
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names)},
        )

        # Treat base contact and bad orientation as failures, and make the
        # non-timeout terminal transition visible to PPO as in V9.
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.base_link_name]), "threshold": 1.0},
        )
        self.terminations.bad_orientation_2 = DoneTerm(func=mdp.bad_orientation_2)
        self.terminations.front_riser_collision = DoneTerm(func=mdp.v11_confirmed_front_riser_contact)
        self.rewards.is_terminated = RewTerm(func=mdp.is_terminated, weight=-5.0)


__all__ = ["DeeproboticsM20V11ContinuousEnvCfg", "M20V11PerceptionCfg"]
