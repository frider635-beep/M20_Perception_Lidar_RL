"""V11.1: saved V11-FD baseline plus the pre-thigh rear posture repair."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v11_flat_descent_env_cfg import DeeproboticsM20V11FlatDescentEnvCfg
from .v135_v11fd_native_alternation_env_cfg import (
    M20V135DescentSupportTransferCfg, M20V135PostStairFlatCfg, M20V135LongTrainCfg,
)


@configclass
class M20V111RearPostureCfg:
    """Single numerical ledger for the three retained descent bounds."""

    timed_support_cycle: bool = False
    leader_support_seconds: float = 0.16
    pair_support_seconds: float = 0.12
    support_loss_seconds: float = 0.04
    hold_grace_seconds: float = 0.60
    cycle_contact_force: float = 15.0
    cycle_slip_limit: float = 0.25
    cycle_plane_tolerance: float = 0.04
    cycle_drive_floor: float = 0.60
    hold_drive_scale: float = 0.25
    cycle_progress_weight: float = 1.0
    cycle_completion_weight: float = 1.0
    support_credit_floor: float = 0.25
    flat_sample_radius: float = 0.25
    flat_lookahead: float = 0.60
    flat_half_width: float = 0.25
    flat_plane_tolerance: float = 0.03
    flat_confirmation_seconds: float = 0.10
    flat_clearance_min: float = 0.40
    flat_clearance_scale: float = 0.10
    flat_tilt_free: float = 0.05
    flat_tilt_scale: float = 0.15
    flat_rate_free: float = 0.30
    flat_rate_scale: float = 1.0
    flat_posture_weight: float = 2.0
    same_tread_handoff: bool = False
    front_airborne_confirmation_frames: int = 2
    front_airborne_min_lift: float = 0.04
    physical_first_tread: bool = True
    adjacent_landing_credit: bool = False
    directional_exit_evidence: bool = True
    rear_extension_free_reach: float = 0.05
    rear_extension_scale: float = 0.12
    rear_extension_max_penalty: float = 1.0
    rear_extension_weight: float = 0.75
    rear_knee_flexion_free: float = 0.20
    rear_knee_flexion_scale: float = 0.25
    rear_knee_flexion_max_penalty: float = 1.0
    rear_knee_flexion_weight: float = 0.50
    rear_roll_weight: float = 0.10
    # Contact-backed level ground at either end of a stair flight. This uses
    # root height above the wheel support plane and no leg-angle target.
    exit_confirmation_frames: int = 5
    exit_support_height_tolerance: float = 0.06
    exit_body_clearance_free_height: float = 0.38
    exit_body_clearance_scale: float = 0.10
    exit_body_clearance_max_penalty: float = 1.0
    exit_body_clearance_weight: float = 1.0


@configclass
class DeeproboticsM20V111V11FDRearPostureEnvCfg(DeeproboticsM20V11FlatDescentEnvCfg):
    """Keep the saved V11-FD run intact except for three rear-leg bounds."""

    v111_rear_posture: M20V111RearPostureCfg = M20V111RearPostureCfg()
    v135_descent_support_transfer: M20V135DescentSupportTransferCfg = M20V135DescentSupportTransferCfg()
    v135_post_stair_flat: M20V135PostStairFlatCfg = M20V135PostStairFlatCfg()
    v135_longtrain: M20V135LongTrainCfg = M20V135LongTrainCfg()

    def __post_init__(self):
        super().__post_init__()
        cfg = self.v111_rear_posture
        # Preserve the saved V11-FD ascent rewards. Only physical B1 entry
        # geometry and descent/exit repair are installed in this branch.
        self.v135_longtrain.descent_velocity_baseline_initial = 0.60
        self.v135_longtrain.descent_velocity_baseline_final = 0.60
        # V11 retains bilateral ascent and its native signed-edge observation.
        # Only the proven descent and exit state machines are shared with V13.5.
        self.rewards.descent_step_complete = None
        self.rewards.descent_rear_follow_delay = None
        self.rewards.descent_rear_roll_consistency = None
        self.rewards.descent_rear_support_progress = None
        self.rewards.v111_descent_rear_extension = None
        self.rewards.v111_descent_rear_knee_flexion = None
        self.rewards.v111_descent_rear_roll = None
        self.rewards.v111_level_exit_body_clearance = None
        self.rewards.v135_descent_rear_transfer_progress = RewTerm(
            func=mdp.v135_descent_rear_transfer_progress_reward, weight=0.8,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
        )
        self.rewards.v135_descent_rear_transfer_completion = RewTerm(
            func=mdp.v135_descent_rear_transfer_completion_reward, weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
        )
        self.rewards.v135_descent_rear_recovery = RewTerm(
            func=mdp.v135_descent_rear_recovery_reward, weight=0.5,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
        )
        self.rewards.v135_post_stair_flat_progress = RewTerm(
            func=mdp.v135_post_stair_flat_progress_reward, weight=0.5,
        )
        self.rewards.v135_post_stair_flat_completion = RewTerm(
            func=mdp.v135_post_stair_flat_completion_reward, weight=1.0,
        )
        self.rewards.v135_post_stair_flat_crawl = RewTerm(
            func=mdp.v135_post_stair_flat_crawl_penalty, weight=0.5,
        )
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.v111_track_lin_vel_with_descent_support, weight=5.0,
            params=dict(self.rewards.track_lin_vel_xy_exp.params),
        )
        self.rewards.world_forward_progress = RewTerm(
            func=mdp.v111_world_forward_with_descent_support, weight=2.0,
            params={"target_speed_fraction": 0.5},
        )
        self.rewards.world_forward_deficit = RewTerm(
            func=mdp.v111_world_forward_deficit_with_descent_support, weight=3.0,
            params={"min_speed_fraction": 0.5, "speed_scale": 0.3},
        )
        # V11.1 keeps its original terrain/command distribution. The shared
        # schedule reads the restored V11 update count and is already annealed.

        # The named 2026-09-02 run is the baseline authority.  Restore source
        # drift added after that run before installing the V11.1 delta.
        self.scene.terrain.max_init_terrain_level = 5
        for terrain in self.scene.terrain.terrain_generator.sub_terrains.values():
            terrain.size = (8.0, 8.0)
        for name in ("random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"):
            self.scene.terrain.terrain_generator.sub_terrains[name].slope_threshold = 0.75
        self.v11_perception.enable_persistent_descent_rear_phase = False
        self.rewards.safe_front_landing_progress = None
        self.rewards.single_front_hard_landing = None
        self.rewards.descent_rear_support_progress = None
        for name, keys in {
            "action_rate_l2": ("descent_rear_hipy_scale", "descent_rear_knee_scale"),
            "action_smooth_l2": ("descent_rear_hipy_scale", "descent_rear_knee_scale"),
            "hipy_joint_pos_penalty": ("descent_rear_joint_names", "descent_rear_active_scale"),
            "knee_joint_pos_penalty": ("descent_rear_joint_names", "descent_rear_active_scale"),
        }.items():
            params = getattr(self.rewards, name).params
            for key in keys:
                params.pop(key, None)
        self.rewards.joint_mirror.func = mdp.joint_mirror
        self.rewards.joint_mirror.params.pop("descent_rear_follow_scale", None)
        for name in (
            "v11_descent_rear_support_event_rate", "v11_fd_safe_support_event_rate",
            "v11_fd_hard_candidate_event_rate",
        ):
            setattr(self.curriculum, name, None)

        for key in (
            "course_active_rate", "rear_extension_max_mean",
            "rear_contact_fraction", "rear_roll_error_mean",
            "rear_knee_flexion_mean", "rear_knee_flexion_rate",
            "top_exit_active_rate", "bottom_exit_active_rate",
            "exit_body_clearance_mean", "exit_low_body_rate",
        ):
            setattr(self.curriculum, f"v111_{key}", None)

        for key in ("active_rate", "rear_extension_ratio", "rear_length_min_ratio",
                    "recovery_event_rate", "support_quality", "whole_body_slip",
                    "whole_body_workspace", "terrain_pitch_error", "ground_evidence_rate",
                    "rear_load_share", "four_wheel_support", "body_path_speed",
                    "body_pitch_rate_abs", "body_rear_lag"):
            setattr(self.curriculum, f"v111_transfer_{key}", CurrTerm(
                func=mdp.v135_descent_support_transfer_metric,
                params={"key": key, "sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            ))
        for key in ("traversal_seen_rate", "active_rate", "quality",
                    "completion_event_rate", "completion_count", "flat_blend"):
            setattr(self.curriculum, f"v111_flat_{key}", CurrTerm(
                func=mdp.v135_post_stair_flat_metric, params={"key": key},
            ))

        for name in (
            "rear_extension_scale", "rear_extension_max_penalty",
            "rear_extension_weight", "rear_knee_flexion_scale",
            "rear_knee_flexion_max_penalty", "rear_knee_flexion_weight",
            "rear_roll_weight", "exit_support_height_tolerance",
            "exit_body_clearance_free_height", "exit_body_clearance_scale",
            "exit_body_clearance_max_penalty", "exit_body_clearance_weight",
        ):
            if float(getattr(cfg, name)) <= 0.0:
                raise RuntimeError(f"V11.1 rear posture value {name} must be positive.")
        if float(cfg.rear_extension_free_reach) < 0.0:
            raise RuntimeError("V11.1 rear extension free reach must be non-negative.")
        if float(cfg.rear_knee_flexion_free) < 0.0:
            raise RuntimeError("V11.1 rear knee free flexion must be non-negative.")
        if int(cfg.exit_confirmation_frames) < 2:
            raise RuntimeError("V11.1 level exit must require temporal confirmation.")
        print("V111_BASE_IS_SAVED_V11_FD=YES", flush=True)
        print("V111_DESCENT_REAR_EXTENSION_BOUND=NO", flush=True)
        print("V111_DESCENT_REAR_KNEE_FOLD_BOUND=NO", flush=True)
        print("V111_DESCENT_REAR_ROLL_BOUND=NO", flush=True)
        print("V111_REAR_THIGH_PARALLEL_PENALTY=NO", flush=True)
        print("V111_TOP_BOTTOM_LEVEL_EXIT_CLEARANCE=NO", flush=True)
        print("V111_ALTERNATION_DELTA=NO", flush=True)
        print("V111_V135_DESCENT_TRANSFER_AND_FLAT=YES", flush=True)
        print("V111_ACTOR_OBS_DIM=63", flush=True)
        print("V111_ACTION_DIM=16", flush=True)


__all__ = ["DeeproboticsM20V111V11FDRearPostureEnvCfg", "M20V111RearPostureCfg"]
