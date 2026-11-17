"""Independent V11-FD task: bilateral ascent, high clearance, stable descent."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v11_continuous_env_cfg import (
    DeeproboticsM20V11ContinuousEnvCfg,
    M20V11PerceptionCfg,
    M20V11RewardsCfg,
)


@configclass
class M20V11FDPerceptionCfg(M20V11PerceptionCfg):
    """V11-FD's deployable geometry and event contracts."""

    front_completion_mode: str = "bilateral_same_riser"
    step_feature_mode: str = "signed_edges"
    feature_contract: str = "v11_signed_edges_v2"
    # wheel_radius=0.09 m + clearance_margin=0.05 m = 0.14 m centre height
    # above the real target tread on the crossing control step.
    clearance_margin: float = 0.05
    require_clearance_at_crossing: bool = True
    enable_persistent_descent_rear_phase: bool = True


@configclass
class M20V11FDRewardsCfg(M20V11RewardsCfg):
    """Separate reward object so V11-FD cannot mutate legacy V11 terms."""

    # Both terms are one-shot and require upper-tread support.  The first is
    # graded by target-relative crossing height; the second only fires after
    # the unchanged 14 cm hard gate.  Bilateral landing remains the dominant
    # event and neither term rewards lifting/crossing without a landing.
    safe_front_landing_progress = RewTerm(
        func=mdp.v11_fd_safe_front_landing_progress_reward, weight=0.5,
    )
    single_front_hard_landing = RewTerm(
        func=mdp.v11_fd_single_front_landing_reward, weight=1.0,
    )


@configclass
class DeeproboticsM20V11FlatDescentEnvCfg(DeeproboticsM20V11ContinuousEnvCfg):
    """8/27 V11 baseline with only high-clearance, descent and flat-turn edits."""

    rewards: M20V11FDRewardsCfg = M20V11FDRewardsCfg()
    v11_perception: M20V11FDPerceptionCfg = M20V11FDPerceptionCfg()

    def __post_init__(self):
        # Install the requested distribution before V11 resolves terrain-type
        # IDs.  Ordered proportions map exactly to 20 columns: 4/4/6/2/2/1/1.
        terrain_cfg = self.scene.terrain.terrain_generator
        existing = terrain_cfg.sub_terrains
        required = {
            "pyramid_stairs", "pyramid_stairs_inv", "boxes", "random_rough",
            "hf_pyramid_slope", "hf_pyramid_slope_inv",
        }
        if not required.issubset(existing):
            raise RuntimeError(f"V11-FD terrain generator is missing {sorted(required - set(existing))}.")
        terrain_cfg.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
            "pyramid_stairs": existing["pyramid_stairs"],
            "pyramid_stairs_inv": existing["pyramid_stairs_inv"],
            "boxes": existing["boxes"],
            "random_rough": existing["random_rough"],
            "hf_pyramid_slope": existing["hf_pyramid_slope"],
            "hf_pyramid_slope_inv": existing["hf_pyramid_slope_inv"],
        }
        for name, proportion in {
            "pyramid_stairs": 0.20,
            "pyramid_stairs_inv": 0.30,
            "boxes": 0.10,
            "random_rough": 0.10,
            "hf_pyramid_slope": 0.05,
            "hf_pyramid_slope_inv": 0.05,
        }.items():
            terrain_cfg.sub_terrains[name].proportion = proportion

        super().__post_init__()

        # A random policy cannot discover the strict landing chain reliably
        # when half of its initial samples start on mid-level stairs.  Keep
        # the requested terrain mixture, but begin every type at level 0--1.
        self.scene.terrain.max_init_terrain_level = 1

        cfg = self.v11_perception
        if cfg.front_completion_mode != "bilateral_same_riser":
            raise RuntimeError("V11-FD must remain bilateral_same_riser.")
        if cfg.feature_contract != "v11_signed_edges_v2" or cfg.step_feature_mode != "signed_edges":
            raise RuntimeError("V11-FD requires the v11_signed_edges_v2 observation contract.")
        if abs((float(cfg.wheel_radius) + float(cfg.clearance_margin)) - 0.14) > 1.0e-9:
            raise RuntimeError("V11-FD crossing clearance must resolve to 0.14 m above the target tread.")

        # No prescribed diagonal or alternating gait on flat turns.
        self.rewards.feet_air_time_ang_z_M20 = None
        self.rewards.rotation_gait_status = None
        self.rewards.rotation_gait_symmetry = None
        self.rewards.pure_yaw_xy_drift = RewTerm(
            func=mdp.v11_pure_yaw_xy_drift_penalty,
            weight=-0.5,
            params={"cmd_lin_threshold": 0.1, "cmd_ang_threshold": 0.1, "speed_scale": 0.35},
        )

        # Bilateral front-to-rear descent has no left/right ordering.
        self.rewards.descent_step_complete = RewTerm(
            func=mdp.v11_descent_step_complete_reward, weight=3.0,
        )
        # A small per-wheel milestone supplies a reachable rear-follow
        # gradient; the full bilateral event above remains dominant.
        self.rewards.descent_rear_support_progress = RewTerm(
            func=mdp.v11_descent_rear_support_progress_reward, weight=0.5,
        )
        self.rewards.descent_rear_follow_delay = RewTerm(
            func=mdp.v11_descent_rear_follow_delay_penalty, weight=-0.5,
        )
        self.rewards.descent_rear_roll_consistency = RewTerm(
            func=mdp.v11_descent_rear_roll_consistency_penalty, weight=-0.2,
        )
        self.rewards.post_first_clearance_surplus = None

        # During only the locked front-complete -> rear-complete descent
        # phase, release enough rear sagittal motion to establish lower-tread
        # support.  Everywhere else retains the baseline pose, mirror and
        # smoothness regularizers unchanged.
        self.rewards.action_rate_l2.params.update({
            "descent_rear_hipy_scale": 0.50,
            "descent_rear_knee_scale": 0.70,
        })
        self.rewards.action_smooth_l2.params.update({
            "descent_rear_hipy_scale": 0.65,
            "descent_rear_knee_scale": 0.80,
        })
        self.rewards.hipy_joint_pos_penalty.params.update({
            "descent_rear_joint_names": ("hl_hipy_joint", "hr_hipy_joint"),
            "descent_rear_active_scale": 0.50,
        })
        self.rewards.knee_joint_pos_penalty.params.update({
            "descent_rear_joint_names": ("hl_knee_joint", "hr_knee_joint"),
            "descent_rear_active_scale": 0.70,
        })
        self.rewards.joint_mirror.func = mdp.v11_descent_phase_joint_mirror
        self.rewards.joint_mirror.params["descent_rear_follow_scale"] = 0.25

        # Remove alternating-only logging from this independent task.  FL/FR
        # first-responsibility diagnostics remain to detect accidental bias.
        for name in (
            "v11_expected_fl_rate", "v11_expected_fr_rate",
            "v11_completed_risers_mean", "v11_alternation_compliance_rate",
            "v11_alternation_violation_rate", "v11_post_first_collision_rate",
            "v11_anchor_valid_rate", "v11_anchor_support_steps",
            "v11_clearance_surplus", "v11_confirmed_violation_rate",
            "v11_rewarded_alternation_rate",
        ):
            setattr(self.curriculum, name, None)
        for key in (
            "crossing_center_mean", "crossing_margin_mean",
            "crossing_min_margin", "crossing_p10_margin", "low_crossing_rate",
            "safe_support_event_rate", "hard_candidate_event_rate",
        ):
            setattr(
                self.curriculum, f"v11_fd_{key}",
                CurrTerm(func=mdp.v11_fd_clearance_metric, params={"key": key}),
            )


__all__ = ["DeeproboticsM20V11FlatDescentEnvCfg", "M20V11FDPerceptionCfg"]
