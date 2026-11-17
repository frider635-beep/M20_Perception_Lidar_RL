"""V13.3: native V13 emergence plus three-riser chain settlement."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v13_v11fd_alternation_env_cfg import (
    DeeproboticsM20V13V11FDAlternationEnvCfg,
    M20V13RewardsCfg,
)


@configclass
class M20V133TransitionCfg:
    """Single numerical ledger for training-only three-step accounting."""

    step_width: float = 0.30
    longitudinal_tolerance: float = 0.08
    lateral_tolerance: float = 0.15
    minimum_height_gain: float = 0.03
    adjacent_progress_weight: float = 0.50
    adjacent_transition_settlement: float = 1.0
    chain_settlement: float = 4.0
    repeat_transition_cost: float = 0.10
    skip_transition_cost: float = 0.02
    simultaneous_transition_cost: float = 0.10
    promotion_min_chain_precision: float = 0.50
    evaluation_min_chain_count: int = 10
    evaluation_min_each_final_side: int = 1
    evaluation_min_chain_precision: float = 0.50
    evaluation_min_velocity_tracking: float = 0.65
    evaluation_ascending_fraction: float = 0.50
    evaluation_step_height: float = 0.12


@configclass
class M20V133RewardsCfg(M20V13RewardsCfg):
    """Keep every V13 reward and add one completed-chain settlement."""

    v133_style = RewTerm(func=mdp.v133_three_step_chain_reward, weight=1.0)
    # Difference terms make the total descending-column objective equal the
    # official DeepRobotics M20 Rough weights at commit 6d317dfb.  They are
    # exactly zero on ascent and every non-descending terrain.
    v133_official_descent_flat_orientation = RewTerm(
        func=mdp.v133_official_descent_flat_orientation_correction, weight=-1.0,
    )
    v133_official_descent_lin_vel_z = RewTerm(
        func=mdp.v133_official_descent_lin_vel_z_correction, weight=-1.0,
    )
    v133_official_descent_action_rate = RewTerm(
        func=mdp.v133_official_descent_action_rate_correction, weight=0.004,
    )
    v133_official_descent_ang_vel_xy = RewTerm(
        func=mdp.v133_official_descent_ang_vel_xy_correction, weight=0.04,
    )
    v133_official_descent_hipx = RewTerm(
        func=mdp.v133_official_descent_joint_pos_cost, weight=-2.6,
        params={
            "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "stand_still_scale": 5.0, "velocity_threshold": 0.5,
            "command_threshold": 0.1, "official_kind": "hipx",
        },
    )
    v133_official_descent_hipy = RewTerm(
        func=mdp.v133_official_descent_joint_pos_cost, weight=-1.4,
        params={
            "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "stand_still_scale": 5.0, "velocity_threshold": 0.5,
            "command_threshold": 0.1, "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "official_kind": "hipy",
        },
    )
    v133_official_descent_knee = RewTerm(
        func=mdp.v133_official_descent_joint_pos_cost, weight=-0.65,
        params={
            "command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "stand_still_scale": 5.0, "velocity_threshold": 0.5,
            "command_threshold": 0.1, "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "official_kind": "knee",
        },
    )
    v133_official_descent_joint_limit_cancel = RewTerm(
        func=mdp.v133_official_descent_joint_limit_cancellation, weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="")},
    )


@configclass
class DeeproboticsM20V133AdjacentTransitionAcquisitionEnvCfg(
    DeeproboticsM20V13V11FDAlternationEnvCfg
):
    """Original 63D V13 actor with simulator-only chain bookkeeping."""

    rewards: M20V133RewardsCfg = M20V133RewardsCfg()
    v133_transition: M20V133TransitionCfg = M20V133TransitionCfg()

    def __post_init__(self):
        super().__post_init__()
        cfg = self.v133_transition

        # The actor observation, command sampler and terrain distribution come
        # directly from V13.  Chain state is available only to rewards, metrics
        # and curriculum; it is never appended to the policy observation.
        self.curriculum.terrain_levels.params.update({
            "v133_three_step_gate": True,
            "v133_min_chain_precision": cfg.promotion_min_chain_precision,
            "official_descent_curriculum": True,
        })
        # Descent uses the same posture, vertical-velocity and action priors as
        # flat locomotion.  There is no persistent tread identity and no
        # front-to-rear completion objective: both pairs are free to negotiate
        # the edge through the native velocity and stability rewards.
        self.v11_perception.enable_persistent_descent_rear_phase = False
        self.rewards.descent_step_complete = None
        self.rewards.descent_rear_support_progress = None
        self.rewards.descent_rear_follow_delay = None
        self.rewards.descent_rear_roll_consistency = None
        self.rewards.flat_orientation_l2.func = mdp.v11_phase_flat_orientation_l2
        self.rewards.lin_vel_z_l2.func = mdp.v11_phase_lin_vel_z_l2
        self.rewards.action_rate_l2.func = mdp.v133_route_action_rate
        self.rewards.action_smooth_l2.func = mdp.v133_route_action_smooth
        self.rewards.hipy_joint_pos_penalty.func = mdp.v133_route_front_joint_pos_penalty
        self.rewards.knee_joint_pos_penalty.func = mdp.v133_route_front_joint_pos_penalty
        for term_name, joint_names in (
            ("v133_official_descent_hipx", self.hipx_joint_names),
            ("v133_official_descent_hipy", self.hipy_joint_names),
            ("v133_official_descent_knee", self.knee_joint_names),
            ("v133_official_descent_joint_limit_cancel", self.leg_joint_names),
        ):
            getattr(self.rewards, term_name).params["asset_cfg"].joint_names = joint_names
        # These V11 route/impact terms have no official M20 counterpart.  Keep
        # them for perceptive ascent and remove them from descending columns.
        for term_name, routed_func in (
            ("world_forward_deficit", mdp.v133_ascent_world_forward_deficit),
            ("world_forward_progress", mdp.v133_ascent_world_forward_progress),
            ("path_lateral_deviation", mdp.v133_ascent_path_lateral_deviation),
            ("path_heading_error", mdp.v133_ascent_path_heading_error),
            ("non_wheel_leg_contact", mdp.v133_ascent_non_wheel_contact),
            ("non_wheel_leg_scrape", mdp.v133_ascent_non_wheel_scrape),
            ("wheel_impact", mdp.v133_ascent_wheel_impact),
            ("stair_straightness", mdp.v133_ascent_stair_straightness),
            ("front_riser_contact", mdp.v133_ascent_front_riser_contact),
        ):
            getattr(self.rewards, term_name).func = routed_func
        # Cancel every remaining V10/V11/V13/V13.3 reward that is absent from
        # the official M20 Rough objective.  These source terms are state-cached
        # within a control step, so re-evaluation cannot mint a second event.
        for source_name in (
            "is_terminated", "landing_event", "platform_exit_success",
            "clearance_progress", "pure_yaw_xy_drift",
            "front_hipy_participation_settlement", "front_knee_excess",
            "landing_center", "landing_edge",
            "v13_opposite_progress", "v13_velocity_coupled_opposite_progress",
            "v13_same_side_progress", "v13_simultaneous_progress",
            "v13_alternating_completion", "v13_same_side_completion", "v133_style",
        ):
            source = getattr(self.rewards, source_name)
            setattr(
                self.rewards, f"v133_official_descent_cancel_{source_name}",
                RewTerm(
                    func=mdp.v133_cancel_nonofficial_descent_reward,
                    weight=-float(source.weight), params={"source_term_name": source_name},
                ),
            )
        self.terminations.illegal_contact.func = mdp.v133_nonofficial_illegal_contact_termination
        self.terminations.front_riser_collision.func = mdp.v133_ascent_front_riser_collision_termination
        for key in (
            "seed_count", "adjacent_count", "chain_count", "repeat_count", "follower_count",
            "skip_count", "simultaneous_count", "chain_fl_count", "chain_fr_count",
            "chain_precision", "chain_fl_fraction", "chain_active_fraction",
            "chain_length_mean", "promotion_eligible_rate",
            "adjacent_progress_active_rate", "adjacent_progress_delta_mean",
            "adjacent_progress_escrow_mean", "adjacent_progress_clawback_mean",
        ):
            setattr(self.curriculum, f"v133_{key}", CurrTerm(func=mdp.v133_chain_metric, params={"key": key}))

        if not 0.0 < float(cfg.longitudinal_tolerance) < 0.5 * float(cfg.step_width):
            raise RuntimeError("V13.3 longitudinal tolerance must distinguish adjacent treads.")
        if (
            float(cfg.lateral_tolerance) <= 0.0 or float(cfg.minimum_height_gain) <= 0.0
        ):
            raise RuntimeError("V13.3 lateral tolerance and height gain must be positive.")
        if float(cfg.chain_settlement) <= 0.0:
            raise RuntimeError("V13.3 chain settlement must be positive.")
        if float(cfg.adjacent_progress_weight) <= 0.0:
            raise RuntimeError("V13.3 adjacent progress weight must be positive.")
        if not 0.0 < float(cfg.adjacent_transition_settlement) < float(cfg.chain_settlement):
            raise RuntimeError("V13.3 adjacent settlement must be positive and below the chain bonus.")
        if not 0.0 <= float(cfg.promotion_min_chain_precision) <= 1.0:
            raise RuntimeError("V13.3 promotion precision must lie in [0, 1].")
        if int(cfg.evaluation_min_chain_count) < 1 or int(cfg.evaluation_min_each_final_side) < 1:
            raise RuntimeError("V13.3 frozen evaluation event minima must be positive.")
        if not 0.0 <= float(cfg.evaluation_min_chain_precision) <= 1.0:
            raise RuntimeError("V13.3 frozen evaluation precision must lie in [0, 1].")
        if not 0.0 <= float(cfg.evaluation_min_velocity_tracking) <= 1.0:
            raise RuntimeError("V13.3 frozen evaluation velocity threshold must lie in [0, 1].")
        for name in ("repeat_transition_cost", "skip_transition_cost", "simultaneous_transition_cost"):
            if float(getattr(cfg, name)) <= 0.0:
                raise RuntimeError(f"V13.3 {name} must be positive.")
        if not 0.0 < float(cfg.evaluation_ascending_fraction) < 1.0:
            raise RuntimeError("V13.3 frozen evaluation must include ascent and descent.")
        if float(cfg.evaluation_step_height) <= 0.0:
            raise RuntimeError("V13.3 frozen evaluation step height must be positive.")
        terrain = self.scene.terrain.terrain_generator.sub_terrains
        for name in ("pyramid_stairs", "pyramid_stairs_inv"):
            if abs(float(terrain[name].step_width) - float(cfg.step_width)) > 1.0e-9:
                raise RuntimeError(f"V13.3 expects {cfg.step_width} m treads, got {terrain[name].step_width}.")

        print("V133_NATIVE_V13_SOFT_REWARDS=YES", flush=True)
        print("V133_THREE_STEP_CHAIN_SETTLEMENT=YES", flush=True)
        print("V133_CHAIN_STATE_IN_ACTOR_OBS=NO", flush=True)
        print("V133_ASCENDING_COMMAND_CURRICULUM=NO", flush=True)
        print("V133_DESCENT_TREAD_LOCK=NO", flush=True)
        print("V133_DESCENT_OFFICIAL_M20_REWARD_ROUTING=YES", flush=True)
        print("V133_DESCENT_OFFICIAL_COMMIT=6d317dfb", flush=True)
        print("V133_FROM_SCRATCH=YES", flush=True)
        print("V133_ACTOR_OBS_DIM=63", flush=True)
        print("V133_ACTION_DIM=16", flush=True)


__all__ = ["DeeproboticsM20V133AdjacentTransitionAcquisitionEnvCfg", "M20V133TransitionCfg"]
