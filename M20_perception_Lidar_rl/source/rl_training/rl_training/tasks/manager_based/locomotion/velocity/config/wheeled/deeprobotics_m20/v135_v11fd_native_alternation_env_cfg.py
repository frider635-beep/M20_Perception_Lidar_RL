"""V13.5-Final: ascent dominance, rear support transfer, persistent flat."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v13_v11fd_alternation_env_cfg import (
    DeeproboticsM20V13V11FDAlternationEnvCfg,
    M20V13RewardsCfg,
)


@configclass
class M20V135PostStairFlatCfg:
    """Single numerical ledger for the post-stair flat outcome."""

    no_riser_confirmation_frames: int = 10
    support_confirmation_frames: int = 5
    wheel_plane_tolerance: float = 0.05
    platform_lookahead: float = 0.60
    platform_half_width: float = 0.25
    clearance_free_ratio: float = 0.90
    clearance_scale_ratio: float = 0.15
    upright_std: float = 0.20
    velocity_std: float = 0.50
    minimum_quality: float = 0.80
    # V13.5-Final: persistent flat-mode crawl prevention
    crawl_clearance_threshold: float = 0.90
    crawl_clearance_scale: float = 0.12


@configclass
class M20V135DescentSupportTransferCfg:
    """V13.5-Final descent support transfer parameters."""

    wheel_radius: float = 0.10
    front_margin: float = 0.05
    contact_force_threshold: float = 20.0
    stable_frames: int = 3
    # Lag penalty
    free_lag: float = 0.06
    lag_scale: float = 0.10
    # Premature collapse penalty
    collapse_threshold: float = 0.04
    collapse_scale: float = 0.08
    # V13.5.1 phase-aware rear-leg timing (rear-transfer only).
    rear_extension_free_ratio: float = 1.15
    rear_extension_ratio_scale: float = 0.15
    recovery_free_ratio: float = 1.10
    recovery_confirmation_frames: int = 3
    recovery_pitch_tolerance: float = 0.2617993878
    persistent_extension_ratio: float = 1.20
    persistent_extension_frames: int = 15
    folded_length_ratio: float = 0.65
    defect_duration: float = 0.30
    ground_sample_radius: float = 0.25
    workspace_free: float = 0.35
    workspace_scale: float = 0.35
    terrain_pitch_free: float = 0.2617993878
    terrain_pitch_scale: float = 0.2617993878
    clearance_floor: float = 0.75
    clearance_quality_scale: float = 0.20
    rolling_slip_scale: float = 0.50
    relative_speed_scale: float = 0.50
    pitch_rate_free: float = 1.0
    pitch_rate_scale: float = 2.0


@configclass
class M20V135LongTrainCfg:
    """One numerical ledger for the first 1000-update interaction curriculum."""

    rollout_steps_per_update: int = 24
    anneal_updates: int = 1000
    early_mixture_probability: float = 0.70
    near_stair_distance_range: tuple[float, float] = (0.35, 0.65)
    early_ascent_command_range: tuple[float, float] = (0.60, 1.20)
    early_lateral_range: tuple[float, float] = (-0.10, 0.10)
    early_yaw_jitter_range: tuple[float, float] = (-0.0872664626, 0.0872664626)
    initial_penalty_scale: float = 0.25
    ascent_penalty_ramp_start_update: int = 200
    early_descent_command_range: tuple[float, float] = (0.25, 0.65)
    descent_velocity_baseline_initial: float = 0.85
    descent_velocity_baseline_final: float = 0.60
    flat_blend_steps: int = 10


@configclass
class M20V135RiserIdentityCfg:
    """Single ledger for locked physical B1/B2/... identity."""

    # Per-column ascent geometry; descent retains its original 0.30m mesh.
    ascent_tread_widths: tuple[float, ...] = (0.30, 0.30, 0.25, 0.35, 0.40, 0.45)
    support_preparation_frames: int = 5
    support_margin_scale: float = 0.10
    step_width: float = 0.30
    longitudinal_tolerance: float = 0.08
    lateral_tolerance: float = 0.15
    overstep_margin: float = 0.12
    first_swing_lift_min: float = 0.03
    b1_confirmation_frames: int = 3
    b1_world_stability_tolerance: float = 0.03
    irrecoverable_behind_distance: float = -0.30
    ordered_speed_multiplier: float = 1.20
    ordinary_speed_multiplier: float = 1.00
    skip_speed_multiplier: float = 0.85


@configclass
class M20V135StageStartsCfg:
    """Opt-in training sampler; formal evaluation uses the default full resets."""
    enabled: bool = False
    probability: float = 0.30
    stable_frames: int = 5
    max_age_steps: int = 4800
    max_cached_envs: int = 8192
    max_captures_per_step: int = 8


@configclass
class M20V135RewardsCfg(M20V13RewardsCfg):
    """V13.5-Final From-Scratch reward ledger.

    Key changes:
    - Ascent: dominance-based shaping (not dense penalties)
    - Descent: new rear support transfer (not V11/V13.4)
    - Flat: persistent crawl guard
    - All penalties greatly reduced
    """

    # ========== Ascent: Completion events ==========
    v135_valid_riser_completion = RewTerm(
        func=mdp.v135_valid_riser_completion_reward, weight=1.5,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_adjacent_chain = RewTerm(
        func=mdp.v135_adjacent_chain_reward, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_alternation_cycle = RewTerm(
        func=mdp.v135_alternation_cycle_reward, weight=1.5,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )

    # ========== Ascent: Error penalties (greatly reduced) ==========
    v135_wrong_side_riser = RewTerm(
        func=mdp.v135_wrong_side_riser_reward, weight=-0.15,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_duplicate_riser = RewTerm(
        func=mdp.v135_duplicate_riser_reward, weight=-0.10,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_skip_riser = RewTerm(
        func=mdp.v135_skip_riser_reward, weight=-0.35,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_dual_skip_riser = RewTerm(
        func=mdp.v135_dual_skip_riser_reward, weight=-0.20,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    # ========== Ascent: Dominance-based shaping (NEW) ==========
    v135_ascent_leader_progress = RewTerm(
        func=mdp.v135_final_leader_progress_reward, weight=0.3,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_ascent_leader_velocity_coupled = RewTerm(
        func=mdp.v135_final_velocity_coupled_leader_reward, weight=0.7,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )

    # ========== Descent: New rear support transfer ==========
    v135_descent_rear_transfer_progress = RewTerm(
        func=mdp.v135_descent_rear_transfer_progress_reward, weight=0.8,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_descent_rear_transfer_completion = RewTerm(
        func=mdp.v135_descent_rear_transfer_completion_reward, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_descent_body_rear_lag = None  # superseded by shared whole-body quality
    v135_descent_premature_collapse = None  # superseded by shared whole-body quality
    v135_descent_rear_early_extension = None  # superseded by shared whole-body quality
    v135_descent_rear_recovery = RewTerm(
        func=mdp.v135_descent_rear_recovery_reward, weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
    )
    v135_descent_persistent_extension = None  # superseded by shared whole-body quality

    # ========== Post-stair: Recovery and persistent flat ==========
    v135_post_stair_flat_progress = RewTerm(
        func=mdp.v135_post_stair_flat_progress_reward, weight=0.5,
    )
    v135_post_stair_flat_completion = RewTerm(
        func=mdp.v135_post_stair_flat_completion_reward, weight=1.0,
    )
    v135_post_stair_flat_crawl = RewTerm(
        func=mdp.v135_post_stair_flat_crawl_penalty, weight=0.5,
    )

    # Note: Native V11 rewards (velocity, orientation, torques, etc.)
    # are inherited from M20V13RewardsCfg and remain active


@configclass
class DeeproboticsM20V135NativeAlternationEnvCfg(
    DeeproboticsM20V13V11FDAlternationEnvCfg
):
    """V13.5-Final From-Scratch: dominance shaping + new descent + persistent flat."""

    rewards: M20V135RewardsCfg = M20V135RewardsCfg()
    v135_descent_support_transfer: M20V135DescentSupportTransferCfg = M20V135DescentSupportTransferCfg()
    v135_post_stair_flat: M20V135PostStairFlatCfg = M20V135PostStairFlatCfg()
    v135_riser_identity: M20V135RiserIdentityCfg = M20V135RiserIdentityCfg()
    v135_longtrain: M20V135LongTrainCfg = M20V135LongTrainCfg()

    v135_stage_starts: M20V135StageStartsCfg = M20V135StageStartsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Disable all legacy V13 rewards (we use new dominance-based shaping)
        self.rewards.v13_opposite_progress = None
        self.rewards.v13_velocity_coupled_opposite_progress = None
        self.rewards.v13_same_side_progress = None
        self.rewards.v13_simultaneous_progress = None
        self.rewards.v13_alternating_completion = None
        self.rewards.v13_same_side_completion = None
        # Final descent has one implementation.  Disable the inherited V11
        # completion/delay/roll terms at configuration time instead of keeping
        # them alive and numerically cancelling their rewards.
        self.rewards.descent_step_complete = None
        self.rewards.descent_rear_follow_delay = None
        self.rewards.descent_rear_roll_consistency = None
        # Parent V13 logging calls the retired V13 ledger as well. Final has
        # one live physical-riser ledger, so these diagnostics must be off.
        for key in (
            "potential_active_rate", "opposite_progress_delta_mean",
            "same_side_progress_delta_mean", "simultaneous_progress_delta_mean",
            "opposite_potential_mean", "seed_event_rate", "compliant_event_rate",
            "violation_event_rate", "simultaneous_completion_event_rate",
            "seed_fl_fraction", "natural_compliance", "progress_precision",
            "simultaneous_progress_fraction",
        ):
            setattr(self.curriculum, f"v13_{key}", None)

        # Replace V11 velocity/forward rewards with modulated versions
        # Preserve V13 strong drive (5.0/2.0/3.0), only modulate during rear transfer
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.v135_modulated_track_lin_vel_xy_exp_l2,
            weight=5.0,  # Preserve V13 strength
            params={"std": 0.5, "command_name": "base_velocity"},
        )
        # A physical Bk->B(k+2) skip must not collect V11's main landing pulse.
        # Adjacent wrong-side landings keep native capability credit but lose
        # strict adjacent/cycle credit, preserving exploration without making
        # the two-riser shortcut economically optimal.
        self.rewards.landing_event = RewTerm(
            func=mdp.v135_no_skip_native_landing_reward,
            weight=5.0,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
        )
        # The ascent terminal payoff represents completion of the whole
        # B1->B2->... chain, independent of how many risers the terrain has.
        self.rewards.platform_exit_success = RewTerm(
            func=mdp.v135_strict_chain_platform_exit_reward,
            weight=5.0,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
        )
        self.rewards.world_forward_progress = RewTerm(
            func=mdp.v135_modulated_world_forward_progress,
            weight=2.0,  # Preserve V13 strength
            params={
                "target_speed_fraction": 0.5,
                "sensor_cfg": SceneEntityCfg("v10_pgtt_scanner"),
            },
        )
        # Replace world_forward_deficit (not world_forward_speed_deficit)
        self.rewards.world_forward_deficit = RewTerm(
            func=mdp.v135_modulated_world_forward_deficit,
            weight=3.0,  # Preserve V13 weight (native returns negative cost)
            params={
                "min_speed_fraction": 0.5,
                "speed_scale": 0.3,
                "sensor_cfg": SceneEntityCfg("v10_pgtt_scanner"),
            },
        )

        # Final command distribution: descent 0.15-0.80, ascent keeps 0.10-2.0.
        # The first 1000-update sampling bias is applied by the command term.
        self.v11_perception.soft_alternation_command_split = True
        self.v11_perception.stair_forward_command_min = 0.10
        self.v11_perception.stair_forward_command_max = None  # No upper limit for ascent
        self.v11_perception.non_ascending_stair_forward_command_min = 0.15
        self.v11_perception.non_ascending_stair_forward_command_max = 0.80

        # Soft perception + strict evaluation: Actor always receives V11/V13's
        # native signed edge geometry. The strict ledger remains privileged.
        self.observations.policy.height_scan.func = mdp.v11_actor_perception

        # Keep V13's native curriculum and start at a useful but non-trivial
        # stair level. Strict alternation never gates terrain promotion.
        self.scene.terrain.max_init_terrain_level = 3

        # Use V135-Final critic state (not legacy V13)
        self.observations.critic.v13_reward_state = ObsTerm(
            func=mdp.v135_final_critic_reward_state,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
            clip=(-2.0, 2.0),
            scale=1.0,
        )

        for key in (
            "traversal_seen_rate", "no_riser_after_stair_rate", "active_rate",
            "contact_coplanar_rate", "clearance_ratio", "quality",
            "completion_event_rate", "completion_count",
            "flat_blend",
        ):
            setattr(
                self.curriculum,
                f"v135_post_stair_{key}",
                CurrTerm(func=mdp.v135_post_stair_flat_metric, params={"key": key}),
            )
        for key in (
            "active_rate", "rear_extension_ratio", "rear_length_min_ratio", "early_extension_rate",
            "recovery_event_rate", "support_quality",
            "whole_body_slip", "whole_body_workspace", "terrain_pitch_error", "ground_evidence_rate",
            "persistent_extension_event_rate", "recovery_pitch_abs",
            "rear_load_share", "four_wheel_support", "body_path_speed",
            "body_pitch_rate_abs", "body_rear_lag",
        ):
            setattr(
                self.curriculum, f"v135_descent_envelope_{key}",
                CurrTerm(
                    func=mdp.v135_descent_support_transfer_metric,
                    params={"key": key, "sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
                ),
            )
        for key in (
            "target_valid_rate", "target_id_mean", "adjacent_chain_length_mean",
            "episode_valid_rate", "gait_speed_factor_mean",
            "ordered_alternation_rate",
            "wrong_side_rate", "duplicate_rate", "skip_rate", "dual_skip_rate",
        ):
            setattr(
                self.curriculum, f"v135_riser_{key}",
                CurrTerm(func=mdp.v135_riser_identity_metric, params={"key": key}),
            )
        for key in (
            "fraction", "early_mixture_probability", "ascent_penalty_scale",
            "descent_penalty_scale", "descent_velocity_baseline",
        ):
            setattr(
                self.curriculum, f"v135_longtrain_{key}",
                CurrTerm(func=mdp.v135_longtrain_metric, params={"key": key}),
            )

        cfg = self.v135_post_stair_flat
        if int(cfg.no_riser_confirmation_frames) < 2 or int(cfg.support_confirmation_frames) < 2:
            raise RuntimeError("V13.5 recovery confirmations must persist for at least two frames.")
        if float(cfg.wheel_plane_tolerance) <= 0.0:
            raise RuntimeError("V13.5 wheel-plane tolerance must be positive.")
        for name in ("clearance_scale_ratio", "upright_std", "velocity_std"):
            if float(getattr(cfg, name)) <= 0.0:
                raise RuntimeError(f"V13.5 recovery scale {name} must be positive.")
        if not 0.0 < float(cfg.minimum_quality) <= 1.0:
            raise RuntimeError("V13.5 minimum recovery quality must lie in (0, 1].")
        identity = self.v135_riser_identity
        if not 0.0 < float(identity.longitudinal_tolerance) < 0.5 * float(identity.step_width):
            raise RuntimeError("V13.5 riser tolerance must distinguish adjacent physical faces.")
        if float(identity.lateral_tolerance) <= 0.0 or float(identity.overstep_margin) <= 0.0:
            raise RuntimeError("V13.5 riser lateral and overstep tolerances must be positive.")
        if int(identity.b1_confirmation_frames) < 2 or float(identity.b1_world_stability_tolerance) <= 0.0:
            raise RuntimeError("V13.5 B1 acquisition must be stable for at least two frames.")
        if float(identity.irrecoverable_behind_distance) >= 0.0:
            raise RuntimeError("V13.5 irrecoverable target distance must be behind the root.")
        if not (
            0.0 < float(identity.skip_speed_multiplier)
            < float(identity.ordinary_speed_multiplier)
            < float(identity.ordered_speed_multiplier)
        ):
            raise RuntimeError("V13.5 ascent speed multipliers must rank skip < ordinary < ordered.")
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import configure_ascent_treads
        configure_ascent_treads(self.scene.terrain.terrain_generator, identity.ascent_tread_widths)
        if float(identity.longitudinal_tolerance) >= 0.5 * min(identity.ascent_tread_widths):
            raise RuntimeError("V135 tolerance aliases the shortest ascent tread")
        longtrain = self.v135_longtrain
        if int(longtrain.rollout_steps_per_update) != 24 or int(longtrain.anneal_updates) != 1000:
            raise RuntimeError("V13.5 LongTrain schedule must match the frozen 24-step/1000-update contract.")
        if not 0.0 <= float(longtrain.early_mixture_probability) <= 1.0:
            raise RuntimeError("V13.5 early mixture probability must lie in [0, 1].")
        if int(longtrain.flat_blend_steps) < 1:
            raise RuntimeError("V13.5 flat blending must last at least one control step.")
        descent = self.v135_descent_support_transfer
        for name in ("rear_extension_ratio_scale", "recovery_pitch_tolerance"):
            if float(getattr(descent, name)) <= 0.0:
                raise RuntimeError(f"V13.5 descent envelope scale {name} must be positive.")
        if not 1.0 < float(descent.recovery_free_ratio) < float(descent.rear_extension_free_ratio) < 2.0:
            raise RuntimeError("V13.5 neutral rear-length envelope must be ordered around ratio 1.")
        if int(descent.recovery_confirmation_frames) < 2:
            raise RuntimeError("V13.5 rear recovery must persist for at least two frames.")
        if float(descent.persistent_extension_ratio) <= float(descent.recovery_free_ratio):
            raise RuntimeError("V13.5 persistent extension threshold must exceed recovery ratio.")
        if int(descent.persistent_extension_frames) < 2:
            raise RuntimeError("V13.5 persistent extension must persist for at least two frames.")
        print("V135_NATIVE_V13_STAIR_OBJECTIVE=YES", flush=True)
        print("V135_REAR_SUPPORT_TRANSFER_FROM_UPDATE_0=YES", flush=True)
        print("V135_POST_STAIR_LOCAL_RECOVERY=YES", flush=True)
        print("V135_PHYSICAL_RISER_IDENTITY=YES", flush=True)
        print("V135_ACTOR_OBS_DIM=63", flush=True)
        print("V135_CRITIC_OBS_DIM=574", flush=True)
        print("V135_ACTION_DIM=16", flush=True)


__all__ = [
    "DeeproboticsM20V135NativeAlternationEnvCfg",
    "M20V135PostStairFlatCfg",
    "M20V135DescentSupportTransferCfg",
    "M20V135RiserIdentityCfg",
    "M20V135LongTrainCfg",
]
