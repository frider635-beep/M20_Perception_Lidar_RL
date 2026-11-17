"""V13.1: strict B1/B2/... identity with soft alternation and top follow."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v13_v11fd_alternation_env_cfg import (
    DeeproboticsM20V13V11FDAlternationEnvCfg,
    M20V13RewardsCfg,
)


V131_VELOCITY_QUALITY_STD = 0.7071067811865476


@configclass
class M20V131OrderingCfg:
    """Single numerical ledger for ordered-riser identity and top behavior."""

    # Both inherited stair generators use a 0.30 m tread.  Startup asserts this
    # instead of silently applying this ledger to a different geometry.
    step_width: float = 0.30
    target_match_tolerance: float = 0.08
    target_lateral_tolerance: float = 0.15
    strict_invalid_teaching_scale: float = 1.0
    # Commit the first front leg as soon as target-qualified motion becomes
    # asymmetric.  These are normalized locked-target potential thresholds,
    # not joint or world-coordinate magic numbers.
    first_mover_potential_threshold: float = 0.05
    first_mover_release_threshold: float = 0.03
    first_mover_separation: float = 0.02
    first_mover_confirmation_frames: int = 3
    velocity_quality_std: float = V131_VELOCITY_QUALITY_STD
    expected_action_rate_hipy_scale: float = 0.35
    expected_action_rate_knee_scale: float = 0.50
    expected_action_smooth_hipy_scale: float = 0.50
    expected_action_smooth_knee_scale: float = 0.50
    top_no_riser_confirmation_frames: int = 3
    plateau_lift_free_height: float = 0.025
    plateau_lift_scale: float = 0.08


@configclass
class M20V131RewardsCfg(M20V13RewardsCfg):
    """Ordered-riser gait style; native V11 speed weights stay untouched."""

    # These B1 terms reuse the B2+ progress weights in a mutually exclusive
    # phase, so the reward ceiling is unchanged while the bootstrap signal is
    # reachable before the first landing.
    v131_b1_progress = RewTerm(
        func=mdp.v131_b1_progress_reward, weight=0.5,
    )
    v131_b1_velocity_coupled_progress = RewTerm(
        func=mdp.v131_b1_velocity_coupled_progress_reward,
        weight=1.5,
        params={"command_name": "base_velocity", "std": V131_VELOCITY_QUALITY_STD},
    )
    v131_expected_progress = RewTerm(
        func=mdp.v131_expected_progress_reward, weight=0.5,
    )
    v131_velocity_coupled_expected_progress = RewTerm(
        func=mdp.v131_velocity_coupled_expected_progress_reward,
        weight=1.5,
        params={"command_name": "base_velocity", "std": V131_VELOCITY_QUALITY_STD},
    )
    v131_same_side_progress = RewTerm(
        func=mdp.v131_same_side_progress_reward, weight=-0.75,
    )
    v131_simultaneous_progress = RewTerm(
        func=mdp.v131_simultaneous_progress_reward, weight=-0.5,
    )
    v131_simultaneous_b1_lift = RewTerm(
        func=mdp.v131_simultaneous_b1_lift_reward, weight=-0.5,
    )
    v131_alternating_completion = RewTerm(
        func=mdp.v131_alternating_completion_reward, weight=1.0,
    )
    v131_nonordered_completion = RewTerm(
        func=mdp.v131_nonordered_completion_reward, weight=-0.5,
    )
    v131_skip_riser = RewTerm(
        func=mdp.v131_skip_riser_reward, weight=-1.0,
    )
    v131_skip_progress = RewTerm(
        func=mdp.v131_skip_progress_reward, weight=-1.0,
    )
    v131_top_follow_progress = RewTerm(
        func=mdp.v131_top_follow_progress_reward, weight=1.0,
    )
    v131_top_follow_completion = RewTerm(
        func=mdp.v131_top_follow_completion_reward, weight=3.0,
    )
    v131_post_top_front_lift = RewTerm(
        func=mdp.v131_post_top_front_lift_penalty, weight=-0.5,
    )


@configclass
class DeeproboticsM20V131V11FDOrderedAlternationEnvCfg(
    DeeproboticsM20V13V11FDAlternationEnvCfg
):
    """Isolated V13.1 task; V11-FD/V13 source behavior is not rewritten."""

    rewards: M20V131RewardsCfg = M20V131RewardsCfg()
    v131_ordering: M20V131OrderingCfg = M20V131OrderingCfg()

    def __post_init__(self):
        super().__post_init__()

        # Replace V13's permissive style layer. Physical completion rewards stay
        # native so the V11 capability-emergence path remains available. V13's
        # continuous historical-best potential remains the style primitive;
        # V13.1 adds adjacency without gating native velocity or progress.
        for name in (
            "v13_opposite_progress",
            "v13_velocity_coupled_opposite_progress",
            "v13_same_side_progress",
            "v13_simultaneous_progress",
            "v13_alternating_completion",
            "v13_same_side_completion",
        ):
            setattr(self.rewards, name, None)
        # V13.1 from-scratch starts at levels 0--1. The generator's final
        # curriculum range is unchanged.
        self.scene.terrain.max_init_terrain_level = 1
        # Promotion remains native on non-ascent terrain. Ascending promotion
        # additionally consumes the per-environment strict episode ledger.
        self.curriculum.terrain_levels.params["strict_ordered_gate"] = True
        self.v11_perception.soft_alternation_curriculum_gate = False

        # Preserve the native action range and physical joint limits. Only the
        # currently expected front swing leg receives additional sagittal
        # freedom, and the relief switches off as soon as that wheel supports
        # the locked tread. The non-expected support leg remains native V13.
        self.rewards.action_rate_l2.func = mdp.v131_expected_swing_action_rate_l2
        self.rewards.action_rate_l2.params.update({
            "expected_hipy_scale": self.v131_ordering.expected_action_rate_hipy_scale,
            "expected_knee_scale": self.v131_ordering.expected_action_rate_knee_scale,
        })
        self.rewards.action_smooth_l2.func = mdp.v131_expected_swing_action_smooth_l2
        self.rewards.action_smooth_l2.params.update({
            "expected_hipy_scale": self.v131_ordering.expected_action_smooth_hipy_scale,
            "expected_knee_scale": self.v131_ordering.expected_action_smooth_knee_scale,
        })

        # V13 metrics describe its permissive ledger and would be misleading in
        # this task.  Replace them with observability for every ordered phase and
        # every failure mode that can otherwise look like success.
        for name in (
            "potential_active_rate",
            "opposite_progress_delta_mean",
            "same_side_progress_delta_mean",
            "simultaneous_progress_delta_mean",
            "opposite_potential_mean",
            "seed_event_rate",
            "compliant_event_rate",
            "violation_event_rate",
            "simultaneous_completion_event_rate",
            "seed_fl_fraction",
            "natural_compliance",
            "progress_precision",
            "simultaneous_progress_fraction",
        ):
            setattr(self.curriculum, f"v13_{name}", None)
        for key in (
            "b1_first_completion_rate",
            "ordered_completion_rate",
            "riser_skip_rate",
            "wrong_side_rate",
            "simultaneous_b1_lift_rate",
            "riser_index_jump_mean",
            "top_follow_success_rate",
            "top_follow_delay_steps",
            "post_top_front_lift_rate",
            "seed_fl_fraction",
            "approach_b1_fraction",
            "alternating_fraction",
            "top_follow_fraction",
            "top_trailing_follow_fraction",
            "plateau_roll_fraction",
            "failed_fraction",
            "ordered_event_rate",
            "skip_event_rate",
            "top_follow_event_rate",
            "ordered_chain_valid_fraction",
            "ordered_speed_gate_mean",
            "ordered_alternation_rate",
            "adjacent_completion_count",
            "skip_completion_count",
            "physical_last_riser_mean",
            "ordered_last_riser_mean",
            "physical_minus_ordered_gap",
            "ordered_progress_reward",
            "b1_progress_active_rate",
            "b1_progress_reward",
            "b1_velocity_coupled_reward",
            "alternating_progress_active_rate",
            "strict_chain_valid_fraction",
            "teaching_chain_active_fraction",
            "teaching_resync_count",
            "ordered_speed_gate_on_actionable_target_mean",
            "ordered_speed_reward",
            "free_velocity_reward",
            "full_velocity_eligibility_rate",
            "same_side_progress_rate",
            "simultaneous_progress_rate",
            "skip_rate",
            "platform_success_rate",
            "completed_risers_mean",
            "skip_progress_active_rate",
            "skip_progress_reward",
            "physical_side_switch_rate",
            "physical_adjacent_riser_rate",
            "strict_adjacent_alternating_rate",
            "skip_given_side_switch_rate",
            "rollout_side_switch_rate",
            "rollout_adjacent_rate",
            "rollout_strict_alt_rate",
            "rollout_skip_rate",
            "rollout_skip_given_switch_rate",
            "first_mover_latch_rate",
            "b1_commit_active_rate",
            "early_opposite_progress_rate",
            "handoff_active_rate",
            "locked_target_progress_active_rate",
            "velocity_escrow_mean",
            "velocity_escrow_paid_rate",
            "velocity_escrow_forfeit_rate",
            "latch_to_b1_completion_rate",
            "b1_progress_escrow_mean",
            "b1_velocity_escrow_mean",
            "b1_escrow_paid_rate",
            "b1_escrow_forfeit_rate",
            "expected_swing_relief_active_rate",
        ):
            setattr(
                self.curriculum,
                f"v131_{key}",
                CurrTerm(func=mdp.v131_ordered_metric, params={"key": key}),
            )

        cfg = self.v131_ordering
        if not (0.0 < float(cfg.target_match_tolerance) < 0.5 * float(cfg.step_width)):
            raise RuntimeError("V13.1 longitudinal tolerance must distinguish adjacent stair faces.")
        if float(cfg.target_lateral_tolerance) <= 0.0:
            raise RuntimeError("V13.1 lateral tolerance must be positive.")
        if float(cfg.strict_invalid_teaching_scale) != 1.0:
            raise RuntimeError("V13.1 local teaching must remain full strength after strict failure.")
        if not 0.0 < float(cfg.first_mover_potential_threshold) < 1.0:
            raise RuntimeError("V13.1 first-mover threshold must lie inside (0, 1).")
        if not 0.0 < float(cfg.first_mover_separation) < float(cfg.first_mover_potential_threshold):
            raise RuntimeError("V13.1 first-mover separation must be positive and below its threshold.")
        if not 0.0 < float(cfg.first_mover_release_threshold) < float(cfg.first_mover_potential_threshold):
            raise RuntimeError("V13.1 first-mover release threshold must lie below its onset threshold.")
        if int(cfg.first_mover_confirmation_frames) < 2:
            raise RuntimeError("V13.1 first-mover commitment must persist for at least two frames.")
        if float(cfg.velocity_quality_std) <= 0.0:
            raise RuntimeError("V13.1 velocity-quality standard deviation must be positive.")
        for name in (
            "expected_action_rate_hipy_scale", "expected_action_rate_knee_scale",
            "expected_action_smooth_hipy_scale", "expected_action_smooth_knee_scale",
        ):
            value = float(getattr(cfg, name))
            if not 0.0 < value <= 1.0:
                raise RuntimeError(f"V13.1 expected-swing scale {name} must lie in (0, 1].")
        if int(cfg.top_no_riser_confirmation_frames) < 1:
            raise RuntimeError("V13.1 top confirmation must require at least one frame.")
        terrain_cfg = self.scene.terrain.terrain_generator.sub_terrains
        for name in ("pyramid_stairs", "pyramid_stairs_inv"):
            actual = float(terrain_cfg[name].step_width)
            if abs(actual - float(cfg.step_width)) > 1.0e-9:
                raise RuntimeError(
                    f"V13.1 ordered geometry expects step_width={cfg.step_width}, "
                    f"but {name} uses {actual}."
                )
        print("V131_NATIVE_VELOCITY_RESTORED=YES", flush=True)
        print("V131_NATIVE_FORWARD_PROGRESS_RESTORED=YES", flush=True)
        print("V131_STRICT_AFFECTS_NATIVE_REWARD=NO", flush=True)
        print("V131_TEACHING_AFTER_FAILURE_SCALE=1.0", flush=True)
        print("V131_SKIP_PROGRESS_ACTIVE=YES", flush=True)
        print("V131_ACTOR_OBS_DIM=63", flush=True)
        print("V131_ACTION_DIM=16", flush=True)


@configclass
class DeeproboticsM20V131RetentionEnvCfg(
    DeeproboticsM20V131V11FDOrderedAlternationEnvCfg
):
    """Fixed medium ascending stairs for the bounded V13 retention test."""

    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.curriculum = False
        generator.num_rows = 1
        self.scene.terrain.max_init_terrain_level = 0
        for terrain in generator.sub_terrains.values():
            terrain.proportion = 0.0
        ascending = generator.sub_terrains["pyramid_stairs_inv"]
        ascending.proportion = 1.0
        ascending.step_height_range = (0.12, 0.12)
        self.curriculum.terrain_levels = None


__all__ = [
    "DeeproboticsM20V131V11FDOrderedAlternationEnvCfg",
    "DeeproboticsM20V131RetentionEnvCfg",
    "M20V131OrderingCfg",
]
