"""V13: native V11-FD emergence plus velocity-coupled soft alternation."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v11_flat_descent_env_cfg import (
    DeeproboticsM20V11FlatDescentEnvCfg,
    M20V11FDPerceptionCfg,
    M20V11FDRewardsCfg,
)


@configclass
class M20V13AlternationCfg:
    """Single numerical ledger for V13's only behavioral delta from V11-FD."""

    minimum_riser_separation: float = 0.15


@configclass
class M20V13RewardsCfg(M20V11FDRewardsCfg):
    """Soft gait-selection layer; native reward weights remain intact."""

    v13_opposite_progress = RewTerm(
        func=mdp.v13_opposite_progress_reward, weight=0.5,
    )
    v13_velocity_coupled_opposite_progress = RewTerm(
        func=mdp.v13_velocity_coupled_opposite_progress_reward,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.7071067811865476},
    )
    v13_same_side_progress = RewTerm(
        func=mdp.v13_same_side_progress_reward, weight=-0.75,
    )
    v13_simultaneous_progress = RewTerm(
        func=mdp.v13_simultaneous_progress_reward, weight=-0.5,
    )
    v13_alternating_completion = RewTerm(
        func=mdp.v13_alternating_completion_reward, weight=1.0,
    )
    v13_same_side_completion = RewTerm(
        func=mdp.v13_same_side_completion_reward, weight=-0.5,
    )


@configclass
class DeeproboticsM20V13V11FDAlternationEnvCfg(DeeproboticsM20V11FlatDescentEnvCfg):
    """Keep V11-FD's 63D actor/safety while alternating one leg per riser."""

    rewards: M20V13RewardsCfg = M20V13RewardsCfg()
    v11_perception: M20V11FDPerceptionCfg = M20V11FDPerceptionCfg()
    v13_alternation: M20V13AlternationCfg = M20V13AlternationCfg()

    def __post_init__(self):
        super().__post_init__()

        # The named 2026-09-02 run is the authority, not the later V11-FD
        # source currently present in this archive.  Restore the material
        # post-run config drift while leaving V11 itself untouched.
        self.scene.terrain.max_init_terrain_level = 5
        for terrain in self.scene.terrain.terrain_generator.sub_terrains.values():
            terrain.size = (8.0, 8.0)
        for name in ("random_rough", "hf_pyramid_slope", "hf_pyramid_slope_inv"):
            self.scene.terrain.terrain_generator.sub_terrains[name].slope_threshold = 0.75

        self.v11_perception.enable_persistent_descent_rear_phase = False
        # V13's one intentional physical-completion delta: one front side may
        # safely complete each successive world-space riser. Both candidates
        # remain physically eligible; alternation is reward style, not a gate.
        self.v11_perception.front_completion_mode = "alternating_risers"
        self.v11_perception.soft_alternation = True
        self.v11_perception.soft_alternation_command_split = False
        self.v11_perception.soft_alternation_curriculum_gate = False
        self.rewards.landing_event.func = mdp.v13_any_front_riser_completion_reward
        self.rewards.landing_center.func = mdp.v13_any_front_riser_center_reward
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
            "v11_descent_rear_support_event_rate",
            "v11_fd_safe_support_event_rate",
            "v11_fd_hard_candidate_event_rate",
        ):
            setattr(self.curriculum, name, None)

        cfg = self.v13_alternation
        if abs(float(cfg.minimum_riser_separation) - float(self.v11_perception.encounter_riser_switch_distance)) > 1.0e-9:
            raise RuntimeError("V13 and V11-FD must use the same distinct-riser separation.")

        for key in (
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
            setattr(
                self.curriculum,
                f"v13_{key}",
                CurrTerm(func=mdp.v13_alternation_metric, params={"key": key}),
            )


__all__ = [
    "DeeproboticsM20V13V11FDAlternationEnvCfg",
    "M20V13AlternationCfg",
]
