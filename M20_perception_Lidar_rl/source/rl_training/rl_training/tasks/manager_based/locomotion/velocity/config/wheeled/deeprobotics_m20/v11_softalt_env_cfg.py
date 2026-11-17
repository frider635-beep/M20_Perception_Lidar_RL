"""V11-only soft-alternation fine-tune configuration."""

from __future__ import annotations

import math

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
class M20V11SoftAltPerceptionCfg(M20V11PerceptionCfg):
    """Stage-1 command and lifecycle settings; actor inputs stay unchanged."""

    soft_alternation: bool = True
    soft_alternation_command_split: bool = True
    stair_forward_command_min: float = 0.20
    stair_forward_command_max: float | None = 0.60
    non_ascending_stair_forward_command_min: float = 0.10
    non_ascending_stair_forward_command_max: float | None = None


@configclass
class M20V11SoftAltRewardsCfg(M20V11RewardsCfg):
    """One-shot reward ledger with weights equal to per-event return."""

    landing_event = None
    platform_exit_success = None
    first_front_completion = RewTerm(func=mdp.v11_first_completion_reward, weight=0.5)
    alternating_completion = RewTerm(func=mdp.v11_alternating_completion_reward, weight=2.5)
    same_side_repeat = RewTerm(func=mdp.v11_same_side_repeat_reward, weight=-1.0)
    simultaneous_completion = RewTerm(func=mdp.v11_simultaneous_completion_reward, weight=-0.5)
    platform_success = RewTerm(func=mdp.v11_softalt_platform_success_reward, weight=3.0)


@configclass
class DeeproboticsM20V11SoftAltEnvCfg(DeeproboticsM20V11ContinuousEnvCfg):
    """Minimal SoftAlt delta over the existing 63-D MLP V11 task."""

    rewards: M20V11SoftAltRewardsCfg = M20V11SoftAltRewardsCfg()
    v11_perception: M20V11SoftAltPerceptionCfg = M20V11SoftAltPerceptionCfg()

    def __post_init__(self):
        super().__post_init__()

        # Preserve 5.0 on flat/rough samples; 0.25 gives an effective 1.25 on
        # V11 forward stair samples without removing the need to advance.
        self.rewards.track_lin_vel_xy_exp.func = mdp.v11_stair_masked_track_lin_vel_xy_exp
        self.rewards.track_lin_vel_xy_exp.params = {
            "command_name": "base_velocity", "std": math.sqrt(0.5), "stair_scale": 0.25,
            "ascending_only": True,
        }
        self.rewards.world_forward_deficit.func = mdp.v11_softalt_world_forward_speed_deficit
        self.rewards.world_forward_deficit.params["ascending_scale"] = 0.5 / 3.0
        self.rewards.world_forward_progress.func = mdp.v11_softalt_world_forward_progress_reward
        self.rewards.world_forward_progress.params["ascending_scale"] = 0.75 / 2.0

        self.curriculum.terrain_levels.params.update({
            "move_up_platform_success_threshold": 0.75,
            "move_up_alternation_compliance_threshold": 0.80,
            "move_up_same_side_repeat_threshold": 0.20,
            "move_up_front_riser_collision_threshold": 0.0,
        })
        soft_metrics = (
            "first_completion_count", "FL_completion_count", "FR_completion_count",
            "alternating_completion_count", "same_side_repeat_count",
            "simultaneous_completion_count", "simultaneous_completion_rate",
            "alternation_compliance_rate", "FL_to_FR_count", "FR_to_FL_count",
            "FL_to_FL_repeat_count", "FR_to_FR_repeat_count",
            "front_riser_collision_rate", "platform_success_rate",
            "stair_forward_velocity_mean",
        )
        for key in soft_metrics:
            setattr(
                self.curriculum, f"v11_{key}",
                CurrTerm(func=mdp.v11_softalt_metric, params={"key": key}),
            )
        # Keep the existing physical-riser mean metric, whose exact public
        # TensorBoard name is already v11_completed_risers_mean.


__all__ = ["DeeproboticsM20V11SoftAltEnvCfg"]
