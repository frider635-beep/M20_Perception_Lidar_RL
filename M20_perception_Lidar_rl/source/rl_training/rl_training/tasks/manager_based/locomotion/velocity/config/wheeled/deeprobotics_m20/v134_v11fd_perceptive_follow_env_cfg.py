"""V13.4: native V13 locomotion with ascent-chain retention and natural descent."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v13_v11fd_alternation_env_cfg import DeeproboticsM20V13V11FDAlternationEnvCfg
from .v133_v11fd_adjacent_transition_env_cfg import M20V133TransitionCfg


@configclass
class M20V134AscentChainCfg(M20V133TransitionCfg):
    """Single ledger for the ascent dividend and descent posture envelope."""

    # Stable B2 and B3 events must outweigh the ordinary-gait advantage while
    # preserving the native +5 physical landing reward for every safe result.
    # The first stable support must match the first world-locked face.  Either
    # FL or FR may discover it; B2 can no longer silently become a new B1.
    require_first_riser_seed: bool = True
    b1_settlement: float = 1.0
    adjacent_transition_settlement: float = 2.0
    chain_settlement: float = 6.0
    # Failed style attempts receive the exact native V13 return.  This removes
    # the learned shortcut in which backing away avoids style costs.
    # Historical-best progress toward the exact adjacent tread makes the rare
    # stable-support outcome learnable.  It is positive-only and bounded; a
    # failed attempt keeps at most this small teaching credit and never creates
    # a negative signal that can be avoided by backing away.
    adjacent_progress_weight: float = 0.50
    repeat_transition_cost: float = 0.0
    skip_transition_cost: float = 0.0
    simultaneous_transition_cost: float = 0.0
    # Descent has no order or target tread.  These bounds only reject the two
    # observed posture extremes: trailing far behind and folding under itself.
    rear_extension_free_reach: float = 0.05
    rear_extension_scale: float = 0.12
    rear_extension_max_penalty: float = 1.0
    rear_extension_weight: float = 0.75
    rear_knee_flexion_free: float = 0.20
    rear_knee_flexion_scale: float = 0.25
    rear_knee_flexion_max_penalty: float = 1.0
    rear_knee_flexion_weight: float = 0.50
    rear_thigh_parallel_free_cosine: float = 0.75
    rear_thigh_parallel_scale: float = 0.20
    rear_thigh_parallel_max_penalty: float = 1.0
    rear_thigh_parallel_weight: float = 0.50
    rear_roll_weight: float = 0.10
    # Retain the final-riser state long enough for both rear wheels to cross;
    # the legacy three-frame target-loss limit expired before confirmation.
    platform_target_loss_release_steps: int = 30
    platform_body_clearance_free_height: float = 0.38
    platform_body_clearance_scale: float = 0.10
    platform_body_clearance_max_penalty: float = 1.0
    platform_body_clearance_weight: float = 1.0


@configclass
class DeeproboticsM20V134PerceptiveFollowEnvCfg(
    DeeproboticsM20V13V11FDAlternationEnvCfg
):
    """Keep V13's entry behavior and add outcome-only gait selection."""

    v133_transition: M20V134AscentChainCfg = M20V134AscentChainCfg()

    def __post_init__(self):
        super().__post_init__()
        cfg = self.v133_transition
        self.v11_perception.platform_target_loss_release_steps = (
            cfg.platform_target_loss_release_steps
        )

        # All additional state is critic-only.  The actor ABI remains the 63D
        # V13 observation used by deployment and by the successful model_5000.
        self.observations.critic.v134_task_state = ObsTerm(
            func=mdp.v134_native_chain_critic_state,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        self.rewards.v134_ascent_chain = RewTerm(
            func=mdp.v134_positive_ascent_chain_reward,
            weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
        )
        self.rewards.v134_descent_rear_extension = RewTerm(
            func=mdp.v13_descent_rear_extension_penalty,
            weight=cfg.rear_extension_weight,
            params={
                "free_reach": cfg.rear_extension_free_reach,
                "reach_scale": cfg.rear_extension_scale,
                "max_penalty": cfg.rear_extension_max_penalty,
                "sensor_cfg": SceneEntityCfg("v10_pgtt_scanner"),
            },
        )
        self.rewards.v134_descent_rear_knee_flexion = RewTerm(
            func=mdp.v134_descent_rear_knee_flexion_penalty,
            weight=cfg.rear_knee_flexion_weight,
            params={
                "free_flexion": cfg.rear_knee_flexion_free,
                "flexion_scale": cfg.rear_knee_flexion_scale,
                "max_penalty": cfg.rear_knee_flexion_max_penalty,
            },
        )
        self.rewards.v134_descent_rear_roll = RewTerm(
            func=mdp.v13_descent_rear_roll_penalty,
            weight=cfg.rear_roll_weight,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
        )
        self.rewards.v134_descent_rear_thigh_parallel = RewTerm(
            func=mdp.v134_descent_rear_thigh_parallel_penalty,
            weight=cfg.rear_thigh_parallel_weight,
            params={
                "free_cosine": cfg.rear_thigh_parallel_free_cosine,
                "cosine_scale": cfg.rear_thigh_parallel_scale,
                "max_penalty": cfg.rear_thigh_parallel_max_penalty,
            },
        )
        self.rewards.v134_platform_body_clearance = RewTerm(
            func=mdp.v134_platform_body_clearance_penalty,
            weight=cfg.platform_body_clearance_weight,
            params={
                "free_height": cfg.platform_body_clearance_free_height,
                "height_scale": cfg.platform_body_clearance_scale,
                "max_penalty": cfg.platform_body_clearance_max_penalty,
                "sensor_cfg": SceneEntityCfg("v10_pgtt_scanner"),
            },
        )

        for key in (
            "seed_count", "adjacent_count", "chain_count", "repeat_count",
            "follower_count", "skip_count", "simultaneous_count",
            "chain_fl_count", "chain_fr_count", "chain_precision",
            "chain_fl_fraction", "chain_active_fraction", "chain_length_mean",
            "adjacent_progress_active_rate", "adjacent_progress_delta_mean",
            "adjacent_progress_escrow_mean", "adjacent_progress_clawback_mean",
            "b1_lock_count", "b1_seed_miss_count", "b1_locked_rate",
            "b1_seed_miss_rate",
        ):
            setattr(
                self.curriculum,
                f"v134_{key}",
                CurrTerm(func=mdp.v133_chain_metric, params={"key": key}),
            )
        for key in (
            "course_active_rate", "live_edge_rate", "edge_distance_mean",
            "drop_height_mean", "rear_extension_max_mean", "rear_contact_fraction",
        ):
            setattr(
                self.curriculum,
                f"v134_descent_{key}",
                CurrTerm(func=mdp.v13_critic_descent_metric, params={"key": key}),
            )
        for key in (
            "rear_thigh_parallel_mean", "rear_thigh_parallel_rate",
            "platform_confirmed_rate", "platform_clearance_active_rate",
            "platform_body_clearance_mean",
            "platform_low_body_rate",
        ):
            setattr(
                self.curriculum,
                f"v134_{key}",
                CurrTerm(func=mdp.v134_posture_metric, params={"key": key}),
            )

        if not 0.0 < float(cfg.longitudinal_tolerance) < 0.5 * float(cfg.step_width):
            raise RuntimeError("V13.4 must distinguish immediately adjacent stair treads.")
        if float(cfg.adjacent_transition_settlement) <= 0.0:
            raise RuntimeError("V13.4 adjacent settlement must be positive.")
        if not bool(cfg.require_first_riser_seed) or float(cfg.b1_settlement) <= 0.0:
            raise RuntimeError("V13.4 must reward only a world-locked physical B1 seed.")
        if float(cfg.chain_settlement) <= float(cfg.adjacent_transition_settlement):
            raise RuntimeError("V13.4 three-step dividend must exceed pairwise settlement.")
        if not 0.0 < float(cfg.adjacent_progress_weight) < float(cfg.adjacent_transition_settlement):
            raise RuntimeError("V13.4 dense adjacent teaching must stay below stable settlement.")
        for name in (
            "repeat_transition_cost", "skip_transition_cost",
            "simultaneous_transition_cost",
        ):
            if float(getattr(cfg, name)) != 0.0:
                raise RuntimeError(f"V13.4 avoidance-prone style term {name} must remain zero.")
        for name in (
            "rear_extension_scale", "rear_extension_max_penalty", "rear_extension_weight",
            "rear_knee_flexion_scale", "rear_knee_flexion_max_penalty",
            "rear_knee_flexion_weight", "rear_roll_weight",
            "rear_thigh_parallel_scale", "rear_thigh_parallel_max_penalty",
            "rear_thigh_parallel_weight", "platform_body_clearance_free_height",
            "platform_body_clearance_scale", "platform_body_clearance_max_penalty",
            "platform_body_clearance_weight",
        ):
            if float(getattr(cfg, name)) <= 0.0:
                raise RuntimeError(f"V13.4 descent bound {name} must be positive.")
        if not 0.0 < float(cfg.rear_thigh_parallel_free_cosine) < 1.0:
            raise RuntimeError("V13.4 rear-thigh free cosine must lie in (0, 1).")
        if int(cfg.platform_target_loss_release_steps) < (
            int(self.v11_perception.platform_no_riser_confirmation_frames)
            + int(self.v11_perception.platform_rear_support_confirmation_frames)
        ):
            raise RuntimeError("V13.4 top-state hold must cover platform confirmation.")

        terrain = self.scene.terrain.terrain_generator.sub_terrains
        for name in ("pyramid_stairs", "pyramid_stairs_inv"):
            if abs(float(terrain[name].step_width) - float(cfg.step_width)) > 1.0e-9:
                raise RuntimeError(
                    f"V13.4 expects {cfg.step_width} m treads, got {terrain[name].step_width}."
                )

        print("V134_BASE_IS_NATIVE_V13=YES", flush=True)
        print("V134_ASCENT_POSITIVE_ADJACENT_SETTLEMENT=YES", flush=True)
        print("V134_ASCENT_WORLD_LOCKED_B1=YES", flush=True)
        print("V134_ASCENT_POSITIVE_ADJACENT_PROGRESS=YES", flush=True)
        print("V134_ADDED_ASCENT_STYLE_COSTS=NO", flush=True)
        print("V134_DESCENT_NATURAL_POSTURE_ENVELOPE=YES", flush=True)
        print("V134_DESCENT_ORDER_OR_TREAD_LOCK=NO", flush=True)
        print("V134_MODEL5000_STUDENT_INITIALIZATION=NO", flush=True)
        print("V134_MODEL5000_ASCENT_TEACHER_ONLY=YES", flush=True)
        print("V134_PLATFORM_CLEARANCE_BEFORE_CONFIRMATION=YES", flush=True)
        print("V134_MODEL5000_ACTION_REFERENCE_REWARD=NO", flush=True)
        print("V134_ACTOR_OBS_DIM=63", flush=True)
        print("V134_CRITIC_OBS_DIM=580", flush=True)
        print("V134_ACTION_DIM=16", flush=True)


__all__ = ["DeeproboticsM20V134PerceptiveFollowEnvCfg", "M20V134AscentChainCfg"]
