"""V13.2: V13 locomotion with recoverable barrier-style alternation."""

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
class M20V132StyleCfg:
    """Single numerical ledger for the complete V13.2 style channel."""

    step_width: float = 0.30
    target_identity_tolerance: float = 0.15
    new_flight_separation: float = 0.60
    b1_lift_onset_progress: float = 0.20

    barrier_delta: float = 0.05
    lead_margin_lower: float = 0.05
    lead_margin_upper: float = 0.95
    expected_progress_weight: float = 1.0
    barrier_progress_weight: float = 1.0
    velocity_coupling_weight: float = 2.0
    same_progress_cost: float = 1.0
    simultaneous_progress_cost: float = 1.5
    ordered_completion_weight: float = 2.0
    wrong_completion_cost: float = 1.5
    simultaneous_event_cost: float = 2.0
    skip_event_cost: float = 4.0
    overshoot_free_distance: float = 0.16
    overshoot_scale: float = 0.12
    overshoot_cost: float = 1.0

    plateau_probe_start: float = 0.14
    plateau_probe_length: float = 0.48
    plateau_corridor_half_width: float = 0.24
    plateau_height_tolerance: float = 0.03
    plateau_min_points: int = 8
    plateau_min_forward_span: float = 0.18
    plateau_confirmation_frames: int = 3
    top_follow_support_confirmation_frames: int = 3
    top_follow_progress_weight: float = 1.5
    top_follow_completion_weight: float = 3.0
    plateau_lift_free_height: float = 0.025
    plateau_lift_scale: float = 0.08
    post_top_lift_cost: float = 0.75

    command_name: str = "base_velocity"
    velocity_tracking_std: float = 0.7071067811865476


@configclass
class M20V132RewardsCfg(M20V13RewardsCfg):
    """One separately learned style return; all native V13 rewards stay intact."""

    v132_style = RewTerm(func=mdp.v132_style_objective, weight=1.0)


@configclass
class DeeproboticsM20V132V11FDBarrierAlternationEnvCfg(
    DeeproboticsM20V13V11FDAlternationEnvCfg
):
    """Independent V13.2 task with a 68-D deployable actor contract."""

    rewards: M20V132RewardsCfg = M20V132RewardsCfg()
    v132_style: M20V132StyleCfg = M20V132StyleCfg()

    def __post_init__(self):
        super().__post_init__()

        # V13.2 replaces only V13's old style layer.  In particular it does not
        # repoint landing_event, clearance settlement, hip participation or
        # landing-centre credit: those remain generic physical V13/V11 rewards.
        for name in (
            "v13_opposite_progress",
            "v13_velocity_coupled_opposite_progress",
            "v13_same_side_progress",
            "v13_simultaneous_progress",
            "v13_alternating_completion",
            "v13_same_side_completion",
        ):
            setattr(self.rewards, name, None)

        self.observations.policy.v132_event_phase = ObsTerm(
            func=mdp.v132_event_phase_observation,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        for name in (
            "potential_active_rate", "opposite_progress_delta_mean",
            "same_side_progress_delta_mean", "simultaneous_progress_delta_mean",
            "opposite_potential_mean", "seed_event_rate", "compliant_event_rate",
            "violation_event_rate", "simultaneous_completion_event_rate",
            "seed_fl_fraction", "natural_compliance", "progress_precision",
            "simultaneous_progress_fraction",
        ):
            setattr(self.curriculum, f"v13_{name}", None)
        for key in (
            "b1_completion_rate", "ordered_completion_rate", "wrong_completion_rate",
            "simultaneous_event_rate", "skip_rate", "plateau_positive_rate",
            "top_follow_success_rate", "expected_progress_delta_mean",
            "same_progress_delta_mean", "simultaneous_progress_delta_mean",
            "barrier_score_mean", "style_reward_mean", "approach_b1_fraction",
            "alternating_fraction", "top_follow_fraction", "plateau_roll_fraction",
        ):
            setattr(
                self.curriculum,
                f"v132_{key}",
                CurrTerm(func=mdp.v132_alternation_metric, params={"key": key}),
            )

        cfg = self.v132_style
        if float(cfg.step_width) <= 0.0:
            raise RuntimeError("V13.2 step width must be positive.")
        if not (0.0 < float(cfg.target_identity_tolerance) <= 0.5 * float(cfg.step_width)):
            raise RuntimeError("V13.2 target identity tolerance must not cross a half-tread boundary.")
        if float(cfg.new_flight_separation) <= float(cfg.step_width):
            raise RuntimeError("V13.2 flight re-arm distance must exceed one tread.")
        terrain_cfg = self.scene.terrain.terrain_generator.sub_terrains
        for name in ("pyramid_stairs", "pyramid_stairs_inv"):
            actual = float(terrain_cfg[name].step_width)
            if abs(actual - float(cfg.step_width)) > 1.0e-9:
                raise RuntimeError(
                    f"V13.2 parity geometry expects step_width={cfg.step_width}, "
                    f"but {name} uses {actual}."
                )
        if not (0.0 < float(cfg.b1_lift_onset_progress) < 1.0):
            raise RuntimeError("V13.2 B1 lift onset must lie strictly inside (0, 1).")
        if not (float(cfg.lead_margin_lower) < float(cfg.lead_margin_upper)):
            raise RuntimeError("V13.2 lead barrier has an empty feasible band.")
        if float(cfg.barrier_delta) <= 0.0:
            raise RuntimeError("V13.2 barrier relaxation must be positive.")
        positive_weights = (
            cfg.expected_progress_weight, cfg.barrier_progress_weight,
            cfg.velocity_coupling_weight, cfg.same_progress_cost,
            cfg.simultaneous_progress_cost, cfg.ordered_completion_weight,
            cfg.wrong_completion_cost, cfg.simultaneous_event_cost,
            cfg.skip_event_cost, cfg.overshoot_cost, cfg.top_follow_progress_weight,
            cfg.top_follow_completion_weight, cfg.post_top_lift_cost,
        )
        if any(float(value) < 0.0 for value in positive_weights):
            raise RuntimeError("V13.2 reward magnitudes must be non-negative; signs live in the settlement code.")
        if float(cfg.overshoot_scale) <= 0.0 or float(cfg.plateau_lift_scale) <= 0.0:
            raise RuntimeError("V13.2 penalty scales must be positive.")
        if not (0.0 <= float(cfg.plateau_probe_start) < float(cfg.plateau_probe_length)):
            raise RuntimeError("V13.2 platform probe interval is invalid.")
        if (
            float(cfg.plateau_corridor_half_width) <= 0.0
            or float(cfg.plateau_height_tolerance) <= 0.0
            or int(cfg.plateau_min_points) < 1
            or float(cfg.plateau_min_forward_span) <= 0.0
        ):
            raise RuntimeError("V13.2 platform evidence geometry is invalid.")
        if float(cfg.velocity_tracking_std) <= 0.0:
            raise RuntimeError("V13.2 velocity tracking standard deviation must be positive.")
        if int(cfg.plateau_confirmation_frames) < 2:
            raise RuntimeError("V13.2 platform recognition requires temporal confirmation.")
        if int(cfg.top_follow_support_confirmation_frames) < 2:
            raise RuntimeError("V13.2 top follower requires stable support confirmation.")


__all__ = [
    "DeeproboticsM20V132V11FDBarrierAlternationEnvCfg",
    "M20V132StyleCfg",
]
