"""V13 actor ABI with balanced ascent pressure and natural descent rear motion."""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v13_critic6d_env_cfg import DeeproboticsM20V13Critic6DEnvCfg


@configclass
class M20V13Critic12DBalanceCfg:
    """Numerical authority for the two measured V13 failure modes."""

    ascent_command_min: float = 0.20
    ascent_command_max: float = 0.60
    descent_command_min: float = 0.10
    descent_command_max: float | None = None
    ascent_velocity_scale: float = 0.25
    ascent_speed_deficit_scale: float = 1.0 / 6.0
    route_target_speed_fraction: float = 0.50
    route_heading_scale: float = 0.50
    route_lateral_tolerance: float = 0.45
    route_lateral_scale: float = 0.35
    rear_extension_free_reach: float = 0.08
    rear_extension_scale: float = 0.12
    rear_extension_max_penalty: float = 1.0
    rear_extension_weight: float = 0.50
    rear_roll_weight: float = 0.20


@configclass
class DeeproboticsM20V13Critic12DEnvCfg(DeeproboticsM20V13Critic6DEnvCfg):
    """Keep the 63D actor while making stair rewards reachable and bounded."""

    v13_critic12d_balance: M20V13Critic12DBalanceCfg = M20V13Critic12DBalanceCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic.v13_descent_state = ObsTerm(
            func=mdp.v13_critic_descent_state,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        cfg = self.v13_critic12d_balance

        # Preserve full-speed flat/rough training. On ascending stair columns,
        # sample moderate commands and reduce only the velocity and speed-
        # deficit pressure. World-route forward progress, heading and lateral
        # terms stay active, so turning away cannot become a profitable pause.
        self.v11_perception.soft_alternation_command_split = True
        self.v11_perception.stair_forward_command_min = cfg.ascent_command_min
        self.v11_perception.stair_forward_command_max = cfg.ascent_command_max
        self.v11_perception.non_ascending_stair_forward_command_min = cfg.descent_command_min
        self.v11_perception.non_ascending_stair_forward_command_max = cfg.descent_command_max
        route_params = {
            "stair_scale": cfg.ascent_velocity_scale,
            "target_speed_fraction": cfg.route_target_speed_fraction,
            "heading_scale": cfg.route_heading_scale,
            "lateral_tolerance": cfg.route_lateral_tolerance,
            "lateral_scale": cfg.route_lateral_scale,
        }
        self.rewards.track_lin_vel_xy_exp.func = mdp.v11_ascent_route_coupled_track_lin_vel_xy_exp
        self.rewards.track_lin_vel_xy_exp.params.update({
            **route_params,
        })
        self.rewards.track_ang_vel_z_exp.func = mdp.v11_ascent_route_coupled_track_ang_vel_z_exp
        self.rewards.track_ang_vel_z_exp.params.update(route_params)
        self.rewards.world_forward_deficit.func = mdp.v11_softalt_world_forward_speed_deficit
        self.rewards.world_forward_deficit.params["ascending_scale"] = (
            cfg.ascent_speed_deficit_scale
        )

        # No tread identity or left/right order is retained on descent. The
        # first term only acts after a rear wheel retreats beyond a generous
        # episode-start free zone; the second uses current wheel contact and
        # rolling speed throughout the descending terrain column.
        self.rewards.v13_descent_rear_extension = RewTerm(
            func=mdp.v13_descent_rear_extension_penalty,
            weight=cfg.rear_extension_weight,
            params={
                "free_reach": cfg.rear_extension_free_reach,
                "reach_scale": cfg.rear_extension_scale,
                "max_penalty": cfg.rear_extension_max_penalty,
                "sensor_cfg": SceneEntityCfg("v10_pgtt_scanner"),
            },
        )
        self.rewards.v13_descent_rear_roll = RewTerm(
            func=mdp.v13_descent_rear_roll_penalty,
            weight=cfg.rear_roll_weight,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
        )
        for key in (
            "course_active_rate", "live_edge_rate", "edge_distance_mean",
            "drop_height_mean", "rear_extension_max_mean", "rear_contact_fraction",
        ):
            setattr(
                self.curriculum,
                f"v13_critic_descent_{key}",
                CurrTerm(func=mdp.v13_critic_descent_metric, params={"key": key}),
            )
        self.curriculum.v13_ascent_route_engagement = CurrTerm(
            func=mdp.v11_ascent_route_engagement_metric,
        )
        if not 0.0 < float(cfg.route_target_speed_fraction) <= 1.0:
            raise RuntimeError("V13 route target-speed fraction must lie in (0, 1].")
        if float(cfg.route_heading_scale) <= 0.0 or float(cfg.route_lateral_scale) <= 0.0:
            raise RuntimeError("V13 route quality scales must be positive.")
        if float(cfg.route_lateral_tolerance) < 0.0:
            raise RuntimeError("V13 route lateral tolerance must be non-negative.")
        print("V13_CRITIC12D_ACTOR_NATIVE_ABI=YES", flush=True)
        print("V13_CRITIC12D_ASCENT_SPEED_PRESSURE_REDUCED=YES", flush=True)
        print("V13_CRITIC12D_ASCENT_TRACKING_ROUTE_COUPLED=YES", flush=True)
        print("V13_CRITIC12D_DESCENT_TREAD_LOCK=NO", flush=True)
        print("V13_CRITIC12D_ACTOR_OBS_DIM=63", flush=True)
        print("V13_CRITIC12D_CRITIC_OBS_DIM=580", flush=True)
        print("V13_CRITIC12D_ACTION_DIM=16", flush=True)


__all__ = ["DeeproboticsM20V13Critic12DEnvCfg", "M20V13Critic12DBalanceCfg"]
