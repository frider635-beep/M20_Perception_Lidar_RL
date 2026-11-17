"""Final from-scratch M20 task: proprioception + elevation to full 16-D action."""

import copy
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR

from .rough_env_cfg import DeeproboticsM20RewardsCfg, DeeproboticsM20RoughEnvCfg


@configclass
class M20SpatialGuidanceCfg:
    """TASK_CALIBRATED ledger for metres, grid geometry, and reward kernels."""

    corridor_half_width: float = 0.15
    h_min: float = 0.02
    d_start: float = 0.35
    d_peak: float = 0.07
    d_hold: float = -0.03
    d_end: float = -0.12
    # OFFICIAL_FROZEN collision radius from the actual M20 USD capsule.
    wheel_collision_radius: float = 0.09
    # TASK_CALIBRATED collision clearance, not a leg-lift margin.
    clearance_margin: float = 0.01
    # TASK_CALIBRATED temporary numerical clamp: 0.85 x a grid search bounded
    # by official soft joint limits. NOT A PHYSICAL OR DEPLOYMENT SAFE LIMIT.
    # DO NOT use it as final task calibration or call it an official range.
    l_workspace_numeric_bound: float = 0.766
    flat_relief_threshold: float = 0.02
    l_flat_tol: float = 0.035
    false_lift_scale: float = 0.10


@configclass
class M20ElevationDomainRandomizationCfg:
    """TASK_CALIBRATED actor perturbations; defaults make actor equal clean input."""

    z_noise_std: float = 0.0
    xy_grid_shift: float = 0.0
    dropout_probability: float = 0.0
    update_period_steps: int = 1
    latency_steps: int = 0


@configclass
class M20PerceptiveRewardsCfg(DeeproboticsM20RewardsCfg):
    # TASK_CALIBRATED additions. All inherited official reward terms are locked
    # by scripts/tools/test_m20_perceptive_acceptance.py, including full params.
    spatial_lift = RewTerm(
        func=mdp.perceptive_spatial_lift,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    false_lift = RewTerm(
        func=mdp.perceptive_false_lift,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class DeeproboticsM20PerceptiveStairEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Independent task; the official Rough task and its action layout remain unchanged."""

    rewards: M20PerceptiveRewardsCfg = M20PerceptiveRewardsCfg()
    spatial_guidance: M20SpatialGuidanceCfg = M20SpatialGuidanceCfg()
    elevation_randomization: M20ElevationDomainRandomizationCfg = M20ElevationDomainRandomizationCfg()

    def __post_init__(self):
        super().__post_init__()
        # Do not alter the official global M20 asset configuration.  A clean
        # submodule stores its byte-identical USD under ``M20/usd``; override
        # only this independent task instance so fresh worktrees are runnable.
        tracked_usd = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "M20/usd/M20.usd")
        self.scene.robot.spawn.usd_path = tracked_usd
        self.scene.num_envs = 4096
        self.scene.terrain.terrain_type = "generator"
        # Keep the official DeepRobotics M20 Rough terrain generator inherited
        # from DeeproboticsM20RoughEnvCfg, including its boxes/random_rough
        # overrides. This task's only terrain control variable is the stair
        # composition: down 20% -> 10%, up 20% -> 30%.
        terrain_generator = self.scene.terrain.terrain_generator
        terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.10
        terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.30
        self.scene.terrain.max_init_terrain_level = 5

        # Preserve the official 17 x 11 privileged critic height scan before
        # changing the actor/reward scanner to the deployable 21 x 15 map.
        self.scene.height_scanner_critic = copy.deepcopy(self.scene.height_scanner)

        # 21 rows x 15 columns: scanner centre +0.25 m forward, yaw aligned.
        self.scene.height_scanner.offset.pos = (0.25, 0.0, 20.0)
        self.scene.height_scanner.ray_alignment = "yaw"
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.10, size=(2.0, 1.4), ordering="xy"
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        actor_elevation_params = {
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "z_noise_std": self.elevation_randomization.z_noise_std,
            "xy_grid_shift": self.elevation_randomization.xy_grid_shift,
            "dropout_probability": self.elevation_randomization.dropout_probability,
            "update_period_steps": self.elevation_randomization.update_period_steps,
            "latency_steps": self.elevation_randomization.latency_steps,
        }
        # Official actor order is untouched; elevation is appended last.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.perceptive_elevation_actor,
            params=actor_elevation_params.copy(),
            noise=None,
            clip=(0.0, 2.0),
            scale=1.0,
        )
        # Preserve the complete official 247-D M20 critic definition. Only its
        # height-scan sensor name changes so the actor's new 315-D scanner does
        # not replace the official 187-D privileged scan.
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_critic")},
            noise=None,
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.60)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_zero_vel_envs = 0.10
        self.commands.base_velocity.rel_only_lin_y_envs = 0.0
        self.commands.base_velocity.rel_only_lin_x_envs = 0.0
        self.commands.base_velocity.rel_only_ang_z_envs = 0.0
        self.commands.base_velocity.perceptive_metrics = True

        self.events.randomize_reset_base = EventTerm(
            func=mdp.reset_perceptive_stair_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "spawn_x_jitter": 0.20,
                "spawn_y_jitter": 0.20,
                "spawn_yaw_jitter": 0.0872664626,
            },
        )
        # Reuse the official rough-terrain distance curriculum for the actual
        # official MeshPyramidStairsTerrainCfg training distribution. Fixed
        # 17/20 cm x 31 cm delivery stairs remain a separate test generator.
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.curriculum.command_levels = None
        self.curriculum.gait_level = None

        # No result reward, early stagnation termination, or task success termination.
        # Official M20 locomotion/stability/contact rewards remain as inherited.
        self.disable_zero_weight_rewards()
