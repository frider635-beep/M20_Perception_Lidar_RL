"""Independent M20 HR-Bridge stair task.

The environment keeps the official M20 Rough 16-D action interface.  The
runner owns action authority and replaces only HR leg actions while S1 is
active.  No legacy per-wheel spatial reward is inherited here.
"""

import copy
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR

from .rough_env_cfg import DeeproboticsM20RoughEnvCfg


@configclass
class M20HRBridgeTriggerCfg:
    min_step_up: float = 0.05
    min_forward_command: float = 0.12
    anticipatory_distance_min: float = 0.08
    anticipatory_distance_max: float = 0.55
    corridor_half_width: float = 0.15
    detection_hysteresis: float = 0.008
    measurement_epsilon: float = 0.0001
    confirm_frames: int = 3
    release_frames: int = 3
    elevation_valid_fraction: float = 0.90


@configclass
class M20HRBridgeTeacherCfg:
    wheel_radius: float = 0.09
    clearance_margin: float = 0.01
    cross_margin: float = 0.01
    landing_height_tolerance: float = 0.035
    support_force_threshold: float = 15.0
    support_frames: int = 4
    vertical_contact_force_threshold: float = 8.0
    bridge_timeout_s: float = 4.0
    bridge_action_delta_limit: float = 0.12


@configclass
class M20HRBridgeRewardCfg:
    # Calibrated against each actor's own return scale. Do not add Base and
    # Bridge returns together: JOINT_SUCCESS intentionally appears in both
    # critics for credit assignment, but is still one environment event.
    hr_success_weight: float = 3.3992834091186523
    perception_miss_weight: float = 6.798566818237305
    joint_success_weight: float = 20.617873382568362
    calibrated: bool = True
    handoff_audit_passed: bool = False
    formal_training_enabled: bool = False


@configclass
class DeeproboticsM20HRBridgeEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Fresh Base + HR Bridge task with a 372-D deployable policy stream."""

    hr_bridge_trigger: M20HRBridgeTriggerCfg = M20HRBridgeTriggerCfg()
    hr_bridge_teacher: M20HRBridgeTeacherCfg = M20HRBridgeTeacherCfg()
    hr_bridge_reward: M20HRBridgeRewardCfg = M20HRBridgeRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        tracked_usd = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "M20/usd/M20.usd")
        self.scene.robot.spawn.usd_path = tracked_usd
        self.scene.num_envs = 4096
        self.scene.terrain.terrain_type = "generator"
        terrain = self.scene.terrain.terrain_generator
        terrain.sub_terrains["pyramid_stairs"].proportion = 0.10
        terrain.sub_terrains["pyramid_stairs_inv"].proportion = 0.30
        terrain.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.20)
        self.scene.terrain.max_init_terrain_level = 1

        # Preserve the official 247-D privileged critic before replacing the
        # actor scanner with the deployable 21 x 15 PGTT elevation map.
        self.scene.height_scanner_critic = copy.deepcopy(self.scene.height_scanner)
        self.scene.height_scanner.offset.pos = (0.25, 0.0, 20.0)
        self.scene.height_scanner.ray_alignment = "yaw"
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.10, size=(2.0, 1.4), ordering="xy"
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # The first 57 columns remain byte-for-byte the official actor order;
        # the 315 deployable elevation values are appended last.
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.perceptive_elevation_actor,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "z_noise_std": 0.0,
                "xy_grid_shift": 0.0,
                "dropout_probability": 0.0,
                "update_period_steps": 1,
                "latency_steps": 0,
            },
            noise=None,
            clip=(0.0, 2.0),
            scale=1.0,
        )
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
        self.commands.base_velocity.perceptive_metrics = False
        self.commands.base_velocity.debug_vis = False

        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.curriculum.command_levels = None
        self.curriculum.gait_level = None

        # Only official Rough rewards survive. Event rewards are owned by the
        # dual-return runner and never become dense per-frame lift shaping.
        self.disable_zero_weight_rewards()
