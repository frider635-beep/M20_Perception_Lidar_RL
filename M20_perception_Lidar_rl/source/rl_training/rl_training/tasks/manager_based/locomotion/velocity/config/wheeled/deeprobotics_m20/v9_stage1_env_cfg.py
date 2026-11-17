"""M20 Perceptive V9 Stage-1: transferred locomotion plus first-riser learning."""

from __future__ import annotations

import copy
import os

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_CFG

from .perceptive_env_cfg import DeeproboticsM20PerceptiveStairEnvCfg
from .v9_stage1_terrains import V9_STAGE1_TERRAINS_CFG


@configclass
class DeeproboticsM20PerceptiveV9Stage1EnvCfg(DeeproboticsM20PerceptiveStairEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = copy.deepcopy(DEEPROBOTICS_M20_CFG).replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "M20/usd/M20.usd")
        # Match the colleague policy's physical nominal pose.  Relative joint
        # observations and action default offsets therefore retain the exact
        # 57-D semantics used by model_9999.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.52)
        self.scene.robot.init_state.joint_pos = {
            ".*_hipx_joint": 0.0,
            "f[l,r]_hipy_joint": -0.6,
            "h[l,r]_hipy_joint": 0.6,
            "f[l,r]_knee_joint": 1.0,
            "h[l,r]_knee_joint": -1.0,
            ".*_wheel_joint": 0.0,
        }

        self.scene.num_envs = 4096
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = copy.deepcopy(V9_STAGE1_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.curriculum.gait_level = None

        # The inherited scanner/observation is the already accepted 57+315D
        # pipeline. Only the V9 terrain and nominal pose differ.
        self.commands.base_velocity.ranges.lin_vel_x = (0.20, 0.60)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (0.40, 0.40)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_zero_vel_envs = 0.0
        # Stage-1 metrics are emitted by V9Stage1Runner because the legacy
        # upstairs logger assumes the official pyramid terrain names.
        self.commands.base_velocity.perceptive_metrics = False
        self.events.randomize_reset_base = EventTerm(
            func=mdp.reset_perceptive_stair_pose, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "spawn_x_jitter": 0.10,
                    "spawn_y_jitter": 0.05, "spawn_yaw_jitter": 0.0261799},
        )
        # A single-riser Stage-1 patch is intentionally tiled in Y. The
        # inherited global terrain-edge termination uses absolute world bounds
        # for a large multi-row map and would truncate valid episodes here.
        self.terminations.terrain_out_of_bounds = None
