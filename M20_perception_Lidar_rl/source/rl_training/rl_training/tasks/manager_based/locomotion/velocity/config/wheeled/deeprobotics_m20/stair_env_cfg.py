"""V6: M20-native control with deployable elevation and fixed square stairs."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import patterns

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg, TerminationsCfg
from .m20_stair_terrains import V6_STAIR_TERRAINS_CFG
from .rough_env_cfg import DeeproboticsM20RoughEnvCfg, DeeproboticsM20RewardsCfg


@configclass
class V6RewardsCfg(DeeproboticsM20RewardsCfg):
    clearance_deficit = RewTerm(func=mdp.v6_clearance_deficit, weight=-1.0)
    riser_collision = RewTerm(
        func=mdp.v6_riser_collision, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel"), "force_threshold": 20.0},
    )
    wheel_crossing = RewTerm(
        func=mdp.v6_wheel_crossing, weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wheel")},
    )
    step_complete = RewTerm(func=mdp.v6_step_complete, weight=1.0)
    full_stair_success = RewTerm(func=mdp.v6_full_stair_success, weight=1.0)
    stair_failure = RewTerm(func=mdp.v6_stair_failure, weight=1.0)
    stair_stagnation = RewTerm(func=mdp.v6_stagnation, weight=1.0)


@configclass
class V6TerminationsCfg(TerminationsCfg):
    stair_success = DoneTerm(func=mdp.v6_success_termination)


@configclass
class DeeproboticsM20StairEnvCfg(DeeproboticsM20RoughEnvCfg):
    rewards: V6RewardsCfg = V6RewardsCfg()
    terminations: V6TerminationsCfg = V6TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        self.scene.terrain.terrain_generator = V6_STAIR_TERRAINS_CFG
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = 0

        # The actor sees only proprioception, command, previous actions and a
        # 21x15 (315-D) scanner patch.  The critic retains privileged base lin vel.
        self.scene.height_scanner.offset.pos = (0.25, 0.0, 20.0)
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.10, size=(2.0, 1.4))
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.v6_local_elevation,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=None,
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        self.observations.critic.base_lin_vel = self.observations.critic.base_lin_vel
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.v6_local_elevation,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=None,
            clip=(-2.0, 2.0),
            scale=1.0,
        )

        # Fixed specialist command distribution: no standing, lateral or yaw tasks.
        self.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.65)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_zero_vel_envs = 0.0
        self.commands.base_velocity.rel_only_lin_y_envs = 0.0
        self.commands.base_velocity.rel_only_lin_x_envs = 0.0
        self.commands.base_velocity.rel_only_ang_z_envs = 0.0
        self.commands.base_velocity.v6_metrics = True

        # Replace broad official spawn randomization with central-flat cardinal spawn.
        self.events.randomize_reset_base = EventTerm(
            func=mdp.reset_v6_task_pose, mode="reset", params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None
        self.curriculum.gait_level = None
        self.scene.terrain.terrain_generator.curriculum = False

        # V6 task terms are integrated by the manager with explicit dt semantics.
        self.rewards.clearance_deficit.weight = 1.0
        self.rewards.riser_collision.weight = 1.0
        self.rewards.wheel_crossing.weight = 1.0
        self.rewards.step_complete.weight = 1.0
        self.rewards.full_stair_success.weight = 1.0
        self.rewards.stair_failure.weight = 1.0
        self.rewards.stair_stagnation.weight = 1.0

        # Keep official control/action/PD/limits untouched; only observation and task
        # terrain are changed in this subclass.
        self.disable_zero_weight_rewards()
