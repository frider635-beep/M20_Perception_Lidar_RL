"""M20 V10 unified first-riser perception task."""

from __future__ import annotations

import copy
import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.assets import ISAACLAB_ASSETS_DATA_DIR

from .rough_env_cfg import DeeproboticsM20RewardsCfg, DeeproboticsM20RoughEnvCfg


@configclass
class M20V10PerceptionCfg:
    min_step_height: float = 0.05
    max_step_height: float = 0.23
    height_valid_tolerance: float = 0.002
    anticipation_distance_min: float = 0.02
    anticipation_distance_max: float = 0.70
    forward_command_threshold: float = 0.05
    corridor_half_width: float = 0.20
    confirmation_frames: int = 2
    hysteresis_frames: int = 2
    track_height_tolerance: float = 0.04
    track_max_forward_jump: float = 0.12
    distance_filter_alpha: float = 0.5
    pgtt_pass_exit_distance: float = -0.15
    pgtt_pass_lost_max_distance: float = -0.05
    pass_confirmation_frames: int = 2
    rearm_no_stair_steps: int = 50
    perception_delay_steps: int = 2
    height_noise_std: float = 0.005
    distance_noise_std: float = 0.010
    frame_hold_probability: float = 0.02
    wheel_radius: float = 0.09
    clearance_margin: float = 0.01
    contact_force_threshold: float = 1.0
    vertical_force_ratio: float = 0.35
    upper_support_tolerance: float = 0.04
    # Official MeshInvertedPyramidStairsTerrainCfg tread depth.
    step_depth: float = 0.30
    full_success_margin: float = 0.25
    # Resolved from the inherited official Rough generator in __post_init__.
    # Official Rough has no V10 single-step topology. Only the columns assigned
    # to pyramid_stairs_inv are Full; every other official terrain stays Rough.
    single_stair_column_start: int = -1
    single_stair_column_end: int = -1
    full_stair_column_start: int = -1
    full_stair_column_end: int = -1
    num_terrain_columns: int = 0
    single_successes_to_full: int = 3
    stair_reset_distance_range: tuple[float, float] = (0.9, 1.4)
    stair_reset_lateral_range: tuple[float, float] = (-0.25, 0.25)
    stair_reset_yaw_jitter_range: tuple[float, float] = (-0.2617993878, 0.2617993878)


@configclass
class M20V10EventRewardsCfg(DeeproboticsM20RewardsCfg):
    # Calibrated against 24 blind six-step rollouts on 5/8/11/14/17/20cm.
    # RewardManager multiplies each one-frame pulse by control_dt=0.02s,
    # FIRST is intentionally only one tenth of its original calibrated pulse:
    # RewardManager applies dt=0.02, yielding +0.824852828 per event.
    first_step_success = RewTerm(func=mdp.v10_first_step_success_reward, weight=41.2426414)
    # Keep the paired FIRST/MISS teaching pulses symmetric. With dt=0.02,
    # this contributes -0.824852828 per miss instead of the old -12.37279244.
    perception_miss = RewTerm(func=mdp.v10_perception_miss_reward, weight=-41.2426414)
    joint_success = RewTerm(func=mdp.v10_joint_success_reward, weight=1649.705658)
    # The function divides delta_progress by control_dt, so the actual
    # RewardManager contribution is exactly 2.0 * delta_progress.
    fr_clearance_progress = RewTerm(func=mdp.v10_fr_clearance_progress_reward, weight=2.0)
    # Exact duplicate of official XY command tracking, enabled by an episode-
    # latched FIRST handoff. It follows live [vx, vy], not b or target geometry.
    post_first_track_lin_vel_xy_exp = RewTerm(
        func=mdp.v10_post_first_track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )


@configclass
class DeeproboticsM20V10UnifiedEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Official Rough locomotion plus sparse first-riser event learning."""

    rewards: M20V10EventRewardsCfg = M20V10EventRewardsCfg()
    v10_perception: M20V10PerceptionCfg = M20V10PerceptionCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.usd_path = f"{ISAACLAB_ASSETS_DATA_DIR}/M20/usd/M20.usd"
        self.scene.num_envs = 4096

        # Keep the complete inherited official M20 Rough terrain generator.
        # Resolve the exact curriculum columns assigned to the ascending stair
        # type using Isaac Lab's own cumulative-proportion column semantics.
        terrain_cfg = self.scene.terrain.terrain_generator
        sub_names = list(terrain_cfg.sub_terrains.keys())
        proportions = [float(item.proportion) for item in terrain_cfg.sub_terrains.values()]
        total_proportion = sum(proportions)
        cumulative = []
        running = 0.0
        for proportion in proportions:
            running += proportion / total_proportion
            cumulative.append(running)
        column_names = []
        for column in range(int(terrain_cfg.num_cols)):
            choice = column / float(terrain_cfg.num_cols) + 0.001
            sub_index = next(index for index, upper in enumerate(cumulative) if choice < upper)
            column_names.append(sub_names[sub_index])
        ascending_columns = [
            column for column, name in enumerate(column_names) if name == "pyramid_stairs_inv"
        ]
        if not ascending_columns or ascending_columns != list(
            range(ascending_columns[0], ascending_columns[-1] + 1)
        ):
            raise RuntimeError(
                "Official Rough pyramid_stairs_inv columns must exist and form one contiguous range."
            )
        self.v10_perception.full_stair_column_start = ascending_columns[0]
        self.v10_perception.full_stair_column_end = ascending_columns[-1] + 1
        self.v10_perception.num_terrain_columns = int(terrain_cfg.num_cols)

        # Independent deployable 21x15 PGTT. The official 17x11 critic scan is
        # preserved byte-for-byte as the inherited height_scanner.
        self.scene.v10_pgtt_scanner = copy.deepcopy(self.scene.height_scanner)
        self.scene.v10_pgtt_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.v10_pgtt_scanner.offset.pos = (0.25, 0.0, 20.0)
        self.scene.v10_pgtt_scanner.ray_alignment = "yaw"
        self.scene.v10_pgtt_scanner.pattern_cfg = patterns.GridPatternCfg(
            resolution=0.10, size=(2.0, 1.4), ordering="xy"
        )
        self.scene.v10_pgtt_scanner.update_period = self.decimation * self.sim.dt

        # Rough policy is exactly 57D; append only delayed/noisy [b,H,D].
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.v10_actor_perception,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
            clip=(0.0, 2.0),
            scale=1.0,
        )
        # Critic remains asymmetric: official 247D + clean PGTT + truth labels.
        self.observations.critic.v10_pgtt = ObsTerm(
            func=mdp.v10_critic_pgtt,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
        )
        self.observations.critic.v10_geometry = ObsTerm(
            func=mdp.v10_critic_geometry,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
        )
        self.observations.critic.v10_teacher = ObsTerm(
            func=mdp.v10_critic_teacher,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
        )

        # Inherit the complete official M20 Rough velocity-command sampler:
        # vx/vy/wz ranges, heading mode, and zero/axis-only sample fractions.
        self.commands.base_velocity.perceptive_metrics = False

        # Keep the inherited official reset byte-for-byte on non-stair Rough
        # terrains; constrain only ascending stair approaches.
        self.events.randomize_reset_base.func = mdp.v10_reset_root_state_stair_conditioned
        self.events.randomize_reset_base.params.update({
            "stair_distance_range": self.v10_perception.stair_reset_distance_range,
            "stair_lateral_range": self.v10_perception.stair_reset_lateral_range,
            "stair_yaw_jitter_range": self.v10_perception.stair_reset_yaw_jitter_range,
        })

        # Restore the official M20 distance-only terrain curriculum. FIRST/MISS
        # remain reward and diagnostic labels and never affect promotion/demotion.
        self.curriculum.terrain_levels = CurrTerm(func=mdp.v10_official_terrain_levels)
        self.curriculum.command_levels = None
        self.curriculum.gait_level = CurrTerm(func=mdp.gait_level_curve)
        self.terminations.terrain_out_of_bounds = None
        self.disable_zero_weight_rewards()


__all__ = ["DeeproboticsM20V10UnifiedEnvCfg", "M20V10PerceptionCfg"]
