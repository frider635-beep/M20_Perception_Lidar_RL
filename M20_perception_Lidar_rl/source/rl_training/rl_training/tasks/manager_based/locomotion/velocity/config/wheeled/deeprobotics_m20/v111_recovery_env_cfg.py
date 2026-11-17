"""Bounded short recovery stages; deployment ABI remains the native 63->16."""
import math
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm, CurriculumTermCfg as CurrTerm
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from .v111_adjacent_env_cfg import DeeproboticsM20V111AdjacentEnvCfg, DeeproboticsM20V111AdjacentPPORunnerCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_recovery as recovery
from rl_training.recovery_assets import resolve_assets


@configclass
class M20V111RecoveryCfg:
    skip_weight: float = 1.5
    overshoot_weight: float = .75
    body_progress_weight: float = 2.
    rear_progress_scale: float = .5
    complete_weight: float = 2.
    contact_force: float = 15.
    plane_tolerance: float = .035
    top_support_seconds: float = .06
    target_speed_fraction: float = 1.
    minimum_speed_fraction: float = .8
    minimum_upright_cosine: float = .8
    minimum_top_clearance: float = .30
    flat_cost_cap: float = 2.
    flat_posture_weight: float = .5
    flat_quality_weight: float = 1.
    flat_quality_speed_scale: float = .5
    straight_stability_weight: float = .5
    straight_yaw_deadband: float = .03
    straight_lateral_deadband: float = .05
    straight_yaw_scale: float = .08
    straight_lateral_scale: float = .12


@configclass
class DeeproboticsM20V111RecoveryEnvCfg(DeeproboticsM20V111AdjacentEnvCfg):
    v111_recovery: M20V111RecoveryCfg = M20V111RecoveryCfg()

    def __post_init__(self):
        super().__post_init__()
        # User-selected default on the migrated host; the CLI may reduce it.
        self.scene.num_envs = 4096
        c = self.v111_recovery
        if any(not math.isfinite(value) or value <= 0 for value in vars(c).values()
               if isinstance(value, (int, float))):
            raise ValueError('Recovery reward scales and budgets must be finite and positive')
        self.rewards.v111_adjacent_skip.weight = c.skip_weight
        self.rewards.v111_adjacent_overshoot = None
        self.rewards.v111_adjacent_complete = None
        # New flat posture and straightness costs are at most 1.0 + 0.5 per
        # second; they cannot erase the native speed reward by multiplication.
        self.rewards.v111_flat_workspace.weight = c.flat_posture_weight
        self.rewards.v111_straight_stability = RewTerm(func=recovery.straight_stability, weight=c.straight_stability_weight)
        self.rewards.v111_recovery_flat_quality = RewTerm(func=recovery.flat_quality, weight=c.flat_quality_weight)
        self.rewards.v111_recovery_swing = RewTerm(func=recovery.overshoot_cost, weight=c.overshoot_weight)
        self.rewards.v111_recovery_progress = RewTerm(func=recovery.body_progress, weight=c.body_progress_weight)
        self.rewards.v111_recovery_complete = RewTerm(func=recovery.whole_completion, weight=c.complete_weight)
        self.commands.base_velocity.rel_only_lin_x_envs = .20
        self.commands.base_velocity.rel_zero_vel_envs = .10
        self.rewards.world_forward_progress.params['target_speed_fraction'] = c.target_speed_fraction
        self.rewards.world_forward_deficit.params['min_speed_fraction'] = c.minimum_speed_fraction
        self.curriculum.terrain_levels.params['whole_body_ascent_gate'] = True
        for key in ('whole_flight', 'progress', 'cost_event'):
            setattr(self.curriculum, 'v111_recovery_'+key, CurrTerm(func=recovery.metric, params={'key': key}))


@configclass
class M20RecoveryPPOCfg(RslRlPpoAlgorithmCfg):
    class_name: str = 'rl_training.v111_recovery_runner:RecoveryPPO'
    teacher_checkpoint: str = ''
    replay_path: str = ''
    anchor_weight: float = 2.
    anchor_front_stair_weight: float = .04
    anchor_flat_leg_weight: float = .04
    anchor_flat_wheel_differential_weight: float = .08
    anchor_ascent_other_weight: float = .35
    anchor_tolerance: float = .05
    anchor_batch_size: int = 256
    value_loss_coef: float = 1.
    use_clipped_value_loss: bool = True
    clip_param: float = .2
    entropy_coef: float = .001
    num_learning_epochs: int = 3
    num_mini_batches: int = 4
    learning_rate: float = 1.e-5
    schedule: str = 'fixed'
    gamma: float = .99
    lam: float = .95
    desired_kl: float = .01
    max_grad_norm: float = 1.
    normalize_advantage_per_mini_batch: bool = False


@configclass
class DeeproboticsM20V111RecoveryPPORunnerCfg(DeeproboticsM20V111AdjacentPPORunnerCfg):
    experiment_name = 'M20-V11.1-AscentRecovery'
    class_name = 'rl_training.v111_recovery_runner:RecoveryRunner'
    num_steps_per_env = 96
    max_iterations = 100
    save_interval = 25
    algorithm = M20RecoveryPPOCfg()

    def __post_init__(self):
        super().__post_init__()
        # Parent EntryFlatRepair installs its own LR in __post_init__.
        self.algorithm.learning_rate = M20RecoveryPPOCfg().learning_rate
        teacher, replay = resolve_assets()
        self.algorithm.teacher_checkpoint = str(teacher)
        self.algorithm.replay_path = str(replay)
