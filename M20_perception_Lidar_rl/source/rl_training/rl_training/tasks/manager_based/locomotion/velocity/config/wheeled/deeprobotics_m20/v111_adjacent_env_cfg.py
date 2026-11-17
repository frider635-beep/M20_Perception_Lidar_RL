"""All-tread landing repair from 35499; B1-only speed shaping remains unchanged."""
import math
from isaaclab.managers import CurriculumTermCfg as CurrTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from .v111_entry_flat_env_cfg import DeeproboticsM20V111EntryFlatEnvCfg, DeeproboticsM20V111EntryFlatPPORunnerCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_adjacent as adjacent


@configclass
class M20V111AdjacentCfg:
    edge_margin: float = .035
    edge_fraction: float = .15
    lateral_margin: float = .03
    contact_force: float = 15.
    plane_tolerance: float = .035
    support_seconds: float = .06
    lift_min: float = .04
    support_weight: float = .8
    skip_weight: float = 4.
    overshoot_weight: float = 3.
    complete_weight: float = 2.
    straight_yaw_deadband: float = .03
    straight_lateral_deadband: float = .05
    straight_yaw_scale: float = .08
    straight_lateral_scale: float = .12
    straight_stability_weight: float = 1.
    straight_sample_fraction: float = .20
    stationary_sample_fraction: float = .10


@configclass
class DeeproboticsM20V111AdjacentEnvCfg(DeeproboticsM20V111EntryFlatEnvCfg):
    v111_adjacent: M20V111AdjacentCfg = M20V111AdjacentCfg()

    def __post_init__(self):
        super().__post_init__()
        c = self.v111_adjacent
        if any(not math.isfinite(value) or value <= 0 for value in vars(c).values() if isinstance(value, (int, float))):
            raise ValueError('Adjacent/straight reward parameters must be finite and positive')
        if c.edge_fraction >= .5:
            raise ValueError('Empty adjacent-tread support window')
        self.rewards.v111_adjacent_support = RewTerm(func=adjacent.support_reward, weight=c.support_weight)
        self.rewards.v111_adjacent_skip = RewTerm(func=adjacent.skip_penalty, weight=c.skip_weight)
        self.rewards.v111_adjacent_overshoot = RewTerm(func=adjacent.overshoot_penalty, weight=c.overshoot_weight)
        self.rewards.v111_adjacent_complete = RewTerm(func=adjacent.complete_reward, weight=c.complete_weight)
        self.rewards.v111_straight_stability = RewTerm(func=adjacent.straight_stability_penalty, weight=c.straight_stability_weight)
        self.commands.base_velocity.rel_only_lin_x_envs = c.straight_sample_fraction
        self.commands.base_velocity.rel_zero_vel_envs = c.stationary_sample_fraction
        for key in ('last_index', 'ever_skipped', 'missed_count', 'complete'):
            setattr(self.curriculum, 'v111_adjacent_'+key, CurrTerm(func=adjacent.metric, params={'key': key}))


@configclass
class DeeproboticsM20V111AdjacentPPORunnerCfg(DeeproboticsM20V111EntryFlatPPORunnerCfg):
    experiment_name = 'M20-V11.1-AdjacentRepair'
    save_interval = 50
