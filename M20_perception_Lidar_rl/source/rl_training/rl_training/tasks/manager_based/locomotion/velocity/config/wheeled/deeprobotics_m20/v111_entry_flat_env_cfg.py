"""Fine-tune the selected V11.1 actor for physical B1 entry and flat stance."""
from isaaclab.managers import CurriculumTermCfg as CurrTerm, RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from .v111_v11fd_rear_posture_env_cfg import DeeproboticsM20V111V11FDRearPostureEnvCfg
from .agents.rsl_rl_ppo_cfg import DeeproboticsM20V111V11FDRearPosturePPORunnerCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_entry_flat as repair
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_terrain_geometry import configure_ascent_treads


@configclass
class M20V111EntryFlatCfg:
    entry_edge_margin: float = .035
    entry_edge_fraction: float = .15
    entry_contact_force: float = 15.
    entry_plane_tolerance: float = .035
    entry_support_seconds: float = .06
    entry_skip_seconds: float = .04
    entry_lift_min: float = .04
    entry_approach_distance: float = .40
    entry_speed_limit: float = .35
    entry_speed_scale: float = .35
    entry_speed_cost_scale: float = 1.
    entry_penalty_weight: float = 3.
    entry_support_weight: float = 2.
    entry_skip_weight: float = 4.
    flat_reach_free: float = .07
    flat_reach_scale: float = .12
    flat_joint_free: float = .30
    flat_joint_scale: float = .50
    flat_joint_cost_scale: float = .10
    flat_posture_weight: float = 2.5
    flat_drive_cost_scale: float = .35
    ascent_treads: tuple[float, ...] = (.24, .26, .28, .30, .34, .38)


@configclass
class DeeproboticsM20V111EntryFlatEnvCfg(DeeproboticsM20V111V11FDRearPostureEnvCfg):
    v111_entry_flat: M20V111EntryFlatCfg = M20V111EntryFlatCfg()

    def __post_init__(self):
        super().__post_init__()
        c = self.v111_entry_flat
        if any(value <= 0 for name, value in vars(c).items() if isinstance(value, (int, float))):
            raise ValueError('Entry/flat reward bounds and weights must be positive')
        if not c.entry_edge_fraction < .5:
            raise ValueError('Invalid entry bounds')
        # No new global cycle, alternation constraint, action filter or actor input.
        self.rewards.v111_entry_overshoot_speed = RewTerm(func=repair.entry_penalty, weight=c.entry_penalty_weight)
        self.rewards.v111_entry_support = RewTerm(func=repair.entry_support_reward, weight=c.entry_support_weight)
        self.rewards.v111_entry_skip = RewTerm(func=repair.entry_skip_penalty, weight=c.entry_skip_weight)
        self.rewards.v111_flat_workspace = RewTerm(func=repair.flat_penalty, weight=c.flat_posture_weight)
        self.rewards.track_lin_vel_xy_exp.func = repair.track_velocity
        self.rewards.world_forward_progress.func = repair.forward_progress
        self.rewards.world_forward_deficit.func = repair.forward_deficit
        configure_ascent_treads(self.scene.terrain.terrain_generator, c.ascent_treads)
        for key in ('entry_certified', 'entry_failed', 'entry_near', 'entry_speed_cost',
                    'entry_overshoot', 'flat_active', 'flat_workspace_cost', 'flat_reach_max'):
            setattr(self.curriculum, 'v111_repair_'+key, CurrTerm(func=repair.metric, params={'key': key}))


@configclass
class DeeproboticsM20V111EntryFlatPPORunnerCfg(DeeproboticsM20V111V11FDRearPosturePPORunnerCfg):
    experiment_name = 'M20-V11.1-EntryFlatRepair'
    max_iterations = 1500
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.e-4
        self.algorithm.entropy_coef = 1.e-3
        self.algorithm.schedule = 'fixed'
