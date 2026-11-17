"""V11.2: discrete stair treads and persistent, geometry-gated flat posture."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .v111_v11fd_rear_posture_env_cfg import DeeproboticsM20V111V11FDRearPostureEnvCfg
from .agents.rsl_rl_ppo_cfg import DeeproboticsM20V111V11FDRearPosturePPORunnerCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v112_terrain import configure_treads
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v111_support_cycle import (
    support_cycle_reward,
)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v112_flat_posture import flat_metric, level_posture_penalty, level_metric
from rl_training.tasks.manager_based.locomotion.velocity.mdp import v112_handoff


@configclass
class M20V112HandoffCfg:
    contact_edge_margin: float = 0.05
    contact_edge_fraction: float = 0.25
    support_corridor_half_width: float = 0.50
    route_lateral_free: float = 0.25
    route_lateral_scale: float = 0.35
    route_lateral_limit: float = 0.80
    route_heading_scale: float = 0.50
    flat_drive_scale: float = 1.0
    flat_workspace_enabled: bool = True
    flat_reach_free: float = 0.05
    flat_reach_scale: float = 0.10
    flat_joint_free: float = 0.25
    flat_joint_scale: float = 0.50
    flat_joint_cost_scale: float = 0.25


@configclass
class DeeproboticsM20V112VariableTreadEnvCfg(DeeproboticsM20V111V11FDRearPostureEnvCfg):
    v112_tread_widths: tuple[float, ...] = (0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44)
    v112_handoff: M20V112HandoffCfg = M20V112HandoffCfg()

    def __post_init__(self):
        super().__post_init__()
        if not (self.v112_handoff.contact_edge_margin > 0
                and 0 < self.v112_handoff.contact_edge_fraction < 0.5
                and self.v112_handoff.flat_drive_scale > 0
                and self.v112_handoff.flat_reach_free > 0
                and self.v112_handoff.flat_reach_scale > 0
                and self.v112_handoff.flat_joint_free > 0
                and self.v112_handoff.flat_joint_scale > 0
                and self.v112_handoff.flat_joint_cost_scale > 0):
            raise ValueError("V11.2 requires positive posture shaping and nonempty contact bounds")
        if not (0 < self.v112_handoff.route_lateral_free < self.v112_handoff.route_lateral_limit
                and self.v112_handoff.route_lateral_scale > 0 and self.v112_handoff.route_heading_scale > 0):
            raise ValueError("Invalid V11.2 route reward bounds")
        self.v111_rear_posture.same_tread_handoff = True
        self.v111_rear_posture.timed_support_cycle = True
        self.v111_rear_posture.adjacent_landing_credit = True
        # The old +/-0.20 corridor shrank to +/-0.10 after wheel/margin
        # subtraction, excluding normal bilateral wheel spacing.
        self.v11_perception.tread_corridor_half_width = self.v112_handoff.support_corridor_half_width
        self.rewards.v112_support_cycle = RewTerm(
            func=support_cycle_reward, weight=self.v111_rear_posture.cycle_progress_weight)
        self.rewards.v112_cycle_completion = RewTerm(func=v112_handoff.cycle_completion,
            weight=self.v111_rear_posture.cycle_completion_weight)
        self.rewards.track_ang_vel_z_exp.func = v112_handoff.track_yaw
        self.rewards.track_lin_vel_xy_exp.func = v112_handoff.track_velocity
        self.rewards.world_forward_progress.func = v112_handoff.forward_progress
        self.rewards.world_forward_deficit.func = v112_handoff.forward_deficit
        terrain = self.scene.terrain.terrain_generator
        # Twenty columns provide only four descending columns: seven widths
        # would silently disappear. Forty retain the original type proportions.
        terrain.num_cols = 40
        stairs, ascent = configure_treads(terrain, self.v112_tread_widths)
        self.v11_perception.stair_terrain_type_ids = stairs
        self.v11_perception.ascending_stair_terrain_type_ids = ascent
        self.v10_perception.full_stair_column_start = ascent[0]
        self.v10_perception.full_stair_column_end = ascent[-1] + 1
        self.v10_perception.num_terrain_columns = terrain.num_cols
        # Independent of exit settlement, traversal history and elapsed time.
        # Flat posture does not depend on the stair support cycle's completion.
        self.rewards.v112_level_posture = RewTerm(
            func=level_posture_penalty, weight=self.v111_rear_posture.flat_posture_weight,
        )
        for key in ("active", "clearance", "tilt", "cost", "blend", "workspace_cost", "reach_max", "joint_deviation", "extension_rate"):
            setattr(self.curriculum, f"v112_level_{key}", CurrTerm(
                func=level_metric, params={"key": key},
            ))
        for key in ("low_rate", "inactive_rate"):
            setattr(self.curriculum, f"v112_level_{key}", CurrTerm(
                func=flat_metric, params={"key": key},
            ))
        for key in ('samples', 'physical_pairs', 'style_cycles', 'offroute_fraction'):
            setattr(self.curriculum, f'v112_route_{key}', CurrTerm(
                func=v112_handoff.route_metric, params={'key': key}))
        for width in self.v112_tread_widths:
            for key in ("samples", "cycle_ready", "cycle_done", "cycle_bad", "cycle_event", "adjacent_credit_blocked"):
                setattr(self.curriculum, f"v112_tread_{round(width * 100)}_{key}", CurrTerm(
                    func=v112_handoff.tread_metric, params={"width": width, "key": key}))


@configclass
class DeeproboticsM20V112VariableTreadPPORunnerCfg(DeeproboticsM20V111V11FDRearPosturePPORunnerCfg):
    experiment_name = "M20-V11.2-VariableTread"
