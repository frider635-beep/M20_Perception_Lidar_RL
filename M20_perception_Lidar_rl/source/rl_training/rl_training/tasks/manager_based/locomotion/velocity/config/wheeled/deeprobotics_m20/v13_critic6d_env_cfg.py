"""Native V13 task with six reward-state values visible only to the critic."""

from __future__ import annotations

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp

from .v13_v11fd_alternation_env_cfg import DeeproboticsM20V13V11FDAlternationEnvCfg


@configclass
class DeeproboticsM20V13Critic6DEnvCfg(DeeproboticsM20V13V11FDAlternationEnvCfg):
    """Exact V13 behavior with an asymmetric 6D reward-state critic input."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic.v13_reward_state = ObsTerm(
            func=mdp.v13_critic_reward_state,
            params={"sensor_cfg": SceneEntityCfg("v10_pgtt_scanner")},
            noise=None,
            clip=(-2.0, 2.0),
            scale=1.0,
        )
        print("V13_CRITIC6D_NATIVE_TASK=YES", flush=True)
        print("V13_CRITIC6D_ACTOR_OBS_DIM=63", flush=True)
        print("V13_CRITIC6D_CRITIC_OBS_DIM=574", flush=True)
        print("V13_CRITIC6D_ACTION_DIM=16", flush=True)


__all__ = ["DeeproboticsM20V13Critic6DEnvCfg"]
