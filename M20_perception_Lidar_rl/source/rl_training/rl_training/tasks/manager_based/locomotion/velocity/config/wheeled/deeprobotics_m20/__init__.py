# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Flat-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20FlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20FlatTrainerCfg",
    },
)

gym.register(
    id="M20-Stair-Official20K-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_env_cfg:DeeproboticsM20StairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20StairOfficial20KPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-Perceptive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_env_cfg:DeeproboticsM20PerceptiveStairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20PerceptiveStairPPORunnerCfg",
    },
)

gym.register(
    id="M20-Perceptive-V9-Stage1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v9_stage1_env_cfg:DeeproboticsM20PerceptiveV9Stage1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20PerceptiveV9Stage1PPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:DeeproboticsM20RoughTrainerCfg",
    },
)

gym.register(
    id="M20-Stair-HRBridge-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hr_bridge_env_cfg:DeeproboticsM20HRBridgeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20HRBridgePPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-Unified-Perceptive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v10_unified_env_cfg:DeeproboticsM20V10UnifiedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V10UnifiedPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-Continuous-Landing-V11-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v11_continuous_env_cfg:DeeproboticsM20V11ContinuousEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V11ContinuousPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-SoftAlt-V11-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v11_softalt_env_cfg:DeeproboticsM20V11SoftAltEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V11SoftAltPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-Flat-Descent-V11FD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v11_flat_descent_env_cfg:DeeproboticsM20V11FlatDescentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V11FlatDescentPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V11.1-V11FD-RearPosture-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v111_v11fd_rear_posture_env_cfg:DeeproboticsM20V111V11FDRearPostureEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V111V11FDRearPosturePPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V11.1-EntryFlatRepair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v111_entry_flat_env_cfg:DeeproboticsM20V111EntryFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.v111_entry_flat_env_cfg:DeeproboticsM20V111EntryFlatPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V11.1-AdjacentRepair-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v111_adjacent_env_cfg:DeeproboticsM20V111AdjacentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.v111_adjacent_env_cfg:DeeproboticsM20V111AdjacentPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V11.1-AscentRecovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v111_recovery_env_cfg:DeeproboticsM20V111RecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.v111_recovery_env_cfg:DeeproboticsM20V111RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V11.2-VariableTread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v112_variable_tread_env_cfg:DeeproboticsM20V112VariableTreadEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.v112_variable_tread_env_cfg:DeeproboticsM20V112VariableTreadPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13-V11FD-Alternation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v13_v11fd_alternation_env_cfg:DeeproboticsM20V13V11FDAlternationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V13V11FDAlternationPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13-V11FD-Alternation-Critic6D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v13_critic6d_env_cfg:DeeproboticsM20V13Critic6DEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V13Critic6DPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13-V11FD-Alternation-Critic12D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v13_critic12d_env_cfg:DeeproboticsM20V13Critic12DEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V13Critic12DPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13.1-V11FD-OrderedAlternation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v131_v11fd_ordered_alternation_env_cfg:DeeproboticsM20V131V11FDOrderedAlternationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V131V11FDOrderedAlternationPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13.1-V11FD-OrderedAlternation-Retention-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v131_v11fd_ordered_alternation_env_cfg:DeeproboticsM20V131RetentionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V131RetentionPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13.2-V11FD-BarrierAlternation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v132_v11fd_barrier_alternation_env_cfg:DeeproboticsM20V132V11FDBarrierAlternationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V132V11FDBarrierAlternationPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13.3-V11FD-FromScratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v133_v11fd_adjacent_transition_env_cfg:DeeproboticsM20V133AdjacentTransitionAcquisitionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V133AdjacentTransitionPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13.4-V11FD-PerceptiveFollow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v134_v11fd_perceptive_follow_env_cfg:DeeproboticsM20V134PerceptiveFollowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V134PerceptiveFollowPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13.5-V11FD-NativeAlternation-v0",
    entry_point="rl_training.v135_training_env:V135TrainingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v135_v11fd_native_alternation_env_cfg:DeeproboticsM20V135NativeAlternationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V135NativeAlternationPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V13.5-RR-ReturnRedistribution-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v135_soft_env_cfg:DeeproboticsM20V135RREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V135RRPPORunnerCfg",
    },
)

gym.register(
    id="M20-Stair-V12-V9Teacher-6D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.v12_teacher_env_cfg:DeeproboticsM20V12V9Teacher6DEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20V12V9Teacher6DPPORunnerCfg",
    },
)
