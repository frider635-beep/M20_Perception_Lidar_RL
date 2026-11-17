# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DeeproboticsM20RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "deeprobotics_m20_rough"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DeeproboticsM20FlatPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "deeprobotics_m20_flat"


@configclass
class DeeproboticsM20StairOfficial20KPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """V6 is a fresh official-style PPO run; no checkpoint is ever loaded."""

    max_iterations = 20000
    save_interval = 250
    experiment_name = "m20_stair_official20k_v1"
    resume = False
    load_run = None
    load_checkpoint = None


@configclass
class DeeproboticsM20PerceptiveStairPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """Strict from-scratch PPO settings for the final perceptive task."""

    max_iterations = 30000
    save_interval = 500
    experiment_name = "m20_stair_perceptive_final"
    resume = False
    load_run = None
    load_checkpoint = None


@configclass
class DeeproboticsM20PerceptiveV9Stage1PPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """Short transfer-learning gate; this is not the final 30K experiment."""

    class_name = "rl_training.v9_stage1_runner:V9Stage1Runner"
    max_iterations = 1500
    save_interval = 250
    experiment_name = "m20_perceptive_v9_stage1_colleague9999"
    resume = False
    load_run = None
    load_checkpoint = None
    v9_base_checkpoint = (
        "/home/ubuntu/m20_perceptive_final/eval_outputs/v9_initialization_gate/"
        "colleague_model9999/model_9999.pt"
    )
