# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class V12TeacherMoEPolicyCfg(RslRlPpoActorCriticCfg):
    """V9 teacher architecture parameters carried by the V12 runner bridge."""

    class_name: str = "V12Teacher6DMoEActorCritic"
    num_moe_experts: int = 4
    moe_gate_hidden_dims: list[int] = [128]
    output_init_scale: float = 0.01


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
    """Single uninterrupted V9 transfer-learning run with frequent gates."""

    class_name = "rl_training.v9_stage1_runner:V9Stage1Runner"
    max_iterations = 30000
    save_interval = 250
    experiment_name = "m20_perceptive_v9_stage1_colleague9999"
    resume = False
    load_run = None
    load_checkpoint = None
    v9_base_checkpoint = (
        "/home/ubuntu/m20_perceptive_final/eval_outputs/v9_initialization_gate/"
        "colleague_model9999/model_9999.pt"
    )


@configclass
class DeeproboticsM20HRBridgePPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """Fresh dual-actor HR Bridge PPO. No pretrained policy is loaded."""

    class_name = "rl_training.hr_bridge_runner:HRBridgeRunner"
    max_iterations = 30000
    save_interval = 250
    experiment_name = "m20_stair_hr_bridge"
    resume = False
    load_run = None
    load_checkpoint = None
    hr_bridge_hidden_dims = [512, 256, 128]


@configclass
class DeeproboticsM20V10UnifiedPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """From-scratch V10 pilot; checkpoint initialization is prohibited."""

    class_name = "rl_training.v10_unified_runner:V10UnifiedRunner"
    max_iterations = 2000
    save_interval = 250
    experiment_name = "M20-V10-Unified-Perceptive"
    resume = False
    load_run = None
    load_checkpoint = None

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "V10UnifiedActorCritic"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.activation = "elu"
        self.policy.actor_obs_normalization = False
        self.policy.critic_obs_normalization = False


@configclass
class DeeproboticsM20V11ContinuousPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """From-scratch V11 configuration; long runs require the landing gate."""

    class_name = "rl_training.v11_continuous_runner:V11ContinuousRunner"
    max_iterations = 2000
    # Preserve denser V11 checkpoints so the landing-learning transition can
    # be inspected and rolled back at 100-iteration resolution.
    save_interval = 100
    experiment_name = "M20-V11-Continuous-Landing"
    resume = False
    load_run = None
    load_checkpoint = None

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "V11ContinuousActorCritic"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.policy.activation = "elu"
        self.policy.actor_obs_normalization = False
        self.policy.critic_obs_normalization = False


@configclass
class DeeproboticsM20V11FlatDescentPPORunnerCfg(DeeproboticsM20V11ContinuousPPORunnerCfg):
    """Independent from-scratch runner for the signed-edge V11-FD task."""

    max_iterations = 30000
    save_interval = 100
    experiment_name = "M20-V11-Flat-Descent"
    resume = False
    load_run = None
    load_checkpoint = None


@configclass
class DeeproboticsM20V111V11FDRearPosturePPORunnerCfg(
    DeeproboticsM20V11FlatDescentPPORunnerCfg
):
    """Saved V11-FD PPO with only rear-posture reward additions."""

    experiment_name = "M20-V11.1-V11FD-RearPosture"
    resume = False
    load_run = None
    load_checkpoint = None


@configclass
class DeeproboticsM20V11SoftAltPPORunnerCfg(DeeproboticsM20V11ContinuousPPORunnerCfg):
    """Short-horizon V11 fine-tune; runner refuses a random initialization."""

    num_steps_per_env = 48
    max_iterations = 1000
    save_interval = 100
    experiment_name = "M20-V11-SoftAlt"
    resume = False
    load_run = None
    load_checkpoint = None
    require_pretrained_checkpoint = True

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 1.0e-3


@configclass
class DeeproboticsM20V13V11FDAlternationPPORunnerCfg(DeeproboticsM20V11FlatDescentPPORunnerCfg):
    """V11-FD-identical PPO/network settings under an isolated V13 log root."""

    experiment_name = "M20-V13-V11FD-Alternation"


@configclass
class DeeproboticsM20V13Critic6DPPORunnerCfg(
    DeeproboticsM20V13V11FDAlternationPPORunnerCfg
):
    """V13-identical PPO with six additional critic-only lifecycle values."""

    experiment_name = "M20-V13-V11FD-Alternation-Critic6D"


@configclass
class DeeproboticsM20V13Critic12DPPORunnerCfg(
    DeeproboticsM20V13Critic6DPPORunnerCfg
):
    """V13 rollout cadence with ascent and descent critic-only context."""

    num_steps_per_env = 24
    max_iterations = 20000
    experiment_name = "M20-V13-V11FD-Alternation-Critic12D"


@configclass
class DeeproboticsM20V131V11FDOrderedAlternationPPORunnerCfg(
    DeeproboticsM20V13V11FDAlternationPPORunnerCfg
):
    """V13-identical PPO/network settings under an isolated V13.1 log root."""

    experiment_name = "M20-V13.1-V11FD-OrderedAlternation"


@configclass
class DeeproboticsM20V131RetentionPPORunnerCfg(
    DeeproboticsM20V131V11FDOrderedAlternationPPORunnerCfg
):
    """Short actor-only retention experiment with a fresh critic/optimizer."""

    class_name = "rl_training.v131_ordered_runner:V131OrderedRetentionRunner"
    max_iterations = 400
    save_interval = 100
    experiment_name = "M20-V13.1-V11FD-OrderedAlternation-Retention"
    require_v131_retention_actor = True


@configclass
class DeeproboticsM20V132V11FDBarrierAlternationPPORunnerCfg(
    DeeproboticsM20V13V11FDAlternationPPORunnerCfg
):
    """From-scratch V13.2 with independently normalised task/style returns."""

    class_name = "rl_training.v132_dual_critic_runner:V132DualCriticRunner"
    experiment_name = "M20-V13.2-V11FD-BarrierAlternation"
    resume = False
    load_run = None
    load_checkpoint = None
    v132_task_advantage_weight = 1.0
    v132_style_advantage_weight = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "V132BarrierActorCritic"


@configclass
class DeeproboticsM20V133AdjacentTransitionPPORunnerCfg(
    DeeproboticsM20V132V11FDBarrierAlternationPPORunnerCfg
):
    """Twenty-thousand-iteration V13.3 run with every network initialized fresh."""

    class_name = "rl_training.v133_transition_runner:V133TransitionRunner"
    experiment_name = "M20-V13.3-V11FD-FromScratch"
    num_steps_per_env = 32
    max_iterations = 20000
    save_interval = 250
    resume = False
    load_run = None
    load_checkpoint = None

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "V133FromScratchActorCritic"
        # Keep the eight-way batching from the prior V13.3 run while shortening
        # the feedback interval to 32 environment steps.
        self.algorithm.num_mini_batches = 8


@configclass
class DeeproboticsM20V134PerceptiveFollowPPORunnerCfg(
    DeeproboticsM20V13V11FDAlternationPPORunnerCfg
):
    """Fresh student with sparse V13 model_5000 ascent rollout segments."""

    class_name = "rl_training.v134_style_prior_runner:V134StylePriorRunner"
    experiment_name = "M20-V13.4-V11FD-PerceptiveFollow"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 250
    resume = False
    load_run = None
    load_checkpoint = None
    require_v134_style_actor = True
    # Start coherent teacher segments sparsely, then anneal their occupancy so
    # ordered-ascent reward must retain the gait in the fresh student.
    v134_teacher_burst_start_probability = 0.025
    v134_teacher_burst_end_probability = 0.005
    v134_teacher_burst_decay_updates = 6000
    v134_teacher_burst_steps = 8

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.class_name = "rl_training.v134_style_prior_runner:V134StylePriorPPO"
        # Keep the native V13 entropy pressure and reviewed learning rate. The
        # teacher affects rollout state occupancy, not an auxiliary actor loss.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 1.0e-3


@configclass
class V135GRUPolicyCfg(RslRlPpoActorCriticCfg):
    """Legacy bridge fields for a 63D GRU Actor and the native MLP Critic."""

    class_name: str = "V135RecurrentActorCritic"
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 128
    rnn_num_layers: int = 1


@configclass
class DeeproboticsM20V135NativeAlternationPPORunnerCfg(
    DeeproboticsM20V13V11FDAlternationPPORunnerCfg
):
    """V13.5-Final From-Scratch: V13-style MLP optimization backbone.

    Network: pure 63D feed-forward MLP Actor
    PPO: Restore V13 exploration strength to promote capability emergence
    Descent: New rear support transfer (not V11/V13.4)
    Flat: Persistent flat mode with crawl guard
    """

    experiment_name = "M20-V13.5-Final-FromScratch"
    num_steps_per_env = 24

    policy = RslRlPpoActorCriticCfg(
        class_name="V11ContinuousActorCritic",
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )

    # Restore V13 exploration strength for from-scratch training
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,  # Restored from 0.001 → 0.003 (V13 level)
        num_learning_epochs=5,  # Restored from 3 → 5 (V13 level)
        num_mini_batches=4,
        learning_rate=1.0e-3,  # Restored from 3e-4 → 1e-3 (V13 level)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  # Restored from 0.008 → 0.01 (V13 level)
        max_grad_norm=1.0,
    )

    # From-scratch: no checkpoint
    resume = False
    load_run = None
    load_checkpoint = None

@configclass
class DeeproboticsM20V12V9Teacher6DPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """Fresh V12 run with V9's MoE/PPO settings and V11's deployable Actor ABI."""

    class_name = "rl_training.v12_teacher_runner:V12Teacher6DRunner"
    max_iterations = 30000
    save_interval = 200
    experiment_name = "M20-V12-V9Teacher-6D"
    resume = False
    load_run = None
    load_checkpoint = None
    # Same rolling completed-episode window used by the V9 robust logger for
    # Train/reward_p10 and Train/reward_p90.
    episode_logging_buffer_size = 100

    def __post_init__(self):
        super().__post_init__()
        self.clip_actions = 5.0
        self.policy = V12TeacherMoEPolicyCfg(
            init_noise_std=0.5,
            noise_std_type="log",
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            num_moe_experts=4,
            moe_gate_hidden_dims=[128],
            output_init_scale=0.01,
        )
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 8
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.desired_kl = 0.006
