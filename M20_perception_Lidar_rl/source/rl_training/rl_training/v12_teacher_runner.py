"""RSL-RL 5.x V9-MoE bridge and ABI gate for the deployable V12 63D Actor."""

from __future__ import annotations

import os
from collections import deque

import torch
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

from .v12_teacher_moe import MoeLayer


class V12RobustLogger(Logger):
    """V9 teacher return-distribution summaries for V12 TensorBoard logs."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        buffer_size = max(int(self.cfg.get("episode_logging_buffer_size", 100)), 100)
        if self.rewbuffer.maxlen != buffer_size:
            self.rewbuffer = deque(maxlen=buffer_size)
            self.lenbuffer = deque(maxlen=buffer_size)
            if hasattr(self, "erewbuffer"):
                self.erewbuffer = deque(maxlen=buffer_size)
                self.irewbuffer = deque(maxlen=buffer_size)
        self._rollout_reward_sum = 0.0
        self._rollout_reward_steps = 0

    def process_env_step(self, rewards, dones, extras, intrinsic_rewards=None) -> None:
        super().process_env_step(rewards, dones, extras, intrinsic_rewards)
        self._rollout_reward_sum += rewards.detach().float().mean().item()
        self._rollout_reward_steps += 1

    def log(self, *args, **kwargs) -> None:
        iteration = kwargs.get("it", args[0] if args else 0)
        super().log(*args, **kwargs)
        if self.writer is not None:
            if self._rollout_reward_steps:
                self.writer.add_scalar(
                    "Train/mean_reward_per_step_rollout",
                    self._rollout_reward_sum / self._rollout_reward_steps,
                    iteration,
                )
            if self.rewbuffer:
                returns = torch.tensor(tuple(self.rewbuffer), dtype=torch.float32)
                self.writer.add_scalar("Train/median_reward", torch.quantile(returns, 0.50).item(), iteration)
                self.writer.add_scalar("Train/reward_p10", torch.quantile(returns, 0.10).item(), iteration)
                self.writer.add_scalar("Train/reward_p90", torch.quantile(returns, 0.90).item(), iteration)
        self._rollout_reward_sum = 0.0
        self._rollout_reward_steps = 0


class V12Teacher6DRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        # The installed OnPolicyRunner resolves ``Logger`` from its own module
        # at construction time.  Scope the V9 robust logger replacement to
        # this V12 process before invoking its parent initializer.
        rsl_on_policy_runner.Logger = V12RobustLogger
        cfg = dict(train_cfg)
        cfg["algorithm"] = dict(cfg["algorithm"])
        cfg["algorithm"].pop("beta", None)
        policy = dict(cfg.pop("policy"))
        if policy.pop("class_name", None) != "V12Teacher6DMoEActorCritic":
            raise RuntimeError("V12 runner received the wrong policy configuration")
        init_std = policy.pop("init_noise_std")
        std_type = policy.pop("noise_std_type")
        actor_norm = policy.pop("actor_obs_normalization")
        critic_norm = policy.pop("critic_obs_normalization")
        actor_hidden = policy.pop("actor_hidden_dims")
        critic_hidden = policy.pop("critic_hidden_dims")
        activation = policy.pop("activation")
        num_experts = policy.pop("num_moe_experts")
        gate_hidden = policy.pop("moe_gate_hidden_dims")
        output_init_scale = policy.pop("output_init_scale")
        if policy.pop("include_depth_input", False):
            raise RuntimeError("V12 Actor must not receive a hidden height-map input")
        if policy:
            raise RuntimeError(f"V12 runner received unsupported policy fields: {sorted(policy)}")
        cfg["obs_groups"] = {"actor": ["policy"], "critic": ["critic"]}
        cfg["actor"] = {
            "class_name": "rl_training.v12_teacher_moe:V12TeacherMoEModel",
            "hidden_dims": actor_hidden,
            "activation": activation,
            "obs_normalization": actor_norm,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": init_std,
                "std_type": std_type,
            },
            "num_moe_experts": num_experts,
            "moe_gate_hidden_dims": gate_hidden,
            "output_init_scale": output_init_scale,
        }
        cfg["critic"] = {
            "class_name": "rl_training.v12_teacher_moe:V12TeacherMoEModel",
            "hidden_dims": critic_hidden,
            "activation": activation,
            "obs_normalization": critic_norm,
            "num_moe_experts": num_experts,
            "moe_gate_hidden_dims": gate_hidden,
        }
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        if not isinstance(self.logger, V12RobustLogger):
            raise RuntimeError("V12 robust p10/p90 logger was not installed")
        actor = self.alg.actor
        critic = self.alg.critic
        action_dim = actor.distribution.output_dim if actor.distribution is not None else -1
        if (actor.obs_dim, action_dim) != (63, 16):
            raise RuntimeError(f"V12 Actor ABI gate failed: got {(actor.obs_dim, action_dim)}, expected (63, 16)")
        for name, model in (("Actor", actor), ("Critic", critic)):
            if not isinstance(model.mlp, MoeLayer) or model.mlp.num_experts != 4:
                raise RuntimeError(f"V12 {name} must use the V9 four-expert MoE architecture")
        self._write_manifest(log_dir)
        print("V12_ACTOR_OBS_DIM=63")
        print("V12_ACTION_DIM=16")
        print("V12_ACTOR_CRITIC_ARCH=4_EXPERT_SOFT_MOE")
        print(f"V12_REWARD_QUANTILES=Train/reward_p10,Train/reward_p90 window={self.logger.rewbuffer.maxlen}")

    @staticmethod
    def _write_manifest(log_dir: str | None) -> None:
        if log_dir is None:
            return
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "actor_observation_manifest.txt"), "w", encoding="ascii") as stream:
            stream.write("M20 V12 actor observation (63D)\\n")
            stream.write("0:3 base_ang_vel; 3:6 projected_gravity; 6:9 command; 9:25 joint_pos; 25:41 joint_vel; 41:57 previous_action; 57 B1; 58 H1; 59 D1; 60 B2; 61 H2; 62 D2\\n")
            stream.write("Actor/Critic architecture: V9-compatible four-expert soft MoE, gate hidden dims 128.\\n")
            stream.write("PGTT height map is training-only Critic/reward information.\\n")


__all__ = ["V12Teacher6DRunner", "V12RobustLogger"]
