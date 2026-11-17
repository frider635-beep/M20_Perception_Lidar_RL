"""Fresh 63D V13.3 runner with signal-counted chain advantages."""

from __future__ import annotations

import os

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

from rl_training.v132_dual_critic import V132DualCriticPPO, v132_combine_advantages


def v133_scale_style_advantage(
    value: torch.Tensor,
    running_second_moment: torch.Tensor,
    signal_count: int | torch.Tensor,
    *,
    minimum_signal_count: int,
    decay: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Scale sparse style evidence without inflating one event to unit variance."""
    if int(minimum_signal_count) < 1:
        raise ValueError("V13.3 minimum style signal count must be positive.")
    if not 0.0 <= float(decay) < 1.0 or float(clip) <= 0.0:
        raise ValueError("V13.3 style RMS parameters are invalid.")
    count = int(torch.as_tensor(signal_count).item())
    if count <= 0:
        return torch.zeros_like(value), running_second_moment, 0.0
    centered = value - value.mean()
    batch_second = centered.square().mean().detach()
    updated = float(decay) * running_second_moment + (1.0 - float(decay)) * batch_second
    scale = min(1.0, count / float(minimum_signal_count))
    normalized = centered / updated.clamp_min(1.0e-8).sqrt()
    return normalized.clamp(-float(clip), float(clip)) * scale, updated, scale


class V133FromScratchPPO(V132DualCriticPPO):
    """Dual-return PPO whose style actor pressure tracks actual event support."""

    def __init__(
        self,
        *args,
        style_rms_decay: float = 0.99,
        style_advantage_clip: float = 2.0,
        style_minimum_signal_count: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.style_rms_decay = float(style_rms_decay)
        self.style_advantage_clip = float(style_advantage_clip)
        self.style_minimum_signal_count = int(style_minimum_signal_count)
        self._style_second_moment = torch.ones((), device=self.device)
        self._style_actor_scale = 0.0
        self._v133_rollout_counts = {
            name: torch.zeros((), device=self.device, dtype=torch.long)
            for name in (
                "seed", "adjacent", "chain", "repeat", "follower", "skip",
                "simultaneous", "chain_fl", "chain_fr",
            )
        }

    def process_env_step(self, obs, rewards, dones, extras):
        raw = self.env.unwrapped
        if not hasattr(raw, "_v133_chain"):
            raise RuntimeError("V13.3 rollout is missing its three-step chain ledger.")
        state = raw._v133_chain
        for name in ("seed", "adjacent", "chain", "repeat", "follower", "skip", "simultaneous"):
            self._v133_rollout_counts[name] += state[f"{name}_event"].long().sum()
        self._v133_rollout_counts["chain_fl"] += (
            state["chain_event"] & (state["last_side"] == 0)
        ).long().sum()
        self._v133_rollout_counts["chain_fr"] += (
            state["chain_event"] & (state["last_side"] == 1)
        ).long().sum()

        # V132's process method requires its barrier ledgers.  V13.3 keeps only
        # the independent return split and consumes the chain ledger above.
        index = self.storage.step
        if self._style_values is None:
            raise RuntimeError("V13.3 style critic was not evaluated before env.step().")
        style_reward = self._style_reward().to(self.device).reshape(-1)
        task_reward = rewards.reshape(-1) - style_reward
        self.storage.style_values[index].copy_(self._style_values.reshape(-1, 1))
        self.storage.style_rewards[index].copy_(style_reward.reshape(-1, 1))
        if "time_outs" in extras:
            timeouts = extras["time_outs"].to(self.device).reshape(-1, 1)
            self.storage.style_rewards[index] += self.gamma * self._style_values * timeouts
        self._rollout_task_reward_sum += task_reward.sum()
        self._rollout_style_reward_sum += style_reward.sum()
        self._rollout_style_nonzero_count += torch.count_nonzero(style_reward)
        self._rollout_sample_count += task_reward.numel()
        self.style_critic.update_normalization(obs)
        PPO.process_env_step(self, obs, task_reward, dones, extras)
        self.style_critic.reset(dones)
        self._style_values = None

    def compute_returns(self, obs):
        # Preserve the exact native PPO task-return path.  Recompute only the
        # independent style channel with its running RMS and evidence scale.
        PPO.compute_returns(self, obs)
        st = self.storage
        st.task_advantages.copy_(st.advantages)
        last_values = self.style_critic(obs).detach()
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.style_values[step + 1]
            not_terminal = 1.0 - st.dones[step].float()
            delta = st.style_rewards[step] + self.gamma * next_values * not_terminal - st.style_values[step]
            advantage = delta + self.gamma * self.lam * not_terminal * advantage
            st.style_returns[step] = advantage + st.style_values[step]
        style, second, scale = v133_scale_style_advantage(
            st.style_returns - st.style_values,
            self._style_second_moment,
            self._rollout_style_nonzero_count,
            minimum_signal_count=self.style_minimum_signal_count,
            decay=self.style_rms_decay,
            clip=self.style_advantage_clip,
        )
        self._style_second_moment.copy_(second)
        self._style_actor_scale = scale
        st.style_advantages.copy_(style)
        st.advantages.copy_(v132_combine_advantages(
            st.task_advantages, style,
            self.task_advantage_weight, self.style_advantage_weight,
        ))

    def update(self):
        counts = {name: int(value) for name, value in self._v133_rollout_counts.items()}
        style_scale = self._style_actor_scale
        style_rms = float(self._style_second_moment.sqrt())
        losses = super().update()
        for stale in (
            "ordered_completion_rate", "wrong_completion_rate",
            "simultaneous_event_per_encounter", "skip_per_encounter",
            "top_follow_success_rate", "deploy_reward_target_agreement",
            "style_target_gate_rate", "physical_completion_count",
            "target_comparable_count",
        ):
            losses.pop(stale, None)
        for name, value in counts.items():
            losses[f"v133_{name}_count"] = float(value)
        failures = counts["repeat"] + counts["skip"] + counts["simultaneous"]
        losses["v133_chain_precision"] = counts["chain"] / max(counts["chain"] + failures, 1)
        losses["v133_chain_fl_fraction"] = counts["chain_fl"] / max(counts["chain"], 1)
        losses["v133_style_actor_scale"] = style_scale
        losses["v133_style_advantage_rms"] = style_rms
        for value in self._v133_rollout_counts.values():
            value.zero_()
        return losses

    def save(self):
        saved = super().save()
        saved["v133_from_scratch"] = True
        saved["v133_style_second_moment"] = self._style_second_moment.detach().cpu()
        return saved


class V133TransitionRunner(OnPolicyRunner):
    """Construct a random native-63D actor, two critics and a fresh optimizer."""

    algorithm_class_name = "rl_training.v133_transition_runner:V133FromScratchPPO"

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        if os.environ.get("V133_MOTOR_PRIOR_CHECKPOINT"):
            raise RuntimeError("V13.3 from-scratch forbids V133_MOTOR_PRIOR_CHECKPOINT.")
        cfg = dict(train_cfg)
        task_weight = float(cfg.pop("v132_task_advantage_weight", 1.0))
        style_weight = float(cfg.pop("v132_style_advantage_weight", 1.0))
        cfg["algorithm"] = dict(cfg["algorithm"])
        cfg["algorithm"].pop("beta", None)
        cfg["algorithm"]["class_name"] = self.algorithm_class_name
        cfg["algorithm"]["style_reward_term"] = "v133_style"
        cfg["algorithm"]["task_advantage_weight"] = task_weight
        cfg["algorithm"]["style_advantage_weight"] = style_weight
        cfg["algorithm"]["style_rms_decay"] = 0.99
        cfg["algorithm"]["style_advantage_clip"] = 2.0
        cfg["algorithm"]["style_minimum_signal_count"] = 8

        policy = dict(cfg.pop("policy"))
        if policy.pop("class_name", None) != "V133FromScratchActorCritic":
            raise RuntimeError("V13.3 runner received the wrong policy configuration.")
        init_std = policy.pop("init_noise_std")
        std_type = policy.pop("noise_std_type")
        if policy.pop("state_dependent_std", False):
            raise RuntimeError("V13.3 requires state-independent action standard deviation.")
        actor_normalization = policy.pop("actor_obs_normalization")
        critic_normalization = policy.pop("critic_obs_normalization")
        actor_hidden = policy.pop("actor_hidden_dims")
        critic_hidden = policy.pop("critic_hidden_dims")
        activation = policy.pop("activation")
        if policy.pop("include_depth_input", False):
            raise RuntimeError("V13.3 does not permit a hidden depth actor input.")
        if policy:
            raise RuntimeError(f"V13.3 runner received unsupported policy fields: {sorted(policy)}")

        cfg["obs_groups"] = {"actor": ["policy"], "critic": ["critic"]}
        cfg["actor"] = {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": actor_hidden,
            "activation": activation,
            "obs_normalization": actor_normalization,
            "distribution_cfg": {
                "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                "init_std": init_std,
                "std_type": std_type,
            },
        }
        cfg["critic"] = {
            "class_name": "rsl_rl.models:MLPModel",
            "hidden_dims": critic_hidden,
            "activation": activation,
            "obs_normalization": critic_normalization,
        }
        super().__init__(env, cfg, log_dir=log_dir, device=device)

        if not isinstance(self.alg, V133FromScratchPPO):
            raise RuntimeError("V13.3 runner did not construct its from-scratch PPO.")
        actor = self.alg.actor
        action_dim = actor.distribution.output_dim if actor.distribution is not None else -1
        if (actor.obs_dim, action_dim) != (63, 16):
            raise RuntimeError(f"V13.3 actor ABI must be 63D->16D, got {actor.obs_dim}D->{action_dim}D.")
        if self.alg.optimizer.state:
            raise RuntimeError("V13.3 from-scratch optimizer must have empty state.")
        if self.current_learning_iteration != 0:
            raise RuntimeError("V13.3 from-scratch iteration must begin at zero.")
        self._write_actor_manifest(log_dir)
        print("V133_FROM_SCRATCH=YES", flush=True)
        print("V133_NO_CHECKPOINT_LOADED=YES", flush=True)
        print("V133_FRESH_ACTOR=YES", flush=True)
        print("V133_FRESH_CRITICS=YES", flush=True)
        print("V133_FRESH_OPTIMIZER=YES", flush=True)

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
        del path, load_cfg, strict, map_location
        raise RuntimeError("V13.3 from-scratch runner refuses checkpoint loading.")

    @staticmethod
    def _write_actor_manifest(log_dir: str | None) -> None:
        if log_dir is None:
            return
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "actor_observation_manifest.txt"), "w", encoding="utf-8") as stream:
            stream.write(
                "M20 V13.3 actor observation (63D)\n"
                "0:63 exact native V13 observation ABI\n"
                "No expected-side, chain, target identity or phase state enters the actor.\n"
                "initialization=random actor, random task critic, random style critic, fresh optimizer\n"
            )


__all__ = [
    "V133FromScratchPPO", "V133TransitionRunner", "v133_scale_style_advantage",
]
