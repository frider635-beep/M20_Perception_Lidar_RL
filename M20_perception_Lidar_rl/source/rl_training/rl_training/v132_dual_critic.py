"""PPO with independently normalised V11-task and V13.2-style returns."""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_optimizer


def v132_normalize_advantage(value: torch.Tensor) -> torch.Tensor:
    """Standardise one return channel without coupling its reward scale to another."""
    return (value - value.mean()) / (value.std(unbiased=False) + 1.0e-8)


def v132_combine_advantages(
    task: torch.Tensor, style: torch.Tensor, task_weight: float, style_weight: float,
) -> torch.Tensor:
    """Add style guidance without attenuating the native task advantage.

    In particular, weights ``1/1`` mean ``task + style`` rather than their
    average.  Therefore a rollout with no style signal is exactly the original
    V13 actor update, preserving the strong velocity-driven emergence path.
    """
    if abs(float(task_weight)) + abs(float(style_weight)) <= 0.0:
        raise ValueError("V13.2 needs at least one non-zero advantage weight.")
    return float(task_weight) * task + float(style_weight) * style


def v132_normalize_style_advantage(
    value: torch.Tensor, style_nonzero_count: int | torch.Tensor,
) -> torch.Tensor:
    """Normalize style only after the rollout contains real style cashflow.

    With an all-zero style reward, a randomly initialized value baseline still
    has non-zero temporal residuals.  Standardizing those residuals would turn
    critic initialization noise into a full-strength actor objective.
    """
    normalized = v132_normalize_advantage(value)
    active = torch.as_tensor(style_nonzero_count, device=value.device) > 0
    return torch.where(active, normalized, torch.zeros_like(normalized))


def v132_zero_scalar_output(model) -> None:
    """Start the auxiliary value baseline at exactly zero, preserving its body."""
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if not linear_layers or linear_layers[-1].out_features != 1:
        raise RuntimeError("V13.2 style critic must end in one scalar Linear layer.")
    nn.init.zeros_(linear_layers[-1].weight)
    if linear_layers[-1].bias is not None:
        nn.init.zeros_(linear_layers[-1].bias)


class V132DualCriticPPO(PPO):
    """One actor, one native-task critic and one barrier-style critic."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        *,
        style_reward_term: str = "v132_style",
        task_advantage_weight: float = 1.0,
        style_advantage_weight: float = 1.0,
        optimizer: str = "adam",
        **kwargs,
    ):
        super().__init__(actor, critic, storage, optimizer=optimizer, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise RuntimeError("V13.2 dual-return PPO currently forbids RND and symmetry augmentation.")
        if self.normalize_advantage_per_mini_batch:
            raise RuntimeError("V13.2 requires rollout-level advantage normalization.")
        import copy
        # Deepcopy retains the exact critic observation ABI while creating
        # independent parameters and normalisation statistics.
        self.style_critic = copy.deepcopy(critic).to(self.device)
        v132_zero_scalar_output(self.style_critic)
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters(), self.style_critic.parameters()),
            lr=self.learning_rate,
        )
        self.style_reward_term = str(style_reward_term)
        self.task_advantage_weight = float(task_advantage_weight)
        self.style_advantage_weight = float(style_advantage_weight)
        shape = self.storage.rewards.shape
        self.storage.style_rewards = torch.zeros(shape, device=self.device)
        self.storage.style_values = torch.zeros(shape, device=self.device)
        self.storage.style_returns = torch.zeros(shape, device=self.device)
        self.storage.style_advantages = torch.zeros(shape, device=self.device)
        self.storage.task_advantages = torch.zeros(shape, device=self.device)
        self._style_values = None
        self._rollout_task_reward_sum = torch.zeros((), device=self.device)
        self._rollout_style_reward_sum = torch.zeros((), device=self.device)
        self._rollout_style_nonzero_count = torch.zeros((), device=self.device, dtype=torch.long)
        self._rollout_event_counts = {
            name: torch.zeros((), device=self.device, dtype=torch.long)
            for name in (
                "encounter", "physical", "ordered", "wrong", "simultaneous", "skip",
                "top_entry", "top_success", "target_comparable", "target_agree",
                "reward_target_samples", "style_target_aligned",
            )
        }
        self._rollout_sample_count = 0
        self.env = None

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str):
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        if not isinstance(alg, V132DualCriticPPO):
            raise RuntimeError("V13.2 runner did not construct V132DualCriticPPO.")
        alg.env = env
        return alg

    def act(self, obs):
        actions = super().act(obs)
        self._style_values = self.style_critic(obs).detach()
        return actions

    def _style_reward(self) -> torch.Tensor:
        if self.env is None:
            raise RuntimeError("V13.2 PPO has no environment bound to its style channel.")
        raw = self.env.unwrapped
        reward_manager = raw.reward_manager
        if self.style_reward_term not in reward_manager.active_terms:
            raise RuntimeError(f"V13.2 style term is not active: {self.style_reward_term!r}")
        if not hasattr(reward_manager, "_step_reward"):
            raise RuntimeError("V13.2 requires RewardManager's exact per-term step buffer.")
        index = reward_manager.active_terms.index(self.style_reward_term)
        weighted_rate = reward_manager._step_reward[:, index]
        if weighted_rate.shape != (raw.num_envs,):
            raise RuntimeError(
                f"V13.2 style step buffer has shape {tuple(weighted_rate.shape)}, "
                f"expected {(raw.num_envs,)}."
            )
        # RewardManager stores value * configured weight (the integrated term
        # divided by dt). Reading it avoids re-evaluating the term after some
        # terminal environments have already reset.
        return weighted_rate.detach().clone() * float(raw.step_dt)

    def process_env_step(self, obs, rewards, dones, extras):
        index = self.storage.step
        if self._style_values is None:
            raise RuntimeError("V13.2 style critic was not evaluated before env.step().")
        raw = self.env.unwrapped
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
        if not hasattr(raw, "_v132_alternation") or not hasattr(raw, "_v132_deploy"):
            raise RuntimeError("V13.2 rollout is missing its reward/deployment state ledgers.")
        reward_state = raw._v132_alternation
        deploy_state = raw._v132_deploy
        event_map = {
            "encounter": reward_state["encounter_event"],
            "physical": reward_state["physical_event"],
            "ordered": reward_state["ordered_event"],
            "wrong": reward_state["wrong_event"],
            "simultaneous": reward_state["simultaneous_event"],
            "skip": reward_state["skip_event"],
            "top_entry": reward_state["top_entry_event"],
            "top_success": reward_state["top_follow_event"],
        }
        for name, event in event_map.items():
            self._rollout_event_counts[name] += event.long().sum()
        comparable = (
            reward_state["target_active"] & deploy_state["settlement_target_valid"]
            & ~reward_state["physical_event"] & ~dones.bool()
        )
        agree = comparable & (
            reward_state["target_index"] == deploy_state["settlement_target_index"]
        )
        self._rollout_event_counts["target_comparable"] += comparable.long().sum()
        self._rollout_event_counts["target_agree"] += agree.long().sum()
        reward_target = reward_state["target_active"] & ~dones.bool()
        self._rollout_event_counts["reward_target_samples"] += reward_target.long().sum()
        self._rollout_event_counts["style_target_aligned"] += (
            reward_target & reward_state["deploy_target_agreement"]
        ).long().sum()
        self.style_critic.update_normalization(obs)
        super().process_env_step(obs, task_reward, dones, extras)
        self.style_critic.reset(dones)
        self._style_values = None

    def compute_returns(self, obs):
        # Native V11/V13 task return is computed by the unmodified PPO path.
        super().compute_returns(obs)
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
        # super().compute_returns() has already normalized the task advantage
        # through the exact native PPO path. Do not normalize it a second time
        # with subtly different variance semantics.
        task_normalized = st.task_advantages
        style_normalized = v132_normalize_style_advantage(
            st.style_returns - st.style_values, self._rollout_style_nonzero_count,
        )
        st.style_advantages.copy_(style_normalized)
        st.advantages.copy_(v132_combine_advantages(
            task_normalized, style_normalized,
            self.task_advantage_weight, self.style_advantage_weight,
        ))

    @staticmethod
    def _clipped_value_loss(values, old_values, returns, clip_param, enabled):
        if not enabled:
            return (returns - values).square().mean()
        clipped = old_values + (values - old_values).clamp(-clip_param, clip_param)
        return torch.maximum((values - returns).square(), (clipped - returns).square()).mean()

    def update(self):
        st = self.storage
        total = st.num_envs * st.num_transitions_per_env
        mini_batch = total // self.num_mini_batches
        if mini_batch < 1:
            raise RuntimeError("V13.2 PPO mini-batch is empty.")
        obs = st.observations.flatten(0, 1)
        actions = st.actions.flatten(0, 1)
        old_values = st.values.flatten(0, 1)
        old_style_values = st.style_values.flatten(0, 1)
        returns = st.returns.flatten(0, 1)
        style_returns = st.style_returns.flatten(0, 1)
        advantages = st.advantages.flatten(0, 1)
        old_log_prob = st.actions_log_prob.flatten(0, 1)
        old_distribution = tuple(value.flatten(0, 1) for value in st.distribution_params)
        losses = {"surrogate": 0.0, "task_value": 0.0, "style_value": 0.0, "entropy": 0.0}
        updates = 0

        for _ in range(self.num_learning_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, self.num_mini_batches * mini_batch, mini_batch):
                ids = permutation[start:start + mini_batch]
                batch_obs = obs[ids]
                self.actor(batch_obs, stochastic_output=True)
                log_prob = self.actor.get_output_log_prob(actions[ids])
                task_values = self.critic(batch_obs)
                style_values = self.style_critic(batch_obs)
                distribution = tuple(value for value in self.actor.output_distribution_params)
                entropy = self.actor.output_entropy

                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        previous = tuple(value[ids] for value in old_distribution)
                        kl_mean = self.actor.get_kl_divergence(previous, distribution).mean()
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size
                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                                self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                        if self.is_multi_gpu:
                            value = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(value, src=0)
                            self.learning_rate = float(value)
                        for group in self.optimizer.param_groups:
                            group["lr"] = self.learning_rate

                ratio = torch.exp(log_prob - old_log_prob[ids].squeeze(-1))
                advantage = advantages[ids].squeeze(-1)
                surrogate = -advantage * ratio
                clipped = -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.maximum(surrogate, clipped).mean()
                task_value_loss = self._clipped_value_loss(
                    task_values, old_values[ids], returns[ids], self.clip_param, self.use_clipped_value_loss,
                )
                style_value_loss = self._clipped_value_loss(
                    style_values, old_style_values[ids], style_returns[ids], self.clip_param, self.use_clipped_value_loss,
                )
                # Each critic keeps the native full value-loss coefficient.
                # Averaging here would silently halve V13's task-critic update.
                value_loss = task_value_loss + style_value_loss
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                # Independent clipping prevents a large auxiliary-critic
                # gradient from shrinking the native actor/task-critic update.
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.style_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses["surrogate"] += float(surrogate_loss.detach())
                losses["task_value"] += float(task_value_loss.detach())
                losses["style_value"] += float(style_value_loss.detach())
                losses["entropy"] += float(entropy.mean().detach())
                updates += 1

        samples = max(self._rollout_sample_count, 1)
        losses["task_reward_per_step"] = float(self._rollout_task_reward_sum / samples)
        losses["style_reward_per_step"] = float(self._rollout_style_reward_sum / samples)
        losses["style_signal_fraction"] = float(self._rollout_style_nonzero_count / samples)
        counts = self._rollout_event_counts
        losses["ordered_completion_rate"] = float(
            counts["ordered"] / counts["physical"].clamp_min(1)
        )
        losses["wrong_completion_rate"] = float(
            counts["wrong"] / counts["physical"].clamp_min(1)
        )
        losses["simultaneous_event_per_encounter"] = float(
            counts["simultaneous"] / counts["encounter"].clamp_min(1)
        )
        losses["skip_per_encounter"] = float(counts["skip"] / counts["encounter"].clamp_min(1))
        losses["top_follow_success_rate"] = float(
            counts["top_success"] / counts["top_entry"].clamp_min(1)
        )
        losses["deploy_reward_target_agreement"] = float(
            counts["target_agree"] / counts["target_comparable"].clamp_min(1)
        )
        losses["style_target_gate_rate"] = float(
            counts["style_target_aligned"] / counts["reward_target_samples"].clamp_min(1)
        )
        losses["physical_completion_count"] = float(counts["physical"])
        losses["target_comparable_count"] = float(counts["target_comparable"])
        for key in ("surrogate", "task_value", "style_value", "entropy"):
            losses[key] /= max(updates, 1)
        st.clear()
        self._rollout_task_reward_sum.zero_()
        self._rollout_style_reward_sum.zero_()
        self._rollout_style_nonzero_count.zero_()
        for count in self._rollout_event_counts.values():
            count.zero_()
        self._rollout_sample_count = 0
        return losses

    def train_mode(self):
        super().train_mode()
        self.style_critic.train()

    def eval_mode(self):
        super().eval_mode()
        self.style_critic.eval()

    def save(self):
        saved = super().save()
        saved["style_critic_state_dict"] = self.style_critic.state_dict()
        saved["v132_dual_return"] = True
        return saved

    def load(self, loaded_dict, load_cfg, strict):
        if "style_critic_state_dict" not in loaded_dict:
            raise RuntimeError("V13.2 refuses a checkpoint without the independent style critic.")
        iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("critic", True):
            self.style_critic.load_state_dict(loaded_dict["style_critic_state_dict"], strict=strict)
        return iteration

    def broadcast_parameters(self):
        super().broadcast_parameters()
        values = [self.style_critic.state_dict()]
        torch.distributed.broadcast_object_list(values, src=0)
        self.style_critic.load_state_dict(values[0])

    def reduce_parameters(self):
        parameters = list(chain(
            self.actor.parameters(), self.critic.parameters(), self.style_critic.parameters(),
        ))
        grads = [parameter.grad.reshape(-1) for parameter in parameters if parameter.grad is not None]
        flattened = torch.cat(grads)
        torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
        flattened /= self.gpu_world_size
        offset = 0
        for parameter in parameters:
            if parameter.grad is not None:
                size = parameter.numel()
                parameter.grad.data.copy_(flattened[offset:offset + size].view_as(parameter.grad))
                offset += size


__all__ = [
    "V132DualCriticPPO", "v132_normalize_advantage", "v132_normalize_style_advantage",
    "v132_combine_advantages", "v132_zero_scalar_output",
]
