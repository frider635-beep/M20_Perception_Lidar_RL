"""Terrain-separated V13.4 PPO with ascent-priority gradient protection."""

from __future__ import annotations

import copy
from itertools import chain

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_optimizer

from rl_training.v132_dual_critic import v132_zero_scalar_output
from rl_training.v133_transition_runner import V133FromScratchPPO, V133TransitionRunner
from rl_training.v134_diagnostics import (
    v134_combine_terrain_advantages,
    v134_masked_mean,
    v134_masked_normalize,
    v134_project_gradient_away,
    v134_priority_gradient_gain,
    v134_redistribute_chain_settlement,
    v134_terrain_normalize,
)


class V134DiagnosticPPO(V133FromScratchPPO):
    """PPO with native, ascent-style and descent-posture value channels."""

    _BEHAVIORS = (
        "seed", "ordered", "chain", "ordinary", "correct_fl", "correct_fr",
        "wrong_jump", "repeat", "simultaneous", "expected_progress",
        "actionable_target",
    )

    def __init__(
        self,
        *args,
        ascent_reward_terms,
        descent_reward_terms,
        ascent_gamma=0.997,
        ascent_lam=0.97,
        ascent_minimum_positive_signals=8,
        descent_minimum_positive_signals=32,
        descent_enable_after_ascent_positive=64,
        ascent_advantage_weight=1.0,
        ascent_advantage_weight_max=4.0,
        ascent_lagrange_rate=0.05,
        ascent_precision_target=0.50,
        descent_advantage_weight=0.35,
        chain_settlement=4.0,
        ascent_gradient_minimum_ratio=1.5,
        ascent_gradient_gain_max=6.0,
        optimizer="adam",
        **kwargs,
    ):
        super().__init__(*args, optimizer=optimizer, **kwargs)
        if self.is_multi_gpu:
            raise RuntimeError(
                "V13.4 terrain-separated PPO currently supports one GPU only; "
                "cross-rank gradient reduction is not implemented."
            )
        self.ascent_reward_terms = tuple(ascent_reward_terms)
        self.descent_reward_terms = tuple(descent_reward_terms)
        self.ascent_gamma = float(ascent_gamma)
        self.ascent_lam = float(ascent_lam)
        self.ascent_minimum_positive_signals = int(ascent_minimum_positive_signals)
        self.descent_minimum_positive_signals = int(descent_minimum_positive_signals)
        self.descent_enable_after_ascent_positive = int(descent_enable_after_ascent_positive)
        self.ascent_advantage_weight = float(ascent_advantage_weight)
        self.ascent_advantage_weight_max = float(ascent_advantage_weight_max)
        self.ascent_lagrange_rate = float(ascent_lagrange_rate)
        self.ascent_precision_target = float(ascent_precision_target)
        self.descent_advantage_weight = float(descent_advantage_weight)
        self.chain_settlement = float(chain_settlement)
        self.ascent_gradient_minimum_ratio = float(ascent_gradient_minimum_ratio)
        self.ascent_gradient_gain_max = float(ascent_gradient_gain_max)
        if not self.ascent_reward_terms or not self.descent_reward_terms:
            raise ValueError("V13.4 requires non-empty ascent and descent reward channels.")
        if not (0.0 < self.ascent_gamma <= 1.0 and 0.0 < self.ascent_lam <= 1.0):
            raise ValueError("V13.4 ascent GAE parameters must lie in (0, 1].")
        if min(
            self.ascent_minimum_positive_signals,
            self.descent_minimum_positive_signals,
            self.descent_enable_after_ascent_positive,
        ) < 1:
            raise ValueError("V13.4 signal gates must be positive.")
        if not 0.0 < self.ascent_advantage_weight <= self.ascent_advantage_weight_max:
            raise ValueError("V13.4 ascent advantage bounds are invalid.")
        if not 0.0 <= self.descent_advantage_weight <= 1.0:
            raise ValueError("V13.4 descent advantage weight must lie in [0, 1].")
        if self.ascent_gradient_minimum_ratio <= 0.0:
            raise ValueError("V13.4 gradient-strength ratio must be positive.")
        if self.ascent_gradient_gain_max < self.ascent_advantage_weight_max:
            raise ValueError("V13.4 gradient gain cap cannot be below the ascent weight cap.")

        self.descent_critic = copy.deepcopy(self.critic).to(self.device)
        v132_zero_scalar_output(self.descent_critic)
        self.optimizer = resolve_optimizer(optimizer)(
            chain(
                self.actor.parameters(), self.critic.parameters(),
                self.style_critic.parameters(), self.descent_critic.parameters(),
            ),
            lr=self.learning_rate,
        )
        shape = self.storage.rewards.shape
        self.storage.descent_rewards = torch.zeros(shape, device=self.device)
        self.storage.descent_values = torch.zeros(shape, device=self.device)
        self.storage.descent_returns = torch.zeros(shape, device=self.device)
        self.storage.descent_advantages = torch.zeros(shape, device=self.device)
        self.storage.ascent_masks = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.storage.descent_masks = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self._descent_values = None
        self._pending_ascent_mask = None
        self._pending_descent_mask = None
        self._pending_pre_b1_actionable = None
        self._rollout_ascent_positive_count = torch.zeros((), device=self.device, dtype=torch.long)
        self._rollout_descent_positive_count = torch.zeros((), device=self.device, dtype=torch.long)
        self._rollout_descent_reward_sum = torch.zeros((), device=self.device)
        self._ascent_positive_total = 0
        self._descent_actor_enabled = False
        self._ascent_actor_scale = 0.0
        self._descent_actor_scale = 0.0
        self._chain_redistributed = 0
        self._chain_success_total = 0
        self._rollout_pre_b1_actionable = torch.zeros((), device=self.device, dtype=torch.long)
        self._rollout_pre_b1_backward = torch.zeros((), device=self.device, dtype=torch.long)
        self._v134_masks = {
            name: torch.zeros(shape, device=self.device, dtype=torch.bool)
            for name in self._BEHAVIORS
        }
        self._v134_diagnostics = {}

    def act(self, obs):
        actions = super().act(obs)
        self._descent_values = self.descent_critic(obs).detach()
        raw = self.env.unwrapped
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.v133_adjacent_transition import (
            v133_ascending_stair_mask,
            v133_descending_stair_mask,
        )
        self._pending_ascent_mask = v133_ascending_stair_mask(raw).detach().clone()
        self._pending_descent_mask = v133_descending_stair_mask(raw).detach().clone()
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.v11_continuous import (
            _v11_front_mobility_gate,
        )
        chain_state = getattr(raw, "_v133_chain", None)
        if chain_state is None:
            raise RuntimeError("V13.4 requires chain state before action sampling.")
        # This is an observation-only audit signal. It measures whether the
        # native V13 policy retreats when a front leg has an actionable B1
        # target but no first support has yet seeded the ordered chain.
        self._pending_pre_b1_actionable = (
            _v11_front_mobility_gate(raw).any(dim=1)
            & self._pending_ascent_mask
            & ~chain_state["chain_valid"]
        ).detach().clone()
        return actions

    def _reward_channel(self, terms, mask):
        raw = self.env.unwrapped
        manager = raw.reward_manager
        if not hasattr(manager, "_step_reward"):
            raise RuntimeError("V13.4 requires RewardManager's per-term step buffer.")
        missing = [name for name in terms if name not in manager.active_terms]
        if missing:
            raise RuntimeError(f"V13.4 reward channel has inactive terms: {missing}")
        indices = [manager.active_terms.index(name) for name in terms]
        weighted_rate = manager._step_reward[:, indices].sum(dim=1)
        return weighted_rate.detach().clone() * float(raw.step_dt) * mask.float()

    def process_env_step(self, obs, rewards, dones, extras):
        if self._style_values is None or self._descent_values is None:
            raise RuntimeError("V13.4 critics were not evaluated before env.step().")
        if self._pending_ascent_mask is None or self._pending_descent_mask is None:
            raise RuntimeError("V13.4 terrain masks were not captured before env.step().")
        if self._pending_pre_b1_actionable is None:
            raise RuntimeError("V13.4 pre-B1 retreat mask was not captured before env.step().")
        raw = self.env.unwrapped
        if not hasattr(raw, "_v133_chain"):
            raise RuntimeError("V13.4 requires the V13.3 chain ledger.")
        state = raw._v133_chain
        index = self.storage.step
        ascent_mask = self._pending_ascent_mask
        descent_mask = self._pending_descent_mask
        if bool((ascent_mask & descent_mask).any()):
            raise RuntimeError("V13.4 terrain channel masks overlap.")
        pre_b1 = self._pending_pre_b1_actionable
        backward = raw.scene["robot"].data.root_lin_vel_b[:, 0] < -0.05
        self._rollout_pre_b1_actionable += pre_b1.long().sum()
        self._rollout_pre_b1_backward += (pre_b1 & backward).long().sum()

        adjacent = state["adjacent_event"]
        repeat = state["repeat_event"]
        follower = state["follower_event"]
        skip = state["skip_event"]
        simultaneous = state["simultaneous_event"]
        ordinary = repeat | follower | skip | simultaneous
        masks = {
            "seed": state["seed_event"],
            "ordered": adjacent,
            "chain": state["chain_event"],
            "ordinary": ordinary,
            "correct_fl": adjacent & (state["last_side"] == 0),
            "correct_fr": adjacent & (state["last_side"] == 1),
            "wrong_jump": skip,
            "repeat": repeat,
            "simultaneous": simultaneous,
            "expected_progress": (state["adjacent_progress_delta"] > 0.0) & ~adjacent,
            "actionable_target": state["adjacent_progress_active"],
        }
        for name, mask in masks.items():
            self._v134_masks[name][index, :, 0].copy_(mask & ascent_mask)
        for name in ("seed", "adjacent", "chain", "repeat", "follower", "skip", "simultaneous"):
            self._v133_rollout_counts[name] += state[f"{name}_event"].long().sum()
        self._v133_rollout_counts["chain_fl"] += (
            state["chain_event"] & (state["last_side"] == 0)
        ).long().sum()
        self._v133_rollout_counts["chain_fr"] += (
            state["chain_event"] & (state["last_side"] == 1)
        ).long().sum()

        ascent_reward = self._reward_channel(self.ascent_reward_terms, ascent_mask)
        descent_reward = self._reward_channel(self.descent_reward_terms, descent_mask)
        native_reward = rewards.reshape(-1) - ascent_reward - descent_reward
        self.storage.ascent_masks[index, :, 0].copy_(ascent_mask)
        self.storage.descent_masks[index, :, 0].copy_(descent_mask)
        self.storage.style_values[index].copy_(self._style_values.reshape(-1, 1))
        self.storage.style_rewards[index].copy_(ascent_reward.reshape(-1, 1))
        self.storage.descent_values[index].copy_(self._descent_values.reshape(-1, 1))
        self.storage.descent_rewards[index].copy_(descent_reward.reshape(-1, 1))
        if "time_outs" in extras:
            timeouts = extras["time_outs"].to(self.device).reshape(-1, 1)
            self.storage.style_rewards[index] += self.ascent_gamma * self._style_values * timeouts
            self.storage.descent_rewards[index] += self.gamma * self._descent_values * timeouts
        self._rollout_task_reward_sum += native_reward.sum()
        self._rollout_style_reward_sum += ascent_reward.sum()
        self._rollout_descent_reward_sum += descent_reward.sum()
        self._rollout_style_nonzero_count += torch.count_nonzero(ascent_reward)
        # Open the ascent actor channel only with exact V13.3 adjacent-target
        # evidence. V13's permissive side-switch shaping can be positive after
        # a skip and therefore is not sufficient evidence on its own.
        exact_ascent_evidence = ascent_mask & (
            (state["adjacent_progress_delta"] > 0.0)
            | state["adjacent_event"] | state["chain_event"]
        )
        self._rollout_ascent_positive_count += torch.count_nonzero(exact_ascent_evidence)
        self._rollout_descent_positive_count += torch.count_nonzero(descent_reward > 0.0)
        self._rollout_sample_count += native_reward.numel()
        self.style_critic.update_normalization(obs)
        self.descent_critic.update_normalization(obs)
        PPO.process_env_step(self, obs, native_reward, dones, extras)
        self.style_critic.reset(dones)
        self.descent_critic.reset(dones)
        self._style_values = None
        self._descent_values = None
        self._pending_ascent_mask = None
        self._pending_descent_mask = None
        self._pending_pre_b1_actionable = None

    @staticmethod
    def _compute_channel_returns(storage, rewards, values, output, last_values, gamma, lam):
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(storage.num_transitions_per_env)):
            next_values = last_values if step == storage.num_transitions_per_env - 1 else values[step + 1]
            not_terminal = 1.0 - storage.dones[step].float()
            delta = rewards[step] + gamma * next_values * not_terminal - values[step]
            advantage = delta + gamma * lam * not_terminal * advantage
            output[step] = advantage + values[step]

    def _update_ascent_multiplier(self):
        counts = self._v133_rollout_counts
        success = int(counts["adjacent"])
        failure = int(counts["repeat"] + counts["skip"] + counts["simultaneous"])
        if success + failure == 0 or int(self._rollout_ascent_positive_count) == 0:
            return
        precision = success / (success + failure)
        if precision < self.ascent_precision_target:
            self.ascent_advantage_weight = min(
                self.ascent_advantage_weight_max,
                self.ascent_advantage_weight * (1.0 + self.ascent_lagrange_rate),
            )
        elif precision > min(1.0, self.ascent_precision_target + 0.10):
            self.ascent_advantage_weight = max(
                1.0, self.ascent_advantage_weight / (1.0 + self.ascent_lagrange_rate),
            )

    def _critic_sensitivity(self, obs):
        if obs["critic"].shape[-1] != 580:
            raise RuntimeError(f"V13.4 requires a 580D critic, got {obs['critic'].shape[-1]}D.")
        out = {}
        critics = (
            ("task", self.critic),
            ("ascent", self.style_critic),
            ("descent", self.descent_critic),
        )
        with torch.inference_mode():
            for critic_name, critic in critics:
                original = critic(obs)
                for group, bounds in (("ascent6", (-12, -6)), ("descent6", (-6, None))):
                    start, stop = bounds
                    for mode in ("zero", "shuffle"):
                        changed_obs = obs.clone()
                        critic_obs = obs["critic"].clone()
                        if mode == "zero":
                            critic_obs[..., start:stop] = 0.0
                        else:
                            critic_obs[..., start:stop] = torch.roll(
                                critic_obs[..., start:stop], shifts=1, dims=0,
                            )
                        changed_obs["critic"] = critic_obs
                        changed = critic(changed_obs)
                        out[f"v134_{critic_name}_{group}_{mode}_delta_v"] = float(
                            (changed - original).abs().mean()
                        )
        return out

    def compute_returns(self, obs):
        PPO.compute_returns(self, obs)
        storage = self.storage
        self._chain_redistributed = v134_redistribute_chain_settlement(
            storage.style_rewards,
            self._v134_masks["ordered"],
            self._v134_masks["chain"],
            storage.dones.bool(),
            self.chain_settlement,
        )
        self._compute_channel_returns(
            storage, storage.style_rewards, storage.style_values, storage.style_returns,
            self.style_critic(obs).detach(), self.ascent_gamma, self.ascent_lam,
        )
        self._compute_channel_returns(
            storage, storage.descent_rewards, storage.descent_values, storage.descent_returns,
            self.descent_critic(obs).detach(), self.gamma, self.lam,
        )
        native = v134_terrain_normalize(
            storage.returns - storage.values, storage.ascent_masks, storage.descent_masks,
        )
        ascent, ascent_scale = v134_masked_normalize(
            storage.style_returns - storage.style_values,
            storage.ascent_masks,
            positive_signal_count=int(self._rollout_ascent_positive_count),
            minimum_positive_signals=self.ascent_minimum_positive_signals,
            clip=2.0,
        )
        self._ascent_positive_total += int(self._rollout_ascent_positive_count)
        if self._ascent_positive_total >= self.descent_enable_after_ascent_positive:
            self._descent_actor_enabled = True
        descent_positive = int(self._rollout_descent_positive_count) if self._descent_actor_enabled else 0
        descent, descent_scale = v134_masked_normalize(
            storage.descent_returns - storage.descent_values,
            storage.descent_masks,
            positive_signal_count=descent_positive,
            minimum_positive_signals=self.descent_minimum_positive_signals,
            clip=2.0,
        )
        self._update_ascent_multiplier()
        storage.task_advantages.copy_(native)
        storage.style_advantages.copy_(ascent)
        storage.descent_advantages.copy_(descent)
        storage.advantages.copy_(v134_combine_terrain_advantages(
            native, ascent, descent, storage.ascent_masks, storage.descent_masks,
            ascent_weight=self.ascent_advantage_weight,
            descent_weight=self.descent_advantage_weight,
        ))
        self._ascent_actor_scale = ascent_scale
        self._descent_actor_scale = descent_scale

        diagnostics = {}
        channels = {
            "task_return": storage.returns,
            "style_return": storage.style_returns,
            "task_advantage": storage.task_advantages,
            "style_advantage": storage.style_advantages,
            "combined_advantage": storage.advantages,
        }
        for behavior, mask in self._v134_masks.items():
            count = int(mask.sum())
            diagnostics[f"v134_{behavior}_sample_count"] = float(count)
            diagnostics[f"v134_{behavior}_sample_fraction"] = count / max(
                storage.num_envs * storage.num_transitions_per_env, 1,
            )
            for channel, value in channels.items():
                mean, _ = v134_masked_mean(value, mask)
                diagnostics[f"v134_{behavior}_{channel}"] = mean
        diagnostics.update(self._critic_sensitivity(obs))
        diagnostics["v134_ascent_positive_total"] = float(self._ascent_positive_total)
        diagnostics["v134_descent_actor_enabled"] = float(self._descent_actor_enabled)
        diagnostics["v134_chain_redistributed"] = float(self._chain_redistributed)
        self._v134_diagnostics = diagnostics

    @staticmethod
    def _clipped_value_loss(values, old_values, returns, clip_param, enabled):
        if not enabled:
            return (returns - values).square().mean()
        clipped = old_values + (values - old_values).clamp(-clip_param, clip_param)
        return torch.maximum(
            (values - returns).square(), (clipped - returns).square(),
        ).mean()

    @staticmethod
    def _actor_grads(loss, parameters):
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return [
            torch.zeros_like(parameter) if grad is None else grad
            for parameter, grad in zip(parameters, grads)
        ]

    def update(self):
        storage = self.storage
        total = storage.num_envs * storage.num_transitions_per_env
        mini_batch = total // self.num_mini_batches
        obs = storage.observations.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        old_values = storage.values.flatten(0, 1)
        old_ascent_values = storage.style_values.flatten(0, 1)
        old_descent_values = storage.descent_values.flatten(0, 1)
        returns = storage.returns.flatten(0, 1)
        ascent_returns = storage.style_returns.flatten(0, 1)
        descent_returns = storage.descent_returns.flatten(0, 1)
        native_advantages = storage.task_advantages.flatten(0, 1).squeeze(-1)
        ascent_advantages = storage.style_advantages.flatten(0, 1).squeeze(-1)
        descent_advantages = storage.descent_advantages.flatten(0, 1).squeeze(-1)
        ascent_masks = storage.ascent_masks.flatten(0, 1).squeeze(-1)
        descent_masks = storage.descent_masks.flatten(0, 1).squeeze(-1)
        old_log_prob = storage.actions_log_prob.flatten(0, 1)
        old_distribution = tuple(value.flatten(0, 1) for value in storage.distribution_params)
        losses = {
            "surrogate": 0.0, "task_value": 0.0, "style_value": 0.0,
            "descent_value": 0.0, "entropy": 0.0, "ascent_gradient_dot": 0.0,
            "ascent_gradient_conflict_rate": 0.0,
            "descent_ascent_conflict_rate": 0.0,
            "ascent_gradient_gain": 0.0, "ascent_gradient_norm": 0.0,
            "native_up_gradient_norm": 0.0,
        }
        updates = 0
        actor_parameters = list(self.actor.parameters())

        for _ in range(self.num_learning_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, self.num_mini_batches * mini_batch, mini_batch):
                ids = permutation[start:start + mini_batch]
                batch_obs = obs[ids]
                self.actor(batch_obs, stochastic_output=True)
                log_prob = self.actor.get_output_log_prob(actions[ids])
                task_values = self.critic(batch_obs)
                ascent_values = self.style_critic(batch_obs)
                descent_values = self.descent_critic(batch_obs)
                distribution = tuple(value for value in self.actor.output_distribution_params)
                entropy = self.actor.output_entropy
                if self.desired_kl is not None and self.schedule == "adaptive":
                    with torch.inference_mode():
                        previous = tuple(value[ids] for value in old_distribution)
                        kl_mean = self.actor.get_kl_divergence(previous, distribution).mean()
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                        for group in self.optimizer.param_groups:
                            group["lr"] = self.learning_rate

                ratio = torch.exp(log_prob - old_log_prob[ids].squeeze(-1))
                clipped_ratio = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
                def surrogate(advantage, mask=None):
                    loss = torch.maximum(
                        -advantage * ratio, -advantage * clipped_ratio,
                    )
                    if mask is None:
                        return loss.mean()
                    return (loss * mask).sum() / mask.sum().clamp_min(1.0)

                up_mask = ascent_masks[ids].float()
                down_mask = descent_masks[ids].float()
                native_up_loss = surrogate(native_advantages[ids], up_mask)
                native_other_loss = surrogate(native_advantages[ids], 1.0 - up_mask)
                ascent_loss = surrogate(ascent_advantages[ids], up_mask)
                descent_loss = surrogate(descent_advantages[ids], down_mask)
                entropy_loss = -self.entropy_coef * entropy.mean()
                native_up_grad = self._actor_grads(native_up_loss, actor_parameters)
                native_other_grad = self._actor_grads(native_other_loss, actor_parameters)
                ascent_grad = self._actor_grads(ascent_loss, actor_parameters)
                descent_grad = self._actor_grads(descent_loss, actor_parameters)
                entropy_grad = self._actor_grads(entropy_loss, actor_parameters)
                if self._ascent_actor_scale > 0.0:
                    native_up_grad, dot, conflict = v134_project_gradient_away(
                        native_up_grad, ascent_grad,
                    )
                    descent_grad, _, down_conflict = v134_project_gradient_away(
                        descent_grad, ascent_grad,
                    )
                    dot = float(dot)
                    ascent_gain, ascent_norm, native_up_norm = v134_priority_gradient_gain(
                        ascent_grad, native_up_grad,
                        base_gain=self.ascent_advantage_weight,
                        minimum_ratio=self.ascent_gradient_minimum_ratio,
                        maximum_gain=self.ascent_gradient_gain_max,
                    )
                else:
                    dot, conflict, down_conflict = 0.0, False, False
                    ascent_gain, ascent_norm, native_up_norm = 0.0, 0.0, 0.0

                task_value_loss = self._clipped_value_loss(
                    task_values, old_values[ids], returns[ids],
                    self.clip_param, self.use_clipped_value_loss,
                )
                ascent_value_loss = self._clipped_value_loss(
                    ascent_values, old_ascent_values[ids], ascent_returns[ids],
                    self.clip_param, self.use_clipped_value_loss,
                )
                descent_value_loss = self._clipped_value_loss(
                    descent_values, old_descent_values[ids], descent_returns[ids],
                    self.clip_param, self.use_clipped_value_loss,
                )
                value_loss = task_value_loss + ascent_value_loss + descent_value_loss
                self.optimizer.zero_grad()
                value_loss.backward()
                for index, parameter in enumerate(actor_parameters):
                    parameter.grad = (
                        native_other_grad[index] + native_up_grad[index]
                        + ascent_gain * ascent_grad[index]
                        + self.descent_advantage_weight * descent_grad[index]
                        + entropy_grad[index]
                    )
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.style_critic.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.descent_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                combined_surrogate = (
                    native_up_loss + native_other_loss
                    + self.ascent_advantage_weight * ascent_loss
                    + self.descent_advantage_weight * descent_loss
                )
                losses["surrogate"] += float(combined_surrogate.detach())
                losses["task_value"] += float(task_value_loss.detach())
                losses["style_value"] += float(ascent_value_loss.detach())
                losses["descent_value"] += float(descent_value_loss.detach())
                losses["entropy"] += float(entropy.mean().detach())
                losses["ascent_gradient_dot"] += dot
                losses["ascent_gradient_conflict_rate"] += float(conflict)
                losses["descent_ascent_conflict_rate"] += float(down_conflict)
                losses["ascent_gradient_gain"] += ascent_gain
                losses["ascent_gradient_norm"] += ascent_norm
                losses["native_up_gradient_norm"] += native_up_norm
                updates += 1

        counts = {name: int(value) for name, value in self._v133_rollout_counts.items()}
        failures = counts["repeat"] + counts["skip"] + counts["simultaneous"]
        for key in losses:
            losses[key] /= max(updates, 1)
        for name, value in counts.items():
            losses[f"v133_{name}_count"] = float(value)
        losses["v133_chain_precision"] = counts["chain"] / max(counts["chain"] + failures, 1)
        losses["v133_chain_fl_fraction"] = counts["chain_fl"] / max(counts["chain"], 1)
        losses["v133_style_actor_scale"] = self._ascent_actor_scale
        losses["v134_descent_actor_scale"] = self._descent_actor_scale
        losses["v134_ascent_advantage_weight"] = self.ascent_advantage_weight
        losses["v134_descent_actor_enabled"] = float(self._descent_actor_enabled)
        pre_b1_samples = int(self._rollout_pre_b1_actionable)
        losses["v134_pre_b1_actionable_samples"] = float(pre_b1_samples)
        losses["v134_pre_b1_backward_samples"] = float(self._rollout_pre_b1_backward)
        losses["v134_pre_b1_backward_rate"] = (
            float(self._rollout_pre_b1_backward) / max(pre_b1_samples, 1)
        )
        losses["v134_chain_success_total"] = float(self._chain_success_total)
        samples = max(self._rollout_sample_count, 1)
        losses["task_reward_per_step"] = float(self._rollout_task_reward_sum / samples)
        losses["style_reward_per_step"] = float(self._rollout_style_reward_sum / samples)
        losses["v134_descent_reward_per_step"] = float(
            self._rollout_descent_reward_sum / samples,
        )
        losses.update(self._v134_diagnostics)
        self._chain_success_total += counts["chain"]

        storage.clear()
        for value in self._v133_rollout_counts.values():
            value.zero_()
        for mask in self._v134_masks.values():
            mask.zero_()
        self._rollout_task_reward_sum.zero_()
        self._rollout_style_reward_sum.zero_()
        self._rollout_descent_reward_sum.zero_()
        self._rollout_style_nonzero_count.zero_()
        self._rollout_ascent_positive_count.zero_()
        self._rollout_descent_positive_count.zero_()
        self._rollout_pre_b1_actionable.zero_()
        self._rollout_pre_b1_backward.zero_()
        self._rollout_sample_count = 0
        self._v134_diagnostics.clear()
        return losses

    def train_mode(self):
        super().train_mode()
        self.descent_critic.train()

    def eval_mode(self):
        super().eval_mode()
        self.descent_critic.eval()

    def save(self):
        saved = super().save()
        saved["descent_critic_state_dict"] = self.descent_critic.state_dict()
        saved["v134_ascent_positive_total"] = self._ascent_positive_total
        saved["v134_descent_actor_enabled"] = self._descent_actor_enabled
        saved["v134_ascent_advantage_weight"] = self.ascent_advantage_weight
        saved["v134_chain_success_total"] = self._chain_success_total
        saved["v134_terrain_separated_three_critic"] = True
        return saved

    def reduce_parameters(self):
        parameters = list(chain(
            self.actor.parameters(), self.critic.parameters(),
            self.style_critic.parameters(), self.descent_critic.parameters(),
        ))
        grads = [
            parameter.grad.reshape(-1)
            for parameter in parameters if parameter.grad is not None
        ]
        flattened = torch.cat(grads)
        torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
        flattened /= self.gpu_world_size
        offset = 0
        for parameter in parameters:
            if parameter.grad is not None:
                size = parameter.numel()
                parameter.grad.data.copy_(
                    flattened[offset:offset + size].view_as(parameter.grad),
                )
                offset += size


class V134DiagnosticRunner(V133TransitionRunner):
    """Construct the unchanged 63D actor and three terrain-specific critics."""

    algorithm_class_name = "rl_training.v134_diagnostic_runner:V134DiagnosticPPO"

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        cfg = dict(train_cfg)
        cfg["algorithm"] = dict(cfg["algorithm"])
        channel_cfg = {
            "ascent_reward_terms": cfg.pop("v134_ascent_reward_terms"),
            "descent_reward_terms": cfg.pop("v134_descent_reward_terms"),
            "ascent_gamma": cfg.pop("v134_ascent_gamma"),
            "ascent_lam": cfg.pop("v134_ascent_lam"),
            "ascent_minimum_positive_signals": cfg.pop("v134_ascent_minimum_positive_signals"),
            "descent_minimum_positive_signals": cfg.pop("v134_descent_minimum_positive_signals"),
            "descent_enable_after_ascent_positive": cfg.pop("v134_descent_enable_after_ascent_positive"),
            "ascent_advantage_weight": cfg.pop("v134_ascent_advantage_weight"),
            "ascent_advantage_weight_max": cfg.pop("v134_ascent_advantage_weight_max"),
            "ascent_lagrange_rate": cfg.pop("v134_ascent_lagrange_rate"),
            "ascent_precision_target": cfg.pop("v134_ascent_precision_target"),
            "descent_advantage_weight": cfg.pop("v134_descent_advantage_weight"),
            "chain_settlement": cfg.pop("v134_chain_settlement"),
            "ascent_gradient_minimum_ratio": cfg.pop("v134_ascent_gradient_minimum_ratio"),
            "ascent_gradient_gain_max": cfg.pop("v134_ascent_gradient_gain_max"),
        }
        cfg["algorithm"].update(channel_cfg)
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        if not isinstance(self.alg, V134DiagnosticPPO):
            raise RuntimeError("V13.4 runner did not construct terrain-separated PPO.")
        print("V134_TERRAIN_SEPARATED_THREE_CRITIC=YES", flush=True)
        print("V134_ASCENT_POSITIVE_SIGNAL_GATE=YES", flush=True)
        print("V134_ASCENT_PRIORITY_GRADIENT_PROJECTION=YES", flush=True)
        print("V134_DESCENT_ACTOR_DELAYED_BY_ASCENT_EVIDENCE=YES", flush=True)
        print("V134_CHAIN_CREDIT_REDISTRIBUTION=YES", flush=True)
        print("V134_NATIVE_V13_ACTION_DISTRIBUTION=YES", flush=True)
        print("V134_PRE_B1_BACKWARD_AUDIT=YES", flush=True)
        print("V134_ASCENT_GRADIENT_MINIMUM_STRENGTH=YES", flush=True)


__all__ = [
    "V134DiagnosticPPO", "V134DiagnosticRunner", "v134_masked_mean",
    "v134_redistribute_chain_settlement",
]
