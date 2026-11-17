"""Read-only helpers for V13.5 PPO attribution.

This module is deliberately not registered as the production runner.  It owns
no reward terms and never changes the V13.5 environment or policy ABI.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io

import torch
from torch import nn
from rsl_rl.algorithms import PPO
from rl_training.v11_continuous_runner import V11ContinuousRunner


NONE = 0
B1_FREE = 1
ALT = 2
WRONG_OR_SAME_SIDE = 3
SKIP_DUAL = 4
ORDINARY_NONSTRICT = 5
LABEL_NAMES = {
    NONE: "NONE", B1_FREE: "B1_FREE", ALT: "ALT",
    WRONG_OR_SAME_SIDE: "WRONG_OR_SAME_SIDE", SKIP_DUAL: "SKIP_DUAL",
    ORDINARY_NONSTRICT: "ORDINARY_NONSTRICT",
}


def classify_v135_frames(strict: dict, landing: dict) -> torch.Tensor:
    """Classify frames using the existing strict ledger; create no second FSM."""
    device = strict["valid"].device
    label = torch.full_like(strict["target_id"], NONE, dtype=torch.long, device=device)
    actionable = strict["target_valid"] | landing["target_active"]
    b1_free = actionable & ~strict["seeded_before"] & ~strict["seeded"]
    physical = landing["landing_event"] | landing["front_crossing_event"].any(1)
    label[actionable & physical] = ORDINARY_NONSTRICT
    label[b1_free & physical] = B1_FREE
    label[strict["wrong_side"]] = WRONG_OR_SAME_SIDE
    label[strict["skip"] | strict["dual_skip"]] = SKIP_DUAL
    label[strict["ordered"]] = ALT
    return label


def expand_event_windows(labels: torch.Tensor, radius: int) -> torch.Tensor:
    """Assign each action frame to the nearest event without crossing env axes."""
    if labels.ndim != 2:
        raise ValueError("labels must be [time, env]")
    out = labels.clone()
    event_types = (ALT, WRONG_OR_SAME_SIDE, SKIP_DUAL, ORDINARY_NONSTRICT, B1_FREE)
    for env_id in range(labels.shape[1]):
        events = torch.nonzero(labels[:, env_id] != NONE, as_tuple=False).flatten()
        for center in events.tolist():
            lo, hi = max(0, center - radius), min(labels.shape[0], center + radius + 1)
            distances = torch.arange(lo, hi, device=labels.device).sub(center).abs()
            current = out[lo:hi, env_id]
            # Event frames keep their exact class. NONE neighbors inherit the
            # nearest event; ties retain the earlier assignment deterministically.
            replace = current == NONE
            current[replace] = labels[center, env_id]
            out[lo:hi, env_id] = current
    return out


def raw_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    """Reproduce installed RSL-RL GAE before its global normalization."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_values)
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        alive = 1.0 - dones[step].float()
        delta = rewards[step] + alive * gamma * next_values - values[step]
        running = delta + alive * gamma * lam * running
        advantages[step] = running
    return advantages, advantages + values


def matched_gait_summary(labels, returns, advantages, terrain_level, command_vx,
                         target_id, episode_progress):
    """Compare ALT/NORMAL only within shared conditioning strata."""
    command_bin = torch.bucketize(
        command_vx, torch.tensor([0.1, 0.5, 1.0, 1.5, 2.0], device=command_vx.device)
    ) - 1
    progress_bin = torch.clamp((episode_progress * 4).long(), 0, 3)
    alt_values, normal_values, alt_adv, normal_adv = [], [], [], []
    levels = torch.unique(terrain_level)
    for level in levels.tolist():
        for cb in range(4):
            for tid in torch.unique(target_id).tolist():
                for pb in range(4):
                    base = ((terrain_level == level) & (command_bin == cb) &
                            (target_id == tid) & (progress_bin == pb))
                    a = base & (labels == ALT)
                    n = base & (labels == ORDINARY_NONSTRICT)
                    if a.any() and n.any():
                        # Equal weight per stratum prevents easy strata with
                        # many NORMAL frames from dominating the comparison.
                        alt_values.append(returns[a].mean())
                        normal_values.append(returns[n].mean())
                        alt_adv.append(advantages[a].mean())
                        normal_adv.append(advantages[n].mean())
    if not alt_values:
        return {"matched_strata": 0}
    stack_mean = lambda xs: torch.stack(xs).mean().item()
    return {
        "matched_strata": len(alt_values),
        "alt_return": stack_mean(alt_values),
        "normal_return": stack_mean(normal_values),
        "alt_advantage_raw": stack_mean(alt_adv),
        "normal_advantage_raw": stack_mean(normal_adv),
    }


@dataclass
class ParameterSnapshot:
    actor: dict[str, torch.Tensor]
    critic: dict[str, torch.Tensor]
    optimizer_sha256: str

    @staticmethod
    def _optimizer_hash(optimizer) -> str:
        stream = io.BytesIO()
        torch.save(optimizer.state_dict(), stream)
        return hashlib.sha256(stream.getvalue()).hexdigest()

    @classmethod
    def capture(cls, runner):
        clone = lambda state: {k: v.detach().cpu().clone() for k, v in state.items()}
        return cls(clone(runner.alg.actor.state_dict()), clone(runner.alg.critic.state_dict()),
                   cls._optimizer_hash(runner.alg.optimizer))

    def assert_unchanged(self, runner):
        for name, before, after in (
            ("actor", self.actor, runner.alg.actor.state_dict()),
            ("critic", self.critic, runner.alg.critic.state_dict()),
        ):
            if before.keys() != after.keys() or any(
                not torch.equal(before[k], after[k].detach().cpu()) for k in before
            ):
                raise RuntimeError(f"frozen diagnosis mutated {name}")
        if self.optimizer_sha256 != self._optimizer_hash(runner.alg.optimizer):
            raise RuntimeError("frozen diagnosis mutated optimizer state")


class V135AttributionPPO(PPO):
    """Diagnostic-only PPO with exact per-gait minibatch attribution."""

    def set_attribution_env(self, env):
        self.attribution_env = env
        self._gait_labels = []
        self.last_attribution = {}
        self._event_totals = {key: 0 for key in ("crossing", "v11_completion", "valid", "ordered",
                                                   "wrong_side", "skip", "dual_skip", "duplicate", "seeded")}

    def process_env_step(self, obs, rewards, dones, extras):
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.v11_continuous import v11_state
        from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_riser_identity import v135_riser_identity_state
        strict = v135_riser_identity_state(self.attribution_env)
        landing = v11_state(self.attribution_env)["landing"]
        self._gait_labels.append(classify_v135_frames(strict, landing).detach().clone())
        for key, value in (("crossing", landing["front_crossing_event"]),
                           ("v11_completion", landing["landing_event"]),
                           ("valid", strict["valid"]), ("ordered", strict["ordered"]),
                           ("wrong_side", strict["wrong_side"]), ("skip", strict["skip"]),
                           ("dual_skip", strict["dual_skip"]), ("duplicate", strict["duplicate"]),
                           ("seeded", strict["seeded"])):
            self._event_totals[key] += int(value.sum())
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs):
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        td_errors = torch.zeros_like(st.rewards)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            alive = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + alive * self.gamma * next_values - st.values[step]
            td_errors[step] = delta
            advantage = delta + alive * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        # Runner calls compute_returns under inference_mode.  Re-enter normal
        # mode so the diagnostic tensors can participate in autograd during
        # the subsequent PPO attribution pass.
        with torch.inference_mode(False):
            self.raw_advantages = (st.returns.detach().clone() - st.values.detach().clone())
            self.td_errors = td_errors.detach().clone()
            st.advantages = ((self.raw_advantages - self.raw_advantages.mean()) /
                             (self.raw_advantages.std() + 1e-8)).clone()
        self._gait_labels = list(expand_event_windows(torch.stack(self._gait_labels), 16).unbind(0))

    @staticmethod
    def _flat_grad(loss, parameters, retain_graph=True):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
        return torch.cat([torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1)
                          for p, g in zip(parameters, gradients)]).detach()

    def update(self):
        if self.rnd or self.symmetry or self.is_multi_gpu:
            raise RuntimeError("V135 attribution supports the production no-RND/no-symmetry single-GPU configuration only")
        labels = torch.stack(self._gait_labels)
        if labels.shape[:2] != self.storage.rewards.shape[:2]:
            raise RuntimeError(f"gait label/storage mismatch: {labels.shape} vs {self.storage.rewards.shape}")
        stats = {name: {"count": 0, "surrogate_sum": 0.0, "ratio_sum": 0.0,
                        "unclipped_sum": 0.0, "clipped_sum": 0.0, "clip_count": 0,
                        "positive_count": 0, "entropy_sum": 0.0} for name in LABEL_NAMES.values()}
        # Unique rollout statistics are taken before samples repeat across epochs.
        for label_id, name in LABEL_NAMES.items():
            mask = labels == label_id
            if mask.any():
                raw = self.raw_advantages.squeeze(-1)[mask]
                stats[name].update({"rollout_count": int(mask.sum()), "reward_mean": float(self.storage.rewards.squeeze(-1)[mask].mean()),
                                    "return_mean": float(self.storage.returns.squeeze(-1)[mask].mean()),
                                    "value_mean": float(self.storage.values.squeeze(-1)[mask].mean()),
                                    "adv_raw_mean": float(raw.mean()), "adv_raw_std": float(raw.std()),
                                    "adv_normalized_mean": float(self.storage.advantages.squeeze(-1)[mask].mean()),
                                    "td_error_mean": float(self.td_errors.squeeze(-1)[mask].mean()),
                                    "old_log_prob_mean": float(self.storage.actions_log_prob.squeeze(-1)[mask].mean()),
                                    "positive_advantage_fraction": float((raw > 0).float().mean()),
                                    "value_error_mean": float((self.storage.values.squeeze(-1)[mask] - self.storage.returns.squeeze(-1)[mask]).mean())})
            else:
                stats[name]["rollout_count"] = 0

        alt_grad = normal_grad = None
        mean_value_loss = mean_surrogate_loss = mean_entropy = 0.0
        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        mini_batch_size = self.storage.num_envs // self.num_mini_batches
        parameters = tuple(self.actor.parameters())
        for update_index, batch in enumerate(generator):
            mb_index = update_index % self.num_mini_batches
            start, stop = mb_index * mini_batch_size, (mb_index + 1) * mini_batch_size
            batch_labels = labels[:, start:stop]
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(self.actor.output_distribution_params)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params).mean()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups: group["lr"] = self.learning_rate
            ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
            advantage_norm = torch.squeeze(batch.advantages)
            objective_unclipped = ratio * advantage_norm
            objective_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage_norm
            objective = torch.minimum(objective_unclipped, objective_clipped)
            surrogate_loss = -objective.mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
            entropy = self.actor.output_entropy
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            for label_id, name in LABEL_NAMES.items():
                mask = batch_labels == label_id
                if mask.any():
                    count = int(mask.sum())
                    target = stats[name]
                    target["count"] += count
                    target["surrogate_sum"] += float(objective[mask].detach().sum())
                    target["ratio_sum"] += float(ratio[mask].detach().sum())
                    target["unclipped_sum"] += float(objective_unclipped[mask].detach().sum())
                    target["clipped_sum"] += float(objective_clipped[mask].detach().sum())
                    target["clip_count"] += int(((ratio[mask] < 1-self.clip_param) | (ratio[mask] > 1+self.clip_param)).sum())
                    target["positive_count"] += int((objective[mask] > 0).sum())
                    target["entropy_sum"] += float(entropy[mask].detach().sum())
            if update_index < self.num_mini_batches:
                for label_id, holder in ((ALT, "alt"), (ORDINARY_NONSTRICT, "normal")):
                    mask = batch_labels == label_id
                    if int(mask.sum()) >= 8:
                        vector = self._flat_grad(-objective[mask].mean(), parameters)
                        if holder == "alt": alt_grad = vector if alt_grad is None else alt_grad + vector
                        else: normal_grad = vector if normal_grad is None else normal_grad + vector

            self.optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += float(value_loss.detach()); mean_surrogate_loss += float(surrogate_loss.detach())
            mean_entropy += float(entropy.mean().detach())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for target in stats.values():
            count = max(1, target["count"])
            target["sample_fraction"] = target.get("rollout_count", 0) / labels.numel()
            target["surrogate_mean"] = target["surrogate_sum"] / count
            target["ratio_mean"] = target["ratio_sum"] / count
            target["unclipped_mean"] = target["unclipped_sum"] / count
            target["clipped_mean"] = target["clipped_sum"] / count
            target["clip_fraction"] = target["clip_count"] / count
            target["positive_surrogate_fraction"] = target["positive_count"] / count
            target["entropy_mean"] = target["entropy_sum"] / count
        gradient = {"status": "INSUFFICIENT_ALT_SAMPLES"}
        if alt_grad is not None and normal_grad is not None:
            alt_norm, normal_norm = alt_grad.norm(), normal_grad.norm()
            gradient = {"status": "PASS", "alt_norm": float(alt_norm), "normal_norm": float(normal_norm),
                        "normal_to_alt_grad_norm_ratio": float(normal_norm / alt_norm.clamp_min(1e-12)),
                        "alt_normal_cosine": float(torch.dot(alt_grad, normal_grad) / (alt_norm * normal_norm).clamp_min(1e-12))}
        self.last_attribution = {"gaits": stats, "gradient": gradient, "events": dict(self._event_totals)}
        self.storage.clear(); self._gait_labels.clear()
        self._event_totals = {key: 0 for key in self._event_totals}
        return {"value": mean_value_loss/num_updates, "surrogate": mean_surrogate_loss/num_updates,
                "entropy": mean_entropy/num_updates}


class V135AttributionRunner(V11ContinuousRunner):
    """Isolated runner that wires the diagnostic PPO only when explicitly constructed."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        cfg = dict(train_cfg)
        cfg["algorithm"] = dict(cfg["algorithm"])
        cfg["algorithm"]["class_name"] = "rl_training.v135_ppo_attribution_runner:V135AttributionPPO"
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        self.alg.set_attribution_env(env.unwrapped)


__all__ = [
    "NONE", "B1_FREE", "ALT", "WRONG_OR_SAME_SIDE", "SKIP_DUAL",
    "ORDINARY_NONSTRICT", "LABEL_NAMES", "classify_v135_frames",
    "expand_event_windows", "raw_gae", "matched_gait_summary", "ParameterSnapshot",
    "V135AttributionPPO", "V135AttributionRunner",
]
