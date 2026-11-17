"""Sparse ascent-teacher rollouts for reward-driven V13.4 learning."""

from __future__ import annotations

import copy
import hashlib
import os

import torch
from rsl_rl.algorithms import PPO

from .v11_continuous_runner import V11ContinuousRunner


class V134StylePriorPPO(PPO):
    """Expose a fresh student to sparse teacher trajectories on ascent only.

    The environment executes the frozen teacher action during a short burst, so
    those transitions contribute rewards, returns, and critic targets.  The
    stored action remains the student's sampled action and its advantage is
    zeroed before PPO, preventing an off-policy teacher action from entering the
    actor surrogate.  Student-controlled transitions before and after a burst
    retain their GAE signal from the ordered-ascent rewards.
    """

    def configure_style_prior(
        self,
        start_probability: float,
        end_probability: float,
        decay_updates: int,
        burst_steps: int,
        env,
        teacher_actor,
    ) -> None:
        if not 0.0 <= float(end_probability) <= float(start_probability) <= 1.0:
            raise ValueError("V13.4 teacher burst probabilities must lie in [0, 1] and decay.")
        if int(decay_updates) < 1 or int(burst_steps) < 1:
            raise ValueError("V13.4 teacher burst duration and decay must be positive.")
        if self.normalize_advantage_per_mini_batch:
            raise RuntimeError("V13.4 teacher masking requires rollout-level advantage normalization.")

        self._v134_teacher_burst_start_probability = float(start_probability)
        self._v134_teacher_burst_end_probability = float(end_probability)
        self._v134_teacher_burst_decay_updates = int(decay_updates)
        self._v134_teacher_burst_steps = int(burst_steps)
        self._v134_update_count = 0
        self._v134_env = getattr(env, "unwrapped", env)
        self._v134_teacher_actor = teacher_actor.eval()
        for parameter in self._v134_teacher_actor.parameters():
            parameter.requires_grad_(False)

        num_envs = int(self.storage.num_envs)
        self._v134_burst_remaining = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._v134_pending_teacher_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.storage.v134_teacher_action_masks = torch.zeros(
            self.storage.num_transitions_per_env,
            num_envs,
            1,
            dtype=torch.bool,
            device=self.device,
        )
        self._v134_eligible_samples = 0
        self._v134_teacher_action_samples = 0
        self._v134_teacher_burst_count = 0

    def _teacher_probability(self) -> float:
        progress = min(
            float(self._v134_update_count) / float(self._v134_teacher_burst_decay_updates),
            1.0,
        )
        return self._v134_teacher_burst_start_probability + progress * (
            self._v134_teacher_burst_end_probability
            - self._v134_teacher_burst_start_probability
        )

    def _eligible_ascent_mask(self) -> torch.Tensor:
        from .tasks.manager_based.locomotion.velocity.mdp.v11_continuous import v11_state
        from .tasks.manager_based.locomotion.velocity.mdp.v133_adjacent_transition import (
            v133_ascending_stair_mask,
        )
        from .tasks.manager_based.locomotion.velocity.mdp.v134_perceptive_follow import (
            v134_ascent_style_anchor_mask,
        )

        state = v11_state(self._v134_env)
        risers = state["risers"]
        positive_command = self._v134_env.command_manager.get_command("base_velocity")[:, 0] > float(
            self._v134_env.cfg.v11_perception.forward_command_threshold
        )
        return v134_ascent_style_anchor_mask(
            v133_ascending_stair_mask(self._v134_env),
            risers["valid"],
            risers["riser_height"],
            float(self._v134_env.cfg.v133_transition.minimum_height_gain),
            positive_command,
        )

    def act(self, obs):
        student_actions = super().act(obs)
        eligible = self._eligible_ascent_mask()
        self._v134_eligible_samples += int(eligible.sum().item())

        # A burst ends immediately if the robot leaves a valid ascending-riser
        # state.  Descent, flat terrain, and the top platform therefore never
        # execute teacher actions.
        self._v134_burst_remaining.masked_fill_(~eligible, 0)
        idle = self._v134_burst_remaining == 0
        starts = eligible & idle & (
            torch.rand(eligible.shape, device=eligible.device) < self._teacher_probability()
        )
        self._v134_burst_remaining[starts] = self._v134_teacher_burst_steps
        teacher_mask = eligible & (self._v134_burst_remaining > 0)
        self._v134_burst_remaining[teacher_mask] -= 1
        self._v134_pending_teacher_mask.copy_(teacher_mask)
        self._v134_teacher_burst_count += int(starts.sum().item())
        self._v134_teacher_action_samples += int(teacher_mask.sum().item())

        if not bool(teacher_mask.any()):
            return student_actions
        with torch.inference_mode():
            teacher_actions = self._v134_teacher_actor(obs, stochastic_output=False)
        return torch.where(teacher_mask.unsqueeze(1), teacher_actions, student_actions)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        step = int(self.storage.step)
        self.storage.v134_teacher_action_masks[step, :, 0].copy_(
            self._v134_pending_teacher_mask
        )
        done_mask = dones.bool().view(-1)
        self._v134_burst_remaining[done_mask] = 0
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        teacher_masks = self.storage.v134_teacher_action_masks
        teacher_count = int(teacher_masks.sum().item())
        masked_advantage_mean = 0.0
        if teacher_count:
            masked_advantage_mean = float(
                self.storage.advantages[teacher_masks].mean().detach().item()
            )
            # compute_returns commonly runs under inference_mode, which makes
            # its newly assigned advantage tensor immutable outside that
            # context.  Clone before masking so PPO receives a normal tensor.
            masked_advantages = self.storage.advantages.clone()
            masked_advantages.masked_fill_(teacher_masks, 0.0)
            self.storage.advantages = masked_advantages

        losses = super().update()
        total_samples = int(teacher_masks.numel())
        losses["v134_teacher_burst_probability"] = float(self._teacher_probability())
        losses["v134_teacher_action_rate"] = float(teacher_count) / float(max(total_samples, 1))
        losses["v134_teacher_action_rate_given_eligible"] = (
            float(teacher_count) / float(max(self._v134_eligible_samples, 1))
        )
        losses["v134_teacher_burst_count"] = float(self._v134_teacher_burst_count)
        losses["v134_teacher_masked_advantage_mean"] = masked_advantage_mean
        losses["v134_teacher_actor_samples_masked"] = float(teacher_count)

        self._v134_update_count += 1
        teacher_masks.zero_()
        self._v134_pending_teacher_mask.zero_()
        self._v134_eligible_samples = 0
        self._v134_teacher_action_samples = 0
        self._v134_teacher_burst_count = 0
        return losses


class V134StylePriorRunner(V11ContinuousRunner):
    """Start fresh; use V13 model_5000 only for sparse ascent rollouts."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        cfg = dict(train_cfg)
        require_actor = bool(cfg.pop("require_v134_style_actor", False))
        start_probability = float(cfg.pop("v134_teacher_burst_start_probability", 0.0))
        end_probability = float(cfg.pop("v134_teacher_burst_end_probability", 0.0))
        decay_updates = int(cfg.pop("v134_teacher_burst_decay_updates", 0))
        burst_steps = int(cfg.pop("v134_teacher_burst_steps", 0))
        super().__init__(env, cfg, log_dir=log_dir, device=device)
        if not require_actor:
            raise RuntimeError("V13.4 sparse teacher runner requires an explicit actor checkpoint.")
        if not isinstance(self.alg, V134StylePriorPPO):
            raise RuntimeError("V13.4 sparse teacher runner received the wrong PPO implementation.")

        checkpoint_env = os.environ.get("V134_ASCENT_TEACHER_CHECKPOINT", "").strip()
        if not checkpoint_env:
            raise RuntimeError("Set V134_ASCENT_TEACHER_CHECKPOINT to the reviewed V13 actor.")
        checkpoint = os.path.realpath(checkpoint_env)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"V13.4 ascent teacher does not exist: {checkpoint}")

        student_before = {
            key: value.detach().cpu().clone()
            for key, value in self.alg.actor.state_dict().items()
        }
        if self.alg.optimizer.state:
            raise RuntimeError("V13.4 optimizer must start fresh.")
        loaded = torch.load(checkpoint, weights_only=False, map_location="cpu")
        expected_actor = loaded["actor_state_dict"]
        teacher_actor = copy.deepcopy(self.alg.actor)
        teacher_actor.load_state_dict(expected_actor, strict=True)
        actual_actor = teacher_actor.state_dict()
        if actual_actor.keys() != expected_actor.keys() or any(
            not torch.equal(actual_actor[key].detach().cpu(), expected_actor[key].detach().cpu())
            for key in actual_actor
        ):
            raise RuntimeError("V13.4 ascent teacher did not load exactly.")
        student_after = self.alg.actor.state_dict()
        if student_after.keys() != student_before.keys() or any(
            not torch.equal(student_after[key].detach().cpu(), student_before[key])
            for key in student_after
        ):
            raise RuntimeError("V13.4 student was initialized from the teacher.")
        if all(
            torch.equal(student_after[key].detach().cpu(), expected_actor[key].detach().cpu())
            for key in student_after
        ):
            raise RuntimeError("V13.4 fresh student unexpectedly equals the V13 teacher.")
        if self.alg.optimizer.state:
            raise RuntimeError("V13.4 unexpectedly inherited optimizer state.")
        if self.current_learning_iteration != 0:
            raise RuntimeError("V13.4 sparse teacher objective must start at iteration zero.")

        self.alg.configure_style_prior(
            start_probability,
            end_probability,
            decay_updates,
            burst_steps,
            env,
            teacher_actor,
        )
        with open(checkpoint, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        print(f"V134_ASCENT_TEACHER_CHECKPOINT={checkpoint}", flush=True)
        print(f"V134_ASCENT_TEACHER_SHA256={digest}", flush=True)
        print("V134_STYLE_TEACHER_EXACT_LOAD=YES", flush=True)
        print("V134_STYLE_STUDENT_FRESH=YES", flush=True)
        print("V134_STYLE_CRITIC_FRESH=YES", flush=True)
        print("V134_STYLE_OPTIMIZER_FRESH=YES", flush=True)
        print("V134_STYLE_START_ITERATION=0", flush=True)
        print(f"V134_TEACHER_BURST_START_PROBABILITY={start_probability}", flush=True)
        print(f"V134_TEACHER_BURST_END_PROBABILITY={end_probability}", flush=True)
        print(f"V134_TEACHER_BURST_DECAY_UPDATES={decay_updates}", flush=True)
        print(f"V134_TEACHER_BURST_STEPS={burst_steps}", flush=True)
        print("V134_TEACHER_ASCENT_ONLY=YES", flush=True)
        print("V134_TEACHER_ACTIONS_EXCLUDED_FROM_ACTOR_PPO=YES", flush=True)
        print("V134_TEACHER_RETURNS_AVAILABLE_TO_CRITIC=YES", flush=True)
        print("V134_ASCENT_REWARD_DRIVES_STUDENT_ACTOR=YES", flush=True)


__all__ = ["V134StylePriorPPO", "V134StylePriorRunner"]
