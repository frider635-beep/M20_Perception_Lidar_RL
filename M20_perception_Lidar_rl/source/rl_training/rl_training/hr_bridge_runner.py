"""Dual-return PPO runner for the independent M20 HR Bridge task."""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from rl_training.tasks.manager_based.locomotion.velocity.mdp.hr_bridge import HRBridgeTracker
from rl_training.tasks.manager_based.locomotion.velocity.mdp.hr_bridge_contracts import (
    STATE_HR_BRIDGE,
    base_action_authority_mask,
    bridge_actor_mask,
    compose_final_action,
)


def _mlp(input_dim, hidden_dims, output_dim):
    layers = []
    last = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(last, width), nn.ELU()))
        last = width
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, init_std=1.0):
        super().__init__()
        self.mlp = _mlp(obs_dim, hidden_dims, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_std)).log())

    def distribution(self, obs):
        mean = self.mlp(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)


class HRBridgePolicy(nn.Module):
    def __init__(self, base_critic_dim, hidden_dims):
        super().__init__()
        self.base_actor = GaussianActor(57, 16, hidden_dims)
        self.bridge_actor = GaussianActor(372, 3, hidden_dims)
        self.base_critic = _mlp(base_critic_dim, hidden_dims, 1)
        self.bridge_critic = _mlp(372, hidden_dims, 1)

    @property
    def action_std(self):
        return torch.cat((self.base_actor.log_std.exp(), self.bridge_actor.log_std.exp()))


@dataclass
class Rollout:
    base_obs: torch.Tensor
    bridge_obs: torch.Tensor
    critic_obs: torch.Tensor
    base_actions: torch.Tensor
    bridge_actions: torch.Tensor
    base_log_prob_dim: torch.Tensor
    bridge_log_prob_dim: torch.Tensor
    base_action_mask: torch.Tensor
    bridge_actor_mask: torch.Tensor
    base_values: torch.Tensor
    bridge_values: torch.Tensor
    base_rewards: torch.Tensor
    bridge_rewards: torch.Tensor
    dones: torch.Tensor
    base_returns: torch.Tensor
    bridge_returns: torch.Tensor
    base_advantages: torch.Tensor
    bridge_advantages: torch.Tensor


class HRBridgeRunner:
    """PPO with separate actors, critics, returns and executed-action masks."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.device = torch.device(device)
        self.log_dir = log_dir
        self.cfg = train_cfg
        self.num_steps_per_env = int(train_cfg["num_steps_per_env"])
        self.save_interval = int(train_cfg["save_interval"])
        self.current_learning_iteration = 0
        self.tracker = HRBridgeTracker(env.unwrapped)
        obs = env.get_observations().to(self.device)
        policy_obs, critic_obs = obs["policy"], obs["critic"]
        if policy_obs.shape[1] != 372:
            raise RuntimeError(f"HR Bridge policy observation must be 372-D, got {policy_obs.shape}")
        if critic_obs.shape[1] != 247:
            raise RuntimeError(f"official Base critic observation must be 247-D, got {critic_obs.shape}")
        hidden = train_cfg.get("hr_bridge_hidden_dims", [512, 256, 128])
        self.policy = HRBridgePolicy(critic_obs.shape[1], hidden).to(self.device)
        alg = train_cfg["algorithm"]
        self.clip_param = float(alg["clip_param"])
        self.gamma = float(alg["gamma"])
        self.lam = float(alg["lam"])
        self.value_loss_coef = float(alg["value_loss_coef"])
        self.entropy_coef = float(alg["entropy_coef"])
        self.num_epochs = int(alg["num_learning_epochs"])
        self.num_mini_batches = int(alg["num_mini_batches"])
        self.max_grad_norm = float(alg["max_grad_norm"])
        self.learning_rate = float(alg["learning_rate"])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        reward_cfg = env.unwrapped.cfg.hr_bridge_reward
        self.hr_success_weight = float(reward_cfg.hr_success_weight)
        self.perception_miss_weight = float(reward_cfg.perception_miss_weight)
        self.joint_success_weight = float(reward_cfg.joint_success_weight)
        self.rewards_calibrated = bool(reward_cfg.calibrated)
        self.handoff_audit_passed = bool(reward_cfg.handoff_audit_passed)
        self.formal_training_enabled = bool(reward_cfg.formal_training_enabled)
        self.previous_final_action = torch.zeros((env.num_envs, 16), device=self.device)
        self.delta_limit = float(env.unwrapped.cfg.hr_bridge_teacher.bridge_action_delta_limit)
        self.writer = None
        self.tot_timesteps = 0
        self._audit_header_written = False
        self._write_architecture_audit(critic_obs.shape[1])

    def _write_architecture_audit(self, critic_dim):
        print("HR_BRIDGE_BASE_ACTOR=57D->16D", flush=True)
        print("HR_BRIDGE_ACTOR=372D->3D", flush=True)
        print(f"HR_BRIDGE_BASE_CRITIC={critic_dim}D->1D", flush=True)
        print("HR_BRIDGE_CRITIC=372D->1D", flush=True)
        print("HR_BRIDGE_PRETRAINED_POLICY=NONE", flush=True)

    def add_git_repo_to_log(self, _path):
        return None

    def _allocate(self):
        t, n, d = self.num_steps_per_env, self.env.num_envs, self.device
        z = lambda *shape: torch.zeros((t, n, *shape), device=d)
        return Rollout(
            z(57), z(372), z(247), z(16), z(3), z(16), z(3),
            z(16), z(1), z(1), z(1), z(1), z(1), z(1),
            z(1), z(1), z(1), z(1),
        )

    @staticmethod
    def _masked_log_prob(dist, actions, mask):
        return (dist.log_prob(actions) * mask).sum(dim=-1, keepdim=True)

    def _audit_handoff(self, policy_obs, bridge_action, step):
        if self.log_dir is None:
            return
        active_ids = (step.state_before == STATE_HR_BRIDGE).nonzero(as_tuple=False).flatten()
        active_ids = active_ids[active_ids < 16]
        if active_ids.numel() == 0:
            return
        path = os.path.join(self.log_dir, "handoff_audit.csv")
        os.makedirs(self.log_dir, exist_ok=True)
        elevation = policy_obs[:, 57:]
        robot = self.env.unwrapped.scene["robot"]
        command = self.env.unwrapped.command_manager.get_command("base_velocity")
        rows = []
        for index in active_ids.tolist():
            row = {
                "env": index,
                "s1_elapsed_steps": int(self.tracker.elapsed_steps[index]),
                "step_up": float(step.riser["riser_height"][index, 3]),
                "riser_distance": float(step.riser["riser_distance"][index, 3]),
                "command_x": float(command[index, 0]),
                "base_ang_x": float(robot.data.root_ang_vel_b[index, 0]),
                "base_ang_y": float(robot.data.root_ang_vel_b[index, 1]),
                "base_ang_z": float(robot.data.root_ang_vel_b[index, 2]),
                "gravity_x": float(robot.data.projected_gravity_b[index, 0]),
                "gravity_y": float(robot.data.projected_gravity_b[index, 1]),
                "gravity_z": float(robot.data.projected_gravity_b[index, 2]),
                "bridge_a0": float(bridge_action[index, 0]),
                "bridge_a1": float(bridge_action[index, 1]),
                "bridge_a2": float(bridge_action[index, 2]),
                "hr_q0": float(policy_obs[index, 18]),
                "hr_q1": float(policy_obs[index, 19]),
                "hr_q2": float(policy_obs[index, 20]),
                "hr_dq0": float(policy_obs[index, 34]),
                "hr_dq1": float(policy_obs[index, 35]),
                "hr_dq2": float(policy_obs[index, 36]),
                "teacher_hr_actual_z": float(robot.data.body_pos_w[index, self.tracker.wheel_body_ids[3], 2]),
                "teacher_clearance": int(self.tracker.clearance_latched[index]),
                "teacher_crossing": int(self.tracker.crossing_latched[index]),
                "teacher_landing": int(self.tracker.landing_latched[index]),
                "teacher_support": int(self.tracker.support_latched[index]),
                "teacher_contact_miss": int(self.tracker.miss_latched[index]),
            }
            for action_index in range(16):
                row[f"previous_final_action_{action_index}"] = float(policy_obs[index, 41 + action_index])
            for column in range(315):
                row[f"elevation_{column}"] = float(elevation[index, column])
            rows.append(row)
        with open(path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if not self._audit_header_written and os.path.getsize(path) == 0:
                writer.writeheader()
            writer.writerows(rows)
        self._audit_header_written = True

    def _collect(self, obs):
        storage = self._allocate()
        for t in range(self.num_steps_per_env):
            policy_obs = obs["policy"]
            critic_obs = obs["critic"]
            state, riser, activation = self.tracker.pre_action_state()
            base_dist = self.policy.base_actor.distribution(policy_obs[:, :57])
            bridge_dist = self.policy.bridge_actor.distribution(policy_obs)
            base_action = base_dist.sample()
            bridge_action = bridge_dist.sample()
            final_action = compose_final_action(
                base_action, bridge_action, state, self.previous_final_action, self.delta_limit
            )
            base_mask = base_action_authority_mask(state)
            bridge_mask = bridge_actor_mask(state)
            base_value = self.policy.base_critic(critic_obs)
            bridge_value = self.policy.bridge_critic(policy_obs)
            next_obs, official_reward, dones, extras = self.env.step(final_action.to(self.env.device))
            next_obs = next_obs.to(self.device)
            dones = dones.to(self.device).reshape(-1, 1)
            official_reward = official_reward.to(self.device).reshape(-1, 1)
            step = self.tracker.post_step(state, riser, activation)
            base_reward = official_reward + step.joint_success_event[:, None] * self.joint_success_weight
            bridge_reward = (
                step.success_event[:, None] * self.hr_success_weight
                - step.miss_event[:, None] * self.perception_miss_weight
                + step.joint_success_event[:, None] * self.joint_success_weight
            )
            storage.base_obs[t] = policy_obs[:, :57]
            storage.bridge_obs[t] = policy_obs
            storage.critic_obs[t] = critic_obs
            storage.base_actions[t] = base_action
            storage.bridge_actions[t] = bridge_action
            storage.base_log_prob_dim[t] = base_dist.log_prob(base_action)
            storage.bridge_log_prob_dim[t] = bridge_dist.log_prob(bridge_action)
            storage.base_action_mask[t] = base_mask
            storage.bridge_actor_mask[t] = bridge_mask
            storage.base_values[t] = base_value
            storage.bridge_values[t] = bridge_value
            storage.base_rewards[t] = base_reward
            storage.bridge_rewards[t] = bridge_reward
            storage.dones[t] = dones
            self._audit_handoff(policy_obs, bridge_action, step)
            self.previous_final_action = final_action.detach()
            self.previous_final_action[dones[:, 0].bool()] = 0.0
            self.tracker.reset(dones)
            obs = next_obs
        return obs, storage

    def _gae(self, rewards, values, dones, last_values):
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_values)
        for t in reversed(range(self.num_steps_per_env)):
            next_value = last_values if t == self.num_steps_per_env - 1 else values[t + 1]
            not_done = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * next_value * not_done - values[t]
            gae = delta + self.gamma * self.lam * not_done * gae
            advantages[t] = gae
        returns = advantages + values
        return returns, advantages

    def _update(self, storage):
        total = self.num_steps_per_env * self.env.num_envs
        batch_size = total // self.num_mini_batches
        flat = lambda x: x.reshape(total, *x.shape[2:])
        tensors = {name: flat(getattr(storage, name)) for name in storage.__dataclass_fields__}
        losses = {"base_surrogate": 0.0, "bridge_surrogate": 0.0, "value": 0.0}
        updates = 0
        for _ in range(self.num_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, batch_size):
                ids = permutation[start : start + batch_size]
                bdist = self.policy.base_actor.distribution(tensors["base_obs"][ids])
                hdist = self.policy.bridge_actor.distribution(tensors["bridge_obs"][ids])
                base_mask = tensors["base_action_mask"][ids]
                bridge_mask = tensors["bridge_actor_mask"][ids, 0]
                old_base_lp = (tensors["base_log_prob_dim"][ids] * base_mask).sum(-1)
                new_base_lp = (bdist.log_prob(tensors["base_actions"][ids]) * base_mask).sum(-1)
                base_adv = tensors["base_advantages"][ids, 0]
                base_adv = (base_adv - base_adv.mean()) / (base_adv.std(unbiased=False) + 1e-8)
                ratio = (new_base_lp - old_base_lp).exp()
                base_loss = -torch.minimum(
                    ratio * base_adv,
                    ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * base_adv,
                ).mean()
                active = bridge_mask > 0.5
                if active.any():
                    old_hlp = tensors["bridge_log_prob_dim"][ids][active].sum(-1)
                    new_hlp = hdist.log_prob(tensors["bridge_actions"][ids])[active].sum(-1)
                    hadv = tensors["bridge_advantages"][ids, 0][active]
                    hadv = (hadv - hadv.mean()) / (hadv.std(unbiased=False) + 1e-8)
                    hratio = (new_hlp - old_hlp).exp()
                    bridge_loss = -torch.minimum(
                        hratio * hadv,
                        hratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * hadv,
                    ).mean()
                    bridge_entropy = hdist.entropy()[active].sum(-1).mean()
                else:
                    bridge_loss = torch.zeros((), device=self.device)
                    bridge_entropy = torch.zeros((), device=self.device)
                base_value = self.policy.base_critic(tensors["critic_obs"][ids])
                bridge_value = self.policy.bridge_critic(tensors["bridge_obs"][ids])
                value_loss = (
                    (base_value - tensors["base_returns"][ids]).square().mean()
                    + (bridge_value - tensors["bridge_returns"][ids]).square().mean()
                )
                base_entropy = (bdist.entropy() * base_mask).sum(-1).mean()
                loss = base_loss + bridge_loss + self.value_loss_coef * value_loss
                loss -= self.entropy_coef * (base_entropy + bridge_entropy)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses["base_surrogate"] += float(base_loss.detach())
                losses["bridge_surrogate"] += float(bridge_loss.detach())
                losses["value"] += float(value_loss.detach())
                updates += 1
        return {key: value / max(updates, 1) for key, value in losses.items()}

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        formal_ready = (
            self.rewards_calibrated and self.handoff_audit_passed and self.formal_training_enabled
        )
        if not formal_ready and num_learning_iterations > 2:
            raise RuntimeError(
                "HR Bridge formal gates are not all passed; only <=2 iteration smoke runs are allowed"
            )
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir else None
        obs = self.env.get_observations().to(self.device)
        for iteration in range(num_learning_iterations):
            started = time.time()
            with torch.no_grad():
                obs, storage = self._collect(obs)
                storage.base_returns, storage.base_advantages = self._gae(
                    storage.base_rewards, storage.base_values, storage.dones,
                    self.policy.base_critic(obs["critic"]),
                )
                storage.bridge_returns, storage.bridge_advantages = self._gae(
                    storage.bridge_rewards, storage.bridge_values, storage.dones,
                    self.policy.bridge_critic(obs["policy"]),
                )
            losses = self._update(storage)
            self.current_learning_iteration = iteration
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            fps = int(self.num_steps_per_env * self.env.num_envs / max(time.time() - started, 1e-6))
            print(
                f"HRBridge iteration={iteration} fps={fps} "
                f"base_loss={losses['base_surrogate']:.5f} "
                f"bridge_loss={losses['bridge_surrogate']:.5f}", flush=True
            )
            if self.writer:
                for key, value in losses.items():
                    self.writer.add_scalar(f"Loss/{key}", value, iteration)
                self.writer.add_scalar("Perf/total_fps", fps, iteration)
                self.writer.add_scalar("State/S1_fraction", storage.bridge_actor_mask.mean(), iteration)
            if self.log_dir and iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path, infos=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            "architecture": "M20-Stair-HRBridge-v0",
        }, path)

    def load(self, path):
        loaded = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(loaded["model_state_dict"])
        self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
        self.current_learning_iteration = int(loaded.get("iter", 0))
