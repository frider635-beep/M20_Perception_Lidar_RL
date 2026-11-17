"""V11 continuous-stair actor-critic.

Unlike V10, no observation field is structurally routed to a particular leg.
The policy consumes the deployable 63D two-riser ABI and produces all 16 M20
actions from a shared representation.
"""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class V11ContinuousActorCritic(nn.Module):
    """RSL-RL actor-critic with a 63D deployable, leg-neutral actor."""

    is_recurrent = False

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], num_actions: int,
        actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128),
        activation="elu", init_noise_std=1.0, noise_std_type="log",
        actor_obs_normalization=False, critic_obs_normalization=False, **kwargs: dict[str, Any],
    ):
        super().__init__()
        if kwargs:
            print("V11ContinuousActorCritic ignored arguments: " + str(sorted(kwargs)))
        self.obs_groups = obs_groups
        self.actor_obs_dim = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        self.critic_obs_dim = sum(obs[g].shape[-1] for g in obs_groups["critic"])
        self.action_dim = num_actions
        if (self.actor_obs_dim, self.action_dim) != (63, 16):
            raise RuntimeError(f"V11 dimensions must be actor=63/action=16, got {self.actor_obs_dim}/{self.action_dim}")
        self.actor = MLP(self.actor_obs_dim, self.action_dim, actor_hidden_dims, activation)
        self.critic = MLP(self.critic_obs_dim, 1, critic_hidden_dims, activation)
        self.actor_obs_normalizer = EmpiricalNormalization(self.actor_obs_dim) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(self.critic_obs_dim) if critic_obs_normalization else nn.Identity()
        if noise_std_type != "log":
            raise ValueError("V11 requires log action standard deviation")
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.action_dim)))
        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _actor_obs(self, obs):
        return torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)

    def _critic_obs(self, obs):
        return torch.cat([obs[g] for g in self.obs_groups["critic"]], dim=-1)

    def _update_distribution(self, observation):
        mean = self.actor(observation)
        self.distribution = Normal(mean, torch.exp(self.log_std).expand_as(mean))

    @property
    def action_mean(self): return self.distribution.mean
    @property
    def action_std(self): return self.distribution.stddev
    @property
    def entropy(self): return self.distribution.entropy().sum(dim=-1)
    def act(self, obs, **kwargs):
        self._update_distribution(self.actor_obs_normalizer(self._actor_obs(obs)))
        return self.distribution.sample()
    def act_inference(self, obs):
        return self.actor(self.actor_obs_normalizer(self._actor_obs(obs)))
    def evaluate(self, obs, **kwargs):
        return self.critic(self.critic_obs_normalizer(self._critic_obs(obs)))
    def get_actions_log_prob(self, actions): return self.distribution.log_prob(actions).sum(dim=-1)


__all__ = ["V11ContinuousActorCritic"]
