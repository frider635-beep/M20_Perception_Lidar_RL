"""Information-isolated actor-critic for M20 V10 unified perception."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class UnifiedActor(nn.Module):
    """60D actor whose H/D inputs have a structural path only to FR actions."""

    def __init__(self, hidden_dims=(512, 256, 128), activation="elu"):
        super().__init__()
        if list(hidden_dims) != [512, 256, 128]:
            raise ValueError(f"V10 shared trunk must be [512, 256, 128], got {hidden_dims}")
        self.shared_trunk = MLP(58, 128, [512, 256], activation)
        self.support_head = nn.Linear(128, 13)
        self.fr_head = MLP(130, 3, [128], activation)
        self._export_input = SimpleNamespace(in_features=60)

    def __getitem__(self, index: int):
        # Isaac Lab's generic exporter only queries actor[0].in_features.
        if index == 0:
            return self._export_input
        raise IndexError(index)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 60:
            raise RuntimeError(f"V10 actor requires 60D input, got {observation.shape[-1]}")
        proprio, b = observation[..., :57], observation[..., 57:58]
        height_distance = observation[..., 58:60]
        latent = self.shared_trunk(torch.cat((proprio, b), dim=-1))
        support = self.support_head(latent)
        fr = self.fr_head(torch.cat((latent, height_distance), dim=-1))
        return torch.cat((support[..., :3], fr, support[..., 3:13]), dim=-1)


class V10UnifiedActorCritic(nn.Module):
    """RSL-RL compatible asymmetric actor-critic with a 60D deployable actor."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="log",
        state_dependent_std=False,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        if kwargs:
            print("V10UnifiedActorCritic ignored arguments: " + str(sorted(kwargs)))
        if state_dependent_std:
            raise ValueError("V10 does not support state-dependent action standard deviation")
        self.obs_groups = obs_groups
        actor_dim = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        critic_dim = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        if actor_dim != 60 or num_actions != 16:
            raise RuntimeError(f"V10 dimensions must be actor=60/action=16, got {actor_dim}/{num_actions}")
        self.actor = UnifiedActor(actor_hidden_dims, activation)
        self.critic = MLP(critic_dim, 1, critic_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(actor_dim) if actor_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(critic_dim) if critic_obs_normalization else nn.Identity()
        self.noise_std_type = noise_std_type
        if noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        elif noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:
            raise ValueError(f"unsupported noise_std_type={noise_std_type}")
        self.distribution = None
        Normal.set_default_validate_args(False)
        self.actor_obs_dim = actor_dim
        self.critic_obs_dim = critic_dim
        self.action_dim = num_actions

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actor_obs(self, obs: TensorDict):
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict):
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def _update_distribution(self, observation):
        mean = self.actor(observation)
        std = torch.exp(self.log_std) if self.noise_std_type == "log" else self.std
        self.distribution = Normal(mean, std.expand_as(mean))

    def act(self, obs: TensorDict, **kwargs):
        observation = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self._update_distribution(observation)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict):
        observation = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(observation)

    def evaluate(self, obs: TensorDict, **kwargs):
        observation = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(observation)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


__all__ = ["UnifiedActor", "V10UnifiedActorCritic"]
