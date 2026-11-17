"""V9-compatible soft mixture-of-experts model for the deployable V12 policy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_nn_activation


class MoeLayer(nn.Module):
    """Evaluate every expert and blend their outputs with a learned softmax gate."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_hidden_dims: list[int],
        gate_hidden_dims: list[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.gate = self._build_mlp(input_dim, self.num_experts, gate_hidden_dims or [], activation)
        self.experts = nn.ModuleList(
            [self._build_mlp(input_dim, output_dim, expert_hidden_dims, activation) for _ in range(self.num_experts)]
        )
        self.last_gate_scores = torch.empty(0)

    @staticmethod
    def _build_mlp(input_dim: int, output_dim: int, hidden_dims: list[int], activation: str) -> nn.Sequential:
        layers: list[nn.Module] = []
        activation_mod = resolve_nn_activation(activation)
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_mod)
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_scores = F.softmax(self.gate(inputs), dim=-1)
        if not torch.jit.is_scripting():
            self.last_gate_scores = gate_scores
        expert_outputs = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        return torch.einsum("be,beo->bo", gate_scores, expert_outputs)


class V12TeacherMoEModel(MLPModel):
    """RSL-RL 5 model matching the V9 teacher's four-expert model contract."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(256, 256, 256),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        num_moe_experts: int = 4,
        moe_gate_hidden_dims: list[int] | None = None,
        output_init_scale: float | None = None,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        model_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MoeLayer(
            self._get_latent_dim(),
            model_output_dim,
            num_experts=num_moe_experts,
            expert_hidden_dims=list(hidden_dims),
            gate_hidden_dims=moe_gate_hidden_dims,
            activation=activation,
        )
        if output_init_scale is not None:
            for expert in self.mlp.experts:
                output_layer = next(module for module in reversed(expert) if isinstance(module, nn.Linear))
                nn.init.orthogonal_(output_layer.weight, gain=output_init_scale)
                nn.init.zeros_(output_layer.bias)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


__all__ = ["MoeLayer", "V12TeacherMoEModel"]
