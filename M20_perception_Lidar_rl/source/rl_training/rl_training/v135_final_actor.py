"""V13.5-Final actor: native MLP plus a GRU64 front-leg residual."""
from __future__ import annotations

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules import RNN
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class V135ResidualGRUModel(MLPModel):
    """RSL-RL 5.x stochastic model with a bounded recurrent residual."""
    is_recurrent = True

    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
                 output_dim: int, hidden_dims=(512, 256, 128), activation: str = "elu",
                 obs_normalization: bool = False, distribution_cfg: dict | None = None,
                 gru_input_dim: int = 32, gru_hidden_dim: int = 64,
                 residual_scale: float = 0.20) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg)
        if output_dim < 6:
            raise ValueError("V135ResidualGRUModel requires at least six actions")
        self.gru_input = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dim, gru_input_dim), resolve_nn_activation(activation)
        )
        self.rnn = RNN(gru_input_dim, gru_hidden_dim, 1, "gru")
        self.gru_output = torch.nn.Linear(gru_hidden_dim, 6)
        torch.nn.init.normal_(self.gru_output.weight, mean=0.0, std=1.0e-3)
        torch.nn.init.zeros_(self.gru_output.bias)
        self.residual_scale = float(residual_scale)
        self._inactive_steps = None

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        padded_latent = super().get_latent(obs)
        # RSL-RL pads complete recurrent sequences during PPO updates.  The
        # GRU consumes that padded layout, while both action branches must use
        # the same unpadded valid-transition order returned by RNN.forward().
        latent = unpad_trajectories(padded_latent, masks) if masks is not None else padded_latent
        parameters = self.mlp(latent)
        gate = ((latent[..., 57] > 0.5) & (latent[..., 58] > 0.03)).to(latent.dtype)
        # Rollout-only lifecycle: after three consecutive non-ascent frames,
        # erase the stored gait phase. Recurrent update batches use their
        # recorded initial hidden state and masks, so this must not mutate it.
        if masks is None and latent.ndim == 2:
            inactive = gate == 0
            if self._inactive_steps is None or self._inactive_steps.shape != inactive.shape:
                self._inactive_steps = torch.zeros_like(inactive, dtype=torch.long)
            self._inactive_steps = torch.where(
                inactive, self._inactive_steps + 1, torch.zeros_like(self._inactive_steps)
            )
            clear = self._inactive_steps >= 3
            if clear.any() and self.rnn.hidden_state is not None:
                self.rnn.reset(clear)
        recurrent = self.rnn(self.gru_input(padded_latent), masks, hidden_state).squeeze(0)
        residual = self.residual_scale * torch.tanh(self.gru_output(recurrent))
        parameters = parameters.clone()
        parameters[..., :6] = parameters[..., :6] + gate.unsqueeze(-1) * residual
        if self.distribution is None:
            return parameters
        if stochastic_output:
            self.distribution.update(parameters)
            return self.distribution.sample()
        return self.distribution.deterministic_output(parameters)

    def reset(self, dones=None, hidden_state=None):
        self.rnn.reset(dones, hidden_state)
        if self._inactive_steps is not None:
            if dones is None:
                self._inactive_steps.zero_()
            else:
                self._inactive_steps[dones == 1] = 0

    def get_hidden_state(self):
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones=None):
        self.rnn.detach_hidden_state(dones)


__all__ = ["V135ResidualGRUModel"]
