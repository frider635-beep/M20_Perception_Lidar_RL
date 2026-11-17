"""RSL-RL recurrent model compatibility for inference-mode rollouts."""

from __future__ import annotations

import torch
from rsl_rl.models import RNNModel


class V135RNNModel(RNNModel):
    """GRU model whose done reset does not mutate an inference tensor in place.

    RSL-RL 5.x collects rollouts under ``torch.inference_mode``. Its bundled
    RNN reset writes into the hidden tensor in place after leaving the GRU
    forward call, which PyTorch rejects. Replacing only that write with an
    out-of-place mask preserves the official recurrent training and exports.
    """

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        if dones is None:
            self.rnn.reset(dones, hidden_state)
            return
        if hidden_state is not None:
            raise NotImplementedError("V13.5 does not reset done environments to a custom hidden state")
        state = self.rnn.hidden_state
        if state is None:
            return
        keep = (dones != 1).to(dtype=torch.float32)[None, :, None]
        if isinstance(state, tuple):
            self.rnn.hidden_state = tuple(value * keep.to(value.dtype) for value in state)
        else:
            self.rnn.hidden_state = state * keep.to(state.dtype)


__all__ = ["V135RNNModel"]
