#!/usr/bin/env python3
"""CPU contract for V13's critic-only six-value reward state."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v13_alternation.py"
spec = importlib.util.spec_from_file_location("v13_alternation_contract", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

encoded = module.v13_encode_critic_reward_state(
    torch.tensor([-1, 0, 1]),
    torch.tensor([False, True, True]),
    torch.tensor([False, True, True]),
    torch.tensor([[0.9, 0.8], [0.2, 0.7], [0.3, 0.6]]),
    torch.tensor([[1.0, 1.0], [0.4, 0.8], [0.5, 0.9]]),
)
torch.testing.assert_close(encoded, torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 0.2, 0.4],
    [0.0, 1.0, 1.0, 1.0, 0.6, 0.9],
]))
assert encoded.shape == (3, 6) and torch.isfinite(encoded).all()

cfg_text = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v13_critic6d_env_cfg.py").read_text()
assert "DeeproboticsM20V13V11FDAlternationEnvCfg" in cfg_text
assert "observations.critic.v13_reward_state" in cfg_text
assert "observations.policy" not in cfg_text
assert "M20V133RewardsCfg" not in cfg_text and "v133_" not in cfg_text.lower()

print("V13_CRITIC6D_CONTRACT: PASS")
