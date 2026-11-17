#!/usr/bin/env python3
"""Regression checks for V12 bilateral front/rear stair completion rewards."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
LANDING = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_landing.py"
CONTINUOUS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py"
REWARDS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v12_teacher_rewards.py"
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v12_teacher_env_cfg.py"

spec = importlib.util.spec_from_file_location("v11_landing_bilateral_pure", LANDING)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

latched = torch.zeros((1, 2), dtype=torch.bool)
age = torch.zeros(1, dtype=torch.long)
active = torch.tensor([True])

# Rear wheels may support sequentially, but both contacts must occur inside
# the configured window before platform completion becomes eligible.
first = module.v11_update_bilateral_support_window(
    latched, age, torch.tensor([[True, False]]), active, 3,
)
assert not first["ready"].item() and first["age"].item() == 1
second = module.v11_update_bilateral_support_window(
    first["latched"], first["age"], torch.tensor([[False, True]]), active, 3,
)
assert second["ready"].item() and second["age"].item() == 0

# A stale one-sided contact expires instead of being retained indefinitely.
state = {"latched": latched, "age": age}
for _ in range(3):
    state = module.v11_update_bilateral_support_window(
        state["latched"], state["age"], torch.tensor([[True, False]]), active, 2,
    )
assert not state["ready"].item() and state["age"].item() == 1

continuous = CONTINUOUS.read_text(encoding="utf-8")
rewards = REWARDS.read_text(encoding="utf-8")
config = CONFIG.read_text(encoding="utf-8")

# Execute the actual pure bounding helper without importing Isaac Lab.  The
# penalty must preserve a useful near-zero gradient while never reaching or
# exceeding its fixed upper bound for arbitrarily long phase gaps.
reward_tree = ast.parse(rewards)
bounded_helper = next(
    node for node in reward_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "_bounded_squared_penalty"
)
helper_namespace = {"torch": torch}
exec(compile(ast.Module(body=[bounded_helper], type_ignores=[]), str(REWARDS), "exec"), helper_namespace)
bounded_squared_penalty = helper_namespace["_bounded_squared_penalty"]
bounded = bounded_squared_penalty(torch.tensor([0.0, 0.5, 1.0, 10.0, 1.0e6]))
assert bounded[0].item() == 0.0
assert torch.all(bounded[1:4] > bounded[:3])
assert torch.all(bounded[1:] >= bounded[:-1])
assert torch.all(bounded >= 0.0) and torch.all(bounded <= 1.0)
assert bounded[-1].item() == 1.0

assert '"bilateral_landing_event"' in continuous
assert '"second_clearance_settlement"' in continuous
assert 'state["candidate_ready"].all(dim=1)' in continuous
assert "v11_update_bilateral_support_window" in continuous
assert "v12_second_front_completion_reward" in rewards
assert "v12_second_front_clearance_settlement_reward" in rewards
assert "v12_bilateral_wheel_height_difference" in rewards
assert "v12_front_phase_time_gap" in rewards
assert "return _bounded_squared_penalty(excess)" in rewards
assert "second_front_completion = RewTerm" in config
assert "second_front_clearance_settlement = RewTerm" in config
assert "bilateral_wheel_height_difference = RewTerm" in config
assert "front_crossing_time_gap = RewTerm" in config
assert "front_support_time_gap = RewTerm" in config
assert "require_bilateral_front_completion = True" in config
assert 'front_completion_mode = "bilateral_same_riser"' in config
assert "self.curriculum.v11_rewarded_alternation_rate = None" in config
assert "require_bilateral_rear_support = True" in config

print("V12_BILATERAL_STAIR_REWARD_REGRESSION: PASS")
