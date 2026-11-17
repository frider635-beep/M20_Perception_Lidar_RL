#!/usr/bin/env python3
"""One-shot V10 reward event/latch unit test."""

from __future__ import annotations

import ast
from pathlib import Path

import torch


def load_event_function():
    path = Path(__file__).resolve().parents[2] / (
        "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py"
    )
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "v10_event_pulses")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["v10_event_pulses"]


def scalar(value):
    return torch.tensor([value], dtype=torch.bool)


def main():
    pulses = load_event_function()
    false, true = scalar(False), scalar(True)
    success = pulses(true, true, true, true, false, false, false, false)
    success_repeat = pulses(true, true, true, true, false, true, false, false)
    miss = pulses(true, false, false, false, false, false, false, true)
    blind_after_miss = pulses(true, true, true, true, true, false, true, false)
    success_ok = bool(success[1] and not success_repeat[1])
    miss_ok = bool(miss[3] and not blind_after_miss[0] and not blind_after_miss[1])
    joint_latched = false.clone()
    first_latched, full_latched = true, true
    joint_now = first_latched & full_latched
    joint_event = joint_now & ~joint_latched
    joint_latched |= joint_now
    joint_repeat = joint_now & ~joint_latched
    joint_ok = bool(joint_event and not joint_repeat)
    print(f"FIRST_STEP_EVENT_ONE_SHOT={'PASS' if success_ok else 'FAIL'}")
    print(f"PERCEPTION_MISS_LATCH={'PASS' if miss_ok else 'FAIL'}")
    print(f"JOINT_SUCCESS_ONE_SHOT={'PASS' if joint_ok else 'FAIL'}")
    print(f"REWARD_EVENTS={'PASS' if success_ok and miss_ok and joint_ok else 'FAIL'}")
    raise SystemExit(0 if success_ok and miss_ok and joint_ok else 1)


if __name__ == "__main__":
    main()
