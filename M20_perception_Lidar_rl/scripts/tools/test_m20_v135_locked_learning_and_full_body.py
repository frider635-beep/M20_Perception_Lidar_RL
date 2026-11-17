#!/usr/bin/env python3
"""Physical-target progress and full-body descent settlement contracts."""

import importlib.util
from pathlib import Path

import torch


root = Path(__file__).resolve().parents[2]
mdp = root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"


def load(name):
    spec = importlib.util.spec_from_file_location(name, mdp / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ascent = load("v135_final_ascent_reward")
descent = load("v135_descent_support_transfer")

# Scanner switching away must not terminate B1 progress while the locked B1
# height and physical target remain valid.
eligible = ascent.v135_final_progress_eligible(
    torch.tensor([True, True, True]),
    torch.tensor([True, True, True]),
    torch.tensor([True, True, False]),
)
assert eligible.tolist() == [True, True, False]

kwargs = dict(crossing_span=0.15, step_width=0.30, wheel_radius=0.10,
              lateral_tolerance=0.15, confirmation_frames=3)
clearance = torch.tensor([[0.8, 0.0]])
crossed = torch.tensor([[True, False]])
forward = torch.tensor([[0.10, -0.10]])
lateral = torch.zeros((1, 2))
no_support = ascent.v135_locked_front_potential(
    clearance, crossed, torch.zeros((1, 2), dtype=torch.long),
    forward, lateral, torch.tensor([True]), **kwargs,
)
with_support = ascent.v135_locked_front_potential(
    clearance, crossed, torch.tensor([[2, 0]]),
    forward, lateral, torch.tensor([True]), **kwargs,
)
future_riser = ascent.v135_locked_front_potential(
    clearance, crossed, torch.zeros((1, 2), dtype=torch.long),
    torch.tensor([[0.36, -0.10]]), lateral, torch.tensor([True]), **kwargs,
)
invalid_height = ascent.v135_locked_front_potential(
    clearance, crossed, torch.zeros((1, 2), dtype=torch.long),
    forward, lateral, torch.tensor([False]), **kwargs,
)
assert with_support[0, 0] > no_support[0, 0] > future_riser[0, 0]
assert future_riser[0, 0] == 0.0
assert invalid_height.eq(0).all()

# A rear-only touchdown, a front wheel unloading, a large pitch, and a
# one-frame contact cannot pay completion. A healthy full-body hold pays once.
steps = torch.zeros(4, dtype=torch.long)
completed = torch.zeros(4, dtype=torch.bool)
rear = torch.tensor([[True, True]] * 4)
front = torch.tensor([[True, True], [True, False], [True, True], [True, True]])
pitch = torch.tensor([0.05, 0.05, 0.40, 0.05])
for frame in range(3):
    steps, completed, event = descent.v135_full_body_completion_step(
        steps, completed, torch.ones(4, dtype=torch.bool),
        torch.ones(4, dtype=torch.bool), rear, front, pitch,
        confirmation_frames=3, pitch_tolerance=0.26,
    )
    assert event.tolist() == ([False, False, False, False] if frame < 2
                            else [True, False, False, True])
steps, completed, event = descent.v135_full_body_completion_step(
    steps, completed, torch.ones(4, dtype=torch.bool),
    torch.ones(4, dtype=torch.bool), rear, front, pitch,
    confirmation_frames=3, pitch_tolerance=0.26,
)
assert not event.any()
# Completion stays latched after unloading and reloading the front wheels.
front[0, 0] = False
steps, completed, event = descent.v135_full_body_completion_step(
    steps, completed, torch.ones(4, dtype=torch.bool),
    torch.ones(4, dtype=torch.bool), rear, front, pitch,
    confirmation_frames=3, pitch_tolerance=0.26,
)
front[0, 0] = True
for _ in range(3):
    steps, completed, event = descent.v135_full_body_completion_step(
        steps, completed, torch.ones(4, dtype=torch.bool),
        torch.ones(4, dtype=torch.bool), rear, front, pitch,
        confirmation_frames=3, pitch_tolerance=0.26,
    )
    assert not event.any()

print("V135_LOCKED_LEARNING_FULL_BODY_CONTRACT: PASS")
