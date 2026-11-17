#!/usr/bin/env python3
"""Static audit for simulator-only first-riser lifecycle distance diagnostics."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py"
RUNNER = ROOT / "source/rl_training/rl_training/v10_unified_runner.py"
if not SOURCE.is_file():
    ROOT = Path(__file__).resolve().parents[1]
    SOURCE = ROOT / "source/v10_unified.py"
    RUNNER = ROOT / "source/v10_unified_runner.py"

source = SOURCE.read_text()
runner = RUNNER.read_text()

results = {
    "D_FIRST_VALID_LATCHED": all(token in source for token in (
        'first_valid_event = detected_valid & ~state["first_valid_seen"]',
        'state["d_first_valid"] = torch.where(',
    )),
    "D_FIRST_LOCK_LATCHED": (
        'state["d_first_lock"] = torch.where(lock_event, detected_d' in source
    ),
    "D_FIRST_B_ON_LATCHED": all(token in source for token in (
        'first_b_on_event = b_raw & ~state["b_on_latched"]',
        'state["d_b_on"] = torch.where(',
    )),
    "DISTANCES_FILTERED_TO_UPSTAIRS": all(token in source for token in (
        "first_valid_event & full_topology",
        "lock_event & full_topology",
        "first_b_on_event & full_topology",
    )),
    "FOUR_PRE_B_ON_BLOCKERS": all(token in source for token in (
        "no_valid_riser =", "not_locked =", "d_out_of_range =", "command_not_forward =",
    )),
    "BLOCKERS_ONLY_BEFORE_FIRST_B_ON": (
        'full_topology & ~state["b_on_latched"] & ~b_raw' in source
    ),
    "TENSORBOARD_DISTANCE_TAGS": all(tag in runner for tag in (
        "V10_Perception/D_first_valid",
        "V10_Perception/D_first_lock",
        "V10_Perception/D_b_on",
    )),
    "TENSORBOARD_BLOCKER_TAGS": all(label in runner for label in (
        "no_valid_riser", "not_locked", "D_out_of_range", "command_not_forward",
    )),
}

for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if all(results.values()) else 1)
