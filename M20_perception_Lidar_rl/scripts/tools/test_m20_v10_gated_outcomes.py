#!/usr/bin/env python3
"""Static and pure-tensor audit for gated-opportunity outcome diagnostics."""

import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py"
RUNNER = ROOT / "source/rl_training/rl_training/v10_unified_runner.py"
if not SOURCE.is_file():
    ROOT = Path(__file__).resolve().parents[1]
    SOURCE = ROOT / "source/v10_unified.py"
    RUNNER = ROOT / "source/v10_unified_runner.py"

source = SOURCE.read_text()
runner = RUNNER.read_text()

tree = ast.parse(source)
function_node = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "v10_gated_outcome_masks"
)
namespace = {"torch": torch}
exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
classify = namespace["v10_gated_outcome_masks"]

gated = torch.ones(6, dtype=torch.bool)
classified = classify(
    gated,
    torch.tensor([True, False, False, False, False, False]),
    torch.tensor([False, False, True, True, True, True]),
    torch.tensor([False, False, False, True, True, True]),
    torch.tensor([False, False, False, False, True, True]),
    torch.tensor([False, False, False, False, False, True]),
)
expected_indices = (0, 1, 2, 3, 5, 4)
pure_classification_pass = all(bool(classified[index][sample]) for sample, index in enumerate(expected_indices))
stacked = torch.stack(classified)
mutually_exclusive = bool((stacked.sum(dim=0) == 1).all())

outcomes = (
    "vertical_contact_before_clearance",
    "clearance_not_reached",
    "clearance_reached_but_no_crossing",
    "crossing_reached_but_no_upper_support",
    "first_success",
)
tags = (
    "VerticalContactBeforeClearance",
    "ClearanceNotReached",
    "ClearanceReachedButNoCrossing",
    "CrossingReachedButNoUpperSupport",
    "FirstSuccess",
)

results = {
    "DIAGNOSTIC_SCOPE_LEVEL2_OFFICIAL_UP": all(token in source for token in (
        "diagnostic_scope = is_stair & (terrain_level >= 2)",
        "gated_stair_up_opportunity = official_up & diagnostic_scope",
    )),
    "OFFICIAL_MOVE_UP_UNCONDITIONAL": (
        "move_up = official_up" in source
        and "move_up = official_up &" not in source
    ),
    "FIVE_OUTCOMES_DEFINED": all(f'"{name}"' in source for name in outcomes),
    "VERTICAL_BEFORE_CLEARANCE_LATCHED": all(token in source for token in (
        '"vertical_contact_before_clearance_latched"',
        'vertical_contact & ~state["precontact_clearance_latched"]',
    )),
    "OUTCOMES_SETTLED_AT_CURRICULUM": all(token in source for token in (
        "v10_gated_outcome_masks(",
        "gated_stair_up_opportunity,",
        "_append_gated_recent(",
        'state["max_clearance_progress"][ids]',
    )),
    "RECENT_100_500_PRESENT": all(token in runner for token in (
        "for window in (100, 500):",
        "V10_Curriculum/GatedOpportunity/Recent{window}",
    )),
    "FIVE_TENSORBOARD_OUTCOMES": all(label in runner for label in tags),
    "UNCLASSIFIED_INTEGRITY_METRIC": "UnclassifiedRate" in runner,
    "PURE_CLASSIFICATION_CASES": pure_classification_pass,
    "OUTCOMES_MUTUALLY_EXCLUSIVE": mutually_exclusive,
    "NO_REWARD_OR_POLICY_WIRING": all(token not in source for token in (
        "gated_outcome_reward", "gated_outcome_observation",
    )),
}

for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if all(results.values()) else 1)
