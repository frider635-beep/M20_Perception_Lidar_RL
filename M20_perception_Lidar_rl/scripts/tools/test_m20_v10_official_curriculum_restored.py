#!/usr/bin/env python3
"""Audit that V10 terrain promotion is the official distance-only curriculum."""

import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py"
RUNNER = ROOT / "source/rl_training/rl_training/v10_unified_runner.py"
CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v10_unified_env_cfg.py"
if not SOURCE.is_file():
    ROOT = Path(__file__).resolve().parents[1]
    SOURCE = ROOT / "source/v10_unified.py"
    RUNNER = ROOT / "source/v10_unified_runner.py"
    CFG = ROOT / "source/v10_unified_env_cfg.py"

source = SOURCE.read_text()
runner = RUNNER.read_text()
cfg = CFG.read_text()
tree = ast.parse(source)
node = next(
    item for item in tree.body
    if isinstance(item, ast.FunctionDef) and item.name == "v10_official_promotion_diagnostics"
)
namespace = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
promote = namespace["v10_official_promotion_diagnostics"]

official_up = torch.tensor([True, True, True, True, False, False])
official_down = torch.tensor([False, False, False, False, True, False])
is_stair = torch.tensor([True, True, True, True, True, False])
level = torch.tensor([0, 1, 2, 9, 9, 9])
first = torch.tensor([False, False, False, False, False, False])
move_up, move_down, _, _, diagnostic_opportunity, would_fail = promote(
    official_up, official_down, is_stair, level, first
)

results = {
    "MOVE_UP_EQUALS_OFFICIAL_UP": bool(torch.equal(move_up, official_up)),
    "MOVE_DOWN_EQUALS_OFFICIAL_DOWN": bool(torch.equal(move_down, official_down)),
    "LEVEL_GE2_FIRST_FALSE_STILL_PROMOTES": bool(move_up[2] and move_up[3]),
    "FIRST_NOT_IN_PROMOTION_EXPRESSION": "move_up = official_up &" not in source,
    "OLD_GATE_CONFIG_REMOVED": "perception_gate_start_level" not in cfg,
    "OFFICIAL_CURRICULUM_REGISTERED": (
        "CurrTerm(func=mdp.v10_official_terrain_levels)" in cfg
    ),
    "LEVEL2_DIAGNOSTICS_RETAINED": (
        diagnostic_opportunity.tolist() == [False, False, True, True, False, False]
        and would_fail.tolist() == [False, False, True, True, False, False]
    ),
    "BLOCKED_RATE_FORCED_ZERO": (
        "torch.zeros_like(stair_gated_official_up_count)" in runner
    ),
    "WOULD_FAIL_DIAGNOSTIC_PRESENT": (
        "V10_Curriculum/WouldFailFirstAtOfficialUpRate" in runner
    ),
}

for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if all(results.values()) else 1)
