#!/usr/bin/env python3
"""Static and pure-math gates for V10 height-conditioned FR diagnostics."""

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
node = next(
    item for item in tree.body
    if isinstance(item, ast.FunctionDef) and item.name == "v10_recent_window_mean"
)
namespace = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
recent_mean = namespace["v10_recent_window_mean"]

ring = {
    "sequence": torch.tensor([0, 1, 2, 3, 4, 5]),
    "bucket": torch.tensor([0, 0, 1, 1, 2, 2]),
    "value": torch.tensor([1.0, 3.0, 10.0, 14.0, 20.0, 30.0]),
}

results = {
    "RECENT_MEAN_BUCKET_FILTER": abs(float(recent_mean(ring, 6, 500, "value", 1)) - 12.0) < 1e-6,
    "RECENT_MEAN_WINDOW_FILTER": abs(float(recent_mean(ring, 6, 2, "value")) - 25.0) < 1e-6,
    "ACTION_IS_FR_3_TO_6": "env.action_manager.action[:, 3:6]" in source,
    "ACTION_SCOPED_PRE_CROSSING": "clearance_active, fr_action_amplitude" in source,
    "ABSOLUTE_LIFT_USES_LOWER_GROUND": (
        "wheel_bottom - state[\"target_lower_z\"]" in source
    ),
    "HEIGHT_LOCKED_PER_ENCOUNTER": (
        'ring["height"][keep_slots] = state["locked_h"][keep_ids]' in source
    ),
    "CLEARANCE_AND_FIRST_SHARE_RING": (
        '"max_clearance_progress"' in source and 'field="first_success"' in runner
    ),
    "SIX_BUCKET_OUTPUTS": (
        'prefix = f"V10_HeightJoint/Recent500/{bucket_label}"' in runner
    ),
    "SIMULATION_ONLY_COMMENT": "simulation-only" in runner,
}

for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if all(results.values()) else 1)
