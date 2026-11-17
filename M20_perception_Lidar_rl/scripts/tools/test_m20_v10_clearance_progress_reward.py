#!/usr/bin/env python3
"""Static and pure-tensor audit for the V10 FR clearance-progress reward."""

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
    CFG = ROOT / "source/config/v10_unified_env_cfg.py"

source = SOURCE.read_text()
runner = RUNNER.read_text()
cfg = CFG.read_text()

tree = ast.parse(source)
function_node = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "v10_clearance_progress_step"
)
namespace = {"torch": torch}
exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
step = namespace["v10_clearance_progress_step"]

lower = torch.tensor([0.0])
upper = torch.tensor([0.17])
active = torch.tensor([True])
maximum = torch.tensor([0.0])
deltas = []
for requested_progress in (0.10, 0.30, 0.20, 1.00):
    wheel_bottom = lower + requested_progress * (upper + 0.01 - lower)
    progress, delta, maximum = step(wheel_bottom, lower, upper, 0.01, active, maximum)
    deltas.append(delta)
total_delta = torch.stack(deltas).sum()
actual_reward = total_delta / 0.02 * 2.0 * 0.02

inactive_progress, inactive_delta, inactive_max = step(
    torch.tensor([0.18]), lower, upper, 0.01, torch.tensor([False]), torch.tensor([0.4])
)

results = {
    "OFFICIAL_ASCENDING_ONLY": 'full_topology & state["target_valid"]' in source,
    "REQUIRES_LOCK_AND_B": all(token in source for token in (
        'state["first_riser_locked"] & b_raw',
        '& ~state["crossing_latched"] & (fr_progress < 0.0)',
    )),
    "TARGET_IS_UPPER_PLUS_1CM": (
        "target_z = upper_z + float(clearance_margin)" in source
        and "clearance_margin: float = 0.01" in cfg
    ),
    "MONOTONIC_DELTA_SUMS_TO_ONE": bool(torch.isclose(total_delta, torch.tensor(1.0))),
    "TOTAL_REWARD_CAP_IS_TWO": bool(torch.isclose(actual_reward, torch.tensor(2.0))),
    "INACTIVE_GIVES_ZERO": bool(inactive_progress.item() == 0.0 and inactive_delta.item() == 0.0),
    "INACTIVE_PRESERVES_MAX": bool(torch.isclose(inactive_max, torch.tensor([0.4])).all()),
    "REWARD_WEIGHT_IS_TWO": (
        "fr_clearance_progress = RewTerm(func=mdp.v10_fr_clearance_progress_reward, weight=2.0)" in cfg
    ),
    "EVENT_WEIGHTS_UNCHANGED": all(token in cfg for token in (
        "weight=412.426414", "weight=-618.639622", "weight=1649.705658",
    )),
    "TENSORBOARD_TAGS_PRESENT": all(tag in runner for tag in (
        "Reward/fr_clearance_progress",
        "V10_Perception/MaxClearanceProgress",
        "V10_Perception/Recent100ClearanceProgressMean",
        "V10_Perception/Recent500ClearanceProgressMean",
    )),
}

for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if all(results.values()) else 1)
