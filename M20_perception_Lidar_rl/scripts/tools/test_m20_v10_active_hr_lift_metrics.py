#!/usr/bin/env python3
"""Legacy filename: geometry and wiring gates for base-frame FR active-lift diagnostics."""

from pathlib import Path

import torch


def quat_apply(quat, vector):
    xyz = quat[:, 1:]
    uv = torch.cross(xyz, vector, dim=1)
    uuv = torch.cross(xyz, uv, dim=1)
    return vector + 2.0 * (quat[:, :1] * uv + uuv)


def quat_apply_inverse(quat, vector):
    inverse = quat.clone()
    inverse[:, 1:] *= -1.0
    return quat_apply(inverse, vector)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py"
RUNNER = ROOT / "source/rl_training/rl_training/v10_unified_runner.py"
if not SOURCE.is_file():
    ROOT = Path(__file__).resolve().parents[1]
    SOURCE = ROOT / "source/v10_unified.py"
    RUNNER = ROOT / "source/v10_unified_runner.py"

source = SOURCE.read_text()
runner = RUNNER.read_text()

quat = torch.tensor([[0.80, 0.20, -0.30, 0.40]])
quat = quat / quat.norm(dim=1, keepdim=True)
local_at_b_on = torch.tensor([[0.20, -0.30, -0.40]])
local_lifted = torch.tensor([[0.20, -0.30, -0.30]])
world_at_b_on = quat_apply(quat, local_at_b_on)
world_lifted = quat_apply(quat, local_lifted)
recovered_at_b_on = quat_apply_inverse(quat, world_at_b_on)
recovered_lifted = quat_apply_inverse(quat, world_lifted)
active_lift = recovered_lifted[:, 2] - recovered_at_b_on[:, 2]

results = {
    "ROLL_PITCH_YAW_REMOVED": torch.allclose(recovered_at_b_on, local_at_b_on, atol=1e-6),
    "UPWARD_RETRACTION_IS_POSITIVE": abs(float(active_lift.item()) - 0.10) < 1e-6,
    "RIGID_WORLD_Z_NOT_USED": "fr_pos[:, 2] - root[:, 2]" not in source,
    "BASE_FRAME_FORMULA": (
        "quat_apply_inverse(asset.data.root_quat_w, fr_pos - root)" in source
    ),
    "BASELINE_LATCHED_AT_FIRST_B_ON": (
        'first_b_on_event, fr_base_z, state["fr_base_z_at_b_on"]' in source
    ),
    "BASELINE_RESET_ON_REARM": (
        '"fr_base_z_reference_valid"' in source
        and '"fr_base_z_at_b_on", "max_active_fr_lift"' in source
    ),
    "PRE_CROSSING_WINDOW_REUSED": (
        'clearance_active,' in source and '& ~state["crossing_latched"]' in source
    ),
    "THREE_SIGNED_ACTIONS": all(
        name in runner for name in ("FRHipXAction", "FRHipYAction", "FRKneeAction")
    ),
    "THREE_ABS_ACTIONS": all(
        name in runner for name in ("AbsFRHipXAction", "AbsFRHipYAction", "AbsFRKneeAction")
    ),
    "ACTIVE_LIFT_TAGS": (
        'metrics[f"{prefix}/MaxActiveFRLift"]' in runner
        and 'metrics[f"{prefix}/ActiveLiftRatio"]' in runner
    ),
    "WORLD_Z_LIFT_RETAINED": "FRAbsoluteMaxLiftCm" in runner,
}

for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if all(results.values()) else 1)
