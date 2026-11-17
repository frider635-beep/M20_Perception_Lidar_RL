#!/usr/bin/env python3
"""Pure geometry and static wiring gates for the V10 HR-to-FR correction."""

import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py"
POLICY = ROOT / "source/rl_training/rl_training/v10_unified_policy.py"
RUNNER = ROOT / "source/rl_training/rl_training/v10_unified_runner.py"
CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v10_unified_env_cfg.py"
if not SOURCE.is_file():
    ROOT = Path(__file__).resolve().parents[1]
    SOURCE = ROOT / "source/v10_unified.py"
    POLICY = ROOT / "source/v10_unified_policy.py"
    RUNNER = ROOT / "source/v10_unified_runner.py"
    CFG = ROOT / "source/v10_unified_env_cfg.py"

source = SOURCE.read_text()
policy = POLICY.read_text()
runner = RUNNER.read_text()
cfg = CFG.read_text()
tree = ast.parse(source)
names = {"v10_first_wheel_geometry", "v10_event_pulses"}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
namespace = {"torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
geometry = namespace["v10_first_wheel_geometry"]
pulses = namespace["v10_event_pulses"]


def geom(position, force):
    return geometry(
        torch.tensor([position], dtype=torch.float32),
        torch.tensor([force], dtype=torch.float32),
        torch.tensor([0.0]), torch.tensor([0.0]),
        torch.tensor([0.0]), torch.tensor([0.10]),
        torch.tensor([1.0]), torch.tensor([0.0]),
        torch.tensor([True]), torch.tensor([False]),
        0.05, 1.0, 0.5, 0.01, 0.005,
    )


clearance = geom((-0.10, 0.0, 0.16), (0.0, 0.0, 0.0))
vertical = geom((0.00, 0.0, 0.10), (10.0, 0.0, 0.0))
support = geom((0.10, 0.0, 0.15), (0.0, 0.0, 10.0))
false = torch.tensor([False])
true = torch.tensor([True])
success = pulses(true, true, true, true, false, false, false, false)
miss = pulses(true, false, false, false, false, false, false, true)

results = {
    "FR_SPECIAL_HEAD_MAPPING": (
        "self.fr_head" in policy
        and "torch.cat((support[..., :3], fr, support[..., 3:13])" in policy
    ),
    "HD_TO_FR_ONLY": "self.fr_head(torch.cat((latent, height_distance), dim=-1))" in policy,
    "HD_TO_HR_DIRECT_PATH_FALSE": "self.hr_head" not in policy,
    "ACTION_16_ORDER_PRESERVED": "support[..., :3]" in policy and "support[..., 3:13]" in policy,
    "FR_BODY_CONTACT_WIRING": (
        'asset.find_bodies("fr_wheel")' in source
        and 'contact.find_bodies("fr_wheel")' in source
    ),
    "FR_ACTION_INDICES": "env.action_manager.action[:, 3:6]" in source,
    "FR_FIRST_CLEARANCE_GEOMETRY": bool(clearance["clearance"] and not clearance["crossing"]),
    "FR_VERTICAL_CONTACT": bool(vertical["vertical_contact"]),
    "FR_CROSSING": bool(support["crossing"]),
    "FR_UPPER_SUPPORT": bool(support["upper_support"]),
    "FR_FIRST_STEP_SUCCESS": bool(success[0] and success[1]),
    "FR_PERCEPTION_MISS": bool(miss[2] and miss[3]),
    "FR_TEACHER_GEOMETRY": all(key in source for key in ('state["fr_z"]', 'state["fr_progress"]')),
    "FR_DENSE_REWARD": (
        "v10_fr_clearance_progress_reward" in source
        and "fr_clearance_progress = RewTerm" in cfg
        and 'get_term_cfg("fr_clearance_progress")' in runner
    ),
    "FULL_REAR_HR_SEMANTICS_RETAINED": (
        'asset.find_bodies("hr_wheel")' in source
        and "full_rear_progress" in source
        and "single_mask, full_mask, state[\"target_valid\"], base_progress, full_rear_progress" in source
    ),
    "JOINT_USES_NEW_FIRST_AND_FULL": (
        "joint_now = full_mask & first_step_success_latched & full_latched_next" in source
    ),
}

for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if all(results.values()) else 1)
