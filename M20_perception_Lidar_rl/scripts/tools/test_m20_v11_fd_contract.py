#!/usr/bin/env python3
"""CPU regression for V11-FD's independent high-clearance contract."""

from __future__ import annotations

import importlib.util
import ast
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
LANDING = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_landing.py"
CONTINUOUS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py"
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_flat_descent_env_cfg.py"
POLICY = ROOT / "source/rl_training/rl_training/v11_continuous_policy.py"
SPATIAL = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/perceptive_spatial.py"

spec = importlib.util.spec_from_file_location("v11_fd_landing_pure", LANDING)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

upper = torch.tensor([[0.17, 0.17]])
crossing = torch.tensor([[True, True]])

# 13.9 cm above the target tread fails; 14.0 cm is the inclusive boundary.
gate = module.v11_crossing_clearance_gate(
    torch.tensor([[0.309, 0.310]]), upper, crossing,
    wheel_radius=0.09, clearance_margin=0.05,
)
assert torch.equal(gate["clear"], torch.tensor([[False, True]]))
assert torch.equal(gate["low"], torch.tensor([[True, False]]))

# No crossing means no completion signal even when the wheel is high.
stationary = module.v11_crossing_clearance_gate(
    torch.tensor([[0.40, 0.40]]), upper, torch.zeros_like(crossing),
    wheel_radius=0.09, clearance_margin=0.05,
)
assert not stationary["clear"].any() and not stationary["low"].any()

# Graded height credit is bounded, and lifting/crossing without a subsequent
# safe support event must release exactly zero reward.
support_progress = module.v11_safe_support_clearance_progress(
    torch.tensor([[0.07, 0.20]]), torch.tensor([[True, False]]),
    required_center_above_tread=0.14,
)
assert torch.allclose(support_progress, torch.tensor([[0.5, 0.0]]))
assert torch.equal(
    module.v11_safe_support_clearance_progress(
        torch.tensor([[0.20, 0.20]]), torch.tensor([[True, True]]),
        required_center_above_tread=0.14,
    ),
    torch.ones((1, 2)),
)

# Dense progress saturates at the 5 cm bottom-clearance target, remains
# target-relative on a later riser and cannot pay before a landing event.
zeros = torch.zeros((1, 2))
escrow = module.v11_update_clearance_escrow(
    zeros, zeros, torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool),
    torch.tensor([[0.03, 0.03]]), torch.tensor([[0.22, 0.389]]),
    torch.tensor([[0.0, 0.17]]), torch.tensor([[0.17, 0.34]]),
    torch.ones((1, 2), dtype=torch.bool), clearance_margin=0.05,
)
assert escrow["clearance"][0, 0]
assert not escrow["clearance"][0, 1]
assert module.v11_settle_bilateral_escrow(
    escrow["escrow"], torch.tensor([False]),
).item() == 0.0

# Execute the real training-side point extractor in isolation.  A physical
# descent is hidden by the legacy contract and retained as two negative Hs by
# signed-edge-v2.
spatial_tree = ast.parse(SPATIAL.read_text(encoding="utf-8"))
extract_node = next(
    node for node in spatial_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "extract_two_risers_points"
)
namespace = {"torch": torch}
exec(compile(ast.Module(body=[extract_node], type_ignores=[]), str(SPATIAL), "exec"), namespace)
extract_two = namespace["extract_two_risers_points"]
x = torch.linspace(1.25, -0.75, 21)
y = torch.linspace(0.70, -0.70, 15)
xx, yy = torch.meshgrid(x, y, indexing="ij")
zz = torch.where(xx < 0.20, 0.34, torch.where(xx < 0.50, 0.17, 0.0))
points = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0)
kwargs = dict(corridor_half_width=0.20, h_min=0.05, d_end=-0.30, min_separation=0.15)
legacy_edges = extract_two(points, signed_edges=False, **kwargs)
signed_edges = extract_two(points, signed_edges=True, **kwargs)
assert not legacy_edges["valid"].any()
assert signed_edges["valid"].all()
assert torch.allclose(signed_edges["riser_height"], torch.tensor([[-0.17, -0.17]]), atol=1.0e-5)

config = CONFIG.read_text(encoding="utf-8")
source = CONTINUOUS.read_text(encoding="utf-8")
policy = POLICY.read_text(encoding="utf-8")

# The pure lifecycle gate must retain an already-locked edge after the front
# wheels complete, even once that edge is no longer present in the forward
# two-riser scan.  Before front completion, the same perception loss remains
# fail-closed.
continuous_tree = ast.parse(source)
phase_node = next(
    node for node in continuous_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "v11_descent_phase_active"
)
phase_namespace = {"torch": torch}
exec(compile(ast.Module(body=[phase_node], type_ignores=[]), str(CONTINUOUS), "exec"), phase_namespace)
descent_phase_active = phase_namespace["v11_descent_phase_active"]
active = torch.tensor([True, True, True])
reference = torch.tensor([True, True, True])
visible = torch.tensor([False, False, True])
front_complete = torch.tensor([False, True, True])
rear_complete = torch.tensor([False, False, True])
assert torch.equal(
    descent_phase_active(active, reference, visible, front_complete, rear_complete),
    torch.tensor([False, True, True]),
)
assert torch.equal(
    descent_phase_active(active, reference, visible, front_complete, rear_complete, retain_after_front=False),
    torch.tensor([False, False, True]),
)

assert 'front_completion_mode: str = "bilateral_same_riser"' in config
assert 'step_feature_mode: str = "signed_edges"' in config
assert 'feature_contract: str = "v11_signed_edges_v2"' in config
assert 'clearance_margin: float = 0.05' in config
assert 'require_clearance_at_crossing: bool = True' in config
assert 'enable_persistent_descent_rear_phase: bool = True' in config
assert 'self.scene.terrain.max_init_terrain_level = 1' in config
assert 'safe_front_landing_progress = RewTerm(' in config
assert 'single_front_hard_landing = RewTerm(' in config
assert 'self.rewards.rotation_gait_status = None' in config
assert 'self.rewards.rotation_gait_symmetry = None' in config
assert 'self.rewards.feet_air_time_ang_z_M20 = None' in config
assert 'self.rewards.descent_rear_support_progress = RewTerm(' in config
assert 'self.rewards.joint_mirror.func = mdp.v11_descent_phase_joint_mirror' in config
assert '"descent_rear_hipy_scale": 0.50' in config
assert '"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20)' in config
assert '"pyramid_stairs": 0.20' in config
assert '"pyramid_stairs_inv": 0.30' in config
assert 'candidate_ready &= state["crossing_clearance_ok"]' in source
assert 'state["clearance_escrow"] = torch.where' in source
assert 'state["front_support_clearance_settlement"] = v11_safe_support_clearance_progress(' in source
assert 'state["front_candidate_event"] = (' in source
assert '~previous_candidate_ready & candidate_ready & ~already_support_credited' in source
assert 'state["front_support_credited_valid"] |= safe_support_event' in source
assert 'torch.quantile(sample[valid], 0.10)' in source
assert 'ascent_risers["valid"] = positive_valid' in source
assert 'descent_risers["valid"] = negative_valid' in source
assert 'def v11_descent_phase_active(' in source
assert 'state["reference_lower_z"][assign] = lower[assign]' in source
assert 'state["rear_support_event"] = (' in source
assert '!= (63, 16)' in policy
assert 'V11 dimensions must be actor=63/action=16' in policy

print("V11_FD_CONTRACT_REGRESSION: PASS")
