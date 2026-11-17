#!/usr/bin/env python3
"""CPU contract for the V11.1 pre-thigh-alignment descent repair."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20"
module_path = MDP / "v111_rear_posture.py"
spec = importlib.util.spec_from_file_location("v111_contract", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

cost = module.v111_rear_knee_flexion_cost(
    torch.tensor([[-0.10, 0.20], [-0.45, -0.70]]),
    free_flexion=0.20,
    flexion_scale=0.25,
    max_penalty=1.0,
)
torch.testing.assert_close(cost, torch.tensor([0.0, 1.0]))

clearance_cost = module.v111_low_body_clearance_cost(
    torch.tensor([0.40, 0.38, 0.33]),
    free_height=0.38, height_scale=0.10, max_penalty=1.0,
)
torch.testing.assert_close(clearance_cost, torch.tensor([0.0, 0.0, 0.25]))

steps = torch.zeros(2, dtype=torch.long)
for frame in range(1, 6):
    exit_state = module.v111_confirm_level_exit(
        steps,
        stair_started=torch.tensor([True, True]),
        no_visible_riser=torch.tensor([True, True]),
        all_wheels_supported=torch.tensor([True, True]),
        support_height_spread=torch.tensor([0.02, 0.08]),
        support_height_tolerance=0.06,
        confirmation_frames=5,
    )
    steps = exit_state["steps"]
    assert exit_state["confirmed"].tolist() == [frame == 5, False]

env_source = (CONFIG / "v111_v11fd_rear_posture_env_cfg.py").read_text()
assert "DeeproboticsM20V11FlatDescentEnvCfg" in env_source
for term in (
    "v111_descent_rear_extension", "v111_descent_rear_knee_flexion",
    "v111_descent_rear_roll", "v111_level_exit_body_clearance",
):
    assert term in env_source, term
for forbidden in (
    "v134_descent_rear_thigh_parallel_penalty", "rear_thigh_parallel_weight",
    "v13_opposite", "alternating_completion",
    "observations.policy", "observations.critic",
):
    assert forbidden not in env_source.lower(), forbidden
assert "V111_REAR_THIGH_PARALLEL_PENALTY=NO" in env_source

agent_source = (CONFIG / "agents/rsl_rl_ppo_cfg.py").read_text()
section = agent_source.split("class DeeproboticsM20V111V11FDRearPosturePPORunnerCfg", 1)[1].split("@configclass", 1)[0]
assert "DeeproboticsM20V11FlatDescentPPORunnerCfg" in section
assert "resume = False" in section
assert "load_run = None" in section and "load_checkpoint = None" in section
for forbidden in ("teacher", "checkpoint_load", "style_prior"):
    assert forbidden not in section.lower(), forbidden

registry = (CONFIG / "__init__.py").read_text()
assert "M20-Stair-V11.1-V11FD-RearPosture-v0" in registry

print("V111_V11FD_REAR_POSTURE_CONTRACT: PASS")
