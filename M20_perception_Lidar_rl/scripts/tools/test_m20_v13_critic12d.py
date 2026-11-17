#!/usr/bin/env python3
"""CPU contract for V13 ascent/descent critic-only state."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v13_alternation.py"
spec = importlib.util.spec_from_file_location("v13_alternation_contract", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

encoded = module.v13_encode_critic_descent_state(
    torch.tensor([False, True]),
    torch.tensor([False, True]),
    torch.tensor([0.4, -0.2]),
    torch.tensor([0.1, 0.16]),
    torch.tensor([0.0, 0.11]),
    torch.tensor([1.0, 0.5]),
)
torch.testing.assert_close(encoded, torch.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, -0.2, 0.16, 0.11, 0.5],
]))
assert encoded.shape == (2, 6) and torch.isfinite(encoded).all()

extension_cost = module.v13_rear_extension_cost(
    torch.tensor([[0.00, 0.08], [0.14, 0.20], [0.50, 0.50]]),
    free_reach=0.08, reach_scale=0.12, max_penalty=1.0,
)
torch.testing.assert_close(extension_cost, torch.tensor([0.0, 0.625, 1.0]))

route_quality = module.v13_route_engagement_quality(
    path_speed=torch.tensor([0.0, 0.5, 0.5, 0.5]),
    command_x=torch.ones(4),
    heading_error=torch.tensor([0.0, 0.0, 1.0, 0.0]),
    lateral_error=torch.tensor([0.0, 0.2, 0.2, 0.8]),
    target_speed_fraction=0.5,
    heading_scale=0.5,
    lateral_tolerance=0.45,
    lateral_scale=0.35,
)
torch.testing.assert_close(route_quality[:2], torch.tensor([0.0, 1.0]))
assert 0.0 < route_quality[2].item() < 0.02
assert 0.0 < route_quality[3].item() < 0.5

v11_module_path = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py"
v11_source = v11_module_path.read_text()
assert "def v11_ascent_route_coupled_track_lin_vel_xy_exp(" in v11_source
assert "def v11_ascent_route_coupled_track_ang_vel_z_exp(" in v11_source

cfg = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v13_critic12d_env_cfg.py").read_text()
assert "DeeproboticsM20V13Critic6DEnvCfg" in cfg
assert "observations.critic.v13_descent_state" in cfg
assert "v13_critic_descent_metric" in cfg
assert "observations.policy" not in cfg
assert "soft_alternation_command_split = True" in cfg
assert "v11_ascent_route_coupled_track_lin_vel_xy_exp" in cfg
assert "v11_ascent_route_coupled_track_ang_vel_z_exp" in cfg
assert "v13_ascent_route_engagement" in cfg
assert "v13_descent_rear_extension_penalty" in cfg
assert "v13_descent_rear_roll_penalty" in cfg
assert "DESCENT_TREAD_LOCK=NO" in cfg

runner = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py").read_text()
section = runner.split("class DeeproboticsM20V13Critic12DPPORunnerCfg", 1)[1].split("@configclass", 1)[0]
assert "num_steps_per_env = 24" in section
print("V13_CRITIC12D_CONTRACT: PASS")
