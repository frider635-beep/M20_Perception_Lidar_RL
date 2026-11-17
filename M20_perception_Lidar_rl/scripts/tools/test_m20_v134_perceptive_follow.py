#!/usr/bin/env python3
"""CPU contract for V13.4's positive ascent and natural descent design."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"
module_path = MDP / "v134_perceptive_follow.py"
spec = importlib.util.spec_from_file_location("v134_contract", module_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

v133_path = MDP / "v133_adjacent_transition.py"
v133_spec = importlib.util.spec_from_file_location("v133_contract", v133_path)
v133 = importlib.util.module_from_spec(v133_spec)
assert v133_spec.loader is not None
v133_spec.loader.exec_module(v133)

# The descent envelope has a real free region and is one-sided: normal
# extension is free, while excessive folding is bounded rather than rewarded.
cost = module.v134_rear_knee_flexion_cost(
    torch.tensor([[-0.10, 0.20], [-0.45, -0.70]]),
    free_flexion=0.20,
    flexion_scale=0.25,
    max_penalty=1.0,
)
torch.testing.assert_close(cost, torch.tensor([0.0, 1.0]))

# The newly observed descent failure is geometric: a rear thigh nearly parallel
# to the torso is bounded, while the broad natural posture region is untouched.
parallel_cost = module.v134_rear_thigh_parallel_cost(
    torch.tensor([[0.50, 0.75], [0.95, 1.00]]),
    free_cosine=0.75,
    cosine_scale=0.20,
    max_penalty=1.0,
)
torch.testing.assert_close(parallel_cost, torch.tensor([0.0, 1.0]))

platform_cost = module.v134_low_body_clearance_cost(
    torch.tensor([0.50, 0.38, 0.28]),
    free_height=0.38,
    height_scale=0.10,
    max_penalty=1.0,
)
torch.testing.assert_close(platform_cost, torch.tensor([0.0, 0.0, 1.0]))

# Last-riser front support must activate posture protection before the rear
# wheels finish platform confirmation; descent and an intermediate visible
# riser must never enter that phase.
platform_active = module.v134_platform_clearance_active(
    torch.tensor([True, True, True, False, True]),
    torch.tensor([True, True, True, True, False]),
    torch.tensor([True, False, True, True, True]),
    torch.tensor([True, True, False, True, True]),
    torch.tensor([False, False, False, False, True]),
)
assert platform_active.tolist() == [True, False, False, False, True]

# Only the first locked world face can seed the chain.  The exact same helper
# accepts either side because leg identity is deliberately absent here.
b1 = v133.v133_match_locked_first_riser(
    torch.tensor([True, True, True]),
    torch.tensor([[1.00, 0.00], [1.30, 0.00], [1.00, 0.20]]),
    torch.tensor([True, True, True]),
    torch.tensor([[1.00, 0.00], [1.00, 0.00], [1.00, 0.00]]),
    torch.tensor([[1.00, 0.00], [1.00, 0.00], [1.00, 0.00]]),
    longitudinal_tolerance=0.08,
    lateral_tolerance=0.15,
)
assert b1.tolist() == [True, False, False]

# Teacher retention is ascent-only.  A descending stair and a riser-free top
# are both excluded even though they share the same actor network.
style_mask = module.v134_ascent_style_anchor_mask(
    torch.tensor([True, False, True]),
    torch.tensor([[True, False], [True, True], [False, False]]),
    torch.tensor([[0.12, 0.00], [0.12, 0.12], [0.00, 0.00]]),
    minimum_height_gain=0.03,
    forward_command=torch.tensor([True, True, False]),
)
assert style_mask.tolist() == [True, False, False]

env_cfg = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v134_v11fd_perceptive_follow_env_cfg.py").read_text()
assert "DeeproboticsM20V13V11FDAlternationEnvCfg" in env_cfg
assert "DeeproboticsM20V133AdjacentTransitionAcquisitionEnvCfg" not in env_cfg
assert "observations.critic.v134_task_state" in env_cfg
assert "observations.policy" not in env_cfg
assert "v134_positive_ascent_chain_reward" in env_cfg
assert "v134_descent_rear_extension" in env_cfg
assert "v134_descent_rear_knee_flexion" in env_cfg
assert "v134_descent_rear_roll" in env_cfg
assert "v134_descent_rear_thigh_parallel" in env_cfg
assert "v134_platform_body_clearance" in env_cfg
assert "v134_descent_follow_reward" not in env_cfg
assert "v134_descent_follow_completion_reward" not in env_cfg
assert "adjacent_progress_weight: float = 0.50" in env_cfg
assert "require_first_riser_seed: bool = True" in env_cfg
assert "b1_settlement: float = 1.0" in env_cfg
assert "repeat_transition_cost: float = 0.0" in env_cfg
assert "skip_transition_cost: float = 0.0" in env_cfg
assert "simultaneous_transition_cost: float = 0.0" in env_cfg
assert "V134_DESCENT_ORDER_OR_TREAD_LOCK=NO" in env_cfg
assert "V134_MODEL5000_STUDENT_INITIALIZATION=NO" in env_cfg
assert "V134_MODEL5000_ASCENT_TEACHER_ONLY=YES" in env_cfg
assert "V134_PLATFORM_CLEARANCE_BEFORE_CONFIRMATION=YES" in env_cfg
assert "V134_MODEL5000_ACTION_REFERENCE_REWARD=NO" in env_cfg
for forbidden in ("reference_action", "behavior_clone"):
    assert forbidden not in env_cfg

runner_cfg = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py").read_text()
section = runner_cfg.split("class DeeproboticsM20V134PerceptiveFollowPPORunnerCfg", 1)[1].split("@configclass", 1)[0]
assert "DeeproboticsM20V13V11FDAlternationPPORunnerCfg" in section
assert "v134_diagnostic_runner" not in section
assert "num_steps_per_env = 24" in section
assert "max_iterations = 30000" in section
assert "V134StylePriorRunner" in section
assert "V134StylePriorPPO" in section
assert "v134_teacher_burst_start_probability = 0.025" in section
assert "v134_teacher_burst_end_probability = 0.005" in section
assert "v134_teacher_burst_decay_updates = 6000" in section
assert "v134_teacher_burst_steps = 8" in section
assert "learning_rate = 1.0e-4" in section
assert "entropy_coef = 1.0e-3" in section
assert "v134_ascent_gradient" not in section
assert "v134_descent_reward_terms" not in section

prior_runner = (ROOT / "source/rl_training/rl_training/v134_style_prior_runner.py").read_text()
assert "teacher_actor.load_state_dict(expected_actor, strict=True)" in prior_runner
assert "V134_STYLE_TEACHER_EXACT_LOAD=YES" in prior_runner
assert "V134_STYLE_STUDENT_FRESH=YES" in prior_runner
assert "V134_STYLE_CRITIC_FRESH=YES" in prior_runner
assert "_project_to_teacher_kl" not in prior_runner
assert "_ascent_teacher_update" not in prior_runner
assert "v134_teacher_action_masks" in prior_runner
assert "masked_advantages.masked_fill_(teacher_masks, 0.0)" in prior_runner
assert "V134_TEACHER_ACTIONS_EXCLUDED_FROM_ACTOR_PPO=YES" in prior_runner
assert "V134_TEACHER_RETURNS_AVAILABLE_TO_CRITIC=YES" in prior_runner
assert "v133_ascending_stair_mask" in prior_runner
assert "v133_descending_stair_mask" not in prior_runner

mdp_source = module_path.read_text()
reward_body = mdp_source.split("def v134_positive_ascent_chain_reward", 1)[1].split("def v134_native_chain_critic_state", 1)[0]
assert 'state["adjacent_event"]' in reward_body
assert 'state["seed_event"]' in reward_body
assert 'state["chain_event"]' in reward_body
assert 'state["adjacent_progress_delta"]' in reward_body
assert "repeat_event" not in reward_body
assert "skip_event" not in reward_body
assert "clawback" not in reward_body

print("V134_NATIVE_ASCENT_NATURAL_DESCENT_CONTRACT: PASS")
