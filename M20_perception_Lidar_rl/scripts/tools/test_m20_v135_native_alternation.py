#!/usr/bin/env python3
"""Contract for native V13 stairs plus local post-stair flat recovery."""

from pathlib import Path
import importlib.util

root = Path(__file__).resolve().parents[2]
config = root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20"
env_text = (config / "v135_v11fd_native_alternation_env_cfg.py").read_text()
events_text = (config.parents[2] / "mdp/events.py").read_text()
commands_text = (config.parents[2] / "mdp/commands.py").read_text()
agent_text = (config / "agents/rsl_rl_ppo_cfg.py").read_text()
registry = (config / "__init__.py").read_text()
runner = agent_text.split("class DeeproboticsM20V135NativeAlternationPPORunnerCfg", 1)[1].split("@configclass", 1)[0]

assert "DeeproboticsM20V13V11FDAlternationEnvCfg" in env_text
assert "M20-Stair-V13.5-V11FD-NativeAlternation-v0" in registry
assert "observations.critic.v13_reward_state" in env_text
assert "func=mdp.v135_final_critic_reward_state" in env_text
assert "self.observations.policy.height_scan.func = mdp.v11_actor_perception" in env_text
assert "self.scene.terrain.max_init_terrain_level = 3" in env_text
for exact in (
    "rollout_steps_per_update: int = 24",
    "anneal_updates: int = 1000",
    "early_mixture_probability: float = 0.70",
    "near_stair_distance_range: tuple[float, float] = (0.35, 0.65)",
    "early_ascent_command_range: tuple[float, float] = (0.60, 1.20)",
    "early_descent_command_range: tuple[float, float] = (0.25, 0.65)",
    "descent_velocity_baseline_initial: float = 0.85",
    "descent_velocity_baseline_final: float = 0.60",
    "flat_blend_steps: int = 10",
):
    assert exact in env_text, exact
assert "v135_early_mixture_probability" in events_text
assert "v135_early_mixture_probability" in commands_text
for term in (
    "v135_post_stair_flat_progress", "v135_post_stair_flat_completion",
    "v135_post_stair_flat_crawl", "v135_descent_rear_transfer_progress",
    "v135_descent_body_rear_lag", "v135_descent_premature_collapse",
    "v135_ascent_leader_progress", "v135_ascent_leader_velocity_coupled",
    "v135_descent_rear_early_extension", "v135_descent_rear_recovery",
    "v135_descent_persistent_extension", "v135_adjacent_chain",
    "v135_alternation_cycle",
):
    assert term in env_text, term
for exact in (
    "no_riser_confirmation_frames: int = 10",
    "support_confirmation_frames: int = 5",
    "wheel_plane_tolerance: float = 0.05",
    "clearance_free_ratio: float = 0.90",
    "minimum_quality: float = 0.80",
):
    assert exact in env_text, exact
for forbidden in (
    "knee_flexion", "thigh_parallel", "rear_swing", "bilateral_cycle",
    "randomize_reset_base.func",
    "v133_transition", "teacher_kl", "wheelbase_min", "opposed_knee",
    "v135_cancel_native_descent", "chain_failed_rate",
):
    assert forbidden not in env_text, forbidden
assert "DeeproboticsM20V13V11FDAlternationPPORunnerCfg" in runner
for exact in ('rnn_type: str = "gru"', "rnn_num_layers: int = 1"):
    assert exact in agent_text, exact
for forbidden in ("max_iterations =", "save_interval =", "V134StylePrior"):
    assert forbidden not in runner, forbidden
assert 'class_name="V11ContinuousActorCritic"' in runner
assert "rnn_hidden_dim=64" not in runner
assert "num_steps_per_env = 24" in runner
launcher_text = (root / "scripts/train_m20_v135_native_alternation_from_scratch.sh").read_text()
assert 'V135_NUM_ENVS:-8192' in launcher_text
assert 'V135_MAX_ITERATIONS:-30000' in launcher_text
assert 'v135_longtrain_final_mlp_native6d_e8192_r24_from_scratch_30k' in launcher_text
assert "resume = False" in runner and "load_run = None" in runner and "load_checkpoint = None" in runner
for disabled in (
    "self.rewards.descent_step_complete = None",
    "self.rewards.descent_rear_follow_delay = None",
    "self.rewards.descent_rear_roll_consistency = None",
):
    assert disabled in env_text, disabled

module_path = config.parents[2] / "mdp/v135_descent_stance.py"
spec = importlib.util.spec_from_file_location("v135_descent_stance_pure", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import torch

ascent_path = config.parents[2] / "mdp/v135_final_ascent_reward.py"
ascent_spec = importlib.util.spec_from_file_location("v135_ascent_pure", ascent_path)
ascent = importlib.util.module_from_spec(ascent_spec)
ascent_spec.loader.exec_module(ascent)
eligible = ascent.v135_final_progress_eligible(
    torch.tensor([True, True, True, False]),
    torch.tensor([True, True, False, True]),
    torch.tensor([True, True, True, True]),
)
assert eligible.tolist() == [True, True, False, False]

schedule_path = config.parents[2] / "mdp/v135_longtrain_curriculum.py"
schedule_spec = importlib.util.spec_from_file_location("v135_longtrain_pure", schedule_path)
schedule = importlib.util.module_from_spec(schedule_spec)
schedule_spec.loader.exec_module(schedule)


class _Cfg:
    rollout_steps_per_update = 24
    anneal_updates = 1000
    early_mixture_probability = 0.70
    initial_penalty_scale = 0.25
    ascent_penalty_ramp_start_update = 200
    descent_velocity_baseline_initial = 0.85
    descent_velocity_baseline_final = 0.60


class _EnvCfg:
    v135_longtrain = _Cfg()


class _Env:
    cfg = _EnvCfg()
    common_step_counter = 0


dummy = _Env()
assert schedule.v135_early_mixture_probability(dummy) == 0.70
assert schedule.v135_ascent_error_scale(dummy) == 0.25
assert schedule.v135_descent_error_scale(dummy) == 0.25
assert schedule.v135_descent_velocity_baseline(dummy) == 0.85
dummy.common_step_counter = 200 * 24
assert schedule.v135_ascent_error_scale(dummy) == 0.25
dummy.common_step_counter = 1000 * 24
assert schedule.v135_early_mixture_probability(dummy) == 0.0
assert schedule.v135_ascent_error_scale(dummy) == 1.0
assert schedule.v135_descent_error_scale(dummy) == 1.0
assert schedule.v135_descent_velocity_baseline(dummy) == 0.60

healthy = module.v135_post_stair_quality(
    torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0]),
    torch.tensor([True]), torch.tensor([True]),
    clearance_free_ratio=0.90, clearance_scale_ratio=0.15,
    upright_std=0.20, velocity_std=0.50,
)
crouched = module.v135_post_stair_quality(
    torch.tensor([0.60]), torch.tensor([0.0]), torch.tensor([0.0]),
    torch.tensor([True]), torch.tensor([True]),
    clearance_free_ratio=0.90, clearance_scale_ratio=0.15,
    upright_std=0.20, velocity_std=0.50,
)
airborne = module.v135_post_stair_quality(
    torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0]),
    torch.tensor([False]), torch.tensor([True]),
    clearance_free_ratio=0.90, clearance_scale_ratio=0.15,
    upright_std=0.20, velocity_std=0.50,
)
assert healthy.item() == 1.0
assert crouched.item() == 0.5
assert airborne.item() == 0.0

transfer_path = config.parents[2] / "mdp/v135_descent_support_transfer.py"
transfer_spec = importlib.util.spec_from_file_location("v135_transfer_pure", transfer_path)
transfer = importlib.util.module_from_spec(transfer_spec)
transfer_spec.loader.exec_module(transfer)
extension_cost = transfer.v135_rear_early_extension_cost(
    torch.tensor([[1.3, 1.3], [1.3, 1.3]]),
    torch.tensor([[0.0, 0.0], [0.4, 0.4]]),
    free_ratio=1.15, ratio_scale=0.15,
)
assert extension_cost[0].item() > 0.99 and extension_cost[1].item() == 0.0
support_quality = transfer.v135_rear_support_quality(
    torch.tensor([[True, True], [True, True], [True, False], [True, True]]),
    torch.tensor([[1.10, 1.10], [1.15, 1.15], [1.10, 1.10], [0.80, 0.80]]),
    extension_free_ratio=1.15,
    recovery_free_ratio=1.10,
)
assert torch.allclose(support_quality, torch.tensor([1.0, 0.0, 0.5, 0.0]), atol=1.0e-6)
velocity_factors = transfer.v135_descent_velocity_factor(
    torch.tensor([True, False, False, False]),
    torch.tensor([0.0, 1.0, 1.0, 1.0]),
    torch.tensor([False, True, True, False]),
    torch.tensor([0.0, 0.0, 1.0, 0.0]),
    baseline=0.60,
)
# After a one-time recovery, re-extension must still reduce speed credit.
assert torch.allclose(velocity_factors, torch.tensor([0.60, 0.70, 1.0, 1.0]))
for exact in (
    "func=mdp.v135_final_leader_progress_reward, weight=0.3",
    "func=mdp.v135_final_velocity_coupled_leader_reward, weight=0.7",
    "func=mdp.v135_adjacent_chain_reward, weight=1.0",
    "func=mdp.v135_alternation_cycle_reward, weight=1.5",
    "func=mdp.v135_descent_rear_recovery_reward, weight=0.5",
    "v135_descent_persistent_extension = None",
    "func=mdp.v135_no_skip_native_landing_reward",
    "func=mdp.v135_strict_chain_platform_exit_reward",
    "persistent_extension_frames: int = 15",
    "ordered_speed_multiplier: float = 1.20",
    "ordinary_speed_multiplier: float = 1.00",
    "skip_speed_multiplier: float = 0.85",
):
    assert exact in env_text, exact

print("V135_NATIVE_STAIRS_POST_STAIR_RECOVERY_CONTRACT: PASS")
