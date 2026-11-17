#!/usr/bin/env python3
"""CPU contracts for native V11-FD plus V13 soft alternation shaping."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v13_alternation.py"
V11_CONTINUOUS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py"
V11_LANDING = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_landing.py"
CURRICULUMS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py"
COMMANDS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/commands.py"
V11_CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_flat_descent_env_cfg.py"
V13_CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v13_v11fd_alternation_env_cfg.py"

TEST_PACKAGE = "v13_contract_mdp"
test_package = types.ModuleType(TEST_PACKAGE)
test_package.__path__ = []
test_rewards = types.ModuleType(f"{TEST_PACKAGE}.rewards")
test_rewards.track_lin_vel_xy_exp = lambda env, std, command_name: env.velocity_quality
test_v11 = types.ModuleType(f"{TEST_PACKAGE}.v11_continuous")
v11_state_holder = {}
test_v11.v11_state = lambda env, sensor_cfg=None: v11_state_holder["state"]
sys.modules[TEST_PACKAGE] = test_package
sys.modules[test_rewards.__name__] = test_rewards
sys.modules[test_v11.__name__] = test_v11
spec = importlib.util.spec_from_file_location(f"{TEST_PACKAGE}.v13_alternation", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
landing_spec = importlib.util.spec_from_file_location("v13_contract_v11_landing", V11_LANDING)
landing_module = importlib.util.module_from_spec(landing_spec)
assert landing_spec.loader is not None
landing_spec.loader.exec_module(landing_module)

v13_source = MODULE.read_text(encoding="utf-8")
v13_cfg = V13_CFG.read_text(encoding="utf-8")

# V11 is inherited, never copied or rewritten, and V13 does not alter landing gates.
assert "(DeeproboticsM20V11FlatDescentEnvCfg)" in v13_cfg
assert V11_CFG.read_text(encoding="utf-8").count("v13_") == 0
assert "candidate_ready &=" not in v13_source

# V13 uses the soft physical candidate path, but must not inherit SoftAlt's
# reduced/split stair-command curriculum. Its command sampling stays V11-FD.
commands_source = COMMANDS.read_text(encoding="utf-8")
assert 'getattr(v11_cfg, "soft_alternation_command_split", False)' in commands_source
assert 'self.v11_perception.soft_alternation_command_split = False' in v13_cfg

# The V13-selected one-leg/riser branch must retain V11-FD's crossing-time
# 14 cm safety gate. A previously high wheel that crosses low cannot complete.
crossing_gate = landing_module.v11_crossing_clearance_gate(
    wheel_center_z=torch.tensor([[1.13, 1.15]]),
    upper_tread_z=torch.tensor([[1.0, 1.0]]),
    crossing_now=torch.tensor([[True, True]]),
    wheel_radius=0.09,
    clearance_margin=0.05,
)
physical_candidate = landing_module.v11_softalt_candidate_ready(
    clearance=torch.tensor([[True, True]]),
    crossing=torch.tensor([[True, True]]),
    support=torch.tensor([[True, True]]),
    vertical_contact=torch.tensor([[False, False]]),
    eligible_sides=torch.tensor([[True, True]]),
    sequence_started=torch.tensor([True]),
    anchor_support_valid=torch.tensor([True]),
    soft_alternation=True,
)
safe_candidate = physical_candidate & crossing_gate["clear"]
assert safe_candidate.tolist() == [[False, True]]
alternating_source = V11_CONTINUOUS.read_text(encoding="utf-8").split(
    "def _v11_alternating_landing_state", 1,
)[1].split("def _v11_bilateral_landing_state", 1)[0]
assert "v11_crossing_clearance_gate(" in alternating_source
assert 'candidate_ready &= state["crossing_clearance_ok"]' in alternating_source

# The bounded potential uses forward-qualified clearance, clean crossing, and
# supported upper-tread arrival. Lifting/crossing dirty cannot mint later credit.
potential = module.v13_front_riser_potential(
    max_clearance_progress=torch.tensor([[0.4, 1.0], [1.0, 1.0]]),
    crossing_clearance_ok=torch.tensor([[False, False], [True, True]]),
    wheel_x=torch.tensor([[0.0, 0.0], [1.05, 1.10]]),
    target_riser_x=torch.tensor([1.0, 1.0]),
    support=torch.tensor([[False, False], [False, True]]),
    crossing_span=0.10,
)
torch.testing.assert_close(potential, torch.tensor([[0.20, 0.50], [0.625, 1.0]]))

# Historical-best increments pay each physical advance once. Regression,
# repeated pose, and ineligible/seed riser progress are all zero-credit.
history = module.v13_historical_best_progress(
    previous_best=torch.tensor([[0.2, 0.5], [0.7, 0.4], [0.1, 0.1]]),
    current_potential=torch.tensor([[0.4, 0.4], [0.6, 0.8], [0.9, 0.9]]),
    eligible=torch.tensor([True, True, False]),
)
torch.testing.assert_close(
    history["delta"], torch.tensor([[0.2, 0.0], [0.0, 0.4], [0.0, 0.0]]),
)
repeated = module.v13_historical_best_progress(
    history["best"], torch.tensor([[0.4, 0.4], [0.6, 0.8], [1.0, 1.0]]),
    torch.tensor([True, True, False]),
)
assert not repeated["delta"].any()

# A monotone-or-oscillating trajectory can settle at most one unit per side on
# one locked riser; revisiting earlier values never increases total return.
best = torch.zeros((1, 2))
settled = torch.zeros((1, 2))
for current in (
    torch.tensor([[0.2, 0.4]]),
    torch.tensor([[0.1, 0.3]]),
    torch.tensor([[0.8, 0.5]]),
    torch.tensor([[0.7, 1.0]]),
    torch.tensor([[1.0, 0.8]]),
):
    update = module.v13_historical_best_progress(best, current, torch.tensor([True]))
    best = update["best"]
    settled += update["delta"]
torch.testing.assert_close(settled, torch.tensor([[1.0, 1.0]]))

# Expected side is the opposite side after the free seed. Overlap is exactly
# simultaneous effective riser progress, not arbitrary bilateral leg motion.
split = module.v13_split_alternation_progress(
    per_side_delta=torch.tensor([[0.2, 0.1], [0.3, 0.4], [0.8, 0.9]]),
    expected_side=torch.tensor([1, 0, 1]),
    eligible=torch.tensor([True, True, False]),
)
torch.testing.assert_close(split["opposite"], torch.tensor([0.0, 0.0, 0.0]))
torch.testing.assert_close(split["same"], torch.tensor([0.1, 0.1, 0.0]))
torch.testing.assert_close(split["simultaneous"], torch.tensor([0.1, 0.3, 0.0]))
torch.testing.assert_close(
    split["opposite"] + split["simultaneous"], torch.tensor([0.1, 0.3, 0.0]),
)
torch.testing.assert_close(
    split["same"] + split["simultaneous"], torch.tensor([0.2, 0.4, 0.0]),
)


def fake_v11_state(
    reference_x: float,
    clearance: tuple[float, float],
    *,
    landing_event: bool = False,
    side: int = -1,
    target_active: bool = True,
):
    return {
        "risers": {"riser_x": torch.tensor([[1.0, 1.3]])},
        "landing": {
            "landing_event": torch.tensor([landing_event]),
            "responsible": torch.tensor([side]),
            "reference_riser_world_xy": torch.tensor([[reference_x, 0.0]]),
            "reference_valid": torch.tensor([True]),
            "target_active": torch.tensor([target_active]),
            "target_riser_x": torch.tensor([1.0]),
            "scores": {"center": torch.tensor([[0.7, 0.4]])},
            "max_clearance_progress": torch.tensor([clearance]),
            "crossing_clearance_ok": torch.tensor([[False, False]]),
            "wheel_yaw": torch.tensor([[[0.8, -0.1, 0.0], [0.8, 0.1, 0.0]]]),
            "support": torch.tensor([[False, False]]),
            "candidate_ready": torch.tensor([[landing_event, False]]),
        },
    }


# Exercise the real state cache/reset/reference path without Isaac physics.
fake_cfg = types.SimpleNamespace(
    v13_alternation=types.SimpleNamespace(minimum_riser_separation=0.15),
    v11_perception=types.SimpleNamespace(wheel_radius=0.09, front_margin=0.05),
)
state_env = types.SimpleNamespace(
    num_envs=1, device="cpu", episode_length_buf=torch.tensor([5]),
    common_step_counter=0, cfg=fake_cfg,
)
v11_state_holder["state"] = fake_v11_state(1.0, (1.0, 0.5), landing_event=True, side=0)
seed_state = module.v13_alternation_state(state_env)
assert seed_state["seed_event"].item() and seed_state["expected_side"].item() == 1
assert not seed_state["opposite_progress_delta"].any()  # first riser is unpaid

# A flat/no-target gap keeps the phase seeded by B1 and emits no stair style
# credit. The next real riser must resume with the same expected side.
state_env.episode_length_buf[:] = 6
state_env.common_step_counter = 1
v11_state_holder["state"] = fake_v11_state(1.0, (0.0, 0.0), target_active=False)
flat_gap_state = module.v13_alternation_state(state_env)
assert flat_gap_state["sequence_started"].item()
assert flat_gap_state["expected_side"].item() == 1
assert not flat_gap_state["opposite_progress_delta"].any()

state_env.episode_length_buf[:] = 7
state_env.common_step_counter = 2
v11_state_holder["state"] = fake_v11_state(1.3, (0.2, 0.4))
next_state = module.v13_alternation_state(state_env)
torch.testing.assert_close(next_state["opposite_progress_delta"], torch.tensor([0.1]))
torch.testing.assert_close(next_state["same_side_progress_delta"], torch.tensor([0.0]))
torch.testing.assert_close(next_state["simultaneous_progress_delta"], torch.tensor([0.1]))
cached_opposite = next_state["opposite_progress_delta"].clone()
v11_state_holder["state"] = fake_v11_state(1.3, (1.0, 1.0))
cached_state = module.v13_alternation_state(state_env)
torch.testing.assert_close(cached_state["opposite_progress_delta"], cached_opposite)

state_env.episode_length_buf[:] = 8
state_env.common_step_counter = 3
v11_state_holder["state"] = fake_v11_state(1.3, (0.1, 0.3))
regressed_state = module.v13_alternation_state(state_env)
assert not regressed_state["opposite_progress_delta"].any()
assert not regressed_state["same_side_progress_delta"].any()

state_env.episode_length_buf[:] = 0
state_env.common_step_counter = 4
reset_state = module.v13_alternation_state(state_env)
assert not reset_state["sequence_started"].item()
assert reset_state["expected_side"].item() == -1
assert not reset_state["opposite_progress_delta"].any()


def transition(expected: int, started: bool, side: int, simultaneous: bool = False):
    return module.v13_update_alternating_sequence(
        torch.tensor([expected]),
        torch.tensor([started]),
        torch.tensor([side]),
        torch.tensor([True]),
        torch.tensor([simultaneous]),
    )


# The target sequence is per physical riser, not merely the first side on a
# bilateral same-riser completion: arbitrary staircase length keeps toggling.
expected, started = -1, False
observed = []
for riser_index, side in enumerate((0, 1, 0, 1, 0, 1), start=1):
    event = transition(expected, started, side)
    observed.append((riser_index, side, event["seed_event"].item(), event["compliant_event"].item()))
    expected = int(event["expected_side"].item())
    started = bool(event["sequence_started"].item())
assert observed[0] == (1, 0, True, False)
assert all(compliant for _, _, _, compliant in observed[1:])
assert [side for _, side, _, _ in observed] == [0, 1, 0, 1, 0, 1]

# Call all six production wrappers. For an identical trajectory, RewardManager
# integration (wrapper output * dt) must be invariant to the control period.
wrapper_state = {
    "opposite_progress_delta": torch.tensor([0.6]),
    "same_side_progress_delta": torch.tensor([0.4]),
    "simultaneous_progress_delta": torch.tensor([0.2]),
    "compliant_event": torch.tensor([True]),
    "violation_event": torch.tensor([True]),
    "simultaneous_completion_event": torch.tensor([False]),
}
module.v13_alternation_state = lambda env, sensor_cfg=None: wrapper_state
for step_dt in (0.01, 0.02, 0.04):
    fake_env = types.SimpleNamespace(step_dt=step_dt, velocity_quality=torch.tensor([0.25]))
    outputs = (
        module.v13_opposite_progress_reward(fake_env),
        module.v13_velocity_coupled_opposite_progress_reward(
            fake_env, command_name="base_velocity", std=0.7071067811865476,
        ),
        module.v13_same_side_progress_reward(fake_env),
        module.v13_simultaneous_progress_reward(fake_env),
        module.v13_alternating_completion_reward(fake_env),
        module.v13_same_side_completion_reward(fake_env),
    )
    integrated = torch.stack(outputs).squeeze(1) * step_dt
    torch.testing.assert_close(
        integrated, torch.tensor([0.6, 0.15, 0.4, 0.2, 1.0, 1.0]),
    )

# The inherited +5 landing term is repointed to every safe one-leg/one-riser
# completion; it is not replaced by an alternation-only task reward.
v11_state_holder["state"] = fake_v11_state(1.0, (1.0, 0.0), landing_event=True, side=0)
assert module.v13_any_front_riser_completion_reward(types.SimpleNamespace()).item() == 1.0
torch.testing.assert_close(
    module.v13_any_front_riser_center_reward(types.SimpleNamespace()), torch.tensor([0.7]),
)

# Simultaneous completion cannot escape the existing -0.5 non-alternating
# completion settlement even when it occurs after staggered dense progress.
wrapper_state["violation_event"] = torch.tensor([False])
wrapper_state["simultaneous_completion_event"] = torch.tensor([True])
fake_env = types.SimpleNamespace(step_dt=0.02)
torch.testing.assert_close(
    module.v13_same_side_completion_reward(fake_env) * fake_env.step_dt,
    torch.tensor([1.0]),
)
wrapper_state["violation_event"] = torch.tensor([True])
wrapper_state["simultaneous_completion_event"] = torch.tensor([False])

# The requested ledger strongly prefers fast opposite progress over same-side
# or bilateral shortcuts while retaining V11's native velocity term unchanged.
def shaped_progress(opposite: float, same: float, velocity_quality: float) -> float:
    simultaneous = min(opposite, same)
    opposite_only = opposite - simultaneous
    same_only = same - simultaneous
    return (
        0.5 * opposite_only + 1.5 * velocity_quality * opposite_only
        - 0.75 * same_only - 0.5 * simultaneous
    )


assert shaped_progress(0.5, 0.0, 1.0) > shaped_progress(0.5, 0.0, 0.0) > 0.0
assert shaped_progress(0.0, 0.5, 1.0) < 0.0
assert shaped_progress(0.5, 0.5, 1.0) < shaped_progress(0.5, 0.0, 1.0)


# Both mirror sequences are legal; the first riser is an unpaid seed.
fl_seed = transition(-1, False, 0)
assert fl_seed["seed_event"].item() and not fl_seed["compliant_event"].item()
assert fl_seed["expected_side"].item() == 1
fr_second = transition(1, True, 1)
assert fr_second["compliant_event"].item() and fr_second["expected_side"].item() == 0
fl_third = transition(0, True, 0)
assert fl_third["compliant_event"].item() and fl_third["expected_side"].item() == 1

fr_seed = transition(-1, False, 1)
assert fr_seed["seed_event"].item() and fr_seed["expected_side"].item() == 0
fl_second = transition(0, True, 0)
assert fl_second["compliant_event"].item() and fl_second["expected_side"].item() == 1

# A wrong side is penalized, but the fixed riser parity still advances. If FL
# seeded B1, then B2 expects FR and B3 expects FL even when FL wrongly took B2.
repeat_fl = transition(1, True, 0)
assert repeat_fl["violation_event"].item() and not repeat_fl["compliant_event"].item()
assert repeat_fl["expected_side"].item() == 0

# A later simultaneous completion is never compliant, but it consumes that
# physical riser and advances the already-fixed parity. It cannot seed B1.
sim_later = transition(1, True, 0, simultaneous=True)
assert sim_later["simultaneous_completion_event"].item()
assert not sim_later["compliant_event"].item() and not sim_later["violation_event"].item()
assert sim_later["expected_side"].item() == 0
sim_first = transition(-1, False, 0, simultaneous=True)
assert not sim_first["seed_event"].item() and not sim_first["sequence_started"].item()

# Reacquiring one world-space face cannot mint another completion event.
same = module.v13_distinct_riser_event(
    torch.tensor([True]), torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 2.0]]),
    torch.tensor([True]), minimum_separation=0.15,
)
next_riser = module.v13_distinct_riser_event(
    torch.tensor([True]), torch.tensor([[1.30, 2.0]]), torch.tensor([[1.0, 2.0]]),
    torch.tensor([True]), minimum_separation=0.15,
)
assert not same.item() and next_riser.item()

# Exact reward ledger and wiring. Old late-window counter-rewards/schedules are
# forbidden because they suppress the native V11 mechanism that produced emergence.
required_cfg_fragments = {
    "v13_opposite_progress": "weight=0.5",
    "v13_velocity_coupled_opposite_progress": "weight=1.5",
    "v13_same_side_progress": "weight=-0.75",
    "v13_simultaneous_progress": "weight=-0.5",
    "v13_alternating_completion": "weight=1.0",
    "v13_same_side_completion": "weight=-0.5",
}


def assert_reward_ledger(config_text: str) -> None:
    for reward_name, weight_fragment in required_cfg_fragments.items():
        assert reward_name in config_text
        reward_block = config_text.split(reward_name, 1)[1].split(")", 1)[0]
        assert weight_fragment in reward_block


assert_reward_ledger(v13_cfg)
assert 'front_completion_mode = "alternating_risers"' in v13_cfg
assert "soft_alternation = True" in v13_cfg
assert "soft_alternation_curriculum_gate = False" in v13_cfg
assert "landing_event.func = mdp.v13_any_front_riser_completion_reward" in v13_cfg
assert "landing_center.func = mdp.v13_any_front_riser_center_reward" in v13_cfg
curriculum_source = CURRICULUMS.read_text(encoding="utf-8")
assert 'getattr(cfg, "soft_alternation_curriculum_gate", True)' in curriculum_source
# Prove that this acceptance check turns red on a material weight drift.
try:
    assert_reward_ledger(v13_cfg.replace("weight=1.5", "weight=1.4", 1))
except AssertionError:
    pass
else:
    raise AssertionError("V13 reward-ledger test failed to reject a wrong coupling weight.")
for forbidden in (
    "v13_alternating_riser_guidance",
    "v13_expected_stage_guidance",
    "v13_sequence_focus_linear_velocity",
    "v13_sequence_focus_forward_progress",
    "guidance_start_iteration",
    "focus_start_iteration",
):
    assert forbidden not in v13_cfg

# V13 contains no termination hook. The word may appear in documentation, so
# inspect executable calls instead of relying on a brittle substring test.
tree = ast.parse(v13_source)
for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
    if isinstance(call.func, ast.Name):
        called = call.func.id
    elif isinstance(call.func, ast.Attribute):
        called = call.func.attr
    else:
        called = ""
    assert "termination" not in called.lower()

print("V13_NATIVE_V11_FAST_ALTERNATION_CONTRACT: PASS")
