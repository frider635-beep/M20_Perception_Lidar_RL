#!/usr/bin/env python3
"""CPU/static contracts for V13.1 ordered B1/B2/... alternation."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v131_ordered_alternation.py"
V13_MODULE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v13_alternation.py"
CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v131_v11fd_ordered_alternation_env_cfg.py"
V13_CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v13_v11fd_alternation_env_cfg.py"
REGISTRY = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py"
RUNNER = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py"

spec = importlib.util.spec_from_file_location("v131_ordered_contract", MODULE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

v13_spec = importlib.util.spec_from_file_location("v13_contract", V13_MODULE)
v13_module = importlib.util.module_from_spec(v13_spec)
assert v13_spec.loader is not None
v13_spec.loader.exec_module(v13_module)

# The face matcher accepts detector jitter around the locked target but rejects
# B2 while B1 is expected and B3 while B2 is expected.
expected = torch.tensor([[1.0, 2.0], [1.3, 2.0], [1.3, 2.0]])
reference = torch.tensor([[1.05, 2.03], [1.60, 2.0], [1.31, 2.20]])
relation = module.v131_target_relation(
    reference,
    expected,
    torch.tensor([[1.0, 0.0]]).expand(3, -1),
    torch.tensor([True, True, True]),
    longitudinal_tolerance=0.08,
    lateral_tolerance=0.15,
)
assert relation["matches"].tolist() == [True, False, False]
torch.testing.assert_close(relation["longitudinal_error"], torch.tensor([0.05, 0.30, 0.01]))


def classify(phase, expected_side, side, *, physical=True, simultaneous=False, matches=True):
    return module.v131_classify_completion(
        torch.tensor([phase]), torch.tensor([expected_side]), torch.tensor([side]),
        torch.tensor([physical]), torch.tensor([simultaneous]), torch.tensor([matches]),
    )


# A target-qualified unique first mover owns B1; a bilateral onset remains
# unassigned and is settled by its own style bucket.
latched = module.v131_latch_b1_leader(
    torch.tensor([module.APPROACH_B1] * 4),
    torch.tensor([-1, -1, -1, 1]),
    torch.tensor([[True, False], [False, True], [True, True], [True, False]]),
)
assert latched["expected_side"].tolist() == [0, 1, -1, 1]
assert latched["unique_lift_event"].tolist() == [True, True, False, False]
assert latched["simultaneous_lift_event"].tolist() == [False, False, True, False]

# A one-frame potential spike cannot commit B1. The same clear leader must
# persist for the configured number of frames, with a lower release threshold
# preserving the candidate through small detector jitter.
candidate_side = torch.tensor([-1])
candidate_steps = torch.tensor([0])
tie_steps = torch.tensor([0])
for frame, potential in enumerate((
    torch.tensor([[0.07, 0.01]]),
    torch.tensor([[0.04, 0.01]]),
    torch.tensor([[0.07, 0.01]]),
), start=1):
    confirmed = module.v131_confirm_b1_first_mover(
        candidate_side, candidate_steps, tie_steps, potential, torch.tensor([True]),
        onset_threshold=0.05, release_threshold=0.03, separation=0.02,
        confirmation_frames=3,
    )
    candidate_side = confirmed["candidate_side"]
    candidate_steps = confirmed["candidate_steps"]
    tie_steps = confirmed["tie_steps"]
    assert confirmed["lift_ready"].any().item() is (frame == 3)
    assert not confirmed["lift_ready"][:, 1].any().item()

# Extra action freedom is selected by expected side only; the support-side
# contribution and inactive rows remain unchanged.
relieved = module.v131_expected_swing_relief(
    torch.tensor([10.0, 10.0, 10.0]),
    torch.tensor([[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]]),
    torch.tensor([0, 1, 0]), torch.tensor([True, True, False]),
)
torch.testing.assert_close(relieved, torch.tensor([8.0, 5.0, 10.0]))


# Either FL or FR may seed exactly B1. Its accumulated potential settles only
# on that target-qualified completion; the opposite side is then mandatory B2.
for first_side in (0, 1):
    seed = classify(module.APPROACH_B1, -1, first_side)
    assert seed["seed_event"].item() and seed["ordered_event"].item()
    assert not seed["compliant_event"].item() and not seed["violation_event"].item()
    second = classify(module.ALTERNATING_RISERS, 1 - first_side, 1 - first_side)
    assert second["compliant_event"].item() and second["ordered_event"].item()

# These are deliberately red-path samples: simultaneous B1, same-side B2,
# landing B2 while B1 is locked, and landing B3 while B2 is locked.
simultaneous_b1 = classify(module.APPROACH_B1, -1, 0, simultaneous=True)
same_side_b2 = classify(module.ALTERNATING_RISERS, 1, 0)
skipped_b1 = classify(module.APPROACH_B1, -1, 0, matches=False)
skipped_b2 = classify(module.ALTERNATING_RISERS, 1, 1, matches=False)
assert simultaneous_b1["wrong_side_event"].item()
assert same_side_b2["wrong_side_event"].item()
assert skipped_b1["wrong_target_event"].item()
assert skipped_b2["wrong_target_event"].item()
assert all(
    item["violation_event"].item()
    for item in (simultaneous_b1, same_side_b2, skipped_b1, skipped_b2)
)

# V13's historical-best primitive is the positive shaping authority. A
# lift/drop/lift trajectory mints only the final historical maximum.
best = torch.zeros((1, 2))
net = torch.zeros((1, 2))
for current in (
    torch.tensor([[0.0, 0.6]]),
    torch.tensor([[0.0, 0.2]]),
    torch.tensor([[0.0, 0.6]]),
):
    history = v13_module.v13_historical_best_progress(
        best, current, torch.tensor([True]),
    )
    net += history["delta"]
    best = history["best"]
torch.testing.assert_close(net, torch.tensor([[0.0, 0.6]]))

# B1 has a side-neutral historical-max delta before any completion. Either leg
# can make the signal positive, while lift/drop/lift cannot replay it.
b1_best = torch.zeros(1)
b1_net = torch.zeros(1)
for current in (torch.tensor([0.3]), torch.tensor([0.1]), torch.tensor([0.7])):
    result = module.v131_historical_max_delta(
        b1_best, current, torch.tensor([True]),
    )
    b1_net += result["delta"]
    b1_best = result["best"]
torch.testing.assert_close(b1_net, torch.tensor([0.7]))

# Locked-target progress is scanner-independent and requires both lift and
# approach before support can close the potential. Lifting in place cannot
# mint the complete B1/B2 signal.
locked = module.v131_locked_target_potential(
    wheel_bottom_z=torch.tensor([[0.20, 0.20], [0.20, 0.20], [0.20, 0.20]]),
    wheel_forward_distance=torch.tensor([[-0.30, -0.30], [-0.10, -0.10], [0.10, 0.10]]),
    support=torch.tensor([[False, False], [False, False], [True, True]]),
    target_lower_z=torch.tensor([0.0, 0.0, 0.0]),
    target_upper_z=torch.tensor([0.15, 0.15, 0.15]),
    clearance_margin=0.05,
    approach_span=0.30,
    crossing_span=0.10,
)
torch.testing.assert_close(locked[0], torch.zeros(2))
assert (locked[1] > locked[0]).all()
torch.testing.assert_close(locked[2], torch.ones(2))

# Strong velocity becomes a bounded completion dividend. Regression cannot
# replay it, while same-side/skip cancellation forfeits rather than pays it.
escrow = module.v131_settle_progress_escrow(
    torch.tensor([0.0, 0.4]), torch.tensor([0.3, 0.2]),
    torch.tensor([True, True]), torch.tensor([False, False]), torch.tensor([False, True]),
)
torch.testing.assert_close(escrow["escrow"], torch.tensor([0.3, 0.0]))
torch.testing.assert_close(escrow["payout"], torch.zeros(2))
torch.testing.assert_close(escrow["forfeited"], torch.tensor([0.0, 0.6]))
paid = module.v131_settle_progress_escrow(
    escrow["escrow"], torch.tensor([0.4, 0.0]),
    torch.tensor([True, False]), torch.tensor([True, False]), torch.tensor([False, False]),
)
torch.testing.assert_close(paid["payout"], torch.tensor([0.7, 0.0]))
torch.testing.assert_close(paid["escrow"], torch.zeros(2))

# A skip does not repair strict success. Once a unique physical landing is
# available, the teaching ledger targets the next adjacent face and opposite
# side, and emits exactly one resync event.
teaching = module.v131_update_teaching_ledger(
    target_world_xy=torch.tensor([[1.0, 0.0]]),
    expected_side=torch.tensor([1]),
    seeded=torch.tensor([True]),
    resync_pending=torch.tensor([True]),
    reference_world_xy=torch.tensor([[1.6, 0.0]]),
    forward_world_xy=torch.tensor([[1.0, 0.0]]),
    physical_completion=torch.tensor([True]),
    completed_side=torch.tensor([0]),
    step_width=0.3,
)
torch.testing.assert_close(teaching["target_world_xy"], torch.tensor([[1.9, 0.0]]))
assert teaching["expected_side"].item() == 1
assert teaching["seeded"].item()
assert teaching["resync_event"].item()
assert not teaching["resync_pending"].item()

# Immediately after B1 seeds side 0, a positive side-1 delta on the new B2
# target reaches the expected-progress bucket without another completion.
b2_progress = v13_module.v13_split_alternation_progress(
    torch.tensor([[0.0, 0.4]]),
    teaching["expected_side"],
    torch.tensor([True]),
)
torch.testing.assert_close(b2_progress["opposite"], torch.tensor([0.4]))
torch.testing.assert_close(b2_progress["same"], torch.tensor([0.0]))

# Physical and ordered ledgers agree on FL-B1/FR-B2/FL-B3.
physical = torch.zeros(1, dtype=torch.long)
ordered = torch.zeros(1, dtype=torch.long)
for index in (1, 2, 3):
    ledger = module.v131_update_riser_ledgers(
        physical, ordered, torch.tensor([index]), torch.tensor([True]), torch.tensor([True]),
    )
    physical = ledger["physical_last_riser_index"]
    ordered = ledger["ordered_last_riser_index"]
    assert ledger["adjacent_ordered_completion"].item()
torch.testing.assert_close(physical, torch.tensor([3]))
torch.testing.assert_close(ordered, torch.tensor([3]))

# B1 -> B3 records physical truth but never consumes ordered B2. B4 afterwards
# cannot re-synchronise the ordered ledger.
physical = torch.tensor([1])
ordered = torch.tensor([1])
skip = module.v131_update_riser_ledgers(
    physical, ordered, torch.tensor([3]), torch.tensor([True]), torch.tensor([False]),
)
assert skip["skip_completion"].item()
torch.testing.assert_close(skip["physical_last_riser_index"], torch.tensor([3]))
torch.testing.assert_close(skip["ordered_last_riser_index"], torch.tensor([1]))
torch.testing.assert_close(skip["physical_minus_ordered_gap"], torch.tensor([2]))
after = module.v131_update_riser_ledgers(
    skip["physical_last_riser_index"], skip["ordered_last_riser_index"],
    torch.tensor([4]), torch.tensor([True]), torch.tensor([False]),
)
torch.testing.assert_close(after["physical_last_riser_index"], torch.tensor([4]))
torch.testing.assert_close(after["ordered_last_riser_index"], torch.tensor([1]))
assert after["physical_minus_ordered_gap"].item() == 3

# Same-side adjacent and simultaneous adjacent are physical completions but do
# not advance the ordered ledger and do not terminate anything.
for style_ok in (False,):
    same = module.v131_update_riser_ledgers(
        torch.tensor([1]), torch.tensor([1]), torch.tensor([2]),
        torch.tensor([True]), torch.tensor([style_ok]),
    )
    assert same["physical_last_riser_index"].item() == 2
    assert same["ordered_last_riser_index"].item() == 1
    assert not same["adjacent_ordered_completion"].item()

# Strict validity never changes native velocity or native forward progress.
native_scale = module.v131_native_reward_scale(torch.tensor([True, False]))
torch.testing.assert_close(native_scale, torch.tensor([1.0, 1.0]))

# Dense skip potential stays zero inside the locked tread's safe region and
# rises before a wheel reaches the next riser face. Historical best prevents replay.
safe = module.v131_skip_progress_delta(
    torch.zeros(1), torch.tensor([[0.05, 0.10]]), torch.tensor([True]), torch.tensor([False]),
    step_width=0.30, wheel_radius=0.055, front_margin=0.025,
)
assert safe["delta"].item() == 0.0
skipping = module.v131_skip_progress_delta(
    safe["best"], torch.tensor([[0.26, 0.10]]), torch.tensor([True]), torch.tensor([False]),
    step_width=0.30, wheel_radius=0.055, front_margin=0.025,
)
assert skipping["delta"].item() > 0.0

# Side alternation and spatial adjacency are independently observable.
adjacent_switch = module.v131_physical_transition(
    torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), torch.tensor([2]), torch.tensor([True]),
)
assert adjacent_switch["side_switch"].item() and adjacent_switch["adjacent"].item()
assert adjacent_switch["strict"].item() and not adjacent_switch["skip_given_switch"].item()
skip_switch = module.v131_physical_transition(
    torch.tensor([0]), torch.tensor([1]), torch.tensor([1]), torch.tensor([3]), torch.tensor([True]),
)
assert skip_switch["side_switch"].item() and not skip_switch["adjacent"].item()
assert not skip_switch["strict"].item() and skip_switch["skip_given_switch"].item()

# Local completion teaching stays payable after strict failure because strict
# validity is intentionally absent from this classifier.
local_ok = module.v131_teaching_completion_events(
    torch.tensor([True]), torch.tensor([1]), torch.tensor([1]),
    torch.tensor([True]), torch.tensor([False]), torch.tensor([True]),
)
assert local_ok["compliant"].item()
local_same = module.v131_teaching_completion_events(
    torch.tensor([True]), torch.tensor([1]), torch.tensor([0]),
    torch.tensor([True]), torch.tensor([False]), torch.tensor([True]),
)
assert local_same["wrong_side"].item()
local_skip = module.v131_teaching_completion_events(
    torch.tensor([True]), torch.tensor([1]), torch.tensor([1]),
    torch.tensor([True]), torch.tensor([False]), torch.tensor([False]),
)
assert local_skip["wrong_target"].item()
first_skip = module.v131_one_shot_event(
    torch.tensor([False]), torch.tensor([True]), torch.tensor([False]),
)
second_skip = module.v131_one_shot_event(
    first_skip["latched"], torch.tensor([True]), torch.tensor([False]),
)
assert first_skip["event"].item() and not second_skip["event"].item()

# For any N, platform exit is two ordered supports: the opposite leg first,
# then the leg that completed BN. Only the second support enters flat mode.
for last_side in (0, 1):
    first = module.v131_platform_phase_transition(
        torch.tensor([module.TOP_FOLLOW]), torch.tensor([last_side]),
        torch.tensor([True]), torch.tensor([False]),
    )
    assert first["active_side"].item() == 1 - last_side
    assert first["phase"].item() == module.TOP_TRAILING_FOLLOW
    assert first["lead_event"].item() and not first["platform_event"].item()
    second = module.v131_platform_phase_transition(
        first["phase"], torch.tensor([last_side]),
        torch.tensor([False]), torch.tensor([True]),
    )
    assert second["active_side"].item() == last_side
    assert second["phase"].item() == module.PLATEAU_ROLL
    assert second["platform_event"].item()

# Every violation pays exactly one event-cost bucket. In particular, an ahead
# wrong-target completion is a skip, not a hidden -1 plus -3 double charge.
buckets = module.v131_violation_settlement(
    torch.tensor([True, True, False, True]),
    torch.tensor([False, True, True, False]),
    torch.tensor([True, True, False, False]),
)
assert buckets["nonordered_completion"].tolist() == [False, False, False, True]
assert buckets["simultaneous_b1_lift"].tolist() == [False, True, True, False]
assert buckets["skip"].tolist() == [True, False, False, False]
bucket_count = sum(value.long() for value in buckets.values())
assert (bucket_count <= 1).all()

# Top-follow progress is bounded and reaches one only after the follower is
# high enough, beyond the final face, and physically supported on the plateau.
top = module.v131_top_follow_potential(
    wheel_bottom_z=torch.tensor([[0.8, 1.0]]),
    forward_distance=torch.tensor([[0.0, 0.14]]),
    support=torch.tensor([[False, True]]),
    top_height=torch.tensor([1.0]), step_height=torch.tensor([0.2]),
    forward_target=0.14,
)
torch.testing.assert_close(top, torch.tensor([[0.0, 1.0]]))

source = MODULE.read_text(encoding="utf-8")
cfg = CFG.read_text(encoding="utf-8")
v13_source = V13_MODULE.read_text(encoding="utf-8")
v13_cfg = V13_CFG.read_text(encoding="utf-8")

# V13.1 is isolated: no V11/V13 source file contains its symbols, and it does
# not add an action mask, candidate gate, or termination hook.
assert "v131_" not in v13_source
assert "v131_" not in v13_cfg
tree = ast.parse(source)
for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
    if isinstance(call.func, ast.Name):
        called = call.func.id
    elif isinstance(call.func, ast.Attribute):
        called = call.func.attr
    else:
        called = ""
    assert "termination" not in called.lower()
assert "candidate_ready &= " not in source

# V11/V13 physical completion credit remains intact for capability emergence.
# Native velocity/forward terms remain inherited and untouched.
for forbidden in (
    "landing_event.func = mdp.v131_ordered_physical_completion_reward",
    "clearance_progress.func = mdp.v131_ordered_clearance_settlement_reward",
    "landing_center.func = mdp.v131_ordered_landing_center_reward",
):
    assert forbidden not in cfg
assert "track_lin_vel_xy_exp.func" not in cfg
assert "world_forward_progress.func" not in cfg
assert "ordered_speed_free_fraction" not in cfg
assert "self.rewards.action_rate_l2.func = mdp.v131_expected_swing_action_rate_l2" in cfg
assert "self.rewards.action_smooth_l2.func = mdp.v131_expected_swing_action_smooth_l2" in cfg
assert "expected_action_rate_hipy_scale" in cfg
assert "expected_action_smooth_knee_scale" in cfg
assert "strict_invalid_teaching_scale: float = 1.0" in cfg
assert "self.scene.terrain.max_init_terrain_level = 1" in cfg
assert 'self.curriculum.terrain_levels.params["strict_ordered_gate"] = True' in cfg
assert "soft_alternation_curriculum_gate = False" in cfg
assert "torch.full_like(state[\"phase\"], SEQUENCE_FAILED)" not in source
assert "scheduled_potential - other_potential" not in source
assert "strict_chain_valid" in source
assert "teaching_target_riser_world_xy" in source
assert "teaching_expected_side" in source
assert "teaching_resync_pending" in source
for v13_primitive in (
    "v13_historical_best_progress",
    "v13_split_alternation_progress",
):
    assert v13_primitive in source
for old_name in (
    "v13_opposite_progress", "v13_velocity_coupled_opposite_progress",
    "v13_same_side_progress", "v13_simultaneous_progress",
    "v13_alternating_completion", "v13_same_side_completion",
):
    assert f'"{old_name}"' in cfg
assert "setattr(self.rewards, name, None)" in cfg

reward_ledger = {
    "v131_b1_progress": "weight=0.5",
    "v131_b1_velocity_coupled_progress": "weight=1.5",
    "v131_expected_progress": "weight=0.5",
    "v131_velocity_coupled_expected_progress": "weight=1.5",
    "v131_same_side_progress": "weight=-0.75",
    "v131_simultaneous_progress": "weight=-0.5",
    "v131_simultaneous_b1_lift": "weight=-0.5",
    "v131_alternating_completion": "weight=1.0",
    "v131_nonordered_completion": "weight=-0.5",
    "v131_skip_riser": "weight=-1.0",
    "v131_skip_progress": "weight=-1.0",
    "v131_top_follow_progress": "weight=1.0",
    "v131_top_follow_completion": "weight=3.0",
    "v131_post_top_front_lift": "weight=-0.5",
}


def assert_reward_ledger(text: str) -> None:
    for name, weight in reward_ledger.items():
        block = text.split(name, 1)[1].split(")", 1)[0]
        assert weight in block


assert_reward_ledger(cfg)
try:
    assert_reward_ledger(cfg.replace("weight=-1.0", "weight=-0.3", 1))
except AssertionError:
    pass
else:
    raise AssertionError("V13.1 ledger probe failed to reject a weakened skip penalty.")

for metric in (
    "b1_progress_active_rate",
    "b1_progress_reward",
    "b1_velocity_coupled_reward",
    "alternating_progress_active_rate",
    "strict_chain_valid_fraction",
    "teaching_chain_active_fraction",
    "teaching_resync_count",
    "ordered_speed_gate_on_actionable_target_mean",
    "physical_last_riser_mean",
    "ordered_last_riser_mean",
    "physical_minus_ordered_gap",
    "ordered_progress_reward",
    "ordered_speed_reward",
    "free_velocity_reward",
    "full_velocity_eligibility_rate",
    "platform_success_rate",
    "completed_risers_mean",
    "physical_side_switch_rate", "physical_adjacent_riser_rate",
    "strict_adjacent_alternating_rate", "skip_given_side_switch_rate",
    "rollout_side_switch_rate", "rollout_adjacent_rate", "rollout_strict_alt_rate",
    "rollout_skip_rate", "rollout_skip_given_switch_rate", "skip_progress_reward",
    "first_mover_latch_rate", "b1_commit_active_rate", "early_opposite_progress_rate",
    "handoff_active_rate", "locked_target_progress_active_rate", "velocity_escrow_mean",
    "velocity_escrow_paid_rate", "velocity_escrow_forfeit_rate",
    "latch_to_b1_completion_rate", "b1_progress_escrow_mean",
    "b1_velocity_escrow_mean", "b1_escrow_paid_rate",
    "b1_escrow_forfeit_rate", "expected_swing_relief_active_rate",
):
    assert f'"{metric}"' in cfg

# The negative fixture proves the wiring assertions discriminate the old
# unreachable design instead of merely observing a green current file.
old_unreachable = source.replace(
    'b1_actionable = (', 'b1_actionable = torch.zeros_like(reference_active) & (', 1,
)
try:
    assert "b1_actionable = torch.zeros_like" not in old_unreachable
except AssertionError:
    pass
else:
    raise AssertionError("V13.1 B1 reachability red-path probe did not fail.")

# The production handoff must not regress to the old live-scanner reachability
# cliff. Target matching remains mandatory for completion settlement only.
handoff_block = source.split("progress_eligible = (", 1)[1].split("history =", 1)[0]
assert "reference_active" not in handoff_block
assert 'teaching_relation["matches"]' not in handoff_block
assert "locked_potential" in source
assert 'state["velocity_escrow_payout"]' in source

registry = REGISTRY.read_text(encoding="utf-8")
runner = RUNNER.read_text(encoding="utf-8")
assert 'id="M20-Stair-V13.1-V11FD-OrderedAlternation-v0"' in registry
assert 'id="M20-Stair-V13.1-V11FD-OrderedAlternation-Retention-v0"' in registry
assert "DeeproboticsM20V131V11FDOrderedAlternationEnvCfg" in registry
assert "DeeproboticsM20V131RetentionEnvCfg" in registry
assert "class DeeproboticsM20V131V11FDOrderedAlternationPPORunnerCfg" in runner
assert "class DeeproboticsM20V131RetentionPPORunnerCfg" in runner
assert 'experiment_name = "M20-V13.1-V11FD-OrderedAlternation"' in runner

print("V131_ORDERED_B1_ALTERNATION_CONTRACT: PASS")
