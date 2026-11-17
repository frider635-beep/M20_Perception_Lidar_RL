#!/usr/bin/env python3
"""Regression checks for V11's forward-only, one-settlement clearance escrow."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
LANDING = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_landing.py"
CONTINUOUS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py"
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_continuous_env_cfg.py"

spec = importlib.util.spec_from_file_location("v11_landing_pure", LANDING)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def update(max_progress, escrow, previous_x, previous_x_valid, wheel_x, wheel_bottom, active):
    return module.v11_update_clearance_escrow(
        max_progress, escrow, previous_x, previous_x_valid, wheel_x, wheel_bottom,
        torch.tensor([[0.0, 0.0]]), torch.tensor([[0.10, 0.10]]), active, 0.01,
    )


zeros = torch.zeros((1, 2))
false = torch.zeros((1, 2), dtype=torch.bool)
true = torch.ones((1, 2), dtype=torch.bool)

# The first riser is side-neutral; after a completion only the opposite side
# can receive the next physical riser's credit.
first_open = module.v11_expected_front_mask(torch.tensor([-1]))
assert first_open.all()
first_fl = module.v11_latch_alternating_front_completion(
    torch.tensor([-1]), torch.tensor([[True, False]]),
)
assert first_fl["event"].item() and first_fl["completed_side"].item() == 0
assert first_fl["next_expected_side"].item() == 1
right_second = module.v11_latch_alternating_front_completion(
    first_fl["next_expected_side"], torch.tensor([[True, True]]),
)
assert right_second["event"].item() and right_second["completed_side"].item() == 1
assert right_second["next_expected_side"].item() == 0
wrong_second = module.v11_latch_alternating_front_completion(
    torch.tensor([1]), torch.tensor([[True, False]]),
)
assert not wrong_second["event"].item() and wrong_second["out_of_turn_ready"].item()

# The first physical completion seeds the sequence but pays nothing.  A later
# alternating completion pays once, while a reset sequence is fail-closed.
assert not module.v11_alternating_reward_event(
    torch.tensor([True]), torch.tensor([False]),
).item()
assert module.v11_alternating_reward_event(
    torch.tensor([True]), torch.tensor([True]),
).item()
assert not module.v11_alternating_reward_event(
    torch.tensor([False]), torch.tensor([False]),
).item()

# A held/reappearing copy of the credited face in slot 0 is rejected; the
# strictly forward slot 1 is selected.  With no forward face, selection fails
# closed instead of silently reusing the same reward target.
next_slot = module.v11_select_next_uncredited_riser(
    torch.tensor([[True, True]]), torch.tensor([[0.0, 0.30]]),
    torch.tensor([True]), minimum_separation=0.15,
)
assert next_slot.item() == 1
no_next = module.v11_select_next_uncredited_riser(
    torch.tensor([[True, False]]), torch.tensor([[0.0, 0.0]]),
    torch.tensor([True]), minimum_separation=0.15,
)
assert no_next.item() == -1

# A wheel may be high at the first sample, but without an observed forward x
# advance it must neither become clear nor accrue escrow.
stationary_high = update(zeros, zeros, zeros, false, torch.tensor([[0.0, 0.0]]), torch.tensor([[0.11, 0.11]]), true)
assert torch.equal(stationary_high["escrow"], zeros)
assert not stationary_high["clearance"].any()

# The same high wheel becomes eligible only while it advances toward the face.
advancing_high = update(
    stationary_high["max_progress"], stationary_high["escrow"], stationary_high["previous_x"],
    stationary_high["previous_x_valid"], torch.tensor([[0.03, 0.03]]), torch.tensor([[0.11, 0.11]]), true,
)
assert torch.allclose(advancing_high["escrow"], torch.ones((1, 2)))
assert advancing_high["clearance"].all()

# Holding the same high pose cannot create another payout.
held_high = update(
    advancing_high["max_progress"], advancing_high["escrow"], advancing_high["previous_x"],
    advancing_high["previous_x_valid"], torch.tensor([[0.03, 0.03]]), torch.tensor([[0.11, 0.11]]), true,
)
assert torch.allclose(held_high["escrow"], advancing_high["escrow"])

# A wheel bottom above the first tread but below the second tread cannot earn
# the second-riser clearance bit.  The threshold remains target-relative.
second_riser_short = module.v11_update_clearance_escrow(
    zeros, zeros, torch.tensor([[0.0, 0.0]]), true,
    torch.tensor([[0.03, 0.03]]), torch.tensor([[0.25, 0.25]]),
    torch.tensor([[0.17, 0.17]]), torch.tensor([[0.34, 0.34]]),
    true, clearance_margin=0.01,
)
assert not second_riser_short["clearance"].any()
assert torch.all(second_riser_short["max_progress"] < 1.0)

# Escrow is released only by the one-shot clearance -> crossing -> support
# landing event; selecting FL must not pay FR's escrow.
responsible = torch.tensor([0])
assert module.v11_settle_clearance_escrow(held_high["escrow"], responsible, torch.tensor([False])).item() == 0.0
assert module.v11_settle_clearance_escrow(held_high["escrow"], responsible, torch.tensor([True])).item() == 1.0

# V11's first front wheel is only an internal latch.  It cannot release the
# main landing/clearance/pose rewards until the second front wheel completes.
first_only = module.v11_front_reward_event(
    torch.tensor([True]), torch.tensor([False]), defer_until_bilateral=True,
)
both_complete = module.v11_front_reward_event(
    torch.tensor([False]), torch.tensor([True]), defer_until_bilateral=True,
)
assert not first_only.item()
assert both_complete.item()
bilateral_escrow = torch.tensor([[0.4, 0.7]])
assert module.v11_settle_bilateral_escrow(
    bilateral_escrow, first_only,
).item() == 0.0
assert torch.allclose(
    module.v11_settle_bilateral_escrow(bilateral_escrow, both_complete),
    torch.tensor([1.1]),
)
assert torch.allclose(
    module.v11_settle_bilateral_escrow(bilateral_escrow, both_complete, average=True),
    torch.tensor([0.55]),
)

# Two classifier spikes are ignored, while three consecutive face-contact frames
# confirm the failure.  Releasing contact resets the debounce streak.
contact_1 = module.v11_confirm_vertical_contact(
    torch.zeros((1, 2), dtype=torch.long),
    torch.tensor([[True, False]]),
    confirmation_frames=3,
)
assert torch.equal(contact_1["steps"], torch.tensor([[1, 0]]))
assert not contact_1["confirmed"].any()
contact_2 = module.v11_confirm_vertical_contact(
    contact_1["steps"], torch.tensor([[True, False]]), confirmation_frames=3,
)
assert not contact_2["confirmed"].any()
contact_3 = module.v11_confirm_vertical_contact(
    contact_2["steps"], torch.tensor([[True, False]]), confirmation_frames=3,
)
assert torch.equal(contact_3["confirmed"], torch.tensor([[True, False]]))
released = module.v11_confirm_vertical_contact(
    contact_2["steps"], torch.tensor([[False, False]]), confirmation_frames=3,
)
assert torch.equal(released["steps"], torch.zeros((1, 2), dtype=torch.long))

# Platform support requires both rear wheels.  They may establish support in
# sequence, but only inside the configured bounded window.
rear_1 = module.v11_update_bilateral_support_window(
    torch.zeros((1, 2), dtype=torch.bool), torch.zeros(1, dtype=torch.long),
    torch.tensor([[True, False]]), torch.tensor([True]), window_frames=15,
)
assert not rear_1["ready"].item()
rear_2 = module.v11_update_bilateral_support_window(
    rear_1["latched"], rear_1["age"],
    torch.tensor([[False, True]]), torch.tensor([True]), window_frames=15,
)
assert rear_2["ready"].item()

# Mobility relief is actor-visible, near-face and fail-closed per front leg.
mobility = module.v11_front_step_mobility_gate(
    actor_b1=torch.tensor([True, True, False]),
    target_active=torch.tensor([True, True, True]),
    wheel_x=torch.tensor([[0.15, 0.15], [0.15, 0.15], [0.15, 0.15]]),
    riser_x=torch.tensor([0.30, 0.30, 0.30]),
    candidate_ready=torch.tensor([[False, True], [False, False], [False, False]]),
    vertical_contact=torch.tensor([[False, False], [True, False], [False, False]]),
    wheel_radius=0.09,
)
assert torch.equal(
    mobility,
    torch.tensor([[True, False], [False, True], [False, False]]),
)
far = module.v11_front_step_mobility_gate(
    actor_b1=torch.tensor([True]),
    target_active=torch.tensor([True]),
    wheel_x=torch.tensor([[0.0, 0.0]]),
    riser_x=torch.tensor([0.30]),
    candidate_ready=torch.tensor([[False, False]]),
    vertical_contact=torch.tensor([[False, False]]),
    wheel_radius=0.09,
)
assert not far.any()

# The relative hip-y term is bounded, needs real hip-y pose participation and
# cannot be maximized by either a knee-only or an excessive hip-y-only pose.
relative = module.v11_bounded_relative_hipy_participation(
    hipy_deviation=torch.tensor([[0.20, 0.0], [0.0, 0.40], [0.40, 0.40]]),
    knee_deviation=torch.tensor([[0.20, 0.40], [0.40, 0.0], [0.0, 0.0]]),
    active=torch.tensor([[True, False], [True, True], [True, False]]),
    target_share=0.50,
    share_half_width=0.10,
    activation_scale=0.20,
)
assert 0.0 < relative[0].item() <= 1.0
assert relative[1].item() == 0.0
assert relative[2].item() == 0.0
assert torch.all((relative >= 0.0) & (relative <= 1.0))

# Latest-value escrow prevents a brief good pose from surviving a later
# knee-only reversion; inactive legs retain their last eligible value.
pose_escrow = torch.zeros((1, 2))
pose_escrow = module.v11_update_latest_participation_escrow(
    pose_escrow, torch.tensor([[0.9, 0.7]]), torch.tensor([[True, True]]),
)
pose_escrow = module.v11_update_latest_participation_escrow(
    pose_escrow, torch.tensor([[0.2, 0.1]]), torch.tensor([[True, False]]),
)
assert torch.allclose(pose_escrow, torch.tensor([[0.2, 0.7]]))
assert module.v11_settle_clearance_escrow(
    pose_escrow, torch.tensor([0]), torch.tensor([False]),
).item() == 0.0
assert torch.allclose(
    module.v11_settle_clearance_escrow(pose_escrow, torch.tensor([0]), torch.tensor([True])),
    torch.tensor([0.2]),
)

# Necessary knee flexion is free; only the active-side excess is penalized and
# the cost stays bounded even for extreme joint deviations.
knee_excess = module.v11_bounded_knee_excess(
    knee_deviation=torch.tensor([[0.55, 0.80], [5.0, 5.0]]),
    active=torch.tensor([[True, True], [True, False]]),
    free_band=0.55,
    excess_scale=0.25,
)
assert torch.allclose(knee_excess["cost"], torch.tensor([0.5, 1.0]))
assert torch.allclose(knee_excess["rate"], torch.tensor([0.5, 1.0]))
assert torch.all((knee_excess["cost"] >= 0.0) & (knee_excess["cost"] <= 1.0))

source = CONTINUOUS.read_text(encoding="utf-8")
config = CONFIG.read_text(encoding="utf-8")
assert '"credited_reference_valid"' in source
assert "same_credited_riser" in source
assert "v11_settle_clearance_escrow" in source
assert "v11_phase_relaxed_action_rate_l2" in source
assert "v11_phase_relaxed_action_smooth_l2" in source
assert "v11_phase_relaxed_front_joint_pos_penalty" in source
assert "v11_front_hipy_participation_settlement_reward" in source
assert "v11_front_knee_excess_penalty" in source
assert 'params={"hipy_active_scale": 0.35, "knee_active_scale": 1.00}' in config
assert 'params={"hipy_active_scale": 0.50, "knee_active_scale": 0.80}' in config
assert '"active_scale": 0.80' in config
assert 'front_hipy_participation_settlement = RewTerm' in config
assert 'front_knee_excess = RewTerm' in config
assert 'require_bilateral_front_completion: bool = True' in config
assert 'front_completion_mode: str = "alternating_risers"' in config
assert 'alternating_clearance_settlement_scale: float = 2.0' in config
assert 'defer_first_front_rewards_until_bilateral: bool = True' in config
assert 'vertical_contact_confirmation_frames: int = 3' in config
assert 'require_bilateral_rear_support: bool = True' in config
assert 'front_riser_contact = RewTerm' in config
assert 'front_riser_collision = DoneTerm' in config
assert 'v11_perception_miss = RewTerm' not in config
assert '"front_collision_event"' in source
assert 'def v11_front_collision_metric' in source
assert 'def v11_perception_miss_reward' not in source
assert 'landing["bilateral_landing_event"]' in source
assert 'def _v11_alternating_landing_state' in source
assert 'v11_select_next_uncredited_riser' in source
assert 'state["expected_side"]' in source
assert 'for slot in range(2):' in source
assert 'v11_alternating_reward_event(event, state["sequence_started"])' in source
assert 'def v11_event_reachability_metric' in source
assert 'landing["candidate_ready"].all(dim=1)' in source
assert 'wheel_yaw[..., 1] >= region["y_min"][:, None]' in source
assert '"free_band": 0.55' in config
assert '"excess_scale": 0.25' in config
assert '"target_share": 0.50' in config
assert '"share_half_width": 0.10' in config
assert 'v11_front_upperleg_torso_angle' in config
assert "front_action_mirror = RewTerm" not in config
assert "rear_action_mirror = RewTerm" not in config
assert "wheel_action_mirror = RewTerm" not in config
assert "clearance_progress = RewTerm(func=mdp.v11_clearance_progress_reward, weight=0.5)" in config
assert "stair_straightness = RewTerm(func=mdp.v11_stair_straightness_penalty, weight=0.05)" in config

print("V11_CLEARANCE_ESCROW_REGRESSION: PASS")
