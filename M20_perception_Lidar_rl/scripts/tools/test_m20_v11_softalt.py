#!/usr/bin/env python3
"""CPU contract tests for V11 SoftAlt event, reward and curriculum semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
LANDING = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_landing.py"
CONTINUOUS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py"
CONFIG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_softalt_env_cfg.py"
PPO = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py"
REGISTRATION = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py"

spec = importlib.util.spec_from_file_location("v11_softalt_landing_pure", LANDING)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def classify(previous: int, side: int):
    ready = torch.zeros((1, 2), dtype=torch.bool)
    ready[0, side] = True
    return module.v11_classify_soft_alternation(torch.tensor([previous]), ready)


def production_ready(previous: int, side: int):
    per_side = torch.zeros((1, 2), dtype=torch.bool)
    per_side[0, side] = True
    ready = module.v11_softalt_candidate_ready(
        per_side, per_side, per_side, torch.zeros_like(per_side),
        ~per_side, torch.tensor([previous >= 0]), torch.tensor([False]),
        soft_alternation=True,
    )
    return module.v11_classify_soft_alternation(torch.tensor([previous]), ready)


def run_sequence(sides: tuple[int, ...]) -> tuple[int, int, int]:
    previous = -1
    physical = alternating = repeats = 0
    for side in sides:
        result = classify(previous, side)
        physical += int(result["event"].item())
        alternating += int(result["alternating_completion"].item())
        repeats += int(result["same_side_repeat"].item())
        previous = int(result["completed_side"].item())
    return physical, alternating, repeats


# Either side can own the first physical completion.
fl_first = classify(-1, 0)
fr_first = classify(-1, 1)
assert fl_first["event"].item() and fl_first["first_completion"].item()
assert fl_first["completed_side"].item() == 0
assert fr_first["event"].item() and fr_first["first_completion"].item()
assert fr_first["completed_side"].item() == 1

# Opposite transitions are rewarded style events; repeats remain physical
# completions but are classified only after completion.
fl_fr = classify(0, 1)
fr_fl = classify(1, 0)
fl_fl = classify(0, 0)
fr_fr = classify(1, 1)
assert fl_fr["event"].item() and fl_fr["alternating_completion"].item()
assert fr_fl["event"].item() and fr_fl["alternating_completion"].item()
assert fl_fl["event"].item() and fl_fl["same_side_repeat"].item()
assert fr_fr["event"].item() and fr_fr["same_side_repeat"].item()
assert run_sequence((0, 1)) == (2, 1, 0)
assert run_sequence((1, 0)) == (2, 1, 0)
assert run_sequence((0, 0)) == (2, 0, 1)

# Exercise the same full candidate gate used by production.  The deliberately
# false anchor and ineligible style mask would reject these repeats in hard V11,
# but must not suppress SoftAlt physical completion.
fl_fl_production = production_ready(0, 0)
fr_fr_production = production_ready(1, 1)
assert fl_fl_production["event"].item() and fl_fl_production["same_side_repeat"].item()
assert fr_fr_production["event"].item() and fr_fr_production["same_side_repeat"].item()
assert not module.v11_softalt_termination(
    torch.tensor([[False, False]]), fl_fl_production["same_side_repeat"],
).item()
assert not module.v11_softalt_termination(
    torch.tensor([[False, False]]), fr_fr_production["same_side_repeat"],
).item()

# Simultaneous completion advances physics but can never receive an alternating
# bonus or be randomly mislabeled as a repeat.
both = module.v11_classify_soft_alternation(
    torch.tensor([0]), torch.tensor([[True, True]]),
)
assert both["event"].item()
assert both["simultaneous_completion"].item()
assert not both["alternating_completion"].item()
assert not both["same_side_repeat"].item()

# Physical completion is invariant to the style ledger.  The exact same FL
# candidate completes for first/expected-FL/expected-FR history values.
physical_sides = [classify(previous, 0)["completed_side"].item() for previous in (-1, 0, 1)]
assert physical_sides == [0, 0, 0]

# Repeats never terminate.  A debounced physical collision still does.
assert not module.v11_softalt_termination(
    torch.tensor([[False, False]]), torch.tensor([True]),
).item()
assert module.v11_softalt_termination(
    torch.tensor([[True, False]]), torch.tensor([False]),
).item()

# RewardManager multiplies raw * weight * dt.  Dt-normalization makes the
# configured weight the exact one-shot contribution.
dt = 0.02
event = torch.tensor([True])
reward_smoke = {}
for name, weight, expected in (
    ("first", 0.5, 0.5), ("alternating", 2.5, 2.5),
    ("repeat", -1.0, -1.0), ("simultaneous", -0.5, -0.5),
    ("platform", 3.0, 3.0),
):
    actual = module.v11_dt_normalized_event(event, dt) * weight * dt
    assert torch.allclose(actual, torch.tensor([expected]))
    reward_smoke[name] = actual.item()

# Promotion fails closed with no valid transitions, accepts the recommended
# boundary, and rejects excess repeat or any confirmed collision episode.
def quality(platform, compliance, repeat, collision, valid=True):
    return module.v11_softalt_quality_gate(
        torch.tensor([platform]), torch.tensor([compliance]), torch.tensor([repeat]),
        torch.tensor([collision]), torch.tensor([valid]),
        platform_threshold=0.75, compliance_threshold=0.80,
        repeat_threshold=0.20, collision_threshold=0.0,
    ).item()


assert quality(0.75, 0.80, 0.20, 0.0)
assert not quality(1.0, 1.0, 0.0, 0.0, valid=False)
assert not quality(1.0, 0.79, 0.21, 0.0)
assert not quality(1.0, 1.0, 0.0, 1.0)

# Reward scaling changes PPO authority, not the curriculum quality definition.
stair = torch.tensor([True, True, True, False])
scaled = torch.tensor([0.25, 0.15, 0.05, 1.0])
restored = module.v11_restore_stair_tracking_quality(scaled, stair, 0.25)
assert torch.allclose(restored, torch.tensor([1.0, 0.60, 0.20, 1.0]))
assert restored[1] > 0.50
assert restored[2] < 0.32

source = CONTINUOUS.read_text(encoding="utf-8")
config = CONFIG.read_text(encoding="utf-8")
ppo = PPO.read_text(encoding="utf-8")
registration = REGISTRATION.read_text(encoding="utf-8")
assert 'candidate_ready = v11_softalt_candidate_ready(' in source
assert 'gate &= landing["eligible_sides"]' in source
assert 'alternating and not bool(getattr(env.cfg.v11_perception, "soft_alternation", False))' in source
assert 'soft_alternation: bool = True' in config
assert 'stair_forward_command_min: float = 0.20' in config
assert 'stair_forward_command_max: float | None = 0.60' in config
assert '"stair_scale": 0.25' in config
assert 'simultaneous_completion_reward, weight=-0.5' in config
assert 'v11_softalt_world_forward_speed_deficit' in config
assert '"ascending_scale"] = 0.5 / 3.0' in config
assert 'v11_softalt_world_forward_progress_reward' in config
assert '"ascending_scale"] = 0.75 / 2.0' in config
assert '"ascending_only": True' in config
assert 'non_ascending_stair_forward_command_min: float = 0.10' in config
assert 'num_steps_per_env = 48' in ppo
assert 'learning_rate = 3.0e-4' in ppo
assert 'entropy_coef = 1.0e-3' in ppo
assert 'require_pretrained_checkpoint = True' in ppo
assert 'M20-Stair-SoftAlt-V11-v0' in registration
assert 'expected_side' not in config
assert 'phase clock' not in config.lower()
assert 'gru' not in config.lower()

print(f"V11_SOFTALT_REWARD_SMOKE={reward_smoke}")
print("V11_SOFTALT_SEQUENCE_SMOKE=FL->FR: physical=2 alternating=1 repeat=0 termination=false")
print("V11_SOFTALT_SEQUENCE_SMOKE=FR->FL: physical=2 alternating=1 repeat=0 termination=false")
print("V11_SOFTALT_SEQUENCE_SMOKE=FL->FL: physical=2 alternating=0 repeat=1 termination=false")
print("V11_SOFTALT_SEQUENCE_SMOKE=FR->FR: physical=2 alternating=0 repeat=1 termination=false")
print("V11_SOFTALT_SIMULTANEOUS_SMOKE=physical=true alternating=false simultaneous=true")
print("V11_SOFTALT_COLLISION_SMOKE=termination=true")
print("V11_SOFTALT_CURRICULUM_SMOKE=flat_perfect=1 stair_perfect=1 stair_0.60=0.60 stair_0.20=0.20")
print("V11_SOFTALT_CONTRACT: PASS")
