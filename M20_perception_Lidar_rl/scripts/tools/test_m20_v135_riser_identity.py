#!/usr/bin/env python3
"""Six deterministic contracts for V13.5 physical-riser identity."""
import importlib.util
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
path = root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v135_riser_identity.py"
spec = importlib.util.spec_from_file_location("v135_identity", path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def classify(seeded, expected, side, error=0.0, simultaneous=False):
    return m.v135_classify_riser_completion(
        torch.tensor([seeded]), torch.tensor([expected]), torch.tensor([side]),
        torch.tensor([True]), torch.tensor([simultaneous]),
        torch.tensor([error]), torch.tensor([0.0]),
        longitudinal_tolerance=0.08, lateral_tolerance=0.15)

fl_b1 = classify(False, -1, 0); fr_b1 = classify(False, -1, 1)
assert fl_b1["seed"].item() and fr_b1["seed"].item()
duplicate = classify(True, 1, 1, error=-0.30)
wrong = classify(True, 1, 0)
skip = classify(False, -1, 0, error=0.30)
dual_skip = classify(False, -1, 0, error=0.30, simultaneous=True)
assert duplicate["duplicate"].item()
assert wrong["wrong_side"].item()
assert skip["skip"].item() and not skip["seed"].item()
assert dual_skip["dual_skip"].item() and not dual_skip["seed"].item()
for event in (duplicate, wrong, skip, dual_skip):
    assert not event["valid"].item()

# Isaac Lab multiplies every configured reward term by dt.  A one-frame event
# must therefore return event/dt so the configured weight is the event value.
normalized = m.v135_event_rate(torch.tensor([True, False]), 0.02)
assert torch.equal(normalized, torch.tensor([50.0, 0.0]))
try:
    m.v135_event_rate(torch.tensor([True]), 0.0)
except ValueError:
    pass
else:
    raise AssertionError("non-positive step_dt must fail closed")

# B1 seeds phase without chain credit. Consecutive strict adjacent completions
# grow the chain, while any ledger error resets it without disabling relearning.
chain = torch.tensor([0, 2, 2], dtype=torch.long)
chain = m.v135_update_adjacent_chain(
    chain,
    torch.tensor([True, True, False]),
    torch.tensor([False, True, False]),
)
assert chain.tolist() == [1, 0, 2]
# An episode's strict score stays failed, while its live adjacent segment can
# earn ordered credit again after a physically correct B(k)->B(k+1) transition.
recovered = m.v135_update_adjacent_chain(
    torch.tensor([0, 0], dtype=torch.long),
    torch.tensor([True, False]),
    torch.tensor([False, False]),
)
assert recovered.tolist() == [1, 0]
factors = m.v135_select_gait_speed_factor(
    torch.tensor([0, 0, 1], dtype=torch.long),
    torch.tensor([False, True, False]),
    ordered=1.2, ordinary=1.0, skip=0.85,
)
assert torch.allclose(factors, torch.tensor([1.0, 0.85, 1.2]))
applied = m.v135_apply_gait_speed_factor(
    torch.tensor([True, True, True, True]),
    torch.tensor([True, True, True, True]),
    torch.tensor([True, True, True, False]),
    torch.tensor([True, False, False, False]),
    torch.tensor([False, False, True, False]),
    torch.tensor([1.2, 1.2, 0.85, 1.2]),
    skip=0.85,
)
# Matching Bk earns the multiplier; unmatched future riser is neutral; a
# confirmed skip stays discounted even while that future riser is in view.
assert torch.allclose(applied, torch.tensor([1.2, 1.0, 0.85, 1.0]))
# Final keeps one live target. Errors only invalidate strict episode success;
# they cannot shut down the live gait phase or subsequent learning.
clean = m.v135_update_episode_validity(
    torch.tensor([True]), torch.tensor([False]), torch.tensor([False]),
    torch.tensor([False]), torch.tensor([False]), torch.tensor([False]))
assert clean.item()
for flags in (
    (wrong["wrong_side"], torch.tensor([False]), torch.tensor([False]), torch.tensor([False]), torch.tensor([False])),
    (torch.tensor([False]), duplicate["duplicate"], torch.tensor([False]), torch.tensor([False]), torch.tensor([False])),
    (torch.tensor([False]), torch.tensor([False]), skip["skip"], torch.tensor([False]), torch.tensor([False])),
    (torch.tensor([False]), torch.tensor([False]), torch.tensor([False]), dual_skip["dual_skip"], torch.tensor([False])),
    (torch.tensor([False]), torch.tensor([False]), torch.tensor([False]), torch.tensor([False]), torch.tensor([True])),
):
    assert not m.v135_update_episode_validity(torch.tensor([True]), *flags).item()

# A correct completion wins when the old target is already behind in that frame.
behind = m.v135_irrecoverable_behind(
    torch.tensor([-0.31]), torch.tensor([True]), torch.tensor([True]), threshold=-0.30)
assert not behind.item()

# A locked B1 can settle after the live scanner has switched to B2, but only
# after the same front wheel safely crosses B1 and bears stable B1 support.
kwargs = dict(wheel_radius=0.05, front_margin=0.02, step_width=0.30,
              clearance_margin=0.01, support_tolerance=0.05,
              contact_threshold=20.0, lateral_tolerance=0.15,
              confirmation_frames=3)
crossed = torch.zeros((1, 2), dtype=torch.bool)
steps = torch.zeros((1, 2), dtype=torch.long)
def locked(previous, forward, bottom, force, upper=0.10):
    global crossed, steps
    crossed, steps, ready = m.v135_locked_target_completion_step(
        crossed, steps, torch.tensor([previous]), torch.tensor([forward]),
        torch.zeros((1, 2)), torch.tensor([bottom]), torch.tensor([force]),
        torch.tensor([upper]), torch.tensor([True]), **kwargs)
    return ready
assert not locked([-0.02, -0.02], [0.01, -0.02], [0.12, 0.12], [0.0, 0.0]).any()
for count in range(2):
    assert not locked([0.01, -0.02], [0.11, -0.02], [0.10, 0.10], [30.0, 0.0]).any()
assert locked([0.11, -0.02], [0.11, -0.02], [0.10, 0.10], [30.0, 0.0]).tolist() == [[True, False]]
# Future-riser height and position, or a crossing below B1 tread, never settle B1.
for bad_forward, bad_bottom in ((0.36, 0.10), (0.11, 0.22), (0.11, 0.02)):
    crossed.zero_(); steps.zero_()
    locked([-0.02, -0.02], [0.01, -0.02], [0.12, 0.12], [0.0, 0.0])
    for _ in range(3):
        assert not locked([0.01, -0.02], [bad_forward, -0.02],
                          [bad_bottom, 0.10], [30.0, 0.0]).any()
crossed.zero_(); steps.zero_()
for _ in range(3):
    assert not locked([0.01, -0.02], [0.11, -0.02], [0.10, 0.10], [30.0, 0.0]).any()

# Converting yaw-frame detections back to world XY keeps P1/P2 identity stable.
root_xy = torch.tensor([[0.0, 0.0]])
target_xy = torch.tensor([[0.30, 0.0]])
forward = torch.tensor([[1.0, 0.0]])
fixed_world = torch.tensor([[[0.30, 0.0], [0.60, 0.0], [0.90, 0.0]]])
for degrees in (0.0, 5.0, 10.0, 15.0, 20.0):
    yaw = torch.deg2rad(torch.tensor(degrees)); c, s = torch.cos(yaw), torch.sin(yaw)
    delta = fixed_world - root_xy[:, None, :]
    local = torch.stack((c * delta[..., 0] + s * delta[..., 1], -s * delta[..., 0] + c * delta[..., 1]), -1)
    reconstructed = torch.stack((
        root_xy[:, None, 0] + c * local[..., 0] - s * local[..., 1],
        root_xy[:, None, 1] + s * local[..., 0] + c * local[..., 1]), -1)
    match = m.v135_match_locked_target_edges(
        reconstructed, torch.ones((1, 3), dtype=torch.bool), root_xy, target_xy, forward,
        step_width=0.30, longitudinal_tolerance=0.08, lateral_tolerance=0.15)
    assert match["valid"].tolist() == [[True, True]]
    assert match["slot"].tolist() == [[0, 1]]
print("V135_PHYSICAL_RISER_IDENTITY_CONTRACT: PASS")
