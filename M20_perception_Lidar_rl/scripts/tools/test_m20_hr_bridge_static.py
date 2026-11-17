"""CPU-only contract tests for M20-Stair-HRBridge-v0."""

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/hr_bridge_contracts.py"
spec = importlib.util.spec_from_file_location("hr_bridge_contracts", CONTRACTS)
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)
STATE_BASE = contracts.STATE_BASE
STATE_HR_BRIDGE = contracts.STATE_HR_BRIDGE
STATE_BLIND_STAIR = contracts.STATE_BLIND_STAIR
base_action_authority_mask = contracts.base_action_authority_mask
bridge_actor_mask = contracts.bridge_actor_mask
compose_final_action = contracts.compose_final_action
event_scores = contracts.event_scores


def main():
    base = torch.arange(48, dtype=torch.float32).reshape(3, 16)
    bridge = torch.tensor([[20.0, 20.0, 20.0], [2.0, 3.0, 4.0], [9.0, 9.0, 9.0]])
    previous = torch.zeros_like(base)
    state = torch.tensor([STATE_BASE, STATE_HR_BRIDGE, STATE_BLIND_STAIR])
    final = compose_final_action(base, bridge, state, previous, 0.12)
    assert torch.equal(final[0], base[0])
    assert torch.equal(final[2], base[2])
    assert torch.equal(final[1, :9], base[1, :9])
    assert torch.equal(final[1, 12:], base[1, 12:])
    assert torch.allclose(final[1, 9:12], torch.full((3,), 0.12))
    base_mask = base_action_authority_mask(state)
    assert base_mask[1, 9:12].sum() == 0 and base_mask[1].sum() == 13
    assert base_mask[0].sum() == 16 and base_mask[2].sum() == 16
    assert torch.equal(bridge_actor_mask(state).flatten(), torch.tensor([0.0, 1.0, 0.0]))

    weights = {"hr_success": 3.4, "perception_miss": 6.8, "joint_success": 20.6}
    base_return = torch.tensor([79.9, -23.2, 79.9, -23.2])
    hr = torch.tensor([True, True, False, False])
    miss = torch.tensor([False, False, True, False])
    stair = torch.tensor([True, False, True, False])
    score = event_scores(base_return, hr, miss, stair, weights)
    assert score[0] == score.max()
    assert torch.isclose(score[1] - score[3], torch.tensor(weights["hr_success"]))
    assert score[0] - score[2] > weights["joint_success"]
    print("STATIC_ACTION_AUTHORITY=PASS")
    print("STATIC_PPO_MASK=PASS")
    print("STATIC_REWARD_ORDERING=PASS")


if __name__ == "__main__":
    main()
