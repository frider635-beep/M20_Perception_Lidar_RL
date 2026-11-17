"""Pure PyTorch contracts shared by simulation, PPO and deployment tests."""

import torch

STATE_BASE = 0
STATE_HR_BRIDGE = 1
STATE_BLIND_STAIR = 2
HR_INDEX = 3


def compose_final_action(base_action, bridge_action, state, previous_final_action, delta_limit):
    if base_action.shape[-1] != 16 or bridge_action.shape[-1] != 3:
        raise ValueError("expected Base 16-D and Bridge 3-D actions")
    final = base_action.clone()
    limited = previous_final_action[:, 9:12] + torch.clamp(
        bridge_action - previous_final_action[:, 9:12], -delta_limit, delta_limit
    )
    active = state == STATE_HR_BRIDGE
    final[:, 9:12] = torch.where(active[:, None], limited, base_action[:, 9:12])
    return final


def base_action_authority_mask(state):
    mask = torch.ones((state.shape[0], 16), device=state.device)
    mask[state == STATE_HR_BRIDGE, 9:12] = 0.0
    return mask


def bridge_actor_mask(state):
    return (state == STATE_HR_BRIDGE).float().unsqueeze(1)


def event_scores(base_return, hr_success, perception_miss, staircase_success, weights):
    """Build a diagnostic score with each environment event counted once.

    PPO uses separate Base and Bridge returns. JOINT_SUCCESS appears in both
    critics for credit assignment, but must not be added twice when presenting
    one episode-level score.
    """
    joint = hr_success & staircase_success
    return base_return + (
        hr_success.float() * weights["hr_success"]
        - perception_miss.float() * weights["perception_miss"]
        + joint.float() * weights["joint_success"]
    )
