"""V13.5-Final Critic Reward State: reflects actual V135 reward lifecycle.

Replaces legacy v13_critic_reward_state which reads ordered_chain_failed (no longer exists).

6D Critic observation:
- expected_FL, expected_FR: which leg is expected next
- seeded: whether B1 acquisition happened
- target_active: whether we're tracking a valid target
- current_potential: current-frame potential (FL or FR depending on seeded)
- best_potential: historical best potential
"""

from __future__ import annotations

import torch


def v135_final_critic_reward_state(env, sensor_cfg=None) -> torch.Tensor:
    """6D critic state reflecting V135 final reward lifecycle.

    Returns:
        [num_envs, 6]: [expected_FL, expected_FR, seeded, target_active,
                        current_potential, best_potential]
    """
    from .v135_riser_identity import v135_riser_identity_state
    from .v135_final_ascent_reward import v135_final_ascent_state

    strict = v135_riser_identity_state(env, sensor_cfg)
    ascent_state = v135_final_ascent_state(env, sensor_cfg)

    # Expected leg: one-hot encoding
    # expected_side=0 → FL, expected_side=1 → FR
    expected_FL = (strict["expected_side"] == 0).float()
    expected_FR = (strict["expected_side"] == 1).float()

    # Seeded flag
    seeded = strict["seeded"].float()

    # Target active
    target_active = strict["target_valid"].float()

    # Current and best potential
    # Before seeded: max(FL, FR)
    # After seeded: expected leg's potential
    fl_current = ascent_state["current_potential"][:, 0]
    fr_current = ascent_state["current_potential"][:, 1]
    fl_best = ascent_state["best_potential"][:, 0]
    fr_best = ascent_state["best_potential"][:, 1]

    # Current potential (what policy is working on right now)
    current_potential = torch.where(
        strict["seeded"],
        # After seed: use expected leg
        torch.where(
            strict["expected_side"] == 0,
            fl_current,
            fr_current,
        ),
        # Before seed: use max of both
        torch.maximum(fl_current, fr_current),
    )

    # Best potential (historical max for current riser)
    best_potential = torch.where(
        strict["seeded"],
        torch.where(strict["expected_side"] == 0, fl_best, fr_best),
        torch.maximum(fl_best, fr_best),
    )
    # Reward settles the old riser on the completion frame; the observation
    # describes the next action's live target and must not reuse old credit.
    same_target = ascent_state["target_id_last"] == strict["target_id"]
    current_potential = torch.where(same_target, current_potential, 0.0)
    best_potential = torch.where(same_target, best_potential, 0.0)

    return torch.stack([
        expected_FL,
        expected_FR,
        seeded,
        target_active,
        current_potential,
        best_potential,
    ], dim=1)


__all__ = ["v135_final_critic_reward_state"]
