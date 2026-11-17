"""V13.5 velocity modulation for ascent gait and descent support quality.

Preserve V13's 5.0/2.0/3.0 drive while changing which gait earns it most efficiently.
"""

from __future__ import annotations

import torch


def v135_modulated_track_lin_vel_xy_exp_l2(env, std: float, command_name: str) -> torch.Tensor:
    """Reward tracking linear velocity with descent modulation.

    Preserves V13 weight 5.0, applies g_d modulation only during rear transfer.
    """
    from .v135_descent_support_transfer import v135_descent_velocity_modulation
    from .v135_riser_identity import v135_ascent_gait_speed_factor

    from isaaclab.envs.mdp.rewards import track_lin_vel_xy_exp

    reward = track_lin_vel_xy_exp(env, std=std, command_name=command_name)

    # Apply the shared LongTrain schedule: early 0.85+0.15P, then
    # continuously anneal to the final 0.60+0.40P gate.
    g_d = v135_descent_velocity_modulation(env)

    return reward * g_d * v135_ascent_gait_speed_factor(env)


def v135_modulated_world_forward_progress(
    env, target_speed_fraction: float = 0.5, sensor_cfg=None,
) -> torch.Tensor:
    """Reward forward progress with descent modulation.

    Calls native V11 function, preserves weight 2.0, applies g_d modulation.
    """
    from .v11_continuous import v11_world_forward_progress_reward
    from .v135_descent_support_transfer import v135_descent_velocity_modulation
    from .v135_riser_identity import v135_ascent_gait_speed_factor

    # Native V11 forward progress
    reward = v11_world_forward_progress_reward(
        env, target_speed_fraction=target_speed_fraction,
    )

    # Apply descent modulation
    g_d = v135_descent_velocity_modulation(env, sensor_cfg)

    return reward * g_d * v135_ascent_gait_speed_factor(env, sensor_cfg)


def v135_modulated_world_forward_deficit(
    env, min_speed_fraction: float = 0.5, speed_scale: float = 0.3,
    sensor_cfg=None,
) -> torch.Tensor:
    """Speed deficit penalty, DISABLED during rear transfer.

    Preserves weight +3.0 (native returns negative cost), sets to 0 during transfer.
    """
    from .v11_continuous import v11_world_forward_speed_deficit
    from .v135_descent_support_transfer import v135_descent_rear_support_transfer_state

    # Native V11 deficit (returns negative cost)
    reward = v11_world_forward_speed_deficit(
        env, min_speed_fraction=min_speed_fraction, speed_scale=speed_scale,
    )

    # Disable during rear transfer
    transfer = v135_descent_rear_support_transfer_state(env, sensor_cfg)

    return torch.where(
        transfer["active"],
        torch.zeros_like(reward),
        reward,
    )


__all__ = [
    "v135_modulated_track_lin_vel_xy_exp_l2",
    "v135_modulated_world_forward_progress",
    "v135_modulated_world_forward_deficit",
]
