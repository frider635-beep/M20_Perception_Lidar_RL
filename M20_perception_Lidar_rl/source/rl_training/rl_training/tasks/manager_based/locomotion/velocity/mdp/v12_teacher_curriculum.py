"""V12 terrain curriculum matching the V9 teacher's effective rule."""

from __future__ import annotations

import torch


def v12_v9_teacher_terrain_levels(
    env, env_ids, lin_vel_reward_term_name: str = "track_lin_vel_xy_exp",
    ang_vel_reward_term_name: str = "track_ang_vel_z_exp",
    forward_progress_reward_term_name: str = "world_forward_progress_score",
    move_up_tracking_threshold: float = 0.50, move_down_tracking_threshold: float = 0.32,
    move_up_forward_progress_threshold: float = 0.45, move_down_forward_progress_threshold: float = 0.20,
    random_reset_at_max: bool = True, allow_move_down: bool = True,
) -> torch.Tensor:
    """Replicate V9's inherited timeout/tracking/progress terrain rule.

    V9's ``terrain_levels_vel`` promotes only after an episode reaches its time
    limit while both the normalized tracking and forward-progress scores clear
    their thresholds.  It demotes on an early termination, insufficient
    tracking, or insufficient forward progress, on every terrain type.
    """
    terrain = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0 or env.common_step_counter == 0:
        return terrain.terrain_levels.float().mean()

    def score(name: str) -> torch.Tensor:
        if name not in env.reward_manager._episode_sums:
            raise RuntimeError(f"V12 curriculum requires reward term {name!r}.")
        episode_mean = env.reward_manager._episode_sums[name][ids] / env.max_episode_length_s
        return episode_mean / max(abs(env.reward_manager.get_term_cfg(name).weight), 1.0e-6)

    tracking = torch.minimum(score(lin_vel_reward_term_name), score(ang_vel_reward_term_name))
    progress = score(forward_progress_reward_term_name)
    timed_out = env.termination_manager.time_outs[ids]
    move_up = (
        timed_out
        & (tracking > move_up_tracking_threshold)
        & (progress > move_up_forward_progress_threshold)
    )
    move_down = (
        (~timed_out)
        | (tracking < move_down_tracking_threshold)
        | (progress < move_down_forward_progress_threshold)
    )
    move_down &= ~move_up
    if not allow_move_down:
        move_down = torch.zeros_like(move_down)
    terrain.update_env_origins(ids, move_up, move_down) if random_reset_at_max else _v12_clamped_update(terrain, ids, move_up, move_down)
    env._v12_terrain_curriculum_debug = {
        "tracking_score_mean": tracking.mean(),
        "tracking_score_min": tracking.min(),
        "forward_progress_score_mean": progress.mean(),
        "forward_progress_score_min": progress.min(),
        "timeout_rate": timed_out.float().mean(),
        "move_up_rate": move_up.float().mean(), "move_down_rate": move_down.float().mean(),
    }
    return terrain.terrain_levels.float().mean()


def _v12_clamped_update(terrain, ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor) -> None:
    levels = (terrain.terrain_levels[ids] + move_up.long() - move_down.long()).clamp(0, terrain.max_terrain_level - 1)
    terrain.terrain_levels[ids] = levels
    terrain.env_origins[ids] = terrain.terrain_origins[levels, terrain.terrain_types[ids]]


def v12_terrain_curriculum_debug(env, env_ids, key: str) -> torch.Tensor:
    return getattr(env, "_v12_terrain_curriculum_debug", {}).get(key, torch.zeros((), device=env.device))


def v12_fixed_gait_level(env, env_ids, level: float = 1.0) -> torch.Tensor:
    """Mirror V9's fixed regularization curriculum for from-scratch training."""
    from .rewards import update_gait_level_from_terrain_mean

    # The legacy scalar reaches exactly one at a terrain mean of three.  V9
    # fixes it there so level-zero starts retain posture and smoothness costs.
    update_gait_level_from_terrain_mean(3.0 if level >= 1.0 else level)
    return torch.tensor(level, device=env.device)
