# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terrain curriculum with synchronized global gait_level update."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())
    # Local import avoids module-load circular dependency.
    from .rewards import update_gait_level_from_terrain_mean

    update_gait_level_from_terrain_mean(mean_level)
    return mean_level


def v11_teacher_gated_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    lin_vel_reward_term_name: str = "track_lin_vel_xy_exp",
    ang_vel_reward_term_name: str = "track_ang_vel_z_exp",
    stair_forward_progress_reward_term_name: str = "world_forward_progress",
    move_up_tracking_threshold: float = 0.50,
    move_down_tracking_threshold: float = 0.32,
    move_up_forward_progress_threshold: float = 0.45,
    move_down_forward_progress_threshold: float = 0.20,
    move_up_platform_success_threshold: float = 0.75,
    move_up_alternation_compliance_threshold: float = 0.80,
    move_up_same_side_repeat_threshold: float = 0.20,
    move_up_front_riser_collision_threshold: float = 0.0,
    random_reset_at_max: bool = True,
    allow_move_down: bool = True,
    strict_ordered_gate: bool = False,
    v133_three_step_gate: bool = False,
    v133_min_chain_precision: float = 0.50,
    v133_descent_closure_gate: bool = False,
    v133_min_descent_closure_rate: float = 0.50,
    official_descent_curriculum: bool = False,
    whole_body_ascent_gate: bool = False,
) -> torch.Tensor:
    """V11 curriculum: V9-quality gate plus true ascent completion.

    Non-stair terrain uses the V9 teacher's timeout and velocity-tracking
    quality gate.  V11's forward-progress reward is deliberately stair-only,
    so applying it to every terrain column would permanently block rough/flat
    columns.  On stair columns it is required; on ascending stair columns a
    reward-side platform confirmation is required as well.  The latter is
    simulator-only curriculum state, never an actor observation.
    """
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    if ids.numel() == 0:
        return terrain.terrain_levels.float().mean()
    if env.common_step_counter == 0:
        mean_level = terrain.terrain_levels.float().mean()
        from .rewards import update_gait_level_from_terrain_mean

        update_gait_level_from_terrain_mean(mean_level)
        return mean_level

    def _normalized_episode_score(term_name: str) -> torch.Tensor:
        if term_name not in env.reward_manager._episode_sums:
            raise RuntimeError(f"V11 terrain curriculum requires reward term {term_name!r}.")
        episode_mean = env.reward_manager._episode_sums[term_name][ids] / env.max_episode_length_s
        weight = env.reward_manager.get_term_cfg(term_name).weight
        return episode_mean / max(abs(weight), 1.0e-6)

    cfg = env.cfg.v11_perception
    stair_type_ids = tuple(cfg.stair_terrain_type_ids)
    ascending_type_ids = tuple(cfg.ascending_stair_terrain_type_ids)
    if not stair_type_ids or not ascending_type_ids:
        raise RuntimeError("V11 terrain curriculum requires resolved stair terrain type IDs.")
    terrain_types = terrain.terrain_types[ids]
    stair_ids = torch.tensor(stair_type_ids, device=env.device, dtype=terrain_types.dtype)
    ascending_ids = torch.tensor(ascending_type_ids, device=env.device, dtype=terrain_types.dtype)
    is_stair = (terrain_types[:, None] == stair_ids[None, :]).any(dim=1)
    is_ascending = (terrain_types[:, None] == ascending_ids[None, :]).any(dim=1)
    is_descending = is_stair & ~is_ascending

    lin_vel_score = _normalized_episode_score(lin_vel_reward_term_name)
    lin_vel_cfg = env.reward_manager.get_term_cfg(lin_vel_reward_term_name)
    stair_scale = float((lin_vel_cfg.params or {}).get("stair_scale", 1.0))
    if stair_scale != 1.0:
        from .v11_landing import v11_restore_stair_tracking_quality

        scaled_mask = is_ascending if (lin_vel_cfg.params or {}).get("ascending_only", False) else is_stair
        lin_vel_score = v11_restore_stair_tracking_quality(lin_vel_score, scaled_mask, stair_scale)
    ang_vel_score = _normalized_episode_score(ang_vel_reward_term_name)
    tracking_score = torch.minimum(lin_vel_score, ang_vel_score)
    forward_progress_score = _normalized_episode_score(stair_forward_progress_reward_term_name)
    timed_out = env.termination_manager.time_outs[ids]

    # V11 reward evaluation updates this state during the final simulated step.
    # Missing state is treated as failure, never as a successful staircase.
    landing_state = getattr(env, "_v11_landing", None)
    if landing_state is None:
        platform_confirmed = torch.zeros(ids.numel(), device=env.device, dtype=torch.bool)
        episode_alternating = torch.zeros(ids.numel(), device=env.device)
        episode_repeats = torch.zeros(ids.numel(), device=env.device)
        episode_simultaneous = torch.zeros(ids.numel(), device=env.device)
        episode_collision = torch.ones(ids.numel(), device=env.device, dtype=torch.bool)
    else:
        platform_confirmed = landing_state["platform_confirmed"][ids]
        episode_alternating = landing_state["episode_alternating_count"][ids].float()
        episode_repeats = landing_state["episode_repeat_count"][ids].float()
        episode_simultaneous = landing_state["episode_simultaneous_count"][ids].float()
        episode_collision = landing_state["episode_front_collision"][ids]

    quality_up = timed_out & (tracking_score > move_up_tracking_threshold)
    stair_progress_up = forward_progress_score > move_up_forward_progress_threshold
    stair_complete = torch.where(is_ascending, platform_confirmed, torch.ones_like(platform_confirmed))
    move_up = quality_up & torch.where(is_stair, stair_progress_up & stair_complete, torch.ones_like(quality_up))
    if whole_body_ascent_gate:
        recovered = getattr(env, '_v111_recovery', None)
        whole = (recovered['whole_flight'][ids] if recovered is not None
                 else torch.zeros_like(move_up))
        move_up &= ~is_ascending | whole

    strict_ordered_success = torch.ones_like(move_up)
    if strict_ordered_gate:
        ordered_state = getattr(env, "_v131_ordered", None)
        required_keys = (
            "strict_chain_valid",
            "episode_skip_latched",
            "ordered_last_riser_index",
            "strict_required_risers",
        )
        if ordered_state is None or any(key not in ordered_state for key in required_keys):
            # Missing task accounting must never silently promote ascent.
            strict_ordered_success = torch.zeros_like(move_up)
        else:
            required = ordered_state["strict_required_risers"][ids]
            completed = ordered_state["ordered_last_riser_index"][ids]
            strict_ordered_success = (
                ordered_state["strict_chain_valid"][ids]
                & ~ordered_state["episode_skip_latched"][ids]
                & (required > 0)
                & (completed >= required)
            )
        move_up &= torch.where(
            is_ascending, strict_ordered_success, torch.ones_like(strict_ordered_success),
        )

    v133_chain_success = torch.ones_like(move_up)
    if v133_three_step_gate:
        chain_state = getattr(env, "_v133_chain", None)
        required = ("episode_chain_success_count", "episode_chain_failure_count")
        if chain_state is None or any(key not in chain_state for key in required):
            v133_chain_success = torch.zeros_like(move_up)
        else:
            successes = chain_state["episode_chain_success_count"][ids].float()
            failures = chain_state["episode_chain_failure_count"][ids].float()
            from .v133_adjacent_transition import v133_episode_chain_quality

            v133_chain_success = v133_episode_chain_quality(
                successes, failures, minimum_precision=v133_min_chain_precision,
            )
        move_up &= torch.where(is_ascending, v133_chain_success, torch.ones_like(v133_chain_success))

    v133_descent_success = torch.ones_like(move_up)
    if v133_descent_closure_gate:
        chain_state = getattr(env, "_v133_chain", None)
        required = ("episode_descent_front_count", "episode_descent_completion_count")
        if chain_state is None or any(key not in chain_state for key in required):
            v133_descent_success = torch.zeros_like(move_up)
        else:
            front = chain_state["episode_descent_front_count"][ids]
            rear = chain_state["episode_descent_completion_count"][ids]
            from .v133_adjacent_transition import v133_descent_closure_quality

            v133_descent_success = v133_descent_closure_quality(
                front, rear, minimum_rate=v133_min_descent_closure_rate,
            )
        move_up &= torch.where(is_descending, v133_descent_success, torch.ones_like(v133_descent_success))

    soft_alternation = bool(getattr(cfg, "soft_alternation", False))
    transition_count = episode_alternating + episode_repeats + episode_simultaneous
    transition_valid = transition_count > 0
    compliance = episode_alternating / transition_count.clamp_min(1.0)
    repeat_rate = episode_repeats / transition_count.clamp_min(1.0)
    platform_rate = platform_confirmed.float()
    collision_rate = episode_collision.float()
    if soft_alternation:
        # A zero-transition episode is explicitly invalid: it cannot obtain a
        # fake 100% compliance rate and advance the ascending-stair curriculum.
        # V13 disables only this style gate: its alternation remains a reward
        # preference while native V11 task-quality promotion stays intact.
        if bool(getattr(cfg, "soft_alternation_curriculum_gate", True)):
            from .v11_landing import v11_softalt_quality_gate

            soft_quality = v11_softalt_quality_gate(
                platform_rate, compliance, repeat_rate, collision_rate, transition_valid,
                platform_threshold=move_up_platform_success_threshold,
                compliance_threshold=move_up_alternation_compliance_threshold,
                repeat_threshold=move_up_same_side_repeat_threshold,
                collision_threshold=move_up_front_riser_collision_threshold,
            )
            move_up &= torch.where(is_ascending, soft_quality, torch.ones_like(soft_quality))
        if landing_state is not None:
            landing_state["stair_episode_count"] += is_stair.sum()
            landing_state["ascending_episode_count"] += is_ascending.sum()
            landing_state["front_collision_episode_count"] += (episode_collision & is_stair).sum()

    quality_down = (~timed_out) | (tracking_score < move_down_tracking_threshold)
    stair_progress_down = forward_progress_score < move_down_forward_progress_threshold
    move_down = quality_down | (is_stair & stair_progress_down)
    move_down &= ~move_up
    if official_descent_curriculum:
        # Match the official M20 Rough distance curriculum on descending
        # columns while retaining V13.3's chain gate on ascending columns.
        command = env.command_manager.get_command("base_velocity")[ids]
        distance = torch.norm(
            asset.data.root_pos_w[ids, :2] - env.scene.env_origins[ids, :2], dim=1,
        )
        official_up = distance > terrain.cfg.terrain_generator.size[0] / 2
        official_down = distance < torch.norm(command[:, :2], dim=1) * env.max_episode_length_s * 0.5
        official_down &= ~official_up
        from .v133_adjacent_transition import v133_route_official_descent_curriculum

        move_up, move_down = v133_route_official_descent_curriculum(
            move_up, move_down, is_descending, official_up, official_down,
        )
    if not allow_move_down:
        move_down = torch.zeros_like(move_down)

    env._v11_terrain_curriculum_debug = {
        "tracking_score_mean": tracking_score.mean(),
        "forward_progress_score_mean": forward_progress_score[is_stair].mean()
        if is_stair.any() else torch.zeros((), device=env.device),
        "stair_platform_success_rate": platform_confirmed[is_ascending].float().mean()
        if is_ascending.any() else torch.zeros((), device=env.device),
        "stair_alternation_compliance_rate": compliance[is_ascending & transition_valid].mean()
        if (is_ascending & transition_valid).any() else torch.full((), float("nan"), device=env.device),
        "stair_same_side_repeat_rate": repeat_rate[is_ascending & transition_valid].mean()
        if (is_ascending & transition_valid).any() else torch.full((), float("nan"), device=env.device),
        "stair_front_riser_collision_rate": collision_rate[is_stair].mean()
        if is_stair.any() else torch.full((), float("nan"), device=env.device),
        "move_up_rate": move_up.float().mean(),
        "move_down_rate": move_down.float().mean(),
        "strict_ordered_promotion_eligible_rate": (
            strict_ordered_success[is_ascending].float().mean()
            if is_ascending.any() else torch.zeros((), device=env.device)
        ),
        "v133_chain_promotion_eligible_rate": (
            v133_chain_success[is_ascending].float().mean()
            if is_ascending.any() else torch.zeros((), device=env.device)
        ),
        "v133_descent_promotion_eligible_rate": (
            v133_descent_success[is_descending].float().mean()
            if is_descending.any() else torch.zeros((), device=env.device)
        ),
    }
    if random_reset_at_max:
        terrain.update_env_origins(ids, move_up, move_down)
    else:
        max_level = terrain.max_terrain_level - 1
        terrain.terrain_levels[ids] = (terrain.terrain_levels[ids] + move_up.long() - move_down.long()).clamp(0, max_level)
        terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]

    mean_level = terrain.terrain_levels.float().mean()
    from .rewards import update_gait_level_from_terrain_mean

    update_gait_level_from_terrain_mean(mean_level)
    return mean_level


def v11_terrain_curriculum_debug(env: ManagerBasedRLEnv, env_ids: Sequence[int], key: str) -> torch.Tensor:
    """Return the latest V11 curriculum diagnostic, or fail closed before reset."""
    debug = getattr(env, "_v11_terrain_curriculum_debug", None)
    if debug is None or key not in debug:
        return torch.zeros((), device=env.device)
    return debug[key]


def gait_level_curve(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Return current global gait_level for logging in curriculum curves."""
    from .rewards import gait_level

    return torch.tensor(gait_level, device=env.device)


def perceptive_stair_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    first_riser_from_spawn: float = 1.10,
    step_depth: float = 0.31,
    num_steps: int = 6,
    top_landing_margin: float = 0.05,
    low_progress_threshold: float = 0.25,
) -> torch.Tensor:
    """Promote only complete stable ascents; demote only clear failures.

    This is training-manager GT.  It updates terrain origins at reset and is
    never registered as an actor or critic observation.
    """
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    terrain: TerrainImporter = env.scene.terrain
    asset: Articulation = env.scene[asset_cfg.name]
    if ids.numel() == 0:
        return terrain.terrain_levels.float().mean()
    # Curriculum is invoked at episode reset while body state still represents
    # the completed episode.  All wheel centres must be beyond the last riser.
    from .perceptive_spatial import _semantic_body_ids

    if not hasattr(env, "_perceptive_wheel_body_ids"):
        env._perceptive_wheel_body_ids = _semantic_body_ids(asset)
    wheel_x = asset.data.body_pos_w[ids][:, env._perceptive_wheel_body_ids, 0] - env.scene.env_origins[ids, None, 0]
    root_progress = asset.data.root_pos_w[ids, 0] - env.scene.env_origins[ids, 0]
    last_riser_x = first_riser_from_spawn + num_steps * step_depth
    stable = (asset.data.projected_gravity_b[ids, 2] < -0.65) & (
        asset.data.projected_gravity_b[ids, :2].abs().amax(dim=1) < 0.55
    )
    success = (wheel_x > last_riser_x + top_landing_margin).all(dim=1) & stable
    fell = ~stable
    very_low_progress = root_progress < low_progress_threshold
    move_down = (~success) & (fell | very_low_progress)
    terrain.update_env_origins(ids, success, move_down)
    env.perceptive_curriculum_success = torch.zeros(env.num_envs, device=env.device)
    env.perceptive_curriculum_demote = torch.zeros(env.num_envs, device=env.device)
    env.perceptive_curriculum_success[ids] = success.float()
    env.perceptive_curriculum_demote[ids] = move_down.float()
    return terrain.terrain_levels.float().mean()


def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)
