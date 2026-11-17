# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import reset_root_state_uniform as _official_reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None) # type: ignore
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def bad_orientation_2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot") # type: ignore
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return (asset.data.projected_gravity_b[:, 2] > 0) | (asset.data.projected_gravity_b[:, :2].abs() > 0.7).any(-1)


def reset_v6_task_pose(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Reset only on the central flat and align the base with a cardinal stair."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = env_ids.to(device=env.device, dtype=torch.long)
    n = ids.numel()
    if n == 0:
        return
    if not hasattr(env, "v6_direction"):
        wheel_ids, _ = asset.find_bodies(".*_wheel")
        env.v6_wheel_body_ids = torch.as_tensor(wheel_ids, device=env.device, dtype=torch.long)
        env.v6_direction = torch.zeros((env.num_envs, 2), device=env.device)
        env.v6_stair_height = torch.full((env.num_envs,), 0.17, device=env.device)
        env.v6_current_level = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env.v6_level_crossed = torch.zeros((env.num_envs, 6, 4), device=env.device, dtype=torch.bool)
        env.v6_level_completed = torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.bool)
        env.v6_collision_latched = torch.zeros((env.num_envs, 6, 4), device=env.device, dtype=torch.bool)
        env.v6_cross_frames = torch.zeros((env.num_envs, 6, 4), device=env.device, dtype=torch.long)
        env.v6_success = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.v6_failure = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.v6_stagnation_history = torch.zeros((env.num_envs, 20, 2), device=env.device)
        env.v6_wheel_z0 = torch.zeros((env.num_envs, 4), device=env.device)
        env.v6_step_complete_event = torch.zeros(env.num_envs, device=env.device)
        env.v6_level_complete_event = torch.zeros((env.num_envs, 6), device=env.device)
        env.v6_success_event = torch.zeros(env.num_envs, device=env.device)
        env.v6_failure_event = torch.zeros(env.num_envs, device=env.device)
    choice = torch.randint(0, 4, (n,), device=env.device)
    dirs = torch.zeros((n, 2), device=env.device)
    dirs[choice == 0, 0] = 1.0
    dirs[choice == 1, 0] = -1.0
    dirs[choice == 2, 1] = 1.0
    dirs[choice == 3, 1] = -1.0
    env.v6_direction[ids] = dirs
    slots = torch.remainder(ids, 64)
    env.v6_stair_height[ids] = torch.where(
        slots >= 58,
        torch.empty(n, device=env.device).uniform_(0.08, 0.15),
        torch.where(slots >= 29, torch.full((n,), 0.20, device=env.device), torch.full((n,), 0.17, device=env.device)),
    )
    state = asset.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, 0:2] += torch.empty((n, 2), device=env.device).uniform_(-0.10, 0.10)
    yaw = torch.atan2(dirs[:, 1], dirs[:, 0]) + torch.empty(n, device=env.device).uniform_(-0.10, 0.10)
    state[:, 3:7] = math_utils.quat_from_euler_xyz(
        torch.zeros(n, device=env.device), torch.zeros(n, device=env.device), yaw
    )
    state[:, 7:] = 0.0
    asset.write_root_state_to_sim(state, env_ids=ids)
    asset.write_joint_state_to_sim(
        asset.data.default_joint_pos[ids], asset.data.default_joint_vel[ids], env_ids=ids
    )
    env.v6_current_level[ids] = 0
    env.v6_level_crossed[ids] = False
    env.v6_level_completed[ids] = False
    env.v6_collision_latched[ids] = False
    env.v6_cross_frames[ids] = 0
    env.v6_success[ids] = False
    env.v6_failure[ids] = False
    env.v6_step_complete_event[ids] = 0
    env.v6_level_complete_event[ids] = 0
    env.v6_success_event[ids] = 0
    env.v6_failure_event[ids] = 0
    env.v6_stagnation_history[ids] = state[:, :2].unsqueeze(1)
    env.v6_wheel_z0[ids] = env.scene.env_origins[ids, 2].unsqueeze(1) + 0.09


def v6_success_termination(env: ManagerBasedEnv) -> torch.Tensor:
    """Terminate immediately after the sixth step-completion latch."""
    if not hasattr(env, "v6_success"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.v6_success


def reset_perceptive_stair_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    spawn_x_jitter: float = 0.20,
    spawn_y_jitter: float = 0.20,
    spawn_yaw_jitter: float = 0.0872664626,
):
    """Reset on the approach, facing world +X, with configured small jitter."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = env_ids.to(device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    state = asset.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, 0] += torch.empty(ids.numel(), device=env.device).uniform_(-spawn_x_jitter, spawn_x_jitter)
    state[:, 1] += torch.empty(ids.numel(), device=env.device).uniform_(-spawn_y_jitter, spawn_y_jitter)
    yaw = torch.empty(ids.numel(), device=env.device).uniform_(-spawn_yaw_jitter, spawn_yaw_jitter)
    state[:, 3:7] = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    state[:, 7:] = 0.0
    asset.write_root_state_to_sim(state, env_ids=ids)
    asset.write_joint_state_to_sim(
        asset.data.default_joint_pos[ids], asset.data.default_joint_vel[ids], env_ids=ids
    )
    # Invalidate reward geometry and elevation sample-and-hold on reset.
    env._perceptive_spatial_step = None
    for name in (
        "_perceptive_vertical_contact_latched",
        "_perceptive_true_precontact_clearance_latched",
        "_perceptive_riser_plane_crossing_latched",
        "_perceptive_upper_tread_support_latched",
    ):
        if hasattr(env, name):
            getattr(env, name)[ids] = False
    if hasattr(env, "_perceptive_elevation_last_step"):
        env._perceptive_elevation_last_step = -1


def v10_reset_root_state_stair_conditioned(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    stair_distance_range: tuple[float, float] = (0.9, 1.4),
    stair_lateral_range: tuple[float, float] = (-0.25, 0.25),
    stair_yaw_jitter_range: tuple[float, float] = (-0.2617993878, 0.2617993878),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Use official Rough reset off stairs and an approach-aligned reset on upstairs.

    The stair mask is exactly the V10 ascending ``pyramid_stairs_inv`` column
    range.  The other official Rough terrain types are delegated unchanged to
    Isaac Lab's ``reset_root_state_uniform`` with the inherited pose/velocity
    ranges.
    """
    ids = env_ids.to(device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    terrain = env.scene.terrain
    cfg = env.cfg.v10_perception
    terrain_types = terrain.terrain_types[ids]
    stair_mask = (
        (terrain_types >= int(cfg.full_stair_column_start))
        & (terrain_types < int(cfg.full_stair_column_end))
    )
    non_stair_ids = ids[~stair_mask]
    if non_stair_ids.numel() > 0:
        _official_reset_root_state_uniform(
            env, non_stair_ids, pose_range, velocity_range, asset_cfg
        )

    stair_ids = ids[stair_mask]
    if stair_ids.numel() == 0:
        return
    asset: Articulation = env.scene[asset_cfg.name]
    count = stair_ids.numel()

    terrain_generator_cfg = terrain.cfg.terrain_generator
    stair_cfg = terrain_generator_cfg.sub_terrains["pyramid_stairs_inv"]
    size_x = float(terrain_generator_cfg.size[0])
    terrain_width = size_x - 2.0 * float(stair_cfg.border_width)
    num_steps = int(
        (size_x - 2.0 * float(stair_cfg.border_width) - float(stair_cfg.platform_width))
        // (2.0 * float(stair_cfg.step_width))
        + 1
    )
    first_riser_radius = 0.5 * (
        terrain_width - 2.0 * num_steps * float(stair_cfg.step_width)
    )

    side = torch.randint(0, 4, (count,), device=env.device)
    cardinal = torch.tensor(
        ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)),
        device=env.device,
    )
    direction = cardinal[side]
    lateral_direction = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
    if (hasattr(env.cfg, "v135_riser_identity") or hasattr(env.cfg, "v112_tread_widths")
            or hasattr(env.cfg, "v111_entry_flat")):
        from .v135_terrain_geometry import v135_terrain_geometry
        geometry = v135_terrain_geometry(env)
        first_riser_radius = geometry["radii"][stair_ids].gather(1, (side % 2)[:, None]).squeeze(1)
    longtrain_cfg = getattr(env.cfg, "v135_longtrain", None)
    if longtrain_cfg is None:
        approach_distance = torch.empty(count, device=env.device).uniform_(*stair_distance_range)
        lateral_range = stair_lateral_range
        yaw_range = stair_yaw_jitter_range
    else:
        from .v135_longtrain_curriculum import (
            v135_early_mixture_probability,
            v135_linear_fraction,
        )

        fraction = v135_linear_fraction(env)
        probability = v135_early_mixture_probability(env)
        approach_distance = torch.empty(count, device=env.device).uniform_(*stair_distance_range)
        near = torch.rand(count, device=env.device) < probability
        near_distance = torch.empty(count, device=env.device).uniform_(
            *longtrain_cfg.near_stair_distance_range
        )
        approach_distance = torch.where(near, near_distance, approach_distance)
        lateral_range = tuple(
            float(early) + fraction * (float(final) - float(early))
            for early, final in zip(longtrain_cfg.early_lateral_range, stair_lateral_range)
        )
        yaw_range = tuple(
            float(early) + fraction * (float(final) - float(early))
            for early, final in zip(longtrain_cfg.early_yaw_jitter_range, stair_yaw_jitter_range)
        )
    lateral_offset = torch.empty(count, device=env.device).uniform_(*lateral_range)
    local_xy = (
        direction * (first_riser_radius - approach_distance)[:, None]
        + lateral_direction * lateral_offset[:, None]
    )

    root_states = asset.data.default_root_state[stair_ids].clone()
    roll = torch.empty(count, device=env.device).uniform_(*pose_range.get("roll", (0.0, 0.0)))
    pitch = torch.empty(count, device=env.device).uniform_(*pose_range.get("pitch", (0.0, 0.0)))
    yaw_jitter = torch.empty(count, device=env.device).uniform_(*yaw_range)
    nominal_yaw = torch.atan2(direction[:, 1], direction[:, 0])
    orientation_delta = math_utils.quat_from_euler_xyz(roll, pitch, nominal_yaw + yaw_jitter)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    z_offset = torch.empty(count, device=env.device).uniform_(*pose_range.get("z", (0.0, 0.0)))
    positions = root_states[:, :3] + env.scene.env_origins[stair_ids]
    positions[:, :2] += local_xy
    positions[:, 2] += z_offset

    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ]
    velocity_ranges = torch.tensor(range_list, device=env.device)
    velocity_noise = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (count, 6), device=env.device
    )
    velocities = root_states[:, 7:13] + velocity_noise

    asset.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=stair_ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=stair_ids)

    if not hasattr(env, "_v10_stair_reset_distance"):
        env._v10_stair_reset_distance = torch.zeros(env.num_envs, device=env.device)
        env._v10_stair_reset_yaw_error = torch.zeros(env.num_envs, device=env.device)
        env._v10_stair_reset_lateral = torch.zeros(env.num_envs, device=env.device)
    env._v10_stair_reset_distance[stair_ids] = approach_distance
    env._v10_stair_reset_yaw_error[stair_ids] = yaw_jitter
    env._v10_stair_reset_lateral[stair_ids] = lateral_offset
