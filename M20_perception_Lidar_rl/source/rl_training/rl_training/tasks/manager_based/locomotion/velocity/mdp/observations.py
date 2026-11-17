# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def v6_local_elevation(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the deployable 21x15 local elevation patch (315 values).

    The scanner is centered 0.25 m in front of the base.  Invalid rays are
    replaced by zero and every patch is expressed relative to its valid local
    minimum, matching the policy-side elevation mapping used on the robot.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(hits)
    safe = torch.where(valid, hits, torch.full_like(hits, float("inf")))
    valid_min = safe.min(dim=1, keepdim=True).values
    valid_min = torch.where(torch.isfinite(valid_min), valid_min, torch.zeros_like(valid_min))
    return torch.nan_to_num(hits - valid_min, nan=0.0, posinf=0.0, neginf=0.0)


def perceptive_elevation_world_map(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return semantic world-Z ray hits as ``[N, 21, 15]`` in metres.

    Axes are semantic and deployable: row zero is front (+1.25 m), row 20 is
    rear (-0.75 m), column zero is left (+0.70 m), and column 14 is right
    (-0.70 m).  Isaac Lab's ``ordering='xy'`` ray buffer is laid out as
    ``[y, x]`` with both coordinates increasing, hence the transpose and the
    two flips below. Invalid rays remain NaN for reward-side geometry.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    terrain_z = sensor.data.ray_hits_w[..., 2].reshape(env.num_envs, -1)
    expected_rays = 21 * 15
    if terrain_z.shape[1] != expected_rays:
        raise RuntimeError(f"M20 elevation requires {expected_rays} rays, got {tuple(terrain_z.shape)}")
    world = terrain_z.reshape(-1, 15, 21).transpose(1, 2).flip(dims=(1, 2))
    return torch.where(torch.isfinite(world), world, torch.full_like(world, float("nan")))


def perceptive_elevation_clean_map(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return clean local terrain height as ``[N, 21, 15]`` in metres."""
    world = perceptive_elevation_world_map(env, sensor_cfg)
    valid = torch.isfinite(world)
    safe = torch.where(valid, world, torch.full_like(world, float("inf")))
    valid_min = safe.flatten(1).amin(dim=1).view(-1, 1, 1)
    valid_min = torch.where(torch.isfinite(valid_min), valid_min, torch.zeros_like(valid_min))
    return torch.where(valid, world - valid_min, torch.zeros_like(world))


def perceptive_elevation_clean(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Flatten the clean semantic map in row-major order to ``[N, 315]``."""
    return perceptive_elevation_clean_map(env, sensor_cfg).flatten(start_dim=1)


def perceptive_elevation_actor(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    z_noise_std: float = 0.0,
    xy_grid_shift: float = 0.0,
    dropout_probability: float = 0.0,
    update_period_steps: int = 1,
    latency_steps: int = 0,
) -> torch.Tensor:
    """Return the cached/noisy map consumed by the actor and critic.

    All perturbations are configuration driven.  The first curriculum stage
    uses their zero/one defaults, so actor elevation equals clean elevation.
    ``update_period_steps`` implements sample-and-hold at policy rate and the
    ring buffer implements whole-policy-step latency without exposing either
    setting to the policy.
    """
    if update_period_steps < 1 or latency_steps < 0:
        raise ValueError("update_period_steps must be >= 1 and latency_steps must be >= 0")
    clean = perceptive_elevation_clean_map(env, sensor_cfg)
    shape = clean.shape
    needs_init = not hasattr(env, "_perceptive_elevation_hold") or env._perceptive_elevation_hold.shape != shape
    if needs_init:
        env._perceptive_elevation_hold = clean.clone()
        env._perceptive_elevation_ring = clean.unsqueeze(0).repeat(latency_steps + 1, 1, 1, 1)
        env._perceptive_elevation_ring_index = 0
        env._perceptive_elevation_last_step = -1

    step = int(getattr(env, "common_step_counter", 0))
    if step != env._perceptive_elevation_last_step and step % update_period_steps == 0:
        actor_map = clean.clone()
        if xy_grid_shift > 0.0:
            # Continuous random translation; normalized grid coordinates use
            # (column, row) order for torch.grid_sample.
            import torch.nn.functional as F

            n = actor_map.shape[0]
            shift_x = torch.empty(n, device=env.device).uniform_(-xy_grid_shift, xy_grid_shift)
            shift_y = torch.empty(n, device=env.device).uniform_(-xy_grid_shift, xy_grid_shift)
            theta = torch.zeros((n, 2, 3), device=env.device, dtype=actor_map.dtype)
            theta[:, 0, 0] = 1.0
            theta[:, 1, 1] = 1.0
            theta[:, 0, 2] = -2.0 * shift_y / 1.4
            theta[:, 1, 2] = -2.0 * shift_x / 2.0
            grid = F.affine_grid(theta, (n, 1, 21, 15), align_corners=True)
            actor_map = F.grid_sample(
                actor_map.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True
            ).squeeze(1)
        if z_noise_std > 0.0:
            actor_map.add_(torch.randn_like(actor_map) * z_noise_std)
        if dropout_probability > 0.0:
            actor_map.masked_fill_(torch.rand_like(actor_map) < dropout_probability, 0.0)
        actor_map.clamp_(min=0.0)
        ring_index = int(env._perceptive_elevation_ring_index)
        env._perceptive_elevation_ring[ring_index] = actor_map
        read_index = (ring_index - latency_steps) % (latency_steps + 1)
        env._perceptive_elevation_hold = env._perceptive_elevation_ring[read_index].clone()
        env._perceptive_elevation_ring_index = (ring_index + 1) % (latency_steps + 1)
        env._perceptive_elevation_last_step = step
    return env._perceptive_elevation_hold.flatten(start_dim=1)
