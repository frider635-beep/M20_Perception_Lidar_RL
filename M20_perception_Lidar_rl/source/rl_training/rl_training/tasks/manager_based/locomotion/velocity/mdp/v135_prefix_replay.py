"""V13.5: replay only self-discovered, contact-backed ascent prefixes.

The actor and PPO weights always start fresh.  A replay restores a physical
state reached by this same policy on the same terrain tile, then restores the
training-only riser ledger without paying the already completed prefix again.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg


_CHAIN_FIELDS = (
    "chain_length", "last_side", "last_riser_world_xy", "last_upper_z",
    "forward_world_xy", "chain_valid", "b1_locked", "b1_locked_world_xy",
    "b1_locked_forward_world_xy", "support_credited_forward",
    "support_credited_valid",
)
_COUNT_KEYS = (
    "capture_b1_fl_count", "capture_b1_fr_count",
    "capture_b2_fl_count", "capture_b2_fr_count",
    "replay_b1_count", "replay_b2_count", "full_chain_count",
    "replay_chain_count",
)


def _ensure_replay_state(env):
    if hasattr(env, "_v135_prefix_pool"):
        return
    env._v135_prefix_pool = {}
    env._v135_pending_prefix = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._v135_replayed_prefix = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._v135_prefix_counts = {
        name: torch.zeros((), device=env.device, dtype=torch.long) for name in _COUNT_KEYS
    }


def _new_snapshot(env, chain):
    robot = env.scene["robot"]
    terrain = env.scene.terrain
    snapshot = {
        "valid": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        "level": torch.zeros_like(terrain.terrain_levels),
        "type": torch.zeros_like(terrain.terrain_types),
        "origin": torch.zeros_like(terrain.env_origins),
        "root": torch.zeros_like(robot.data.root_state_w),
        "joint_pos": torch.zeros_like(robot.data.joint_pos),
        "joint_vel": torch.zeros_like(robot.data.joint_vel),
    }
    for name in _CHAIN_FIELDS:
        snapshot[name] = torch.zeros_like(chain[name])
    return snapshot


def v135_prefix_replay_reset(
    env, env_ids, pose_range, velocity_range,
    stair_distance_range=(0.9, 1.4),
    stair_lateral_range=(-0.25, 0.25),
    stair_yaw_jitter_range=(-0.2617993878, 0.2617993878),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Keep V13's native reset, replacing it only with safe local snapshots."""
    from .events import v10_reset_root_state_stair_conditioned

    ids = env_ids.to(device=env.device, dtype=torch.long)
    v10_reset_root_state_stair_conditioned(
        env, ids, pose_range, velocity_range,
        stair_distance_range=stair_distance_range,
        stair_lateral_range=stair_lateral_range,
        stair_yaw_jitter_range=stair_yaw_jitter_range,
        asset_cfg=asset_cfg,
    )
    _ensure_replay_state(env)
    env._v135_pending_prefix[ids] = 0
    env._v135_replayed_prefix[ids] = 0
    if ids.numel() == 0 or not env._v135_prefix_pool:
        return

    terrain = env.scene.terrain
    pool = env._v135_prefix_pool

    def available(stage):
        if stage not in pool:
            return torch.zeros(ids.numel(), device=env.device, dtype=torch.bool)
        snap = pool[stage]
        return (
            snap["valid"][ids]
            & (snap["level"][ids] == terrain.terrain_levels[ids])
            & (snap["type"][ids] == terrain.terrain_types[ids])
            & ((snap["origin"][ids] - terrain.env_origins[ids]).abs().amax(dim=1) < 1.0e-5)
        )

    b1, b2 = available(1), available(2)
    cfg = env.cfg.v133_transition
    selected = torch.rand(ids.numel(), device=env.device) < float(cfg.prefix_replay_probability)
    use_b2 = selected & b2 & (
        (~b1) | (torch.rand(ids.numel(), device=env.device) < float(cfg.prefix_b2_share))
    )
    use_b1 = selected & b1 & ~use_b2
    robot = env.scene[asset_cfg.name]
    for stage, mask in ((1, use_b1), (2, use_b2)):
        replay_ids = ids[mask]
        if replay_ids.numel() == 0:
            continue
        snap = pool[stage]
        root = snap["root"][replay_ids].clone()
        root[:, :3] += terrain.env_origins[replay_ids]
        robot.write_root_state_to_sim(root, env_ids=replay_ids)
        robot.write_joint_state_to_sim(
            snap["joint_pos"][replay_ids], snap["joint_vel"][replay_ids],
            env_ids=replay_ids,
        )
        env._v135_pending_prefix[replay_ids] = stage
        env._v135_replayed_prefix[replay_ids] = stage
        env._v135_prefix_counts[f"replay_b{stage}_count"] += replay_ids.numel()


def v135_chain_state(env, sensor_cfg=None):
    """Rebuild the saved prefix after V13.3's normal episode-reset clearing."""
    from .v133_adjacent_transition import v133_three_step_chain_state

    _ensure_replay_state(env)
    chain = v133_three_step_chain_state(env, sensor_cfg=sensor_cfg)
    pending = env._v135_pending_prefix
    for stage in (1, 2):
        ids = torch.nonzero(pending == stage, as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        snap = env._v135_prefix_pool[stage]
        # The first observation may have seen the next live face before this
        # ledger is restored.  It must not count that face as a new B1 lock.
        chain["b1_lock_count"] -= chain["b1_locked"][ids].long().sum()
        for name in _CHAIN_FIELDS:
            chain[name][ids] = snap[name][ids]
        chain["seed_event"][ids] = False
        chain["adjacent_event"][ids] = False
        chain["chain_event"][ids] = False
        chain["adjacent_progress_best"][ids] = 0.0
        chain["adjacent_progress_escrow"][ids] = 0.0
        pending[ids] = 0
    return chain


def _capture_prefixes(env, chain):
    from .v133_adjacent_transition import v133_ascending_stair_mask

    _ensure_replay_state(env)
    robot = env.scene["robot"]
    terrain = env.scene.terrain
    root = robot.data.root_state_w
    _, qx, qy, _ = root[:, 3:7].unbind(dim=1)
    upright = 1.0 - 2.0 * (qx.square() + qy.square())
    speed = torch.linalg.norm(root[:, 7:9], dim=1)
    cfg = env.cfg.v133_transition
    safe = (
        v133_ascending_stair_mask(env)
        & (upright >= float(cfg.prefix_min_upright_cosine))
        & (speed <= float(cfg.prefix_max_planar_speed))
        & ~env.termination_manager.terminated
        & ~env.termination_manager.time_outs
    )
    for stage, event in (
        (1, chain["seed_event"]),
        (2, chain["adjacent_event"] & (chain["chain_length"] == 2)),
    ):
        ids = torch.nonzero(safe & event, as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        if stage not in env._v135_prefix_pool:
            env._v135_prefix_pool[stage] = _new_snapshot(env, chain)
        snap = env._v135_prefix_pool[stage]
        snap["valid"][ids] = True
        snap["level"][ids] = terrain.terrain_levels[ids]
        snap["type"][ids] = terrain.terrain_types[ids]
        snap["origin"][ids] = terrain.env_origins[ids]
        snap["root"][ids] = root[ids]
        snap["root"][ids, :3] -= terrain.env_origins[ids]
        snap["joint_pos"][ids] = robot.data.joint_pos[ids]
        snap["joint_vel"][ids] = robot.data.joint_vel[ids]
        for name in _CHAIN_FIELDS:
            snap[name][ids] = chain[name][ids]
        for side, label in ((0, "fl"), (1, "fr")):
            env._v135_prefix_counts[f"capture_b{stage}_{label}_count"] += (
                chain["last_side"][ids] == side
            ).long().sum()


def v135_positive_ascent_chain_reward(env, sensor_cfg=None) -> torch.Tensor:
    """Pay B1/adjacent/B3 events and collect only true supported prefixes."""
    from .v133_adjacent_transition import v133_ascending_stair_mask

    chain = v135_chain_state(env, sensor_cfg=sensor_cfg)
    _capture_prefixes(env, chain)
    replayed = env._v135_replayed_prefix > 0
    env._v135_prefix_counts["full_chain_count"] += (chain["chain_event"] & ~replayed).long().sum()
    env._v135_prefix_counts["replay_chain_count"] += (chain["chain_event"] & replayed).long().sum()
    cfg = env.cfg.v133_transition
    settlement = (
        float(cfg.b1_settlement) * chain["seed_event"].float()
        + float(cfg.adjacent_progress_weight) * chain["adjacent_progress_delta"]
        + float(cfg.adjacent_transition_settlement) * chain["adjacent_event"].float()
        + float(cfg.chain_settlement) * chain["chain_event"].float()
    )
    return settlement * v133_ascending_stair_mask(env).float() / float(env.step_dt)


def v135_native_chain_critic_state(env, sensor_cfg=None) -> torch.Tensor:
    """The same 12 privileged values as V13.4, with replay ledger restored."""
    from .v13_alternation import v13_critic_descent_state

    chain = v135_chain_state(env, sensor_cfg=sensor_cfg)
    valid = chain["chain_valid"]
    expected = (1 - chain["last_side"]).clamp(0, 1)
    ascent = torch.stack((
        (valid & (expected == 0)).float(),
        (valid & (expected == 1)).float(),
        valid.float(),
        (chain["chain_length"].float() / 3.0).clamp(0.0, 1.0),
        chain["adjacent_progress_active"].float(),
        chain["adjacent_progress_best"],
    ), dim=1)
    return torch.cat((ascent, v13_critic_descent_state(env, sensor_cfg=sensor_cfg)), dim=1)


def v135_chain_metric(env, env_ids, key: str):
    from .v133_adjacent_transition import v133_chain_metric

    v135_chain_state(env)
    return v133_chain_metric(env, env_ids, key)


def v135_prefix_metric(env, env_ids, key: str):
    del env_ids
    _ensure_replay_state(env)
    if key in env._v135_prefix_counts:
        return env._v135_prefix_counts[key].float()
    if key == "replay_active_rate":
        return (env._v135_replayed_prefix > 0).float().mean()
    if key in ("snapshot_b1_available_rate", "snapshot_b2_available_rate"):
        stage = 1 if "b1" in key else 2
        snap = env._v135_prefix_pool.get(stage)
        return (
            snap["valid"].float().mean() if snap is not None
            else torch.zeros((), device=env.device)
        )
    raise KeyError(key)


__all__ = [
    "v135_prefix_replay_reset", "v135_chain_state",
    "v135_positive_ascent_chain_reward", "v135_native_chain_critic_state",
    "v135_chain_metric", "v135_prefix_metric",
]
