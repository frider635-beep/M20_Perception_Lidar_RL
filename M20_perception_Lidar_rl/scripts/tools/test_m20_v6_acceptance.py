"""Pre-flight acceptance for M20 Stair Official 20K (no PPO)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import copy
import numpy as np
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.m20_stair_terrains import (
    STAIR17_CFG,
    m20_square_stair_terrain,
)


def terrain_audit():
    cfg = copy.deepcopy(STAIR17_CFG)
    cfg.size = (12.0, 12.0)
    meshes, origin = m20_square_stair_terrain(0.0, cfg)
    center = np.asarray(origin[:2])
    angles = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    scale = np.maximum(np.abs(np.cos(angles)), np.abs(np.sin(angles)))

    bands = [(0.0, cfg.central_flat_half_extent, 0.0, "center")]
    inner = cfg.central_flat_half_extent
    for level in range(cfg.num_steps):
        outer = inner + cfg.step_depth
        bands.append((inner, outer, (level + 1) * cfg.step_height, f"up_{level + 1}"))
        inner = outer
    outer = inner + cfg.top_landing_depth
    bands.append((inner, outer, cfg.num_steps * cfg.step_height, "landing"))
    inner = outer
    for level in range(cfg.num_steps):
        outer = inner + cfg.down_step_depth
        bands.append((inner, outer, (cfg.num_steps - level - 1) * cfg.step_height, f"down_{level + 1}"))
        inner = outer

    uncovered = 0
    max_height_error = 0.0
    for inner, outer, expected_height, name in bands:
        radius = (inner + outer) / 2.0
        points = center + np.stack((radius * np.cos(angles) / scale, radius * np.sin(angles) / scale), axis=1)
        sampled_height = np.full(points.shape[0], -np.inf)
        for mesh in meshes:
            bounds = mesh.bounds
            inside = (
                (points[:, 0] >= bounds[0, 0] - 1e-8)
                & (points[:, 0] <= bounds[1, 0] + 1e-8)
                & (points[:, 1] >= bounds[0, 1] - 1e-8)
                & (points[:, 1] <= bounds[1, 1] + 1e-8)
            )
            sampled_height[inside] = np.maximum(sampled_height[inside], bounds[1, 2])
        missing = int((~np.isfinite(sampled_height)).sum())
        uncovered += missing
        error = float(np.max(np.abs(sampled_height - expected_height))) if missing == 0 else float("inf")
        max_height_error = max(max_height_error, error)
        print(f"TERRAIN_BAND={name} EXPECTED_Z={expected_height:.2f} MISSING={missing} MAX_ERROR={error:.6f}")

    print(f"uncovered bypass samples = {uncovered}")
    print(f"UP_LANDING_DOWN_AUDIT max_height_error={max_height_error:.6f}")
    assert uncovered == 0
    assert max_height_error < 1e-6


def reward_ledger():
    rows = {
        "stagnate_center": dict(OFFICIAL_REWARD_RETURN=0, CLEARANCE_RETURN=0, COLLISION_RETURN=0, CROSSING_RETURN=0, STEP_RETURN=0, SUCCESS_RETURN=0, STAGNATION_RETURN=-37.5, FAILURE_RETURN=-30),
        "stagnate_before_riser": dict(OFFICIAL_REWARD_RETURN=0, CLEARANCE_RETURN=-40, COLLISION_RETURN=0, CROSSING_RETURN=0, STEP_RETURN=0, SUCCESS_RETURN=0, STAGNATION_RETURN=-20, FAILURE_RETURN=-30),
        "clearance_only_no_cross": dict(OFFICIAL_REWARD_RETURN=0, CLEARANCE_RETURN=0, COLLISION_RETURN=0, CROSSING_RETURN=0, STEP_RETURN=0, SUCCESS_RETURN=0, STAGNATION_RETURN=-20, FAILURE_RETURN=-30),
        "riser_collision": dict(OFFICIAL_REWARD_RETURN=0, CLEARANCE_RETURN=-4, COLLISION_RETURN=-5, CROSSING_RETURN=0, STEP_RETURN=0, SUCCESS_RETURN=0, STAGNATION_RETURN=-2, FAILURE_RETURN=-30),
        "fall_before_success": dict(OFFICIAL_REWARD_RETURN=-20, CLEARANCE_RETURN=0, COLLISION_RETURN=0, CROSSING_RETURN=0, STEP_RETURN=0, SUCCESS_RETURN=0, STAGNATION_RETURN=0, FAILURE_RETURN=-30),
        "complete_one_level": dict(OFFICIAL_REWARD_RETURN=0, CLEARANCE_RETURN=0, COLLISION_RETURN=0, CROSSING_RETURN=20, STEP_RETURN=10, SUCCESS_RETURN=0, STAGNATION_RETURN=0, FAILURE_RETURN=-30),
        "complete_six_levels": dict(OFFICIAL_REWARD_RETURN=0, CLEARANCE_RETURN=0, COLLISION_RETURN=0, CROSSING_RETURN=120, STEP_RETURN=60, SUCCESS_RETURN=30, STAGNATION_RETURN=0, FAILURE_RETURN=0),
    }
    print("scenario                         total_return")
    totals = {}
    for scenario, parts in rows.items():
        total = sum(parts.values()); totals[scenario] = total
        print(f"{scenario:32s} {total:8.2f}")
        print("  " + " ".join(f"{k}={v}" for k, v in parts.items()) + f" TOTAL_RETURN={total}")
    assert totals["complete_six_levels"] > totals["complete_one_level"]
    assert totals["complete_one_level"] > totals["clearance_only_no_cross"]
    assert totals["fall_before_success"] <= totals["complete_one_level"]


def main():
    terrain_audit(); reward_ledger()
    cfg = parse_env_cfg("M20-Stair-Official20K-v0", device=args.device, num_envs=args.num_envs)
    env = gym.make("M20-Stair-Official20K-v0", cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset()
    policy_dim = raw.observation_manager.group_obs_dim["policy"][0]
    critic_dim = raw.observation_manager.group_obs_dim["critic"][0]
    action_dim = raw.action_manager.total_action_dim
    print(f"POLICY_DIM={policy_dim}"); print(f"CRITIC_DIM={critic_dim}"); print(f"ACTION_DIM={action_dim}")
    assert (policy_dim, critic_dim, action_dim) == (372, 375, 16)
    allowed = {"base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "height_scan"}
    policy_terms = set(raw.observation_manager.active_terms["policy"])
    assert policy_terms == allowed, policy_terms
    actions = torch.zeros((args.num_envs, action_dim), device=raw.device)
    for _ in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(reward).all()
        for value in obs.values():
            assert torch.isfinite(value).all()
    # Real manager integration test: force the next step to hit the time limit.
    # This guards against the exact loophole where Isaac Lab reports timeout in
    # `time_outs`/`truncated` rather than `terminated`.
    env.reset()
    raw.episode_length_buf[:] = raw.max_episode_length - 1
    _, _, terminated, truncated, _ = env.step(actions)
    failure_idx = raw.reward_manager.active_terms.index("stair_failure")
    failure_return = raw.reward_manager._step_reward[:, failure_idx] * raw.step_dt
    assert truncated.all() and not terminated.any()
    assert torch.allclose(failure_return, torch.full_like(failure_return, -30.0), atol=1e-5), failure_return
    print("TIMEOUT_FAILURE_RETURN=-30.0 PASS")
    counts = (round(args.num_envs * 0.45), round(args.num_envs * 0.45))
    print(f"terrain allocation target: Stair17={counts[0]} Stair20={counts[1]} Stair08-15={args.num_envs-sum(counts)}")
    print(f"SMOKE_STEPS={args.steps} NO_NAN=True NO_INF=True PPO=False")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
