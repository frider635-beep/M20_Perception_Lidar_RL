#!/usr/bin/env python3
"""Small Isaac runtime smoke for the isolated V13.1 task (no learning)."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument(
    "--task",
    type=str,
    default="M20-Stair-V13.1-V11FD-OrderedAlternation-v0",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from rl_training.tasks.manager_based.locomotion.velocity import mdp


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    obs, _ = env.reset()
    raw = env.unwrapped
    assert raw.action_manager.total_action_dim == 16
    assert obs["policy"].shape == (args.num_envs, 63)
    assert torch.isfinite(obs["policy"]).all()

    actions = torch.zeros((args.num_envs, 16), device=raw.device)
    seen_phases: set[int] = set()
    previous_chain_valid = torch.ones(args.num_envs, device=raw.device, dtype=torch.bool)
    for _ in range(args.steps):
        obs, reward, terminated, truncated, _ = env.step(actions)
        state = mdp.v131_ordered_state(raw)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(reward).all()
        assert state["phase"].shape == (args.num_envs,)
        assert state["expected_side"].shape == (args.num_envs,)
        assert state["teaching_expected_side"].shape == (args.num_envs,)
        assert state["teaching_target_riser_world_xy"].shape == (args.num_envs, 2)
        assert state["locked_target_potential"].shape == (args.num_envs, 2)
        assert state["first_mover_latched"].shape == (args.num_envs,)
        assert state["velocity_escrow"].shape == (args.num_envs,)
        assert state["b1_progress_escrow"].shape == (args.num_envs,)
        assert state["b1_velocity_escrow"].shape == (args.num_envs,)
        assert state["expected_swing_relief_active"].shape == (args.num_envs,)
        assert torch.isin(
            state["phase"],
            torch.tensor(
                [mdp.APPROACH_B1, mdp.ALTERNATING_RISERS, mdp.TOP_FOLLOW,
                 mdp.TOP_TRAILING_FOLLOW, mdp.PLATEAU_ROLL],
                device=raw.device,
            ),
        ).all()
        assert torch.isin(
            state["expected_side"], torch.tensor([-1, 0, 1], device=raw.device),
        ).all()
        assigned_active = (
            (state["phase"] == mdp.ALTERNATING_RISERS)
            | (state["phase"] == mdp.TOP_FOLLOW)
            | (state["phase"] == mdp.TOP_TRAILING_FOLLOW)
            | (state["phase"] == mdp.PLATEAU_ROLL)
        )
        assert (state["expected_side"][assigned_active] >= 0).all()
        assert (~state["skip_event"] | state["violation_event"]).all()
        assert torch.equal(state["ordered_chain_valid"], state["strict_chain_valid"])
        assert (~state["strict_chain_valid"] | previous_chain_valid).all()
        assert ((state["ordered_speed_gate"] >= 0.0) & (state["ordered_speed_gate"] <= 1.0)).all()
        assert (state["physical_last_riser_index"] >= state["ordered_last_riser_index"]).all()
        assert torch.equal(
            state["physical_minus_ordered_gap"],
            state["physical_last_riser_index"] - state["ordered_last_riser_index"],
        )
        assert torch.isfinite(state["b1_progress_delta"]).all()
        assert torch.isfinite(state["expected_progress_delta"]).all()
        assert torch.isfinite(state["locked_target_potential"]).all()
        assert ((state["locked_target_potential"] >= 0.0)
                & (state["locked_target_potential"] <= 1.0)).all()
        assert ((state["velocity_escrow"] >= 0.0) & (state["velocity_escrow"] <= 1.0)).all()
        assert ((state["b1_progress_escrow"] >= 0.0)
                & (state["b1_progress_escrow"] <= 1.0)).all()
        assert ((state["b1_velocity_escrow"] >= 0.0)
                & (state["b1_velocity_escrow"] <= 1.0)).all()
        assert (~state["expected_swing_relief_active"] | ~state["expected_target_support"]).all()
        assert (~state["first_mover_event"] | state["first_mover_latched"]).all()
        previous_chain_valid = state["strict_chain_valid"].clone()
        assert not (terminated & truncated).any()
        seen_phases.update(int(value) for value in state["phase"].unique().tolist())

    counts = {phase: int((state["phase"] == phase).sum()) for phase in sorted(seen_phases)}
    print(
        f"V131_RUNTIME_STEP_SMOKE: PASS envs={args.num_envs} steps={args.steps} "
        f"action_dim=16 phase_counts={counts}",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
