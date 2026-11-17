#!/usr/bin/env python3
"""Isaac smoke and V11-FD delta audit for the V11.1 rear-posture branch."""

import argparse
import copy
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import rl_training.tasks  # noqa: F401
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

TASK = "M20-Stair-V11.1-V11FD-RearPosture-v0"
REFERENCE = Path(__file__).resolve().parents[2] / "logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch/params"


def _remove(mapping, path):
    node = mapping
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]


def _diff(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [f"{path}: keys {sorted(set(left) ^ set(right))}"]
        out = []
        for key in sorted(left):
            out.extend(_diff(left[key], right[key], f"{path}.{key}" if path else key))
        return out
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return [f"{path}: length {len(left)} != {len(right)}"]
        out = []
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            out.extend(_diff(a, b, f"{path}[{index}]"))
        return out
    try:
        if isinstance(left == right, bool) and left == right:
            return []
    except Exception:
        pass
    return [] if repr(left) == repr(right) else [f"{path}: {left!r} != {right!r}"]


def main():
    import yaml

    saved_env = yaml.load((REFERENCE / "env.yaml").read_text(), Loader=yaml.UnsafeLoader)
    saved_agent = yaml.load((REFERENCE / "agent.yaml").read_text(), Loader=yaml.UnsafeLoader)
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    candidate = copy.deepcopy(cfg.to_dict())
    _remove(candidate, ("v111_rear_posture",))
    for name in (
        "v111_descent_rear_extension", "v111_descent_rear_knee_flexion",
        "v111_descent_rear_roll", "v111_level_exit_body_clearance",
    ):
        _remove(candidate, ("rewards", name))
    for name in list(candidate["curriculum"]):
        if name.startswith("v111_"):
            _remove(candidate, ("curriculum", name))
    for name in ("safe_front_landing_progress", "single_front_hard_landing", "descent_rear_support_progress"):
        if name not in saved_env["rewards"]:
            assert candidate["rewards"].pop(name) is None
    for name in (
        "v11_descent_rear_support_event_rate", "v11_fd_safe_support_event_rate",
        "v11_fd_hard_candidate_event_rate",
    ):
        if name not in saved_env["curriculum"]:
            assert candidate["curriculum"].pop(name) is None
    for section in (
        "actions", "commands", "observations", "events", "terminations",
        "rewards", "curriculum",
    ):
        differences = _diff(saved_env[section], candidate[section], section)
        if differences:
            raise AssertionError("unexpected V11.1 drift from saved V11-FD:\n" + "\n".join(differences[:30]))
    for key, value in saved_env["v11_perception"].items():
        differences = _diff(value, candidate["v11_perception"][key], f"v11_perception.{key}")
        if differences:
            raise AssertionError("unexpected V11.1 perception drift:\n" + "\n".join(differences))
    assert candidate["v11_perception"]["enable_persistent_descent_rear_phase"] is False
    assert candidate["v11_perception"]["soft_alternation"] is False
    for path in (
        ("scene", "terrain", "terrain_generator"),
        ("scene", "terrain", "max_init_terrain_level"),
    ):
        left, right = saved_env, candidate
        for key in path:
            left, right = left[key], right[key]
        differences = _diff(left, right, ".".join(path))
        if differences:
            raise AssertionError("unexpected V11.1 terrain drift:\n" + "\n".join(differences[:30]))

    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    candidate_agent = agent.to_dict()
    for section in ("policy", "algorithm"):
        for key, value in saved_agent[section].items():
            differences = _diff(value, candidate_agent[section][key], f"{section}.{key}")
            if differences:
                raise AssertionError("unexpected V11.1 PPO drift:\n" + "\n".join(differences[:30]))
    for key in ("num_steps_per_env", "max_iterations", "save_interval", "resume", "load_run", "load_checkpoint"):
        assert saved_agent[key] == candidate_agent[key], key

    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    obs = raw.observation_manager.compute()
    assert obs["policy"].shape[1] == 63 and raw.action_manager.total_action_dim == 16
    assert not raw.cfg.v11_perception.enable_persistent_descent_rear_phase
    terms = raw.reward_manager.active_terms
    assert "v111_descent_rear_extension" in terms
    assert "v111_descent_rear_knee_flexion" in terms
    assert "v111_descent_rear_roll" in terms
    assert "v111_level_exit_body_clearance" in terms
    assert not any("thigh" in name.lower() for name in terms)
    assert raw.reward_manager.get_term_cfg("v111_descent_rear_extension").func is mdp.v111_descent_rear_extension_penalty
    assert raw.reward_manager.get_term_cfg("v111_descent_rear_knee_flexion").func is mdp.v111_descent_rear_knee_flexion_penalty
    assert raw.reward_manager.get_term_cfg("v111_descent_rear_roll").func is mdp.v111_descent_rear_roll_penalty
    assert raw.reward_manager.get_term_cfg("v111_level_exit_body_clearance").func is mdp.v111_level_exit_body_clearance_penalty
    env.reset()
    actions = torch.zeros((args.num_envs, 16), device=raw.device)
    for _ in range(args.steps):
        observations, rewards, terminated, truncated, extras = env.step(actions)
        assert torch.isfinite(observations["policy"]).all()
        assert torch.isfinite(observations["critic"]).all()
        assert torch.isfinite(rewards).all()
    print(
        f"V111_RUNTIME_SMOKE: PASS envs={args.num_envs} steps={args.steps} "
        f"actor={obs['policy'].shape[1]} critic={obs['critic'].shape[1]} action=16 thigh_penalty=NO",
        flush=True,
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
