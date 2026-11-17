#!/usr/bin/env python3
"""Short, isolated V13.5 PPO continuation with gait/gradient attribution."""

from __future__ import annotations

import argparse, json, os, traceback
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--updates", type=int, default=10)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=13542)
parser.add_argument("--output", default="logs/diagnostics/v135_ppo_attribution/continuation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import rl_training.tasks  # noqa
from torch.utils.tensorboard import SummaryWriter
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.utils import check_nan
from rl_training.v135_ppo_attribution_runner import V135AttributionRunner

TASK = "M20-Stair-V13.5-V11FD-NativeAlternation-v0"


def main():
    torch.manual_seed(args.seed)
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(TASK, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner = V135AttributionRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(args.checkpoint, map_location=agent.device)
    obs = wrapped.get_observations().to(agent.device)
    writer = SummaryWriter(str(output / "tensorboard"))
    records = []
    runner.alg.train_mode()
    for update in range(args.updates):
        with torch.inference_mode():
            for _ in range(agent.num_steps_per_env):
                actions = runner.alg.act(obs)
                obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.device))
                check_nan(obs, rewards, dones)
                obs, rewards, dones = obs.to(agent.device), rewards.to(agent.device), dones.to(agent.device)
                runner.alg.process_env_step(obs, rewards, dones, extras)
            runner.alg.compute_returns(obs)
        losses = runner.alg.update()
        record = {"update": update, "loss": losses, **runner.alg.last_attribution}
        records.append(record)
        for gait_name, values in record["gaits"].items():
            tb_name = "NORMAL" if gait_name == "ORDINARY_NONSTRICT" else ("SKIP" if gait_name == "SKIP_DUAL" else gait_name)
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"Diag/V135PPO/{tb_name}/{key}", value, update)
        for key, value in record["gradient"].items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"Diag/V135PPO/Gradient/{key}", value, update)
        writer.flush()
        (output / "updates.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"V135_ATTRIBUTION_UPDATE={update} gradient={record['gradient']['status']}", flush=True)
    writer.close(); wrapped.close(); app.close()


if __name__ == "__main__":
    try: main()
    except BaseException:
        traceback.print_exc(); os._exit(1)
