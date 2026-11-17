#!/usr/bin/env python3
"""Measure official-vs-V10 reward balance in train-faithful PPO rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--steps", type=int, default=984, help="Must be a multiple of PPO horizon.")
parser.add_argument("--horizon", type=int, default=24)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import rl_training.tasks  # noqa: F401
from rl_training.tasks.manager_based.locomotion.velocity import mdp
from rl_training.v10_unified_policy import UnifiedActor


TASK = "M20-Stair-Unified-Perceptive-v0"
V10_NAMES = (
    "first_step_success",
    "fr_clearance_progress",
    "perception_miss",
    "joint_success",
    "post_first_track_lin_vel_xy_exp",
)
GROUP_NAMES = (
    "official", "official_xy", "FIRST", "clearance", "miss", "FULL",
    "post_first_velocity", "v10_added", "all_reward",
)


def load_policy(path: str, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = UnifiedActor().to(device).eval()
    actor.load_state_dict(
        {k.removeprefix("actor."): v for k, v in checkpoint["model_state_dict"].items() if k.startswith("actor.")},
        strict=True,
    )
    log_std = checkpoint["model_state_dict"]["log_std"].to(device)
    return actor, torch.exp(log_std), int(checkpoint.get("iter", -1))


def stats(x: torch.Tensor) -> dict[str, float]:
    x = x.double().reshape(-1)
    if x.numel() == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "var": float("nan"),
                "sum": 0.0, "abs_sum": 0.0, "nonzero": 0}
    return {
        "n": int(x.numel()), "mean": float(x.mean()), "std": float(x.std(unbiased=False)),
        "var": float(x.var(unbiased=False)), "sum": float(x.sum()), "abs_sum": float(x.abs().sum()),
        "nonzero": int((x.abs() > 1.0e-12).sum()),
    }


def gae_reward_trace(rewards: torch.Tensor, dones: torch.Tensor, gamma_lam: float) -> torch.Tensor:
    # Additive reward-only portion of PPO GAE. The critic TD residual is deliberately separate.
    out = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        carry = rewards[t] + gamma_lam * (~dones[t]).unsqueeze(-1) * carry
        out[t] = carry
    return out


def main():
    if args.steps % args.horizon:
        raise ValueError("steps must be a multiple of horizon")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deliberately preserve the complete production training config: terrain distribution,
    # command sampler, curricula, randomization, observation corruption and actuator delays.
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset()
    actor, action_std, checkpoint_iteration = load_policy(args.checkpoint, raw.device)

    reward_names = list(raw.reward_manager._term_names)
    missing = [name for name in V10_NAMES if name not in reward_names]
    if missing:
        raise RuntimeError(f"missing V10 reward terms: {missing}; available={reward_names}")
    v10_idx = {name: reward_names.index(name) for name in V10_NAMES}
    official_idx = [i for i, name in enumerate(reward_names) if name not in V10_NAMES]
    full_topology = int(getattr(mdp, "V10_TOPOLOGY_FULL", 3))
    dt = float(raw.step_dt)
    gamma_lam = args.gamma * args.lam

    all_component_rows = []
    all_term_rows = []
    batch_rows = []
    episode_rows = []
    episode_acc = torch.zeros((args.num_envs, len(GROUP_NAMES)), device=raw.device)
    episode_steps = torch.zeros(args.num_envs, device=raw.device, dtype=torch.long)
    reward_match_max_error = 0.0
    post_first_match_max_error = 0.0
    post_first_inactive_max_abs = 0.0
    post_first_active_samples = 0
    post_first_mismatch_samples = 0
    post_first_first_mismatch = None
    first_seen_in_episode = torch.zeros(args.num_envs, device=raw.device, dtype=torch.bool)

    batch_components = []
    batch_terms = []
    batch_masks = []
    batch_dones = []

    for step in range(args.steps):
        pre_state = mdp.v10_state(raw)
        full_mask = pre_state["topology"] == full_topology
        with torch.inference_mode():
            mean = actor(obs["policy"])
            action = mean + torch.randn_like(mean) * action_std
            next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated

        # RewardManager stores func*configured_weight; env return additionally applies step_dt.
        terms = raw.reward_manager._step_reward.detach().clone() * dt
        official = terms[:, official_idx].sum(dim=1)
        official_xy = terms[:, reward_names.index("track_lin_vel_xy_exp")]
        first = terms[:, v10_idx["first_step_success"]]
        clearance = terms[:, v10_idx["fr_clearance_progress"]]
        miss = terms[:, v10_idx["perception_miss"]]
        full = terms[:, v10_idx["joint_success"]]
        post_first_velocity = terms[:, v10_idx["post_first_track_lin_vel_xy_exp"]]
        v10 = first + clearance + miss + full + post_first_velocity
        total = official + v10
        comp = torch.stack(
            (official, official_xy, first, clearance, miss, full,
             post_first_velocity, v10, total), dim=1
        )
        reward_match_max_error = max(reward_match_max_error, float((total - reward).abs().max()))

        # Independent runtime contract: the duplicate is exactly the official
        # XY tracking contribution from FIRST onward, surviving any encounter
        # rearm and ending only with episode reset.
        first_now = first.abs() > 1.0e-12
        expected_active = first_seen_in_episode | first_now
        official_xy = terms[:, reward_names.index("track_lin_vel_xy_exp")]
        expected_post = official_xy * expected_active.float()
        post_first_match_max_error = max(
            post_first_match_max_error, float((post_first_velocity - expected_post).abs().max())
        )
        mismatch = (post_first_velocity - expected_post).abs() > 1.0e-7
        post_first_mismatch_samples += int(mismatch.sum())
        if post_first_first_mismatch is None and mismatch.any():
            env_id = int(mismatch.nonzero(as_tuple=False)[0])
            runtime_state = mdp.v10_state(raw)
            post_first_first_mismatch = {
                "step": step, "env_id": env_id, "done": bool(done[env_id]),
                "first_term": float(first[env_id]), "seen_before": bool(first_seen_in_episode[env_id]),
                "expected_active": bool(expected_active[env_id]),
                "runtime_latch": bool(runtime_state["post_first_velocity_latched"][env_id]),
                "runtime_encounter_first_latch": bool(runtime_state["first_step_success_latched"][env_id]),
                "episode_length_buf": int(raw.episode_length_buf[env_id]),
                "official_xy": float(official_xy[env_id]), "post_reward": float(post_first_velocity[env_id]),
            }
        if (~expected_active).any():
            post_first_inactive_max_abs = max(
                post_first_inactive_max_abs, float(post_first_velocity[~expected_active].abs().max())
            )
        post_first_active_samples += int(expected_active.sum())
        first_seen_in_episode = expected_active
        first_seen_in_episode[done] = False

        if full_mask.any():
            all_component_rows.append(comp[full_mask].cpu())
            all_term_rows.append(terms[full_mask].cpu())
            episode_acc[full_mask] += comp[full_mask]
            episode_steps[full_mask] += 1

        ended = done & (episode_steps > 0)
        if ended.any():
            for env_id in ended.nonzero(as_tuple=False).flatten().tolist():
                row = {"env_id": env_id, "steps": int(episode_steps[env_id])}
                row.update({name: float(episode_acc[env_id, i]) for i, name in enumerate(GROUP_NAMES)})
                episode_rows.append(row)
        episode_acc[done] = 0.0
        episode_steps[done] = 0

        batch_components.append(comp)
        batch_terms.append(terms)
        batch_masks.append(full_mask)
        batch_dones.append(done)

        if len(batch_components) == args.horizon:
            c = torch.stack(batch_components)  # [T,N,groups]
            tr = torch.stack(batch_terms)
            mask = torch.stack(batch_masks)
            dones = torch.stack(batch_dones)
            trace = gae_reward_trace(c, dones, gamma_lam)
            selected = c[mask]
            selected_trace = trace[mask]
            selected_terms = tr[mask]
            batch_id = step // args.horizon
            row = {"batch": batch_id, "start_step": step - args.horizon + 1, "end_step": step,
                   "full_samples": int(mask.sum()), "full_envs": int(mask.any(dim=0).sum()),
                   "first_event_envs": int(((c[:, :, GROUP_NAMES.index("FIRST")].abs() > 1e-12) & mask).any(dim=0).sum()),
                   "full_event_envs": int(((c[:, :, GROUP_NAMES.index("FULL")].abs() > 1e-12) & mask).any(dim=0).sum())}
            signed_den = float(selected[:, -1].sum())
            abs_term_den = float(selected_terms.abs().sum())
            for i, name in enumerate(GROUP_NAMES):
                s = stats(selected[:, i])
                gs = stats(selected_trace[:, i])
                row.update({f"{name}_sum": s["sum"], f"{name}_mean": s["mean"],
                            f"{name}_var": s["var"], f"{name}_abs_sum": s["abs_sum"],
                            f"{name}_signed_share": s["sum"] / signed_den if abs(signed_den) > 1e-12 else float("nan"),
                            f"{name}_term_abs_mass_share": s["abs_sum"] / abs_term_den if abs_term_den > 0 else float("nan"),
                            f"{name}_gae_reward_mean": gs["mean"], f"{name}_gae_reward_var": gs["var"],
                            f"{name}_gae_reward_abs_sum": gs["abs_sum"]})

            # FIRST-centered PPO sub-batch: all 24 transitions for envs with a FIRST event.
            first_env = (
                (c[:, :, GROUP_NAMES.index("FIRST")].abs() > 1e-12) & mask
            ).any(dim=0)
            if first_env.any():
                fm = mask[:, first_env]
                fc = c[:, first_env][fm]
                ft = trace[:, first_env][fm]
                for i, name in enumerate(GROUP_NAMES):
                    fs, fgs = stats(fc[:, i]), stats(ft[:, i])
                    row.update({f"firstwin_{name}_sum": fs["sum"], f"firstwin_{name}_var": fs["var"],
                                f"firstwin_{name}_gae_reward_mean": fgs["mean"],
                                f"firstwin_{name}_gae_reward_var": fgs["var"]})
            batch_rows.append(row)
            batch_components.clear(); batch_terms.clear(); batch_masks.clear(); batch_dones.clear()

        obs = next_obs
        if (step + 1) % 120 == 0:
            print(f"PROGRESS step={step+1}/{args.steps} full_samples={sum(x.shape[0] for x in all_component_rows)} episodes={len(episode_rows)}", flush=True)

    components = torch.cat(all_component_rows, dim=0)
    terms_all = torch.cat(all_term_rows, dim=0)
    signed_total = float(components[:, -1].sum())
    abs_term_total = float(terms_all.abs().sum())
    summary = {
        "task": TASK, "checkpoint": args.checkpoint, "checkpoint_iteration": checkpoint_iteration,
        "num_envs": args.num_envs, "steps": args.steps, "ppo_horizon": args.horizon,
        "gamma": args.gamma, "lambda": args.lam, "seed": args.seed, "step_dt": dt,
        "full_topology_id": full_topology, "full_transition_samples": int(components.shape[0]),
        "completed_full_stair_episodes": len(episode_rows), "reward_return_match_max_error": reward_match_max_error,
        "post_first_match_max_error": post_first_match_max_error,
        "post_first_inactive_max_abs": post_first_inactive_max_abs,
        "post_first_active_samples": post_first_active_samples,
        "post_first_mismatch_samples": post_first_mismatch_samples,
        "post_first_first_mismatch": post_first_first_mismatch,
        "post_first_contract_pass": post_first_match_max_error <= 1.0e-7 and post_first_inactive_max_abs <= 1.0e-12,
        "reward_terms": reward_names, "official_terms": [reward_names[i] for i in official_idx],
        "v10_added_terms": list(V10_NAMES), "groups": {},
    }
    for i, name in enumerate(GROUP_NAMES):
        s = stats(components[:, i])
        s["signed_return_share"] = s["sum"] / signed_total if abs(signed_total) > 1e-12 else float("nan")
        s["term_abs_mass_share"] = s["abs_sum"] / abs_term_total if abs_term_total > 0 else float("nan")
        summary["groups"][name] = s

    term_rows = []
    for i, name in enumerate(reward_names):
        s = stats(terms_all[:, i])
        s.update({"term": name, "signed_return_share": s["sum"] / signed_total if abs(signed_total) > 1e-12 else float("nan"),
                  "term_abs_mass_share": s["abs_sum"] / abs_term_total if abs_term_total > 0 else float("nan")})
        term_rows.append(s)

    def write_csv(path: Path, rows: list[dict]):
        if not rows:
            return
        keys = sorted({k for row in rows for k in row})
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(out_dir / "term_stats.csv", term_rows)
    write_csv(out_dir / "ppo_batch_stats.csv", batch_rows)
    write_csv(out_dir / "episode_stats.csv", episode_rows)
    print("REWARD_BALANCE_AUDIT=PASS")
    print(f"POST_FIRST_RUNTIME_CONTRACT={'PASS' if summary['post_first_contract_pass'] else 'FAIL'}")
    print(f"OUTPUT_DIR={out_dir}")
    print(json.dumps(summary["groups"], indent=2), flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
