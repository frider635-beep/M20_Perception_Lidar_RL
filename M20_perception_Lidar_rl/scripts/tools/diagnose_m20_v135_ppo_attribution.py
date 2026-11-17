#!/usr/bin/env python3
"""Frozen V13.5 checkpoint attribution and recurrent-memory probes."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run_dir", required=True)
parser.add_argument("--checkpoints", nargs="+", default=["latest"])
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=480)
parser.add_argument("--seed", type=int, default=13542)
parser.add_argument("--window", type=int, choices=(16, 32), default=16)
parser.add_argument("--output_root", default="logs/diagnostics/v135_ppo_attribution")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rl_training.v11_continuous_runner import V11ContinuousRunner
from rl_training.v135_ppo_attribution_runner import (
    ALT, B1_FREE, LABEL_NAMES, NONE, ORDINARY_NONSTRICT, SKIP_DUAL,
    WRONG_OR_SAME_SIDE, ParameterSnapshot, classify_v135_frames,
    expand_event_windows, matched_gait_summary, raw_gae,
)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v11_continuous import v11_state
from rl_training.tasks.manager_based.locomotion.velocity.mdp.v135_riser_identity import v135_riser_identity_state

TASK = "M20-Stair-V13.5-V11FD-NativeAlternation-v0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoints(run_dir: Path, requested: list[str]) -> list[Path]:
    existing = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not existing:
        raise FileNotFoundError(f"no model_*.pt under {run_dir}")
    iterations = np.asarray([int(p.stem.split("_")[-1]) for p in existing])
    selected = []
    for item in requested:
        if item == "latest":
            candidate = existing[-1]
        else:
            target = int(item)
            candidate = existing[int(np.abs(iterations - target).argmin())]
        if candidate not in selected:
            selected.append(candidate)
    return selected


def tensor(x):
    return x.detach().cpu().numpy()


def term_indices(raw):
    major_tokens = (
        "track_lin_vel", "world_forward", "landing", "clearance", "v13_", "v135_",
        "action_rate", "action_smooth", "termination", "orientation", "base_height",
    )
    names = list(raw.reward_manager.active_terms)
    chosen = [i for i, name in enumerate(names) if any(token in name for token in major_tokens)]
    return names, chosen


def summarize_by_label(labels, rewards, returns, values, adv_raw, adv_norm):
    result = {}
    for label_id, name in LABEL_NAMES.items():
        mask = labels == label_id
        count = int(mask.sum())
        entry = {"sample_count": count, "sample_fraction": count / labels.size}
        if count:
            for key, array in (("reward_mean", rewards), ("return_mean", returns),
                               ("value_mean", values), ("adv_raw_mean", adv_raw),
                               ("adv_raw_std", adv_raw), ("adv_normalized_mean", adv_norm)):
                entry[key] = float(array[mask].mean()) if key != "adv_raw_std" else float(array[mask].std())
            entry["value_error_mean"] = float((values[mask] - returns[mask]).mean())
            entry["positive_advantage_fraction"] = float((adv_raw[mask] > 0).mean())
        result[name] = entry
    return result


def probe_expected_side(obs, hidden, expected, eligible, groups, seed):
    """Fit read-only Torch probes with episode-group-disjoint splits."""
    idx = np.flatnonzero(eligible & ((expected == 0) | (expected == 1)))
    if idx.size < 100 or np.unique(expected[idx]).size < 2:
        return {"status": "INSUFFICIENT_SAMPLES", "samples": int(idx.size)}
    unique_groups = np.unique(groups[idx])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    cut = max(1, int(0.8 * len(unique_groups)))
    train_groups = set(unique_groups[:cut].tolist())
    train = idx[np.array([g in train_groups for g in groups[idx]])]
    test = idx[np.array([g not in train_groups for g in groups[idx]])]
    if test.size < 20:
        return {"status": "INSUFFICIENT_TEST_GROUPS", "samples": int(idx.size)}
    y = expected.astype(np.int64)
    output = {"status": "PASS", "samples": int(idx.size), "train": int(train.size), "test": int(test.size)}

    def metrics(y_true, probability):
        prediction = (probability >= 0.5).astype(np.int64)
        cm = np.zeros((2, 2), dtype=np.int64)
        for truth, pred in zip(y_true, prediction): cm[truth, pred] += 1
        recalls = [cm[c, c] / max(1, cm[c].sum()) for c in (0, 1)]
        # Mann-Whitney AUC without allocating an O(N_pos*N_neg) matrix.
        order = np.argsort(probability, kind="mergesort")
        sorted_scores = probability[order]
        ranks = np.empty(order.size, dtype=np.float64)
        lo = 0
        while lo < order.size:
            hi = lo + 1
            while hi < order.size and sorted_scores[hi] == sorted_scores[lo]: hi += 1
            ranks[order[lo:hi]] = 0.5 * (lo + hi - 1) + 1.0
            lo = hi
        n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
        auc = float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / max(1, n_pos * n_neg))
        return {"accuracy": float((prediction == y_true).mean()),
                "balanced_accuracy": float(np.mean(recalls)), "auc": auc,
                "confusion_matrix": cm.tolist()}

    def train_probe(x_train, y_train, x_test, nonlinear):
        torch.manual_seed(seed)
        mean = x_train.mean(0, keepdim=True)
        std = x_train.std(0, keepdim=True).clamp_min(1.0e-6)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        model = (torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], 32), torch.nn.Tanh(),
                                     torch.nn.Linear(32, 1)) if nonlinear
                 else torch.nn.Linear(x_train.shape[1], 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
        target = y_train.float().unsqueeze(1)
        generator = torch.Generator().manual_seed(seed)
        for _ in range(120):
            order = torch.randperm(x_train.shape[0], generator=generator)
            for start in range(0, order.numel(), 2048):
                batch = order[start:start + 2048]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x_train[batch]), target[batch])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            return torch.sigmoid(model(x_test)).squeeze(1).numpy()

    for feature_name, features in (("raw63", obs), ("gru128", hidden)):
        output[feature_name] = {}
        train_x = torch.from_numpy(features[train]).float()
        test_x = torch.from_numpy(features[test]).float()
        train_y = torch.from_numpy(y[train])
        for model_name, nonlinear in (("logistic", False), ("small_mlp", True)):
            probability = train_probe(train_x, train_y, test_x, nonlinear)
            output[feature_name][model_name] = metrics(y[test], probability)
    return output


def evaluate_checkpoint(checkpoint: Path, output: Path, runtime: dict):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Load the runner config before environment construction.  IsaacLab's
    # registry objects are shared and environment creation may mutate config
    # instances; this is also the ordering used by the production smoke test.
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = V11ContinuousRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), map_location=agent_cfg.device)
    snapshot = ParameterSnapshot.capture(runner)
    runner.alg.actor.eval(); runner.alg.critic.eval()
    obs = wrapped.get_observations()
    runner.alg.actor.reset()
    names, reward_columns = term_indices(raw)
    major_names = [names[i] for i in reward_columns]
    episode_id = torch.zeros(args.num_envs, device=raw.device, dtype=torch.long)
    buffers: dict[str, list[np.ndarray]] = {key: [] for key in (
        "policy_obs", "hidden", "value", "reward", "done", "label_event", "terrain_level",
        "terrain_type", "command_vx", "root_xyz", "base_velocity", "wheel_xyz", "contact",
        "support", "target_id", "target_world_xy", "target_valid", "seeded", "expected_side",
        "valid", "wrong_side", "duplicate", "skip", "dual_skip", "chain_failed",
        "v11_crossing", "v11_completion", "current_potential", "best_potential",
        "episode_id", "episode_step", "reward_components",
        "step", "env_id",
    )}
    asset = raw.scene["robot"]
    env_ids = torch.arange(args.num_envs, device=raw.device, dtype=torch.long)
    for step_index in range(args.steps):
        policy_obs = obs["policy"]
        with torch.inference_mode():
            value = runner.alg.critic(obs).squeeze(-1)
            actions = runner.alg.actor(obs, stochastic_output=False)
            hidden = runner.alg.actor.get_hidden_state()
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
            hidden = hidden[-1]
        next_obs, rewards, dones, extras = wrapped.step(actions)
        strict = v135_riser_identity_state(raw)
        base = v11_state(raw)
        landing = base["landing"]
        labels = classify_v135_frames(strict, landing)
        v13 = getattr(raw, "_v13_alternation", {})
        current_potential = v13.get("potential", torch.zeros((args.num_envs, 2), device=raw.device))
        best_potential = v13.get("best_potential", torch.zeros((args.num_envs, 2), device=raw.device))
        level = getattr(raw.scene.terrain, "terrain_levels", torch.zeros(args.num_envs, device=raw.device, dtype=torch.long))
        terrain_type = raw.scene.terrain.terrain_types
        wheel_xyz = asset.data.body_pos_w[:, raw._v11_front_body_ids]
        contacts = landing.get("vertical_contact_now", torch.zeros((args.num_envs, 2), device=raw.device, dtype=torch.bool))
        values = {
            "policy_obs": policy_obs, "hidden": hidden, "value": value, "reward": rewards,
            "done": dones, "label_event": labels, "terrain_level": level,
            "terrain_type": terrain_type, "command_vx": raw.command_manager.get_command("base_velocity")[:, 0],
            "root_xyz": asset.data.root_pos_w, "base_velocity": asset.data.root_lin_vel_w,
            "wheel_xyz": wheel_xyz, "contact": contacts, "support": landing["support"],
            "target_id": strict["target_id"], "target_world_xy": strict["target_world_xy"],
            "target_valid": strict["target_valid"], "seeded": strict["seeded"],
            "expected_side": strict["expected_side"], "valid": strict["valid"],
            "wrong_side": strict["wrong_side"], "duplicate": strict["duplicate"],
            "skip": strict["skip"], "dual_skip": strict["dual_skip"],
            "chain_failed": strict["ordered_chain_failed"],
            "v11_crossing": landing["front_crossing_event"], "v11_completion": landing["landing_event"],
            "current_potential": current_potential, "best_potential": best_potential,
            "episode_id": episode_id, "episode_step": raw.episode_length_buf,
            "reward_components": raw.reward_manager._step_reward[:, reward_columns],
            "step": torch.full_like(env_ids, step_index), "env_id": env_ids,
        }
        for key, value_tensor in values.items(): buffers[key].append(tensor(value_tensor))
        episode_id += dones.long()
        runner.alg.actor.reset(dones)
        obs = next_obs
    with torch.inference_mode():
        last_values = runner.alg.critic(obs).squeeze(-1)
    rewards_t = torch.as_tensor(np.stack(buffers["reward"]), device=raw.device)
    values_t = torch.as_tensor(np.stack(buffers["value"]), device=raw.device)
    dones_t = torch.as_tensor(np.stack(buffers["done"]), device=raw.device)
    adv_raw_t, returns_t = raw_gae(rewards_t, values_t, dones_t, last_values,
                                   runner.alg.gamma, runner.alg.lam)
    adv_norm_t = (adv_raw_t - adv_raw_t.mean()) / (adv_raw_t.std() + 1.0e-8)
    event_labels_t = torch.as_tensor(np.stack(buffers["label_event"]), device=raw.device)
    window_labels_t = expand_event_windows(event_labels_t, args.window)
    buffers["label_window"] = [tensor(window_labels_t)]
    buffers["advantage_raw"] = [tensor(adv_raw_t)]
    buffers["advantage_normalized"] = [tensor(adv_norm_t)]
    buffers["return"] = [tensor(returns_t)]
    arrays = {key: (value[0] if key in {"label_window", "advantage_raw", "advantage_normalized", "return"}
                    else np.stack(value)) for key, value in buffers.items()}
    flat = lambda key: arrays[key].reshape(-1, *arrays[key].shape[2:])
    labels_flat = flat("label_window")
    episode_progress = flat("episode_step").astype(np.float32) / float(raw.max_episode_length)
    summary = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "iteration": int(checkpoint.stem.split("_")[-1]), "seed": args.seed,
        "num_envs": args.num_envs, "steps": args.steps, "window_radius": args.window,
        "abi": {"actor_obs": int(obs["policy"].shape[-1]), "critic_obs": int(obs["critic"].shape[-1]),
                "action": int(raw.action_manager.total_action_dim), "gru_hidden": 128},
        "labels": summarize_by_label(labels_flat, flat("reward"), flat("return"), flat("value"),
                                     flat("advantage_raw"), flat("advantage_normalized")),
        "matched": matched_gait_summary(
            torch.as_tensor(labels_flat), torch.as_tensor(flat("return")),
            torch.as_tensor(flat("advantage_raw")), torch.as_tensor(flat("terrain_level")),
            torch.as_tensor(flat("command_vx")), torch.as_tensor(flat("target_id")),
            torch.as_tensor(episode_progress)).copy(),
        "reward_component_names": major_names,
        "reward_weights": {name: float(raw.reward_manager.get_term_cfg(name).weight) for name in names},
        "production_contract": {
            "task": TASK, "runner": "rl_training.v11_continuous_runner:V11ContinuousRunner",
            "policy_class": "V135RecurrentActorCritic", "rollout_steps": int(agent_cfg.num_steps_per_env),
            "ppo": {key: getattr(agent_cfg.algorithm, key) for key in (
                "num_learning_epochs", "num_mini_batches", "learning_rate", "clip_param",
                "gamma", "lam", "desired_kl", "entropy_coef")},
        },
    }
    boundary = flat("valid") | flat("wrong_side") | flat("skip") | flat("dual_skip") | flat("done")
    eligible = flat("seeded").astype(bool) & ~boundary.astype(bool) & (flat("episode_step") > 5)
    groups = (flat("episode_id").astype(np.int64) * args.num_envs +
              np.tile(np.arange(args.num_envs), args.steps))
    summary["memory_probe"] = probe_expected_side(
        flat("policy_obs"), flat("hidden"), flat("expected_side"), eligible, groups, args.seed)
    # A crossing is an earlier kinematic phase and is not expected to settle
    # the strict ledger.  Detector disagreement is therefore evaluated only
    # at V11 physical completion events.
    physical = flat("v11_completion").astype(bool)
    strict_any = flat("valid").astype(bool) | flat("wrong_side").astype(bool) | flat("skip").astype(bool) | flat("dual_skip").astype(bool)
    actionable = flat("target_valid").astype(bool)
    disagreement = actionable & physical & ~strict_any
    summary["detector"] = {
        "actionable_physical_events": int((actionable & physical).sum()),
        "label_disagreement_count": int(disagreement.sum()),
        "label_disagreement_rate": float(disagreement.sum() / max(1, (actionable & physical).sum())),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "trace.npz", **arrays, reward_component_names=np.asarray(major_names))
    # Deterministic detector audit windows, drawn from the already recorded
    # physical ledger.  Missing categories are explicitly reported.
    audit = {}
    audit_sources = {"ALT": ALT, "ORDINARY_NONSTRICT": ORDINARY_NONSTRICT,
                     "SKIP_DUAL": SKIP_DUAL, "B1_FREE": B1_FREE}
    trace_keys = ("step", "env_id", "episode_id", "target_id", "expected_side", "wheel_xyz", "contact",
                  "v11_crossing", "support", "valid", "wrong_side", "skip", "dual_skip", "duplicate", "chain_failed")
    event_labels = arrays["label_event"]
    for audit_name, audit_id in audit_sources.items():
        centers = np.argwhere(event_labels == audit_id)[:20]
        rows = []
        for center, env_id in centers:
            lo, hi = max(0, int(center)-args.window), min(args.steps, int(center)+args.window+1)
            rows.append({key: arrays[key][lo:hi, env_id].tolist() for key in trace_keys})
        audit[audit_name] = {"candidate_count": int((event_labels == audit_id).sum()), "sampled": rows}
    (output / "event_traces.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    snapshot.assert_unchanged(runner)
    wrapped.close()
    return summary


def main():
    run_dir = Path(args.run_dir).resolve()
    output_root = Path(args.output_root).resolve()
    checkpoints = resolve_checkpoints(run_dir, args.checkpoints)
    if len(checkpoints) != 1:
        raise ValueError("one checkpoint per Isaac process is required; use run_m20_v135_attribution_suite.sh")
    runtime = {
        "ppo": {"class": f"{PPO.__module__}.{PPO.__qualname__}", "file": inspect.getsourcefile(PPO)},
        "storage": {"class": f"{RolloutStorage.__module__}.{RolloutStorage.__qualname__}", "file": inspect.getsourcefile(RolloutStorage)},
        "runner": {"class": f"{OnPolicyRunner.__module__}.{OnPolicyRunner.__qualname__}", "file": inspect.getsourcefile(OnPolicyRunner)},
    }
    reports = []
    for checkpoint in checkpoints:
        out = output_root / checkpoint.stem
        print(f"V135_ATTRIBUTION_CHECKPOINT={checkpoint}", flush=True)
        reports.append(evaluate_checkpoint(checkpoint, out, runtime))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runtime_classes.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (output_root / "checkpoint_summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(output_root / "tensorboard"))
    for report in reports:
        step = report["iteration"]
        for source, tag in (("ALT", "ALT"), ("ORDINARY_NONSTRICT", "NORMAL"), ("SKIP_DUAL", "SKIP")):
            values = report["labels"][source]
            writer.add_scalar(f"Diag/V135PPO/{tag}/sample_fraction", values["sample_fraction"], step)
            for field, out_name in (("adv_raw_mean", "adv_raw_mean"), ("return_mean", "return_mean"),
                                    ("value_error_mean", "value_error_mean")):
                if field in values: writer.add_scalar(f"Diag/V135PPO/{tag}/{out_name}", values[field], step)
        writer.add_scalar("Diag/V135PPO/Detector/label_disagreement_rate",
                          report["detector"]["label_disagreement_rate"], step)
        probe = report["memory_probe"]
        if probe.get("status") == "PASS":
            for feature, tag in (("raw63", "raw63_probe_accuracy"), ("gru128", "gru128_probe_accuracy")):
                writer.add_scalar(f"Diag/V135PPO/Memory/{tag}", probe[feature]["logistic"]["accuracy"], step)
                writer.add_scalar(f"Diag/V135PPO/Memory/{feature}_mlp_accuracy",
                                  probe[feature]["small_mlp"]["accuracy"], step)
    writer.close()
    print(f"V135_PPO_ATTRIBUTION_FROZEN=PASS output={output_root}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
