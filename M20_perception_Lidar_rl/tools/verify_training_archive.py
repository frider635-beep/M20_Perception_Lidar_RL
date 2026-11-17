#!/usr/bin/env python3
"""Verify that the archived run contains all 30000 training iterations."""

from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch"
models = list(RUN.glob("model_*.pt"))
expected_periodic = {f"model_{step}.pt" for step in range(0, 30000, 100)}
actual_names = {path.name for path in models}
missing = sorted(expected_periodic - actual_names)
if missing or "model_29999.pt" not in actual_names:
    raise RuntimeError(f"Incomplete checkpoint archive: missing={missing[:10]}")
if len(models) != 301:
    raise RuntimeError(f"Expected 301 checkpoints, found {len(models)}")

checkpoint = torch.load(RUN / "model_29999.pt", map_location="cpu", weights_only=True)
if checkpoint.get("iter") != 29999:
    raise RuntimeError(f"Final checkpoint iteration is {checkpoint.get('iter')!r}, expected 29999")

events = list(RUN.glob("events.out.tfevents.*"))
if len(events) != 1:
    raise RuntimeError(f"Expected one TensorBoard event file, found {len(events)}")
accumulator = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
accumulator.Reload()
for tag in ("Train/mean_reward", "Train/mean_episode_length"):
    points = accumulator.Scalars(tag)
    if len(points) != 30000 or points[-1].step != 29999:
        raise RuntimeError(
            f"Incomplete TensorBoard series {tag}: count={len(points)}, last_step={points[-1].step}"
        )
print("TRAINING_ARCHIVE=PASS checkpoints=301 scalar_iterations=30000 final_checkpoint_iter=29999")
