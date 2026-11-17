#!/usr/bin/env python3
"""Audit required lifecycle TensorBoard tags from a V10 smoke run."""

from __future__ import annotations

import argparse
import math

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
args = parser.parse_args()
events = EventAccumulator(args.run_dir)
events.Reload()
tags = set(events.Tags()["scalars"])

required = {
    "Perception/b_raw", "Perception/b_actor", "Perception/H_raw", "Perception/H_actor",
    "Perception/D_raw", "Perception/D_actor", "Perception/first_riser_locked_fraction",
    "Perception/first_riser_consumed_fraction", "Perception/b_duration_per_encounter",
    "Perception/D_at_b_on", "Perception/D_at_b_raw_off", "Perception/D_at_b_actor_off",
    "Perception/teacher_success_while_b_active", "Perception/teacher_miss_while_b_active",
    "Perception/b_active_flat", "Perception/b_active_rough", "Perception/b_active_single_step",
    "Perception/b_active_full_stair", "Perception/false_trigger_flat", "Perception/false_trigger_rough",
    "Encounter/N_stair_encounter", "Encounter/N_first_step_success", "Encounter/N_perception_miss",
    "Encounter/N_full_stair_success", "Encounter/N_joint_success", "Encounter/first_step_success_rate",
    "Encounter/perception_miss_rate", "Encounter/full_stair_success_rate", "Encounter/joint_success_rate",
    "Encounter/blind_only_success_rate",
}
missing = sorted(required - tags)
latest = {tag: events.Scalars(tag)[-1].value for tag in required if tag in tags}
finite = all(math.isfinite(value) for value in latest.values())
flat_ok = latest.get("Perception/b_active_flat", 1.0) <= 1e-6 and latest.get("Perception/false_trigger_flat", 1.0) <= 1e-6
rough_ok = all(tag in latest for tag in ("Perception/b_active_rough", "Perception/false_trigger_rough"))
encounter_ok = latest.get("Encounter/N_stair_encounter", 0.0) > 0 and not missing and finite

print(f"MISSING_TAGS={missing}")
print(f"FLAT_B_ACTIVE={latest.get('Perception/b_active_flat', float('nan')):.8f}")
print(f"FLAT_FALSE_TRIGGER={latest.get('Perception/false_trigger_flat', float('nan')):.8f}")
print(f"ROUGH_B_ACTIVE={latest.get('Perception/b_active_rough', float('nan')):.8f}")
print(f"ROUGH_FALSE_TRIGGER={latest.get('Perception/false_trigger_rough', float('nan')):.8f}")
print(f"N_STAIR_ENCOUNTER={latest.get('Encounter/N_stair_encounter', 0.0):.0f}")
print(f"ROUGH_FALSE_GATE_AUDIT={'PASS' if flat_ok and rough_ok else 'FAIL'}")
print(f"ENCOUNTER_METRICS={'PASS' if encounter_ok else 'FAIL'}")
raise SystemExit(0 if flat_ok and rough_ok and encounter_ok else 1)
