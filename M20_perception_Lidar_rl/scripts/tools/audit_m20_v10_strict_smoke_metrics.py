#!/usr/bin/env python3
"""Audit strict V10 TensorBoard schema while preserving official logger output."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--definition", default="eval_outputs/v10_metrics_definition.md")
args = parser.parse_args()

events = EventAccumulator(args.run_dir)
events.Reload()
tags = set(events.Tags()["scalars"])

official_required = {
    "Train/mean_reward", "Train/mean_episode_length", "Loss/value_function", "Loss/surrogate",
    "Loss/entropy", "Loss/learning_rate", "Perf/total_fps", "Policy/mean_noise_std",
    "Curriculum/terrain_levels", "Curriculum/gait_level",
    "Episode_Termination/time_out", "Episode_Termination/bad_orientation_2",
    "Episode_Reward/action_rate_l2", "Episode_Reward/action_smooth_l2", "Episode_Reward/ang_vel_xy_l2",
    "Episode_Reward/bad_orientation_penalty", "Episode_Reward/contact_forces",
    "Episode_Reward/feet_air_time_ang_z_M20", "Episode_Reward/feet_slide_ang_z_cmd",
    "Episode_Reward/first_step_success", "Episode_Reward/flat_orientation_l2",
    "Episode_Reward/hipx_joint_pos_penalty", "Episode_Reward/hipy_joint_pos_penalty",
    "Episode_Reward/joint_acc_l2", "Episode_Reward/joint_acc_wheel_l2", "Episode_Reward/joint_mirror",
    "Episode_Reward/joint_success", "Episode_Reward/joint_torques_l2",
    "Episode_Reward/knee_joint_pos_penalty", "Episode_Reward/lin_vel_z_l2",
    "Episode_Reward/perception_miss", "Episode_Reward/rotation_gait_status",
    "Episode_Reward/rotation_gait_symmetry", "Episode_Reward/stand_still",
    "Episode_Reward/track_ang_vel_z_exp", "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/undesired_contacts",
}

required = {
    "V10_Perception/Flat/FalseTriggerRate", "V10_Perception/Rough/FalseTriggerRate",
    "V10_Perception/Flat/BActiveFraction", "V10_Perception/Rough/BActiveFraction",
    "V10_EventDensity/FirstStep", "V10_EventDensity/Miss", "V10_EventDensity/SingleStepComplete",
    "V10_EventDensity/FullStairSuccess", "V10_EventDensity/Joint",
}

scope_fields = {
    "AllStair": ("FirstStepSuccessRate", "PerceptionMissRate"),
    "Single": ("FirstStepSuccessRate", "PerceptionMissRate", "SingleStepCompleteRate"),
    "Full": (
        "FirstStepSuccessRate", "PerceptionMissRate", "FullStairSuccessRate",
        "JointSuccessRate", "BlindOnlySuccessRate",
    ),
}
buckets = ("05_08", "08_11", "11_14", "14_17", "17_20", "20_23")
for window in ("Lifetime", "Recent100", "Recent500"):
    for scope, fields in scope_fields.items():
        prefix = f"V10_Task/{window}/{scope}"
        required.add(f"{prefix}/NEncounter")
        required.update(f"{prefix}/{field}" for field in fields)
        if scope == "AllStair":
            continue
        for bucket in buckets:
            bucket_prefix = f"{prefix}/{bucket}"
            required.add(f"{bucket_prefix}/NEncounter")
            required.update(f"{bucket_prefix}/{field}" for field in fields)

missing_v10 = sorted(required - tags)
missing_official = sorted(official_required - tags)
latest = {tag: events.Scalars(tag)[-1].value for tag in required if tag in tags}
finite = all(math.isfinite(value) for value in latest.values())

forbidden_ambiguous = {
    "Perception/first_step_success", "Perception/perception_miss",
    "Task/full_stair_success", "Task/joint_success",
    "Encounter/full_stair_success_rate", "Encounter/joint_success_rate",
}
ambiguous_present = sorted(forbidden_ambiguous & tags)
definition_ok = Path(args.definition).is_file()
terrain_level = events.Scalars("Curriculum/terrain_levels")[-1].value if "Curriculum/terrain_levels" in tags else float("nan")
gait_level = events.Scalars("Curriculum/gait_level")[-1].value if "Curriculum/gait_level" in tags else float("nan")
expected_gait = 0.0 if terrain_level <= 0.0 else (math.exp(terrain_level - 3.0) if terrain_level < 3.0 else 1.0)
gait_link_ok = math.isfinite(gait_level) and abs(gait_level - expected_gait) <= 1.0e-5

print(f"MISSING_V10_TAGS={missing_v10}")
print(f"MISSING_OFFICIAL_TAGS={missing_official}")
print(f"AMBIGUOUS_OLD_TAGS={ambiguous_present}")
print(f"FINITE_METRICS={finite}")
print(f"METRIC_DEFINITION_DOC={'PASS' if definition_ok else 'FAIL'}")
print(f"RECENT100_METRICS={'PASS' if any('/Recent100/' in tag for tag in tags) else 'FAIL'}")
print(f"RECENT500_METRICS={'PASS' if any('/Recent500/' in tag for tag in tags) else 'FAIL'}")
print(f"OFFICIAL_METRICS_UNCHANGED={'PASS' if not missing_official else 'FAIL'}")
print(f"TERRAIN_LEVEL_LATEST={terrain_level:.8f}")
print(f"GAIT_LEVEL_LATEST={gait_level:.8f}")
print(f"EXPECTED_GAIT_LEVEL={expected_gait:.8f}")
print(f"OFFICIAL_TERRAIN_GAIT_LINK={'PASS' if gait_link_ok else 'FAIL'}")
passed = not missing_v10 and not missing_official and not ambiguous_present and finite and definition_ok and gait_link_ok
raise SystemExit(0 if passed else 1)
