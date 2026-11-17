#!/usr/bin/env python3
"""Runtime-resolved M20 V10 action, wheel body, and contact mapping audit."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch

import rl_training.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply_inverse


TASK = "M20-Stair-Unified-Perceptive-v0"
LEGS = {
    "FL": ("fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint"),
    "FR": ("fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint"),
    "HL": ("hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint"),
    "HR": ("hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint"),
}
WHEELS = {
    "FL": "fl_wheel",
    "FR": "fr_wheel",
    "HL": "hl_wheel",
    "HR": "hr_wheel",
}


def main():
    cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.gait_level = None
    env = gym.make(TASK, cfg=cfg)
    raw = env.unwrapped
    env.reset()

    robot = raw.scene["robot"]
    contact = raw.scene.sensors["contact_forces"]
    action_mapping = []
    offset = 0
    for term_name in raw.action_manager.active_terms:
        term = raw.action_manager._terms[term_name]
        for local_index, (joint_name, joint_id) in enumerate(zip(term._joint_names, term._joint_ids)):
            action_mapping.append((offset + local_index, term_name, joint_name, int(joint_id)))
        offset += term.action_dim

    index_by_joint = {joint_name: index for index, _, joint_name, _ in action_mapping}
    print("ACTION_INDEX_MAPPING")
    for index, term_name, joint_name, joint_id in action_mapping:
        print(f"action[{index}]={joint_name} term={term_name} resolved_joint_id={joint_id}")

    all_leg_ok = True
    for leg, names in LEGS.items():
        indices = [index_by_joint[name] for name in names]
        expected = list(range(indices[0], indices[0] + 3))
        all_leg_ok &= indices == expected
        print(f"{leg}_LEG_ACTION_INDICES={indices}")

    body_ids = {}
    contact_ids = {}
    for leg, body_name in WHEELS.items():
        ids, names = robot.find_bodies(body_name)
        sensor_ids, sensor_names = contact.find_bodies(body_name)
        if len(ids) != 1 or names != [body_name]:
            raise RuntimeError(f"{leg} robot body resolution failed: ids={ids} names={names}")
        if len(sensor_ids) != 1 or sensor_names != [body_name]:
            raise RuntimeError(
                f"{leg} contact body resolution failed: ids={sensor_ids} names={sensor_names}"
            )
        body_ids[leg] = int(ids[0])
        contact_ids[leg] = int(sensor_ids[0])
        print(f"{leg}_WHEEL_BODY_ID={body_ids[leg]}")
        print(f"{leg}_CONTACT_ID={contact_ids[leg]}")

    body_pos = robot.data.body_pos_w[0]
    base_pos = robot.data.root_pos_w[0:1]
    base_quat = robot.data.root_quat_w[0:1]
    local_x = {}
    for leg in WHEELS:
        relative = body_pos[body_ids[leg] : body_ids[leg] + 1] - base_pos
        local = quat_apply_inverse(base_quat, relative)
        local_x[leg] = float(local[0, 0].item())
        print(f"{leg}_WHEEL_BASE_X={local_x[leg]:.9f}")

    front_ok = min(local_x["FL"], local_x["FR"]) > max(local_x["HL"], local_x["HR"])
    fr_front = local_x["FR"] > local_x["HR"]
    hr_rear = local_x["HR"] < local_x["FR"]
    riser_x = max(local_x.values()) + 1.0
    fr_distance = riser_x - local_x["FR"]
    hr_distance = riser_x - local_x["HR"]
    fr_reaches_first = fr_distance < hr_distance
    print(f"FR_DISTANCE_TO_FORWARD_RISER={fr_distance:.9f}")
    print(f"HR_DISTANCE_TO_FORWARD_RISER={hr_distance:.9f}")
    print(f"FR_IS_FRONT_WHEEL={'PASS' if fr_front and front_ok else 'FAIL'}")
    print(f"HR_IS_REAR_WHEEL={'PASS' if hr_rear and front_ok else 'FAIL'}")
    print(f"FR_REACHES_FORWARD_RISER_BEFORE_HR={'PASS' if fr_reaches_first else 'FAIL'}")

    expected_indices = {
        "FL": [0, 1, 2], "FR": [3, 4, 5], "HL": [6, 7, 8], "HR": [9, 10, 11]
    }
    mapping_ok = all([index_by_joint[name] for name in LEGS[leg]] == value for leg, value in expected_indices.items())
    print(f"ACTION_ORDER_FL_FR_HL_HR={'PASS' if mapping_ok and all_leg_ok else 'FAIL'}", flush=True)

    env.close()
    simulation_app.close()
    passed = mapping_ok and all_leg_ok and front_ok and fr_reaches_first
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
