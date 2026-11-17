#!/usr/bin/env python3
"""Export MuJoCo M20 wheel-position derivatives for Isaac alignment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from run_m20_v10_fr_mujoco import DEFAULT_Q, JOINTS, LEGS, prepare_model, quat_matrix


WHEELS = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    model_path = prepare_model(args.output_root, 0.14)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in WHEELS]
    qpos_indices = {}
    for name in JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_indices[name] = model.jnt_qposadr[joint_id]

    def set_pose(values: np.ndarray) -> np.ndarray:
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = (0.0, 0.0, 0.58)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        for index, name in enumerate(JOINTS):
            data.qpos[qpos_indices[name]] = values[index]
        mujoco.mj_forward(model, data)
        rotation = quat_matrix(data.xquat[base_id])
        return np.stack([rotation.T @ (data.xpos[body_id] - data.xpos[base_id]) for body_id in body_ids])

    baseline = set_pose(DEFAULT_Q)
    rows = []
    for joint_name in LEGS:
        values = DEFAULT_Q.copy()
        values[JOINTS.index(joint_name)] += 0.1
        derivative = (set_pose(values) - baseline) / 0.1
        row = {"joint": joint_name}
        for wheel_index, wheel_name in enumerate(WHEELS):
            for axis_index, axis_name in enumerate("xyz"):
                row[f"{wheel_name}_{axis_name}"] = float(derivative[wheel_index, axis_index])
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["joint"] + [f"{wheel}_{axis}" for wheel in WHEELS for axis in "xyz"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"MUJOCO_KINEMATIC_SIGNS_CSV={output}")


if __name__ == "__main__":
    main()
