#!/usr/bin/env python3
"""Interactive MuJoCo runner for the deployable 60D M20 V10 FR policy."""

from __future__ import annotations

import argparse
import csv
import pathlib
import select
import sys
import termios
import threading
import time
import tty
import zipfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort


LEGS = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
)
WHEELS = ("fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint")
JOINTS = LEGS + WHEELS
DEFAULT_Q = np.array(
    [0.0, -0.3, 0.6, 0.0, -0.3, 0.6, 0.0, 0.3, -0.6, 0.0, 0.3, -0.6,
     0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4 + [5.0] * 4, dtype=np.float64)
KP = np.array([80.0] * 12 + [0.0] * 4, dtype=np.float64)
KD = np.array([2.0] * 12 + [0.6] * 4, dtype=np.float64)
EFFORT = np.array([76.4] * 12 + [21.6] * 4, dtype=np.float64)
FIRST_RISER_X = 1.10
WHEEL_RADIUS = 0.09
ISAAC_SEED_ROOT_POS = np.array([0.10, 0.0, 0.58], dtype=np.float64)
ISAAC_SEED_ROOT_QUAT = np.array(
    [0.996057391166687, 0.08741617202758789, -0.015043945983052254, 0.0013203024864196777],
    dtype=np.float64,
)
ISAAC_SEED_JOINT_POS = np.array(
    [
        0.0015389707405120134, -0.00043283700942992054, -0.00017251968383791283,
        0.0015327270375564694, -0.000985044240951527, -0.00029000043869020775,
        0.0009767933515831828, -0.00037847161293030895, -0.00012645721435544655,
        0.0009201568318530917, -0.00039057135581971325, -0.0002614617347717063,
        0.0, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)
ISAAC_SEED_JOINT_VEL = np.array(
    [
        1.8437542021274567, 0.30995020642876625, -0.08901192806661129,
        1.8285946547985077, 0.23492082953453064, 0.2789134345948696,
        1.6792802512645721, 0.4431046172976494, -0.21751202642917633,
        1.4287951588630676, 0.37243228405714035, -0.06634377874433994,
        -2.1388936042785645, 0.5279945209622383, 0.04766632802784443,
        0.6881817430257797,
    ],
    dtype=np.float64,
)
ISAAC_SEED_BASE_ANG_VEL = np.array(
    [-0.004328412935137749, 0.04010326787829399, -0.006960929837077856],
    dtype=np.float64,
)
IMU_RAW_TO_BASE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    output = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(output, quaternion)
    return output.reshape(3, 3)


def yaw_from_quat(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def prepare_model(
    root: pathlib.Path, stair_height: float, align_usd_collision_shapes: bool = False
) -> pathlib.Path:
    """Prepare an isolated official M20 model with a six-step staircase."""
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".m20_assets_ready"
    if not marker.exists():
        archive = pathlib.Path("/home/ubuntu/M20_sdk_deploy.zip")
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                if not name.startswith("M20_sdk_deploy/M20_description/"):
                    continue
                relative = name.split("M20_sdk_deploy/", 1)[1]
                destination = root / relative
                if name.endswith("/"):
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(handle.read(name))
        marker.write_text("ready\n")

    model_dir = root / "M20_description/m20_mjcf/mjcf"
    if align_usd_collision_shapes:
        robot_xml = model_dir / "M20.xml"
        tree = ET.parse(robot_xml)
        changed = False
        for geom in tree.getroot().iter("geom"):
            if geom.get("class") == "collision" and geom.get("type") == "cylinder":
                geom.set("type", "capsule")
                changed = True
        if changed:
            tree.write(robot_xml, encoding="unicode")
    staircase = model_dir / "v10_user_stairs.xml"
    lines = [
        '<mujoco model="v10_user_stairs">',
        '  <worldbody>',
        '    <body name="v10_staircase" pos="0 0 0">',
    ]
    for index in range(6):
        height = (index + 1) * stair_height
        depth = 0.31 if index < 5 else 1.31
        start_x = 1.10 + index * 0.31
        center_x = start_x + depth / 2.0
        lines.append(
            '      <geom name="v10_stair_%d" type="box" pos="%.6f 0 %.6f" '
            'size="%.6f 0.5 %.6f" rgba="0.42 0.46 0.52 1"/>'
            % (index + 1, center_x, height / 2.0, depth / 2.0, height / 2.0)
        )
    lines.extend(('    </body>', '  </worldbody>', '</mujoco>'))
    staircase.write_text("\n".join(lines) + "\n")

    model_path = model_dir / "M20_v10_user_eval.xml"
    model_path.write_text(
        '<mujoco model="M20_v10_user_eval">\n'
        '  <include file="M20.xml"/>\n'
        '  <include file="scene.xml"/>\n'
        '  <include file="v10_user_stairs.xml"/>\n'
        '</mujoco>\n'
    )
    return model_path


class ElevationScanner:
    """The trained 21x15 yaw-aligned scanner: +1.25..-0.75 by +0.70..-0.70."""

    x_coordinates = np.linspace(1.25, -0.75, 21, dtype=np.float64)
    y_coordinates = np.linspace(0.70, -0.70, 15, dtype=np.float64)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, base_id: int):
        self.model = model
        self.data = data
        self.base_id = base_id
        self.geom_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.geom_id = np.array([-1], dtype=np.int32)

    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        base = self.data.xpos[self.base_id]
        yaw = yaw_from_quat(self.data.xquat[self.base_id])
        cosine, sine = np.cos(yaw), np.sin(yaw)
        points = np.full((21, 15, 3), np.nan, dtype=np.float64)
        start_z = max(float(base[2]) + 2.0, 2.5)
        for row, local_x in enumerate(self.x_coordinates):
            for column, local_y in enumerate(self.y_coordinates):
                world_x = base[0] + cosine * local_x - sine * local_y
                world_y = base[1] + sine * local_x + cosine * local_y
                origin = np.array([world_x, world_y, start_z], dtype=np.float64)
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    np.array([0.0, 0.0, -1.0]),
                    self.geom_group,
                    1,
                    self.base_id,
                    self.geom_id,
                )
                if distance >= 0.0:
                    points[row, column] = (local_x, local_y, start_z - distance)
        valid = np.isfinite(points[..., 2])
        return points, valid


class V10PGTT:
    """NumPy port of the deployable V10 first-riser lock and delay pipeline."""

    def __init__(self):
        self.delay_steps = 2
        self.feature_ring = np.zeros((self.delay_steps + 1, 3), dtype=np.float32)
        self.ring_index = 0
        self.reset()

    def reset(self) -> None:
        self.valid_count = 0
        self.invalid_count = 0
        self.no_stair_count = 0
        self.locked = False
        self.consumed = False
        self.visible = False
        self.locked_h = 0.0
        self.locked_d_raw = 0.0
        self.locked_d_filtered = 0.0
        self.pass_count = 0
        self.feature_ring.fill(0.0)
        self.ring_index = 0
        self.actor_feature = np.zeros(3, dtype=np.float32)

    @staticmethod
    def detect(points: np.ndarray, valid: np.ndarray) -> tuple[bool, float, float]:
        # Isaac converts semantic front-to-rear rows into increasing local x.
        points = points[::-1]
        valid = valid[::-1]
        lower = points[:-1]
        upper = points[1:]
        pair_valid = valid[:-1] & valid[1:]
        height = np.maximum(upper[..., 2] - lower[..., 2], 0.0)
        edge_x = 0.5 * (lower[..., 0] + upper[..., 0])
        edge_y = 0.5 * (lower[..., 1] + upper[..., 1])
        distance = edge_x
        corridor = np.abs(edge_y) <= 0.20
        candidates = pair_valid & corridor & (height >= 0.048) & (distance > -0.30)
        fallback = pair_valid & corridor & (height > 0.005) & (distance > -0.30)
        candidates |= fallback
        if not np.any(candidates):
            wide = np.abs(edge_y) <= 0.40
            candidates = pair_valid & wide & (height >= 0.048) & (distance > -0.30)
        if not np.any(candidates):
            return False, 0.0, 0.0
        score = np.where(candidates, np.abs(distance), np.inf)
        flat_index = int(np.argmin(score.reshape(-1)))
        detected_h = float(height.reshape(-1)[flat_index])
        detected_d = float(distance.reshape(-1)[flat_index])
        detected_valid = 0.048 <= detected_h <= 0.232
        return detected_valid, detected_h if detected_valid else 0.0, detected_d if detected_valid else 0.0

    def update(self, points: np.ndarray, valid: np.ndarray, command_x: float) -> dict[str, float | bool]:
        detected_valid, detected_h, detected_d = self.detect(points, valid)
        self.valid_count = self.valid_count + 1 if detected_valid else 0
        lock_event = self.valid_count >= 2 and not self.locked and not self.consumed
        if lock_event:
            self.locked = True
            self.locked_h = detected_h
            self.locked_d_raw = detected_d
            self.locked_d_filtered = detected_d

        height_match = abs(detected_h - self.locked_h) <= 0.04
        distance_continuity = detected_d <= self.locked_d_raw + 0.12
        track_visible = self.locked and detected_valid and height_match and distance_continuity
        self.invalid_count = 0 if track_visible else self.invalid_count + 1
        held_visible = track_visible or (self.locked and self.invalid_count <= 2)
        if track_visible and not lock_event:
            self.locked_d_raw = detected_d
            self.locked_d_filtered = 0.5 * detected_d + 0.5 * self.locked_d_filtered
            self.locked_h = 0.5 * detected_h + 0.5 * self.locked_h
        self.visible = held_visible and not self.consumed

        passed_signal = self.locked and (
            (track_visible and self.locked_d_filtered <= -0.15)
            or ((not held_visible) and self.locked_d_filtered <= -0.05)
        )
        self.pass_count = self.pass_count + 1 if passed_signal else 0
        if self.pass_count >= 2:
            self.consumed = True
            self.visible = False

        in_window = 0.02 <= self.locked_d_filtered <= 0.70
        b_raw = (
            self.locked and not self.consumed and self.visible and in_window and command_x > 0.05
        )
        h_raw = self.locked_h if b_raw else 0.0
        d_raw = self.locked_d_filtered if b_raw else 0.0
        feature_input = np.array([float(b_raw), h_raw, d_raw], dtype=np.float32)
        self.feature_ring[self.ring_index] = feature_input
        read_index = (self.ring_index - self.delay_steps) % self.feature_ring.shape[0]
        self.actor_feature = self.feature_ring[read_index].copy()
        self.actor_feature[1:] *= self.actor_feature[0]
        self.ring_index = (self.ring_index + 1) % self.feature_ring.shape[0]

        self.no_stair_count = 0 if detected_valid else self.no_stair_count + 1
        if self.consumed and self.no_stair_count >= 50:
            delayed_feature = self.actor_feature.copy()
            self.reset()
            self.actor_feature = delayed_feature

        return {
            "detected_valid": detected_valid,
            "detected_h": detected_h,
            "detected_d": detected_d,
            "locked": self.locked,
            "visible": self.visible,
            "consumed": self.consumed,
            "b_raw": b_raw,
            "b": float(self.actor_feature[0]),
            "h": float(self.actor_feature[1]),
            "d": float(self.actor_feature[2]),
        }


class Runner:
    def __init__(
        self,
        model: mujoco.MjModel,
        policy_path: pathlib.Path,
        delay_steps: int,
        mask_pgtt: bool = False,
        initial_observation: np.ndarray | None = None,
        first_b_observation: np.ndarray | None = None,
        replay_actions: list[np.ndarray] | None = None,
        physics_dt: float = 0.005,
        preserve_blind_support: bool = False,
        preserve_blind_wheels: bool = False,
        reset_default_joints: bool = False,
        isaac_seed_reset: bool = False,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.fr_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr_wheel")
        gyro_ids = np.flatnonzero(model.sensor_type == mujoco.mjtSensor.mjSENS_GYRO)
        if gyro_ids.size != 1:
            raise RuntimeError(f"Expected one MuJoCo gyro sensor, found {gyro_ids.size}")
        self.gyro_sensor_id = int(gyro_ids[0])
        self.gyro_sensor_adr = int(model.sensor_adr[self.gyro_sensor_id])
        self.qpos_indices = {}
        self.dof_indices = {}
        for name in JOINTS:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise RuntimeError(f"Missing M20 joint: {name}")
            self.qpos_indices[name] = model.jnt_qposadr[joint_id]
            self.dof_indices[name] = model.jnt_dofadr[joint_id]
        for name in WHEELS:
            model.dof_armature[self.dof_indices[name]] = 0.00243216

        self.policy = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        policy_input = self.policy.get_inputs()[0]
        policy_output = self.policy.get_outputs()[0]
        if policy_input.shape[-1] != 60 or policy_output.shape[-1] != 16:
            raise RuntimeError(f"Expected V10 policy 60->16, got {policy_input.shape}->{policy_output.shape}")
        self.policy_input = policy_input.name
        self.policy_output = policy_output.name
        self.scanner = ElevationScanner(model, self.data, self.base_id)
        self.pgtt = V10PGTT()
        self.command = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(16, dtype=np.float32)
        self.normal_previous_action = np.zeros(16, dtype=np.float32)
        self.blind_previous_action = np.zeros(16, dtype=np.float32)
        self.policy_action = np.zeros(16, dtype=np.float32)
        self.applied_action = np.zeros(16, dtype=np.float32)
        self.action_queue = [np.zeros(16, dtype=np.float32) for _ in range(delay_steps)]
        self.delay_steps = delay_steps
        self.mask_pgtt = mask_pgtt
        self.initial_observation = initial_observation
        self.first_b_observation = first_b_observation
        self.first_b_observation_used = False
        self.replay_actions = replay_actions
        self.replay_index = 0
        self.preserve_blind_support = preserve_blind_support
        self.preserve_blind_wheels = preserve_blind_wheels
        self.reset_joint_values = DEFAULT_Q.copy() if reset_default_joints else np.zeros(16)
        self.isaac_seed_reset = isaac_seed_reset
        self.mode = "STAND"
        self.reset_requested = False
        self.stop_requested = False
        self.control_dt = 0.02
        self.physics_dt = physics_dt
        self.next_control_time = 0.0
        self.last_pgtt: dict[str, float | bool] = {}
        self.last_observation = np.zeros(60, dtype=np.float32)
        self.reset()

    def fr_base_z(self) -> float:
        rotation = quat_matrix(self.data.xquat[self.base_id])
        relative = rotation.T @ (self.data.xpos[self.fr_body_id] - self.data.xpos[self.base_id])
        return float(relative[2])

    def fr_riser_gap(self) -> float:
        """Horizontal clearance from the FR tire front edge to the first riser."""
        return float(FIRST_RISER_X - (self.data.xpos[self.fr_body_id, 0] + WHEEL_RADIUS))

    def fr_wheel_bottom_z(self) -> float:
        return float(self.data.xpos[self.fr_body_id, 2] - WHEEL_RADIUS)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = ISAAC_SEED_ROOT_POS if self.isaac_seed_reset else (0.0, 0.0, 0.58)
        self.data.qpos[3:7] = ISAAC_SEED_ROOT_QUAT if self.isaac_seed_reset else (1.0, 0.0, 0.0, 0.0)
        reset_joint_values = ISAAC_SEED_JOINT_POS if self.isaac_seed_reset else self.reset_joint_values
        for index, name in enumerate(JOINTS):
            self.data.qpos[self.qpos_indices[name]] = reset_joint_values[index]
        self.data.qvel[:] = 0.0
        if self.isaac_seed_reset:
            self.data.qvel[3:6] = ISAAC_SEED_BASE_ANG_VEL
            for index, name in enumerate(JOINTS):
                self.data.qvel[self.dof_indices[name]] = ISAAC_SEED_JOINT_VEL[index]
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.command.fill(0.0)
        self.previous_action.fill(0.0)
        self.normal_previous_action.fill(0.0)
        self.blind_previous_action.fill(0.0)
        self.policy_action.fill(0.0)
        self.applied_action.fill(0.0)
        self.action_queue = [np.zeros(16, dtype=np.float32) for _ in range(self.delay_steps)]
        self.pgtt.reset()
        self.mode = "STAND"
        self.next_control_time = 0.0
        self.reset_requested = False
        print("RESET_COMPLETE mode=STAND pose=official_M20_default", flush=True)

    def observation(self) -> np.ndarray:
        rotation = quat_matrix(self.data.xquat[self.base_id])
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        gyro_raw = self.data.sensordata[self.gyro_sensor_adr : self.gyro_sensor_adr + 3]
        angular_velocity = (IMU_RAW_TO_BASE @ gyro_raw) * 0.25
        joint_position = np.array(
            [self.data.qpos[self.qpos_indices[name]] for name in JOINTS], dtype=np.float32
        )
        joint_velocity = np.array(
            [self.data.qvel[self.dof_indices[name]] for name in JOINTS], dtype=np.float32
        )
        relative_position = joint_position - DEFAULT_Q.astype(np.float32)
        relative_position[12:] = 0.0
        scaled_velocity = joint_velocity * 0.05
        return np.concatenate((
            angular_velocity.astype(np.float32),
            projected_gravity.astype(np.float32),
            self.command,
            relative_position,
            scaled_velocity,
            self.previous_action,
            self.pgtt.actor_feature,
        )).astype(np.float32)

    def control_step(self) -> None:
        points, valid = self.scanner.scan()
        self.last_pgtt = self.pgtt.update(points, valid, float(self.command[0]))
        if self.mask_pgtt:
            self.pgtt.actor_feature.fill(0.0)
            self.last_pgtt["b"] = 0.0
            self.last_pgtt["h"] = 0.0
            self.last_pgtt["d"] = 0.0
        if self.mode != "RL":
            self.policy_action.fill(0.0)
            self.applied_action.fill(0.0)
            self.previous_action.fill(0.0)
            return
        observation = self.observation()
        if self.initial_observation is not None:
            observation = self.initial_observation.copy()
            self.initial_observation = None
        elif (
            self.first_b_observation is not None
            and not self.first_b_observation_used
            and observation[57] > 0.5
        ):
            observation = self.first_b_observation.copy()
            self.first_b_observation_used = True
        self.last_observation = observation.copy()
        if observation.shape != (60,):
            raise RuntimeError(f"V10 observation mismatch: {observation.shape}")
        if self.replay_actions is not None and self.replay_index < len(self.replay_actions):
            action = self.replay_actions[self.replay_index].copy()
            self.replay_index += 1
        else:
            if (self.preserve_blind_support or self.preserve_blind_wheels) and observation[57] > 0.5:
                normal_observation = observation.copy()
                normal_observation[41:57] = self.normal_previous_action
                normal_action = self.policy.run(
                    [self.policy_output], {self.policy_input: normal_observation[None, :]}
                )[0][0].astype(np.float32)
                blind_observation = observation.copy()
                blind_observation[41:57] = self.blind_previous_action
                blind_observation[57:60] = 0.0
                blind_action = self.policy.run(
                    [self.policy_output], {self.policy_input: blind_observation[None, :]}
                )[0][0].astype(np.float32)
                self.normal_previous_action = normal_action.copy()
                self.blind_previous_action = blind_action.copy()
                if self.preserve_blind_wheels:
                    normal_action[12:16] = blind_action[12:16]
                    action = normal_action
                else:
                    blind_action[3:6] = normal_action[3:6]
                    action = blind_action
            else:
                action = self.policy.run(
                    [self.policy_output], {self.policy_input: observation[None, :]}
                )[0][0].astype(np.float32)
                self.normal_previous_action = action.copy()
                self.blind_previous_action = action.copy()
        action = np.clip(action, -100.0, 100.0)
        self.policy_action = action
        if self.delay_steps:
            self.action_queue.append(action.copy())
            self.applied_action = self.action_queue.pop(0)
        else:
            self.applied_action = action.copy()
        self.previous_action = action.copy()

    def physics_step(self) -> None:
        joint_position = np.array(
            [self.data.qpos[self.qpos_indices[name]] for name in JOINTS], dtype=np.float64
        )
        joint_velocity = np.array(
            [self.data.qvel[self.dof_indices[name]] for name in JOINTS], dtype=np.float64
        )
        torque = np.zeros(16, dtype=np.float64)
        if self.mode == "DAMPING":
            torque = -KD * joint_velocity
        else:
            action = self.applied_action if self.mode == "RL" else np.zeros(16)
            scaled = np.clip(action.astype(np.float64) * ACTION_SCALE, -100.0, 100.0)
            position_target = DEFAULT_Q.copy()
            position_target[:12] += scaled[:12]
            velocity_target = scaled[12:]
            torque[:12] = KP[:12] * (position_target[:12] - joint_position[:12]) - KD[:12] * joint_velocity[:12]
            torque[12:] = KD[12:] * (velocity_target - joint_velocity[12:])
        torque = np.clip(torque, -EFFORT, EFFORT)
        self.data.ctrl[:] = 0.0
        for actuator_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name in JOINTS:
                self.data.ctrl[actuator_id] = torque[JOINTS.index(name)]
        mujoco.mj_step(self.model, self.data)

    def key_callback(self, keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "b" or keycode == 259:
            self.reset_requested = True
        elif key == "z":
            self.mode = "STAND"
            self.command.fill(0.0)
        elif key == "c":
            self.mode = "RL"
        elif key == "r":
            self.mode = "DAMPING"
            self.command.fill(0.0)
        elif key == "w" and self.mode == "RL":
            self.command[0] = min(float(self.command[0]) + 0.1, 0.6)
        elif key == "s" and self.mode == "RL":
            self.command[0] = max(float(self.command[0]) - 0.1, -0.5)
        elif key == "a" and self.mode == "RL":
            self.command[1] = min(float(self.command[1]) + 0.1, 0.5)
        elif key == "d" and self.mode == "RL":
            self.command[1] = max(float(self.command[1]) - 0.1, -0.5)
        elif key == "q" and self.mode == "RL":
            self.command[2] = min(float(self.command[2]) + 0.1, 0.7)
        elif key == "e" and self.mode == "RL":
            self.command[2] = max(float(self.command[2]) - 0.1, -0.7)
        elif key == " " and self.mode == "RL":
            self.command.fill(0.0)
        elif key == "x" or keycode == 256:
            self.stop_requested = True
        else:
            return
        print(
            "MODE=%s COMMAND vx=%+.2f vy=%+.2f wz=%+.2f"
            % (self.mode, self.command[0], self.command[1], self.command[2]),
            flush=True,
        )


class TerminalKeyboard:
    def __init__(self, runner: Runner):
        self.runner = runner
        self.file_descriptor = sys.stdin.fileno()
        self.old_settings = None
        self.thread = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("--terminal-control requires a foreground terminal")
        self.old_settings = termios.tcgetattr(self.file_descriptor)
        tty.setcbreak(self.file_descriptor)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.runner.stop_requested:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                key = sys.stdin.read(1)
                self.runner.key_callback(256 if key == "\x1b" else ord(key))

    def close(self) -> None:
        if self.old_settings is not None:
            termios.tcsetattr(self.file_descriptor, termios.TCSADRAIN, self.old_settings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=pathlib.Path, required=True)
    parser.add_argument("--height", type=float, default=0.14)
    parser.add_argument("--delay-steps", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seconds", type=float, default=3600.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--terminal-control", action="store_true")
    parser.add_argument("--auto-command-x", type=float, default=0.0)
    parser.add_argument("--mask-pgtt", action="store_true")
    parser.add_argument("--observation-csv", type=pathlib.Path)
    parser.add_argument("--initial-observation-csv", type=pathlib.Path)
    parser.add_argument("--first-b-observation-csv", type=pathlib.Path)
    parser.add_argument("--action-replay-csv", type=pathlib.Path)
    parser.add_argument("--action-replay-limit", type=int, default=None)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--preserve-blind-support", action="store_true")
    parser.add_argument("--preserve-blind-wheels", action="store_true")
    parser.add_argument("--align-usd-collision-shapes", action="store_true")
    parser.add_argument(
        "--reset-default-joints",
        action="store_true",
        help="Use the old crouched reset. By default, match Isaac's observed zero-radian reset state.",
    )
    parser.add_argument(
        "--isaac-seed-reset",
        action="store_true",
        help="Reset from the physical state captured at Isaac fixed-14 evaluation step zero.",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/ubuntu/m20_perceptive_final/eval_outputs/mujoco_v10_fr_model23000"),
    )
    args = parser.parse_args()
    model_path = prepare_model(
        args.output_root, args.height, align_usd_collision_shapes=args.align_usd_collision_shapes
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = args.physics_dt
    initial_observation = None
    if args.initial_observation_csv is not None:
        with args.initial_observation_csv.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["frame"] == "INITIAL_COMMAND_FRAME":
                    initial_observation = np.array(
                        [float(row[f"obs_{i}"]) for i in range(60)], dtype=np.float32
                    )
                    break
        if initial_observation is None:
            raise RuntimeError("INITIAL_COMMAND_FRAME missing from initial observation CSV")
    first_b_observation = None
    if args.first_b_observation_csv is not None:
        with args.first_b_observation_csv.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["frame"] == "FIRST_B_ON_FRAME":
                    first_b_observation = np.array(
                        [float(row[f"obs_{i}"]) for i in range(60)], dtype=np.float32
                    )
                    break
        if first_b_observation is None:
            raise RuntimeError("FIRST_B_ON_FRAME missing from first-b observation CSV")
    replay_actions = None
    if args.action_replay_csv is not None:
        with args.action_replay_csv.open(newline="") as stream:
            replay_actions = [
                np.array([float(row[f"action_{i}"]) for i in range(16)], dtype=np.float32)
                for row in csv.DictReader(stream)
            ]
        if args.action_replay_limit is not None:
            replay_actions = replay_actions[: args.action_replay_limit]
    runner = Runner(
        model,
        args.policy,
        args.delay_steps,
        mask_pgtt=args.mask_pgtt,
        initial_observation=initial_observation,
        first_b_observation=first_b_observation,
        replay_actions=replay_actions,
        physics_dt=args.physics_dt,
        preserve_blind_support=args.preserve_blind_support,
        preserve_blind_wheels=args.preserve_blind_wheels,
        reset_default_joints=args.reset_default_joints,
        isaac_seed_reset=args.isaac_seed_reset,
    )
    if args.auto_command_x != 0.0:
        runner.mode = "RL"
        runner.command[0] = args.auto_command_x
    viewer = None
    keyboard = None
    if not args.headless:
        from mujoco import viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(model, runner.data, key_callback=runner.key_callback)
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -22.0
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = (0.9, -0.15, 0.35)
    if args.terminal_control:
        keyboard = TerminalKeyboard(runner)
        keyboard.start()
    print("MUJOCO_POLICY_LOAD=PASS OBS_60_ALIGNMENT=PASS ACTION_16_ALIGNMENT=PASS", flush=True)
    print("CONTROL: Z=stand C=RL W/S=vx A/D=vy Q/E=yaw Space=stop B=reset R=damping X/Esc=exit", flush=True)
    print(
        "RESET joint state="
        + (
            "isaac_fixed14_seed"
            if args.isaac_seed_reset
            else ("configured_default" if args.reset_default_joints else "isaac_observed_zero")
        ),
        flush=True,
    )

    total_steps = int(args.seconds / runner.physics_dt)
    control_every = int(round(runner.control_dt / runner.physics_dt))
    last_status = -1
    b_control_steps = 0
    fr_reference = None
    max_active_fr_lift = 0.0
    max_precontact_fr_lift = 0.0
    min_fr_riser_gap = float("inf")
    fr_contact_or_passed = False
    precontact_gap_thresholds = (0.20, 0.10, 0.05, 0.02, 0.01)
    max_fr_lift_by_gap = {threshold: 0.0 for threshold in precontact_gap_thresholds}
    max_fr_bottom_z_by_gap = {threshold: -float("inf") for threshold in precontact_gap_thresholds}
    max_fr_action_abs = np.zeros(3, dtype=np.float64)
    observation_frames = {}
    best_d_error = float("inf")
    previous_frame = None
    first_b_seen = False
    try:
        for step in range(total_steps):
            tick_start = time.perf_counter()
            if runner.reset_requested:
                runner.reset()
            if runner.stop_requested or (viewer is not None and not viewer.is_running()):
                break
            if step % control_every == 0:
                runner.control_step()
                if "INITIAL_COMMAND_FRAME" not in observation_frames:
                    observation_frames["INITIAL_COMMAND_FRAME"] = (
                        step,
                        runner.last_observation.copy(),
                        runner.policy_action.copy(),
                    )
                actor_b = float(runner.last_observation[57])
                actor_d = float(runner.last_observation[59])
                current_frame = (
                    step,
                    runner.last_observation.copy(),
                    runner.policy_action.copy(),
                )
                if actor_b > 0.5 and not first_b_seen:
                    if previous_frame is not None:
                        observation_frames["PRE_B_ON_FRAME"] = previous_frame
                    observation_frames["FIRST_B_ON_FRAME"] = current_frame
                    first_b_seen = True
                previous_frame = current_frame
                d_error = abs(actor_d - 0.40)
                if actor_b > 0.5 and d_error < best_d_error:
                    best_d_error = d_error
                    observation_frames["B_D040_FRAME"] = (
                        step,
                        runner.last_observation.copy(),
                        runner.policy_action.copy(),
                    )
                if float(runner.last_pgtt.get("b", 0.0)) > 0.5:
                    b_control_steps += 1
                    current_fr_z = runner.fr_base_z()
                    current_fr_gap = runner.fr_riser_gap()
                    current_fr_bottom_z = runner.fr_wheel_bottom_z()
                    if fr_reference is None:
                        fr_reference = current_fr_z
                    current_fr_lift = current_fr_z - fr_reference
                    max_active_fr_lift = max(max_active_fr_lift, current_fr_lift)
                    min_fr_riser_gap = min(min_fr_riser_gap, current_fr_gap)
                    if current_fr_gap > 0.0:
                        max_precontact_fr_lift = max(max_precontact_fr_lift, current_fr_lift)
                        for threshold in precontact_gap_thresholds:
                            if current_fr_gap >= threshold:
                                max_fr_lift_by_gap[threshold] = max(
                                    max_fr_lift_by_gap[threshold], current_fr_lift
                                )
                                max_fr_bottom_z_by_gap[threshold] = max(
                                    max_fr_bottom_z_by_gap[threshold], current_fr_bottom_z
                                )
                    else:
                        fr_contact_or_passed = True
                    max_fr_action_abs = np.maximum(
                        max_fr_action_abs, np.abs(runner.policy_action[3:6])
                    )
            runner.physics_step()
            if viewer is not None:
                viewer.sync()
                remaining = runner.physics_dt - (time.perf_counter() - tick_start)
                if remaining > 0.0:
                    time.sleep(remaining)
            elapsed_second = int(runner.data.time)
            if elapsed_second != last_status:
                last_status = elapsed_second
                state = runner.last_pgtt
                if viewer is not None and hasattr(viewer, "set_texts"):
                    viewer.set_texts((
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        "V10 FR PGTT model_29999",
                        "b=%.0f H=%.3f D=%.3f\nFR action=[%+.2f %+.2f %+.2f]\n"
                        "FR precontact lift=%.1f cm gap=%.1f cm"
                        % (
                            state.get("b", 0.0),
                            state.get("h", 0.0),
                            state.get("d", 0.0),
                            runner.policy_action[3],
                            runner.policy_action[4],
                            runner.policy_action[5],
                            100.0 * max_precontact_fr_lift,
                            100.0 * runner.fr_riser_gap(),
                        ),
                    ))
                print(
                    "t=%.1f mode=%s x=%.3f vx=%.3f b=%.0f H=%.3f D=%.3f valid=%s "
                    "locked=%s FR_precontact_lift=%.3f FR_gap=%.3f"
                    % (
                        runner.data.time,
                        runner.mode,
                        runner.data.qpos[0],
                        runner.data.qvel[0],
                        state.get("b", 0.0),
                        state.get("h", 0.0),
                        state.get("d", 0.0),
                        state.get("detected_valid", False),
                        state.get("locked", False),
                        max_precontact_fr_lift,
                        runner.fr_riser_gap(),
                    ),
                    flush=True,
                )
    finally:
        if args.observation_csv is not None:
            args.observation_csv.parent.mkdir(parents=True, exist_ok=True)
            fields = ["frame", "step"] + [f"obs_{i}" for i in range(60)] + [
                f"action_{i}" for i in range(16)
            ]
            with args.observation_csv.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for frame_name, (frame_step, observation, action) in observation_frames.items():
                    row = {"frame": frame_name, "step": frame_step}
                    row.update({f"obs_{i}": value for i, value in enumerate(observation)})
                    row.update({f"action_{i}": value for i, value in enumerate(action)})
                    writer.writerow(row)
            print(f"MUJOCO_OBSERVATION_CSV={args.observation_csv}", flush=True)
        print(
            "TRACE_SUMMARY mask_pgtt=%s final_x=%.6f b_steps=%d max_FR_action_abs=%s "
            "max_active_FR_lift=%.6f max_precontact_FR_lift=%.6f "
            "min_FR_riser_gap=%.6f FR_contact_or_passed=%s"
            % (
                args.mask_pgtt,
                runner.data.qpos[0],
                b_control_steps,
                np.array2string(max_fr_action_abs, precision=5),
                max_active_fr_lift,
                max_precontact_fr_lift,
                min_fr_riser_gap,
                fr_contact_or_passed,
            ),
            flush=True,
        )
        print(
            "PRECONTACT_LIFT_BY_GAP "
            + " ".join(
                "gap_ge_%02dcm=%.6f" % (round(threshold * 100.0), max_fr_lift_by_gap[threshold])
                for threshold in precontact_gap_thresholds
            ),
            flush=True,
        )
        print(
            "PRECONTACT_WHEEL_BOTTOM_Z_BY_GAP "
            + " ".join(
                "gap_ge_%02dcm=%.6f" % (round(threshold * 100.0), max_fr_bottom_z_by_gap[threshold])
                for threshold in precontact_gap_thresholds
            ),
            flush=True,
        )
        if keyboard is not None:
            keyboard.close()
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
