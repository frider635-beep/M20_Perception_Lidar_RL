#!/usr/bin/env python3
"""Standalone MuJoCo viewer for the 372-D M20 perceptive policy.

This is an evaluation/deployment-side file. It does not import or modify the
IsaacLab training environment. The actor input is assembled as:
3 base angular velocity + 3 projected gravity + 3 command + 16 relative joint
position + 16 scaled relative joint velocity + 16 previous action + 315
21x15 elevation.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import select
import sys
import termios
import threading
import time
import tty
import zipfile

import mujoco
import numpy as np
import onnxruntime as ort


WHEELS = ["fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint"]
LEGS = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
]
JOINTS = LEGS + WHEELS
ROBOT_ORDER = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
]
# The vendor MJCF joint coordinates already match the policy coordinates.
# Encoder direction/offset calibration belongs at the ROS/hardware boundary;
# applying it again inside MuJoCo folds the legs through their joint limits.
JOINT_DIR = np.ones(16, np.float32)
POS_OFFSET = np.zeros(16, np.float32)
DEFAULT = np.array([0, -.6, 1, 0, -.6, 1, 0, .6, -1, 0, .6, -1, 0, 0, 0, 0], np.float32)
LIEDOWN_ROBOT = np.array([
    -.438, -1.16, 2.76, 0., .438, -1.16, 2.76, 0.,
    -.438, 1.16, -2.76, 0., .438, 1.16, -2.76, 0.,
], np.float32)
LIEDOWN = np.array([LIEDOWN_ROBOT[ROBOT_ORDER.index(name)] for name in JOINTS], np.float32)
LIE_TARGET = np.array([
    0.0, -1.51076, 2.76,
    0.0, -1.51076, 2.76,
    0.0, 1.51076, -2.76,
    0.0, 1.51076, -2.76,
    0.0, 0.0, 0.0, 0.0,
], np.float32)
SCALE = np.array([.125, .25, .25] * 4 + [5., 5., 5., 5.], np.float32)
KP = np.array([80.] * 12 + [0.] * 4, np.float32)
KD = np.array([2.] * 12 + [.6] * 4, np.float32)
EFFORT = np.array([76.4] * 12 + [21.6] * 4, np.float32)


def quat_matrix(q):
    out = np.zeros((9,), dtype=np.float64)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def prepare_xml(root: pathlib.Path, height: float) -> pathlib.Path:
    """Extract official M20 assets and make an evaluation-only stair include."""
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".assets_ready"
    if not marker.exists():
        archive = pathlib.Path("/home/ubuntu/M20_sdk_deploy.zip")
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.startswith("M20_sdk_deploy/M20_description/"):
                    rel = name.split("M20_sdk_deploy/", 1)[1]
                    target = root / rel
                    if name.endswith("/"):
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(zf.read(name))
        marker.write_text("ready\n")
    stair = root / "M20_description/m20_mjcf/mjcf/stair_user_eval.xml"
    lines = [
        '<mujoco model="stair_user_eval">',
        '  <option gravity="0 0 -9.81"/>',
        '  <worldbody>',
        '    <body name="staircase" pos="0 0 0">',
    ]
    for i in range(6):
        z = (i + 1) * height
        depth = .31 if i < 5 else 1.11
        x = 1.10 + i * .31 + depth / 2
        lines.append(f'      <geom name="stair{i+1}" type="box" pos="{x:.6f} 0 {z/2:.6f}" size="{depth/2:.6f} 0.5 {z/2:.6f}"/>')
    lines += ['    </body>', '  </worldbody>', '</mujoco>']
    stair.write_text("\n".join(lines) + "\n")
    # Keep the top-level XML beside M20.xml so its ../meshes compiler path
    # resolves exactly as in the vendor package.
    model = root / "M20_description/m20_mjcf/mjcf/M20_user_eval.xml"
    model.write_text('<mujoco model="M20_user_eval">\n  <include file="M20.xml"/>\n  <include file="scene.xml"/>\n  <include file="stair_user_eval.xml"/>\n</mujoco>\n')
    return model


class Runner:
    def __init__(self, model, policy, command_x, delay_steps):
        self.m = model
        self.d = mujoco.MjData(model)
        self.command = np.array([command_x, 0.0, 0.0], dtype=np.float32)
        self.mode = "LIEDOWN"
        self.stand_start_q = LIE_TARGET.copy()
        self.stand_start_t = 0.0
        self.stand_duration = 2.0
        self.reset_requested = False
        self.stop_requested = False
        self.session = ort.InferenceSession(policy, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        if inp.shape[-1] != 372 or out.shape[-1] != 16:
            raise RuntimeError(f"Expected 372->16, got {inp.shape}->{out.shape}")
        self.qidx, self.didx = {}, {}
        for name in JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"missing joint {name}")
            self.qidx[name] = model.jnt_qposadr[jid]
            self.didx[name] = model.jnt_dofadr[jid]
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.prev_action = np.zeros(16, np.float32)
        self.pending = np.zeros(16, np.float32)
        self.delay_steps = delay_steps
        self.delay_queue = [np.zeros(16, np.float32) for _ in range(delay_steps)]
        self.last_action = np.zeros(16, np.float32)
        self.t = 0.0
        self.control_dt = 0.02
        self.physics_dt = 0.001

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [0., 0., .20]
        self.d.qpos[3:7] = [1., 0., 0., 0.]
        for i, name in enumerate(JOINTS):
            self.d.qpos[self.qidx[name]] = LIE_TARGET[i]
        mujoco.mj_forward(self.m, self.d)
        self.d.ctrl[:] = 0.0
        self.prev_action.fill(0)
        self.pending.fill(0)
        self.delay_queue = [np.zeros(16, np.float32) for _ in range(self.delay_steps)]
        self.last_action.fill(0)
        self.mode = "LIEDOWN"
        self.stand_start_q = LIE_TARGET.copy()
        self.stand_start_t = self.t
        self.command[:] = 0.0
        self.reset_requested = False
        self.t = 0.0

    def elevation(self):
        # 21 rows front->rear (+1.25 to -0.75), 15 cols left->right (+.70 to -.70).
        R = quat_matrix(self.d.qpos[3:7])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        hits = np.full((21, 15), np.nan, np.float32)
        for row in range(21):
            x = 1.25 - .10 * row
            for col in range(15):
                y = .70 - .10 * col
                p = self.d.qpos[0:3] + Rz @ np.array([x, y, 20.0])
                geomid = np.array([-1], dtype=np.int32)
                terrain_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
                dist = mujoco.mj_ray(self.m, self.d, p, np.array([0., 0., -1.]), terrain_group, 1, -1, geomid)
                if dist >= 0:
                    hits[row, col] = p[2] - dist
        valid = np.isfinite(hits)
        if not valid.any():
            return np.zeros(315, np.float32)
        minimum = np.nanmin(hits)
        return np.nan_to_num(hits - minimum, nan=0., posinf=0., neginf=0.).reshape(-1).astype(np.float32)

    def obs(self):
        R = quat_matrix(self.d.qpos[3:7])
        proj_g = R.T @ np.array([0., 0., -1.])
        # MuJoCo free-joint angular velocity is qvel[3:6]; convert to base frame.
        ang = (R.T @ self.d.qvel[3:6]) * .25
        q_raw = np.array([self.d.qpos[self.qidx[n]] for n in JOINTS], np.float32)
        dq_raw = np.array([self.d.qvel[self.didx[n]] for n in JOINTS], np.float32)
        q = (q_raw - POS_OFFSET) * JOINT_DIR
        dq = dq_raw * JOINT_DIR
        qrel = q - DEFAULT
        qrel[12:] = 0.
        dqrel = dq * .05
        return np.concatenate(([*ang], [*proj_g], self.command, qrel, dqrel, self.prev_action, self.elevation())).astype(np.float32)

    def key_callback(self, keycode):
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "b" or keycode == 259:
            # B or MuJoCo Backspace reset: defer to the physics thread.
            self.reset_requested = True
        elif key == "z":
            q_raw = np.array([self.d.qpos[self.qidx[n]] for n in JOINTS], np.float32)
            self.stand_start_q = (q_raw - POS_OFFSET) * JOINT_DIR
            self.stand_start_t = self.t
            self.mode = "STAND"
            self.command[:] = 0.0
        elif key == "c":
            if self.mode != "STAND":
                print("Press Z to stand before entering RL mode", flush=True)
                return
            self.mode = "RL"
        elif key == "x":
            self.mode = "LIEDOWN"
            self.command[:] = 0.0
        elif key == "r":
            self.mode = "DAMPING"
            self.command[:] = 0.0
        elif key == "w" and self.mode == "RL":
            self.command[0] = min(float(self.command[0]) + 0.1, 0.8)
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
            self.command[:] = 0.0
        elif keycode == 256:
            self.stop_requested = True
        else:
            return
        print(
            f"MODE={self.mode} COMMAND vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} "
            f"wz={self.command[2]:+.2f}", flush=True,
        )

    def policy_step(self):
        if self.mode != "RL":
            self.pending.fill(0.0)
            self.last_action.fill(0.0)
            self.prev_action.fill(0.0)
            return self.last_action.copy(), DEFAULT.copy()
        raw = self.session.run(None, {"obs": self.obs()[None, :]})[0][0].astype(np.float32)
        raw = np.clip(raw, -100., 100.)
        if self.delay_steps == 0:
            self.pending = raw.copy()
        else:
            self.delay_queue.append(raw.copy())
            self.pending = self.delay_queue.pop(0)
        self.last_action = raw.copy()
        scaled = np.clip(raw * SCALE, -100., 100.)
        target = scaled.copy()
        target[:12] += DEFAULT[:12]
        return raw, target

    def physics_step(self):
        q_raw = np.array([self.d.qpos[self.qidx[n]] for n in JOINTS], np.float32)
        dq_raw = np.array([self.d.qvel[self.didx[n]] for n in JOINTS], np.float32)
        q = (q_raw - POS_OFFSET) * JOINT_DIR
        dq = dq_raw * JOINT_DIR
        tau = np.zeros(16, np.float32)
        if self.mode == "DAMPING":
            tau = -KD * dq
        elif self.mode == "LIEDOWN":
            lie_kp = np.array([300.0] * 12 + [0.0] * 4, np.float32)
            lie_kd = np.array([4.5] * 12 + [3.0] * 4, np.float32)
            tau[:12] = lie_kp[:12] * (LIE_TARGET[:12] - q[:12]) - lie_kd[:12] * dq[:12]
            tau[12:] = -lie_kd[12:] * dq[12:]
        else:
            delayed = self.pending if self.mode == "RL" else np.zeros(16, np.float32)
            scaled = np.clip(delayed * SCALE, -100., 100.)
            cmd = scaled.copy(); cmd[:12] += DEFAULT[:12]
            if self.mode == "STAND":
                u = np.clip((self.t - self.stand_start_t) / self.stand_duration, 0.0, 1.0)
                blend = u * u * (3.0 - 2.0 * u)
                cmd = self.stand_start_q + blend * (DEFAULT - self.stand_start_q)
                stand_kp = np.array([200.0] * 12 + [0.0] * 4, np.float32)
                stand_kd = np.array([4.0] * 12 + [1.0] * 4, np.float32)
                tau[:12] = stand_kp[:12] * (cmd[:12] - q[:12]) - stand_kd[:12] * dq[:12]
                tau[12:] = -stand_kd[12:] * dq[12:]
            else:
                tau[:12] = KP[:12] * (cmd[:12] - q[:12]) - KD[:12] * dq[:12]
                tau[12:] = KD[12:] * (cmd[12:] - dq[12:])
        tau = np.clip(tau, -EFFORT, EFFORT)
        self.d.ctrl[:] = 0.
        for ai in range(self.m.nu):
            jid = self.m.actuator_trnid[ai, 0]
            qadr = self.m.jnt_qposadr[jid]; dadr = self.m.jnt_dofadr[jid]
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            idx = JOINTS.index(name)
            self.d.ctrl[ai] = tau[idx] * JOINT_DIR[idx]
        mujoco.mj_step(self.m, self.d)
        self.t += self.physics_dt


class TerminalKeyboard:
    def __init__(self, runner):
        self.runner = runner
        self.fd = sys.stdin.fileno()
        self.old_settings = None
        self.thread = None

    def start(self):
        if not sys.stdin.isatty():
            raise RuntimeError("--terminal-control requires a foreground terminal")
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.runner.stop_requested:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if readable:
                key = sys.stdin.read(1)
                if key == "\x1b":
                    self.runner.stop_requested = True
                else:
                    self.runner.key_callback(ord(key))

    def close(self):
        if self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--height", type=float, default=.14)
    ap.add_argument("--command-x", type=float, default=.4)
    ap.add_argument("--seconds", type=float, default=8.)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--terminal-control", action="store_true")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--delay-steps", type=int, choices=(0, 1), default=0)
    ap.add_argument("--output-root", default="/home/ubuntu/m20_perceptive_final/eval_outputs/mujoco_model8500_user")
    a = ap.parse_args()
    root = pathlib.Path(a.output_root)
    xml = prepare_xml(root, a.height)
    policy = pathlib.Path(a.policy)
    if not policy.exists():
        raise RuntimeError(f"missing exported policy: {policy}")
    model = mujoco.MjModel.from_xml_path(str(xml)); model.opt.timestep = .001
    runner = Runner(model, str(policy), a.command_x, a.delay_steps); runner.reset()
    viewer = None
    terminal_keyboard = None
    if a.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, runner.d, key_callback=runner.key_callback)
    if a.terminal_control:
        terminal_keyboard = TerminalKeyboard(runner)
        terminal_keyboard.start()
        print("CONTROL: Z=stand C=RL W/S=vx A/D=vy Q/E=yaw Space=stop X=lie R=damping B=reset Esc=exit", flush=True)
    steps = int(a.seconds / runner.physics_dt); control_every = int(round(runner.control_dt / runner.physics_dt)); rows=[]; target = np.zeros(16,np.float32)
    for step in range(steps):
        tick_start = time.perf_counter()
        # The stock MuJoCo viewer can reset MjData itself (Backspace/menu).
        # Detect that time jump and replace the raw XML reset with our safe pose.
        if runner.d.time + 1.0e-6 < runner.t:
            runner.reset_requested = True
        if runner.reset_requested:
            runner.reset()
            target.fill(0.0)
            print("RESET_COMPLETE mode=LIEDOWN", flush=True)
        if runner.stop_requested or (viewer and not viewer.is_running()):
            break
        if step % control_every == 0:
            raw, target = runner.policy_step()
            runner.prev_action = raw.copy()
        runner.physics_step()
        if step % control_every == 0:
            rows.append([runner.t, runner.d.qpos[0], runner.d.qvel[0], runner.d.qpos[2], *runner.last_action])
        if viewer:
            viewer.sync()
            remaining = runner.physics_dt - (time.perf_counter() - tick_start)
            if remaining > 0:
                time.sleep(remaining)
        if not viewer and step % 400 == 0:
            print(f"t={runner.t:.2f} x={runner.d.qpos[0]:.3f} vx={runner.d.qvel[0]:.3f} z={runner.d.qpos[2]:.3f}", flush=True)
    csv_path=root / "rollout.csv"
    with csv_path.open("w") as f:
        f.write("time,base_x,base_vx,base_z," + ",".join(f"action_{i}" for i in range(16)) + "\n")
        for row in rows: f.write(",".join(map(str,row))+"\n")
    print(f"MUJOCO_POLICY_LOAD=PASS OBS_372_ALIGNMENT=PASS ACTION_16_ALIGNMENT=PASS ROLLOUT_CSV={csv_path}")
    if terminal_keyboard:
        terminal_keyboard.close()
    if viewer: viewer.close()


if __name__ == "__main__": main()
