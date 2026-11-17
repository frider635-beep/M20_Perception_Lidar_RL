#!/usr/bin/env python3
"""Closed-loop MuJoCo validation of a saved native-63D M20 actor."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import json
from itertools import count
import math
import os
from pathlib import Path
import sys
import time

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from policy import JOINT_NAMES, SavedPolicy, sha256
from perception import Perception
from reset_placement import place_on_terrain
from terrain_loader import attach_terrain, is_terrain_geom, static_terrain_flex_ids, terrain_ray
import teleop
from interaction import Interaction, PRONE_JOINT_POS

def default_checkpoint():
    candidates = [REPO.parent / "M20_perception_Lidar_rl/interview_prep/model_35300.pt",
                  REPO / "interview_prep/model_35300.pt"]
    return next((p for p in candidates if p.is_file()), candidates[0])


class Robot:
    def __init__(self, saved, args):
        self.saved = saved
        spec = mujoco.MjSpec.from_file(str(args.robot_xml.resolve()))
        spec.option.timestep = args.physics_dt
        spec.option.gravity = saved.cfg["sim"]["gravity"]
        overrides = {k: getattr(args, k) for k in ("stair_height", "tread_depth", "stair_count", "stair_start", "spawn_xy")}
        self.terrain = attach_terrain(spec, args.terrain_xml, overrides)
        self.model = model = spec.compile()
        self.data = data = mujoco.MjData(model)
        self.base_id = model.body("base_link").id
        self.root = int(model.joint("floating_base").qposadr[0])
        self.qids = np.asarray([model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.vids = np.asarray([model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self.aids = np.asarray([model.actuator(n).id for n in JOINT_NAMES])
        if model.nu != 16 or len(set(self.qids)) != 16 or len(set(self.aids)) != 16:
            raise ValueError("M20 joint/actuator mapping must contain 16 distinct entries")
        jids = np.asarray([model.joint(n).id for n in JOINT_NAMES])
        if not np.array_equal(model.actuator_trnid[self.aids, 0], jids):
            raise ValueError("Named actuators do not drive the matching named joints")
        if not np.allclose(model.actuator_gear[self.aids, 0], 1.):
            raise ValueError("Nonunit actuator gears are unsupported")
        model.dof_armature[self.vids] = saved.armature
        self.wheels = np.asarray([model.body(f"{leg}_wheel").id for leg in ("fl", "fr", "hl", "hr")])
        self.terrain_geoms = frozenset(g for g in range(model.ngeom) if is_terrain_geom(model, g))
        self.flex = static_terrain_flex_ids(model)
        self.groups = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        # Camera visibility only. Collision masks and friction are unchanged.
        for g in range(model.ngeom):
            if model.geom(g).name.startswith("terrain_wall_"):
                model.geom_rgba[g, 3] = args.wall_alpha
        self.perception = Perception(model, data, self.base_id, saved, args.perception, args.seed)
        self.reset()

    def reset(self, prone=False, snapshot=None):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        x, y = self.terrain.spawn_xy
        yaw = self.terrain.spawn_yaw
        d.qpos[self.root:self.root+7] = (x, y, 0., math.cos(yaw/2), 0., 0., math.sin(yaw/2))
        pose = PRONE_JOINT_POS if prone else self.saved.nominal
        d.qpos[self.qids] = pose
        mujoco.mj_forward(m, d)
        if snapshot is None:
            place_on_terrain(m, d, root_body_id=self.base_id, root_qpos_adr=self.root,
                             terrain_geom_ids=self.terrain_geoms, terrain_flex_ids=self.flex)
        else:
            snapshot = np.asarray(snapshot)
            if snapshot.shape != d.qpos.shape or not np.isfinite(snapshot).all():
                raise ValueError("Invalid standing reset snapshot")
            d.qpos[:] = snapshot
        mujoco.mj_forward(m, d)
        if snapshot is None and not np.allclose(d.qpos[self.qids], pose):
            raise RuntimeError("Joint-order/reset roundtrip failed")
        self.perception.reset()

    def ground(self):
        pos = self.data.xpos[self.base_id]
        origin = np.asarray([pos[0], pos[1], self.terrain.ray_top])
        distance, _, _ = terrain_ray(self.model, self.data, origin, np.asarray([0., 0., -1.]), self.groups, self.flex)
        if distance < 0:
            raise RuntimeError("No terrain under the robot")
        return float(origin[2] - distance)

    def observe(self, command, previous):
        d, m = self.data, self.model
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 1)
        gravity = d.xmat[self.base_id].reshape(3, 3).T @ np.asarray([0., 0., -1.])
        feature = self.perception.observe(float(command[0]))
        return self.saved.observation(velocity[:3], gravity, command, d.qpos[self.qids],
                                      d.qvel[self.vids], previous, feature)

    def control(self, action, mode="policy"):
        s, d = self.saved, self.data
        q, v = d.qpos[self.qids], d.qvel[self.vids]
        target = s.nominal + s.action_scale * action
        vtarget = np.r_[np.zeros(12), s.action_scale[12:] * action[12:]]
        torque = s.kp * (target-q) + s.kd * (vtarget-v)
        if mode == "damping":
            torque = -s.kd * v
        d.ctrl[self.aids] = np.clip(torque, -s.effort, s.effort)

    def control_pose(self, target, damping=False):
        s, d = self.saved, self.data
        torque = -s.kd*d.qvel[self.vids]
        if not damping:
            torque += s.kp*(target-d.qpos[self.qids])
        d.ctrl[self.aids] = np.clip(torque, -s.effort, s.effort)

    def contacts(self):
        support = np.zeros(4, dtype=bool)
        nonwheel = 0
        for contact in self.data.contact:
            a, b = (int(g) for g in contact.geom)
            if a in self.terrain_geoms:
                robot_geom = b
            elif b in self.terrain_geoms:
                robot_geom = a
            else:
                continue
            if robot_geom < 0:
                continue
            body = self.model.geom_bodyid[robot_geom]
            ids = np.flatnonzero(self.wheels == body)
            if len(ids) and abs(contact.frame[2]) > .5 and contact.dist <= .001:
                support[ids[0]] = True
            elif not len(ids):
                nonwheel += 1
        return support, nonwheel


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    p.add_argument("--config-dir", type=Path,
                   help="Original training params/ containing env.yaml and agent.yaml; otherwise uses adjacent params/ or a SHA-256-matched local checkpoint")
    p.add_argument("--robot-xml", type=Path, default=HERE / "assets/m20/mjcf/M20.xml")
    p.add_argument("--terrain-xml", type=Path, default=HERE / "terrains/stairs_ascent_walls.xml")
    display = p.add_mutually_exclusive_group()
    display.add_argument("--headless", action="store_true", help="No interactive window; can still record --video")
    display.add_argument("--viewer", action="store_true", help="Open the interactive window (default)")
    p.add_argument("--keyboard", action="store_true", help="Hold 8/5/4/6 to move, 7/9 to turn; starts at zero command")
    p.add_argument("--command", type=float, nargs=3, default=(.5, 0., 0.), metavar=("VX", "VY", "WZ"))
    p.add_argument("--duration", type=float, default=None,
                   help="Total simulated seconds; viewer default is unlimited (0), headless default is 30")
    p.add_argument("--settle-seconds", type=float, default=1.)
    p.add_argument("--physics-dt", type=float, default=.001)
    p.add_argument("--perception", choices=("clean", "noisy"), default="clean",
                   help="Both retain saved confirmation/hysteresis/delay; noisy adds saved feature noise/drop-hold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wall-alpha", type=float, default=.25, help="Display opacity only; walls still collide")
    p.add_argument("--stair-height", type=float)
    p.add_argument("--tread-depth", type=float)
    p.add_argument("--stair-count", type=float)
    p.add_argument("--stair-start", type=float)
    p.add_argument("--spawn-xy", type=float, nargs=2)
    p.add_argument("--output", type=Path, help="New result directory; defaults to runs/<timestamp>")
    p.add_argument("--video", type=Path, help="Optional MP4; supports EGL in --headless mode")
    p.add_argument("--require-goal", action="store_true", help="Exit 2 unless the geometric finish criterion passes")
    p.add_argument("--stop-at-goal", action="store_true", help="Explicitly exit at the finish; default keeps running")
    p.add_argument("--stop-on-fall", action=argparse.BooleanOptionalAction, default=None,
                   help="Viewer default keeps the window open for reset; headless default stops on fall")
    args = p.parse_args()
    if args.duration is None:
        args.duration = 30. if args.headless else 0.
    if args.stop_on_fall is None:
        args.stop_on_fall = args.headless
    if not np.isfinite([*args.command, args.duration, args.settle_seconds, args.physics_dt, args.wall_alpha]).all():
        p.error("Arguments must be finite")
    if not (0 < args.physics_dt <= .005 and args.duration >= 0 and args.settle_seconds >= 0 and 0 <= args.wall_alpha <= 1):
        p.error("Require valid nonnegative duration/settle time, physics-dt<=0.005 and wall-alpha in [0,1]")
    if args.headless and (args.duration == 0 or args.settle_seconds >= args.duration):
        p.error("Headless runs require duration>settle-seconds>=0")
    if args.headless and args.keyboard:
        p.error("--keyboard requires the interactive window")
    return args


def open_displays(args, model, data):
    resources = ExitStack()
    viewer = keyboard = terminal = renderer = writer = None
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135., -25., 4.
    try:
        if not args.headless:
            keyboard = teleop.TeleopKeyboard()
            resources.callback(keyboard.close)
            viewer, threads = teleop.launch_viewer(model, data, keyboard.on_keycode)
            resources.callback(teleop.close_viewer, viewer, threads)
            with viewer.lock():
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        if args.keyboard:
            terminal = teleop.TerminalInput(keyboard)
            terminal.start()
            resources.callback(terminal.close)
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            if args.video.exists():
                raise FileExistsError(f"Refusing to overwrite existing video: {args.video}")
            model.vis.global_.offwidth = 1280
            model.vis.global_.offheight = 720
            renderer = mujoco.Renderer(model, height=720, width=1280)
            resources.callback(renderer.close)
            writer = imageio.get_writer(str(args.video), fps=1./(2*teleop.POLICY_DT))
            resources.callback(writer.close)
    except BaseException:
        resources.close()
        raise
    return resources, keyboard, terminal, viewer, camera, renderer, writer


def run(args, *, event_source=None):
    torch.set_num_threads(1)
    saved = SavedPolicy(args.checkpoint, args.config_dir)
    substeps = round(saved.period / args.physics_dt)
    if not math.isclose(substeps * args.physics_dt, saved.period):
        raise ValueError("Saved policy period must be an integer multiple of physics-dt")
    robot = Robot(saved, args)
    m, d = robot.model, robot.data
    output = (args.output or HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    teleop.POLICY_DT = saved.period
    session = Interaction(robot, interactive=not args.headless or event_source is not None,
                          settle_seconds=args.settle_seconds)
    resources, keyboard, terminal, viewer, camera, renderer, writer = open_displays(args, m, d)
    session.keyboard = keyboard
    records = {k: [] for k in ("time", "episode", "root_pos", "root_quat", "qpos", "qvel",
                               "observation", "action", "command", "wheel_support", "nonwheel_contacts",
                               "mode", "pose_target")}
    fell, bypass, finished, status, support_steps = False, False, False, "timeout", 0
    episode_fell, episode_bypass, episode_finished, current_episode = False, False, False, 0
    steps_done, chunks, goal_events = 0, [], []
    initial_x = float(d.xpos[robot.base_id, 0])
    max_x, max_height, feature_frames = initial_x, float(d.xpos[robot.base_id, 2]), 0
    metadata = {**saved.metadata, **robot.terrain.summary(), "robot_xml": str(args.robot_xml.resolve()),
                "robot_xml_sha256": sha256(args.robot_xml), "terrain_xml_sha256": sha256(args.terrain_xml),
                "physics_dt": args.physics_dt, "physics_enabled": True, "perception_mode": args.perception,
                "proprioception_noise": False, "actuator_delay_steps": 0,
                "simulator": f"MuJoCo {mujoco.__version__}", "seed": args.seed,
                "command": list(args.command), "keyboard": args.keyboard,
                "wall_alpha": args.wall_alpha, "duration": args.duration, "settle_seconds": args.settle_seconds,
                "interactive": session.interactive, "stop_at_goal": args.stop_at_goal,
                "stop_on_fall": args.stop_on_fall,
                "fall_criterion": "base clearance <0.22m or projected_gravity_z >-0.35 (deployment baseline)",
                "finish_criterion": "all wheels beyond terrain goal+wheel_radius, inside width, no observed bypass/fall, sustained upward terrain contacts; not strict per-riser gait acceptance"}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("[MODEL] " + str(saved.path), flush=True)
    print(f"[CONFIG] {saved.config_path.parent} ({saved.metadata['config_resolution']})", flush=True)
    if saved.metadata["config_source_checkpoint"]:
        print("[SOURCE] SHA-256 identical: " + saved.metadata["config_source_checkpoint"], flush=True)
    print(f"[CONTRACT] 63 -> 16, iteration={saved.iteration}, control={saved.period:g}s, physics={args.physics_dt:g}s", flush=True)
    print("[OUTPUT] " + str(output), flush=True)
    if viewer:
        print("[KEYS] Z stand, C policy, R reset to stationary stand, Backspace reset prone; 8/5 forward/back, 4/6 lateral, 7/9 yaw (--keyboard); X lie, Esc quit", flush=True)
    try:
        for step in count():
            if args.duration > 0 and step >= math.ceil(args.duration / saved.period):
                break
            tick = time.perf_counter()
            if terminal:
                terminal.poll()
            events = keyboard.take_events() if keyboard else []
            if event_source is not None:
                events.extend(event_source(step))
            for event in events:
                if session.event(event):
                    status = "user_exit"
                    break
            if status == "user_exit":
                break
            if current_episode != session.episode:
                current_episode = session.episode
                episode_fell, episode_bypass, episode_finished, support_steps = False, False, False, 0
            session.update()
            command = keyboard.velocity_command() if args.keyboard else np.asarray(args.command, dtype=np.float32)
            active = session.mode == "policy"
            if not active:
                command = np.zeros(3, dtype=np.float32)
            obs = robot.observe(command, session.previous)
            action = session.action(obs)
            target = session.pose_target()
            support, other_contact = robot.contacts()
            for key, value in dict(time=float(d.time), episode=session.episode, root_pos=d.xpos[robot.base_id],
                                   root_quat=d.xquat[robot.base_id], qpos=d.qpos, qvel=d.qvel,
                                   observation=obs, action=action, command=command,
                                   wheel_support=support, nonwheel_contacts=other_contact,
                                   mode=session.mode, pose_target=target).items():
                records[key].append(np.copy(value))
            for _ in range(substeps):
                if active:
                    robot.control(action)
                else:
                    robot.control_pose(target, damping=session.mode == "damping")
                mujoco.mj_step(m, d)
                if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                    raise RuntimeError("MuJoCo produced a nonfinite physical state")
            mujoco.mj_forward(m, d)
            session.previous = action.copy()
            steps_done += 1
            pos = d.xpos[robot.base_id]
            max_x, max_height = max(max_x, float(pos[0])), max(max_height, float(pos[2]))
            feature_frames += int(obs[57] > .5)
            gravity_z = -d.xmat[robot.base_id].reshape(3, 3)[2, 2]
            falling_now = bool(active and (pos[2] < robot.ground() + .22 or gravity_z > -.35))
            episode_fell |= falling_now
            fell |= falling_now
            wheels = d.xpos[robot.wheels]
            inside = bool(np.all(np.abs(wheels[:, 1]) <= robot.terrain.course_width/2))
            episode_bypass |= bool(active and not inside and pos[0] < robot.terrain.goal_x)
            bypass |= episode_bypass
            support, _ = robot.contacts()
            crossed = bool(np.all(wheels[:, 0] > robot.terrain.goal_x + saved.perception["wheel_radius"]))
            support_steps = support_steps + 1 if crossed and inside and support.all() and not episode_fell and not episode_bypass else 0
            new_goal = not episode_finished and support_steps * saved.period >= .5
            if new_goal:
                episode_finished, finished = True, True
                goal_events.append(dict(step=step, time=float(d.time), episode=session.episode))
                print("[GOAL] Finish recorded; session continues unless --stop-at-goal is set", flush=True)
            if step % 50 == 0 or falling_now or new_goal:
                print(f"mode={session.mode} t={d.time:6.2f} x={pos[0]:6.3f} z={pos[2]:5.3f} BHD={np.round(obs[57:],3).tolist()} fall={falling_now}", flush=True)
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = pos
                before = float(d.time)
                viewer.sync()
                if d.time < before:  # native viewer Backspace also changes MjData
                    session.event("reset")
                if not viewer.is_running():
                    status = "user_exit"
                    break
            if renderer and step % 2 == 0:
                camera.lookat[:] = pos
                renderer.update_scene(d, camera=camera)
                writer.append_data(renderer.render())
            if falling_now and args.stop_on_fall:
                status = "fallen"
                break
            if new_goal and args.stop_at_goal:
                status = "goal_reached"
                break
            if falling_now and session.interactive:
                session.pending_policy = False
                session.enter("damping", "fall detected; R resets standing, Backspace resets prone")
            # Unlimited viewer sessions persist bounded chunks instead of
            # accumulating the entire trajectory in RAM.
            if len(records["time"]) >= 4096:
                part = f"trajectory_{len(chunks):05d}.npz"
                np.savez_compressed(output / part, **{k: np.asarray(v) for k, v in records.items()})
                chunks.append(part)
                for values in records.values():
                    values.clear()
            if viewer:
                time.sleep(max(0., saved.period - (time.perf_counter()-tick)))
    except KeyboardInterrupt:
        status = "user_exit"
    except Exception:
        status = "runtime_error"
        raise
    finally:
        d.ctrl[:] = 0.
        resources.close()
        np.savez_compressed(output / "trajectory.npz", **{k: np.asarray(v) for k, v in records.items()})
        result = dict(status=status, goal_reached=finished, fell=fell, observed_bypass=bypass,
                      control_steps=steps_done, simulated_seconds=steps_done*saved.period,
                      final_position=d.xpos[robot.base_id].tolist(), max_x=max_x, max_base_height=max_height,
                      frames_with_active_edge=feature_frames, reset_count=session.episode,
                      final_mode=session.mode, transitions=session.transitions,
                      goal_events=goal_events, trajectory_parts=chunks + ["trajectory.npz"],
                      output=str(output), strict_gait_acceptance="not_measured")
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print("[RESULT] " + json.dumps(result), flush=True)
    return 2 if args.require_goal and not finished else 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
