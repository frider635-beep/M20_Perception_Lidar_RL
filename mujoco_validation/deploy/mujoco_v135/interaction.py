"""Prone/stand/ready/policy lifecycle adapted from the source deployment UI."""
from __future__ import annotations

import numpy as np


# These two transition poses and the phase duration are migrated from
# deploy_real_ws/rl_training/deploy/deploy_mujoco/deploy_mujoco.py.
# The final standing pose always comes from the selected checkpoint config.
PRONE_JOINT_POS = np.array([
    -.438, -1.16, 2.76, .438, -1.16, 2.76,
    -.438, 1.16, -2.76, .438, 1.16, -2.76, 0., 0., 0., 0.])
PRE_STAND_JOINT_POS = np.array([
    0., -.6, 1., 0., -.6, 1., 0., .6, -1., 0., .6, -1., 0., 0., 0., 0.])
STAND_PHASE_DURATION = 2.5
POLICY_BLEND_DURATION = .3


def smoothstep(fraction):
    u = float(np.clip(fraction, 0., 1.))
    return u*u*(3.-2.*u)


class Interaction:
    def __init__(self, robot, keyboard=None, *, interactive=True, settle_seconds=1.):
        self.robot, self.keyboard = robot, keyboard
        self.interactive, self.settle_seconds = interactive, settle_seconds
        self.previous = np.zeros(16, dtype=np.float32)
        self.pending_policy = False
        self.standing_snapshot = None
        self.episode = 0
        self.transitions = []
        robot.reset(prone=interactive)
        self.enter("prone" if interactive else "ready", "startup")

    def enter(self, mode, reason):
        self.mode = mode
        self.started = float(self.robot.data.time)
        self.previous.fill(0.)
        self.robot.perception.reset()
        if self.keyboard:
            self.keyboard.clear_motion()
        self.transitions.append(dict(time=self.started, episode=self.episode, mode=mode, reason=reason))
        print(f"[MODE] {mode} ({reason})", flush=True)

    def event(self, key):
        r = self.robot
        if key == "escape":
            return True
        if key == "reset":
            r.reset(prone=True)
            self.episode += 1
            self.pending_policy = False
            self.standing_snapshot = None
            self.enter("prone", "Backspace: initial prone pose")
        elif key == "r":
            # Reset is unconditional, including before the first successful Z.
            # Restore the most recent completed stand's world pose, zero all
            # velocities/forces and clear policy/perception/command history.
            r.reset(prone=False, snapshot=self.standing_snapshot)
            self.episode += 1
            self.pending_policy = False
            self.enter("ready", "R: stationary standing reset")
        elif key == "z" and self.mode in {"prone", "damping", "lying"}:
            self.initial_pose = r.data.qpos[r.qids].copy()
            self.pending_policy = False
            self.enter("standing", "Z")
        elif key == "c":
            if self.mode == "standing":
                self.pending_policy = True
                print("[MODE] C queued until standing is complete", flush=True)
            elif self.mode == "ready":
                self.enter("policy", "C")
            elif self.mode != "policy":
                print("[MODE] Press Z and wait for ready before C", flush=True)
        elif key == "x" and self.mode not in {"prone", "lying"}:
            self.initial_pose = r.data.qpos[r.qids].copy()
            self.pending_policy = False
            self.enter("lying", "X")
        return False

    def update(self):
        r = self.robot
        elapsed = float(r.data.time) - self.started
        if not self.interactive and self.mode == "ready" and elapsed >= self.settle_seconds:
            self.enter("policy", "batch settle complete")
        elif self.mode == "standing" and elapsed >= 2*STAND_PHASE_DURATION:
            pos = r.data.xpos[r.base_id]
            upright = r.data.xmat[r.base_id].reshape(3, 3)[2, 2] > .8
            if pos[2] >= r.ground() + .42 and upright:
                self.standing_snapshot = r.data.qpos.copy()
                self.enter("ready", "Z standing complete")
                if self.pending_policy:
                    self.pending_policy = False
                    self.enter("policy", "queued C")
        elif self.mode == "lying" and elapsed >= STAND_PHASE_DURATION:
            self.enter("prone", "lying complete")

    def pose_target(self):
        elapsed = float(self.robot.data.time) - self.started
        if self.mode == "standing":
            if elapsed < STAND_PHASE_DURATION:
                blend = smoothstep(elapsed/STAND_PHASE_DURATION)
                return self.initial_pose + blend*(PRE_STAND_JOINT_POS-self.initial_pose)
            blend = smoothstep((elapsed-STAND_PHASE_DURATION)/STAND_PHASE_DURATION)
            return PRE_STAND_JOINT_POS + blend*(self.robot.saved.nominal-PRE_STAND_JOINT_POS)
        if self.mode == "lying":
            blend = smoothstep(elapsed/STAND_PHASE_DURATION)
            return self.initial_pose + blend*(PRONE_JOINT_POS-self.initial_pose)
        if self.mode == "prone":
            return PRONE_JOINT_POS
        return self.robot.saved.nominal

    def action(self, observation):
        if self.mode != "policy":
            return np.zeros(16, dtype=np.float32)
        action = self.robot.saved(observation)
        if self.interactive:
            blend = np.clip((self.robot.data.time-self.started)/POLICY_BLEND_DURATION, 0., 1.)
            action = (action*blend).astype(np.float32)
        return action
