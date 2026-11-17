"""Keyboard/viewer lifecycle migrated from deploy_mujoco.py; no policy dependencies."""
from __future__ import annotations
import fcntl
import os
import select
import termios
import threading
import time
from collections import deque
import numpy as np
import mujoco
from keyboard_input import X11WindowKeys
from key_bindings import MOTION_COMMANDS, glfw_motion_key
POLICY_DT = .02
TELEOP_KEY_TIMEOUT = .5
TELEOP_VELOCITIES = MOTION_COMMANDS
TELEOP_LIMITS = np.max(np.abs(list(TELEOP_VELOCITIES.values())), axis=0)
TELEOP_RISE_TIME = 1.0
TELEOP_STOP_TIME = .3

class TeleopKeyboard:
    """Share MuJoCo window and terminal key presses with the physics loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[str] = deque()
        self._last_motion_press: dict[str, float] = {}
        self._window_keys = X11WindowKeys.open()
        self._command = np.zeros(3, dtype=np.float64)
        self._terminal_physical = False

    def on_keycode(self, keycode: int) -> None:
        motion = glfw_motion_key(keycode)
        with self._lock:
            if self._window_keys is not None:
                self._window_keys.note_viewer_event()
                # MuJoCo invokes this callback on PRESS, not GLFW_REPEAT/RELEASE.
                # Motion keys therefore come from physical state, not a timeout.
                if motion is not None:
                    return
        if motion is not None:
            self.on_char(motion)
        elif keycode == 256:  # GLFW escape
            self.on_char("escape")
        elif keycode == 259:  # GLFW backspace (also resets the native viewer)
            self.on_char("reset")
        elif 65 <= keycode <= 90:
            self.on_char(chr(keycode).lower())

    def on_char(self, key: str) -> None:
        key = "escape" if key == "\x1b" else "reset" if key in {"\x7f", "\x08"} else key.lower()
        with self._lock:
            if key in TELEOP_VELOCITIES:
                if not self._terminal_physical:
                    self._last_motion_press[key] = time.monotonic()
            elif key in {"z", "c", "x", "r", "h", "escape", "reset"}:
                self._events.append(key)

    def note_terminal_event(self) -> None:
        with self._lock:
            # Once verified, delayed terminal repeats after focus loss must
            # never re-enable the character-timeout fallback.
            self._terminal_physical |= (self._window_keys is not None
                                        and self._window_keys.note_terminal_event())
            if self._terminal_physical:
                self._last_motion_press.clear()

    def take_events(self) -> list[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def clear_motion(self) -> None:
        with self._lock:
            self._last_motion_press.clear()
            self._command.fill(0.)

    def close(self) -> None:
        with self._lock:
            if self._window_keys is not None:
                self._window_keys.close()
                self._window_keys = None
            self._last_motion_press.clear()
            self._command.fill(0.)

    def velocity_command(self) -> np.ndarray:
        now = time.monotonic()
        with self._lock:
            self._last_motion_press = {
                key: seen for key, seen in self._last_motion_press.items()
                if now - seen <= TELEOP_KEY_TIMEOUT
            }
            if self._window_keys is not None and self._window_keys.focused():
                pressed = self._window_keys.pressed()
                self._last_motion_press.clear()
            elif self._terminal_physical or not self._last_motion_press:
                # Focus loss must not leave a decaying movement command behind.
                self._command.fill(0.)
                return self._command.astype(np.float32)
            else:
                pressed = set(self._last_motion_press)
            target = np.zeros(3, dtype=np.float64)
            for key in pressed:
                target += TELEOP_VELOCITIES[key]
            target = np.clip(target, -TELEOP_LIMITS, TELEOP_LIMITS)
            # Reverse through zero first. Each call is one policy tick, so a
            # paused viewer or slow render cannot jump straight to full speed.
            braking = (target * self._command < 0) | (target == 0)
            goal = np.where(braking, 0., target)
            rate = TELEOP_LIMITS / np.where(braking, TELEOP_STOP_TIME, TELEOP_RISE_TIME)
            self._command += np.clip(goal - self._command, -rate * POLICY_DT, rate * POLICY_DT)
            return self._command.astype(np.float32)

class TerminalInput:
    """Read the same keys from a focused terminal when stdin is a TTY."""

    def __init__(self, keyboard: TeleopKeyboard) -> None:
        self.keyboard = keyboard
        self.fd = 0
        self.saved_attrs = None
        self.saved_flags = None

    def start(self) -> bool:
        if not os.isatty(self.fd):
            return False
        self.saved_attrs = termios.tcgetattr(self.fd)
        self.saved_flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        attrs = termios.tcgetattr(self.fd)
        attrs[3] &= ~(termios.ECHO | termios.ICANON)
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.saved_flags | os.O_NONBLOCK)
        return True

    def poll(self) -> None:
        if self.saved_attrs is None:
            return
        while select.select([self.fd], [], [], 0.0)[0]:
            try:
                keys = os.read(self.fd, 64)
            except BlockingIOError:
                break
            if not keys:
                break
            self.keyboard.note_terminal_event()
            for key in keys:
                self.keyboard.on_char(chr(key))

    def close(self) -> None:
        if self.saved_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved_attrs)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.saved_flags)
            self.saved_attrs = None

def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData, key_callback=None):
    from mujoco import viewer as mj_viewer
    existing_threads = set(threading.enumerate())
    viewer = mj_viewer.launch_passive(model, data, key_callback=key_callback)
    viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
    return viewer, viewer_threads

def close_viewer(viewer, viewer_threads: list[threading.Thread]) -> None:
    if viewer is None:
        return
    viewer.close()
    # MuJoCo 3.10 close() requests exit but does not join its GL thread.
    for thread in viewer_threads:
        thread.join(timeout=5.0)
    if any(thread.is_alive() for thread in viewer_threads):
        raise RuntimeError("MuJoCo viewer thread did not stop after close()")
