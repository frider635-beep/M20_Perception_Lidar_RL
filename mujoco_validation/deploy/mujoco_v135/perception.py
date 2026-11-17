"""NumPy counterpart of V11 signed-edge extraction and actor-side filtering.

The detector is compared against the training tensor function in test_contract.py.
No terrain parameters, reward state or contact labels enter the actor features.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from terrain_loader import terrain_ray, static_terrain_flex_ids


def extract_risers(points, cfg):
    if points.shape != (21, 15, 3):
        raise ValueError("Expected a 21x15 yaw-local point grid")
    p = points[::-1]
    low, up = p[:-1], p[1:]
    signed = up[..., 2] - low[..., 2]
    height = signed if cfg["step_feature_mode"] == "signed_edges" else np.maximum(signed, 0.)
    x, y = .5 * (low[..., 0] + up[..., 0]), .5 * (low[..., 1] + up[..., 1])
    lower = np.minimum(low[..., 2], up[..., 2])
    valid = (np.isfinite(low[..., 2]) & np.isfinite(up[..., 2])
             & (np.abs(y) <= cfg["corridor_half_width"])
             & (np.abs(height) >= cfg["min_step_height"] - cfg["height_valid_tolerance"]))
    yi = np.argmax(np.where(valid, np.abs(height), -np.inf), axis=1)
    rows = np.arange(len(yi))
    rh, rx, ry, rl = (a[rows, yi] for a in (height, x, y, lower))
    rv = valid.any(axis=1) & (rx > cfg["passed_distance"])
    first = int(np.argmin(np.where(rv, np.abs(rx), np.inf)))
    second_valid = rv & (rx > rx[first] + cfg["min_riser_separation"]) & rv.any()
    if cfg["step_feature_mode"] == "signed_edges":
        second_valid &= rh * rh[first] > 0.
    second = int(np.argmin(np.where(second_valid, rx, np.inf)))
    mask = np.asarray([rv.any(), second_valid.any()])
    def pair(a):
        return np.where(mask, a[[first, second]], 0.)
    return dict(valid=mask, riser_height=pair(rh), riser_distance=pair(rx),
                riser_x=pair(rx), riser_y=pair(ry), riser_lower_world=pair(rl))


def actor_feature(risers, command_x, cfg):
    valid = risers["valid"] & (np.abs(risers["riser_height"]) <= cfg["max_step_height"] + cfg["height_valid_tolerance"])
    distance = risers["riser_distance"]
    promote = not valid[0] or distance[0] < cfg["anticipation_distance_min"]
    primary = int(promote)
    b1 = (valid[primary] and cfg["anticipation_distance_min"] <= distance[primary] <= cfg["anticipation_distance_max"]
          and command_x > cfg["forward_command_threshold"])
    b2 = not promote and valid[1] and b1
    return np.asarray([float(b1), risers["riser_height"][primary] if b1 else 0., distance[primary] if b1 else 0.,
                       float(b2), risers["riser_height"][1] if b2 else 0., distance[1] if b2 else 0.], dtype=np.float32)


class Perception:
    def __init__(self, model, data, base_id, saved, mode="clean", seed=42):
        self.model, self.data, self.base_id = model, data, base_id
        self.cfg, self.mode = saved.perception, mode
        self.rng = np.random.default_rng(seed)
        scanner = saved.cfg["scene"]["v10_pgtt_scanner"]
        pattern, offset = scanner["pattern_cfg"], scanner["offset"]["pos"]
        if scanner["ray_alignment"] != "yaw" or pattern["direction"] != [0., 0., -1.]:
            raise ValueError("Only saved yaw-aligned downward rays are supported")
        size, resolution = pattern["size"], pattern["resolution"]
        nx, ny = (round(v / resolution) + 1 for v in size)
        if (nx, ny) != (21, 15):
            raise ValueError("Checkpoint does not use the 21x15 two-edge scanner")
        xx, yy = np.meshgrid(np.linspace(size[0]/2, -size[0]/2, nx) + offset[0],
                             np.linspace(size[1]/2, -size[1]/2, ny) + offset[1], indexing="ij")
        self.local_xy = np.stack((xx, yy), -1)
        self.ray_height = float(offset[2])
        self.groups = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.flex_ids = static_terrain_flex_ids(model)
        self.reset()

    def reset(self):
        self.tracked = np.zeros(6, dtype=np.float32)
        self.confirm = np.zeros(2, dtype=int)
        self.invalid = np.zeros(2, dtype=int)
        self.last = np.zeros(6, dtype=np.float32)
        self.ring = deque([np.zeros(6, dtype=np.float32) for _ in range(self.cfg["perception_delay_steps"] + 1)],
                          maxlen=self.cfg["perception_delay_steps"] + 1)

    def filter(self, detected):
        bits = detected[[0, 3]] > .5
        prior = self.tracked[[0, 3]] > .5
        self.confirm = np.where(bits, self.confirm + 1, 0)
        self.invalid = np.where(bits, 0, self.invalid + 1)
        active = (bits & (self.confirm >= self.cfg["confirmation_frames"])) | (
            prior & (self.invalid <= self.cfg["hysteresis_frames"]))
        for slot, offset in enumerate((0, 3)):
            if bits[slot]:
                self.tracked[offset+1:offset+3] = detected[offset+1:offset+3]
            self.tracked[offset] = active[slot]
            self.tracked[offset+1:offset+3] *= active[slot]
        noisy = self.tracked.copy()
        if self.mode == "noisy":
            if self.rng.random() < self.cfg["frame_hold_probability"]:
                noisy = self.last.copy()
            noisy[[1, 4]] += self.rng.normal(0, self.cfg["height_noise_std"], 2)
            noisy[[2, 5]] += self.rng.normal(0, self.cfg["distance_noise_std"], 2)
        noisy[[0, 2, 3, 5]] = np.maximum(noisy[[0, 2, 3, 5]], 0.)
        noisy[[1, 4]] = np.clip(noisy[[1, 4]], -self.cfg["max_step_height"], self.cfg["max_step_height"])
        # Training writes at the current ring index and reads index-delay.
        self.ring.append(noisy)
        feature = self.ring[0].copy()
        feature[1:3] *= feature[0]
        feature[4:6] *= feature[3]
        self.last = feature
        return feature

    def observe(self, command_x):
        rot = self.data.xmat[self.base_id].reshape(3, 3)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        co, si = np.cos(yaw), np.sin(yaw)
        base = self.data.xpos[self.base_id]
        points = np.concatenate((self.local_xy, np.full((21, 15, 1), np.nan)), axis=-1)
        origin, down = np.zeros(3), np.asarray([0., 0., -1.])
        for row in range(21):
            for col in range(15):
                x, y = self.local_xy[row, col]
                origin[:] = (base[0] + co*x - si*y, base[1] + si*x + co*y, base[2] + self.ray_height)
                distance, _, _ = terrain_ray(self.model, self.data, origin, down, self.groups, self.flex_ids)
                if distance >= 0 and np.isfinite(distance):
                    points[row, col, 2] = origin[2] - distance
        self.last_points = points
        if not np.isfinite(points[..., 2]).any():
            raise RuntimeError("All terrain rays missed; refusing to substitute flat-ground observations")
        return self.filter(actor_feature(extract_risers(points, self.cfg), command_x, self.cfg))
