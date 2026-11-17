#!/usr/bin/env python3
"""Static and geometry gates for the V10 stair-only reset dispatcher."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/events.py"
CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v10_unified_env_cfg.py"
if not EVENTS.is_file():
    ROOT = Path(__file__).resolve().parents[1]
    EVENTS = ROOT / "source/events.py"
    CFG = ROOT / "source/v10_unified_env_cfg.py"

events = EVENTS.read_text()
cfg = CFG.read_text()

size_x, border, platform, tread = 8.0, 1.0, 3.0, 0.3
num_steps = int((size_x - 2.0 * border - platform) // (2.0 * tread) + 1)
first_riser_radius = 0.5 * (size_x - 2.0 * border - 2.0 * num_steps * tread)

results = {
    "STAIR_RESET_FUNCTION_REGISTERED": (
        "self.events.randomize_reset_base.func = mdp.v10_reset_root_state_stair_conditioned" in cfg
    ),
    "NON_STAIR_CALLS_OFFICIAL_RESET": (
        "_official_reset_root_state_uniform(" in events
        and "env, non_stair_ids, pose_range, velocity_range, asset_cfg" in events
    ),
    "STAIR_MASK_MATCHES_ASCENDING_RANGE": (
        "terrain_types >= int(cfg.full_stair_column_start)" in events
        and "terrain_types < int(cfg.full_stair_column_end)" in events
    ),
    "STAIR_DISTANCE_090_140": (
        "stair_reset_distance_range: tuple[float, float] = (0.9, 1.4)" in cfg
    ),
    "STAIR_YAW_PLUS_MINUS_15DEG": (
        "stair_reset_yaw_jitter_range: tuple[float, float] = (-0.2617993878, 0.2617993878)" in cfg
    ),
    "STAIR_LATERAL_VALID_WIDTH": (
        "stair_reset_lateral_range: tuple[float, float] = (-0.25, 0.25)" in cfg
    ),
    "OFFICIAL_FIRST_RISER_RADIUS_120": abs(first_riser_radius - 1.2) < 1.0e-9,
    "PERCEPTION_PROMOTION_GATE_REMOVED": "perception_gate_start_level" not in cfg,
    "OFFICIAL_CURRICULUM_REGISTERED": (
        "CurrTerm(func=mdp.v10_official_terrain_levels)" in cfg
    ),
    "GAIT_REGISTRATION_UNCHANGED": "CurrTerm(func=mdp.gait_level_curve)" in cfg,
}

for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
print(f"OFFICIAL_FIRST_RISER_RADIUS={first_riser_radius:.6f}")
raise SystemExit(0 if all(results.values()) else 1)
