#!/usr/bin/env python3
"""Static gates for official V10 stair terrain and terrain/gait curriculum wiring."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "source/rl_training").is_dir():
    ROOT = Path(__file__).resolve().parents[1]

if (ROOT / "source/v10_unified_env_cfg.py").is_file():
    env_source = (ROOT / "source/v10_unified_env_cfg.py").read_text()
    terrain_source = ""
    curriculum_source = ""
else:
    cfg_dir = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20"
    mdp_dir = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"
    env_source = (cfg_dir / "v10_unified_env_cfg.py").read_text()
    terrain_source = (cfg_dir / "perceptive_terrains.py").read_text()
    curriculum_source = (mdp_dir / "curriculums.py").read_text()

official_generator = (
    "M20_PERCEPTIVE_TERRAINS_CFG" in env_source
    and "V10_UNIFIED_TERRAINS_CFG" not in env_source
)
official_inv = (
    not terrain_source
    or (
        'ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs_inv"]' in terrain_source
        and '"pyramid_stairs_inv": OFFICIAL_PYRAMID_STAIRS_INV_CFG' in terrain_source
    )
)
terrain_curriculum = "CurrTerm(func=mdp.terrain_levels_vel)" in env_source
gait_curve = "CurrTerm(func=mdp.gait_level_curve)" in env_source and "self.curriculum.gait_level = None" not in env_source
gait_update = not curriculum_source or "update_gait_level_from_terrain_mean(mean_level)" in curriculum_source
official_tread = "step_depth: float = 0.30" in env_source
full_only = "full_stair_column_start: int = 0" in env_source

results = {
    "OFFICIAL_INVERTED_PYRAMID_TERRAIN": official_generator and official_inv,
    "OFFICIAL_TERRAIN_LEVELS_VEL": terrain_curriculum,
    "OFFICIAL_GAIT_LEVEL_CURVE": gait_curve,
    "OFFICIAL_TERRAIN_GAIT_UPDATE": gait_update,
    "OFFICIAL_TREAD_030": official_tread,
    "OFFICIAL_ALL_COLUMNS_FULL": full_only,
}
for name, passed in results.items():
    print(f"{name}={'PASS' if passed else 'FAIL'}")
raise SystemExit(0 if all(results.values()) else 1)
