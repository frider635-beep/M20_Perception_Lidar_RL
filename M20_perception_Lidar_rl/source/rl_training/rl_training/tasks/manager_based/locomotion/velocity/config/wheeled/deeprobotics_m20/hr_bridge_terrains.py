"""Independent fixed stair terrains for HR Bridge gates and evaluators."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


@configclass
class HRBridgeFixedStairCfg(SubTerrainBaseCfg):
    function = None
    step_heights: tuple[float, ...] = (0.05,)
    step_depth: float = 0.31
    levels: int = 6
    first_riser_from_spawn: float = 1.10
    spawn_x: float = 1.25


def hr_bridge_fixed_stair(difficulty: float, cfg: HRBridgeFixedStairCfg):
    heights = tuple(float(value) for value in cfg.step_heights)
    index = min(int(float(difficulty) * len(heights)), len(heights) - 1)
    height = heights[index]
    sx, sy = cfg.size
    first_x = cfg.spawn_x + cfg.first_riser_from_spawn

    def box(x0, x1, top_z):
        thickness = max(float(top_z), 0.02)
        return trimesh.creation.box(
            (x1 - x0, sy, thickness),
            trimesh.transformations.translation_matrix(
                ((x0 + x1) * 0.5, sy * 0.5, top_z - thickness * 0.5)
            ),
        )

    meshes = [box(0.0, sx, 0.0)]
    for level in range(1, cfg.levels + 1):
        x0 = first_x + (level - 1) * cfg.step_depth
        meshes.append(box(x0, sx, level * height))
    return meshes, np.asarray((cfg.spawn_x, sy * 0.5, 0.0), dtype=np.float64)


HRBridgeFixedStairCfg.function = hr_bridge_fixed_stair


def make_hr_bridge_fixed_terrain(step_heights=(0.05,), num_cols=16, first_riser_distance=1.10):
    return TerrainGeneratorCfg(
        seed=241,
        curriculum=False,
        difficulty_range=(0.0, 1.0),
        size=(8.0, 8.0),
        border_width=1.0,
        num_rows=len(step_heights),
        num_cols=num_cols,
        color_scheme="height",
        use_cache=False,
        sub_terrains={
            "hr_bridge_fixed_stair": HRBridgeFixedStairCfg(
                function=hr_bridge_fixed_stair,
                proportion=1.0,
                step_heights=tuple(step_heights),
                first_riser_from_spawn=first_riser_distance,
            )
        },
    )

