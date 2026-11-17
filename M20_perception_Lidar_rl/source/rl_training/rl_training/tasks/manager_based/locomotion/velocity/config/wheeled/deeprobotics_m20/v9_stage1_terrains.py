"""Independent V9 Stage-1 terrain: approach plus one ascending riser.

The terrain is deliberately separate from the official six-level training
terrain.  Difficulty is quantized to the four requested first-riser heights;
the actor still receives the same 21x15 elevation sensor as the perceptive
task.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


@configclass
class V9Stage1RiserCfg(SubTerrainBaseCfg):
    function = None
    step_heights: tuple[float, ...] = (0.05, 0.08, 0.11, 0.14)
    step_depth: float = 0.31
    first_riser_from_spawn: float = 1.10
    top_landing_depth: float = 1.20
    spawn_x: float = 1.25


def v9_stage1_riser_terrain(difficulty: float, cfg: V9Stage1RiserCfg):
    """Generate flat approach, one riser, and a finite upper landing."""
    heights = tuple(float(v) for v in cfg.step_heights)
    index = min(int(float(difficulty) * len(heights)), len(heights) - 1)
    height = heights[index]
    sx, sy = cfg.size
    first_x = cfg.spawn_x + cfg.first_riser_from_spawn
    top_end = first_x + cfg.step_depth + cfg.top_landing_depth
    if first_x < 0.5 or top_end > sx - 0.25:
        raise ValueError("V9 Stage-1 riser does not fit terrain patch")

    def box(x0: float, x1: float, top_z: float) -> trimesh.Trimesh:
        thickness = max(top_z, 0.02)
        return trimesh.creation.box(
            (x1 - x0, sy, thickness),
            trimesh.transformations.translation_matrix(
                ((x0 + x1) * 0.5, sy * 0.5, top_z - thickness * 0.5)
            ),
        )

    meshes = [box(0.0, sx, 0.0)]
    meshes.append(box(first_x, first_x + cfg.step_depth + cfg.top_landing_depth, height))
    return meshes, np.asarray((cfg.spawn_x, sy * 0.5, 0.0), dtype=np.float64)


V9Stage1RiserCfg.function = v9_stage1_riser_terrain

V9_STAGE1_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=9109,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    size=(8.0, 8.0),
    border_width=1.0,
    num_rows=4,
    num_cols=32,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "v9_stage1_riser": V9Stage1RiserCfg(
            function=v9_stage1_riser_terrain, proportion=1.0
        )
    },
)
