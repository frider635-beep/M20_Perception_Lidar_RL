"""V10 unified perceptive curriculum terrains.

Rows encode the five requested height bands. Columns are deliberately grouped
so the runtime curriculum can switch topology without regenerating meshes.
"""

from __future__ import annotations

import copy

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass


HEIGHT_BANDS = ((0.05, 0.08), (0.08, 0.11), (0.11, 0.14), (0.14, 0.17), (0.17, 0.20))


@configclass
class V10StepTerrainCfg(SubTerrainBaseCfg):
    function = None
    full_staircase: bool = False
    step_depth: float = 0.31
    num_steps: int = 6
    first_riser_from_spawn: float = 1.10
    top_landing_depth: float = 1.20
    spawn_x: float = 1.25
    height_override: float | None = None


def _height_from_difficulty(difficulty: float) -> float:
    scaled = float(np.clip(difficulty, 0.0, 1.0)) * len(HEIGHT_BANDS)
    index = min(int(scaled), len(HEIGHT_BANDS) - 1)
    fraction = np.clip(scaled - index, 0.0, 1.0)
    lower, upper = HEIGHT_BANDS[index]
    return float(lower + fraction * (upper - lower))


def v10_step_terrain(difficulty: float, cfg: V10StepTerrainCfg):
    """Generate either one deployable riser or a six-riser staircase."""
    height = float(cfg.height_override) if cfg.height_override is not None else _height_from_difficulty(difficulty)
    size_x, size_y = cfg.size
    first_x = cfg.spawn_x + cfg.first_riser_from_spawn
    count = cfg.num_steps if cfg.full_staircase else 1
    landing_start = first_x + count * cfg.step_depth
    if landing_start + cfg.top_landing_depth > size_x - 0.25:
        raise ValueError("V10 stair does not fit the terrain patch")

    def box(x0: float, x1: float, top_z: float) -> trimesh.Trimesh:
        thickness = max(top_z, 0.02)
        return trimesh.creation.box(
            (x1 - x0, size_y, thickness),
            trimesh.transformations.translation_matrix(
                ((x0 + x1) * 0.5, size_y * 0.5, top_z - thickness * 0.5)
            ),
        )

    meshes = [box(0.0, size_x, 0.0)]
    for level in range(count):
        x0 = first_x + level * cfg.step_depth
        x1 = first_x + (level + 1) * cfg.step_depth
        meshes.append(box(x0, x1, (level + 1) * height))
    meshes.append(box(landing_start, size_x, count * height))
    origin = np.asarray((cfg.spawn_x, size_y * 0.5, 0.0), dtype=np.float64)
    return meshes, origin


V10StepTerrainCfg.function = v10_step_terrain

FLAT = terrain_gen.MeshPlaneTerrainCfg(proportion=0.125)
BOXES = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["boxes"])
BOXES.proportion = 0.125
RANDOM_ROUGH = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["random_rough"])
RANDOM_ROUGH.proportion = 0.125
SINGLE_STEP = V10StepTerrainCfg(function=v10_step_terrain, proportion=0.50, full_staircase=False)
FULL_STAIR = V10StepTerrainCfg(function=v10_step_terrain, proportion=0.125, full_staircase=True)


V10_UNIFIED_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=1010,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=32,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "flat": FLAT,
        "boxes": BOXES,
        "random_rough": RANDOM_ROUGH,
        "single_step": SINGLE_STEP,
        "full_staircase": FULL_STAIR,
    },
)

# Column ranges follow TerrainGenerator's cumulative-proportion assignment.
V10_FLAT_COLUMNS = tuple(range(0, 4))
V10_ROUGH_COLUMNS = tuple(range(4, 12))
V10_SINGLE_COLUMNS = tuple(range(12, 28))
V10_FULL_COLUMNS = tuple(range(28, 32))


__all__ = [
    "HEIGHT_BANDS",
    "V10_UNIFIED_TERRAINS_CFG",
    "V10_FLAT_COLUMNS",
    "V10_ROUGH_COLUMNS",
    "V10_SINGLE_COLUMNS",
    "V10_FULL_COLUMNS",
]
