"""Official M20 stair training distribution plus fixed delivery terrains."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

import copy


@configclass
class M20PerceptiveStairCfg(SubTerrainBaseCfg):
    function = None
    step_heights: tuple[float, ...] = MISSING
    step_depth: float = 0.31
    num_steps: int = 6
    first_riser_from_spawn: float = 1.10
    top_landing_depth: float = 1.20
    spawn_x: float = 1.25


def m20_perceptive_stair_terrain(difficulty: float, cfg: M20PerceptiveStairCfg):
    """Build a flat or six-riser staircase spanning the complete patch width.

    Isaac Lab generates curriculum difficulty in row bands.  Quantizing the
    normalized difficulty therefore maps rows exactly to the configured
    height ledger while retaining its normal terrain-origin curriculum.
    """
    level = min(int(float(difficulty) * len(cfg.step_heights)), len(cfg.step_heights) - 1)
    step_height = float(cfg.step_heights[level])
    if cfg.num_steps != 6 or cfg.step_depth <= 0.0:
        raise ValueError("perceptive stairs require six positive-depth treads")
    size_x, size_y = cfg.size
    first_riser_x = cfg.spawn_x + cfg.first_riser_from_spawn
    last_top_end = first_riser_x + cfg.num_steps * cfg.step_depth + cfg.top_landing_depth
    if first_riser_x < 1.0 or last_top_end > size_x - 0.5:
        raise ValueError("stair approach/landing does not fit the terrain patch")

    def box(x0: float, x1: float, height: float) -> trimesh.Trimesh:
        thickness = height if height > 0.0 else 0.02
        top_z = height if height > 0.0 else 0.0
        return trimesh.creation.box(
            (x1 - x0, size_y, thickness),
            trimesh.transformations.translation_matrix(((x0 + x1) / 2.0, size_y / 2.0, top_z - thickness / 2.0)),
        )

    meshes = [box(0.0, size_x, 0.0)]
    if step_height > 0.0:
        # Nested slabs cover the entire width; lateral bypass is impossible.
        for step in range(cfg.num_steps):
            x0 = first_riser_x + step * cfg.step_depth
            x1 = first_riser_x + (step + 1) * cfg.step_depth
            meshes.append(box(x0, x1, (step + 1) * step_height))
        meshes.append(box(first_riser_x + cfg.num_steps * cfg.step_depth, size_x, cfg.num_steps * step_height))
    return meshes, np.asarray((cfg.spawn_x, size_y / 2.0, 0.0), dtype=np.float64)


M20PerceptiveStairCfg.function = m20_perceptive_stair_terrain

# Fixed full-width stairs are delivery/acceptance terrains only. They are not
# mixed into the PPO training distribution.
PERCEPTIVE_STAIR_CFG = M20PerceptiveStairCfg(
    function=m20_perceptive_stair_terrain,
    proportion=1.0,
    step_heights=(0.00, 0.03, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20),
)

M20_DELIVERY_ACCEPTANCE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    size=(8.0, 5.0),
    border_width=2.0,
    num_rows=8,
    num_cols=32,
    color_scheme="height",
    use_cache=False,
    sub_terrains={"full_width_ascending_stairs": PERCEPTIVE_STAIR_CFG},
)

# Deterministic two-type delivery suite.  Unlike the legacy eight-row ledger
# above, each sub-terrain has exactly one possible riser height, so runtime
# tests can explicitly select 17 or 20 cm without relying on curriculum-row
# difficulty sampling details.
DELIVERY_STAIR_17_CFG = M20PerceptiveStairCfg(
    function=m20_perceptive_stair_terrain,
    proportion=0.5,
    step_heights=(0.17,),
)
DELIVERY_STAIR_20_CFG = M20PerceptiveStairCfg(
    function=m20_perceptive_stair_terrain,
    proportion=0.5,
    step_heights=(0.20,),
)
M20_DELIVERY_17_20_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=False,
    difficulty_range=(0.0, 1.0),
    size=(8.0, 5.0),
    border_width=2.0,
    num_rows=1,
    num_cols=2,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "stairs_17cm_x_31cm": DELIVERY_STAIR_17_CFG,
        "stairs_20cm_x_31cm": DELIVERY_STAIR_20_CFG,
    },
)


def _single_delivery_cfg(name: str, stair_cfg: M20PerceptiveStairCfg) -> TerrainGeneratorCfg:
    single = copy.deepcopy(stair_cfg)
    single.proportion = 1.0
    return TerrainGeneratorCfg(
        seed=42,
        curriculum=False,
        difficulty_range=(0.0, 1.0),
        size=(8.0, 5.0),
        border_width=2.0,
        num_rows=1,
        num_cols=1,
        color_scheme="height",
        use_cache=False,
        sub_terrains={name: single},
    )


M20_DELIVERY_17_TERRAINS_CFG = _single_delivery_cfg("stairs_17cm_x_31cm", DELIVERY_STAIR_17_CFG)
M20_DELIVERY_20_TERRAINS_CFG = _single_delivery_cfg("stairs_20cm_x_31cm", DELIVERY_STAIR_20_CFG)

# The actual perceptive training distribution reuses the official Isaac Lab
# MeshPyramidStairsTerrainCfg objects byte-for-byte in their geometry fields:
# 8x8 m patch, 1 m internal border, 3 m platform, 0.30 m tread, 0.05-0.23 m
# continuous difficulty. These values imply exactly six surrounding steps.
OFFICIAL_PYRAMID_STAIRS_CFG = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs"])
OFFICIAL_PYRAMID_STAIRS_INV_CFG = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["pyramid_stairs_inv"])
# With Isaac Lab's default center spawn, ``pyramid_stairs`` slopes downward
# when moving away from the center while ``pyramid_stairs_inv`` slopes upward.
# This task commands +X away from the center, so the official inverted class is
# the actual ascending staircase.  Do not spend half of PPO on descending.
OFFICIAL_PYRAMID_STAIRS_CFG.proportion = 0.0
OFFICIAL_PYRAMID_STAIRS_INV_CFG.proportion = 1.0

M20_PERCEPTIVE_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    difficulty_range=tuple(ROUGH_TERRAINS_CFG.difficulty_range),
    size=tuple(ROUGH_TERRAINS_CFG.size),
    border_width=ROUGH_TERRAINS_CFG.border_width,
    num_rows=ROUGH_TERRAINS_CFG.num_rows,
    num_cols=32,
    horizontal_scale=ROUGH_TERRAINS_CFG.horizontal_scale,
    vertical_scale=ROUGH_TERRAINS_CFG.vertical_scale,
    slope_threshold=ROUGH_TERRAINS_CFG.slope_threshold,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_inv": OFFICIAL_PYRAMID_STAIRS_INV_CFG,
    },
)
