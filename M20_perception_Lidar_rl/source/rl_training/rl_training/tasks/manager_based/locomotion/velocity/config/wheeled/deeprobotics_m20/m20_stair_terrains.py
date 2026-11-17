"""Fixed, four-sided up/landing/down M20 staircase terrain used by V6."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


@configclass
class M20SquareStairCfg(SubTerrainBaseCfg):
    function = None
    step_height: float = MISSING
    step_depth: float = 0.31
    num_steps: int = 6
    central_flat_half_extent: float = 1.10
    top_landing_depth: float = 0.60
    down_step_depth: float = 0.31


def m20_square_stair_terrain(difficulty: float, cfg: M20SquareStairCfg):
    """Make complete square annuli: six steps up, a landing, then six down."""
    del difficulty
    height_min = cfg.step_height if isinstance(cfg.step_height, (float, int)) else cfg.step_height[0]
    if height_min <= 0 or cfg.step_depth <= 0 or cfg.down_step_depth <= 0 or cfg.num_steps != 6:
        raise ValueError("V6 stairs require six positive, fixed-depth levels")
    center = np.asarray((cfg.size[0] / 2.0, cfg.size[1] / 2.0), dtype=np.float64)
    meshes: list[trimesh.Trimesh] = [
        trimesh.creation.box(
            (2 * cfg.central_flat_half_extent, 2 * cfg.central_flat_half_extent, 0.02),
            trimesh.transformations.translation_matrix((*center, -0.01)),
        )
    ]

    def slab(width: float, depth: float, x: float, y: float, height: float):
        # A zero-height final descending tread is represented by a thin box
        # whose top surface is exactly at ground level.
        box_height = height if height > 0.0 else 0.02
        center_z = height / 2.0 if height > 0.0 else -0.01
        meshes.append(
            trimesh.creation.box(
                (width, depth, box_height),
                trimesh.transformations.translation_matrix((x, y, center_z)),
            )
        )

    def square_annulus(inner: float, outer: float, height: float):
        """Cover a complete square ring, including all four corners."""
        slab(2 * outer, outer - inner, center[0], center[1] + (inner + outer) / 2, height)
        slab(2 * outer, outer - inner, center[0], center[1] - (inner + outer) / 2, height)
        slab(outer - inner, 2 * inner, center[0] + (inner + outer) / 2, center[1], height)
        slab(outer - inner, 2 * inner, center[0] - (inner + outer) / 2, center[1], height)

    inner = cfg.central_flat_half_extent
    step_height = (
        float(cfg.step_height)
        if isinstance(cfg.step_height, (float, int))
        else float(np.random.uniform(cfg.step_height[0], cfg.step_height[1]))
    )
    for level in range(cfg.num_steps):
        outer = inner + cfg.step_depth
        h = (level + 1) * step_height
        square_annulus(inner, outer, h)
        inner = outer

    # The top landing prevents an immediate drop after the sixth riser.
    outer = inner + cfg.top_landing_depth
    h = cfg.num_steps * step_height
    square_annulus(inner, outer, h)
    inner = outer

    # Continue with six complete descending annuli.  The last tread is at z=0,
    # so it joins the surrounding terrain without a cliff or discontinuity.
    for level in range(cfg.num_steps):
        outer = inner + cfg.down_step_depth
        h = (cfg.num_steps - level - 1) * step_height
        square_annulus(inner, outer, h)
        inner = outer

    if inner > min(cfg.size) / 2.0 + 1e-6:
        raise ValueError(f"V6 up/landing/down staircase radius {inner:.2f} exceeds terrain size {cfg.size}")
    return meshes, np.asarray((*center, 0.0), dtype=np.float64)


M20SquareStairCfg.function = m20_square_stair_terrain

STAIR17_CFG = M20SquareStairCfg(function=m20_square_stair_terrain, proportion=0.45, step_height=0.17)
STAIR20_CFG = M20SquareStairCfg(function=m20_square_stair_terrain, proportion=0.45, step_height=0.20)
LOW_STAIR_CFG = M20SquareStairCfg(
    function=m20_square_stair_terrain, proportion=0.10, step_height=(0.08, 0.15)
)

V6_STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    curriculum=False,
    # 1.10 m center + 6x0.31 m up + 0.60 m landing + 6x0.31 m down
    # requires a 5.42 m half-extent.  Twelve meters leaves a safe flat margin.
    size=(12.0, 12.0),
    border_width=2.0,
    num_rows=1,
    num_cols=64,
    color_scheme="height",
    use_cache=False,
    sub_terrains={"stair17": STAIR17_CFG, "stair20": STAIR20_CFG, "stair08_15": LOW_STAIR_CFG},
)
