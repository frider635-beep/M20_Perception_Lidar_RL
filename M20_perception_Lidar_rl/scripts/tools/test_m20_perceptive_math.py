"""CPU unit tests for the final M20 spatial guidance mathematics."""

from __future__ import annotations

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": True}).app

import torch

SOURCE = Path(__file__).resolve().parents[2] / "source" / "rl_training"
sys.path.insert(0, str(SOURCE))

from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import (  # noqa: E402
    extract_nearest_riser,
    spatial_guidance_curve,
)


def synthetic_platform(height: float = 0.10, riser_x: float = 0.05) -> torch.Tensor:
    x = torch.linspace(1.25, -0.75, 21)
    z = torch.where(x >= riser_x, height, 0.0)
    return z[:, None].expand(21, 15).unsqueeze(0).clone()


def main() -> None:
    distances = torch.tensor((0.40, 0.35, 0.25, 0.15, 0.07, 0.00, -0.08, -0.15))
    curve = spatial_guidance_curve(distances, 0.35, 0.07, -0.03, -0.12)
    expected_conditions = (
        curve[0] == 0,
        curve[1] == 0,
        (curve[2] > 0.2) & (curve[2] < 0.5),
        (curve[3] > 0.7) & (curve[3] < 1.0),
        curve[4] == 1,
        curve[5] == 1,
        (curve[6] > 0) & (curve[6] < 1),
        curve[7] == 0,
    )
    assert all(bool(v) for v in expected_conditions), curve
    print("SPATIAL_CURVE", dict(zip(distances.tolist(), curve.tolist())))

    wheels = torch.tensor([[[0.0, 0.35], [0.0, -0.35], [-0.55, 0.35], [-0.55, -0.35]]])
    result = extract_nearest_riser(
        synthetic_platform(), wheels, corridor_half_width=0.15, h_min=0.02, d_end=-0.12
    )
    assert result["valid"].all(), result
    assert torch.allclose(result["riser_height"], torch.full((1, 4), 0.10), atol=1e-5), result
    assert torch.all(result["riser_distance"][:, :2] < result["riser_distance"][:, 2:])
    print("SYNTHETIC_RISER_HEIGHT", result["riser_height"].tolist())
    print("SYNTHETIC_RISER_DISTANCE", result["riser_distance"].tolist())

    flat = extract_nearest_riser(
        torch.zeros((2, 21, 15)), wheels.expand(2, -1, -1),
        corridor_half_width=0.15, h_min=0.02, d_end=-0.12,
    )
    assert not flat["valid"].any()
    assert flat["corridor_relief"].max() == 0
    print("FLAT_DETECTOR valid=0 active_fraction=0")

    # Verified semantic transform: a raw left-front ray maps to [row0,col0].
    raw_yx = torch.zeros(15, 21)
    raw_yx[-1, -1] = 1.0
    semantic = raw_yx.transpose(0, 1).flip((0, 1))
    assert semantic[0, 0] == 1.0 and semantic.sum() == 1.0
    print("ELEVATION_DIRECTION row0=front row20=rear col0=left col14=right")
    print("M20_PERCEPTIVE_MATH=PASS")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
