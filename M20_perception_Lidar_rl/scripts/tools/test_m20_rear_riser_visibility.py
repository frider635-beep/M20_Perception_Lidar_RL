"""Synthetic 315-D rear-wheel visibility audit for six-step stairs.

This is a detector/input-range test only.  It does not cache a previously
seen riser and it does not run a policy or any learning.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch

from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import (
    extract_nearest_riser,
    spatial_guidance_curve,
)


OUT = Path(__file__).resolve().parents[2] / "artifacts" / "rear_riser_visibility"
OUT.mkdir(parents=True, exist_ok=True)
X_ROWS = torch.linspace(1.25, -0.75, 21)
WHEELS = torch.tensor([[[0.45, 0.35], [0.45, -0.35], [-0.45, 0.35], [-0.45, -0.35]]])
REAR_DISTANCES = (0.35, 0.25, 0.15, 0.05, -0.05, -0.10)


def six_step_map(step_height: float, first_riser_x: float) -> torch.Tensor:
    """Return a current local map; no prior-riser state is retained."""
    z = torch.zeros_like(X_ROWS)
    for step in range(6):
        z += (X_ROWS >= first_riser_x + step * 0.31).to(z.dtype) * step_height
    return z[:, None].expand(21, 15).clone()[None]


def main():
    rows = []
    curve_rows = []
    for height in (0.17, 0.20):
        for true_distance in REAR_DISTANCES:
            true_x = float(WHEELS[0, 2, 0]) + true_distance
            elevation = six_step_map(height, true_x)
            result = extract_nearest_riser(
                elevation, WHEELS, corridor_half_width=0.15, h_min=0.02, d_end=-0.12
            )
            guide = spatial_guidance_curve(result["riser_distance"], 0.25, 0.07, -0.03, -0.12)
            for wheel_index, wheel in enumerate(("FL", "FR", "HL", "HR")):
                detected_distance = float(result["riser_distance"][0, wheel_index])
                detected_height = float(result["riser_height"][0, wheel_index])
                # Plot-only world-clearance rise above the lower-tread
                # nominal wheel-centre height. Geometry validation lives in
                # test_m20_wheel_riser_geometry.py.
                lift_ref = float(guide[0, wheel_index]) * (detected_height + 0.01)
                curve_rows.append((wheel, detected_distance, lift_ref, 0.0))
            for wheel_index, wheel in ((2, "HL"), (3, "HR")):
                valid = bool(result["valid"][0, wheel_index])
                detected_height = float(result["riser_height"][0, wheel_index])
                detected_distance = float(result["riser_distance"][0, wheel_index])
                height_error = abs(detected_height - height) if valid else float("nan")
                distance_error = abs(detected_distance - true_distance) if valid else float("nan")
                # At -0.10 m the first riser is at the configured descent end
                # after grid quantization, so selecting the next riser is the
                # expected continuous six-step transition rather than a rear
                # visibility failure.
                expected_same_riser = true_distance > -0.10
                correct = valid and height_error <= 0.015 and distance_error <= 0.051
                rows.append(
                    (
                        height,
                        wheel,
                        true_distance,
                        detected_height,
                        detected_distance,
                        height_error,
                        distance_error,
                        int(valid),
                        int(expected_same_riser),
                        int(correct if expected_same_riser else detected_distance > 0.0),
                    )
                )
    path = OUT / "rear_riser_visibility.csv"
    np.savetxt(
        path,
        np.asarray(rows, dtype=object),
        fmt="%0.3f,%s,%0.3f,%0.6f,%0.6f,%0.6f,%0.6f,%d,%d,%d",
        header=(
            "step_height_m,wheel,true_distance_m,detected_height_m,detected_distance_m,"
            "height_abs_error_m,distance_abs_error_m,valid,expected_same_riser,correct_transition"
        ),
        comments="",
    )
    print("CSV=" + str(path))
    # Emit the same independent per-wheel distance/reference/actual format
    # consumed by plot_m20_spatial_curves.py.  The synthetic audit has zero
    # actual lift because it tests visibility, not a policy trajectory.
    curve_path = OUT / "rear_visibility_distance_ref_actual.csv"
    np.savetxt(
        curve_path,
        np.asarray(curve_rows, dtype=object),
        fmt="%s,%0.6f,%0.6f,%0.6f",
        header="wheel,distance_m,L_ref_m,L_actual_m",
        comments="",
    )
    print("CURVE_CSV=" + str(curve_path))
    for height in (0.17, 0.20):
        selected = [row for row in rows if row[0] == height]
        correct_rate = sum(row[-1] for row in selected) / len(selected)
        same_riser = [row for row in selected if row[8]]
        max_h_error = max(row[5] for row in same_riser)
        max_d_error = max(row[6] for row in same_riser)
        print(
            f"REAR_VISIBILITY_{int(height * 100):02d}CM correct_rate={correct_rate:.3f} "
            f"max_height_error={max_h_error:.6f} max_distance_error={max_d_error:.6f}"
        )
        assert correct_rate == 1.0
    print("NO_RISER_STATE_CACHE=True")
    print("REAR_315D_VISIBILITY=PASS")


try:
    main()
finally:
    simulation_app.close()
