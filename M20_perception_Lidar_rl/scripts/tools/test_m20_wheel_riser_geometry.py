"""Exact 2-D wheel-circle versus stair-riser geometry acceptance."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": True}).app

from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import (
    distance_to_vertical_segment,
    minimum_safe_wheel_center_height,
    spatial_guidance_curve,
)


OUT = Path(__file__).resolve().parents[2] / "artifacts" / "wheel_riser_geometry"
OUT.mkdir(parents=True, exist_ok=True)
RADIUS = 0.09
DISTANCES = np.asarray((0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.0, -0.03, -0.05, -0.10, -0.15))
MARGINS = (0.0, 0.005, 0.01, 0.015, 0.02)
STARTS = (0.25, 0.35, 0.45)


def no_collision(d, z, z_lower, z_upper, r_clear):
    segment = distance_to_vertical_segment(d, z, z_lower, z_upper)
    vertical_ok = segment >= r_clear - 1.0e-7
    upper_ok = (d > 0.0) | (z >= z_upper + r_clear - 1.0e-7)
    return vertical_ok & upper_ok, segment


def sweep(height, margin, d_start=0.25):
    d = torch.tensor(DISTANCES, dtype=torch.float64)
    lower = torch.zeros_like(d)
    upper = torch.full_like(d, height)
    r_clear = RADIUS + margin
    guide = spatial_guidance_curve(d, d_start, 0.07, -0.03, -0.12)
    nominal = lower + RADIUS
    crossing = upper + r_clear
    preview = nominal + guide * (crossing - nominal)
    geom = minimum_safe_wheel_center_height(d, lower, upper, r_clear)
    ref = torch.maximum(preview, geom)
    valid, segment_distance = no_collision(d, ref, lower, upper, r_clear)
    return d, guide, preview, geom, ref, valid, segment_distance


def main():
    all_rows = []
    for height in (0.17, 0.20):
        d, guide, preview, geom, ref, valid, segment = sweep(height, 0.01)
        for values in zip(d, guide, preview, geom, ref, segment, valid):
            all_rows.append((height, 0.0, height, RADIUS, 0.10, *[float(x) for x in values[:-1]], int(values[-1])))
        assert bool(valid.all())
        print(f"{int(height * 100)}CM_GEOMETRIC_CLEARANCE=PASS")
        for index in (0, 6, 8, 9, 10, 13):
            print(
                f"HEIGHT={height:.2f} d={float(d[index]):+.3f} z_geom_required={float(geom[index]):.6f} "
                f"z_ref_world={float(ref[index]):.6f} segment_distance={float(segment[index]):.6f}"
            )
    csv_path = OUT / "geometry_sweep_17_20.csv"
    np.savetxt(
        csv_path,
        np.asarray(all_rows),
        delimiter=",",
        header="height_m,z_lower,z_upper,R,R_clear,d,g,z_preview,z_geom_required,z_ref_world,segment_distance,no_collision",
        comments="",
    )
    print("CSV=" + str(csv_path))

    margin_rows = []
    for height in (0.17, 0.20):
        for margin in MARGINS:
            d, guide, preview, geom, ref, valid, segment = sweep(height, margin)
            margin_rows.append((height, margin, ref.max().item(), segment.min().item(), int(valid.all())))
    np.savetxt(
        OUT / "margin_sweep.csv",
        np.asarray(margin_rows),
        delimiter=",",
        header="height_m,margin_m,max_z_ref_world,min_segment_distance,all_no_collision",
        comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, height in zip(axes, (0.17, 0.20)):
        dense_d = torch.linspace(0.50, -0.15, 400, dtype=torch.float64)
        for start in STARTS:
            guide = spatial_guidance_curve(dense_d, start, 0.07, -0.03, -0.12)
            nominal = RADIUS
            crossing = height + RADIUS + 0.01
            preview = nominal + guide * (crossing - nominal)
            geom = minimum_safe_wheel_center_height(
                dense_d, torch.zeros_like(dense_d), torch.full_like(dense_d, height), RADIUS + 0.01
            )
            ref = torch.maximum(preview, geom)
            axis.plot(dense_d, ref, label=f"d_start={start:.2f}")
        axis.invert_xaxis()
        axis.grid(alpha=0.3)
        axis.set(title=f"{int(height*100)} cm", xlabel="d (m)", ylabel="z_ref_world (m)")
        axis.legend()
    png_path = OUT / "preview_window_sweep.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print("PNG=" + str(png_path))
    print("RECOMMENDED_CLEARANCE_MARGIN=0.010")
    print("RECOMMENDED_D_START=0.350")
    print("OFFICIAL_PARAMS_UNCHANGED=True NO_PHASE=True NO_CONTACT_STATE=True")


try:
    main()
finally:
    simulation_app.close()
