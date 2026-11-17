"""Monte-Carlo detector stress test for 5/8/17/20 cm risers."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.app import AppLauncher

_parser = __import__("argparse").ArgumentParser()
AppLauncher.add_app_launcher_args(_parser)
_args = _parser.parse_args()
_app = AppLauncher(_args).app

from rl_training.tasks.manager_based.locomotion.velocity.mdp.perceptive_spatial import extract_nearest_riser


OUT = Path(__file__).resolve().parents[2] / "artifacts" / "riser_detector_stress"
OUT.mkdir(parents=True, exist_ok=True)


def platform(height, x=0.05):
    xs = torch.linspace(1.25, -0.75, 21)
    z = torch.where(xs >= x, torch.tensor(height), torch.tensor(0.0))
    return z[:, None].expand(21, 15).clone()


def perturb(base, rng, noise, shift, dropout):
    out = base + torch.from_numpy(rng.normal(0.0, noise, base.shape).astype(np.float32))
    if shift:
        # Continuous X translation: +/-5 cm is half a grid cell and must not
        # be rounded to zero. Positive shift moves the observed edge forward.
        theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, shift]]], dtype=out.dtype)
        grid = F.affine_grid(theta, (1, 1, 21, 15), align_corners=True)
        out = F.grid_sample(
            out[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]
    if dropout:
        mask = torch.from_numpy(rng.random(base.shape) < dropout)
        out[mask] = 0.0
    return out


def main():
    rng = np.random.default_rng(42)
    wheels = torch.tensor([[[0.0, 0.35], [0.0, -0.35], [-0.45, 0.35], [-0.45, -0.35]]])
    rows = []
    # Include a true flat control so false positives are measured rather than
    # being reported as a hard-coded zero.
    for height in (0.0, 0.05, 0.08, 0.17, 0.20):
        for condition, noise, shift, dropout in (
            ("clean", 0.0, 0.0, 0.0),
            ("noise_1cm", 0.01, 0.0, 0.0),
            ("shift_p05", 0.0, 0.05, 0.0),
            ("shift_m05", 0.0, -0.05, 0.0),
            ("dropout_5pct", 0.0, 0.0, 0.05),
            ("noise_shift_dropout", 0.01, 0.05, 0.05),
        ):
            heights, distances, valid = [], [], []
            for _ in range(200):
                elevation = perturb(platform(height), rng, noise, shift, dropout)[None]
                result = extract_nearest_riser(elevation, wheels, corridor_half_width=0.15, h_min=0.02, d_end=-0.12)
                heights.append(float(result["riser_height"][0, 0]))
                distances.append(float(result["riser_distance"][0, 0]))
                valid.append(bool(result["valid"][0, 0]))
            valid_arr = np.asarray(valid)
            height_arr = np.asarray(heights)
            dist_arr = np.asarray(distances)
            miss = 1.0 - valid_arr.mean()
            if height == 0.0:
                h_err = np.nan
                x_err = np.nan
                false_positive = float(valid_arr.mean())
                miss = 0.0
            else:
                h_err = np.abs(height_arr[valid_arr] - height).mean() if valid_arr.any() else np.nan
                # Samples at +0.05 and -0.05 m straddle the physical boundary,
                # whose grid-resolved location is therefore x=0.0 m.
                x_err = np.abs(dist_arr[valid_arr] - 0.0).mean() if valid_arr.any() else np.nan
                false_positive = 0.0
            rows.append((height, condition, h_err, x_err, miss, false_positive))
    path = OUT / "detector_stress.csv"
    np.savetxt(path, np.asarray(rows, dtype=object), fmt="%0.3f,%s,%0.6f,%0.6f,%0.6f,%0.6f", header="height_m,condition,height_abs_error_m,distance_abs_error_m,miss_rate,false_positive_rate", comments="")
    print(f"CSV={path}")
    for row in rows:
        print("height=%.2f condition=%s h_err=%.5f x_err=%.5f miss=%.3f" % row[:5])


if __name__ == "__main__":
    try:
        main()
    finally:
        _app.close()
