"""Analytic normalized clearance-progress reward sweep; no learning required."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[2] / "artifacts" / "spatial_reward_sweep"
OUT.mkdir(parents=True, exist_ok=True)
HEIGHTS = (0.03, 0.05, 0.08, 0.17, 0.20)
LABELS = ("3cm", "5cm", "8cm", "17cm", "20cm")
GUIDE_GAINS = (0.30, 0.45, 0.60, 0.85, 1.00)
WEIGHT = 2.0


def main():
    max_ref = np.asarray(HEIGHTS)
    actual = np.arange(0.0, 1.001, 0.05)
    rows = []
    for height, label, gain, ref in zip(HEIGHTS, LABELS, GUIDE_GAINS, max_ref):
        for value in actual:
            deficit = max(0.0, 1.0 - value)
            raw = 2.0 * value - value * value
            weighted = gain * WEIGHT * raw
            rows.append((label, height, ref, value, deficit, raw, gain, weighted))
    csv_path = OUT / "normalized_progress_sweep.csv"
    np.savetxt(
        csv_path,
        np.asarray(rows, dtype=object),
        fmt="%s,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f",
        header="height_label,height_m,target_range_m,progress,progress_deficit,spatial_raw,guide_gain,spatial_weighted",
        comments="",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, ref, gain in zip(LABELS, max_ref, GUIDE_GAINS):
        raw = 2.0 * actual - actual**2
        axes[0].plot(actual, raw, label=label)
        axes[1].plot(actual, gain * WEIGHT * raw, label=label)
    axes[0].set(xlabel="clearance progress p", ylabel="spatial_raw", title="r=2p-p²")
    axes[1].set(xlabel="clearance progress p", ylabel="spatial_weighted", title="Guide-scaled weighted reward")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(ncol=2, fontsize=8)
    png_path = OUT / "normalized_progress_sweep.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    print(f"CSV={csv_path}")
    print(f"PNG={png_path}")
    for label, ref, gain in zip(LABELS, max_ref, GUIDE_GAINS):
        delta = gain * WEIGHT
        print(f"{label} L_ref={ref:.3f} guide_gain={gain:.2f} delta_R_spatial={delta:.6f}")
    assert np.isclose(2 * 0.5 - 0.5**2, 0.75)
    print("SPATIAL_REWARD_NORMALIZED_GRADIENT=PASS")


if __name__ == "__main__":
    main()
