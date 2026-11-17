"""Export independent FL/FR/HL/HR distance/reference/actual CSV and PNG."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[2] / "artifacts" / "spatial_curves"
WHEELS = ("FL", "FR", "HL", "HR")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="runtime CSV with wheel,distance_m,L_ref_m,L_actual_m columns")
    args = parser.parse_args()
    source = Path(args.csv)
    data = np.genfromtxt(source, delimiter=",", names=True, dtype=None, encoding="utf-8")
    required = {"wheel", "distance_m", "L_ref_m", "L_actual_m"}
    if not data.dtype.names or not required.issubset(data.dtype.names):
        raise ValueError(f"expected columns {sorted(required)}, got {data.dtype.names}")
    OUT.mkdir(parents=True, exist_ok=True)
    for wheel in WHEELS:
        rows = data[data["wheel"] == wheel]
        if rows.size == 0:
            raise ValueError(f"missing {wheel} samples")
        rows = np.atleast_1d(rows)
        order = np.argsort(rows["distance_m"])[::-1]
        distance = rows["distance_m"][order]
        ref = rows["L_ref_m"][order]
        actual = rows["L_actual_m"][order]
        csv_path = OUT / f"{wheel}_distance_ref_actual.csv"
        np.savetxt(
            csv_path,
            np.column_stack((distance, ref, actual)),
            delimiter=",",
            header="distance_m,L_ref_m,L_actual_m",
            comments="",
        )
        fig, axis = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        axis.plot(distance, ref, label="L_ref")
        axis.plot(distance, actual, label="L_actual")
        axis.set(xlabel="distance to selected riser (m)", ylabel="wheel lift (m)", title=wheel)
        axis.invert_xaxis()
        axis.grid(alpha=0.3)
        axis.legend()
        png_path = OUT / f"{wheel}_distance_ref_actual.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        print(f"{wheel}_CSV={csv_path}")
        print(f"{wheel}_PNG={png_path}")


if __name__ == "__main__":
    main()
