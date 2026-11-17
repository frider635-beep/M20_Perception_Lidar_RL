#!/usr/bin/env python3
"""Audit a deterministic deployable S1 exit rule against simulator labels."""

import argparse
import csv
import json
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--min_success_rows", type=int, default=100)
args = parser.parse_args()


def main():
    with open(args.input) as handle:
        rows = list(csv.DictReader(handle))
    teacher = [
        int(row["teacher_clearance"]) and int(row["teacher_crossing"])
        and int(row["teacher_landing"]) and int(row["teacher_support"])
        and not int(row["teacher_contact_miss"])
        for row in rows
    ]
    # Candidate rule uses only deployable values. Thresholds are deliberately
    # evaluated, not installed into the state machine by this audit tool.
    predicted = []
    for row in rows:
        gravity_xy = math.hypot(float(row["gravity_x"]), float(row["gravity_y"]))
        hr_dq = math.sqrt(sum(float(row[f"hr_dq{i}"]) ** 2 for i in range(3))) if "hr_dq0" in row else math.inf
        predicted.append(
            float(row["riser_distance"]) < -0.03
            and gravity_xy < 0.35
            and float(row["gravity_z"]) < -0.75
            and hr_dq < 1.0
        )
    tp = sum(p and t for p, t in zip(predicted, teacher))
    fp = sum(p and not t for p, t in zip(predicted, teacher))
    fn = sum((not p) and t for p, t in zip(predicted, teacher))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    enough = sum(teacher) >= args.min_success_rows
    passed = enough and precision >= 0.99 and recall >= 0.95
    result = {
        "rows": len(rows), "teacher_success_rows": sum(teacher),
        "precision": precision, "recall": recall,
        "enough_success_rows": enough,
        "DEPLOYABLE_HANDOFF_AUDIT": "PASS" if passed else "FAIL",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
