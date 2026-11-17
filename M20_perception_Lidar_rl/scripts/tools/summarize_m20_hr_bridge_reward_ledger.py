#!/usr/bin/env python3
"""Summarize the additive M20 HR Bridge reward ledger."""

import argparse
import csv
import glob
import json
import math
import os
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def median(rows, key):
    return statistics.median(float(row[key]) for row in rows) if rows else 0.0


def summarize(rows, term_names):
    term_means = {name: mean(rows, f"episode_term__{name}") for name in term_names}
    s1_term_means = {name: mean(rows, f"s1_term__{name}") for name in term_names}
    return {
        "episodes": len(rows),
        "full_success_rate": mean(rows, "full_stair_success"),
        "hr_bridge_success_rate": mean(rows, "hr_bridge_success"),
        "perception_miss_rate": mean(rows, "perception_miss"),
        "joint_success_rate": mean(rows, "joint_success"),
        "s1_activation_rate": sum(float(row["s1_steps"]) > 0 for row in rows) / max(len(rows), 1),
        "mean_episode_return": mean(rows, "episode_return"),
        "median_episode_return": median(rows, "episode_return"),
        "mean_s1_return": mean(rows, "s1_return"),
        "median_s1_return": median(rows, "s1_return"),
        "mean_s1_steps": mean(rows, "s1_steps"),
        "median_s1_steps": median(rows, "s1_steps"),
        "mean_s1_base_vx": mean(rows, "s1_mean_base_vx"),
        "mean_final_progress": mean(rows, "final_root_progress"),
        "term_means": term_means,
        "s1_term_means": s1_term_means,
        "term_sum": sum(term_means.values()),
        "term_sum_error": sum(term_means.values()) - mean(rows, "episode_return"),
    }


def main():
    paths = sorted(glob.glob(os.path.join(args.input, "ledger_*.csv")))
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("no ledger rows found")
    term_names = [key.removeprefix("episode_term__") for key in rows[0] if key.startswith("episode_term__")]
    modes = sorted({row["mode"] for row in rows})
    report = {"files": len(paths), "rows": len(rows), "term_names": term_names, "modes": {}}
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        by_height = {}
        for height in sorted({float(row["height_m"]) for row in selected}):
            group = [row for row in selected if float(row["height_m"]) == height]
            by_height[f"{height:.2f}"] = summarize(group, term_names)
        report["modes"][mode] = {
            "all": summarize(selected, term_names),
            "s1_activated": summarize([row for row in selected if int(row["s1_steps"]) > 0], term_names),
            "success": summarize([row for row in selected if int(row["full_stair_success"])], term_names),
            "failure": summarize([row for row in selected if not int(row["full_stair_success"])], term_names),
            "by_height": by_height,
        }

    gamma = 0.99
    timeout_steps = round(4.0 / 0.02)
    miss_weight = 6.798566818237305
    discount = gamma ** timeout_steps
    report["event_accounting"] = {
        "HR_SUCCESS_WEIGHT": 3.3992834091186523,
        "PERCEPTION_MISS_WEIGHT": miss_weight,
        "JOINT_SUCCESS_WEIGHT": 20.617873382568362,
        "bridge_A": 24.017156791687015,
        "bridge_B": 3.3992834091186523,
        "bridge_C": -miss_weight,
        "bridge_D": 0.0,
        "bridge_timeout_current": 0.0,
        "timeout_steps_4s": timeout_steps,
        "gamma_power_timeout": discount,
        "entry_value_of_minus_miss_at_timeout": -miss_weight * discount,
        "timeout_penalty_needed_to_match_immediate_miss": -miss_weight / discount,
    }

    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "reward_ledger_summary.json")
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2)

    md = ["# M20 HR Bridge Reward Ledger", ""]
    md.append(f"Files: {len(paths)}; episodes: {len(rows)}; official terms: {len(term_names)}")
    md.append("")
    md.append("| Mode | Episodes | Success | Miss | Mean return | Median return | Mean S1 return | Mean S1 vx | Mean S1 steps | Sum error |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode in modes:
        item = report["modes"][mode]["all"]
        md.append(
            f"| {mode} | {item['episodes']} | {item['full_success_rate']:.3f} | "
            f"{item['perception_miss_rate']:.3f} | {item['mean_episode_return']:.3f} | "
            f"{item['median_episode_return']:.3f} | {item['mean_s1_return']:.3f} | "
            f"{item['mean_s1_base_vx']:.3f} | {item['mean_s1_steps']:.1f} | {item['term_sum_error']:.2e} |"
        )
    md.extend(["", "## Additive Official Term Means", ""])
    md.append("| Term | actor | bridge_hold | stop_on_s1 |")
    md.append("| --- | ---: | ---: | ---: |")
    for name in term_names:
        values = [report["modes"][mode]["all"]["term_means"][name] for mode in ("actor", "bridge_hold", "stop_on_s1")]
        md.append(f"| {name} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    md.extend(["", "## Timeout Discount", ""])
    event = report["event_accounting"]
    for key, value in event.items():
        md.append(f"- {key}: {value}")
    with open(os.path.join(args.output, "reward_ledger_report.md"), "w") as handle:
        handle.write("\n".join(md) + "\n")
    print(json.dumps({
        "files": len(paths), "rows": len(rows),
        "modes": {mode: report["modes"][mode]["all"] for mode in modes},
        "event_accounting": report["event_accounting"],
    }, indent=2))


if __name__ == "__main__":
    main()
