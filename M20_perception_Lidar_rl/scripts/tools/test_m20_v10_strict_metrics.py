#!/usr/bin/env python3
"""Pure semantic gates for V10 topology-strict events and encounter metrics."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py"
RUNNER = ROOT / "source/rl_training/rl_training/v10_unified_runner.py"
ENV_CFG = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v10_unified_env_cfg.py"
if not SOURCE.is_file():
    # Isolated pre-sync layout used by this task's /tmp gate.
    ROOT = Path(__file__).resolve().parents[1]
    SOURCE = ROOT / "source/v10_unified.py"
    RUNNER = ROOT / "source/v10_unified_runner.py"
    ENV_CFG = ROOT / "source/v10_unified_env_cfg.py"


def load_functions():
    tree = ast.parse(SOURCE.read_text())
    names = {
        "v10_route_events", "v10_metric_event_masks", "v10_recent_window_counts",
        "v10_topology_ids", "v10_height_bucket", "v10_official_promotion_diagnostics",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "torch": torch,
        "V10_TOPOLOGY_FLAT": 0,
        "V10_TOPOLOGY_ROUGH": 1,
        "V10_TOPOLOGY_SINGLE": 2,
        "V10_TOPOLOGY_FULL": 3,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def b(value):
    return torch.tensor([value], dtype=torch.bool)


def f(value):
    return torch.tensor([value], dtype=torch.float32)


def route(fn, topology, first=False, miss=False, base=0.0, hr=0.0):
    return fn(
        b(topology == 2), b(topology == 3), b(True), f(base), f(hr), b(True),
        b(first), b(miss), b(False), b(False), b(False), 0.31, 0.25,
    )


def masks(fn, topology, *, lock=False, first=False, miss=False, single=False, full=False, joint=False, blind=False):
    return fn(
        torch.tensor([topology]), b(lock), b(first), b(miss), b(single), b(full), b(joint), b(blind)
    )


def count(mask):
    return int(mask.sum().item())


def main():
    fn = load_functions()
    route_fn = fn["v10_route_events"]
    masks_fn = fn["v10_metric_event_masks"]

    # TEST 1: a single-step first-step success cannot emit full or joint.
    r1 = route(route_fn, 2, first=True, base=0.0, hr=0.0)
    m1 = masks(masks_fn, 2, lock=True, first=True, full=bool(r1["full_event"]), joint=bool(r1["joint_event"]))
    test1 = (
        count(m1["single_first_success"]) == 1
        and count(m1["full_success"]) == 0
        and count(m1["joint_success"]) == 0
    )

    # TEST 2: single route completion is explicit and never aliases full/joint.
    r2 = route(route_fn, 2, first=True, base=0.30, hr=0.10)
    m2 = masks(
        masks_fn, 2, single=bool(r2["single_event"]), full=bool(r2["full_event"]),
        joint=bool(r2["joint_event"]),
    )
    test2 = (
        count(m2["single_complete"]) == 1
        and count(m2["full_success"]) == 0
        and count(m2["joint_success"]) == 0
    )

    # TEST 3: full first-step success without L6 completion is not full/joint.
    r3 = route(route_fn, 3, first=True, base=0.50, hr=0.30)
    test3 = not bool(r3["full_event"]) and not bool(r3["joint_event"])

    # TEST 4: full first-step success plus six-step completion emits full+joint.
    r4 = route(route_fn, 3, first=True, base=1.81, hr=1.61)
    m4 = masks(masks_fn, 3, first=True, full=bool(r4["full_event"]), joint=bool(r4["joint_event"]))
    test4 = (
        count(m4["full_first_success"]) == 1
        and count(m4["full_success"]) == 1
        and count(m4["joint_success"]) == 1
    )

    # TEST 5: miss then L6 completion emits full+blind, never joint.
    r5 = route(route_fn, 3, first=False, miss=True, base=1.81, hr=1.61)
    m5 = masks(
        masks_fn, 3, miss=True, full=bool(r5["full_event"]),
        joint=bool(r5["joint_event"]), blind=bool(r5["blind_only_event"]),
    )
    test5 = (
        count(m5["full_miss"]) == 1
        and count(m5["full_success"]) == 1
        and count(m5["blind_only_success"]) == 1
        and count(m5["joint_success"]) == 0
    )

    # TEST 6/7: rough/flat locks are false triggers, never stair denominators.
    m6 = masks(masks_fn, 1, lock=True)
    test6 = (
        count(m6["rough_false_trigger"]) == 1
        and count(m6["all_stair_encounter"]) == 0
        and count(m6["single_encounter"]) == 0
        and count(m6["full_encounter"]) == 0
    )
    m7 = masks(masks_fn, 0, lock=True)
    test7 = (
        count(m7["flat_false_trigger"]) == 1
        and count(m7["all_stair_encounter"]) == 0
        and count(m7["single_encounter"]) == 0
        and count(m7["full_encounter"]) == 0
    )

    # Last 100/500 are encounter sequence windows, not iterations or control steps.
    ring = {
        "sequence": torch.arange(100, 600, dtype=torch.long),
        "bucket": torch.zeros(500, dtype=torch.long),
        "joint_success": torch.arange(100, 600) >= 500,
    }
    n100, d100 = fn["v10_recent_window_counts"](ring, 600, 100, field="joint_success")
    n500, d500 = fn["v10_recent_window_counts"](ring, 600, 500, field="joint_success")
    recent100 = (int(n100), int(d100)) == (100, 100)
    recent500 = (int(n500), int(d500)) == (100, 500)

    # Official promotion/demotion are passed through byte-for-byte on every
    # terrain. FIRST only labels the former level>=2 gate opportunities.
    move_up, move_down, blind, opportunity, gated_opportunity, would_fail = fn[
        "v10_official_promotion_diagnostics"
    ](
        torch.tensor([True, True, True, True, True, False]),
        torch.tensor([False, False, False, False, False, True]),
        torch.tensor([True, True, True, True, False, False]),
        torch.tensor([0, 1, 2, 3, 4, 4]),
        torch.tensor([False, False, False, True, False, False]),
    )
    low_stair_official = move_up[:2].tolist() == [True, True]
    high_stair_official = move_up[2:4].tolist() == [True, True]
    non_stair_official = move_up[4:].tolist() == [True, False]
    down_untouched = move_down.tolist() == [False, False, False, False, False, True]
    official_curriculum = all((
        low_stair_official, high_stair_official, non_stair_official, down_untouched,
        blind.tolist() == [True, True, True, False, False, False],
        opportunity.tolist() == [True, True, True, True, False, False],
        gated_opportunity.tolist() == [False, False, True, True, False, False],
        would_fail.tolist() == [False, False, True, False, False, False],
    ))

    route_source = inspect.getsource(route_fn)
    promotion_source = inspect.getsource(fn["v10_official_promotion_diagnostics"])
    source_text = SOURCE.read_text()
    env_cfg_source = ENV_CFG.read_text()
    full_only = "full_mask & target_valid" in route_source
    joint_only = "full_mask & first_step_success_latched" in route_source
    runner_source = RUNNER.read_text()
    official_unchanged = "super().log(locs" in runner_source
    promotion_gate_removed = "perception_gate_start_level" not in env_cfg_source
    official_curriculum_registered = "CurrTerm(func=mdp.v10_official_terrain_levels)" in env_cfg_source
    official_distance_semantics = (
        "distance > terrain.cfg.terrain_generator.size[0] / 2" in source_text
        and "* env.max_episode_length_s * 0.5" in source_text
        and ") & ~official_up" in source_text
    )
    no_perception_demotion = (
        "move_down = official_down" in promotion_source
        and 'move_down = state["perception_miss_latched"]' not in source_text
    )
    no_joint_promotion = "joint_success" not in promotion_source
    official_gait_mapping = (
        "update_gait_level_from_terrain_mean(mean_level)" in source_text
        and "CurrTerm(func=mdp.gait_level_curve)" in env_cfg_source
    )
    curriculum_tags = all(tag in runner_source for tag in (
        "Curriculum/StairTerrainLevelMean",
        "Curriculum/NonStairTerrainLevelMean",
        "Curriculum/StairOfficialUpRate",
        "Curriculum/StairPromotionRate",
        "Curriculum/StairPromotionBlockedByFirstRate",
    ))

    results = {
        "TEST1_SINGLE_FIRST_ONLY": test1,
        "TEST2_SINGLE_COMPLETE_NOT_FULL": test2,
        "TEST3_FULL_FIRST_WITHOUT_L6": test3,
        "TEST4_FULL_FIRST_AND_L6": test4,
        "TEST5_FULL_MISS_BLIND_ONLY": test5,
        "TEST6_ROUGH_DENOMINATOR_ISOLATION": test6,
        "TEST7_FLAT_DENOMINATOR_ISOLATION": test7,
        "FULL_EVENT_FULL_ONLY": full_only,
        "JOINT_EVENT_FULL_ONLY": joint_only,
        "ROUGH_NOT_IN_STAIR_DENOMINATOR": test6,
        "SINGLE_NOT_FULL_SUCCESS": test1 and test2,
        "RECENT100_METRICS": recent100,
        "RECENT500_METRICS": recent500,
        "OFFICIAL_CURRICULUM_ALL_LEVELS": official_curriculum,
        "LEVEL_0_1_STAIR_USE_OFFICIAL_UP": low_stair_official,
        "LEVEL_GE_2_STAIR_USE_OFFICIAL_UP": high_stair_official,
        "NON_STAIR_OFFICIAL_UP_UNCHANGED": non_stair_official,
        "MOVE_DOWN_IGNORES_FIRST_MISS": down_untouched,
        "PERCEPTION_PROMOTION_GATE_REMOVED": promotion_gate_removed,
        "OFFICIAL_CURRICULUM_REGISTERED": official_curriculum_registered,
        "OFFICIAL_DISTANCE_SEMANTICS": official_distance_semantics,
        "NO_PERCEPTION_FORCED_DEMOTION": no_perception_demotion,
        "NO_JOINT_PROMOTION": no_joint_promotion,
        "OFFICIAL_GAIT_MAPPING": official_gait_mapping,
        "CURRICULUM_TENSORBOARD_TAGS": curriculum_tags,
        "OFFICIAL_METRICS_UNCHANGED": official_unchanged,
    }
    for name, passed in results.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
