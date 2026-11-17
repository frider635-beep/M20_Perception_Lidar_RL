#!/usr/bin/env python3
"""Pure unit and source-isolation gates for V10 deployable lifecycle."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = Path(__file__).resolve().parents[2] / (
    "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py"
)


def load_functions():
    tree = ast.parse(SOURCE.read_text())
    names = {"v10_event_pulses", "v10_deployable_lifecycle_step", "v10_delay_pipeline_step"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def make_state():
    zf = lambda: torch.zeros(1)
    zb = lambda: torch.zeros(1, dtype=torch.bool)
    zl = lambda: torch.zeros(1, dtype=torch.long)
    return {
        "valid_count": zl(), "invalid_count": zl(), "first_riser_locked": zb(),
        "first_riser_consumed": zb(), "locked_h": zf(), "locked_d_raw": zf(),
        "locked_d_filtered": zf(), "first_riser_visible": zb(), "pass_count": zl(),
    }


def cfg():
    return SimpleNamespace(
        confirmation_frames=2, track_height_tolerance=.04, track_max_forward_jump=.12,
        hysteresis_frames=2, distance_filter_alpha=.5, pgtt_pass_exit_distance=-.15,
        pgtt_pass_lost_max_distance=-.05, pass_confirmation_frames=2,
        anticipation_distance_min=.02, anticipation_distance_max=.70, forward_command_threshold=.05,
    )


def same_state(before, after):
    return all(torch.equal(before[key], after[key]) for key in before)


def main():
    fn = load_functions()
    lifecycle = fn["v10_deployable_lifecycle_step"]
    pulses = fn["v10_event_pulses"]
    delay = fn["v10_delay_pipeline_step"]
    state = make_state()
    for d in (.60, .50):
        out = lifecycle(state, cfg(), torch.tensor([True]), torch.tensor([.17]), torch.tensor([d]), torch.tensor([.4]))
    assert bool(state["first_riser_locked"] and out["b_raw"])
    baseline = {key: value.clone() for key, value in state.items()}
    pulses(*(torch.tensor([value]) for value in (True, True, True, True, False, False, False, False)))
    success_isolated = same_state(baseline, state) and bool(out["b_raw"])
    pulses(*(torch.tensor([value]) for value in (True, False, False, False, False, False, False, True)))
    miss_isolated = same_state(baseline, state) and bool(out["b_raw"])

    ring = torch.zeros((1, 3, 3))
    index = 0
    observed = []
    for raw_b in (1.0, 1.0, 0.0, 0.0, 0.0):
        feature = torch.tensor([[raw_b, .17 if raw_b else 0.0, .30 if raw_b else 0.0]])
        actor, index = delay(ring, index, feature, 2)
        observed.append(float(actor[0, 0]))
    delay_ok = observed == [0.0, 0.0, 1.0, 1.0, 0.0]

    track = make_state()
    switch_count = 0
    last = None
    for d in (.60, .50, .40, .30, .20, .10, .00, -.10, .10):
        was_locked = bool(track["first_riser_locked"])
        out = lifecycle(track, cfg(), torch.tensor([True]), torch.tensor([.17]), torch.tensor([d]), torch.tensor([.4]))
        current = float(track["locked_d_raw"])
        if was_locked and last is not None and not bool(track["first_riser_consumed"]) and current - last > .12:
            switch_count += 1
        if bool(track["first_riser_locked"]):
            last = current
    lock_ok = switch_count == 0 and float(track["locked_d_raw"]) <= -.099
    for _ in range(3):
        out = lifecycle(track, cfg(), torch.tensor([True]), torch.tensor([.17]), torch.tensor([-.20]), torch.tensor([.4]))
    consumed = bool(track["first_riser_consumed"])
    l2_l6 = []
    for d in (.10, .10, .10, .10, .10):
        out = lifecycle(track, cfg(), torch.tensor([True]), torch.tensor([.17]), torch.tensor([d]), torch.tensor([.4]))
        l2_l6.append(bool(out["b_raw"]))
    l2_ok = consumed and not any(l2_l6)

    lifecycle_source = inspect.getsource(lifecycle)
    banned = ("body_pos_w", "contact", "first_step_success", "perception_miss", "vertical_contact")
    source_ok = not any(term in lifecycle_source for term in banned)
    whole_source = SOURCE.read_text()
    no_flush = "_v10_feature_ring[consumed_event]" not in whole_source

    print(f"TEST1_TEACHER_SUCCESS_ISOLATION={'PASS' if success_isolated else 'FAIL'}")
    print(f"TEST2_TEACHER_MISS_ISOLATION={'PASS' if miss_isolated else 'FAIL'}")
    print(f"TEST3_DELAY_SEQUENCE={observed}")
    print(f"DELAY_BUFFER_SEMANTICS={'PASS' if delay_ok and no_flush else 'FAIL'}")
    print(f"TARGET_SWITCH_COUNT={switch_count}")
    print(f"FIRST_RISER_LOCK={'PASS' if lock_ok else 'FAIL'}")
    print(f"L2_L6_B_RAW={l2_l6}")
    print(f"L2_L6_REACTIVATION_BLOCK={'PASS' if l2_ok else 'FAIL'}")
    print(f"TEACHER_LIFECYCLE_ISOLATION={'PASS' if success_isolated and miss_isolated else 'FAIL'}")
    print(f"DEPLOYABLE_PERCEPTION_LIFECYCLE={'PASS' if source_ok else 'FAIL'}")
    passed = success_isolated and miss_isolated and delay_ok and no_flush and lock_ok and l2_ok and source_ok
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
