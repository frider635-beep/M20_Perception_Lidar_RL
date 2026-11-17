#!/usr/bin/env python3
"""Pure-math PGTT geometry regression; does not import or launch Isaac Lab."""

from __future__ import annotations


D_START, D_PEAK, D_HOLD, D_END = 0.35, 0.07, -0.03, -0.12
CLEARANCE_MARGIN = 0.01


def smoothstep01(value: float) -> float:
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


def timing_progress(distance: float) -> float:
    if distance > D_PEAK:
        return smoothstep01((D_START - distance) / (D_START - D_PEAK))
    if distance >= D_HOLD:
        return 1.0
    return 1.0 - smoothstep01((D_HOLD - distance) / (D_HOLD - D_END))


def pgtt(distance: float, step_up: float, actual_geometric_lift: float):
    timing = timing_progress(distance)
    final = step_up + CLEARANCE_MARGIN
    desired = timing * final
    absolute = min(max(actual_geometric_lift / final, 0.0), 1.0)
    raw = timing * (2.0 * absolute - absolute * absolute)
    return timing, final, desired, absolute, raw, 2.0 * raw


def main():
    expected = {0.05: 0.06, 0.08: 0.09, 0.11: 0.12, 0.14: 0.15, 0.17: 0.18, 0.20: 0.21}
    for height, required in expected.items():
        assert abs(pgtt(0.20, height, 0.0)[1] - required) < 1.0e-12

    print("distance,timing_progress,final_required_lift,desired_lift_now,absolute_progress,raw_spatial,weighted_spatial")
    for distance in (0.40, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05):
        values = pgtt(distance, 0.17, 0.01)
        print(f"{distance:.2f}," + ",".join(f"{value:.8f}" for value in values))
        assert abs(values[1] - 0.18) < 1.0e-12
        assert abs(values[3] - 1.0 / 18.0) < 1.0e-12
        assert values[4] < 1.0

    print("step_up,final_required_lift,distance,timing_progress,desired_lift_now")
    for step_up in expected:
        for distance in (0.30, 0.20, 0.10, 0.05):
            timing, final, desired, *_ = pgtt(distance, step_up, 0.0)
            print(f"{step_up:.2f},{final:.2f},{distance:.2f},{timing:.8f},{desired:.8f}")

    print("HEIGHT_CONTROLS_AMPLITUDE=TRUE")
    print("DISTANCE_CONTROLS_TIMING=TRUE")
    print("DISTANCE_CHANGES_FINAL_AMPLITUDE=FALSE")
    print("17CM_1CM_ABSOLUTE_PROGRESS=0.05555556")
    print("17CM_1CM_CAN_SATURATE_PROGRESS=FALSE")
    print("17CM_1CM_CAN_GET_FULL_SPATIAL=FALSE")
    print("PGTT_HEIGHT_TIMING_UNIT_TEST=PASS")


if __name__ == "__main__":
    main()
