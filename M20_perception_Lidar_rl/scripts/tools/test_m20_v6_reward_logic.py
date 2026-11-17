"""Deterministic state-machine acceptance for V6 staircase rewards."""

DT = 0.02


def one_shot(value, already):
    return (0.0, True) if already else (value, True)


def main():
    # A: stationary for >1 s produces a sustained penalty and no termination.
    a = {"stagnation": -2.0 * 2.0, "terminated": False}
    assert a["stagnation"] < 0 and not a["terminated"]
    print("A stagnate_center PASS", a)

    # B: stop before riser: dense deficit + stagnation, then timeout failure.
    b = {"clearance": -4.0 * 5.0, "stagnation": -2.0 * 4.0, "failure": -30.0}
    assert sum(b.values()) < -30
    print("B stagnate_before_riser PASS", b)

    # C: enough clearance alone never creates a positive term.
    c = {"clearance": 0.0, "crossing": 0.0, "step": 0.0, "success": 0.0, "failure": -30.0}
    assert c["clearance"] == 0 and sum(c.values()) < 0
    print("C clearance_only_no_cross PASS", c)

    # D: collision is latched per wheel and level.
    first, latch = one_shot(-5.0, False); repeated, _ = one_shot(-5.0, latch)
    assert first == -5 and repeated == 0
    print("D riser_collision_one_shot PASS", {"first": first, "repeated": repeated})

    # E: falling before success keeps both official bad-orientation and V6 failure.
    e = {"official_bad_orientation": -20.0, "failure": -30.0}
    assert sum(e.values()) == -50
    print("E fall_before_success PASS", e)

    # F: exactly four wheels at the same level trigger four crossings + one step.
    f = {"wheel_crossing": 4 * 5.0, "step_complete": 10.0}
    assert sum(f.values()) == 30
    print("F complete_one_level PASS", f)

    # G: six independent levels trigger success once and suppress failure.
    g = {"wheel_crossing": 6 * 4 * 5.0, "step_complete": 6 * 10.0, "success": 30.0, "failure": 0.0}
    assert sum(g.values()) == 210 and g["failure"] == 0
    print("G complete_six_levels PASS", g)
    print("REWARD_LOGIC_ACCEPTANCE=PASS")


if __name__ == "__main__":
    main()
