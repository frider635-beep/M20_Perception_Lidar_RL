# V13.1 ordered alternation contract

## Authority and frozen surface

V13.1 inherits the current V13 task, which in turn is audited against the saved
V11-FD run under `logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch`.

The following remain identical to V13: actor and critic observations, action
range, commands, events, terminations, V11 physical landing and safety gates,
physical joint limits, PPO, network, and native velocity/forward-progress
rewards. V13.1 starts in the low curriculum rows and does not mask actions,
remove physical candidates, or add a style termination.

All numerical parameters and weights have one authority:
`v131_v11fd_ordered_alternation_env_cfg.py`. This document specifies semantics,
not a second numerical ledger.

## Ordered ascent state

1. `APPROACH_B1`: the first observed ascending face is locked in world space as
   B1. A persistent target-qualified asymmetric first-mover onset latches FL or FR. That
   same leg must safely clear, cross, and support B1. A simultaneous onset has
   no arbitrary tie-break, earns no side-neutral overlap credit, and is settled
   once as a style violation.
2. `ALTERNATING_RISERS`: after B1, the expected side toggles and the expected
   face moves exactly one configured tread width and one observed riser height
   forward. Dense progress is evaluated continuously against that locked world
   target; it does not require the live two-slot scanner to keep selecting the
   same face. A physically safe completion pays only if both expected side and
   expected adjacent face match.
3. `TOP_FOLLOW`: after a correct Bn completion, if the predicted adjacent Bn+1
   is absent for the confirmation window, the last completing leg is the top
   leader and the opposite leg must follow onto that same top tread.
4. `PLATEAU_ROLL`: after follower support is confirmed, stair progress is off.
   Native velocity remains active while unnecessary front-wheel lift above the
   confirmed plateau receives a bounded soft cost.
5. `SEQUENCE_FAILED`: a skipped target, wrong side, simultaneous B1 lift, or
   simultaneous completion makes this encounter ineligible for later ordered
   task credit. Physical locomotion and episode safety continue unchanged.

The first completed stair settles only its accumulated B1 escrow; merely moving
past B1 without stable support forfeits that balance. Native V13/V11 physical
rewards remain unchanged.

## Reward settlement and anti-farming

- Native landing, clearance, pose, velocity, and forward-progress rewards are
  inherited unchanged. Strict accounting never gates native locomotion.
- Expected-side dense shaping is a locked-target historical-best difference.
  Its pre-face component requires both lift and approach, so static lift or
  rolling alone cannot complete the potential; lowering and lifting again
  cannot replay an already credited maximum.
- Before B1 commitment, bilateral overlap is removed from the side-neutral
  positive signal. After commitment, only the committed leg may progress B1;
  the other leg settles into same-side/simultaneous cost buckets. B1 progress
  and its velocity coupling are accumulated but exposed to PPO only after the
  committed leg makes a valid B1 completion; skip and simultaneous settlement
  forfeit the escrow.
- During B2+, velocity quality accumulates only with expected-side target
  progress in a bounded per-riser escrow. Correct adjacent opposite-side
  completion pays it; wrong-side, simultaneous, or skip behavior forfeits it.
- Same-side and simultaneous progress are cost-only buckets. Wrong completion,
  simultaneous B1 onset, and skipped face are disjoint one-shot cost buckets,
  so one event cannot be silently charged twice. A skip also forfeits the
  remaining ordered sequence.
- Top-follow progress terminates after the opposite front wheel and then the
  final-riser leg are stably supported on the platform.
- While a locked adjacent target is actionable, only the expected front swing
  leg receives additional hip-y/knee action-rate and second-order-smoothness
  relief. The support leg remains at native V13 regularization and the relief
  ends on stable target support. Action scale and joint limits never change.

## Acceptance criteria

- Both mirror sequences are valid: FL-B1/FR-B2/... and FR-B1/FL-B2/....
- B2 cannot seed B1; B1-to-B3 cannot masquerade as an adjacent transition.
- The leg that first crosses the B1 lift-onset threshold must complete B1.
- Same-side, simultaneous, target-skip, and target-reacquisition cases are
  distinguishable in metrics and cannot receive ordered positive settlement.
- One-, two-, three-, and longer-stair flights all enter top-follow after their
  true final riser, then enter plateau-roll after the other front leg follows.
- Runtime acceptance requires fixed-rollout comparison with V13/V11-FD. Static
  tests and an environment step smoke prove wiring/reachability only, not gait
  quality.

No V13.1 training is started merely by adding this task or its launcher.
