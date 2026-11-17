# V13.5 native stair behavior and post-stair flat recovery

## Authority and frozen surface

V13.5 inherits the saved native V13 run under
`logs/rsl_rl/M20-V13-V11FD-Alternation/2026-09-09_00-34-12_v13_v11fd_fast_alternation_from_scratch`.
Actor observations remain 63D and actions remain 16D. The Actor now passes the
unchanged 63D stream through a one-layer 128D GRU before the native MLP head.
The Critic remains an MLP and receives the native 568D privileged observation
plus V13's 6D reward lifecycle state, for 574D total. The six values never enter
the Actor.
PPO hyperparameters remain identical. Rollout length is 48 at 4096 environments,
preserving the prior 196608 samples per update while doubling recurrent BPTT
context from 0.48 seconds to 0.96 seconds.

The restored historical V13.4 descent-follow objective is active at full
strength from update zero. There is no update-8000 objective switch or ramp.
It follows perceived descent outcomes without prescribing a left/right order.
Direct knee-angle and thigh-parallel costs remain absent.

## Local post-stair recovery

V13.5 adds one bounded training-only settlement after a real traversal:

- ascent eligibility begins only after V11 confirms the final platform;
- descent eligibility begins after a physically completed front descent;
- the raw local scanner must then report no riser for a confirmation window;
- all four wheels must contact and their bottoms must lie on one plane;
- body clearance is normalized to the episode's neutral starting clearance;
- upright posture and commanded forward speed contribute to recovery quality.

Historical-best quality progress can be earned once, and stable high quality
pays one completion event. Lowering and raising the body cannot replay credit.
The detector prescribes no joint angle, leg ordering, or rear-leg trajectory.
It does not alter action dimensions, terminations, terrain, native velocity,
native ascent completion, or V13 soft-alternation reward weights.

V13.5 uses one ascent authority for target id, seed state, and expected side.
Native V13 shaping and its critic lifecycle encoding consume that strict state;
they do not advance a second parity ledger. Internally the scanner retains
three ordered edges. The Actor still receives two 3-value tuples: P1 is the
edge matching the locked target and P2 is the real adjacent edge one tread
ahead. Geometric passage alone never promotes P1.

B1 becomes valid only after the same world-space face remains stable for the
configured confirmation window. Skip, dual-skip, or an unrecoverably rearward
target latches `ordered_chain_failed`. A failed chain receives no later ordered
success or progress credit, while the episode and native V11 locomotion rewards
continue. P1/P2 distances remain root-relative terrain geometry; leg-relative
distances are used only by crossing, support, and overstep tracking.

## Acceptance

- Reference audit allow-lists only the GRU Actor, restored descent-follow
  objective, post-stair recovery, and their metrics.
- Ordinary flat motion before a stair cannot activate recovery.
- A crouched, airborne, or non-coplanar outcome cannot complete recovery.
- Static and Isaac step tests establish wiring and finite execution only.
  Sim2sim remains required to judge posture and gait quality.

## Physical riser identity

The first observed ascending face is locked in world coordinates as B1. Either
front leg may complete B1 and seed the alternating phase. The target then moves
exactly one configured tread per valid completion. Wrong-side, duplicate,
skip, and dual-skip events neither advance the target nor reseed the phase.
Dense overstep cost begins when a front wheel approaches the face after the
locked target. This layer changes reward accounting only; it does not freeze a
leg, mask an action, or modify native physical completion and velocity rewards.

No training is started merely by changing this task.
