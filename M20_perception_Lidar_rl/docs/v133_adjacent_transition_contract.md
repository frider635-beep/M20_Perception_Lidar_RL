# V13.3 native-observation three-step-chain contract

## Frozen V13 surface

V13.3 trains from random initialization and directly inherits V13. Actor input
remains the native 63 values, including the two forward stair faces and robot
state. Expected side, target identity, contact state and gait phase never enter
the actor observation. Commands, terrain distribution, physical completion,
safety, network widths and all six V13 soft-alternation rewards remain on the
V13 path.

V13.3 changes the rollout sampling cadence and adds a training-only chain
ledger. Its numerical values have one authority in executable configuration;
this document specifies behavior only.

Descending stairs use the official DeepRobotics M20 Rough reward and distance
curriculum at upstream commit `6d317dfb33060226139e38600510e1751372eb5d`.
There is no persistent tread identity, front-to-rear closure task, or diagonal
action prescription. V11 route, stair-impact and descent-completion terms are
zero on descending terrain. Difference terms restore the official posture,
vertical-velocity, action-rate, joint-position and joint-limit totals for the
entire descending terrain column, including approach and run-out. No descent
phase or contact state enters the actor observation.
The V11-added illegal-body-contact and front-riser-collision terminations are
also disabled on descending columns, matching the official M20 task. Native
time-out and base-height terminations remain active.

## Three-riser settlement

The first unambiguous front-wheel support on a live ascending tread seeds either
mirror chain. Support requires physical contact, wheel-bottom agreement with the
tread top, wheel placement inside the tread corridor and persistence for the
configured confirmation frames. Confirmation restarts if a scanner slot changes
world-space face. Per-side credited progress is monotonic for the episode, so a
previous tread cannot be replayed after advancing. This training-side detector does not depend on V11's stricter
landing reward event and does not enter the actor observation.
The following two completions must each be on the next adjacent ascending face
and must alternate front side. FL-B1/FR-B2/FL-B3 and its mirror both settle.
Every later correct adjacent side switch can settle an overlapping three-step
window.

After the first stable support, a training-only potential follows only the
opposite front wheel toward the exactly adjacent predicted tread. Its pre-face
component requires lift and forward approach together; crossing and stable
support close the bounded potential. Historical-best increments are paid as
they occur so PPO receives a signal before the rare completion event. The
unsettled amount is held in escrow: repeated-side, skipped, simultaneous or
terminal outcomes claw it back in full, while a correct adjacent completion
keeps it. Lowering and lifting again cannot replay an increment, and scanner
loss freezes rather than fabricates progress.

A same-tread follower is recorded but neither advances nor clears the active
chain; this permits ordinary transient support and the natural top follower.
Repeated side, skipped face and simultaneous completion receive distinct small
costs and reset the chain from the actual physical outcome.

Each correct opposite-side completion on the immediately adjacent face pays a
small transition settlement. Completing B1-B2-B3 pays an additional chain
bonus; the two transition settlements plus that bonus preserve the configured
full-chain value. Stable B3 support settles immediately, and every later correct
adjacent switch settles its own overlapping window. Partial progress toward any
non-adjacent face and same-tread support receive no added V13.3 reward. Native
V13 rewards continue to teach locomotion and natural phase separation.

## Curriculum and checkpoint selection

Ascending-terrain promotion retains the native locomotion gates and also
requires at least one completed three-step chain with sufficient episode chain
precision. Descending stairs use the official distance-only promotion rule;
there is no rear-closure gate. Missing ascent-chain accounting fails the ascent
gate closed.

Training saves dense checkpoints. After training, the launcher automatically
runs every saved actor without PPO updates on a fixed equal ascending and
descending stair setup.
The report preserves separate leaders for chain precision, chain count and
velocity. At least one checkpoint must pass finite-value, mirror-side, chain
count, chain precision and velocity-retention gates or the sweep fails. Among
qualified checkpoints, the selected policy is chosen lexicographically by chain
precision, chain count and then velocity; it is never assumed to be the last
iteration.

## Acceptance

CPU contracts cover both mirror chains and prove that follower, repeat, skip
and simultaneous outcomes cannot obtain a three-step settlement. A frozen V13
model known to show transient alternation must produce observable support seeds
and adjacent-transition events under the same detector. Isaac runtime
smoke must construct a 63D-to-16D actor, retain all native V13 soft rewards,
exclude every descent closure reward, and reconstruct the official M20 reward
totals on descending terrain while stepping finite observations, actions and
rewards. A complete PPO update is a
separate reachability check; gait quality requires the frozen checkpoint sweep
and sim2sim inspection.
