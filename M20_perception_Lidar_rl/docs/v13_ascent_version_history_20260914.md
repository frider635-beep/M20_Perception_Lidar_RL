# V13 ascent-training version history

This ledger separates intended mechanisms from measured behavior. Numerical
authority remains in each executable task configuration; this file records the
experiments and why each branch was kept or rejected.

## Shared baseline

V13 and all descendants start from V11-FD locomotion: a 16-D action, native
velocity and forward-progress objectives, perceptive stair geometry, physical
landing and safety terms, and a `[512, 256, 128]` ELU actor. Unless stated
otherwise, runs use 8192 environments, PPO learning rate `1e-3`, adaptive KL
with target `0.01`, five learning epochs, discount `0.99`, GAE `0.95`, clip
`0.2`, and entropy coefficient `0.003`.

## V13: native 63-D soft alternation

- Actor observation stayed 63-D: angular velocity, gravity, command, 16 joint
  positions, 16 joint velocities, previous 16-D action, and 6-D two-face stair
  geometry. Expected side and phase were not observations.
- Ascent completion changed from both front wheels completing the same riser to
  either safely landed front wheel completing a distinct riser.
- The first completion seeded parity without an extra bonus. Later expected-side
  progress received `+0.5`, velocity-coupled progress `+1.5`, and completion
  `+1.0`; repeated-side progress received `-0.75`, simultaneous progress
  `-0.5`, and wrong/repeated completion `-0.5`.
- Native landing (`+5`), velocity and forward progress stayed active for every
  physically valid result. Alternation did not gate actions, terminations or
  terrain promotion.
- Training used 24 steps per environment, four mini-batches, 30k requested
  iterations, seed 42, and random initialization.
- A tread was considered distinct after only 0.15 m while the actual tread was
  0.30 m. This did not enforce immediate adjacency.
- Measured result: visual left/right phase separation appeared near model 5000,
  including the observed FL-B1/FR-B2/FL-B3 pattern in sim2sim. Frozen strict
  evaluation nevertheless measured zero strict ordered chains and roughly
  0.79--0.89 skip rate. At model 5000 velocity tracking was about 0.837 and skip
  rate about 0.891. The positive V13 style return was only about 0.02--0.03% of
  native locomotion return, while simultaneous-progress fraction rose to about
  0.87 late in training. PPO therefore preserved speed and lost the transient
  adjacent pattern.

## V13.1: strict B1/B2 world-target state machine

- Actor stayed native 63-D. The additional ordered state existed only in the
  reward/metric ledger.
- The first observed face was locked as B1. An asymmetric first mover had to
  persist for three frames before FL or FR commitment. Hysteresis reduced noisy
  relatching.
- The committed leg had to clear, cross and stably support B1. B2 and later
  targets were predicted exactly one 0.30 m tread and one observed riser height
  forward. Wrong side, simultaneous action and skipped face invalidated strict
  sequence credit.
- Dense B1 and later progress was escrowed and paid only after the matching
  stable support event. Skip or wrong completion forfeited the escrow. A
  top-follow and plateau-roll state handled the final riser.
- Action-rate and action-smooth penalties were relaxed only for the expected
  front swing leg: hip-y scales `0.35/0.50` and knee scales `0.50/0.50`; action
  range and physical joint limits were unchanged.
- Initial terrain was reduced to levels 0--1. PPO remained V13-identical: 24
  rollout steps, four mini-batches, 30k requested iterations, seed 42, from
  scratch.
- Measured near iteration 7034: strict alternation remained zero; B1 escrow paid
  about 2.6%, forfeited about 48.4%, and only about 3.5% of latches completed
  B1. Native speed reward remained about 4.19 and rollout skip averaged about
  0.898. The strict positive event was too sparse; swing relief could also make
  ballistic skips easier.

## V13.2: 68-D deployable phase state and dual-return PPO

- Actor grew from 63-D to 68-D: one scalar for expected FL/unset/FR and a 4-D
  one-hot for approach-B1, alternating-risers, top-follow and plateau-roll.
- A PGTT/odometry/FK state machine anchored B1 in world space, recovered integer
  riser index, imposed fixed odd/even side parity, used a relaxed log barrier for
  lead margin, and required positive platform-plane evidence before top-follow.
- The six V13 soft terms were replaced by one `v132_style` channel. Expected
  progress, barrier progress, velocity coupling, wrong side, simultaneous,
  skip, overshoot, top-follow and plateau-lift outcomes were settled in that
  channel.
- PPO used separate native-task and style critics. Both advantages were
  normalized independently and added with equal weights; an all-zero style
  rollout was forced to produce zero style policy gradient.
- Training retained 24 rollout steps, four mini-batches, `1e-3` adaptive-KL PPO,
  30k requested iterations, seed 42, and random initialization.
- The run reached iteration 19464. Late metrics reported velocity reward about
  4.15 and style reward about 1.51 per episode, but native landing and platform
  success were both zero; B1 completion was about 0.0003 while the internal
  ordered-completion metric was about 0.288 and skip about 0.327. This meant the
  style ledger could grow without the desired physical stair completion. The
  68-D deployment state also conflicted with the desired native 63-D ABI.

## First V13.3: 76-D explicit acquisition state (rejected)

- Actor grew to 76-D by appending 13 values: expected side, target valid,
  forward target distance, relative target height, current tread height, four
  course-phase values and four leg-phase values.
- This branch tried to make the instruction explicit, balanced the initial B1
  side, used phase-aware milestone shaping and a separate style return, and
  experimented with longer rollout horizons before reducing them.
- The formal 300-iteration gate ended with 807 lifts, 26 clearances, two
  crossings, zero stable supports and zero ordered adjacent completions. Support
  fell from 19 events in iterations 0--49 to zero throughout 200--299. Target
  agreement was 0.8757, below its 0.98 gate.
- It was rejected because positive upstream milestones could survive a wrong or
  skipped completion, physical completion collapsed, and the 13-D deployment
  state was complex and fragile.

## Second V13.3: native 63-D contact-backed three-step chain

- The actor returned to the original V13 63-D observation. Target identity,
  contact, expected side and chain phase are training-only.
- A stable-support detector requires contact, wheel-bottom/tread-height
  agreement, placement in the live tread corridor and multi-frame persistence.
  It records exact 0.30 m adjacent transitions with 0.08 m longitudinal
  tolerance and prevents replay of an already credited tread.
- Correct adjacent side switches pay `+1`; B1-B2-B3 pays another `+4`. Same-tread
  follower support does not advance or erase a valid chain. Ascent curriculum
  promotion requires at least one chain and chain precision at least 0.50.
- Training changed to 48 rollout steps and eight mini-batches. With 8192
  environments this doubles samples per update while preserving the original
  49,152 samples per mini-batch. PPO learning rate and adaptive KL target remain
  `1e-3` and `0.01`. Checkpoints are saved every 250 iterations; training is
  fixed at 20k and always starts from random weights. Every checkpoint is then
  frozen-evaluated rather than assuming the final checkpoint is best.
- The first long native-63D run stopped near iteration 2594 with 607 adjacent
  events, one full chain, 6648 repeats, 2802 followers, 106838 skips and 153
  simultaneous events. Velocity reward was about 4.08 and forward progress
  about 0.895. Since repeat/skip/simultaneous were only metrics, ordinary fast
  climbing remained the better return.

## Current V13.3 candidate (validated, long run not yet started)

- The same 63-D contact-backed chain remains. Small ascent-only costs were added:
  repeat `0.10`, skip `0.02`, simultaneous `0.10`. They reset the chain from the
  actual outcome; native V13 rewards remain active on ascent.
- After a stable seed, the opposite front wheel receives bounded continuous
  progress only toward the exactly adjacent predicted tread. Progress is paid
  immediately for credit assignment but held in escrow: repeat, skip,
  simultaneous or terminal outcomes claw it back in full, while correct
  adjacent support keeps it. This retains V13's natural 63-D phase path without
  repeating V13.1's completion-only sparse escrow or the old 76-D milestones
  that survived failed attempts.
- Descending terrain now follows the current official DeepRobotics M20 Rough
  objective at upstream commit `6d317dfb`: official posture, vertical velocity,
  action-rate, joint-position and joint-limit totals, distance-only curriculum,
  and no persistent tread lock, rear-follow sequence or diagonal prescription.
  V11/V13 stair-specific rewards and added contact/riser terminations are masked
  on descending columns. Ascent keeps perceptive V11/V13 rewards and chain
  teaching.
- CPU contracts, V11/V13 regressions, a 16-environment by 8-step Isaac smoke,
  and one complete 2048-environment by 48-step PPO update pass. The subsequent
  from-scratch training configuration uses 32 rollout steps to shorten the
  policy-update feedback interval. The runtime actor
  is 63-D to 16-D and actor, both critics and optimizer are freshly initialized.
- These checks prove wiring and trainability only. They do not yet prove that
  full-chain frequency rises or that descent posture matches the official policy
  after learning. No new long run has been started from this candidate.
