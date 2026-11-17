# V13--V13.5 cross-version root-cause review

Date: 2026-09-19

## Conclusion

The ascent failure is a mixed cause with a clear ordering:

1. **VISITATION_STARVATION is primary.** Exact adjacent, contact-backed
   alternating events are extremely rare compared with skip/ordinary climbing.
   A larger terminal cycle bonus cannot affect PPO before trajectories visit
   that event.
2. **RETURN_ORDERING is strongly suspected but not yet attribution-proven.**
   Native velocity and forward rewards remain large while style return is tiny,
   and later optimization consistently increases ordinary/simultaneous/skip
   occupancy. A matched return/advantage comparison has not yet been run, so
   this is not promoted to a standalone measured verdict.
3. **Detector/proxy mismatch remains an unresolved secondary risk.** Visual V13
   phase separation did not become a strict physical adjacent chain. Later
   contact-backed detectors do see some adjacent events, so the detector is not
   globally dead, but event-window traces are still required to separate a true
   skip from a ledger false negative.

The descent failure is an **objective endpoint mismatch**: lower-tread contact
or rear-follow completion can settle while the rear legs remain in an abnormal
extended/folded posture. The bad posture is therefore a valid terminal state of
the current reward episode, not evidence that PPO cannot descend.

## Measured evidence

### V13

- Visual phase separation appeared near model 5000, but frozen strict
  evaluation found zero ordered chains and approximately 0.79--0.89 skip rate.
- At model 5000 velocity tracking was approximately 0.837.
- Positive style return was only approximately 0.02--0.03% of native locomotion
  return.
- TensorBoard simultaneous-progress fraction rose from 0 to 0.774 around the
  middle of the run and to 0.870 at the end.

This means V13 demonstrated that alternating-looking dynamics are reachable,
but it did not demonstrate stable physical B1/B2/B3 adjacency.

### V13.1

- At the end of the 30k run, native velocity reward was 4.110 and forward
  progress was 0.929.
- Absolute ordered-event rate was only about 9.2e-5.
- Strict adjacent alternating rate peaked near 0.00974 and ended near 0.00685.
- Rollout adjacent rate fell from about 0.417 in the middle to 0.0208 at the
  end, while rollout skip rate ended near 0.715.
- B1 escrow paid rate ended near 0.105, while forfeit rate ended near 0.649.

The strict signal was reachable, but its occupancy and retained credit were too
small; later PPO optimization moved away from it.

### V13.2

- The run reached 19464 iterations with velocity reward about 4.150.
- Internal ordered-completion rate was about 0.288 and style reward grew, while
  native landing reward and physical completion were effectively zero.
- B1 completion was only about 2.74e-4 and skip rate about 0.327.

The style ledger could report progress without the desired physical outcome.
This invalidates internal style metrics as proof of gait quality and does not
prove that a larger style advantage would retain the gait.

### V13.3

- At iteration 2595, velocity reward was about 4.063 and forward progress about
  0.898.
- Lifetime detector counts were approximately 608 adjacent events, one full
  chain, and 107176 skips.

The full-chain event was technically reachable but too rare to shape the
on-policy state distribution.

### V13.4

- At iteration 6844, velocity reward was about 4.285 and forward progress about
  0.906.
- Lifetime counts were 286 adjacent events, three chains, and about 60383 skips.
- Adjacent-progress active rate declined to about 1.37e-4 despite reaching a
  higher earlier maximum.
- Earlier exploration/three-critic variants increased backward approach and
  failed to produce chains; the sparse-only chain settlement was stopped
  because a larger settlement could not help before adjacent occupancy existed.

This rejects teacher/action-offset/multi-critic complexity as a demonstrated
solution to the core occupancy problem.

### Current V13.5 Final run

- At iteration 7420, velocity reward was about 4.196 and forward progress about
  0.865.
- Strict target-valid rate was about 0.298, but target-id mean was only 1.051.
- Strict ordered-alternation rate remained exactly zero through the run.
- Cumulative skip ratio had risen to about 1.435.
- Leader progress rewards were nonzero (about 0.0078 and 0.0198), proving that
  local shaping is reachable, while valid strict completion remained zero.

This is not evidence that a strict cycle reward is too small. It is evidence
that the event which would trigger that reward is absent from the rollout.

## Deterministic accounting defect

Isaac Lab's active `RewardManager.compute()` multiplies every term by `dt`.
V13 progress and completion pulses divide by `env.step_dt` before returning.
The V13.5 code used by the reviewed run returned raw one-frame booleans for the
physical-riser events (`valid`, `wrong`, `duplicate`, `skip`, and `dual_skip`).
At a 0.02 s control interval their configured event values were therefore only
2% of the apparent weights. This concrete accounting defect weakened the strict
layer relative to native locomotion. It is now fixed on disk by converting each
event pulse to an event rate (`event / env.step_dt`) before RewardManager applies
its integration factor. The reviewed checkpoints remain evidence from the old
accounting because no continuation or new training has been started.

It does not explain zero strict occupancy by itself: multiplying a never-visited
event still produces no gradient.

## Descent evidence

- A historical V13.5 descent-follow run reported completion rate around 0.83
  while visual inspection still showed rear-leg trailing/extension.
- The current Final run pays rear-transfer completion (about 0.18 episode-rate
  reward near iteration 7420), while body-rear-lag and premature-collapse costs
  are essentially zero.
- Earlier V13.4 contact-only descent completion also remained high while rear
  knee flexion became the dominant exploit.

The endpoint detector pays for reaching lower support but does not require a
neutral recovered rear geometry to remain stable afterward. This explains why
different versions alternate between rear extension and folding rather than
converging on a normal support-transfer posture.

## What is not established

- **CRITIC_ALIASING:** no matched empirical-return versus value-bias result.
- **PPO_GRADIENT_CONFLICT:** no per-gait actor-gradient norm/cosine result.
- **SEQUENCE_MEMORY_FAILURE:** no grouped raw-observation versus hidden-state
  probe, and native MLP V13 already showed phase-separated motion.
- **DETECTOR_MISMATCH:** plausible for some visual candidates, but no reviewed
  event-window disagreement sample yet.

## One next intervention

Run the frozen, no-update attribution on V13 model 5000 and V13.5 models 5000
and 7400, using matched ascent conditions and 32-step event windows. The run
must report strict-label disagreement, raw return/value/advantage before
normalization, sample fractions, surrogate sums, and ALT versus ordinary
gradient cosine. Do not change reward weights until this distinguishes strict
event starvation from detector false negatives and matched return ordering.
