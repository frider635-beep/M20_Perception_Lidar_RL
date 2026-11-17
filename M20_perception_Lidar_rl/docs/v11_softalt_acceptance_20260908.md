# V11 SoftAlt frozen acceptance contract — 2026-09-08

Scope: a new V11-only fine-tune task. V13, the 63-D actor observation ABI,
the MLP architecture, gamma, lambda, PPO clip, epochs, mini-batches and
desired KL are frozen.

Control: the existing V11 task and its checkpoint remain unchanged. Candidate:
V11 SoftAlt initialized from an ABI-compatible V11 checkpoint, with 4096
environments and 48 steps per environment (196608 samples/update).

Static acceptance:

- both FL and FR may be the first physical completion;
- FL->FR and FR->FL each produce one alternating transition;
- FL->FL and FR->FR each produce one repeat event and do not terminate;
- changing `expected_side` cannot change physical completion;
- a confirmed physical front-riser collision still terminates;
- first, alternating, repeat and platform one-shot contributions are exactly
  +0.5, +2.5, -1.0 and +3.0 after RewardManager dt scaling.

Short-train decision set: 500--1000 iterations, compared with the frozen V11
baseline under the same evaluation protocol. GO requires alternation compliance
at least 0.80 and trending toward 0.90, platform success no more than 0.10
absolute below baseline, and flat locomotion at least 0.95 of baseline.
Persistent compliance below 0.60 with good platform success and non-zero repeat
rate is `NO_GO_MLP_TEMPORAL_MEMORY`; do not continue reward-weight escalation.

False positive: high compliance caused by zero/near-zero valid transitions.
Guard: first completions are excluded and a zero transition denominator is NaN
and invalid for promotion.

False negative: a valid geometric completion is rejected only because its side
does not match an internal ledger. Guard: replay each physical sequence with
opposite `expected_side` values and require identical physical events.
