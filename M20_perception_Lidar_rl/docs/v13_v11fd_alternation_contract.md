# V13 V11-FD fast-alternation contract

The authority is the saved configuration under
`logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch`,
not later edits to the V11-FD source in this archive.

V13 preserves the saved V11-FD actor 63D, action 16D, terrain, commands,
velocity/progress/safety reward weights, PPO and network settings. Its one
intentional task-unit change is necessary for the requested gait: ascent
completion becomes one safely landed front side on each successive physical
riser, instead of requiring both front sides to finish the same riser. The
existing +5 generic completion term is retained for every such safe physical
completion, independent of style. V13 then adds a soft preference over which
side should take the next riser. It does not gate candidates, mask actions,
terminate style failures, add a phase clock, or expose expected side to actor.

## Sequence identity

- For stairs B1...Bn, both `FL-B1, FR-B2, FL-B3...` and
  `FR-B1, FL-B2, FR-B3...` are legal for any observed stair count.
- The first single-side front completion is a free, unpaid seed.
- A completion is V11-FD's existing clearance -> clean crossing -> upper-tread
  support event, not lift onset or raw contact.
- The single-side branch explicitly retains V11-FD's crossing-time 14 cm
  wheel-centre clearance gate; a wheel that rises early but crosses low cannot
  complete or unlock the latter half of the potential.
- Only a different world-space riser may advance the sequence. Reacquiring the
  same face cannot mint another event.
- Once B1 seeds a side, expected side is fixed by physical-riser parity. Every
  later distinct riser advances that parity: a wrong B2 is penalized but B3
  still receives the originally scheduled side instead of shifting the gait.
- A simultaneous FL/FR B1 cannot seed the phase. A later simultaneous riser is
  not compliant, but consumes that riser and advances the fixed parity.
- Flat/no-target intervals never clear the seeded phase and never fabricate a
  stair reward; the next observed physical riser resumes the stored ordering.

## Dense shaping ledger

For each front side, bounded effective riser potential is

`Phi = 0.50 * clearance + 0.25 * clean_crossing + 0.25 * safe_support`.

Clearance is V11's forward-qualified maximum progress; the later components
are unavailable without a clean crossing. Only positive historical-best
increments on the locked next riser are credited. All increments and event
pulses are divided by `env.step_dt`, so integrated return is independent of the
control period. Opposite-only, same-only and simultaneous progress are three
mutually exclusive settlement buckets: an overlapping increment cannot first
collect the full positive opposite reward and then pay only a smaller overlap
penalty.

The only six V13 reward terms are:

| Term | Weight | Meaning |
|---|---:|---|
| opposite progress | +0.50 | dense next-side effective progress |
| velocity x opposite progress | +1.50 | prefer alternation that preserves native V11 velocity quality |
| same-side progress | -0.75 | reject repeated-side progress |
| simultaneous progress | -0.50 | reject overlap in effective riser progress only |
| alternating completion | +1.00 | confirm safe correct-side completion |
| same-side completion | -0.50 | penalize safe repeated or simultaneous completion |

Velocity quality is the native `track_lin_vel_xy_exp` signal in `[0, 1]` with
the saved V11 standard deviation. V13 does not reduce the native V11 velocity
reward or stair progress, and preserves the generic completion weight while
changing its unit to one side/one riser. There is no
900--1200 counter-reward window: guidance is present from scratch so it can
shape the gait at the same time the strong velocity reward drives emergence.
The old SoftAlt alternation-based terrain-promotion gate is explicitly disabled
for V13; native V11 tracking/progress/platform curriculum remains active, so
style is never a prerequisite for receiving harder terrain. The inherited
weak landing-centre term is also settled on every physical completion rather
than secretly using the old actual-previous-leg alternation signal.

## Frozen acceptance criteria

1. Constructed V13 config matches the named V11-FD snapshot except the explicit
   one-side-per-riser soft completion mode, its generic completion function,
   the six V13 style rewards, and diagnostic curriculum terms.
2. CPU contracts cover both mirror sequences, unpaid first seed, historical
   best increments, repeated-side and simultaneous penalties, world-space
   riser identity, and exact weights.
3. No V13 action/candidate/termination gate exists; the V11 landing state is
   consumed read-only.
4. Behavioral success is never inferred from aggregate training reward. V11
   and V13 checkpoints must later be evaluated in the same fixed rollout
   matrix. V13 succeeds only if alternation rises without losing V11-FD
   completion, speed, or survival.

False positive: ordinary bilateral leg motion or a reacquired riser looks like
alternation. The effective-potential definition and world-space identity gate
prevent that.

False negative: V11 clearance/crossing/support progress is unreachable, so V13
emits no shaping. `potential_active_rate`, per-side progress, compliance, and
simultaneous metrics must be observed during training; syntax/config checks do
not prove behavioral emergence.
