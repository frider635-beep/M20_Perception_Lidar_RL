# V13.4 structured-exploration long run

The reviewed V13.4 task was launched from scratch on 2026-09-16 with seed 42,
8192 environments, a 32-step rollout and 20000 iterations. No actor, critic or
optimizer checkpoint was loaded.

The run uses the native 63D actor and 16D action interface. Three 580D critics
separate native locomotion, ascent alternation and descent posture returns.
Training-only ascent exploration applies temporally correlated, mirror-symmetric
offsets to FL/FR hip-y and knee actions. Exact adjacent-target evidence opens
the ascent actor channel; permissive V13 side-switch rewards alone cannot open
it. Once open, the ascent gradient receives conflict projection and a bounded
minimum strength relative to the native ascent gradient.

Systemd unit: `m20-v134-explore-retain-r32-20k.service`

Console log: `logs/v134_explore_retain_r32_20260916_000436.console.log`

Run directory:
`logs/rsl_rl/M20-V13.4-V11FD-PerceptiveFollow/2026-09-16_00-10-33_v134_native63_terrain3critic_explore_retain_r32_from_scratch_20k_final`

The static contracts, V13/V13.3 regressions, exact configuration audit,
32-environment runtime smoke and a separate complete PPO update passed before
launch. Initial observed iteration time was approximately 3.8 seconds.

## Superseded run diagnosis

The run was stopped at iteration 11712 after fixed-checkpoint inspection found
two reward/exploration exploits. Structured ascent exploration remained at high
strength over every ascending-terrain sample because its decay clock required a
complete chain, while the run produced no chains. Near an actionable stair
target the frozen actor moved backward in roughly one fifth of samples. On
descent, active rear-follow samples showed rear-knee deviation almost entirely
in the flexion direction, while the contact-only completion objective still
reported high success.

The replacement V13.4 keeps the task name, native actor ABI, three critics and
rollout cadence. It gates correlated offsets per front leg with the native
actionable-target mobility state and decays them from exact adjacent-progress
evidence. Descent adds a bounded one-sided rear-knee flexion cost across
forward-commanded descending terrain. Extension remains free and no rear gait
order or tread identity is introduced.

## Replacement from-scratch run

The corrected V13.4 run started from a fresh actor, three fresh critics and a
fresh optimizer with seed 42, 8192 environments, rollout 32 and 20000 target
iterations.

The first replacement launch was stopped during its initial rollout audit when
the evidence-only decay was found to retain the same zero-evidence deadlock as
the superseded run. The final launch adds a bounded rollout decay clock.

Systemd unit: `m20-v134-actionable-kneeflex-bounded-r32-20k.service`

Console log:
`logs/v134_actionable_kneeflex_bounded_r32_20260916_113339.console.log`

Run directory:
`logs/rsl_rl/M20-V13.4-V11FD-PerceptiveFollow/2026-09-16_11-34-49_v134_actionable_bounded_exploration_kneeflex_r32_from_scratch_20k`

## Native-V13 replacement after sim2sim rejection

The exploration/three-critic branch was stopped after sim2sim showed that the
policy could neither climb nor descend. At about iteration 3000 its pre-B1
backward rate had risen to roughly one third, while adjacent and chain counts
were still zero. This rejected action offsets, terrain-separated actor
advantages, gradient projection, delayed descent activation and the descent
follow state machine as a combined training design.

The replacement directly inherits V13. Native velocity/forward progress and
the standard single-critic PPO again control stair entry. The actor remains
63D; twelve training-only values make the single critic 580D. The learned
Gaussian is the only action exploration source, with entropy coefficient
0.005. After B1, bounded historical-best progress toward the exact adjacent
tread pays up to +0.5; stable adjacent B2 support pays +2 and a completed
B1-B2-B3 chain pays an additional +6. There are no V13.4 skip, repeat,
simultaneous or failed-attempt costs, so retreating before B1 cannot avoid a
V13.4 penalty.

Descent has no V13.4 follower state. It retains V13 locomotion and adds bounded
course-wide costs only for excessive rearward reach and excessive rear-knee
folding, plus symmetric rolling consistency. These terms do not select a leg
or retain a target tread.

The V13/V13.3/V13.4 CPU contracts, exact V13 delta audit, 32-environment Isaac
step smoke and a complete 512-environment by 32-step PPO update passed. The
from-scratch long run started with seed 42, 8192 environments, rollout 32 and
20000 target iterations.

Systemd unit:
`m20-v134-native-positive-chain-r32-20k.service`

Console log:
`logs/v134_native_positive_chain_r32_20260916_150239.console.log`

Run directory:
`logs/rsl_rl/M20-V13.4-V11FD-PerceptiveFollow/2026-09-16_15-03-18_v134_native_v13_positive_chain_natural_descent_r32_from_scratch_20k`

That sparse-only launch was stopped at iteration 26 after comparison with the
2026-09-14 native-63D V13.3 run showed it repeated the same occupancy failure:
larger settlement cannot help before an adjacent event exists. The final
candidate therefore adds the bounded positive adjacent-progress teaching
signal described above and logs target-active, delta, escrow and clawback
separately. It was restarted from random weights after all audits passed.

Active systemd unit:
`m20-v134-dense-observed-r32-20k.service`

Active console log:
`logs/v134_dense_observed_r32_20260916_151100.console.log`

Active run directory:
`logs/rsl_rl/M20-V13.4-V11FD-PerceptiveFollow/2026-09-16_15-11-34_v134_native_v13_dense_adjacent_observed_natural_descent_r32_from_scratch_20k`
