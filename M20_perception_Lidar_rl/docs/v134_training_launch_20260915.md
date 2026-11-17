# V13.4 formal training launch — 2026-09-15

The formal from-scratch run uses task
`M20-Stair-V13.4-V11FD-PerceptiveFollow-v0`, seed 42, 8192 environments,
32 steps per environment, eight PPO mini-batches and 20000 iterations. The
runner reported a fresh actor, fresh native/ascent/descent critics, a fresh
optimizer and no loaded checkpoint.

Resource basis: the strict configuration audit, 32-environment Isaac step smoke
and a 64-environment complete PPO update passed. Immediately before launch the
RTX 4090 had about 46 GiB free and host MemAvailable was 93 GiB. The formal
process measured about 9.9 GiB compute memory after startup and is isolated in
a systemd unit with a 64 GiB host-memory limit.

Service: `m20-v134-terrain3critic-chaincredit-r32-20k-b.service`.

Console log: `logs/v134_terrain3critic_chaincredit_r32_20260915_2326.console.log`.

Run directory:
`logs/rsl_rl/M20-V13.4-V11FD-PerceptiveFollow/2026-09-15_23-26-07_v134_native63_terrain3critic_chaincredit_r32_from_scratch_20k_final`.

The launcher passed the V13.3/V13.4 CPU contracts, strict reference audit and
32-environment Isaac runtime smoke before starting training. The formal run
continued through iteration 25 with no
traceback, non-finite policy observation or CUDA out-of-memory error. No ascent
positive samples existed yet, so the ascent actor scale correctly remained
zero rather than turning early penalties into stair avoidance. These facts
establish startup and wiring only; they do not establish learned gait quality.
