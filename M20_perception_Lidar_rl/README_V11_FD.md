# M20 V11-FD

V11-FD is an independent from-scratch task derived from the saved
`2026-08-27_18-14-26_v11_from_scratch` configuration. It does not prescribe
which front leg moves first and does not use alternating-riser credit.

## Contracts

- Task: `M20-Stair-Flat-Descent-V11FD-v0`
- Experiment: `M20-V11-Flat-Descent`
- Actor: static `obs[1,63] -> actions[1,16]`
- Feature contract: `v11_signed_edges_v2`
- Ascent completion: `bilateral_same_riser`
- Crossing threshold: wheel centre at least 0.14 m above the real target tread
  (0.05 m wheel-bottom clearance with the configured 0.09 m wheel radius)
- Descent completion: bilateral front support followed by bilateral rear support;
  no left/right gait ordering

The height requirement is evaluated on the control step where each front wheel
crosses the target face. A high pose before the face is not sufficient. A low
crossing clears that wheel's pending clearance credit and cannot complete the
bilateral landing event.

For from-scratch reachability, V11-FD starts terrain types at the lowest two
difficulty rows. Crossing height is sampled at the real face, but its bounded
progress is released only after the same wheel reaches upper-tread support
without a confirmed face collision. Each wheel that passes the full hard
clearance-and-support gate also receives a subordinate one-shot milestone;
the bilateral event remains the main ascent reward. Lifting, crossing, target
loss and reset by themselves never release these shaping terms.

## Training

```bash
./scripts/train_m20_v11_flat_descent_from_scratch.sh
```

The launcher fixes seed 42, 8192 environments, 30000 iterations and a 100
iteration checkpoint interval. It does not resume or load a checkpoint.

Before promoting a run, audit its saved configuration against the source-run
baseline:

```bash
python scripts/tools/check_m20_v11_fd_baseline_delta.py \
  logs/rsl_rl/M20-V11-Flat-Descent/<run>
```

Checkpoint selection must use closed-loop clean/noisy clearance, descent and
flat-command metrics. The final iteration is not selected automatically.

## Export

```bash
./policy_transform.sh <model.pt> --output <model.onnx> \
  --feature-contract v11_signed_edges_v2
```

Run this from `m20_depth_deploy`. Export writes ONNX metadata and a matching
`.feature_contract` sidecar. sim2sim and NX startup fail closed when the sidecar,
producer layout or model contract does not match. Legacy V11/V12 artifacts use
`v11_positive_risers_v1` and remain positive-height only.

## Verified locally

- CPU high-clearance, signed-edge, old V11 and V12 regressions pass.
- Training and deployment Python syntax checks pass.
- C++ positive/signed extractor regression passes.
- A 32-environment, one-iteration Isaac PPO smoke completed with Actor 63D,
  actions 16D and the V11-FD reward set.

Full 30000-iteration training, sim2sim behavioral acceptance and real-robot
acceptance have not yet been run.
