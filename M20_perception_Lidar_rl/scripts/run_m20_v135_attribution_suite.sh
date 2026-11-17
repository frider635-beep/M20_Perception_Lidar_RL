#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20_wzh/bin/python}"
RUN_DIR="${V135_ATTR_RUN_DIR:-logs/rsl_rl/M20-V13.5-V11FD-NativeAlternation/2026-09-18_14-38-55_v135_strict_identity_gru128_r48_from_scratch_30k}"
OUTPUT="${V135_ATTR_OUTPUT:-logs/diagnostics/v135_ppo_attribution}"
export PYTHONPATH="$ROOT/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

read -r -a checkpoints <<< "${V135_ATTR_CHECKPOINTS:-2600 2800 3000 3200 3400 latest}"
for checkpoint in "${checkpoints[@]}"; do
  "$PYTHON" scripts/tools/diagnose_m20_v135_ppo_attribution.py \
    --headless --device cuda:0 --run_dir "$RUN_DIR" --checkpoints "$checkpoint" \
    --num_envs "${V135_ATTR_ENVS:-128}" --steps "${V135_ATTR_STEPS:-1000}" \
    --window "${V135_ATTR_WINDOW:-16}" --output_root "$OUTPUT"
done
