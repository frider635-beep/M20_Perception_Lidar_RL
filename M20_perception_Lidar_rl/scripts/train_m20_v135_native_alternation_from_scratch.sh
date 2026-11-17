#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
M20_PYTHON="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20_wzh/bin/python}"
[[ -x "$M20_PYTHON" ]] || { echo "Training Python not found: $M20_PYTHON" >&2; exit 2; }
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

"$M20_PYTHON" scripts/tools/test_m20_v13_v11fd_alternation.py
"$M20_PYTHON" scripts/tools/test_m20_v135_riser_identity.py
"$M20_PYTHON" scripts/tools/test_m20_v135_native_alternation.py
"$M20_PYTHON" scripts/tools/test_m20_v135_behavior_credit.py
"$M20_PYTHON" scripts/tools/test_m20_v135_training_report.py
"$M20_PYTHON" scripts/tools/audit_m20_v135_reference.py --headless
"$M20_PYTHON" scripts/tools/test_m20_v135_runtime_smoke.py --headless --num_envs 16 --steps 8 --ppo_update --stage_restore_smoke
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13.5-V11FD-NativeAlternation-v0 \
  --seed "${V135_SEED:-42}" \
  --num_envs "${V135_NUM_ENVS:-8192}" \
  --max_iterations "${V135_MAX_ITERATIONS:-30000}" \
  --headless \
  --run_name "${V135_RUN_NAME:-v135_longtrain_final_mlp_native6d_e8192_r24_from_scratch_30k}"
