#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
M20_PYTHON="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20_wzh/bin/python}"
[[ -x "$M20_PYTHON" ]] || { echo "Training Python not found: $M20_PYTHON" >&2; exit 2; }
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

"$M20_PYTHON" scripts/tools/test_m20_v11_fd_contract.py
"$M20_PYTHON" scripts/tools/test_m20_v111_rear_posture.py
"$M20_PYTHON" scripts/tools/test_m20_v111_exit_direction.py
"$M20_PYTHON" scripts/tools/test_m20_v111_v135_descent_runtime.py --headless --num_envs 16
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V11.1-V11FD-RearPosture-v0 \
  --seed "${V111_SEED:-42}" \
  --num_envs "${V111_NUM_ENVS:-8192}" \
  --max_iterations "${V111_MAX_ITERATIONS:-30000}" \
  --headless \
  --run_name "${V111_RUN_NAME:-v111_v11fd_rear_extension_knee_roll_from_scratch}"
