#!/usr/bin/env bash
# Train V13.2 from scratch. This script never starts itself or stops other runs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_SH="${M20_CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
M20_ENV="${M20_TRAIN_ENV:-/home/ubuntu/miniconda3/envs/m20_wzh}"
ISAACLAB_SH="${M20_ISAACLAB_SH:-/home/ubuntu/.holosoma_deps/IsaacLab/isaaclab.sh}"
MAX_ITERATIONS="${M20_MAX_ITERATIONS:-30000}"
NUM_ENVS="${M20_NUM_ENVS:-8192}"
RUN_NAME="${M20_RUN_NAME:-v132_v11fd_barrier_alternation_from_scratch}"
[[ -r "$CONDA_SH" ]] || { echo "Conda setup not found: $CONDA_SH" >&2; exit 2; }
[[ -x "$ISAACLAB_SH" ]] || { echo "IsaacLab launcher not found: $ISAACLAB_SH" >&2; exit 2; }
source "$CONDA_SH"
if ! conda activate "$M20_ENV"; then
  echo "Cannot activate training environment: $M20_ENV" >&2
  exit 2
fi
M20_PYTHON="${CONDA_PREFIX:-}/bin/python"
[[ -x "$M20_PYTHON" ]] || { echo "Training Python not found after activation: $M20_PYTHON" >&2; exit 2; }
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

"$M20_PYTHON" scripts/tools/test_m20_v132_barrier_alternation.py
"$M20_PYTHON" scripts/tools/audit_m20_v13_v11fd_reference.py --headless
"$M20_PYTHON" scripts/tools/test_m20_v132_runtime_smoke.py --headless --num_envs 16 --steps 8

exec "$ISAACLAB_SH" -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13.2-V11FD-BarrierAlternation-v0 \
  --seed 42 \
  --max_iterations "$MAX_ITERATIONS" \
  --num_envs "$NUM_ENVS" \
  --headless \
  --run_name "$RUN_NAME" "$@"
