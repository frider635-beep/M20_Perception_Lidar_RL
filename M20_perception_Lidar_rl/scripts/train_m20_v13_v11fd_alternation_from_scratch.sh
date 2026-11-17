#!/usr/bin/env bash
# Train V13 from scratch: native V11-FD plus velocity-coupled soft alternation.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_SH="${M20_CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
M20_ENV="${M20_TRAIN_ENV:-/home/ubuntu/miniconda3/envs/m20_wzh}"
ISAACLAB_SH="${M20_ISAACLAB_SH:-/home/ubuntu/.holosoma_deps/IsaacLab/isaaclab.sh}"
MAX_ITERATIONS="${M20_MAX_ITERATIONS:-30000}"
NUM_ENVS="${M20_NUM_ENVS:-8192}"
RUN_NAME="${M20_RUN_NAME:-v13_v11fd_fast_alternation_from_scratch}"
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

# Fail before allocating the long run if V13 has drifted from the named V11
# run beyond the explicit alternation whitelist.
"$M20_PYTHON" scripts/tools/audit_m20_v13_v11fd_reference.py --headless

exec "$ISAACLAB_SH" -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13-V11FD-Alternation-v0 \
  --seed 42 \
  --max_iterations "$MAX_ITERATIONS" \
  --num_envs "$NUM_ENVS" \
  --headless \
  --run_name "$RUN_NAME" "$@"
