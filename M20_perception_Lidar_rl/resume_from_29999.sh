#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_ROOT="$ROOT_DIR"
RUN_NAME="2026-09-02_17-15-45_v11_fd_from_scratch"
CHECKPOINT="model_29999.pt"
CONDA_SH="${M20_CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
M20_ENV="${M20_TRAIN_ENV:-/home/ubuntu/miniconda3/envs/m20_wzh}"
ISAACLAB_SH="${M20_ISAACLAB_SH:-/home/ubuntu/.holosoma_deps/IsaacLab/isaaclab.sh}"

[[ -r "$CONDA_SH" ]] || { echo "Conda setup not found: $CONDA_SH" >&2; exit 2; }
[[ -x "$ISAACLAB_SH" ]] || { echo "IsaacLab launcher not found: $ISAACLAB_SH" >&2; exit 2; }
[[ -r "$TRAINING_ROOT/logs/rsl_rl/M20-V11-Flat-Descent/$RUN_NAME/$CHECKPOINT" ]] || {
  echo "Archived checkpoint is missing." >&2
  exit 2
}

source "$CONDA_SH"
conda activate "$M20_ENV"
export TERM=xterm
export PYTHONPATH="$TRAINING_ROOT/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
cd "$TRAINING_ROOT"

exec "$ISAACLAB_SH" -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-Flat-Descent-V11FD-v0 \
  --seed 42 \
  --max_iterations "${M20_MAX_ITERATIONS:-30000}" \
  --num_envs "${M20_NUM_ENVS:-8192}" \
  --headless \
  --resume \
  --load_run "$RUN_NAME" \
  --checkpoint "$CHECKPOINT" \
  --run_name v11_fd_resume "$@"
