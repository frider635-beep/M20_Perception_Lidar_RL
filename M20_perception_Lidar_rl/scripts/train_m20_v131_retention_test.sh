#!/usr/bin/env bash
# Bounded Test A: V13 model_5000 actor, fresh V13.1 critic/optimizer, fixed stairs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_SH="${M20_CONDA_SH:-/home/ubuntu/miniconda3/etc/profile.d/conda.sh}"
M20_ENV="${M20_TRAIN_ENV:-/home/ubuntu/miniconda3/envs/m20_wzh}"
ISAACLAB_SH="${M20_ISAACLAB_SH:-/home/ubuntu/.holosoma_deps/IsaacLab/isaaclab.sh}"
MAX_ITERATIONS="${M20_MAX_ITERATIONS:-400}"
NUM_ENVS="${M20_NUM_ENVS:-8192}"
RUN_NAME="${M20_RUN_NAME:-v131_signal_reachability_v13_model5000_retention}"
DEFAULT_ACTOR="$ROOT_DIR/logs/rsl_rl/M20-V13-V11FD-Alternation/2026-09-09_00-34-12_v13_v11fd_fast_alternation_from_scratch/model_5000.pt"
export V131_RETENTION_ACTOR_CHECKPOINT="${V131_RETENTION_ACTOR_CHECKPOINT:-$DEFAULT_ACTOR}"

[[ -r "$CONDA_SH" ]] || { echo "Conda setup not found: $CONDA_SH" >&2; exit 2; }
[[ -x "$ISAACLAB_SH" ]] || { echo "IsaacLab launcher not found: $ISAACLAB_SH" >&2; exit 2; }
[[ -f "$V131_RETENTION_ACTOR_CHECKPOINT" ]] || {
  echo "Retention actor not found: $V131_RETENTION_ACTOR_CHECKPOINT" >&2
  exit 2
}
if (( MAX_ITERATIONS < 300 || MAX_ITERATIONS > 500 )); then
  echo "Test A must remain within 300--500 iterations, got $MAX_ITERATIONS" >&2
  exit 2
fi

source "$CONDA_SH"
conda activate "$M20_ENV"
M20_PYTHON="${CONDA_PREFIX:-}/bin/python"
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

exec "$ISAACLAB_SH" -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13.1-V11FD-OrderedAlternation-Retention-v0 \
  --seed 42 \
  --max_iterations "$MAX_ITERATIONS" \
  --num_envs "$NUM_ENVS" \
  --headless \
  --run_name "$RUN_NAME" "$@"
