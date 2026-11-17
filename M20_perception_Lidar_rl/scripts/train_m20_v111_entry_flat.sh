#!/usr/bin/env bash
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_ROOT"
M20_PYTHON="${M20_PYTHON:-python3}"
export PYTHONPATH="$TRAIN_ROOT/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
# Resolve effective overrides before importing Isaac. The Python entry point
# enforces the same single ledger, including direct invocations.
M20_LIMIT_VALUES=$("$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/repair_workstation_limits.py "$@")
read -r M20_EFFECTIVE_ENVS M20_EFFECTIVE_UPDATES <<< "$M20_LIMIT_VALUES"
M20_INITIAL_ARGS=(--initial-checkpoint "${M20_INITIAL_CHECKPOINT:-$TRAIN_ROOT/interview_prep/model_35300.pt}")
for M20_TRAIN_ARG in "$@"; do
  if [[ "$M20_TRAIN_ARG" == --resume ]]; then
    M20_INITIAL_ARGS=()
  fi
done
exec "$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V11.1-EntryFlatRepair-v0 --headless \
  --num_envs "$M20_EFFECTIVE_ENVS" --max_iterations "$M20_EFFECTIVE_UPDATES" --seed 42 \
  "${M20_INITIAL_ARGS[@]}" \
  --run_name from35300_entry_flat "$@"
