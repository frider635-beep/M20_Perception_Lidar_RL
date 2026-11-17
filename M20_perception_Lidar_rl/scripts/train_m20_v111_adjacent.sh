#!/usr/bin/env bash
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export M20_INITIAL_CHECKPOINT="${M20_INITIAL_CHECKPOINT:-$TRAIN_ROOT/interview_prep/model_35300.pt}"
exec bash "$TRAIN_ROOT/scripts/train_m20_v111_entry_flat.sh" \
  --task M20-Stair-V11.1-AdjacentRepair-v0 --run_name from35300_adjacent_straight "$@"
