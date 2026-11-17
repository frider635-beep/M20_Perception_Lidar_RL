#!/usr/bin/env bash
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_ROOT"
M20_PYTHON="${M20_PYTHON:-python3}"
M20_CHECK_ONLY=0
M20_TRAIN_ARGS=()
for M20_ARG in "$@"; do
  if [[ "$M20_ARG" == --check-only ]]; then
    M20_CHECK_ONLY=1
  else
    M20_TRAIN_ARGS+=("$M20_ARG")
  fi
done
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/repair_workstation_limits.py "${M20_TRAIN_ARGS[@]}"
export M20_RECOVERY_TEACHER="$("$M20_PYTHON" source/rl_training/rl_training/recovery_assets.py --teacher)"
export M20_RECOVERY_REPLAY="$("$M20_PYTHON" source/rl_training/rl_training/recovery_assets.py --replay)"
export M20_INITIAL_CHECKPOINT="${M20_INITIAL_CHECKPOINT:-$M20_RECOVERY_TEACHER}"
M20_EXPECT_VALUE=''
for M20_ARG in "${M20_TRAIN_ARGS[@]}"; do
  if [[ "$M20_EXPECT_VALUE" == checkpoint ]]; then
    export M20_INITIAL_CHECKPOINT="$M20_ARG"
    M20_EXPECT_VALUE=''
    continue
  elif [[ "$M20_EXPECT_VALUE" == task ]]; then
    if [[ "$M20_ARG" != M20-Stair-V11.1-AscentRecovery-v0 ]]; then
      echo 'The recovery launcher only accepts the AscentRecovery task.' >&2
      exit 2
    fi
    M20_EXPECT_VALUE=''
    continue
  fi
  case "$M20_ARG" in
    --initial-checkpoint) M20_EXPECT_VALUE=checkpoint ;;
    --initial-checkpoint=*) export M20_INITIAL_CHECKPOINT="${M20_ARG#*=}" ;;
    --task) M20_EXPECT_VALUE=task ;;
    --task=*)
      if [[ "${M20_ARG#*=}" != M20-Stair-V11.1-AscentRecovery-v0 ]]; then
        echo 'The recovery launcher only accepts the AscentRecovery task.' >&2
        exit 2
      fi ;;
    --resume|--resume=*)
      echo 'Recovery stages start with fresh optimizer state. Select a saved model with --initial-checkpoint PATH.' >&2
      exit 2 ;;
  esac
done
if [[ -n "$M20_EXPECT_VALUE" ]]; then
  echo 'Missing value for --task or --initial-checkpoint.' >&2
  exit 2
fi
M20_VALIDATION_ROOT="${M20_VALIDATION_ROOT:-$TRAIN_ROOT/../mujoco_validation}"
"$M20_PYTHON" scripts/tools/m20_recovery_preflight.py \
  --checkpoint "$M20_INITIAL_CHECKPOINT" --teacher "$M20_RECOVERY_TEACHER" \
  --replay "$M20_RECOVERY_REPLAY" --validation-root "$M20_VALIDATION_ROOT"
if (( M20_CHECK_ONLY )); then
  exit 0
fi
M20_STAGE_DIR="$TRAIN_ROOT/artifacts/m20_repair_stages/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$M20_STAGE_DIR"
export M20_REPAIR_RUN_RECEIPT="$M20_STAGE_DIR/run_dir.txt"
bash "$TRAIN_ROOT/scripts/train_m20_v111_entry_flat.sh" \
  --task M20-Stair-V11.1-AscentRecovery-v0 --run_name reference_anchored "${M20_TRAIN_ARGS[@]}"
IFS= read -r M20_SAVED_RUN < "$M20_REPAIR_RUN_RECEIPT"
"$M20_PYTHON" scripts/tools/m20_repair_acceptance.py \
  --run-dir "$M20_SAVED_RUN" --output "$M20_STAGE_DIR/acceptance" \
  --validation-root "$M20_VALIDATION_ROOT"
