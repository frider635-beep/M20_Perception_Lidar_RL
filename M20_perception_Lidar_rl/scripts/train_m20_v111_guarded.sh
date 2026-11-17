#!/usr/bin/env bash
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TRAIN_ROOT"
M20_PYTHON="${M20_PYTHON:-python3}"
export PYTHONPATH="$TRAIN_ROOT/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
M20_TOTAL=30000
M20_INTERVAL=100
M20_ENVS=4096
M20_CHECK_ONLY=0
M20_BASELINE=''
M20_SESSION=''
while (($#)); do
  case "$1" in
    --check-only) M20_CHECK_ONLY=1; shift ;;
    --total-updates) M20_TOTAL="$2"; shift 2 ;;
    --segment-updates) M20_INTERVAL="$2"; shift 2 ;;
    --num_envs) M20_ENVS="$2"; shift 2 ;;
    --baseline-results) M20_BASELINE="$2"; shift 2 ;;
    --output) M20_SESSION="$2"; shift 2 ;;
    *) echo "Unknown guarded launch argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$M20_BASELINE" || -z "$M20_SESSION" ]]; then
  echo '--baseline-results and a new --output directory are required.' >&2
  exit 2
fi
export M20_RECOVERY_TEACHER="$("$M20_PYTHON" source/rl_training/rl_training/recovery_assets.py --teacher)"
export M20_RECOVERY_REPLAY="$("$M20_PYTHON" source/rl_training/rl_training/recovery_assets.py --replay)"
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/repair_workstation_limits.py \
  --num_envs "$M20_ENVS" --max_iterations "$M20_INTERVAL"
"$M20_PYTHON" scripts/tools/m20_recovery_preflight.py \
  --checkpoint "$M20_RECOVERY_TEACHER" --teacher "$M20_RECOVERY_TEACHER" --replay "$M20_RECOVERY_REPLAY"
"$M20_PYTHON" - "$M20_TOTAL" "$M20_INTERVAL" "$M20_BASELINE" "$M20_SESSION" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('scripts/reinforcement_learning/rsl_rl').resolve()))
from guarded_recovery import TASK, validate_session
validate_session(TASK, int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
print('[M20] Guarded session inputs validated; no simulator started.', flush=True)
print('[M20] Behavior failures are recorded and training continues; runtime/resource checks remain active.', flush=True)
PY
if (( M20_CHECK_ONLY )); then
  exit 0
fi
export M20_REPAIR_RUN_RECEIPT="$M20_SESSION/run_dir.txt"
exec "$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V11.1-AscentRecovery-v0 --headless --device cuda:0 \
  --num_envs "$M20_ENVS" --max_iterations "$M20_INTERVAL" --seed 42 \
  --initial-checkpoint "$M20_RECOVERY_TEACHER" --run_name guarded_30000_from35300 \
  --guarded-total-updates "$M20_TOTAL" --guard-baseline "$M20_BASELINE" --guard-output "$M20_SESSION"
