#!/usr/bin/env bash
set -euo pipefail
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
M20_PYTHON="${M20_PYTHON:-python3}"
if ! command -v "$M20_PYTHON" >/dev/null 2>&1; then
  echo "Python environment not found: $M20_PYTHON; set M20_PYTHON to an environment with torch, mujoco, numpy, PyYAML." >&2
  exit 1
fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec "$M20_PYTHON" "$SCRIPT_ROOT/deploy/mujoco_v135/run.py" "$@"
