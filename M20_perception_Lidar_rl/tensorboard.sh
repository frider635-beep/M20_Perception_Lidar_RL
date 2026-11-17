#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch"
PYTHON_BIN="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Training Python not found: $PYTHON_BIN" >&2; exit 2; }
exec "$PYTHON_BIN" -m tensorboard.main --logdir "$LOG_DIR" --port "${M20_TENSORBOARD_PORT:-6007}" "$@"
