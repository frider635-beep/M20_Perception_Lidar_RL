#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$ROOT_DIR/SHA256SUMS"
PYTHON_BIN="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20/bin/python}"

[[ -r "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Training Python not found: $PYTHON_BIN" >&2; exit 2; }
cd "$ROOT_DIR"
sha256sum --check --quiet "$MANIFEST"
bash -n train_from_scratch.sh resume_from_29999.sh tensorboard.sh
bash -n scripts/train_m20_v11_flat_descent_from_scratch.sh
"$PYTHON_BIN" tools/verify_training_archive.py
echo "M20_PERCEPTION_LIDAR_RL_VERIFY=PASS"
