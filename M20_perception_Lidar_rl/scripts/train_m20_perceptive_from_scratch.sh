#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for arg in "$@"; do
    case "$arg" in
        --resume|--resume=*|--checkpoint|--checkpoint=*|--load_run|--load_run=*)
            echo "ERROR: final perceptive training is strictly from scratch" >&2
            exit 2
            ;;
    esac
done

source /home/ubuntu/.holosoma_deps/miniconda3/bin/activate /home/ubuntu/.holosoma_deps/miniconda3/envs/hssim

ISAACLAB_DIR="/home/ubuntu/.holosoma_deps/IsaacLab"
export PYTHONUNBUFFERED=1
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

exec "$ISAACLAB_DIR/isaaclab.sh" -p \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task M20-Stair-Perceptive-v0 --headless "$@"
