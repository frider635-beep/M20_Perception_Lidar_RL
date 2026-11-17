#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ubuntu/桌面/js_project/kunyu_ppo"
CONDA_ACTIVATE="/home/ubuntu/miniconda3/bin/activate"
CONDA_ENV="/home/ubuntu/miniconda3/envs/m20"

cd "$PROJECT_DIR"
source "$CONDA_ACTIVATE" "$CONDA_ENV"
export TERM=xterm
export PYTHONPATH="$PROJECT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

exec /home/ubuntu/.holosoma_deps/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V12-V9Teacher-6D-v0 \
  --max_iterations 30000 \
  --num_envs "${NUM_ENVS:-8192}" \
  --headless \
  --run_name "${RUN_NAME:-v12_v9_teacher_6d_from_scratch}" "$@"
