#!/usr/bin/env bash
# Native V13 behavior with a 63D actor and 574D asymmetric critic.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

M20_PYTHON="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20_wzh/bin/python}"
[[ -x "$M20_PYTHON" ]] || { echo "Training Python not found: $M20_PYTHON" >&2; exit 2; }

export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
"$M20_PYTHON" scripts/tools/test_m20_v13_v11fd_alternation.py
"$M20_PYTHON" scripts/tools/test_m20_v13_critic6d.py
"$M20_PYTHON" scripts/tools/test_m20_v13_critic6d_runtime.py --headless --num_envs 16 --steps 8
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13-V11FD-Alternation-Critic6D-v0 \
  --seed "${V13_CRITIC6D_SEED:-42}" \
  --max_iterations "${V13_CRITIC6D_MAX_ITERATIONS:-30000}" \
  --num_envs "${V13_CRITIC6D_NUM_ENVS:-8192}" \
  --headless \
  --run_name "${V13_CRITIC6D_RUN_NAME:-v13_native_task_actor63_critic574_from_scratch}"
