#!/usr/bin/env bash
# V13 63D actor with balanced stair rewards and a 580D asymmetric critic.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
M20_PYTHON="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20_wzh/bin/python}"
[[ -x "$M20_PYTHON" ]] || { echo "Training Python not found: $M20_PYTHON" >&2; exit 2; }
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
"$M20_PYTHON" scripts/tools/test_m20_v13_v11fd_alternation.py
"$M20_PYTHON" scripts/tools/test_m20_v13_critic6d.py
"$M20_PYTHON" scripts/tools/test_m20_v13_critic12d.py
"$M20_PYTHON" scripts/tools/audit_m20_v13_critic12d_reference.py --headless
"$M20_PYTHON" scripts/tools/test_m20_v13_critic12d_runtime.py --headless --num_envs 64 --steps 16
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13-V11FD-Alternation-Critic12D-v0 \
  --seed "${V13_CRITIC12D_SEED:-42}" \
  --max_iterations "${V13_CRITIC12D_MAX_ITERATIONS:-20000}" \
  --num_envs "${V13_CRITIC12D_NUM_ENVS:-8192}" \
  --headless \
  --run_name "${V13_CRITIC12D_RUN_NAME:-v13_actor63_critic580_routecoupled_rollout24_from_scratch_20k}"
