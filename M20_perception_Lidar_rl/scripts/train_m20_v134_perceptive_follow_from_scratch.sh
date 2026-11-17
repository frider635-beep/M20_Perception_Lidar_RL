#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
M20_PYTHON="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20_wzh/bin/python}"
[[ -x "$M20_PYTHON" ]] || { echo "Training Python not found: $M20_PYTHON" >&2; exit 2; }
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"
export V134_ASCENT_TEACHER_CHECKPOINT="${V134_ASCENT_TEACHER_CHECKPOINT:-$ROOT_DIR/logs/rsl_rl/M20-V13-V11FD-Alternation/2026-09-09_00-34-12_v13_v11fd_fast_alternation_from_scratch/model_5000.pt}"
[[ -f "$V134_ASCENT_TEACHER_CHECKPOINT" ]] || {
  echo "Reviewed V13 ascent teacher not found: $V134_ASCENT_TEACHER_CHECKPOINT" >&2
  exit 3
}

"$M20_PYTHON" scripts/tools/test_m20_v13_v11fd_alternation.py
"$M20_PYTHON" scripts/tools/test_m20_v133_adjacent_transition.py
"$M20_PYTHON" scripts/tools/test_m20_v134_perceptive_follow.py
"$M20_PYTHON" scripts/tools/audit_m20_v134_reference.py --headless
"$M20_PYTHON" scripts/tools/test_m20_v134_runtime_smoke.py --headless --num_envs 32 --steps 8
"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13.4-V11FD-PerceptiveFollow-v0 \
  --seed "${V134_SEED:-42}" \
  --num_envs "${V134_NUM_ENVS:-2048}" \
  --headless \
  --run_name "${V134_RUN_NAME:-v134_fresh_ascent_teacher_earlytop_r24_2048_10k}"
