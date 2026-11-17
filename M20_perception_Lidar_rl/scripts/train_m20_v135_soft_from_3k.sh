#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
M20_PYTHON="${M20_TRAIN_PYTHON:-/home/ubuntu/miniconda3/envs/m20_wzh/bin/python}"
[[ -x "$M20_PYTHON" ]] || { echo "Training Python not found: $M20_PYTHON" >&2; exit 2; }
export TERM=xterm
export PYTHONPATH="$ROOT_DIR/source/rl_training${PYTHONPATH:+:$PYTHONPATH}"

# Inherit V135 tests to ensure compatibility
"$M20_PYTHON" scripts/tools/test_m20_v13_v11fd_alternation.py
"$M20_PYTHON" scripts/tools/test_m20_v135_riser_identity.py
"$M20_PYTHON" scripts/tools/test_m20_v135_native_alternation.py
"$M20_PYTHON" scripts/tools/audit_m20_v135_reference.py --headless
"$M20_PYTHON" scripts/tools/test_m20_v135_runtime_smoke.py --headless --num_envs 16 --steps 8 --ppo_update

echo "V135-Soft: Soft-learning / Strict-validation approach"
echo "  - Continuous single-leg progress shaping (dense positive signals)"
echo "  - Strict validation for target_id++ (same as V135)"
echo "  - Overlap cost instead of hard support lock"
echo "  - Chain-failed does NOT shutdown local shaping"
echo ""

"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13.5-Soft-v0 \
  --seed "${V135_SOFT_SEED:-42}" \
  --num_envs "${V135_SOFT_NUM_ENVS:-4096}" \
  --max_iterations 30000 \
  --headless \
  --run_name "${V135_SOFT_RUN_NAME:-v135_soft_continuous_progress_gru128_r32_from_3k}"
