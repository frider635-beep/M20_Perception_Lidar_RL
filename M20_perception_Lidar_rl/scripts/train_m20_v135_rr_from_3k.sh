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

echo "V135-RR: Return Redistribution with gait quality economics"
echo "  - Escrow 30% of V11 task rewards (speed, forward progress)"
echo "  - Release based on gait quality G_k = D_k * S_k * C_k"
echo "  - Weaken skip/dual_skip penalties (economics handles it)"
echo "  - Alternate gait becomes MORE profitable than normal gait"
echo "  - Warm start from V135 3000-iteration checkpoint"
echo ""

# Check if V135 checkpoint exists
V135_CHECKPOINT="${V135_RR_CHECKPOINT:-logs/rsl_rl/m20_stair_v135_native_alternation/*/model_3000.pt}"
if ! compgen -G "$V135_CHECKPOINT" > /dev/null; then
    echo "WARNING: V135 checkpoint not found at $V135_CHECKPOINT"
    echo "Expected: logs/rsl_rl/m20_stair_v135_native_alternation/*/model_3000.pt"
    echo "Will train from scratch instead (not recommended)"
    RESUME_ARG=""
else
    FOUND_CHECKPOINT=$(compgen -G "$V135_CHECKPOINT" | head -1)
    echo "Found V135 checkpoint: $FOUND_CHECKPOINT"
    RESUME_ARG="--resume_path $FOUND_CHECKPOINT"
fi

"$M20_PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-V13.5-RR-ReturnRedistribution-v0 \
  --seed "${V135_RR_SEED:-42}" \
  --num_envs "${V135_RR_NUM_ENVS:-4096}" \
  --max_iterations 30000 \
  --headless \
  --run_name "${V135_RR_RUN_NAME:-v135_rr_gait_economics_from_v135_3k}" \
  $RESUME_ARG
