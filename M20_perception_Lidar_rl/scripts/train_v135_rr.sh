#!/bin/bash
# V13.5-RR Training Script
# Return Redistribution: bind V11 task reward to gait quality
# From 3000-iteration V135 checkpoint warm start

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# V135 checkpoint at ~3000 iterations (capability emergence point)
V135_CHECKPOINT="/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/logs/rsl_rl/M20-V13.5-V11FD-NativeAlternation/2026-09-18_14-38-55_v135_strict_identity_gru128_r48_from_scratch_30k/model_3000.pt"

if [ ! -f "$V135_CHECKPOINT" ]; then
    echo "❌ V135 checkpoint not found: $V135_CHECKPOINT"
    echo "Expected V135 run at ~3000 iterations for warm start"
    exit 1
fi

echo "🚀 Starting V13.5-RR training (Return Redistribution)"
echo "📦 Warm start from: $V135_CHECKPOINT"
echo "🎯 Key changes:"
echo "   - Escrow 30% of V11 task rewards"
echo "   - Release proportional to gait quality G_k"
echo "   - Weaken skip/dual_skip penalties (0.5 → 0.1)"
echo "   - Rollout: 4096×32 (faster iteration)"
echo ""

python source/rl_training/scripts/rsl_rl/train.py \
    --task M20-Stair-V13.5-RR-ReturnRedistribution-v0 \
    --num_envs 32 \
    --headless \
    --resume \
    --load_run "$V135_CHECKPOINT" \
    "$@"
