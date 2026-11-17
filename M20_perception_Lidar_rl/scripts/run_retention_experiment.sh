#!/bin/bash
# V135 Alternation Retention 实验快速启动

set -e

echo "=================================="
echo "V135 Alternation Retention 实验"
echo "=================================="
echo ""

# 检查环境
if [ ! -f "logs/m20_stair_v135/model_3000.pt" ]; then
    echo "❌ 错误: 找不到 V135 3000轮 checkpoint"
    echo "   需要的路径: logs/m20_stair_v135/model_3000.pt"
    echo ""
    echo "请先完成以下步骤:"
    echo "1. 训练 V135 到至少 3000 轮"
    echo "2. 确保 checkpoint 存在"
    exit 1
fi

echo "✓ 找到 V135 3000轮 checkpoint"
echo ""

# 选择实验模式
echo "选择实验模式:"
echo "1) Phase 1: 诊断 (eval V135 3k checkpoint)"
echo "2) Phase 2: 训练 V135-RR (from 3k warm start)"
echo "3) Phase 3: 对比评估 (V135 vs V135-RR)"
echo "4) 完整流程 (Phase 1+2+3)"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "=== Phase 1: 诊断 V135 失败模式 ==="
        echo ""
        python scripts/tools/eval_v135_ppo_gait_diagnostics.py \
            --checkpoints \
                logs/m20_stair_v135/model_2800.pt \
                logs/m20_stair_v135/model_3000.pt \
                logs/m20_stair_v135/model_3200.pt \
            --seeds 42,123,456 \
            --num_episodes 200 \
            --output diagnostics_v135_phase1.csv

        echo ""
        echo "✓ 诊断完成，结果保存到: diagnostics_v135_phase1.csv"
        echo ""
        echo "下一步:"
        echo "1. 查看 CSV 中的关键指标"
        echo "2. 根据 V135_PPO_GAIT_DIAGNOSTICS.md 中的决策树判断失败模式"
        echo "3. 如果是 Reward Ranking 问题，运行 Phase 2"
        ;;

    2)
        echo ""
        echo "=== Phase 2: 训练 V135-RR ==="
        echo ""
        echo "启动训练 (warm start from 3000轮)..."
        ./scripts/train_v135_rr.sh
        ;;

    3)
        echo ""
        echo "=== Phase 3: 对比评估 ==="
        echo ""

        if [ ! -f "logs/m20_stair_v135_rr/model_5000.pt" ]; then
            echo "❌ 错误: 找不到 V135-RR 5000轮 checkpoint"
            echo "   请先完成 Phase 2 训练"
            exit 1
        fi

        python scripts/tools/eval_v135_ppo_gait_diagnostics.py \
            --checkpoints \
                logs/m20_stair_v135/model_3000.pt \
                logs/m20_stair_v135/model_5000.pt \
                logs/m20_stair_v135_rr/model_3000.pt \
                logs/m20_stair_v135_rr/model_5000.pt \
            --seeds 42,123,456 \
            --num_episodes 200 \
            --output comparison_v135_vs_rr.csv

        echo ""
        echo "✓ 对比评估完成"
        echo ""
        echo "检查以下指标判断成功:"
        echo "- ordered_alternation_rate: V135-RR @5k 应该 >= V135 @3k"
        echo "- ALT/return_mean: V135-RR 中 ALT 应该 > NORMAL"
        echo "- 详见 comparison_v135_vs_rr.csv"
        ;;

    4)
        echo ""
        echo "=== 完整流程 ==="
        echo ""

        # Phase 1
        echo "Phase 1: 诊断..."
        python scripts/tools/eval_v135_ppo_gait_diagnostics.py \
            --checkpoints logs/m20_stair_v135/model_3000.pt \
            --seeds 42 \
            --num_episodes 100 \
            --output diagnostics_phase1.csv

        echo ""
        echo "✓ Phase 1 完成"
        echo ""
        read -p "诊断结果确认是 Reward Ranking 问题吗? (y/n): " confirm

        if [ "$confirm" != "y" ]; then
            echo "请根据诊断结果调整实验方案"
            exit 0
        fi

        # Phase 2
        echo ""
        echo "Phase 2: 训练 V135-RR..."
        echo "这将需要数小时到数天，请在 TensorBoard 中监控"
        echo ""
        read -p "是否启动训练? (y/n): " start_train

        if [ "$start_train" = "y" ]; then
            ./scripts/train_v135_rr.sh
        else
            echo "跳过训练，请手动运行: ./scripts/train_v135_rr.sh"
        fi
        ;;

    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "更多信息查看:"
echo "- V135_ALTERNATION_RETENTION_EXPERIMENT.md (实验方案)"
echo "- V135_RR_README.md (RR 机制说明)"
echo "- V135_PPO_GAIT_DIAGNOSTICS.md (诊断指南)"
echo "=================================="
