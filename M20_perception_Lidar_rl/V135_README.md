# V135 Alternation Retention 项目

**解决 M20 人形机器人爬楼梯时交替步态学习后退化的问题**

## 快速开始

```bash
# 1. 运行诊断，识别失败模式
./scripts/run_retention_experiment.sh
# 选择 Option 1: Phase 1 诊断

# 2. 根据诊断结果决定干预方案
# 如果是 Reward Ranking 问题:
./scripts/run_retention_experiment.sh
# 选择 Option 2: Phase 2 训练 V135-RR

# 3. 对比评估
./scripts/run_retention_experiment.sh
# 选择 Option 3: Phase 3 对比
```

## 项目结构

```
M20_perception_Lidar_rl/
├── V135_ALTERNATION_RETENTION_EXPERIMENT.md  # 实验方案总览
├── V135_RR_README.md                          # Return Redistribution 机制
├── V135_PPO_GAIT_DIAGNOSTICS.md              # 诊断指南
│
├── scripts/
│   ├── run_retention_experiment.sh            # 实验启动脚本
│   ├── train_v135_rr.sh                       # V135-RR 训练脚本
│   └── tools/
│       └── eval_v135_ppo_gait_diagnostics.py  # 诊断评估工具
│
└── source/rl_training/.../mdp/
    ├── v135_return_redistribution.py          # RR 实现
    └── v135_ppo_gait_diagnostics.py           # 诊断实现
```

## 核心问题

V13.1 retention test 发现：
- ✓ 3000轮学会交替步态 (ordered_alternation_rate ~0.6)
- ✗ 5000轮退化回普通步态 (rate ~0.3)
- ? 不清楚退化根本原因

## 解决方案

### 1. 诊断工具 (Gait-Stratified PPO Diagnostics)

**目的**: 识别三种失败模式

| 失败模式 | 特征 | 解决方案 |
|---------|------|----------|
| Sample Imbalance | p_ALT << p_NORMAL, A_ALT > 0 | Prioritized replay |
| Reward Ranking | A_ALT < A_NORMAL | V135-RR ✓ |
| Gradient Conflict | A_ALT > 0 但漂移 | Multi-objective PPO |

**关键输出**:
```
ALT/sample_frac         # 交替步态样本占比
ALT/advantage_mean      # 平均优势值
ALT_vs_NORMAL_advantage # 交替 vs 普通的优势差
```

### 2. V135-RR (Return Redistribution)

**假设**: 如果诊断确认 Reward Ranking 问题，RR 应该能解决

**核心机制**:
```python
# 托管 30% 任务奖励，基于步态质量释放
r_immediate = 0.7 * r_task
r_escrow = 0.3 * r_task

# 步态质量评估
G_k = D_k * S_k * C_k
# D_k: 单腿主导性
# S_k: 期望腿占比
# C_k: 物理正确性

# 完成时释放
R_release = E_k * [0.3 + 0.7 * G_k]
```

**期望效果**:
- 交替步态总回报 > 普通步态
- ordered_alternation_rate 稳定在 >0.6
- 继续训练不退化

## TensorBoard 监控

### V135-RR 关键指标

```
v135_rr_escrow          # 托管金额 (应持续增长)
v135_rr_G_k             # 步态质量 (应出现双峰分布)
v135_rr_released        # 释放金额 (与 G_k 正相关)
v135_rr_D_k             # 单腿主导性
v135_rr_S_k             # 期望腿占比
```

### PPO 诊断指标

```
ppo_gait_diag/ALT/sample_frac
ppo_gait_diag/ALT/advantage_mean
ppo_gait_diag/ALT/return_mean
ppo_gait_diag/comparison/ALT_vs_NORMAL_advantage
```

## 文档导航

| 文档 | 用途 |
|------|------|
| [V135_ALTERNATION_RETENTION_EXPERIMENT.md](V135_ALTERNATION_RETENTION_EXPERIMENT.md) | 完整实验方案、时间表、成功标准 |
| [V135_RR_README.md](V135_RR_README.md) | RR 机制详细说明、数学推导 |
| [V135_PPO_GAIT_DIAGNOSTICS.md](V135_PPO_GAIT_DIAGNOSTICS.md) | 诊断工具使用、决策树 |

## 当前状态

- [x] V135-RR 实现完成
- [x] PPO 诊断工具完成
- [x] 训练脚本和文档完成
- [ ] Phase 1: 诊断 V135 3k checkpoint
- [ ] Phase 2: 训练 V135-RR
- [ ] Phase 3: 对比评估

## 联系与反馈

实验日志更新在 `V135_ALTERNATION_RETENTION_EXPERIMENT.md` 底部

---

**下一步**: 运行 `./scripts/run_retention_experiment.sh` 开始 Phase 1 诊断
