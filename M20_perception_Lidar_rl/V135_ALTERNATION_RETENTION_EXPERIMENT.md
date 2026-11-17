# V135 Alternation Retention 实验方案

**目标**: 解决 V13.1 retention test 暴露的问题 - 已学会的交替步态在继续训练后会退化

## 问题陈述

### 观察到的现象

V13.1 在 NO_GO_RETURN_ORDERING 测试中：
1. ✓ 3000轮时能学会交替步态 (ordered_alternation_rate ~0.6)
2. ✗ 继续训练到 5000轮后退化回普通步态 (rate 降到 ~0.3)
3. ? 不清楚退化的根本原因

### 核心问题

**为什么 PPO 会偏离已经学会的交替步态？**

## 实验设计

### Phase 1: 诊断 - 识别失败模式

**目标**: 确定 V135 在 3000→5000 轮之间发生了什么

**方法**: 使用 gait-stratified PPO diagnostics

```bash
# 评估关键 checkpoints
python scripts/tools/eval_v135_ppo_gait_diagnostics.py \
    --checkpoints \
        logs/m20_stair_v135/model_2800.pt \
        logs/m20_stair_v135/model_3000.pt \
        logs/m20_stair_v135/model_3200.pt \
        logs/m20_stair_v135/model_3500.pt \
    --seeds 42,123,456 \
    --num_episodes 200 \
    --output diagnostics_v135_3k_5k.csv
```

**判断标准**:

1. **Sample Imbalance** (样本失衡)
   ```
   症状: ALT_vs_NORMAL_sample_ratio < 0.2
        且 ALT/advantage_mean > 0
   
   解释: 交替步态很好，但样本太少
   解决: Prioritized experience replay
   ```

2. **Reward Ranking** (奖励排序错误)
   ```
   症状: ALT/return_mean < NORMAL/return_mean
        或 ALT/advantage_mean < 0
   
   解释: 普通步态的总回报更高
   解决: V135-RR (Return Redistribution)
   ```

3. **Gradient Conflict** (梯度冲突)
   ```
   症状: ALT/advantage_mean > 0
        且 ALT/sample_frac 稳定但策略仍漂移
   
   解释: 多目标优化冲突（交替 vs 速度）
   解决: Multi-objective PPO 或 curriculum
   ```

### Phase 2: 干预 - V135-RR

**假设**: 如果诊断确认 Reward Ranking 问题，RR 机制应该能解决

**实验配置**:

```bash
# Warm start from V135 3000轮 checkpoint
./scripts/train_v135_rr.sh

# 对比组: 继续训练原始 V135
./scripts/train_m20_v135.sh --resume logs/.../model_3000.pt
```

**V135-RR 核心机制**:

```python
# 1. Escrow 30% 任务奖励
r_immediate = 0.7 * r_task
r_escrow = 0.3 * r_task  # 累积到当前 riser

# 2. 计算步态质量
G_k = D_k * S_k * C_k
# D_k: 单腿主导性 [0,1]
# S_k: 期望腿占比 [0,1]
# C_k: 物理正确性 {0,1}

# 3. 基于质量释放托管
on_valid_completion:
    R_release = E_k * [0.3 + 0.7 * G_k]
```

**期望结果**:

```
时间范围: 100-1000 轮

100-300 轮应该看到:
- v135_rr_escrow 持续积累 (托管金额上升)
- v135_rr_G_k 出现双峰分布 (好坏步态分化)
- v135_rr_released 与 G_k 正相关

1000 轮后应该看到:
- ordered_alternation_rate 上升 (交替占比增加)
- Episode return 分布: ALT > NORMAL
- target_id 分布扩散 (更多高阶完成)
```

**成功判据**:

```
核心: J_alternate > J_normal
即: 交替步态的期望总回报 > 普通步态

验证方法:
1. TensorBoard: ALT/return_mean > NORMAL/return_mean
2. Behavior: ordered_alternation_rate 稳定在 >0.6
3. Robustness: 继续训练到 5000 轮不退化
```

### Phase 3: 对比实验矩阵

| 实验组 | Checkpoint | 机制 | 预期结果 |
|-------|-----------|------|----------|
| **V135-baseline** | 从头训练 | 原始 V135 | 3000轮学会，5000轮退化 |
| **V135-continue** | 3000轮续训 | 原始 V135 | 退化（对照组） |
| **V135-RR** | 3000轮续训 | Return Redistribution | 保持或提升 |
| **V135-RR-scratch** | 从头训练 | Return Redistribution | 直接学到交替 |

### Phase 4: 消融实验

如果 V135-RR 成功，验证哪个机制起作用：

```python
# Ablation 1: 只有 escrow，无质量调制
R_release = E_k * 1.0  # 全部释放

# Ablation 2: 只有质量调制，无 escrow
r_immediate = r_task  # 全部即时发放
quality_bonus = 0.3 * r_task * G_k  # 质量加成

# Ablation 3: 不同 escrow 比例
η ∈ {0.1, 0.3, 0.5, 0.7}

# Ablation 4: 不同基线释放
g0 ∈ {0.0, 0.2, 0.3, 0.5}
```

## 实验执行计划

### Week 1: 诊断

```bash
# Day 1-2: 收集 V135 3k→5k 的诊断数据
python scripts/tools/eval_v135_ppo_gait_diagnostics.py ...

# Day 3: 分析失败模式
# - 制作可视化图表
# - 确定主导因素
# - 决定干预方案
```

### Week 2: V135-RR 训练

```bash
# Day 1: 启动 V135-RR 从 3000轮 warm start
./scripts/train_v135_rr.sh

# Day 2-7: 监控关键指标
# - v135_rr_escrow (应持续增长)
# - v135_rr_G_k (应出现双峰)
# - ordered_alternation_rate (应上升)
# - 每天截图 TensorBoard 贴进实验日志
```

### Week 3: 验证与对比

```bash
# 评估 V135-RR 最终效果
python scripts/tools/eval_v135_ppo_gait_diagnostics.py \
    --checkpoints \
        logs/v135_baseline/model_5000.pt \
        logs/v135_rr/model_5000.pt \
    --output comparison_v135_vs_rr.csv

# 如果成功，启动消融实验
# 如果失败，回到 Phase 1 重新诊断
```

## 数据收集清单

### 每 100 轮记录

- [ ] TensorBoard 截图 (所有 v135_rr_* 指标)
- [ ] 诊断 CSV (gait-stratified metrics)
- [ ] Checkpoint 备份
- [ ] 行为视频 (optional，关键轮次)

### 关键对比

| 指标 | V135 @3k | V135 @5k | V135-RR @3k | V135-RR @5k |
|------|----------|----------|-------------|-------------|
| ordered_alternation_rate | | | | |
| ALT/return_mean | | | | |
| NORMAL/return_mean | | | | |
| ALT_vs_NORMAL_advantage | | | | |
| target_id_mean | | | | |

## 失败应对预案

### 如果 V135-RR 不工作

1. **Escrow 不够强**
   ```python
   # 增加托管比例
   η: 0.3 → 0.5
   
   # 降低基线释放
   g0: 0.3 → 0.1
   ```

2. **质量度量不敏感**
   ```python
   # 增加质量指数
   G_k = (D_k^α * S_k^β * C_k)^γ
   α, β, γ: 1.0 → 1.5
   ```

3. **根本不是 Reward Ranking 问题**
   ```
   回到 Phase 1，重新诊断
   可能需要实现其他解决方案:
   - Prioritized replay (if Sample Imbalance)
   - Multi-objective PPO (if Gradient Conflict)
   ```

## 预期产出

### 代码

- [x] v135_return_redistribution.py
- [x] v135_ppo_gait_diagnostics.py
- [x] eval_v135_ppo_gait_diagnostics.py
- [ ] (如需要) v135_prioritized_replay.py
- [ ] (如需要) v135_multi_objective_ppo.py

### 文档

- [x] V135_RR_README.md
- [x] V135_PPO_GAIT_DIAGNOSTICS.md
- [x] V135_ALTERNATION_RETENTION_EXPERIMENT.md (本文档)
- [ ] 实验报告 (Week 3 产出)

### 数据

- [ ] diagnostics_v135_3k_5k.csv (Phase 1)
- [ ] comparison_v135_vs_rr.csv (Phase 3)
- [ ] TensorBoard logs (持续)
- [ ] 行为视频 (可选)

## 成功标准

### Minimal Success (最小成功)

```
V135-RR @5000轮:
- ordered_alternation_rate > 0.5
- 不低于 @3000轮 的水平
```

### Full Success (完全成功)

```
V135-RR @5000轮:
- ordered_alternation_rate > 0.7
- ALT/return_mean 显著高于 NORMAL/return_mean
- 行为稳定（继续训练到 8000 轮不退化）
- 机制可解释（消融实验确认关键因素）
```

## 参考文献

- V13.1 retention test analysis (2026-09-18)
- PPO paper: Schulman et al. (2017)
- Reward shaping: Ng et al. (1999)
- Multi-objective RL: Roijers et al. (2013)

## 实验日志

### 2026-09-18

- [x] 实现 V135-RR 核心机制
- [x] 实现 gait-stratified PPO diagnostics
- [x] 创建训练脚本和文档
- [ ] 启动 Phase 1 诊断实验

---

**下一步**: 在 V135 3000轮 checkpoint 上运行诊断，确认失败模式
