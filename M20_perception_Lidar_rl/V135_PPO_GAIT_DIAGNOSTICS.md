# V135 Gait-Stratified PPO Diagnostics

**目的**: 诊断 PPO 为什么会偏离交替步态 (alternation gait)

## 背景

V13.1 retention test 发现：即使已学会交替步态，重新训练后也会退化回普通步态。

三种可能原因：

### 1. Sample Imbalance（样本不平衡）
```
p_ALT << p_NORMAL  但  A_ALT > A_NORMAL
```
交替步态很有价值，但样本太少，被普通步态的大量样本淹没。

### 2. Reward Ranking（奖励排序错误）
```
A_ALT < A_NORMAL
```
交替步态的真实回报比普通步态低（奖励设计问题）。

### 3. Gradient Conflict（梯度冲突）
```
A_ALT > A_NORMAL  但策略仍然漂移
```
多目标梯度冲突（例如：交替步态 vs 速度最大化）。

## 诊断指标

### Per-Gait Metrics

对每种步态类型 (NORMAL, ALT, SKIP, DUAL_SKIP) 分别统计：

| 指标 | 含义 | 用途 |
|------|------|------|
| `sample_frac` | 样本占比 | 识别 sample imbalance |
| `advantage_mean` | 平均优势值 | 识别 reward ranking |
| `return_mean` | 平均回报 | 确认真实价值 |
| `surrogate_mean` | PPO代理损失 | 看哪种步态被强化 |
| `clip_frac` | 裁剪比例 | 看更新是否稳定 |

### Cross-Gait Comparisons

| 指标 | 含义 | 判断标准 |
|------|------|----------|
| `ALT_vs_NORMAL_advantage` | 交替 vs 普通的优势差 | >0 = 交替更优 |
| `ALT_vs_NORMAL_return` | 交替 vs 普通的回报差 | >0 = 交替更赚钱 |
| `ALT_vs_NORMAL_sample_ratio` | 交替/普通样本比 | <<1 = 样本失衡 |

## 使用方法

### 方法 1: 训练中实时监控

在 PPO 训练循环中插入诊断 hook（需要修改 rl_games）：

```python
from rl_training.tasks.manager_based.locomotion.velocity.mdp import (
    v135_ppo_gait_diagnostics_hook
)

# In PPO update, after computing advantages:
diag_metrics = v135_ppo_gait_diagnostics_hook(
    env=env,
    advantages=advantages,
    values=values,
    returns=returns,
    old_log_probs=old_log_probs,
    new_log_probs=new_log_probs,
    clip_param=0.2,
)

# Log to TensorBoard
for key, value in diag_metrics.items():
    writer.add_scalar(key, value, iteration)
```

### 方法 2: 离线评估固定 checkpoints

```bash
python scripts/tools/eval_v135_ppo_gait_diagnostics.py \
    --checkpoints \
        logs/m20_stair_v135/model_2800.pt \
        logs/m20_stair_v135/model_3000.pt \
        logs/m20_stair_v135/model_3200.pt \
    --seeds 42,123,456 \
    --num_episodes 100 \
    --output diagnostics_v135.csv
```

## 诊断决策树

```
1. 查看 ALT_vs_NORMAL_sample_ratio
   │
   ├─ < 0.1 (严重失衡)
   │  └─> 问题: Sample Imbalance
   │     解决: 
   │     - 增加交替步态的经验重放权重
   │     - 使用 prioritized experience replay
   │     - 调整探索策略偏向交替
   │
   ├─ 0.1-0.5 (轻度失衡)
   │  └─> 查看 ALT_vs_NORMAL_advantage
   │     │
   │     ├─ > 0 (交替确实更优)
   │     │  └─> 问题: Sample Imbalance (but mild)
   │     │     解决: 同上，或接受（可能自然演化）
   │     │
   │     └─ < 0 (交替不如普通)
   │        └─> 问题: Reward Ranking 错误
   │           解决:
   │           - 检查奖励设计（V135-RR 尝试解决这个）
   │           - 增加交替奖励权重
   │           - 减少速度奖励权重
   │
   └─ > 0.5 (样本充足)
      └─> 查看 ALT_vs_NORMAL_advantage
          │
          ├─ > 0 但策略仍漂移
          │  └─> 问题: Gradient Conflict
          │     解决:
          │     - 多目标优化 (Pareto, weighted sum)
          │     - Separate critics for different objectives
          │     - Curriculum: 先学交替，再加速度
          │
          └─ < 0
             └─> 问题: Reward Ranking 错误
                解决: 同上
```

## 期望观察

### V13.5 (Native Alternation)

如果 V13.5 确实能保持交替：
```
ALT/sample_frac: 0.6-0.8          # 多数样本是交替
ALT/advantage_mean: > 0           # 交替有正优势
ALT_vs_NORMAL_advantage: > 0.5    # 明显优于普通
```

### V135-RR (Return Redistribution)

如果 RR 机制有效：
```
ALT/return_mean > NORMAL/return_mean  # 交替更赚钱
ALT/advantage_mean > NORMAL/advantage_mean  # 优势也更高
ALT/sample_frac 上升趋势            # 策略向交替收敛
```

## TensorBoard 可视化

诊断指标会输出到 TensorBoard 的 `ppo_gait_diag/` 命名空间：

```
ppo_gait_diag/NORMAL/sample_frac
ppo_gait_diag/NORMAL/advantage_mean
ppo_gait_diag/ALT/sample_frac
ppo_gait_diag/ALT/advantage_mean
...
ppo_gait_diag/comparison/ALT_vs_NORMAL_advantage
ppo_gait_diag/comparison/ALT_vs_NORMAL_return
```

## 实现细节

### Gait Classification

基于 V135 riser identity state：
- `NORMAL (0)`: 默认（完成但不是交替）
- `ALT (1)`: `s["ordered"]` = True
- `SKIP (2)`: `s["skip"]` = True  
- `DUAL_SKIP (3)`: `s["dual_skip"]` = True

### Advantage Computation

使用 GAE (Generalized Advantage Estimation)：
```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

注意：诊断在 **episode 层面** 聚合，不是 step 层面。

## 局限性

1. **Snapshot metrics**: 当前实现只看 PPO 更新时的瞬时状态
   - 未追踪历史演化（需要存储 episode trajectories）
   
2. **Causal ambiguity**: 相关性 ≠ 因果
   - A_ALT > A_NORMAL 可能是因为交替策略更好
   - 也可能是因为只有"容易"的 episode 才会交替
   
3. **Multi-step credit assignment**: 
   - 一个 episode 可能同时包含交替和普通步态
   - 当前分类是 per-step，但回报是 per-episode
   
## 下一步

如果诊断确认某个问题：

1. **Sample Imbalance** → 实现 `v135_prioritized_replay.py`
2. **Reward Ranking** → V135-RR 已经在尝试解决
3. **Gradient Conflict** → 实现 `v135_multi_objective_ppo.py`

## 参考

- 设计来源: 2026-09-18 retention test 分析
- 对比实验: V13.1 NO_GO_RETURN_ORDERING
- 理论基础: Schulman et al., PPO (2017)
