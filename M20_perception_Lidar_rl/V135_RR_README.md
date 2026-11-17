# V13.5-RR: Return Redistribution

**核心思路**：改变奖励经济学，让交替步态比普通步态更赚钱

## 设计理念

V13.5 的问题：通过检测错误和惩罚来引导，但普通步态仍然能通过高速度和前进获得足够高的总回报。

V13.5-RR 的解决方案：**把 V11 的高价值任务奖励与步态质量绑定**

### 奖励经济学对比

| 步态类型 | V13.5 | V13.5-RR |
|---------|-------|----------|
| **普通步态** | 高任务奖励 - 小交替惩罚 | 70%任务奖励 + 低G_k释放(~30%) |
| **交替步态** | 高任务奖励 + 小交替奖励 | 70%任务奖励 + 高G_k释放(~100%) |
| **谁更赚钱** | 普通步态 | **交替步态** ✓ |

## 核心机制

### 1. Escrow（托管）30%任务奖励

```python
# 每一步
r_immediate = 0.7 * r_task  # 立即发放70%
r_escrow = 0.3 * r_task     # 托管30%，累积到当前riser
```

### 2. 步态质量计算

```
G_k = D_k * S_k * C_k

其中：
- D_k: 单腿主导性 = |p_FL - p_FR| / (p_FL + p_FR)
  → 1 = 单腿主导，0 = 双腿同时
  
- S_k: 期望腿占比 = p_expected / (p_FL + p_FR)  
  → 1 = 只有期望腿，0 = 只有另一条腿
  
- C_k: 物理正确性 = {1 if valid, 0 if skip/dual_skip}
  → 防止跳阶
```

### 3. 基于步态质量释放托管奖励

```
valid completion 时释放：
R_release = E_k * [0.3 + 0.7 * G_k]

例子：
- 普通步态 (G_k=0.2): 释放 37% 的托管
- 交替步态 (G_k=0.9): 释放 93% 的托管
```

## 与 V13.5 的主要区别

| 特性 | V13.5 | V13.5-RR |
|------|-------|----------|
| 奖励结构 | 事件驱动（完成/错误） | 经济驱动（托管+释放） |
| skip惩罚 | -1.0 | -0.3（经济已处理） |
| dual_skip惩罚 | -2.0 | -0.5（经济已处理） |
| 学习信号 | 稀疏（只在完成时） | 托管持续积累 |
| 优化目标 | 避免错误 | 最大化总回报 |

## 保持不变的部分

- GRU128, 63D Actor, 574D Critic
- Physical riser identity (B1/B2/B3)
- P1/P2 elevation pattern
- B1 任意腿先迈
- Strict target_id++ validation
- Post-stair recovery
- Descent follow

## 训练启动

```bash
# 从 V135 3000轮 checkpoint warm start（能力涌现点）
./scripts/train_v135_rr.sh

# 或手动指定参数
./scripts/train_m20_v135_rr_from_3k.sh
```

## 期望结果

### 100-300轮内应该看到：

1. **托管金额持续积累** (`v135_rr_escrow` 上升)
2. **步态质量开始分化** (`v135_rr_G_k` 出现高低两组)
3. **释放金额反映质量** (高G_k episode释放更多)

### 1000轮后应该看到：

4. **ordered_alternation_rate 上升** (交替步态占比增加)
5. **Episode return 分布**：交替步态 > 普通步态
6. **target_id 分布扩散** (更多B2/B3完成)

## TensorBoard 关键指标

```
v135_rr_D_k            # 单腿主导性
v135_rr_S_k            # 期望腿占比
v135_rr_C_k            # 物理正确性
v135_rr_G_k            # 综合步态质量
v135_rr_escrow         # 当前托管金额
v135_rr_released       # 释放金额
v135_riser_ordered_alternation_rate  # 交替率
v135_riser_target_id_mean             # 平均完成阶数
```

## 判断标准

**成功标志**：
```
J_alternate > J_normal  # 交替步态的总回报更高
ordered_rate ↑         # 交替发生率上升
G_k 分布双峰           # 明显区分好坏步态
```

**失败标志**：
```
J_normal > J_alternate  # 普通步态仍更赚钱
→ 需要调整 η (escrow fraction) 或 g0 (baseline release)
```

## 参数调节指南

如果交替优势不够明显：

1. **增加托管比例**: `η: 0.3 → 0.5`（更多奖励与步态绑定）
2. **降低基线释放**: `g0: 0.3 → 0.2`（放大质量差异）
3. **提高质量指数**: `α,β,γ: 1.0 → 1.5`（放大G_k差异）

## 参考

- 设计来源: 2026-09-18 用户提议 "把 V11 的高价值 task reward 与交替质量绑定"
- Base: V13.5 (Physical riser identity + GRU128)
- 核心改变: 奖励经济学，不是行为规则
