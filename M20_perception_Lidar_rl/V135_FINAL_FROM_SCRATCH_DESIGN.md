# V13.5-Final From-Scratch 设计方案

**设计日期**: 2026-09-18  
**目标**: 从零训练，不依赖 V13 checkpoint，实现三大目标

## 核心目标

$$
\boxed{
\text{上楼逐级交替}
+
\text{下楼自然支撑转移}
+
\text{楼梯}\rightarrow\text{平地不匍匐}
}
$$

## 设计原则

1. **网络架构**: MLP 主干负责基础 locomotion，小 GRU residual 只负责上楼 gait memory
2. **状态机**: 错误影响成绩但不停止学习，target 自动重同步
3. **上楼**: 正向 dominance shaping，大幅削弱负向 penalties
4. **下楼**: 全新 rear support transfer，不用 V11/V13.4 逻辑
5. **Flat**: Persistent flat mode + crawl guard
6. **PPO**: 恢复 V13 探索强度，促进能力涌现

---

## 一、网络架构重构

### 1.1 Actor 设计

**主干 (Base Action)**:
```
63D observation → MLP [512, 256, 128] → 16D base_action
```

**时序分支 (Gait Memory Residual)**:
```
63D observation → Linear(63→32) → GRU(64) → Linear → 6D front_leg_residual
```

**最终输出**:
```python
a_t = base_action + g_up * front_leg_residual

# g_up: gate for ascent only
# front_leg_residual: only affects FL/FR 6 joints
# residual range: ±0.20 via tanh
```

### 1.2 GRU 初始化与清零

**输出层权重**:
```python
W_out ~ N(0, 1e-3)
bias = 0
```

训练初期 residual ≈ 0，策略退化为纯 MLP。

**GRU 清零条件**:
```python
# 连续 3 帧不在上楼状态
if consecutive_non_ascent >= 3:
    h_gru = 0
```

### 1.3 Ascent Gate

```python
g_up = P1_valid & (H1 > 0.03)

# P1_valid: 上楼有效 riser target
# H1 > 0: 上楼方向
# 下楼 (H1 < 0): g_up = 0
# 平地 (P1_valid = 0): g_up = 0
```

### 1.4 Critic

保持不变:
```
574D → MLP [512, 256, 128] → 1D value
```

不使用 recurrent。

---

## 二、上楼状态机重构

### 2.1 核心变更

**旧**: `ordered_chain_failed` - 一次错误后停止学习  
**新**: `strict_episode_valid` - 只影响成绩，不停止学习

```python
# 只作为成绩标记
strict_episode_valid = True  # episode 开始

# 任何错误发生
if wrong_side or duplicate or skip or dual_skip:
    strict_episode_valid = False  # 本 episode 不算 strict 成功

# 但学习继续！target 继续推进
```

### 2.2 Target 重同步逻辑

**Valid Completion (正确完成)**:
```python
target += step_width
expected_side = flip(expected_side)
```

**Wrong-Side Completion (错边完成)**:
```python
strict_episode_valid = False
target += step_width  # 仍然推进
expected_side = opposite(actual_completed_side)  # 根据实际设置
```

**Skip/Dual-Skip (跳阶)**:
```python
strict_episode_valid = False

# 根据 longitudinal distance 计算实际跨越的台阶数
skip_count = floor(longitudinal_distance / step_width)

# 重同步到实际物理位置
target += skip_count * step_width
expected_side = opposite(actual_responsible_side)
```

### 2.3 学习 Gate 移除

**旧**:
```python
eligible = seeded & (~chain_failed) & target_match
```

**新**:
```python
eligible = seeded & target_valid & target_match
# 不再检查 chain_failed
```

**Critic Sequence**:

**旧**:
```python
started = seeded & (~chain_failed)
```

**新**:
```python
started = seeded
# 犯错后仍然有 gait phase
```

---

## 三、上楼 Reward 重构

### 3.1 删除 Dense Negative Penalties

删除:
- `v13_same_side_progress_penalty`
- `v13_simultaneous_progress_penalty`
- 所有 dense style penalties

### 3.2 Dominance-Based Shaping

**B1 Stage (未 seeded)**:
```python
d_L, d_R = historical_best_deltas  # ≥ 0

leader = max(d_L, d_R)
dominance = |d_L - d_R| / (d_L + d_R + eps)

reward_B1 = leader * dominance
```

**效果**:
- 单腿主导 (d_L=0.1, d_R=0): D≈1, 赚满
- 双腿同时 (d_L=d_R=0.1): D=0, 不赚钱

**B2+ Stage (已 seeded)**:
```python
e = d_expected  # 期望腿
o = d_other     # 另一条腿

share = e / (e + o + eps)

reward_B2 = e * share
```

**效果**:
- 期望腿独立: S≈1, 完整收益
- 两腿同时: S≈0.5, 减半
- 另一腿抢: e≈0, 无收益

### 3.3 Velocity Coupling (保留)

```python
Q_v = exp(-(v_x - v_x_cmd)^2 / sigma^2)

r_gait = 0.5 * r_leader + 1.5 * Q_v * r_leader
```

快速有效的单腿交替最优。

### 3.4 Completion Penalties (大幅削弱)

| Event | 旧权重 | 新权重 |
|-------|--------|--------|
| valid_completion | +3.0 | **+3.0** |
| wrong_side | -1.0 | **-0.25** |
| duplicate | -0.5 | **-0.10** |
| skip | -1.0 | **-0.50** |
| dual_skip | -0.5 | **-0.25** |
| overstep | -1.0 | **-0.25** |

主梯度来自正向 progress，不是 punishment。

---

## 四、下楼全新设计

### 4.1 删除旧模块

不再使用:
- V11 `rear_follow_delay`
- V11 `rear_support_progress`
- V13.4 `vertical_drop_follow`
- `rear_roll_reward`

### 4.2 Rear Support Transfer 三阶段

**Phase 1: Horizontal Crossing (前向跨越)**

```python
p_x^i = clip((x_i - x_{i,0}) / (x_e + r_w + m - x_{i,0}), 0, 1)
```

后轮向前靠近并跨过 riser edge。

**Phase 2: Gated Descent (门控下降)**

```python
p_z^i = clip((z_{i,0} - z_i) / Delta_h, 0, 1)

# 关键：不单独奖励 p_z
p_xz^i = p_x^i * p_z^i
```

**防折腿**: 如果 `p_x≈0` (还没跨过)，即使 `p_z>0` (已经下降)，`p_xz≈0` 不赚钱。

**Phase 3: Stable Support (稳定支撑)**

```python
# 三个条件同时满足 3 帧
if (x_i > x_e + r_w + m) and 
   (|z_bottom - z_lower| < eps_z) and 
   (F_z > F_min) for 3 frames:
    s_i = 1
```

### 4.3 Per-Leg Aggregate

```python
phi_i = 0.45 * p_x^i + 0.35 * p_xz^i + 0.20 * s_i
```

### 4.4 Two-Leg Coordination

```python
P_rear = 0.5 * mean(phi_HL, phi_HR) + 0.5 * min(phi_HL, phi_HR)

# mean: 允许一条先走
# min: 第二条必须跟上
```

**Reward**:
```python
r_rear = [P_rear^best_t - P_rear^best_{t-1}]_+
```

只奖励历史最好增量。

### 4.5 防拖腿: Body-Rear Lag

```python
Delta_x_b = body_forward_distance_since_front_complete
Delta_x_r = min(Delta_x_HL, Delta_x_HR)  # 最慢后腿

L = Delta_x_b - Delta_x_r

# 允许 6cm 自然 lag
c_lag = [(L - 0.06) / 0.10]_+^2
c_lag = clip(c_lag, max=1)

penalty = -0.6 * c_lag
```

### 4.6 防折腿: Premature Collapse

```python
# 如果还没向前 (p_x < 0.3)，就已经下坠 (Delta_z > 0.04)
c_collapse^i = (1 - p_x^i) * [(Delta_z_i - 0.04) / 0.08]_+^2

penalty = -0.35 * c_collapse
```

### 4.7 速度收益调制

```python
# Rear transfer active 期间
g_d = 0.3 + 0.7 * P_rear

r_forward_eff = g_d * r_forward
r_velocity_eff = g_d * r_velocity

# 后腿没下来：只能兑现 30%
# 后腿跟下来：恢复完整收益
```

同时关闭 speed-deficit penalty，允许自然减速换支撑。

### 4.8 Knee Penalty (仅 Safety)

```python
# 极弱，只防极端折叠
knee_penalty = -0.15 * knee_flexion_cost
```

不负责教学。

---

## 五、Persistent FLAT Mode

### 5.1 状态转换

```
STAIR → RECOVERY → FLAT (persistent)
```

**进入 RECOVERY**:
- 上楼: `platform_exit_event`
- 下楼: `rear_transfer_complete` (不是 `front_complete`)

**确认 RECOVERY**:
```python
# 连续条件
no_riser (10 frames) &
all_four_contact &
coplanar &
(clearance + upright + velocity quality >= 0.80)
for 5 frames

→ flat_mode = True
```

**FLAT Mode 持续**:
```python
# 只要在平地，就持续生效
flat_mode &= raw_no_riser

# 直到遇到新 riser 或 episode reset
```

### 5.2 Flat-Only Crawl Penalty

```python
rho_h = h_clearance / h_neutral

if flat_mode and raw_no_riser:
    if rho_h < 0.90:
        deficit = (0.90 - rho_h) / 0.12
        c_crawl = clip(deficit^2, max=1)
        penalty = -1.0 * c_crawl
```

**关键**: 只在确认离开楼梯后生效，不干扰上下楼自然步态。

---

## 六、Command Split

### 6.1 Ascent Command

```python
v_x_up ∈ [0.10, 2.0] m/s  # 保持 V13 范围
```

高速压力有助于能力涌现，不改变。

### 6.2 Descent Command

```python
v_x_down ∈ [0.15, 0.80] m/s  # 大幅降低
```

从训练第 0 轮起，下楼不需要追 2m/s。

### 6.3 Flat Command

```python
v_x_flat ∈ [0.10, 2.0] m/s  # 保持 V11 范围
```

---

## 七、PPO 配置恢复 V13 探索强度

### 7.1 从 Fine-Tune 回到 From-Scratch

| 参数 | Fine-tune (之前) | From-Scratch (新) | 理由 |
|------|------------------|-------------------|------|
| `num_envs` | 4096 | **4096** | 保持 |
| `num_steps_per_env` | 48 | **32** | 减少 wall-clock，足够 BPTT |
| `learning_rate` | 3e-4 | **1e-3** | 恢复探索，促进涌现 |
| `entropy_coef` | 1e-3 | **3e-3** | 恢复探索 |
| `num_learning_epochs` | 3 | **5** | 恢复 V13 设置 |
| `desired_kl` | 0.008 | **0.01** | 恢复 V13 设置 |
| `init_std` | 1.0 | **1.0** | 保持 |
| `max_grad_norm` | 1.0 | **1.0** | 保持 |

### 7.2 初始化

```python
# 全部随机初始化
resume = False
load_run = None
load_checkpoint = None
```

---

## 八、Terrain Curriculum

### 8.1 不绑定 Strict Gait

**错误**:
```python
# 不要这样
if strict_ordered_success_rate > 0.8:
    advance_terrain_level()
```

**正确**:
```python
# 基于 basic traversability
if progress_rate > threshold:
    advance_terrain_level()
```

机器人应该先学会"与楼梯有意义的互动"，再优化 gait。

---

## 九、清理历史模块

### 9.1 不引用的模块

保留源码但不在 V135-Final task 中引用:
- `v131_ordered_alternation.py` (V13.1 strict reward)
- `v132_barrier_alternation.py` (double critic)
- `v133_adjacent_transition.py` (chain settlement)
- `v12_teacher_*` (teacher)
- `v135_prefix_replay.py`
- `v135_return_redistribution.py` (RR)
- V11 `rear_follow_delay`
- V13.4 `teacher_burst`

### 9.2 V11/V13 文件保持不动

不修改:
- `v11_continuous.py`
- `v11_landing.py`
- `v13_alternation.py`
- `v13_v11fd_alternation_env_cfg.py`

---

## 十、最终 Reward Ledger

### 10.1 Ascent (上楼)

| Term | Weight | Note |
|------|--------|------|
| Leader progress (bare) | +0.5 | Dominance-based |
| Leader progress (vel-coupled) | +1.5 | Q_v * leader |
| Valid completion | +3.0 | Strict correct |
| Wrong-side | -0.25 | 削弱 |
| Duplicate | -0.10 | 削弱 |
| Skip | -0.50 | 削弱 |
| Dual-skip | -0.25 | 削弱 |
| Overstep | -0.25 | 削弱 |

### 10.2 Descent (下楼)

| Term | Weight | Note |
|------|--------|------|
| Rear transfer progress | +1.0 | Historical best |
| Body-rear lag | -0.6 | 防拖腿 |
| Premature collapse | -0.35 | 防折腿 |
| Knee safety | -0.15 | 仅极端 |

### 10.3 Post-Stair (楼梯后)

| Term | Weight | Note |
|------|--------|------|
| Recovery progress | +0.5 | Quality delta |
| Recovery completion | +3.0 | Sparse |
| Flat crawl guard | -1.0 | Persistent |

### 10.4 Native V11 Rewards (保持)

- `track_lin_vel_xy` (下楼期间调制)
- `track_ang_vel_z` (下楼期间调制)
- `lin_vel_z_l2`
- `ang_vel_xy_l2`
- `orientation_l2`
- `dof_torques_l2`
- `dof_acc_l2`
- `action_rate_l2`
- `dof_pos_limits`
- (其他 V11 原生)

---

## 十一、实施检查清单

### Phase 1: Network Architecture
- [ ] 创建 `V135FromScratchActorCritic` 类
- [ ] MLP 主干: 63→512→256→128→16
- [ ] GRU 分支: 63→32→GRU64→6D residual
- [ ] 实现 ascent gate 和 GRU 清零
- [ ] Residual 初始化和限幅

### Phase 2: Ascent State Machine
- [ ] 修改 `v135_riser_identity.py`
- [ ] `ordered_chain_failed` → `strict_episode_valid`
- [ ] 实现 target 重同步 (wrong-side, skip)
- [ ] 移除 `eligible` 和 `started` 的 chain_failed 检查

### Phase 3: Ascent Rewards
- [ ] 修改 `v135_final_ascent_reward.py`
- [ ] 实现 B1 dominance shaping
- [ ] 实现 B2+ expected share shaping
- [ ] Velocity coupling
- [ ] 削弱 completion penalties

### Phase 4: Descent (全新)
- [ ] 创建 `v135_descent_support_transfer.py`
- [ ] 实现 p_x, p_xz, stable support
- [ ] Two-leg coordination
- [ ] Body-rear lag penalty
- [ ] Premature collapse penalty
- [ ] 速度收益调制

### Phase 5: Post-Stair
- [ ] 修改 `v135_descent_stance.py`
- [ ] Persistent flat_mode
- [ ] Flat-only crawl penalty
- [ ] Traversal completion 改用 rear_transfer_complete

### Phase 6: Config
- [ ] 修改 `v135_v11fd_native_alternation_env_cfg.py`
- [ ] 整合所有新 reward terms
- [ ] Command split (ascent/descent/flat)
- [ ] 修改 `rsl_rl_ppo_cfg.py`
- [ ] 恢复 V13 PPO 参数

### Phase 7: Cleanup
- [ ] 移除 task config 中对历史模块的引用
- [ ] 保留源码但不引用
- [ ] 确认 terrain curriculum 不绑定 strict gait

### Phase 8: Validation
- [ ] Python 语法编译
- [ ] Isaac Sim 16 env smoke test
- [ ] Recurrent PPO update test
- [ ] V13 contract tests

---

## 十二、预期训练动态

### 0-500 轮: Feed-Forward Dominance

- GRU residual ≈ 0
- 策略近似纯 MLP
- 学习基础 locomotion 和楼梯互动
- 预期出现不稳定的交替萌芽

### 500-1500 轮: Gait Emergence

- GRU 开始学习 phase memory
- Dominance shaping 引导单腿主导
- 预期出现明显的交替步态
- 下楼 support transfer 逐渐形成

### 1500-3000 轮: Gait Refinement

- 交替步态稳定
- 速度与 gait 耦合
- 下楼不拖腿、不折腿
- Flat mode 防匍匐

### 3000+ 轮: Stability Test

- 交替步态不应退化
- ordered_alternation_rate 应保持 >0.5
- 所有三大目标应稳定实现

---

## 十三、成功标准

### Minimal Success

- **1500 轮内**: ordered_alternation_rate > 0.3
- **3000 轮**: ordered_alternation_rate > 0.5
- **5000 轮**: 保持不退化

### Full Success

- **3000 轮**: ordered_alternation_rate > 0.7
- 下楼无拖腿、无折腿
- 楼梯后无匍匐
- 继续到 8000 轮不退化

---

## 十四、与之前方案的对比

| 维度 | 之前 (Fine-tune) | 现在 (From-Scratch) |
|------|------------------|---------------------|
| 网络 | 完整 GRU128 | MLP + 小 GRU residual |
| 初始化 | 从 V135 3000 轮 | 全部随机 |
| PPO LR | 3e-4 | 1e-3 |
| Rollout | 48 | 32 |
| 下楼 | V13.4 vertical-drop | 全新 support transfer |
| 哲学 | 稳定已涌现能力 | 促进能力自然涌现 |

---

## 十五、风险与应对

### 风险 1: 涌现延迟

**现象**: 3000 轮仍未出现明显交替

**应对**:
- 检查 GRU residual 是否被过度初始化抑制
- 检查 dominance shaping 是否生效
- 考虑提高 entropy 到 0.005

### 风险 2: 下楼拖腿

**现象**: body-rear lag penalty 不足

**应对**:
- 提高 lag penalty 权重 0.6 → 0.8
- 降低速度收益调制基线 0.3 → 0.2

### 风险 3: GRU 不收敛

**现象**: GRU 始终输出噪声

**应对**:
- 检查 ascent gate 是否正确
- 检查 GRU 清零是否正确
- 考虑增加 GRU hidden size 64 → 96

---

**总结**: V13.5-Final From-Scratch 是一个"早期近似 V13 MLP + 逐渐学习 gait memory"的设计，配合全新的下楼 support transfer 和 persistent flat mode，从零训练实现三大目标。
