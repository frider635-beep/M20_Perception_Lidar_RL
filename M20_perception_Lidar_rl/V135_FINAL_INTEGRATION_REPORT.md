# V13.5-Final 整合完成报告

## 执行日期
2026-09-18

## 核心目标
将 V135 收束为最终版本，删除 RR 分支，整合所有改动到单一稳定版本。

## 关键修改

### 1. 状态机重构 (v135_riser_identity.py)

**核心变更**: `ordered_chain_failed` → `episode_ordered_valid`

**语义转变**:
- 旧: 一次错误后，`ordered_chain_failed=True`，**停止后续学习**
- 新: 一次错误后，`episode_ordered_valid=False`，**只影响成绩，不停止学习**

**Target 重同步逻辑**:
```python
# Valid completion: 标准推进
target += step_width
expected_side = flip

# Wrong-side: 推进 target，根据实际完成腿设置 expected
target += step_width
expected_side = opposite(actual_completed_side)

# Skip/dual-skip: 重同步到实际物理位置
skip_count = (longitudinal / step_width).floor()
target += skip_count * step_width
expected_side = opposite(actual_responsible_side)
```

**影响范围**:
- ✓ Completion 不再关闭学习
- ✓ Overstep 不再使用 chain_failed
- ✓ Metric 改为 episode_valid_rate

### 2. 上楼 Reward 重构 (v135_final_ascent_reward.py)

**删除**:
- 所有 dense negative style rewards (same-side penalty, simultaneous penalty)
- 不再使用 V13 的复杂 penalty 堆叠

**新增**: Dominance-based positive shaping

```python
# B1 stage: 奖励主导腿
leader = max(dL, dR)
dominance = |dL - dR| / (dL + dR + eps)
reward_b1 = leader * dominance

# B2+ stage: 奖励期望腿的占比
expected_delta = d_expected
other_delta = d_other
share = expected / (expected + other + eps)
reward_b2 = expected * share
```

**权重设置**:
- Bare leader progress: 0.5x
- Velocity-coupled leader progress: 1.5x

**Completion penalties** (大幅削弱):
- valid_completion: +3.0 (不变)
- wrong_side: -1.0 → **-0.25**
- duplicate: -0.5 → **-0.1**
- skip: -1.0 → **-0.25**
- dual_skip: -0.5 → **-0.25**
- overstep: -1.0 → **-0.25**

### 3. 下楼约束完善 (v135_descent_stance.py)

**新增三个 follow-active 约束**:

```python
# 1. Rear knee folding (新增)
v135_descent_rear_knee_folding_penalty
- free_flexion: 0.20
- flexion_scale: 0.25
- weight: 0.50
- 只在 descent follow active 期间生效

# 2. Rear extension (参数收紧)
v135_descent_rear_extension_penalty
- free_reach: 0.08 → 0.06
- weight: 0.50 → 0.75

# 3. Rear roll (权重降低)
v135_descent_rear_roll_penalty
- weight: 0.20 → 0.12
```

**Traversal completion 修改**:
```python
# 旧: descent["front_complete"]
# 新: v134_descent_follow_state()["completion_event"]
```

现在下楼结束判定基于 **rear-follow completion**，而非前轮下完就算。

### 4. Persistent FLAT Mode (v135_descent_stance.py)

**状态机**:
```
STAIR → RECOVERY → FLAT (persistent)
```

**新增状态**: `flat_mode`
- 在 recovery settlement 确认后设置为 True
- 只要 `raw_no_riser=True` 就持续生效
- Episode reset 时清零

**新增 Crawl Penalty**:
```python
v135_post_stair_flat_crawl_penalty
- 只在 flat_mode & no_riser 期间生效
- clearance_threshold: 0.90
- clearance_scale: 0.12
- weight: 1.0

deficit = clamp((0.90 - clearance_ratio) / 0.12, min=0)
cost = clamp(deficit^2, max=1)
```

**关键**: 这个 penalty **不在楼梯上生效**，避免干扰自然楼梯步态。

### 5. PPO 参数调整 (rsl_rl_ppo_cfg.py)

**DeeproboticsM20V135NativeAlternationPPORunnerCfg** 改名为:
**DeeproboticsM20V135FinalPPORunnerCfg**

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `num_steps_per_env` | 48 | **32** | 提高更新频率 |
| `learning_rate` | 1.0e-3 | **3.0e-4** | 稳定已涌现能力 |
| `entropy_coef` | 0.003 | **0.001** | 减少探索干扰 |
| `num_learning_epochs` | 5 | **3** | 减少过拟合风险 |
| `desired_kl` | 0.01 | **0.008** | 更保守的更新 |

**初始化策略**: 从 V135 3000-3400 轮 checkpoint 继续训练，不从零开始。

### 6. 配置结构简化

**删除文件**:
- `v135_return_redistribution.py` (未完成的 RR 分支)
- `v135_soft_env_cfg.py` (Soft 分支配置)
- `DeeproboticsM20V135RRPPORunnerCfg` (RR runner config)

**保留文件**:
- `v135_riser_identity.py` (重构后的状态机)
- `v135_final_ascent_reward.py` (新的 dominance shaping)
- `v135_descent_stance.py` (增强的下楼约束 + flat mode)
- `v135_soft_progress.py` (保留，供参考，但不直接使用)
- `v135_ppo_gait_diagnostics.py` (诊断工具，独立)

## 验证结果

### Python 语法编译

```
✓ v135_riser_identity.py
✓ v135_final_ascent_reward.py
✓ v135_descent_stance.py
✓ v135_v11fd_native_alternation_env_cfg.py
✓ rsl_rl_ppo_cfg.py
```

### 待验证

- [ ] Isaac Sim 16 env smoke test
- [ ] Recurrent PPO update (1 batch)
- [ ] V13 contract tests

## 最终 Reward Ledger

### Ascent (上楼)

| Term | Weight | Function |
|------|--------|----------|
| Valid completion | +3.0 | v135_valid_riser_completion |
| Leader progress (bare) | +0.5 | v135_ascent_leader_progress |
| Leader progress (vel-coupled) | +1.5 | v135_ascent_leader_velocity_coupled |
| Wrong-side | -0.25 | v135_wrong_side_riser |
| Duplicate | -0.1 | v135_duplicate_riser |
| Skip | -0.25 | v135_skip_riser |
| Dual-skip | -0.25 | v135_dual_skip_riser |
| Overstep | -0.25 | v135_overstep_progress |

### Descent (下楼)

| Term | Weight | Function |
|------|--------|----------|
| Follow progress | +1.0 | v135_descent_follow_progress |
| Follow completion | +2.0 | v135_descent_follow_completion |
| Rear extension | -0.75 | v135_descent_active_extension |
| Rear knee fold | -0.50 | v135_descent_active_knee_fold |
| Rear roll | -0.12 | v135_descent_active_roll |

### Post-Stair (楼梯后)

| Term | Weight | Function |
|------|--------|----------|
| Recovery progress | +0.5 | v135_post_stair_flat_progress |
| Recovery completion | +3.0 | v135_post_stair_flat_completion |
| Flat crawl guard | -1.0 | v135_post_stair_flat_crawl |

### Native V11 Rewards (保持不变)

- track_lin_vel_xy
- track_ang_vel_z
- lin_vel_z_l2
- ang_vel_xy_l2
- orientation_l2
- dof_torques_l2
- dof_acc_l2
- action_rate_l2
- dof_pos_limits
- (其他 V11 原生奖励)

## 设计原则回顾

### 核心思想

1. **错误影响成绩，不毁学习**
   - `episode_ordered_valid` 只是一个成绩标记
   - Target 自动重同步到实际物理位置
   - 后续 episode 仍然能获得有效的 gait-learning signal

2. **正向塑造优于负向惩罚**
   - Dominance/share: 告诉策略"什么值得做"
   - 不是用一堆 penalty 告诉"什么不能做"
   - 主要梯度来自 positive progress，不是 punishment

3. **约束只在需要时生效**
   - Rear knee fold: 只在 descent follow active
   - Flat crawl: 只在确认离开楼梯后
   - 避免全局约束干扰楼梯自然步态

4. **持续约束不能有时间限制**
   - Flat mode: 一旦进入，只要 no_riser 就持续
   - 不是"站直5帧后就没人管"
   - 防止策略钻空子（站直→拿奖励→重新趴下）

## 下一步

### 实验执行

```bash
# 1. 从 V135 3000轮 checkpoint 启动
python scripts/train_v135_final.py \
    --checkpoint logs/m20_stair_v135/model_3000.pt

# 2. 监控关键指标 (TensorBoard)
- ordered_alternation_rate (应保持 >0.5)
- episode_valid_rate (严格成功率)
- target_id_mean (平均完成台阶数)
- v135_ascent_leader_progress (新的 shaping signal)
- v135_post_stair_flat_crawl (flat mode 约束)

# 3. 对比实验
- V135-baseline: 继续原始 V135 到 5000 轮（预期退化）
- V135-Final: 本次修改版本（预期保持）
```

### 成功标准

**Minimal Success**:
- ordered_alternation_rate 保持 ≥0.5 到 5000 轮
- 不低于 3000 轮水平

**Full Success**:
- ordered_alternation_rate 上升到 ≥0.7
- 下楼无拖腿/折叠
- 楼梯后自然恢复平地 locomotion，无匍匐
- 行为稳定（继续到 8000 轮不退化）

## 已知限制

1. **未验证实际运行**
   - 编译通过 ≠ Isaac Sim 能跑
   - 需要 16 env smoke test

2. **Metric 名称可能需要调整**
   - `episode_valid_rate` 可能与某些 logger 冲突
   - 需要确认 TensorBoard 显示正常

3. **新 reward 函数可能需要微调**
   - Dominance/share 的权重 (0.5 vs 1.5) 基于理论推导
   - 可能需要根据实际训练调整

4. **Checkpoint 兼容性未测试**
   - 从 V135 3000 轮加载到 V135-Final 可能有状态不匹配
   - 可能需要手动处理 state_dict

## 文件清单

### 修改的文件

1. `v135_riser_identity.py` - 状态机重构 + target 重同步
2. `v135_descent_stance.py` - 下楼约束 + flat mode + crawl penalty
3. `v135_v11fd_native_alternation_env_cfg.py` - Reward ledger + config
4. `rsl_rl_ppo_cfg.py` - PPO 参数调整
5. `mdp/__init__.py` - 删除 RR import

### 新增的文件

1. `v135_final_ascent_reward.py` - Dominance-based shaping

### 删除的文件

1. `v135_return_redistribution.py` - RR 分支
2. `v135_soft_env_cfg.py` - Soft 分支配置
3. `rsl_rl_ppo_cfg.py::DeeproboticsM20V135RRPPORunnerCfg` - RR runner

### 备份文件

1. `v135_riser_identity.py.backup` - 原始状态机备份

---

**总结**: V13.5-Final 将所有改动整合为单一稳定版本，核心是"错误不停学习 + 正向塑造 + 精准约束"。所有 Python 语法编译通过，待 Isaac Sim 实际验证。
