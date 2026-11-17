# V13.5-Final Bug Fix Progress Report

**日期**: 2026-09-19  
**状态**: 正在按优先级修复确定性 bug

---

## ✅ 已修复 (Priority 1-2)

### 1. 网络 Actor 接线 ✓
**文件**: `v135_final_actor.py` (新建)

**问题**: 
- 旧的 `v135_from_scratch_actor.py` 使用过时的 ActorCritic API
- V11ContinuousRunner 不识别该 class

**修复**:
- 创建 `V135ResidualGRUModel` (RSL-RL 5.x API)
- MLP backbone + GRU64 front-leg residual
- Ascent gate 从 obs[57:59] 提取: `g_up = (B1_valid>0.5) & (H1>0.03)`
- Residual 只影响 action[0:6]

### 2. Descent Perception 失明 ✓
**文件**: `v135_final_perception.py` (新建)

**问题**: 
- V135 用 strict target 替换了所有 6D perception
- 下楼时 target_valid=False → 6D 全是 0

**修复**:
- Ascent: 使用 strict P1/P2
- Descent: 使用 V11 native signed edge (H<0)
- Flat: 自然归零

### 3. Skip Resync Bug ✓
**文件**: `v135_riser_identity.py`

**问题 1**: `step_width` 在 skip 分支内定义，外部使用会 UnboundLocalError

**修复**: 在所有分支前统一定义 `step_width`

**问题 2**: Skip 少推进一级
- 旧: 完成 B2 → target=B2 (错误，已完成)
- 新: 完成 B2 → target=B3 (正确，下一目标)

**修复**: `skip_advance_count = risers_completed + 1`

### 4. Ascent Reward step_dt ✓
**文件**: `v135_final_ascent_reward.py`

**问题**: 使用 `env.cfg.sim.dt=0.005` 而非 `env.step_dt=0.02`

**修复**: 改为 `float(env.step_dt)`

### 5. Best Potential Reset ✓
**文件**: `v135_final_ascent_reward.py`

**问题**: 只在 valid 时 reset，target 改变后仍用旧 best

**修复**: 在所有 target transition 时 reset
```python
target_transition = valid | wrong_side | skip | dual_skip
```

---

## 🔄 进行中 (Priority 2-3)

### 6. 旧 V13 Reward 仍然生效
**文件**: `v135_v11fd_native_alternation_env_cfg.py`

**问题**: 
- M20V135RewardsCfg 继承 M20V13RewardsCfg
- 旧的 same-side, simultaneous, V13 progress 全部还在

**待修复**: 在 __post_init__ 中显式禁用
```python
self.rewards.v13_opposite_progress = None
self.rewards.v13_same_side_progress = None
self.rewards.v13_simultaneous_progress = None
# ... 等等
```

### 7. Legacy V13 读取不存在的 ordered_chain_failed
**文件**: `v13_alternation.py`

**问题**: 
- V135 已改为 episode_ordered_valid
- V13 critic 仍读取 ordered_chain_failed → KeyError

**待修复**: 创建 `v135_final_critic_reward_state()`

### 8. Descent Support Transfer 使用 placeholder
**文件**: `v135_descent_support_transfer.py`

**问题**: 
- edge_x = root_pos_w[:,0] (假的)
- z_lower = root_pos_w[:,2] - 0.30 (假的，真实高度 0.10-0.24m)

**待修复**: 从 V11 descent state 读取真实 geometry
```python
edge_world_xy = descent["reference_world_xy"]
z_lower = descent["target_lower_z"]
```

### 9. Descent Wheel Height 计算错误
**待修复**: `rear_bottom = rear_z - wheel_radius`

### 10. Descent Reward Delta Bug
**问题**: `P_rear_best.roll(1, dims=0)` roll 的是 env 维度

**待修复**: 
```python
old_best = s["P_rear_best"].clone()
new_best = max(old_best, P_rear)
delta = new_best - old_best
```

### 11. Descent Contact ID 不安全
**待修复**: 分别使用 `asset.find_bodies()` 和 `sensor.find_bodies()`

### 12. Descent 不支持多级楼梯
**待修复**: 每次 descent reference 改变时重新初始化

### 13. Velocity Modulation 未接入
**待修复**: 
```python
r_vel_final = g_d * r_vel
r_forward_final = g_d * r_forward
```

### 14. Command Split 未接入
**待修复**: 添加 `M20V135PerceptionCfg`

### 15. Post-Stair 仍读取 V13.4
**待修复**: 改为读取 `v135_descent_rear_support_transfer`

### 16. Persistent Flat 可能丢失
**待修复**: 只在确认进入真实 stair 后才清 flat_mode

### 17. Anti-Crawl 未导出
**待修复**: 添加到 `__all__`

### 18. Contract Tests 仍是旧版本
**待修复**: 更新测试期望

### 19. PPO Config class_name 错误
**待修复**: 改为 `rl_training.v135_final_actor:V135ResidualGRUModel`

---

## 修复策略

按照你的建议，优先级顺序：

**第一优先** (已完成 ✓):
- ✅ 网络 runner (v135_final_actor.py)
- ✅ Descent perception (v135_final_perception.py)

**第二优先** (部分完成):
- ✅ Ascent state/resync bug
- ✅ Ascent reward step_dt
- 🔄 旧 reward 双重接线

**第三优先**:
- 🔄 Descent support transfer 真实 geometry
- 🔄 Descent reward delta bug
- 🔄 Descent 多级支持

**第四优先**:
- 🔄 Velocity modulation
- 🔄 Command split
- 🔄 Persistent flat
- 🔄 Contract tests

---

## 下一步行动

1. 继续修复第二优先级：禁用旧 V13 rewards
2. 创建 v135_final_critic_reward_state
3. 修复 descent support transfer geometry
4. 修复所有 descent bugs
5. 接入 velocity modulation 和 command split
6. 更新 contract tests
7. Smoke test: 16 env × 8 step, 1 PPO update

---

**预计完成时间**: 继续工作 1-2 小时可完成所有修复

**最终目标**: 所有确定性 bug 修复后，直接 from-scratch 长训，不再改方案
