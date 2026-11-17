# V13.5-Final 剩余修复清单

**日期**: 2026-09-19  
**状态**: 部分修复完成，剩余关键问题需继续

---

## ✅ 本轮已完成

1. ✅ **Final Perception 修复** - 使用现有的 v135_target_actor_perception
2. ✅ **Velocity Modulation 修复** - 保持 V13 权重 (5.0/2.0/3.0)
3. ✅ **Command Split 接入** - 下楼 0.15-0.80, 上楼 0.10-2.0

---

## ⚠️ 剩余必须修复（阻止训练）

### 1. Runner 接入 V135ResidualGRUModel ⚠️

**问题**: 
- V11ContinuousRunner 只认 `("V11ContinuousActorCritic", "V135RecurrentActorCritic")`
- 新 Actor 会被拒绝

**修复位置**: `source/rl_training/rl_training/v11_continuous_runner.py`

**修复方法**:
```python
# Line ~50
if policy_class not in ("V11ContinuousActorCritic", "V135RecurrentActorCritic", "V135ResidualGRUActor"):
    raise RuntimeError(...)

residual_gru_actor = policy_class == "V135ResidualGRUActor"

# Line ~70
if residual_gru_actor:
    cfg["actor"]["class_name"] = "rl_training.v135_final_actor:V135ResidualGRUModel"
    # Pass correct params: gru_hidden_dim=64, gru_input_dim=32, residual_scale=0.20
```

**PPO Config 修改**:
```python
# rsl_rl_ppo_cfg.py
self.policy.class_name = "V135ResidualGRUActor"  # Not the full path
rnn_hidden_dim=64  # Not 128
```

### 2. Descent Support Transfer 6个子问题 ⚠️

**文件**: `v135_descent_support_transfer.py`

#### 2a. forward_world_xy 不存在
```python
# 当前: forward_world_xy = base["forward_world_xy"]  # KeyError
# 修复:
heading = env.command_manager._terms["v11_path_heading_w"].heading  # or similar
forward_world_xy = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
```

#### 2b. descent["reference_id"] 不存在
```python
# 当前: current_ref_id = descent.get("reference_id", zeros)  # 永远是0
# 修复: 用 reference position 判断
ref_xy = descent["reference_world_xy"]
ref_changed = (ref_xy - s["last_reference_world_xy"]).norm(dim=-1) > 0.10
s["last_reference_world_xy"] = torch.where(ref_changed[:, None], ref_xy, s["last_reference_world_xy"])
```

#### 2c. 保存 rear_init_world_xy 而非 rear_init_x
```python
# State 初始化:
"rear_init_world_xy": torch.zeros((n, 2, 2), device=d),  # Not rear_init_x

# 保存时:
rear_world_xy = rear_pos[:, :, :2]
s["rear_init_world_xy"] = torch.where(front_event[:, None, None], rear_world_xy, s["rear_init_world_xy"])
```

#### 2d. 真实 step drop (不是 0.30)
```python
# 当前: step_height = 0.30
# 修复:
expected_drop = (
    s["rear_init_z"] - wheel_radius - s["z_lower"][:, None]
).clamp(0.04, 0.30)
p_z_raw = (s["rear_init_z"] - rear_z) / (expected_drop + 1e-6)
```

#### 2e. Contact IDs 安全
```python
# 初始化时:
asset = env.scene["robot"]
sensor = env.scene.sensors["contact_forces"]
rear_body_ids = asset.find_bodies(["hl_wheel", "hr_wheel"])
rear_contact_ids = sensor.find_bodies(["hl_wheel", "hr_wheel"])
env._v135_rear_body_ids = rear_body_ids
env._v135_rear_contact_ids = rear_contact_ids

# 使用时:
forces_z = sensor.data.net_forces_w[:, env._v135_rear_contact_ids, 2]
```

#### 2f. P_rear ∈ [0,1]
```python
# 当前: 0.5 * (phi_hl + phi_hr) + 0.5 * min(...)  # 可能 >1
# 修复:
mean_phi = 0.5 * (phi_hl + phi_hr)
P_rear = 0.5 * mean_phi + 0.5 * torch.minimum(phi_hl, phi_hr)
# 确保 P_rear ∈ [0,1]
```

### 3. Body-Rear Lag 定义修复 ⚠️

**问题**: 当前用 absolute position, 应该用 delta

**修复**:
```python
# State 初始化:
"body_init_forward": torch.zeros(n, device=d),
"rear_init_forward": torch.zeros((n, 2), device=d),

# Front complete 时保存:
body_forward_0 = ((body_world_xy - s["edge_world_xy"]) * s["forward_world_xy"]).sum(-1)
s["body_init_forward"] = torch.where(front_event, body_forward_0, s["body_init_forward"])

rear_forward_0 = ((rear_world_xy - s["edge_world_xy"][:, None, :]) * s["forward_world_xy"][:, None, :]).sum(-1)
s["rear_init_forward"] = torch.where(front_event[:, None], rear_forward_0, s["rear_init_forward"])

# 计算 lag:
delta_body = body_forward_current - s["body_init_forward"]
delta_rear = rear_forward_current - s["rear_init_forward"]
min_delta_rear = delta_rear.min(dim=1)[0]
lag = delta_body - min_delta_rear
```

### 4. Post-Stair 连接新 Rear Transfer ⚠️

**文件**: `v135_descent_stance.py`

**修复**:
```python
# v135_post_stair_flat_state():
from .v135_descent_support_transfer import v135_descent_rear_support_transfer_state

rear = v135_descent_rear_support_transfer_state(env, sensor_cfg)

traversal_event = (
    landing["platform_exit_event"]
    | rear["completion_event"]  # 需要在 support_transfer 中添加
)
```

**v135_descent_support_transfer.py 添加**:
```python
# 在 state 中添加:
s["completion_event"] = completion_now & ~s["completion_prev"]
s["completion_prev"] = completion_now
```

### 5. Flat Mode 稳定性 ⚠️

**文件**: `v135_descent_stance.py`

**修复**:
```python
# 当前: flat_mode &= raw_no_riser  # 一帧误检就清除
# 修复:
flat_mode |= event  # Set on recovery complete

# 只在确认进入真实 stair 后清除:
ascent_active = ... # from v135_riser_identity
descent_active = ... # from v11 descent
stair_confirmed = (ascent_active | descent_active)  # 连续3帧

flat_mode &= ~stair_confirmed

# Crawl penalty 仍然:
active = flat_mode & raw_no_riser
```

---

## 📋 完整修复顺序（建议）

1. ✅ Final perception (已完成)
2. ✅ Velocity modulation weights (已完成)
3. ✅ Command split (已完成)
4. ⚠️ **Runner 接入 (必须)** 
5. ⚠️ **Descent 6个子问题 (必须)**
6. ⚠️ **Body-rear lag (必须)**
7. ⚠️ **Post-stair 连接 (必须)**
8. ⚠️ **Flat mode 稳定 (重要)**
9. ⚠️ Contract tests 更新 (重要)

---

## 预计修复时间

- Runner 接入: 30分钟
- Descent 6问题: 1小时
- Body-rear lag: 20分钟
- Post-stair: 20分钟
- Flat mode: 10分钟
- Tests: 20分钟

**总计**: ~2.5小时

---

## 下一步

1. 继续修复剩余 6项
2. Python 编译测试
3. 16 env smoke test
4. 如果通过 → 4096 env from-scratch 长训

**不要**: 继续改方案或添加新功能

---

**最后更新**: 2026-09-19
