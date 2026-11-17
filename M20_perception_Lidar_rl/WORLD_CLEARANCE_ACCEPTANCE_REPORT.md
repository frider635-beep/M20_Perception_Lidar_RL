# M20-Stair-Perceptive-v0 World Clearance 验收报告

日期：2026-08-12  
执行机：`ubuntu@192.168.1.244`  
项目目录：`/home/ubuntu/m20_perceptive_final`  
Task：`M20-Stair-Perceptive-v0`

## 1. 本轮目标

本轮只解决一个问题：确保 M20 wheel 在接触楼梯 vertical riser 之前，已经获得足够的真实几何 clearance。

本轮未启动任何 PPO、64/128/512/4096-env learning 或 20K Full Training。

## 2. 冻结接口与官方参数

接口保持：

```text
Actor  = 372D
Critic = 247D
Action = 16D
```

以下 DeepRobotics 官方配置保持冻结：

- Action definition 和 16D joint order
- Action scale
- Leg/Wheel PD
- Actuator configuration
- Effort/velocity limits
- 22 个 official reward term 及其全部参数
- `height_scanner_base`
- `terrain_height_threshold=0.06`
- `high_terrain_penalty_scale=0.1`

回归结果：

```text
OFFICIAL_LOCKED_REWARDS=22 FULL_PARAMS=PASS
OFFICIAL_LOCKED_PARAMS=PASS
POLICY_DIM=372 CRITIC_DIM=247 ACTION_DIM=16
NO_NAN=True NO_INF=True
NO_EXPLICIT_GAIT_STATE=PASS
```

## 3. 官方 Wheel Collision Geometry

读取来源：

```text
/home/ubuntu/m20_perceptive_final/deep_robotics_model/M20/usd/M20.usd
/home/ubuntu/m20_perceptive_final/deep_robotics_model/M20/urdf/M20.urdf
```

URDF 中 wheel collision 为 cylinder：

```text
radius = 0.09 m
length = 0.054 m
```

实际 Task 加载的 USD collision prototype 为 Capsule：

```text
radius ≈ 0.090000004 m
height ≈ 0.054000001 m
scale = 1
```

四轮一致：

```text
OFFICIAL_WHEEL_RADIUS_FL=0.090000004
OFFICIAL_WHEEL_RADIUS_FR=0.090000004
OFFICIAL_WHEEL_RADIUS_HL=0.090000004
OFFICIAL_WHEEL_RADIUS_HR=0.090000004
OFFICIAL_WHEEL_GEOMETRY=PASS
```

Visual STL 没有用于碰撞半径推断。代码采用官方冻结半径 `R=0.09 m`。

## 4. 删除旧的 Leg-Relative 假设

正式废弃：

```text
F(h)=h+0.02
```

作为 leg-relative lift target 的语义。

楼梯高度不再被直接转换成单腿收缩量。Wheel clearance 可以由以下运动共同产生：

- 腿收缩
- 其他腿伸展
- Base rise
- 整体 body motion

`L_leg_relative` 仅保留为诊断量，不再作为主要几何 reward target。

## 5. Wheel-Riser 几何模型

参数定义：

```text
d = x_riser - x_wheel
R = 0.09 m
clearance_margin = 0.01 m
R_clear = 0.10 m
```

TASK_CALIBRATED margin 测试范围：

```text
0, 0.005, 0.010, 0.015, 0.020 m
```

本轮采用 `0.010 m`。该值是 collision clearance margin，不是腿额外抬升量。

几何真值同时检查：

1. Wheel center 到 finite vertical riser segment 的最小欧氏距离不小于 `R_clear`。
2. Wheel center 越过 riser 后不穿入 upper tread。

World-Z reference：

```text
z_nominal_world = z_lower + R
z_cross         = z_upper + R_clear
z_preview(d)    = z_nominal_world + g(d) × (z_cross-z_nominal_world)
z_ref_world(d)  = max(z_preview, z_geom_required)
```

`z_ref_world` 表示 wheel center 应达到的 world height，不是 knee、joint trajectory、IK 或 leg-relative target。

## 6. 17 cm / 20 cm Geometry Sweep

测试距离：

```text
+0.40, +0.35, +0.30, +0.25, +0.20, +0.15,
+0.10, +0.07, +0.05, 0, -0.03, -0.05, -0.10, -0.15 m
```

关键结果，以下高度以 lower tread world-Z 为 0：

| Stair | d | z_geom_required | z_ref_world |
|---:|---:|---:|---:|
| 17 cm | +0.40 | 0.000000 | 0.090000 |
| 17 cm | +0.10 | 0.000000 | 0.256667 |
| 17 cm | +0.05 | 0.256603 | 0.270000 |
| 17 cm | 0.00 | 0.270000 | 0.270000 |
| 20 cm | +0.40 | 0.000000 | 0.090000 |
| 20 cm | +0.10 | 0.000000 | 0.284444 |
| 20 cm | +0.05 | 0.286603 | 0.300000 |
| 20 cm | 0.00 | 0.300000 | 0.300000 |

所有 `z_actual=z_ref_world` 样本均通过 circle-vs-segment 和 upper-tread collision validation：

```text
17CM_GEOMETRIC_CLEARANCE=PASS
20CM_GEOMETRIC_CLEARANCE=PASS
```

## 7. Spatial Preview Window

对比测试：

```text
d_start = 0.25 / 0.35 / 0.45 m
```

采用：

```text
d_start = +0.35 m
d_peak  = +0.07 m
d_hold  = -0.03 m
d_end   = -0.12 m
```

选择 `0.35 m` 的原因是：相比 `0.25 m` 更早建立 world-Z reference，同时没有采用 `0.45 m` 的更长远场引导窗口。

这些参数属于 TASK_CALIBRATED，不是官方参数。

## 8. Normalized Clearance Progress Reward

旧 exponential deficit reward 已删除。

当前对每个 active wheel 定义：

```text
p = clip(
    (z_actual_world-z_start) /
    (z_target-z_start),
    0,
    1
)

r_spatial = 2p-p²
```

性质：

```text
p=0.0 → reward=0.00
p=0.5 → reward=0.75
p=1.0 → reward=1.00
```

超过 target 后 reward 不再增加。只对 active wheels 求平均；没有 active wheel 时 Spatial Reward 为 0。

因此 3 cm 等小台阶也具有完整 raw `0→1` 动态范围：

```text
SPATIAL_REWARD_NORMALIZED_GRADIENT=PASS
```

## 9. FIRST_RISER_PRELIFT FK Reward Competition

测试保持：

```text
height_scanner_base sees flat terrain
hipy scale = 1.0
knee scale = 1.0
```

没有提前强制触发官方 relaxation。

实验约束：

- 每个 FK alpha 恢复相同 root pose 和零速度。
- 根据实际 FK wheel X 重新平移 root，使 wheel 保持在相同潜在 riser collision band。
- 禁止通过把 wheel 后移远离 riser 获得虚假的 clearance PASS。
- 只接受 `progress=1` 且实际 circle/segment geometry 无碰撞的姿态。

结果：

| Stair | ΔSpatial，w=2 | ΔOfficial Cost | ΔNet |
|---:|---:|---:|---:|
| 17 cm | +1.733 | −0.273 | +1.460 |
| 20 cm | +1.770 | −0.429 | +1.341 |

Spatial weight sweep：

```text
2, 3, 4, 5, 6, 8
```

两种高度的最小通过值均为当前值 `2.0`：

```text
PRELIFT_REWARD_COMPETITION_17=PASS MIN_WEIGHT=2.0
PRELIFT_REWARD_COMPETITION_20=PASS MIN_WEIGHT=2.0
```

没有提高 Spatial weight，也没有修改任何官方 reward weight。

静态 action-rate、action-smooth 和 torque 没有用于伪造 PASS；orientation cost 已纳入 official cost。

## 10. 官方 High-Terrain Relaxation

官方机制复核保持：

| Terrain | hipy scale | knee scale |
|---|---:|---:|
| Flat | 1.0 | 1.0 |
| 8 cm | 0.1 | 0.1 |
| 17 cm | 0.1 | 0.1 |
| 20 cm | 0.1 | 0.1 |

第一 riser pre-lift 时，官方 base-centered scanner 仍看到平地，因此 scale 仍为 1.0。这一官方行为未修改。

新的 normalized reward 在该 full-strength official posture penalty 条件下仍得到正 `ΔNet`。

## 11. 新增和修改内容

核心实现：

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/perceptive_spatial.py
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/observations.py
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/perceptive_env_cfg.py
```

验收工具：

```text
scripts/tools/inspect_m20_wheel_collision_geometry.py
scripts/tools/test_m20_wheel_riser_geometry.py
scripts/tools/sweep_m20_spatial_reward.py
scripts/tools/sweep_m20_fk_reward_competition.py
```

验收产物：

```text
artifacts/wheel_riser_geometry/
artifacts/spatial_reward_sweep/normalized_progress_sweep.csv
artifacts/spatial_reward_sweep/normalized_progress_sweep.png
artifacts/fk_reward_competition/fk_reward_competition.csv
```

## 12. 最终判定

```text
OFFICIAL_WHEEL_GEOMETRY=PASS
17CM_GEOMETRIC_CLEARANCE=PASS
20CM_GEOMETRIC_CLEARANCE=PASS
SPATIAL_REWARD_NORMALIZED_GRADIENT=PASS
PRELIFT_REWARD_COMPETITION_17=PASS
PRELIFT_REWARD_COMPETITION_20=PASS
NO_PHASE=True
NO_CONTACT_STATE=True
OFFICIAL_PARAMS_UNCHANGED=True
STATIC_GEOMETRY_AND_REWARD_ACCEPTANCE=PASS
```

本结论仅表示静态几何与 reward optimization logic 通过，不表示 PPO 已经学会爬楼。

## 13. 当前禁止项

仍然禁止自动启动：

- 64/128-env learning
- 512-env training
- 4096-env training
- 20K Full Training

等待用户下一步指令。
