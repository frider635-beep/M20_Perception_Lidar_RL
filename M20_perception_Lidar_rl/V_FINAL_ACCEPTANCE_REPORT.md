# M20-Stair-Perceptive-v0 静态几何与 Reward 验收

日期：2026-08-12  
执行机：`ubuntu@192.168.1.244`  
目录：`/home/ubuntu/m20_perceptive_final`

## 结论

本轮完成了 wheel world-Z 几何 clearance 与 normalized progress reward。没有启动任何 learning。

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
FULL_TRAINING_STARTED=False
```

注意：这只是静态 geometry/FK/reward 验收，不代表 PPO 已学会爬楼。官方 `height_scanner_base` 在第一 riser pre-lift 时仍保持 posture scale=1.0；该官方语义未被修改。

## 冻结接口

- Actor：372D = 官方 57D proprio + 315D Elevation
- Critic：247D = `3+3+3+3+16+16+16+187`
- Action：16D，官方 joint order 不变
- 官方 Action、scale、PD、actuator、limits、22 reward、domain randomization 全部不变
- `terrain_height_threshold=0.06`
- `high_terrain_penalty_scale=0.1`

回归：

```text
OFFICIAL_LOCKED_REWARDS=22 FULL_PARAMS=PASS
POLICY_DIM=372 CRITIC_DIM=247 ACTION_DIM=16
SMOKE_ENVS=2 STEPS=5 NO_NAN=True NO_INF=True
NO_EXPLICIT_GAIT_STATE=PASS
```

## 官方 Wheel Collision Geometry

资产：

```text
/home/ubuntu/m20_perceptive_final/deep_robotics_model/M20/usd/M20.usd
/home/ubuntu/m20_perceptive_final/deep_robotics_model/M20/urdf/M20.urdf
```

URDF collision 是 cylinder：`radius=0.09 m, length=0.054 m`。实际 Task USD prototype 是 Capsule：`radius≈0.090000004 m, height≈0.054000001 m`。Visual STL 没有用于估算 collision radius。

```text
OFFICIAL_WHEEL_RADIUS_FL=0.090000004
OFFICIAL_WHEEL_RADIUS_FR=0.090000004
OFFICIAL_WHEEL_RADIUS_HL=0.090000004
OFFICIAL_WHEEL_RADIUS_HR=0.090000004
OFFICIAL_WHEEL_GEOMETRY=PASS
```

代码使用冻结值 `R=0.09 m`。TASK_CALIBRATED clearance margin 经 `0/5/10/15/20 mm` sweep 后采用 `m=0.01 m`，因此 `R_clear=0.10 m`。

## 几何模型

旧假设 `F(h)=h+0.02 → leg-relative lift target` 已废弃。

当前定义：

```text
z_nominal_world = z_lower + R
z_cross         = z_upper + R_clear
z_preview(d)    = z_nominal_world + g(d)*(z_cross-z_nominal_world)
z_ref_world(d)  = max(z_preview, z_geom_required)
```

`z_geom_required` 同时通过以下真值检查：

- wheel center 到有限 vertical riser segment 的最小欧氏距离 `>=R_clear`
- wheel center 越过 riser 后满足 upper-tread clearance

`z_ref_world` 是 wheel-center world height，不是 knee、IK 或 leg-relative target。Base rise 合法计入 clearance。`L_leg_relative` 只作诊断。

17/20 cm 关键结果（lower tread world-Z 归零）：

| stair | d | z_geom_required | z_ref_world |
|---:|---:|---:|---:|
| 17 cm | +0.10 | 0.100000 | 0.256667 |
| 17 cm | +0.05 | 0.256603 | 0.270000 |
| 17 cm | 0.00 | 0.270000 | 0.270000 |
| 20 cm | +0.10 | 0.100000 | 0.284444 |
| 20 cm | +0.05 | 0.286603 | 0.300000 |
| 20 cm | 0.00 | 0.300000 | 0.300000 |

完整距离 sweep 包含 `+0.40 ... -0.15 m`，所有 `z_actual=z_ref_world` 样本均无 vertical-riser 或 upper-tread 穿入。

## Preview window

静态比较 `d_start=0.25/0.35/0.45 m`。为避免最大 forward command 下在最后几厘米才快速建立 target，采用：

```text
d_start=+0.35 m
d_peak =+0.07 m
d_hold =-0.03 m
d_end  =-0.12 m
```

这些仍是 TASK_CALIBRATED 参数，不属于官方参数。

## Normalized Spatial Reward

旧 exponential deficit 已删除。新 reward：

```text
p = clip((z_actual_world-z_start)/(z_target-z_start), 0, 1)
r = 2p-p²
```

- `p=0 → 0`
- `p=0.5 → 0.75`
- `p=1 → 1`
- 超过 target 不再增加
- 只对 active wheels 平均；无 active wheel 时为 0

因此 3 cm 等小台阶也有完整 raw `0→1` 学习动态范围，不再只有旧公式约 0.05 的增量。

## FIRST_RISER_PRELIFT FK Competition

每个 FK alpha 样本都恢复相同的 precontact root pose和零速度，并重新平移 root，使前轮保持在同一潜在碰撞带，而不是通过关节折叠把轮子后移远离 riser。只接受实际 joint FK 达到 `progress=1` 且 circle/segment geometry 为真的样本。

| stair | ΔSpatial，w=2 | ΔOfficial Cost | ΔNet |
|---:|---:|---:|---:|
| 17 cm | +1.733 | −0.273 | +1.460 |
| 20 cm | +1.770 | −0.429 | +1.341 |

`w_spatial=2/3/4/5/6/8` sweep 中，两者最小正值均是当前 `2.0`：

```text
PRELIFT_REWARD_COMPETITION_17=PASS MIN_WEIGHT=2.0
PRELIFT_REWARD_COMPETITION_20=PASS MIN_WEIGHT=2.0
```

没有提升 spatial weight，也没有修改任何官方 penalty。Spatial weighted maximum=2，与官方 locomotion reward 同量级，不是数量级压制。

静态 pose 下 action-rate、action-smooth 和 torque 不具备可靠动态含义，未用于伪造 PASS；orientation 已计入 official cost。

## 官方高地形放松机制

复核保持：

| terrain | hipy scale | knee scale |
|---|---:|---:|
| flat | 1.0 | 1.0 |
| 8 cm | 0.1 | 0.1 |
| 17 cm | 0.1 | 0.1 |
| 20 cm | 0.1 | 0.1 |

第一 riser pre-lift 时官方 base-centered scanner 仍看到平地，因此 scale=1.0。新 normalized reward 的 FK competition 在该真实 full-strength 条件下仍得到正 ΔNet；官方机制没有被删除、替代或提前强制放松。

## 产物

```text
scripts/tools/inspect_m20_wheel_collision_geometry.py
scripts/tools/test_m20_wheel_riser_geometry.py
scripts/tools/sweep_m20_spatial_reward.py
scripts/tools/sweep_m20_fk_reward_competition.py
artifacts/wheel_riser_geometry/
artifacts/spatial_reward_sweep/
artifacts/fk_reward_competition/
```

## 当前禁止项

- 64/128-env learning
- 512-env training
- 4096-env training
- 20K Full Training

本报告完成后停止，不自动开始 PPO。
