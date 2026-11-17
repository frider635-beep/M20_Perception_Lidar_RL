# V11 仓库实现与训练报告

更新日期：2026-08-29  
项目：`kunyu_ppo`  
任务 ID：`M20-Stair-Continuous-Landing-V11-v0`

## 1. 结论摘要

V11 已作为独立 Gym/Isaac Lab 任务接入仓库。2026-08-29 起，V11 默认契约为 `alternating_risers`：FL/FR 自主决定首阶，之后只由反侧跨下一道未结算立面，再逐阶翻转。净空按当前目标立面的真实上踏面高度计算，不提高固定抬腿高度。

与 V10 相比，V11 不再保留 `H/D -> FR 三关节` 的网络直连。实际训练使用共享 MLP：Actor 为 63D 输入、16D 输出；FL/FR 没有结构性动作通路差异。

已完成静态检查、纯几何门控和 Isaac Lab PPO 冒烟。2026-08-27 的当前版本针对“前肢大臂与躯干长时间平行、主要依靠 knee/小臂爬楼”的奖励捷径做了定向修改：越阶时主要释放 hip-y，knee 仅保留必要活动空间；hip-y 参与分不再逐步稠密发放，而是保存最新姿态分并在成功落台后一次性结算。

交替实现保留 RewardManager 40 项和 Actor `63D -> 16D` ABI，并将 V12 显式固定为 `bilateral_same_riser`。首个物理跨阶只确定交替相位，不发正奖，避免 reset farming；后续只有 `expected_side` 能结算净空、越面、支撑和 hip-y 信用。

GPU Manager 冒烟已通过：V11 为 2 env/1 iteration，V12 为 2 env/1 iteration。从 `model_25300.pt` 运行的 256 env/10 iteration 可达性短训中，目标、净空、越面、支撑和首个物理完成均有非零信号，但后续交替奖励事件仍为零，且前轮碰撞终止仍高。该结果只证明状态机接线可达，不是候选模型验收；`25300/29700` clean-ray/标准噪声闭环对拍、正式 A/B 续训、sim2sim 和真机验收仍未执行。

> 本文后续保留了 2026-08-27 的双前轮同立面实现记录，作为 V12 `bilateral_same_riser` 的历史语义参考；若与本节冲突，以本节及当前配置为准。

## 2. V10 到 V11 的变化

| 项目 | V10 | V11 |
|---|---|---|
| Actor 感知 | `57D proprio + B/H/D = 60D` | `57D proprio + B1/H1/D1 + B2/H2/D2 = 63D` |
| 高程图目标 | 首道有效立面，之后 consume | 每 control step 提取当前与下一道正立面 |
| 跨阶腿 | FR 几何、奖励与网络路径均被显式偏置 | FL/FR 对称候选；最先完成条件者锁定为责任轮 |
| 落点 | 无轮心 XY 区间 | 两立面间的 X 安全区间，加可感知 corridor 的 Y 区间 |
| 落点正奖励 | 无 | 安全支撑事件、中心弱 shaping、边缘越界惩罚 |
| 无效几何 | 不适用 | 安全矩形为空时 fail closed，不给落点正奖励 |
| 旧 checkpoint | 60D | 与 63D ABI 不兼容，不允许续训加载 |

## 3. 代码位置与职责

| 位置 | 职责 |
|---|---|
| `V11_repo.md` | V11 设计、当前奖励参数、训练状态与验收说明 |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_continuous_env_cfg.py` | V11 感知参数、reward、Actor/Critic observation 覆盖 |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/perceptive_spatial.py` | `extract_two_risers_points`：从 21×15 yaw-local 高程射线点提取两道立面 |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_landing.py` | 可单元测试的安全矩形、中心/越界评分、责任轮锁定纯几何函数 |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py` | 连续感知状态、确认/hold、责任轮 lifecycle 和 V11 reward 函数 |
| `source/rl_training/rl_training/v11_continuous_runner.py` | RSL-RL 5.x 配置桥接、63D/16D 硬 gate、Actor observation manifest |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py` | V11 PPO runner 参数：从零训练、2000 iterations、每 250 iterations 保存 |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py` | Gym 任务注册 |
| `source/rl_training/rl_training/actuator/PDActuator.py` | Isaac Lab 0.54 的执行器配置公开 API 兼容修复 |

## 4. 连续高程图信息链路

### 4.1 两立面提取

`extract_two_risers_points` 直接处理实际 ray-hit 点，输出 `[N, 2]` 的：

```text
valid, riser_height, riser_distance, riser_x, riser_y, riser_lower_world
```

输出按 yaw-local `+x` 的前进方向排序。第一道立面允许在刚越过后保留到 `passed_distance`，第二道必须比第一道至少远 `min_riser_separation`。缺失立面通过 `valid=False` 表示，距离不会作为 Actor 的有效数值使用。

### 4.2 Actor 63D ABI

```text
0:3    base_ang_vel
3:6    projected_gravity
6:9    velocity command
9:25   joint position
25:41  joint velocity
41:57  previous action
57:60  B1, H1, D1
60:63  B2, H2, D2
```

两组 `B/H/D` 需要连续确认后才进入 Actor；短时缺失可在有限 hysteresis 窗口内 hold。随后还施加配置的高度/距离噪声、帧保持概率和 2 step 延迟。零或非向前 command 时，新的 Actor 跨阶 bit 不会激活。

Critic 维度为 568D：官方 critic 输入、315D 清洁 PGTT 地图与 6D 两立面真值特征。Actor 不接收 315D PGTT 或 reward-side 几何状态。

## 5. 显式安全落点与责任轮

### 5.1 安全矩形

当前实现使用第一道立面给出轮心必须越过的单侧 X 下界：

```text
x_min = x_riser1 + wheel_radius + front_margin
```

第二道立面不再被当作 X 后边界，避免把策略拉向人工的“第二立面前固定落点”。因此该区域在 X 方向是单侧的，Y 方向仍为双侧安全走廊。

横向区间在当前实现中为高程图可感知的前向 corridor：

```text
y_min = y_riser1 - tread_corridor_half_width + wheel_radius + lateral_margin
y_max = y_riser1 + tread_corridor_half_width - wheel_radius - lateral_margin
```

`y_max <= y_min` 时区域无效；实现不会扩大区间或奖励该落点。当前中心 shaping 只对横向 `y_ref` 进行弱约束，没有人工 X 中心目标。

当前参数台账位于 `M20V11PerceptionCfg`：轮半径 `0.09`、前后/横向 margin 均为 `0.01`、corridor 半宽为 `0.20`。

### 5.2 责任轮 lifecycle

候选仅包括 FL 与 FR。每个候选分别累计：

1. 越过立面之前达到轮底净空；
2. 越过第一道立面；
3. 在安全矩形内形成上踏面支撑；
4. 没有被判定为立面碰撞。

最先满足上述条件的轮锁定为本 encounter 的内部责任轮，但 V11 的主奖励仍需等待另一前轮完成。若两轮同一 step 同时满足，纯函数在 FL/FR 之间随机 tie-break，避免固定低索引产生训练侧偏置。

新当前立面相对已锁定的世界坐标目标发生切换，或目标连续丢失超过配置阈值时，责任轮状态被释放，以便下一阶重新选择。清除/reset 时同样释放。

## 6. V11 reward 配置

V10 的 `first_step_success`、`perception_miss`、`joint_success`、`fr_clearance_progress` 和 `post_first_track_lin_vel_xy_exp` 在 V11 环境中显式设为 `None`，避免 FR 专用项泄漏。

当前 RewardManager 实际登记 `40` 项。V11 不是只使用本节新增的楼梯奖励，而是以官方 M20 Rough 的速度跟踪、姿态、接触和动作约束为底座，再用 V11 的越阶事件和几何判据覆盖其中一部分。

物理仿真步长为 `0.005 s`，`decimation=4`，因此一个 control step 为 `0.02 s`。RewardManager 对普通项按下式计入 episode return：

```text
reward contribution = term_value * weight * 0.02
```

`clearance_progress` 和 `front_hipy_participation_settlement` 在函数内除以 `step_dt`，抵消上述缩放，所以它们按每个立面的一次性 escrow 值结算。`landing_event` 和 `platform_exit_success` 没有抵消 `step_dt`，权重虽然是 `+5.0`，但单次 pulse 的实际 return 贡献约为 `+0.1`。

### 6.1 核心越阶结算链

每只前轮都必须独立完成：

```text
轮心 X 继续向立面前进时累计 clearance escrow
    -> 轮心越过当前立面 crossing
    -> 轮子在安全 X/Y 走廊和踏面高度内建立向上支撑 support
    -> 该轮 candidate_ready
```

`clearance` 只在轮子仍位于立面之前、当前 X 比上帧增加且轮底净空进度创新高时写入 escrow。只把轮子抬高、X 方向没有推进不会获得净空信用；中途抬高也不会立即发奖。

第一只前轮 `candidate_ready` 后只锁定内部责任轮。只有左右前轮都完成 `clearance -> crossing -> support`，才发放：

| 结算项 | 权重 | 单次实际贡献 | 结算语义 |
|---|---:|---:|---|
| `landing_event` | `+5.0` | 约 `+0.10` | 双前轮安全完成时的一帧 pulse |
| `clearance_progress` | `+0.50` | 最高约 `+1.00` | 左右前轮 escrow 求和，每侧范围 `[0,1]` |
| `front_hipy_participation_settlement` | `+0.10` | 最高 `+0.10` | 双前腿最新 hip-y 参与分取平均 |

这条结算链保证主要结果信用遵循 `clearance -> crossing -> support`，且每个锁定立面只结算一次。

| V11 项 | 权重 | 含义 |
|---|---:|---|
| `landing_event` | `+5.0` | V11 双前轮均完成净空、越面和安全支撑后的单次 pulse；第一前轮只锁定内部责任，不立即发奖 |
| `clearance_progress` | `+0.50` | 净空进度 escrow；只在同一轮完成净空、越面和上踏面支撑后结算，避免原地抬高反复领奖 |
| `front_hipy_participation_settlement` | `+0.10` | 保存越阶门控内最新的 hip-y/knee 相对参与分，仅在 `landing_event` 时一次性结算 |
| `front_knee_excess` | `-0.20` | 越阶活跃侧 knee 超过免罚带后的有界惩罚，不受通用高地形 `0.1` 缩放削弱 |
| `landing_center` | `+0.10` | 双前轮完成后，对责任轮横向靠近安全走廊中心的弱 shaping |
| `landing_edge` | `+0.20` | 函数返回负值；分配责任轮后取两只前轮的最大越界量，用于约束较差一侧 |
| `front_riser_contact` | `-1.0` | 任一前轮当前正在抵住立面时的有界逐步惩罚 |

除上述项外，仍保留 M20 官方运动、姿态、接触和 command-following reward。落点奖励是学习信号，而 `front_riser_collision`、base 接触和坏姿态则是硬终止条件。

### 6.2 越阶门控

新的大臂释放、hip-y 参与奖励和 knee 超限惩罚共用逐前腿门控。单侧只在下列条件同时满足时激活：

1. Actor 已确认可见的第一道立面，即 `B1=True`；
2. 当前立面 encounter 仍有效；
3. 轮心距立面不超过两倍轮半径，对当前 `0.09 m` 轮半径即最近约 `0.18 m`；
4. 该侧尚未完成安全支撑；
5. 该侧没有发生立面接触。

这个门控是 fail-closed 的：感知失效、距离过远、已落台或撞立面时，不会继续享受姿态和动作惩罚释放。

### 6.3 action-rate 与名义姿态释放

| 约束 | hip-y 活跃缩放 | knee 活跃缩放 | 设计意图 |
|---|---:|---:|---|
| `action_rate_l2` | `0.35` | `1.00` | 主要解开 hip-y 的快速调整，knee 不释放 |
| `action_smooth_l2` | `0.50` | `0.80` | hip-y 允许较明显参与，knee 只作小幅放宽 |
| 名义关节姿态惩罚 | `0.10` | `0.80` | 大臂允许离开默认姿态，同时恢复对过度屈 knee 的约束 |

缩放只在上述越阶门控内生效，平地、远离立面和越阶完成后恢复原始惩罚强度。这里没有直接奖励 action 幅值或关节速度，因为那会引入“为拿奖而抖大臂”的新捷径。

### 6.4 hip-y 相对参与分与 escrow

对每条活跃前腿，使用相对默认姿态的绝对关节偏差：

```text
hip_share = |delta_hip_y| / (|delta_hip_y| + |delta_knee|)
relative  = clip(1 - |hip_share - 0.50| / 0.10, 0, 1)
activation = tanh(|delta_hip_y| / 0.20)
pose_score = active * relative * activation
```

`pose_score` 范围为 `[0, 1]`。它在 hip-y 和 knee 约各贡献一半时达到相对项峰值，hip-y 贡献低于 `40%` 或高于 `60%` 时相对项归零。`tanh` 激活抑制“两个关节都几乎不动，但比例恰好”的虚假高分，同时不鼓励无上限增大偏差。

训练中不立即发放该分数，而是逐侧保存“最新的有效分数”。若策略先用一次合理 hip-y 姿态，随后又回到 knee-only 姿态，后者会覆盖前者；不保存历史最大值。V11 中只有两只前轮都完成 `clearance -> crossing -> support` 时，左右两侧 escrow 才取平均并以单次 pulse 结算。

### 6.5 knee 免罚带与有界惩罚

```text
excess = clip((|delta_knee| - 0.55) / 0.25, 0, 1)
knee_cost = active * excess^2
```

`|delta_knee| <= 0.55 rad` 时不惩罚，给正常越阶屈膝留出空间；超出后平方增长，到 `0.80 rad` 偏差时达到单侧上限。该项直接使用越阶门控，不经过继承姿态惩罚中的通用高地形 `0.1` 缩放，避免关键阶段约束弱到失效。

### 6.6 TensorBoard 诊断指标

| 指标 | 用途 |
|---|---|
| `v11_front_mobility_gate_rate` | 越阶释放门控的激活比例 |
| `v11_front_hipy_deviation` / `v11_front_knee_deviation` | 门控内两类关节相对默认姿态的偏差 |
| `v11_front_hipy_share` | hip-y 在 hip-y+knee 总姿态偏差中的占比 |
| `v11_front_knee_excess` / `v11_front_knee_excess_rate` | knee 有界 cost 及超过免罚带的活跃侧比例 |
| `v11_front_hipy_escrow` | 当前尚未结算的最新 hip-y 参与分 |
| `v11_front_upperleg_torso_angle` | 大臂轴线与躯干前向轴的夹角，单位 rad |
| `v11_front_upperleg_parallel_cosine` | 大臂与躯干前向的平行程度；越接近 `1` 越平行 |
| `v11_front_riser_contact_rate` | 当前 step 中至少一只前轮被分类为立面接触的环境比例 |
| `v11_front_collision_event_rate` | 本 step 新完成 3 帧确认的前轮立面碰撞事件比例 |
| `v11_front_collision_latched_rate` | 当前 encounter 已锁存确认碰撞的环境比例 |

上述三个碰撞率指标都只用于诊断，没有作为独立 reward 项。其中原始接触状态仍另外驱动 `front_riser_contact` 逐帧惩罚。大臂夹角两项也仍为纯诊断，用于先验证“大臂平行”是否与失败强相关，避免在没有实证前引入新的绝对角度目标。

### 6.7 禁止单前轮盲爬

V11 不再允许“一只前轮安全落台，另一只前轮顶住立面辅助爬升”作为有效完成路径：

1. `require_bilateral_front_completion=True`，平台完成必须先有两只前轮的 `candidate_ready`；
2. `defer_first_front_rewards_until_bilateral=True`，第一只前轮完成时只记录内部责任侧，`landing_event`、净空 escrow 和 hip-y 参与 escrow 均等待第二只前轮完成后才结算；成功时净空结算包含两侧 escrow，hip-y 结算取两侧平均，避免第二前轮只有硬要求却没有结果信用；
3. 前轮立面接触连续 `3` 个 control frame 后确认为 `front_riser_collision`，当前 episode 立即终止；1–2 帧接触分类器尖峰不会直接误杀；
4. 确认前的每个实时接触帧都产生 `front_riser_contact` 惩罚；
5. 上踏面支撑判据同时检查 X、Y、轮底高度和向上支撑力，不再允许轮心在横向安全走廊外被认为安全落台。
6. 平台完成启用 `require_bilateral_rear_support=True`；左右后轮必须在 `15` 帧有限窗口内依次或同时建立支撑，且窗口完成后还需满足既有的平台支撑确认帧数。

`v11_perception_miss` 已从 RewardManager 移除。原实现只在责任轮尚未分配时记录 MISS，与“任何确认碰撞都必须终止”的硬判据不一致。现在奖励层只保留实时 `front_riser_contact` 惩罚，所有新确认碰撞统一记入上述三个 TensorBoard 诊断和 `Episode_Termination/front_riser_collision`。

V12 已有独立的第二前轮完成和净空结算项，因此在 V12 中保留第一前轮的原始发奖时序，避免与第二前轮奖励重复；双前轮完成和立面碰撞终止仍由共享 V11 底层保护。

### 6.8 平台完成与后轮支撑

`platform_exit_success` 权重为 `+5.0`，一次性实际贡献约 `+0.10`。上楼平台不能仅通过 Actor 的 `B1/B2=0` 确认，必须同时满足：

1. 左右前轮已完成 `candidate_ready`；
2. 原始 reward-side 高程图连续 `3` 帧没有检测到立面；
3. 左右后轮都已越过锁定立面；
4. 左右后轮在 `15` 帧有限窗口内依次或同时建立向上支撑；
5. 双后轮支撑准备完成后，继续满足既有的 `2` 帧平台支撑确认。

因此单后轮落台、后轮尚未越面、短时感知丢失或无持续支撑都不会发放平台成功奖励。

### 6.9 完整 40 项奖励/惩罚总账

下表是当前配置经过 `disable_zero_weight_rewards()` 和 V11 `__post_init__` 覆盖后，RewardManager 中实际存在的项，不包括权重为零或已设为 `None` 的历史项。“惩罚幅度”表示函数返回负值、配置中使用正权重的等价惩罚写法。

#### 结果、路线与楼梯专项

| 项 | 有效符号/权重 | 阶段与作用 |
|---|---:|---|
| `landing_event` | `+5.0` | 双前轮安全完成单帧 pulse |
| `platform_exit_success` | `+5.0` | 完成双前轮、无立面和双后轮平台确认后单帧 pulse |
| `clearance_progress` | `+0.50` | 只在成功后结算双前轮净空 escrow |
| `front_hipy_participation_settlement` | `+0.10` | 成功时结算双前腿最新 hip-y 姿态参与分 |
| `landing_center` | `+0.10` | 双前轮完成后对责任轮的横向中心弱 shaping |
| `landing_edge` | 惩罚幅度 `0.20` | 分配责任后惩罚左右前轮中较大的越界量 |
| `world_forward_progress` | `+2.0` | 楼梯且向前 command 下，沿采样世界路线达到命令速度 `50%` 后饱和 |
| `world_forward_deficit` | 惩罚幅度 `3.0` | 世界路线速度低于命令速度 `50%` 时的有界惩罚 |
| `path_lateral_deviation` | 惩罚幅度 `1.35` | 偏离采样路线超过 `0.25 m` 后有界惩罚 |
| `path_heading_error` | 惩罚幅度 `1.00` | 相对采样世界航向的有界 yaw 误差惩罚 |
| `stair_straightness` | 惩罚幅度 `0.05` | 楼梯前进时惩罚机体横向速度和 yaw 角速度 |

#### 前腿姿态、动作和关节约束

| 项 | 权重 | 阶段与作用 |
|---|---:|---|
| `front_knee_excess` | `-0.20` | 越阶活跃侧 knee 超过 `0.55 rad` 免罚带后的有界惩罚 |
| `hipx_joint_pos_penalty` | `-0.4` | 全四腿 hip-x 偏离默认姿态，越阶时不释放 |
| `hipy_joint_pos_penalty` | `-0.1` | 前腿越阶门控内使用 `0.10` active scale，其余关节保留完整代价 |
| `knee_joint_pos_penalty` | `-0.1` | 前腿越阶门控内使用 `0.80` active scale |
| `joint_pos_limits` | `-5.0` | 全部 `12` 个腿部关节的软位置限制 |
| `joint_mirror` | `-0.03` | 对角腿 `FL<->HR`、`FR<->HL` 的 hip-x/hip-y/knee 姿态镜像 |
| `action_rate_l2` | `-0.014` | 一阶动作变化；门控内 hip-y/knee 缩放为 `0.35/1.00` |
| `action_smooth_l2` | `-0.025` | 二阶动作平滑；门控内 hip-y/knee 缩放为 `0.50/0.80` |
| `joint_torques_l2` | `-2.5e-5` | `12` 个腿部关节力矩代价 |
| `joint_acc_l2` | `-2e-7` | 腿部关节加速度代价 |
| `joint_acc_wheel_l2` | `-1e-7` | 轮关节加速度代价 |

前腿 hip-y/knee 名义姿态惩罚在高地形上还会乘继承的 `high_terrain_penalty_scale=0.1`。因此越阶门控内的前腿 hip-y 名义姿态约束很弱，knee 则依靠 `0.80` active scale 和不受高地形缩放的 `front_knee_excess` 保持约束。

#### 机体姿态、接触和终止代价

| 项 | 有效符号/权重 | 作用 |
|---|---:|---|
| `lin_vel_z_l2` | `-2.0` | 机体竖直速度惩罚 |
| `ang_vel_xy_l2` | `-0.06` | roll/pitch 角速度惩罚 |
| `flat_orientation_l2` | `-0.50` | 有界度地抑制持续 roll/pitch，已较平地配置大幅放宽 |
| `front_riser_contact` | `-1.0` | 任一前轮当前接触活跃立面的逐 control-step 惩罚 |
| `non_wheel_leg_contact` | 惩罚幅度 `2.2` | 楼梯前进时非轮腿部超过 `20 N` 的强接触 |
| `non_wheel_leg_scrape` | 惩罚幅度 `1.1` | 非轮腿部接触楼梯且以超过 `0.08 m/s` 滑动 |
| `wheel_impact` | 惩罚幅度 `0.8` | 轮子初次接触时向下速度超过 `0.10 m/s` 的冲击 |
| `undesired_contacts` | `-1.0` | 继承的全局非轮部件接触惩罚 |
| `contact_forces` | `-1.5e-4` | 轮子接触力超过通用阈值的惩罚 |
| `bad_orientation_penalty` | `-1000` | 坏姿态的一帧大惩罚，并同时触发终止 |
| `is_terminated` | `-5.0` | 所有非 timeout 终止的终端惩罚，单帧实际约 `-0.10` |

`non_wheel_leg_contact` 和 `undesired_contacts` 存在有意的覆盖重叠：楼梯上非轮部件的强接触可能同时受到通用接触代价和 V11 楼梯专项代价。

#### 速度跟踪与楼梯阶段基本不生效的继承项

| 项 | 权重 | 实际门控 |
|---|---:|---|
| `track_lin_vel_xy_exp` | `+5.0` | 全局 XY 命令速度跟踪，楼梯上为主要密集前进奖励 |
| `track_ang_vel_z_exp` | `+3.0` | yaw 命令跟踪；楼梯 `wz=0`，因此鼓励不偏航 |
| `stand_still` | `-1.0` | 主要在零 command 时惩罚关节偏离，楼梯强制向前 command 下通常为零 |
| `feet_air_time_ang_z_M20` | `+50.0` | 只在 yaw command 下生效，楼梯阶段通常为零 |
| `feet_slide_ang_z_cmd` | `-2.0` | 只在低线速、高 yaw command 时生效 |
| `rotation_gait_status` | `+2.0` | 原地转向步态奖励，楼梯阶段通常为零 |
| `rotation_gait_symmetry` | `+15.0` | 原地转向接触对称奖励，楼梯阶段通常为零 |

V11 楼梯 column 的 command 被强制为 `vx>0, vy=0, wz=0`。因此上述转向步态项虽然仍在 RewardManager 中，但不是爬楼的主要信号。

### 6.10 硬终止与诊断语义

当前 TerminationManager 包含 `4` 项：

| 终止项 | 判据 |
|---|---|
| `time_out` | episode 达到时间上限，不计入 `is_terminated` 非 timeout 惩罚 |
| `illegal_contact` | `base_link` 接触力超过 `1.0` |
| `bad_orientation_2` | 机体翻倒或 roll/pitch 超过安全判据 |
| `front_riser_collision` | 任一前轮立面接触连续 `3` 个 control step 后确认 |

前轮立面接触在确认前的每一帧都会产生 `front_riser_contact` 惩罚，第 `3` 帧确认后另外触发 `front_riser_collision` 和 `is_terminated`。这允许 `1-2` 帧短时分类尖峰，但不允许持续顶住立面爬升。

`v11_perception_miss` 已不再是 reward。`v11_front_collision_event_rate` 统一统计所有新确认碰撞，不再根据责任轮是否已分配分叉。

### 6.11 楼梯课程晋级和降级

V11 使用 `v11_teacher_gated_terrain_levels`。对楼梯 column，晋级必须同时满足：

1. episode 正常运行到 timeout；
2. 线速度与角速度跟踪的较差一项仍大于 `0.50`；
3. 楼梯世界前进得分大于 `0.45`；
4. 对上楼 column，还必须完成双前轮和双后轮的 `platform_confirmed`。

任一下列条件会导致降级：

- episode 因非 timeout 原因提前终止；
- 速度跟踪得分低于 `0.32`；
- 楼梯前进得分低于 `0.20`。

### 6.12 当前奖励结构的解读要点

1. **密集前进信用强于单次落台 pulse。** `track_lin_vel_xy_exp` 每步最高约 `+0.10`，`world_forward_progress` 每步最高约 `+0.04`，而 `landing_event` 单次约 `+0.10`。楼梯专项中较大的结果信用主要来自双前轮 clearance escrow。
2. **持续顶立面的盲爬路径已被硬终止，但容忍最多两帧短时接触。** 容忍帧仍逐帧扣分，第三帧确认后终止。
3. **双前轮完成条件对称，但落点中心奖励仍只读取责任轮。** `landing_edge` 会检查左右两侧中的较差侧，但 `landing_center` 不是双轮平均。如果后续观察到第二前轮经常勉强落在边缘，应先使用新 run 数据验证，再决定是否改成双前轮平均或较差侧中心分。
4. **V11 没有直接的 `FL<->FR action_mirror` 或轮高/越面时差惩罚。** 左右协调主要由双前轮结算、直线性、路线跟踪和碰撞终止间接约束。当前仍存在的 `joint_mirror` 是对角腿姿态镜像，不是左右前腿同步。

## 7. Actor、Runner 与训练配置

实际使用的是 RSL-RL 5.x 原生 `MLPModel`，不是 V10 的 FR 专用网络：

```text
Actor:  63 -> 512 -> 256 -> 128 -> 16
Critic: 568 -> 512 -> 256 -> 128 -> 1
```

`V11ContinuousRunner` 将仓库中旧式的 `policy` 配置转换为 RSL-RL 5.x 的独立 `actor`/`critic` 配置，并拒绝：

- Actor 输入不是 63D；
- action 输出不是 16D；
- 隐式 depth-image 输入；
- 非 V11 的 policy 标识或未识别字段。

默认训练配置为 fresh run：`resume=False`、`load_run=None`、`load_checkpoint=None`。V10 60D checkpoint 不能被本任务使用；但维度和网络结构一致的 V11 63D checkpoint 可以显式使用 `--resume --load_run ... --checkpoint ...` 续训。本轮 hip-y/knee 奖励修改没有改变 Actor/Critic 结构或 16D action ABI。

## 8. 已执行验证

| 检查 | 结果 | 证据/范围 |
|---|---|---|
| Python 编译 | 通过 | V11 MDP、配置、runner、执行器兼容文件的 `py_compile` |
| 纯几何 | 通过 | 安全矩形、空矩形 fail-closed、FL/FR responsibility 锁定语义 |
| Actor ABI | 通过 | Isaac Lab ObservationManager 实际打印 policy `(63,)` |
| Critic ABI | 通过 | 实际打印 critic `(568,)` |
| Action ABI | 通过 | ActionManager 实际打印 16D |
| Reward 接线 | 通过 | 当前 RewardManager 实际登记 40 项，含 hip-y 结算项和 knee 超限项 |
| PPO 冒烟 | 通过 | 2 environments、1 iteration，rollout、反传和 checkpoint 完成 |
| 连续释放版本冒烟 | 通过 | 加入 encounter release 与确认/hold 后再次完成 2 环境、1 iteration |
| hip-y/knee 纯函数回归 | 通过 | 目标比例、最新值覆盖、只在 landing 结算、knee 免罚带和有界上限均通过 |
| V12 双侧楼梯奖励回归 | 通过 | V11 修改未破坏 V12 双侧 escrow/支撑奖励逻辑 |
| 事件触发冒烟 | 通过 | 256 environments、从 V11 `model_8000.pt` 恢复、10 iterations；新结算、knee 惩罚和全部新诊断量均出现非零值 |
| 禁盲爬 Manager 接线 | 通过 | 2 environments、1 iteration；RewardManager 实载 40 项且无 `v11_perception_miss`，TerminationManager 实载 `front_riser_collision`，CurriculumManager 实载 22 项并含 3 个碰撞诊断，Actor 63D/action 16D 不变 |
| 2 帧历史禁盲爬事件冒烟（已被 3 帧版本替代） | 接线通过，效果未验收 | 256 environments、`model_8000.pt`、10 iterations；`front_riser_collision` 真实非零并最高约 `0.251`，旧策略未产生双前轮主结算 |
| 3 帧碰撞/双后轮精炼冒烟 | 接线通过，效果未验收 | 256 environments、`model_8000.pt`、3 iterations；实时接触率、新确认事件率和锁存率均真实非零，`front_riser_collision` 最新约 `0.0412` |

运行环境为 Isaac Lab `0.54.3`、Isaac Sim `5.1.0.0`、RSL-RL `5.0.1`。训练前还修复了旧执行器配置模块路径，并初始化了仓库锁定的 `deep_robotics_model` 子模块。

## 9. 正式训练状态

以下是 2026-08-24 的历史从零 pilot，保留作为早期基线记录：

```text
task:       M20-Stair-Continuous-Landing-V11-v0
run name:   v11_from_scratch
envs:       4096
iterations: 2000
checkpoint: every 250 iterations
```

输出目录：

```text
logs/rsl_rl/M20-V11-Continuous-Landing/
2026-08-24_18-01-35_v11_from_scratch/
```

### 启动 Bash 指令

首次在新工作区运行时，先初始化仓库锁定的机器人资产子模块：

```bash
cd /home/ubuntu/桌面/js_project/kunyu_ppo
git submodule update --init --recursive
```

随后以 Isaac Lab 的 Python 启动从零 V11 PPO：

```bash
cd /home/ubuntu/桌面/js_project/kunyu_ppo
TERM=xterm \
CONDA_PREFIX=/home/ubuntu/miniconda3/envs/env_isaaclab \
PATH=/home/ubuntu/miniconda3/envs/env_isaaclab/bin:$PATH \
PYTHONPATH="$PWD/source/rl_training" \
/home/ubuntu/IsaacLab/isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-Continuous-Landing-V11-v0 \
  --max_iterations 20000 \
  --headless \
  --run_name v11_from_scratch
```

该配置默认使用 4096 个 environments。从零训练时不要传入 V10 checkpoint；需要对当前奖励改动做控制变量实验时，可以从同 ABI 的 V11 checkpoint 续训。

当前 hip-y/knee 修改的推荐起点是原 V11 `model_8000.pt`，而不是已经在旧奖励下继续训练到更后期的 checkpoint。原因是后期策略可能已经固化“knee-only”局部最优，会弱化新 reward 的可观测效果。

需特别注意：Python 训练进程在启动时就加载了 reward/config，不会因为源码在磁盘上被修改而热更新。2026-08-27 修改前已经启动的 V11 训练进程仍使用旧逻辑；必须新建训练进程才能实际使用本文的调整。

256 环境、10 iteration 的事件冒烟已确认 `landing_event`、hip-y settlement、knee excess 以及新诊断量都能真实触发。这只能证明接线有效，不能说明策略已学会更协调的跨阶步态。

## 10. 当前限制与下一轮验收要求

以下项目尚未完成，不能省略：

1. **完整 XY 语义仍有限。** 当前 Y 区间来自固定宽度的可感知 corridor，不是从高程图显式检测出的踏面侧缘。它是 deployable 的安全带假设，而不是真实侧边界测量。
2. **第二阶与全楼梯成功率未验证。** 现有 `landing_event` 是每个前轮 encounter 的局部跨阶事件，不等价于 FULL 成功。
3. **镜像公平性未验证。** 必须分别统计 FL/FR 被选择比例、成功率、前轮碰撞率和镜像场景一致性。
4. **几何层诊断仍不完整。** hip-y/knee、责任轮和大臂夹角已有 TensorBoard 指标，但矩形有效率、空矩形率、X/Y 误差和立面切换次数仍未完整记录。
5. **感知鲁棒性未做系统 sweep。** 应覆盖台阶高度/深度、yaw、噪声、点云缺失、短时遮挡，以及零/后退/侧移 command。
6. **部署导出未验证。** ONNX/JIT、真机高程图字段顺序、延迟和控制频率仍需独立验收。
7. **`v11_continuous_policy.py` 当前未被训练入口使用。** 实际训练由 runner bridge 构造 RSL-RL 原生共享 MLP；该遗留文件应在后续清理或改造成明确的测试/导出模块，避免误认为它是生效网络。

## 11. 推荐的训练后判据

完成 pilot 后，应至少输出并与 V10 基线对照：

- 第一/第二立面检测有效率和误检率；
- 安全矩形有效率、空矩形率；
- 责任轮 FL/FR 选择比例、成功率与前轮碰撞率；
- 轮心 X/Y 是否在区间内、距离中心的误差、边缘越界率；
- 第二立面碰撞率、逐阶成功率、整段楼梯成功率；
- command 方向速度、倒退比例；
- 镜像地形的一致性；
- 同 seed、同地形与同 command 分布下的 V10/V11 对照。

对 2026-08-27 hip-y/knee 修改，还应使用同一个 `model_8000.pt`、同 seed 和同地形集做旧/新 reward A/B，重点检查：

- `landing_event`、通过率和前进速度不能因释放大臂而显著退化；
- `v11_front_hipy_share` 不再长时间落在低 hip-y 贡献区，但也不应因为奖励而长期饱和在 `0.50`；
- `v11_front_knee_excess_rate` 与 `v11_front_knee_excess` 相比旧策略下降，同时不应出现 knee 几乎不动导致净空失败；
- `v11_front_hipy_escrow` 必须与 `landing_event` 一起解读：高 escrow 但无 landing 不是成功，而是“有姿态、无结果”；
- `v11_front_upperleg_torso_angle` 和 `v11_front_upperleg_parallel_cosine` 应按成功/失败 encounter 分组，不要仅看全局均值；
- sim2sim 视频需同时检查大臂是否真实参与越阶、是否出现高频抖动，以及左右前肢是否出现新的不对称。

如果 hip-y 仍无法参与，优先检查门控激活率和 escrow 是否有效结算，不要立即继续增大 reward 权重。如果 hip-y 已参与但出现抖动，应先收紧 `action_smooth_l2` 的 hip-y 缩放，而不是恢复对 hip-y 名义姿态的强约束。如果 knee 超限仍高，先用分布确认 `0.55 rad` 免罚带是否过宽，再决定调整免罚带、超限尺度或权重。

任何“V11 优于 V10”或“策略自主选择迈腿”的结论，都必须基于这些当前 V11 run 的统计结果，而不能使用旧 V10 checkpoint 或旧报告替代。
