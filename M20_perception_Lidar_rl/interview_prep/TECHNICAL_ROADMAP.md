# 技术路线图

## 一句话主线

这个项目的核心不是“训练一个能上楼梯的策略”这么粗，而是把真实雷达/点云可部署感知压缩成稳定的几何契约，再把它接进官方 M20 rough locomotion 的 PPO 框架，让策略在 16D 轮腿动作空间里学会跨台阶、保持速度指令、并逐步消除单腿捷径。

## 0. 项目底座：官方 M20 Rough Locomotion

底座是 Isaac Lab + RSL-RL PPO 的 M20 PRO 轮腿 locomotion：

- Actor 动作保持官方 16D：12 个腿关节位置目标 + 4 个轮速。
- 基础观测来自官方本体感觉、速度指令、关节状态、历史动作等。
- 任务不是切换“平地策略/楼梯策略”，而是在同一个 actor 里让感知只补足楼梯局部几何。

面试表达：

> 我保留了官方轮腿控制接口和 command-following locomotion，只把可部署的局部几何感知作为额外输入和 reward 监督接进去，避免做一个只在仿真 teacher label 上成立的楼梯特化策略。

## 1. V10：第一阶感知 handoff

V10 的最小闭环是：

```text
LiDAR / Point Cloud
  -> Local Elevation Map
  -> First-riser extraction
  -> B/H/D
  -> 57D official proprioception + 3D geometry = 60D actor
  -> 16D M20 action
```

关键设计：

- `B` 表示前方是否存在有效第一阶。
- `H` 表示第一阶高度。
- `D` 表示沿接近方向到第一阶立面的距离。
- 感知 reward 只负责教 FIRST；FIRST 后重新交给实时 command velocity tracking。

当时解决的问题：

- 旧 reward 中 FIRST/MISS 相对官方 locomotion reward 过大，MISS 方差会主导训练。
- 新版本把 FIRST/MISS 调成对称教学脉冲，并恢复官方速度跟踪为主信号。

可讲亮点：

- 没把整张高度图塞给 actor，而是先做部署友好的几何压缩。
- 用 reward audit 量化了新增项相对官方项的质量占比，避免“楼梯奖励淹没 locomotion”。
- 明确留下证据边界：reward balance 恢复不等于完整楼梯成功已经解决。

## 2. V11：两立面连续感知 + 去 FR 偏置

V11 从“第一阶 B/H/D”升级为“两道立面连续状态”：

```text
official 57D
  + B1/H1/D1
  + B2/H2/D2
  = 63D actor
```

核心变化：

- 从 21x15 yaw-local ray-hit 点中提取当前和下一道立面。
- Actor 维持 63D -> 16D ABI，且不再有 FR 专用动作路径。
- Critic/奖励侧可以使用 PGTT 清洁地图与几何真值，但 actor 不接收 315D PGTT 或 reward-side 状态。
- FL/FR 都是候选轮，责任轮由物理完成事件锁定，避免固定 FR 偏置。

V11 的关键状态机：

```text
target riser active
  -> clearance escrow: 轮底净空且向立面前进
  -> crossing: 轮心越过当前立面
  -> support: 上踏面安全支撑
  -> candidate_ready
  -> responsibility / alternation / bilateral settlement
```

安全设计：

- landing region 的 X 是越过第一立面的单侧下界，不用第二立面伪造固定落点。
- Y 是可感知 corridor 的安全区间。
- 垂直立面接触要经确认帧后终止，防止盲爬或撞面后继续吃奖励。
- V10 的 FR 专用 `first_step_success/perception_miss/joint_success/fr_clearance_progress/post_first_track_lin_vel_xy_exp` 在 V11 中显式断开。

面试表达：

> V11 最大的工程价值是把“看见台阶”变成了一个连续、对称、可部署的几何状态机。Actor 只拿 6 个几何标量，复杂的 PGTT 和接触判据只在训练奖励和 critic 中使用，所以部署接口不会被仿真特权信息污染。

## 3. V11-FD：上下台阶与 signed edge contract

V11-FD 是从 V11 派生的独立从零任务，重点是把上台阶/下台阶统一进 signed edge contract：

- Actor 仍是静态 `obs[1,63] -> actions[1,16]`。
- Feature contract 从正立面扩展为 `v11_signed_edges_v2`。
- Ascent completion 使用 `bilateral_same_riser`。
- Descent completion 要求双前轮支撑后双后轮支撑，不规定左右 gait 顺序。
- 导出时要求 ONNX metadata 和 `.feature_contract` sidecar 匹配，部署端 fail closed。

面试表达：

> V11-FD 不是简单加一个下楼奖励，而是把感知契约从正高度立面扩展为 signed edge，让同一个 63D actor 接口可以表达上台阶和下台阶，同时把部署侧 feature contract 变成强约束。

## 4. V12：V9 teacher 风格任务 + V11 6D 可部署感知

V12 可以作为“教师/对照任务”来讲：

- 继承 V11 的 63D actor 输入：57D proprioception + 两个 B/H/D。
- 明确固定 `front_completion_mode = bilateral_same_riser`。
- 使用 `v11_positive_risers_v1`，保留正立面合同，不混入 V11-FD signed descent。
- Actor/Critic 架构改为 V9 兼容的 4-expert soft MoE。
- Reward/curriculum 向 V9 teacher 对齐，包括地形分布、reset/domain randomization、dense map-based stair rewards。
- PGTT map 用于 reward/critic，不进入 actor observation group。

V12 的定位：

- 不是替代 V11-FD 的部署策略，而是把 V9 teacher 的强监督风格移植到 V11 的 6D 可部署感知接口。
- 适合回答“你如何构造 teacher、如何保证 teacher 没偷看部署时拿不到的信息”。

面试表达：

> V12 的意义是做一个强 teacher/control arm：它用 V9 证明过有效的 dense map reward 和 MoE 表达能力，但 actor 仍然只看 V11 的 6D 几何接口。这样 teacher 强，但部署输入仍受约束。

## 5. V13/V13.1/V13.2：步态顺序与样式奖励演化

V13 系列围绕 V11-FD 的“单侧/交替/有序交替”继续做任务设计：

- V13 保留 V11-FD 的 63D actor/action/terrain/command/基础 reward，只把 ascent completion 改成逐物理立面一侧安全完成，并增加软交替偏好。
- V13 不 mask action、不用样式失败终止、不把 expected side 暴露给 actor。
- V13.2 则显式扩展到 68D actor：前 63D 是冻结 V11 输入，后 5D 是可部署有限状态输出，用来承载有序 gait memory。

面试可选讲法：

> 如果面试官追问“feed-forward policy 怎么记住上一阶是哪条腿”，我可以讲 V13.2：要么显式加一个部署可复现的状态 ledger 到 actor 输入，要么换 recurrent/history carrier。我们选择了 5D finite-state ABI，并规定它只能由 LiDAR/PGTT、base pose、joint/FK 推出，不能用 reward/contact ledger 偷看。

## 6. 训练与验证路线

建议按事件讲，不按周历讲：

1. 静态契约：task registration、actor/action dim、feature contract、runner class、reward disconnect。
2. 纯函数单测：riser extraction、landing region、clearance/crossing/support、bilateral/alternation gate。
3. Isaac PPO smoke：少量 env/iteration 确认 Manager、Reward、Termination、Curriculum 真接线。
4. 中等规模短训：只证明信号可达，例如 landing/settlement/collision diagnostics 非零。
5. 固定 rollout matrix：clean/noisy、不同台阶高度/深度/yaw/遮挡/命令方向，对 V10/V11/V11-FD/V12/V13 做同条件比较。
6. 导出与 sim2sim：ONNX/JIT、feature contract sidecar、部署输入字段顺序、延迟和控制频率。
7. 真机验收：只在前面全部通过后做，指标包括逐阶成功率、碰撞率、倒退率、速度跟踪、左右公平性。

## 7. 面试追问清单

- 为什么不用整张高度图直接进 actor？
- 为什么 V11 要从 FR 特化改成 FL/FR 对称？
- 如何防止 reward-side PGTT/接触信息泄漏到部署 actor？
- V11 的 clearance escrow 为什么要求“轮子前进”而不是只看抬高？
- V12 为什么要保留 V9 teacher 的 MoE 和 dense reward？
- V11-FD 的 signed edge contract 对部署有什么价值？
- 怎么证明模型真的学会了，而不是靠 reward farming 或 reset farming？
- 当前哪些结论已经验证，哪些只能叫候选或未验证？

## 8. 当前待补证据

- V11/V12 的正式训练结果和固定 rollout 对照。
- V11-FD 完整 30000 iteration、sim2sim、真机验收。
- V13/V13.2 在统一 matrix 下相对 V11-FD 的速度、成功率、生存率和交替合规性。
- 部署端实际 LiDAR/PGTT producer 与训练端 `actor_observation_manifest` 的逐字段一致性。
