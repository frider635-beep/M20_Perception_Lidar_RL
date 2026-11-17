# 面试表达稿

## 2 分钟版本

我做的是 M20 PRO 轮腿机器人在楼梯/复杂地形上的雷达感知步态训练。项目的关键不是单纯堆 reward，而是把 LiDAR/点云形成的局部高程图压缩成可部署的几何接口，再接到官方 command-conditioned rough locomotion 的 PPO 框架里。

第一版 V10 用第一阶的 `B/H/D` 扩展官方 57D observation，形成 60D actor，动作仍保持官方 16D。它解决的是“感知教第一阶，之后 locomotion 继续服从实时速度指令”。后来我发现旧 FIRST/MISS reward 会压过官方 locomotion，于是做了 reward audit，把新增感知项调整成更小、更对称的教学信号。

V11 的主要升级是从单第一阶变成两立面连续感知：actor 是 `57D + B1/H1/D1 + B2/H2/D2 = 63D`，并且去掉了 FR 专用动作路径。跨阶不再指定哪条前腿，而是通过 clearance、crossing、support、front-riser collision 这些几何事件锁定责任轮和结算奖励。复杂 PGTT 和接触信息只在 reward/critic 侧使用，actor 不偷看部署拿不到的信息。

V12 则是一个 teacher/control-arm：它保留 V11 的 63D 可部署输入，但移植 V9 teacher 的 4-expert soft MoE、dense map reward、terrain curriculum 和 reset/domain randomization，用来比较强 teacher 是否能在同一部署接口上提供更好的学习信号。

## 5 分钟版本结构

1. 背景：M20 是轮腿机器人，动作空间是 12 个腿关节位置 + 4 个轮速，底座是官方 rough locomotion。
2. 约束：部署时不能依赖仿真特权 label，所以 actor 只能接收本体感觉、命令和雷达几何压缩特征。
3. V10：第一阶 B/H/D，解决 first-riser handoff；reward audit 防止感知奖励主导 locomotion。
4. V11：两立面 6D，63D actor，FL/FR 对称，状态机结算 clearance -> crossing -> support。
5. V11-FD：signed edge contract，把上台阶/下台阶和 feature contract sidecar 接进部署约束。
6. V12：V9 teacher 风格 MoE 与 dense reward，但 actor 输入仍受 V11 6D 限制。
7. 验证：静态契约、纯函数单测、Isaac PPO smoke、短训信号可达、固定 rollout、sim2sim、真机。

## 深挖回答素材

### 为什么不直接把 315D 高度图喂给 actor？

因为部署一致性和泛化风险更重要。高度图直接进 actor 会让模型依赖仿真扫描分辨率、缺失模式和隐式地图结构；这个项目选择把高度图先变成 `B/H/D` 或两组 `B/H/D`，让 actor 只看到任务必须的几何量。PGTT 大图留给 critic/reward，负责训练时提供更密集的监督。

### V11 为什么要去掉 FR 特化？

V10 的 FR 专用网络路径和奖励会形成结构性偏置，策略容易变成“固定右前腿处理第一阶”。V11 改成 FL/FR 对称候选，责任轮来自物理完成事件，tie-break 也避免固定低索引偏置。

### clearance escrow 的价值是什么？

它防止 reward farming。只把腿抬高但没有向立面推进，不应该获得净空信用；净空进度必须和轮子朝当前立面前进绑定，并且要等 crossing/support 后结算。

### V12 teacher 怎么避免偷看？

V12 的 reward 函数可以读取训练侧 PGTT map，critic 也可以更强，但 actor observation group 仍然是 V11 的 63D。runner 里有 actor/action 维度 gate 和 MoE 结构 gate，训练脚本也固定 V12 task。

### 我会主动说的边界

目前这份资料是基于仓库静态证据整理。已有项目文档中，V11/V11-FD/V12 的完整训练、sim2sim 和真机验收并不是全部闭环完成的状态；面试中应把“接线可达、合同已固定、短训有信号”和“策略已最终验收”分开讲。
