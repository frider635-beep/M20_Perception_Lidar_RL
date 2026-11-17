# V135 上楼三轮支撑补丁（未启用长训）

直接修改原V13.5，不新增任务。当前正在训练的进程不会热加载此次修改。

## 修改与接线
1. v135_riser_identity.py：目标高度pool及native target_match改用真实踏面横向边界，保留纵向身份和正高度筛选；不再把首条扫描ray的lane永久锁成±0.15m。原scanner helper无face参数时保留原行为。wheel接触仍使用前轮真实踏面范围。
2. v135_ascent_support.py：摆动前腿之外的另三轮（对侧前轮+两后轮）是支撑组。B1依照实际progress领先侧，seed后依照expected侧。连续载荷质量、支撑腿workspace、支撑轮接触点滑移、地形相对pitch/roll、角速度形成[0,1]质量。不是固定关节姿态，不硬锁动作，不要求摆动腿接地。
3. v135_forward_credit.py直接调用支撑模块；只将原有有限前进额外信用乘0.5+0.5Q，基础速度收益保留1。信用预算仍按当前台阶新前进距离消耗，原地无额外奖励。非ascent active显式factor1，缺地形/NaN不能获得好支撑信用。
4. v135_behavior_metrics.py新增每一级frames/height_missing/valid/ordered/error及episode严格prefix计数，级数来自真实flight_riser_count；增加三轮支撑、slip、pitch/roll误差、ground evidence指标及分母。Logger每步输出完整key集合避免漏计。

## 范围
下楼、flat恢复、Actor63 MLP、Critic574、Action16、PPO8192×24、reward配置权重、训练terrain均未改变。当前仍是0.30m踏面/现有级数分布；尚未新增1/3/6/10/12级terrain训练，不能声称任意长度泛化已解决。

## 验证与限制
所有V135 CPU回归通过；新测试覆盖左右对称三轮支撑、缺后支撑、摆动腿workspace不误罚、无地形/NaN不给信用、四方向横移匹配、越界不匹配。compile通过。16env×8step+一次真实PPO update通过，最终日志/tmp/v135_midflight_final_smoke.log。独立复核发现NaN可污染非上楼分支，已加finite隔离并复测。

这是代码验证，不是新策略行为验收。三轮载荷门槛20N等暂沿用已有下楼质量的量级；不等于已证明动力学稳定，不包括真实CoM动态支撑裕度。共享whole-body只读数据的pitch/ground/workspace误差会影响信用，必须查看ground_valid和分项指标。下一步用32100同场景对照支撑/速度/整段交替，并验证下楼/flat无回归，不能仅看reward。

## 已启动2026-09-21
用户最终要求从34200追加30000轮，终点64200。服务m20-v135-three-support-34200-plus30000，PID62232；日志logs/rsl_rl/M20-V13.5-Final-FromScratch/2026-09-21_14-17-34_v135_three_support_resume34200_plus30000_e8192_r24。实测34205至34210更新，checkpoint34200落盘，manifest源码哈希匹配，8192×24。此前to43100服务已停止，无重复训练。
