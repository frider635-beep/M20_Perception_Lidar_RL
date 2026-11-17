# V13.5 训练评估与报告口径

当前任务仍为 `M20-Stair-V13.5-V11FD-NativeAlternation-v0`。训练启动时自动写入
`v135_manifest.json`，记录实际 Actor/Critic/PPO 类、观测和动作维度、rollout、奖励权重、
Git HEAD 以及 V13.5 相关源码哈希。训练正常结束或收到可捕获的中断时，自动从本次
TensorBoard 事件文件生成 `V135_TRAINING_REPORT.md` 和 `v135_training_report.json`。
已有运行可显式指定目录生成快照：

```bash
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  scripts/tools/report_m20_v135_training.py \
  --run-dir logs/rsl_rl/M20-V13.5-Final-FromScratch/<具体运行目录> \
  --status manual_snapshot
```

报告按事件计数求和后再除以真实分母，不对每个环境的比率简单平均。开头与末尾各取
20% 日志点列出趋势。所有行为项均带 `Diag/V135Behavior/` 前缀。
当前 m20_wzh 安装的 RSL-RL logger 对 rollout 内的逐步 `extras` 取平均，并只从首步
读取键名。采集器因此每步输出完整键集，同时将事件次数按 rollout 长度换算；
`test_m20_v135_training_report.py` 用首末帧合计 3 次事件验证 TensorBoard 最终仍记为 3。
上楼事件同时按 terrain level 与命令速度区间 `[0.1,0.5)`、`[0.5,1.0)`、
`[1.0,1.5)`、`[1.5,2.0)` 记账。报告分别给出各层的交替/合法完成、跳级/有效帧和路径速度；
这是两个边际分层，不能替代 terrain × command × stair phase 的严格匹配实验。

| 目标 | 分子或观测量 | 分母与判读 |
|---|---|---|
| 上楼阶段访问 | 完整起点 episode 中到达物理 B1、严格 B2/B3 的数量 | 完整起点上楼 episode；另记局部起点帧，不能把局部成功充作整段成功 |
| 上楼交替与跳级 | 物理账本的 valid/ordered/wrong/skip/dual 事件、严格平台事件 | 分别报告合法完成数和完整起点 episode 数；严格平台事件是账本代理，不等于视频复核 |
| 上楼速度 | 路径方向速度求和 | 完整起点上楼帧数；与交替率同时看 |
| 下楼姿态 | 工作空间质量、超长/折叠/后伸帧、持续异常、俯仰误差、轮地滑移 | 真正进入下降监测的帧数；异常可以重叠，不能相加成成功率 |
| 楼梯出口 | 几何出口、稳定恢复、恢复耗时、未恢复退出 | 恢复耗时仅对成功样本计算，必须同时看未恢复退出比例 |

`v135_stage_starts.enabled` 默认为 `False`。未来若启用局部起点采样，报告会另记局部
帧和事件；正式完整起点的分母不会混入局部 episode。源码已经过恢复机制冒烟测试，
但尚无真实 B1/B2 合法前缀经复位后稳定延续的实测证据，因此默认不启用。

报告不会自动将 reward 增长解释成交替改善，也不会自动给出
`RETURN_ORDERING`、`VISITATION_STARVATION` 等因果结论。训练后还需对同一批 checkpoint
做固定 terrain、command、seed 的冻结评估，并抽查严格账本与真实物理接触是否一致。
若轨迹没有这一层证据，报告在“未完成验证”中明确保留这一项。
