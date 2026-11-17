# V13.5 上楼交替与下楼支撑漏洞审查

审查对象是现有 `M20-Stair-V13.5-V11FD-NativeAlternation-v0`，没有新建任务。训练设置仍为纯 MLP Actor 63D→16D、574D Critic、8192 env × 24 rollout、5 epoch、4 minibatch、1e-3 adaptive、entropy 0.003、KL 0.01。上楼原生速度与前进权重保持 5.0/2.0。

## 可由代码证明的漏洞与修复

1. 整段 episode 一次错误后，局部相邻链曾被永久关闭。现在整段严格成绩仍失败，但 live target 重同步后，新的正确相邻片段能再次获得局部链与速度收益。
2. 上楼的速度加成曾在已离开可行动台阶后仍留在 ascent terrain 上。现在加成只在严格当前目标与 V11 可行动 riser 匹配时支付。确认跳级后的速度折扣在扫描器暂时指向后续阶时仍保留，直到正确局部动作恢复。
3. 单腿 dense progress 曾能由 V11 当前扫描到的未来 riser 产生，即使它不是严格目标 Bk。现在只有 `target_match` 才更新历史最好 potential 并支付进度。
4. 下楼 `recovery_done` 曾允许短暂站好后再次后伸，马上恢复完整速度收益。现在后轮支撑质量在恢复事件后仍持续控制速度收益；恢复质量同时检查后腿长度相对中性姿态的上下两侧，避免折叠腿被当成正常姿态。
5. 新增只读全身指标：四轮承重、后轮法向载荷占比、机身沿路径速度、俯仰角速度、机身与后轮前进差、后腿最短长度比。它们不进入 Actor/Critic，也不增加 reward。
6. 冻结回放又发现 4 次“同一前轮安全跨过 B1 后，在 B1 上稳定支撑”，V11 landing 仍为 0。现有 strict ledger 增加锁定当前物理阶的跨面与支撑结算：必须先在目标阶高度安全跨面，再在目标阶踏面内连续承重；未来阶高度、越界位置、低净空和瞬时接触均不结算。它只推进已有 live target，没有第二套状态机。

核算事件金额时不能只看配置 weight：Isaac Lab 对 reward term 再乘 `step_dt`。V11 原生 landing/platform 单帧脉冲在 `step_dt=0.02`、weight=5.0 下每次实际为 0.1；V13.5 strict event 已按 `1/step_dt` 归一，配置 weight 才是每次实际支付额。

## 冻结模型证据

使用旧的 `model_5000.pt`、固定种子 13542、128 env × 1000 control steps 做确定性回放；不调用 optimizer。该 checkpoint 由修复前的奖励训练，回放环境采用当前代码，因此只能审计其现有动作与新门控的可达性，不能当作修改后的训练结果。

在 128,000 个 transition 中，记录到 1,590 次 V11 前轮 crossing，但只有 1 次 V11 稳定 landing、0 次严格 valid/ordered。V11 可行动台阶帧为 10,890，其中 strict 目标匹配 2,277；扫描器指向 strict 目标前方下一阶的帧为 9,149。短窗口中的 816 次 crossing 有 532 次指向下一阶、只有 83 次匹配 strict 目标，且 0 次前轮安全支撑结算。这说明旧策略大量移动到未来阶附近，却极少完成当前阶支撑；尚不能仅凭这些计数断言是感知误差、动力学跳级还是 detector 过严。

下楼全身诊断窗口共 10,487 帧，其中后腿长度比低于 0.85 的帧为 1,148；四轮同时达到接触阈值的比例约 0.63。后轮过长（>1.15）的帧为 0。此 checkpoint 更需要检查折叠/失载与前后支撑转移，而非单纯增加“后伸处罚”。

## 验证和边界

Python compile、V13/V13.5 contract tests、`16 env × 8 step + 1 PPO update` runtime smoke、Final reference audit 通过；Actor/Critic/Action 维度和 PPO 参数未改。按用户最新授权，现有任务从零开始 8192 env × 24 rollout × 30000 iterations，systemd unit 为 `m20-v135-locked-settlement-20260919.service`，run name 为 `v135_locked_settlement_e8192_r24_from_scratch_30k_20260919`。启动后已观察到 Learning iteration 6/30000，服务 active，未与旧 run 同名覆盖。

这不是交替最优性或自然下楼的行为验收。训练中仍需在相同 terrain/command 下查看 crossing 前后完整事件窗口，确认当前 Bk 的轮底高度、真实踏面接触、前后轮载荷与机身俯仰。不能凭本次计数直接提高 entropy 或宣布奖励排序已经逆转。
