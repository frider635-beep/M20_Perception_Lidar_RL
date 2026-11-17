# V11.1 出口恢复与跨级信用修复（2026-09-23）

## 范围
直接修改原 M20-Stair-V11.1-V11FD-RearPosture-v0，不引入交替。Actor 63D、Critic 568D、Action 16D、原 MLP/PPO 不变。V13.5 rear-support-transfer 与 whole-body-descent 文件未修改，奖励权重也未修改。

## 证据与修复
原 bilateral landing 将 crossing/support 累积在 reference 上，但扫描槽变化后直接采用新槽的高度与位置；只有 responsible 已确定才允许清空。这允许未完成旧阶的历史参与新阶判断。

V11.1 现在在沿路径的台阶身份切换时清空旧历史；横向扫描漂移不作为换阶。若上一阶未双前轮完成，新 encounter 的 landing、support、clearance settlement 信用关闭。物理候选仍可完成以便重新同步，下一正常逐级事件可以恢复信用，不永久关闭后续学习。

当前踏面支撑必须在下一 riser 前（轮半径留边），连续三帧有效；crossing 必须有该 encounter 的轮子在边缘前的历史。站在台阶上、空抬、跨到下一踏面不能直接算当前完成。仍使用原生扫描边缘，没有引入额外 Actor 观测或左右顺序。

V13.5 平地恢复原来要求所有扫描边缘消失。V11.1 改为忽略位于所有轮子身后的边缘；轮下/前方边缘仍阻断恢复，仍要求全轮越过出口与前方平面射线覆盖。重入楼梯要求前方真实边缘持续确认，不能被身后的保留目标重置。V13.5 本身默认行为保持原样。

## 验证
- Python compile / git diff --check：PASS。
- V111_DIRECTIONAL_EXIT：PASS，身后、轮下、前方、旋转坐标。
- V111_ADJACENT_TRAJECTORIES：PASS，跳 B1 后 B2 不补领；B3 正常完成恢复；reset；站在台阶上无 crossing；跨出踏面无 support。
- V111_V11FD_REAR_POSTURE_CONTRACT、V11_FD_CONTRACT_REGRESSION：PASS。
- V135_BEHAVIOR_CREDIT、V135_EXIT_GEOMETRY：PASS。
- 实际 Isaac 16 env ×24 step +1 PPO update：PASS；V11 model_29999 Actor/Critic/optimizer 加载 PASS。
- 运行日志：artifacts/systemd/v111_exit_adjacent_smoke.log。

## 限制与运行状态
这是奖励判据修复，不是动作硬限制，不能证明策略已不再跳级。扫描漏检仍需真实回放核对；速度收益未修改，行为效果必须用新训练结果验收。
当前既有训练未停止或重启，检查时32920/59999，它仍使用进程启动时的旧实现。修改只有新进程/resume 后生效。没有声称当前曲线属于修复版。

## 用户指定 checkpoint 后续执行
用户指定 model_32900.pt 后，旧服务已停止（MainPID=0），启动 m20-v111-exit-adjacent-resume32900.service。8192×24，延续原59999终点，剩余27099轮；源码与checkpoint哈希见 artifacts/systemd/v111_exit_adjacent_resume32900_receipt.json。

## 首次入阶物理 B1 修复（2026-09-23 23:00）
用户进一步确认：只有平地入阶直接落 B2，后续逐级、下楼和出口正常。本轮仅增加 V11.1 的 physical_first_tread 开关。

现象：上一补丁检测扫描槽之间的漏阶，但首次扫描若已是 B2，没有旧 reference 可比较。
真因（代码层）：首次 assign_reference 没有物理 B1 校验；尚未证明该漏洞是用户回放的唯一原因。
修法：复用 V13.5 真实 terrain column 与 flight_anchor 几何，仅为 reward 侧生成 B1/B2 边界；只从匹配真实 B1 的扫描缓存高度。初始只见 B2 时不初始化 B1 高度。B1 目标保持到双前轮实际完成，随后回到原 bilateral 逻辑。若双前轮越过 B2 且接触持续3帧，记首次跳级失败，不给B1完成信用，释放入口锁并保持跨级信用关闭以允许后续局部恢复。
不改 Actor/native observation、网络、PPO、reward 权重、下楼与平地恢复。没有左右交替要求。

判据：V111_PHYSICAL_B1_ENTRY PASS（首见B2拒绝，B1扫描提升到B2仍锁B1，正常B1支撑可结算，空抬不结算）；原 adjacent 回归 PASS；16env×24step+1PPO update 与 V11 checkpoint 加载 PASS；出口方向与V11FD CPU回归 PASS。日志 artifacts/systemd/v111_first_tread_smoke.log。
运行：本轮仅改代码与测试，未重启长训。新规则需下次resume才能进入训练；未验证实际策略跳级率改善，不宣称已保证第一步必踩B1。对于无B1高度证据时保持不发信用；仅支持已核验的 inverted-pyramid ascent geometry。

## 物理 B1 修复版 resume
用户授权重启后，停止 m20-v111-exit-adjacent-resume32900.service（MainPID=0）。检查最新完整 checkpoint 为原run/model_38400.pt，torch.load验证包含actor/critic/optimizer及iter=38400。启动 m20-v111-physical-b1-resume38400.service，8192×24，剩余21599轮，终点59999。启动脚本 artifacts/systemd/start_v111_physical_b1_resume.sh；哈希收据 v111_physical_b1_resume_receipt.json；日志 v111_physical_b1_resume38400.log。同机资源约39GiB GPU空余、86GiB可用RAM，旧同规模训练内存需求已实测，服务MemoryMax72G。

## 同级先后支撑交接（2026-09-24）
用户目标再次明确：任意前腿先上 B1，对侧跟到 B1 支撑，再以同样方式上 B2；每级领先侧自由，不是 FL/B1→FR/B2。

原先 candidate_ready 使用各腿历史 support 的或累计；左腿已经去 B2、右腿后到 B1 也可能形成双前轮完成。现在 V11.1 的 same_tread_handoff 要求两前轮当前同时在同一级连续有效支撑，事件单次锁存；换阶漏阶判断与首次B1释放使用这个物理完成状态。该级曾出现一条前腿先稳定支撑，另一条随后跟上，才允许完整落脚信用。连续两帧双前轮无接触且都高于下级踏面4cm时，该级落脚信用关闭；仍允许物理完成用于局部重新同步，不终止episode、不永久关闭后续学习。没有强制左右交替、动作锁、网络/PPO修改、下楼或出口恢复修改。

新增观测日志 v111_handoff_bilateral_confirmed / unsupported_handoff / bilateral_landing_event，区分物理完成和实际有资格领取的奖励事件。

测试：V111_SAME_TREAD_HANDOFF、ADJACENT_TRAJECTORIES、PHYSICAL_B1_ENTRY PASS；涵盖左右镜像、错时错层支撑不得结算、真正共同支撑一次结算、同步落地/腾空无完整奖励、reset。实际16env×24step+一次PPO更新与checkpoint加载 PASS；CPU出口方向、V11FD、V111合同PASS；compile/diff检查PASS。日志 artifacts/systemd/v111_handoff_smoke.log。

旧训练11:14被SIGKILL（服务Result=signal，status=9），日志未提供发送者/根因，不归因为训练数值崩溃。最新完整模型model_49500.pt已torch.load核验actor/critic/optimizer/iter。用户授权后启动m20-v111-handoff-resume49500.service，8192×24，继续10499轮到59999。源码哈希与来源见artifacts/systemd/v111_handoff_resume49500_receipt.json。

限制：此次改变落脚收益资格，不是动作硬约束，速度收益仍保留；不能凭单测宣称步态已经解决。真实策略的双前腿腾空与错层比例需后续回放验证。
