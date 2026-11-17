# V13.5 可变踏深与入阶承重修复（代码已验，未启动新训练）

## 契约与参照

参照为 `.codex-scratch/v135_tread_entry_before/` 中本轮修改前代码，以及 three-support 从34200续训的配置。用户授权只修上楼真实踏深一致性和入阶承重信用；原下楼、楼梯到平地、Actor/Critic/Action、PPO保持。没有创建新task、没有动作门控或额外Actor输入。

既有事实：固定30cm训练不能证明更深踏面泛化；一次新腿progress才触发的支撑额外信用不能覆盖完整入阶过程；左右轮跨不同台阶的高度差不能直接解释成真实横坡。旧文档 `v135_three_wheel_ascent_patch_20260921.md` 是旧运行版本事实，其“B1用progress argmax”和“横坡拟合”已被本补丁替代。

## 接线

- `v135_terrain_geometry.py`：沿用安装版Isaac curriculum column mapping；只拆分原6个ascent列，踏深按cfg的 `ascent_tread_widths`。默认30/30/25/35/40/45cm，其他列及其比例、下楼30cm不变。阶高仍按原difficulty分布生成。每个flight等踏深；不声称支持同一flight内非均匀踏深。
- `v135_ascent_contract.py`、`v135_riser_identity.py`：每环境真实几何决定B1半径、face宽度、target推进、skip resync、P1/P2匹配、first-swing范围与flight结束；保留唯一ledger。阶数是每环境tensor，最大阶数仅用于完整输出metric keys。
- `events.py`：V135 ascent reset按真实B1半径布置；非V135及非上楼路径保留。`commands.py`仅补齐V135只读几何指标，command采样不变。
- `v135_final_ascent_reward.py`、`v135_forward_credit.py`：potential落脚域及有限前进预算跟随真实踏深；不再使用固定30cm。25cm时旧radius+margin落脚窗口为空，margin现限制在轮子占用之外可用踏面的半宽，30cm及以上保留旧边界。
- `v135_ascent_support.py`：B1使用已有first-swing身份并结合真实离地证据；B2+也以实际单前腿摆动判定，不拿expected-side冒充实际摆腿。五帧准备证据、摆动、真实upper-tread支撑接受分别统计，不增加第二套阶序FSM。三个阶段质量取最小，落脚阶段只统计连续确认窗口；等待不能稀释差的摆动/准备证据。双前腿离地是坏支撑样本，不能当作“不属于单腿摆动”而漏计。
- 支撑质量使用惯性COM、实际接触点构成的XY支撑凸包、载荷、工作空间、滚动接触滑移、姿态/角速度。只在上楼假定生成台阶无横坡，保留自然纵向pitch。质心投影是稳定性代理，不是完整动态可行性或wrench锥证明。
- `v135_valid_riser_completion_reward`在原合法事件结算既有信用：原event rate乘 `0.5+0.5*settlement_quality`。配置权重不变，差承重仍保留一半strict completion；native velocity、forward、landing信用不改。没有原地奖励、重复落脚奖励、额外长期bonus。原有有限前进bonus继续读取当前摆动/落脚支撑质量。
- `v135_behavior_metrics.py`、`v135_training_report.py`：分踏深记录完整严格链、合法/交替/skip、B1滚转、入阶分阶段帧数、质心投影支撑区外比例、completion承重质量；显式分母。manifest包含踏深分布及events/commands源码hash。报告不把missing数据当零。

## 验证

- 27个相关Python模块compile通过。
- 8份V135 CPU回归脚本全部通过，包括混合踏深同batch、四方向、短踏面支撑窗口、B2不能冒充B1、first swing、有限前进信用、report计数。
- 实际加载当前安装版 `.../m20_wzh/IsaacLab/.../trimesh/mesh_terrains.py`，生成五种踏深mesh，独立采样mesh表面跃迁；物理边线与ledger逐一一致。不是用同一个layout公式生成测试真值。
- 支撑state测试：无摆动不默认FL、真实左右对称、质心外移、退化支撑线、同step重复调用、双前腿离地、落脚后等待不洗掉坏质量、换target清空信用。
- 最终runtime日志 `/tmp/v135_tread_entry_verified_smoke.log`：16env×8step、24-step rollout及一次真实MLP PPO update均PASS；Actor63/Critic574/Action16、MLP和PPO合同不变；真实列分布接线PASS。
- 下楼/平地五个模块与备份逐字节一致：whole_body_descent、descent_support_transfer、descent_stance、velocity_modulation、exit_geometry。
- 测试前资源实查：GPU总49140MiB、占9829MiB，内存可用96GiB。沿用已有16env小烟测规模，未启动8192env长训。
- `git diff --check`通过。pyflakes未安装，未执行该项。
- multi-round-review独立Codex reviewer启动失败（usage limit），未完成独立复审，不冒充多审通过；本轮完成自审、实际回归与runtime测试。

## 运行状态与未验证项

2026-09-22实查旧服务 `m20-v135-three-support-34200-plus30000.service` 是inactive/dead，Result=success，ExecMainStatus=0；日志止于43315/64200，最新checkpoint43300（09-21 21:10）。本轮没有stop/restart，不知道为何旧任务提前结束，不能声称到达64200或仍在训练。

本补丁未启动新训练，也未证明新策略在大踏深上改善。下一次启用前/后应在同checkpoint、固定seed、rise/tread/command交叉组合下做冻结对照，逐阶核验完整链及B1承重，并回归下楼和平地。当前训练mesh覆盖5–8级，不代表所有级数或任意非规则楼梯泛化。共享policy即使下楼代码不变，也可能在续训后退化，必须用行为回放验收。

## 用户授权续训已启动（2026-09-22 11:36）

从上轮model_43300.pt恢复actor/critic/optimizer，追加30000轮，目标73300；8192×24。服务m20-v135-tread-entry-43300-plus30000，PID474968。实际日志43300→43302连续更新，V135_RESUME_CONTROL_STEP=1039200；新manifest源码哈希一致，踏深30/30/25/35/40/45cm，Actor63/Critic574/Action16核对通过。

日志目录：`logs/rsl_rl/M20-V13.5-Final-FromScratch/2026-09-22_11-36-22_v135_tread_entry_resume43300_plus30000_e8192_r24`。资源账、checkpoint hash和启动脚本见artifacts/systemd/v135_tread_entry_resume43300_launch.json；未动同机Arcus训练。这里只确认训练启动，尚无新策略行为结论。
