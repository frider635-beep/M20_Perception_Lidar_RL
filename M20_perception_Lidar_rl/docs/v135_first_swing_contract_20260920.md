# V13.5 首次摆腿与严格逐级链修改验收 — 2026-09-20

## 范围
直接修改现有 M20-Stair-V13.5-V11FD-NativeAlternation-v0。下楼、平地恢复、Actor native 63D MLP、Critic 574D 结构、16D action、8192×24 PPO 设置和所有 reward 配置权重保持原样。Critic 的 potential 数据对齐修复不改变维度。

## 修复
- B1 身份根据当前实际 inverted_pyramid_stairs_terrain 的几何边界与环境 origin 校验；扫描器最先看到 B2 不再将其编号为 B1。目标沿楼梯面法向推进，不沿带初始 yaw 偏差的机身方向外推。
- FL/FR 任意腿首先离地且超过有效抬腿阈值，就锁定第一次尝试；纯垂直空抬同样计入。空抬返回、跳级落地、双腿同时首摆都会永久取消本 episode 完整严格链资格。
- B1 合法完成仍要求物理 crossing、目标踏面高度、接触与连续稳定支撑。原生 landing_event 本身不能绕过确认。
- 连续 shaping 的高度信用必须同时有接近目标与向前摆动。远处抬高、原地抬高不能单独赚 clearance 信用；历史最好增量仍防往复刷分。
- 完成帧先快照旧目标和相位，结算 Bk 奖励，再使用已推进的 B(k+1)。Critic 若已切到新 target，不混入旧 target potential。
- 完整成功统计按实际生成的楼梯级数，不以 B3 为终点。错误后允许局部任务重同步，但不恢复严格 episode 成功。

## 新增观测指标
Diag/V135Behavior/Ascent 下记录 first_lift_event、first_lift_b1_event、first_lift_empty_event、first_lift_skip_event、b1_identity_rejected、native_unconfirmed_event、full_chain_episodes、strict_prefix_sum、flight_length_sum。
训练报告增加首摆 B1 成功/空抬/跳级比例、B1 后 B2 条件比例、完整严格链比例。
首摆事件与落地事件可能跨 rollout/统计窗口，短窗口比值不能直接视为无偏成功率；优先看充分长窗口与 episode 完整链计数。

## 真实执行结果
- 改动 Python 文件 py_compile：PASS。
- 全部六个 test_m20_v135_* 非 runtime 脚本：PASS。
- 首摆新测试：FL/FR 对称、多种楼梯长度、B2 冒充 B1、纯空抬、双腿首摆、跳级、episode reset、旧目标奖励快照：PASS。
- Isaac 16 env × 8 step + 一次真实 MLP PPO update：PASS。
  最终日志 /tmp/v135_first_swing_runtime_final.log 含 V135_NATIVE_MLP_PPO_UPDATE: PASS 和 V135_RUNTIME_SMOKE: PASS。
- 下楼/平地/Actor 等冻结文件对照本轮前 SHA256：一致。
- 当前长训服务仍 active，MainPID=2608537，与修改前相同；本轮未停止、覆盖或重启它。

## 限制与下一次验收
本轮是代码和接线验收，未用新 reward 训练新策略，未证明已有 checkpoint 首次抬腿必定落 B1。当前运行的长训仍使用启动时载入的旧实现。
B1 几何校验针对已核对的无 holes inverted pyramid 生成器；变更 terrain 生成器需要重新验证，不默认支持任意楼梯网格。
保持相同 seeds/terrain/commands 回放对照，必须同时确认首摆 B1、B1→B2、完整链提升，以及下楼和到平地行为不退化。共享策略续训可能改变下楼行为，即使冻结文件不变，仍需行为回归。

## 现象 / 真因 / 判据 / 修法
现象：可抬腿但首步可能空抬或越过 B1，且仅看后续 completion 容易掩盖首次失败。
代码层原因：首可见 edge 作为 B1、完成与推进在同帧改变 reward 参照、没有不可覆盖的首次摆腿证据。此处是代码漏洞结论，不将历史全部训练退化归因于它。
判据：真实物理 B1、首次接触落点和稳定支撑、未中断的逐级异侧 prefix。
修法：物理边界校验、首摆证据锁定、旧目标快照结算和全楼梯长度指标。

## 本轮改动文件
- `source/rl_training/rl_training/v135_training_report.py`
- `source/rl_training/rl_training/v135_behavior_metrics.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v135_final_critic.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v135_forward_credit.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v135_final_ascent_reward.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v135_riser_identity.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v135_v11fd_native_alternation_env_cfg.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v135_ascent_contract.py`
- `scripts/tools/test_m20_v135_first_swing.py`
