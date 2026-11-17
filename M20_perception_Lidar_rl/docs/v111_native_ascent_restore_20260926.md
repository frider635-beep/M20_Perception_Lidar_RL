# V11.1 恢复原 V11-FD 正常上楼（2026-09-26）

## 现象与纠正
用户明确：以 vv11flat 为基准，其正常上楼自然；vv11.1 应仅修改下楼和入口切换。此前增加的整段上楼顺序与周期约束不再符合当前目标。

## 真因及修法
关闭 timed_support_cycle、same_tread_handoff、adjacent_landing_credit；撤下周期奖励、替换完成奖励、周期监控及全程平地姿态奖励；删除速度包装器的 cycle_drive。发现额外速度参数漂移：现改为继承父配置参数并调用项目原 track_lin_vel_xy_exp。
保留 physical_first_tread，仅第一级入口的几何锁定；完成后原 bilateral 路径继续。关闭handoff后入口完成须读原candidate_ready，跳级失败不能访问未创建的adjacent ledger，已修。
保留 V13.5 下楼支撑移植；此前明确要求的出口恢复暂保留，已向用户询问最新范围是否要进一步删除。因而不能声称所有差异严格只有下楼和入口：出口恢复是明确列出的暂留项。

## 判据与实际执行
基准是 logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch/params/env.yaml，区别于后来发生漂移的当前V11-FD源码默认值。
通过：actions/commands/observations/events/terminations、地形和原perception参数对照；除显式下楼替换外，所有原reward参数一致；入口结束后的同一上楼轨迹，逐帧landing tensor与不带V111配置的原实现一致；真实环境下楼外速度包装器与原函数数值一致；原model_29999的actor/critic/optimizer恢复、实际PPO更新通过。日志 artifacts/v111_native_ascent_20260926/smoke_final.log。
旧周期训练已正常结束；模型已经学到的偏差不会因奖励回退自动消失。因此恢复原V11-FD checkpoint，而非继续使用周期训练权重。新服务 m20-v111-native-ascent-resume29999.service；脚本、源码快照和资源记录在同一artifacts目录。
尚未验证：新训练后视觉步态与sim2sim。未重复异构复审：这是按用户要求撤回先前改动，采用原基准直接对照。pyflakes缺失仍跳过。
