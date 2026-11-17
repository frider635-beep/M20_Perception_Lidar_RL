# V11.1 后腿支撑转移与楼梯出口恢复

任务 ID 保持 `M20-Stair-V11.1-V11FD-RearPosture-v0`。从 V11-FD 原生 `model_29999.pt` 恢复 Actor、Critic、优化器；V11.1 的旧 `model_5800.pt` 未被选用。当前续训保留 V11 的 bilateral_same_riser 上楼与 63D signed-edge Actor，不接任何 V13/V13.5 交替奖励或 strict riser ledger。

## 接线

- V11.1 配置直接接 V13.5 的 rear support transfer、whole-body support quality、recovery event 和 post-stair flat progress/completion/crawl。
- 关闭 V11 原生 rear follow/delay/roll/completion，以及 V11.1 旧后伸、膝折、rear-roll、出口高度惩罚，避免重复目标。
- 速度 5.0、世界前进 2.0、速度不足 3.0 继续使用原权重；只在下楼支撑转移时使用 V13.5 门控。上楼不乘交替因子。
- V11.1 的下楼门控固定最终基线，防止跨任务 resume 重启 V13.5 早期退火。出楼梯奖励融合沿用 10 控制步。
- 新指标为 `v111_transfer_*` 与 `v111_flat_*`。旧 `v111_*` 姿态指标禁用。

## 验证

`test_m20_v111_v135_descent_runtime.py`：16 环境，实际 Isaac 环境，V11 原生 checkpoint 严格加载并恢复 optimizer，24 步 rollout 和一次 PPO update 通过。`test_m20_v11_fd_contract.py`、`test_m20_v111_rear_posture.py` 的纯函数回归、Python 编译与 `git diff --check` 通过。旧 `test_m20_v111_runtime_smoke.py` 检查的是已弃用的 V11.1 相对旧基线差异，未作为当前配置验收；活动 launcher 已接新运行检查。

## 训练

服务 `m20-v111-v135-descent-resume29999-v2.service`，日志 `artifacts/systemd/v111_v135_descent_resume29999_v2.log`。从 29999 开始继续 30000 轮，8192 环境 ×24 rollout，训练已实际推进至 30022，并保存 `model_30000.pt`。首次启动因跨任务 checkpoint 不在 V11.1 日志根下失败；通过只读符号链接将原 V11-FD run 暴露给 RSL 训练入口后成功，原 checkpoint 未移动或覆盖。

资源启动前：整卡 49140 MiB/已用 9683 MiB，RAM 可用约 92 GiB，磁盘可用 245 GiB；服务 MemoryMax=72G。依据同机相同 8192×24 V13.5 长训约 8 GiB 新增显存、约 7 GiB 主机内存。

尚未证明新策略的下楼/平地闭环行为优于 V11-FD，需后续固定 checkpoint sim2sim 对比，特别看 rear load、四轮支撑、pitch 和出口恢复。
