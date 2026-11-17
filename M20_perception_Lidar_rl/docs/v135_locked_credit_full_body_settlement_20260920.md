# V13.5 锁定目标信用与全身下楼结算修复（2026-09-20）

任务保持 `M20-Stair-V13.5-V11FD-NativeAlternation-v0`。本次没有改变 reward 权重、Actor/Critic/Action ABI、PPO 或训练分布，也没有启动长训。

## 现象、真因、判据、修法

1. 上楼时 V11 scanner 先转到 B(k+1)，而 strict ledger 仍在等待 Bk 稳定支撑。旧 dense progress 同时读取 scanner 的 `target_riser_x`、clearance 与 `target_match`，因此对 Bk 的信用在目标尚未完成时归零。现改为只在首次匹配时锁定 Bk 下踏面，随后以 strict ledger 的物理 `target_world_xy`、方向、upper height、crossing 与 support 证据计算 potential。只有 strict target id 改变才清历史最佳；scanner 换槽不清。抬腿、crossing 和 support 信用均限制在 Bk 的踏面范围内，未来台阶不能冒领。
2. 下楼旧 `completion_now` 只检查两后轮稳定支撑和历史 `front_complete`，前轮当前卸载或机身大俯仰时仍可结算，而且卸载后重新接触会重复触发。现要求目标仍有效、两后轮下踏面稳定、两前轮当前承重、机身俯仰在现有 recovery 容差内，连续现有 `stable_frames` 帧后一次性锁存 completion。recovery 与完成后的速度质量也要求当前前轮承重；离开下降目标后不继续压低平地速度收益。

## 已执行验收

- 新的锁定目标/全身结算契约测试：先 RED，修复后 PASS。
- V13 与 V13.5 原有物理 identity、native alternation 合同测试：PASS。
- Python compile、`git diff --check`：PASS。
- V13.5 reference audit：PASS。
- 16 env × 8 step 运行时冒烟加 1 次 recurrent/MLP 适配的实际 PPO update：PASS；Actor 63D、Critic 574D、Action 16D，rollout 24。

这些测试证明代码接线与单次更新能运行，**不证明**目标 gait 在继续训练后会出现或保持。旧 17100 轮 checkpoint 的行为不会因 reward 代码改变而自动变好；重启长训前仍应做固定 checkpoint 的事件窗口复核。
