# V13.5 训练至 17100 轮的失败归因

任务仍是 `M20-Stair-V13.5-V11FD-NativeAlternation-v0`。本次从零训练为 8192 env × 24 rollout，run 为 `2026-09-19_22-52-38_v135_locked_settlement_e8192_r24_from_scratch_30k_20260919`。2026-09-20 约 17100 轮后，因目标行为未形成且下楼支撑恶化，已停止 `m20-v135-locked-settlement-20260919.service`。最新保留 `model_17100.pt`；没有重启训练。

## 时间序列：任务回报上升，目标行为下降

以下是该 run 的 TensorBoard scalar，数值是对应记录点，不代表逐 episode 成功率；特别是 `v135_riser_skip_rate` 使用项目自身分母，不能直接解释为“10% episode 跳级”。

| 指标 | 约 1000 轮 | 约 5000 轮 | 约 17000 轮 |
|---|---:|---:|---:|
| 速度跟踪奖励 | 1.715 | 3.704 | 3.955 |
| strict target id 均值 | 1.000 | 1.033 | 1.010 |
| ordered alternation 指标 | 0 | 0 | 0.000786 |
| V11 前轮 crossing 指标 | 0.0696 | 0.1395 | 0.1457 |
| V11 前轮 support 指标 | 0.00019 | 0.00062 | 0.00175 |
| V11 physical completion 指标 | 0 | 0 | 0 |
| rear-transfer completion reward | 0.188 | 0.236 | 0.496 |
| descent support quality | 0.653 | 0.551 | 0.522 |
| descent 四轮支撑 | 0.518 | 0.421 | 0.320 |
| descent recovery 俯仰绝对值（rad） | 0.020 | 0.070 | 0.442 |
| descent rear load share | 0.427 | 0.516 | 0.672 |
| descent body–rear lag（m） | -0.130 | -0.084 | 0.089 |

速度和后轮“完成”收益提高，没有伴随逐级交替或健康全身支撑。rear load share 增加而四轮支撑下降，更像前轮卸载、后部独自承重，不能把 rear completion 当作自然下楼的证据。

## 冻结策略同条件对照

只读 `frozen_probe.py`，固定种子 13542、同一任务代码、128 env × 500 step，对 `model_5000.pt` 和 `model_17000.pt` 分别回放；不调用 optimizer。回放环境分布相同，但两 checkpoint 的轨迹不同，以下计数是行为对照，不是因果 PPO-gradient 证明。

| 事件/条件 | 5000 | 17000 |
|---|---:|---:|
| V11 前轮 crossing | 734 | 723 |
| crossing 指向下一物理阶 | 495 | 419 |
| crossing 指向 strict 当前阶 | 74 | 77 |
| 安全跨 B1 且同轮稳定 B1 支撑的代理事件 | 18 | 0 |
| strict valid completion | 4 | 0 |
| strict ordered completion | 0 | 0 |
| 下楼 active/recovery 窗口帧 | 3567 | 5427 |
| 后腿长度比 <0.85 的窗口帧 | 296（8.3%） | 776（14.3%） |
| 四轮支撑窗口占比 | 53.4% | 46.1% |
| 后轮载荷占比窗口均值 | 56.3% | 68.3% |
| 机身–后轮前进差窗口均值 | -0.085 m | +0.103 m |

早期代理物理事件多于 strict valid，说明 detector 仍值得逐帧核查；但后期同条件下连代理 B1 安全支撑也消失，不能把 17000 轮没有交替全部归罪于 detector。代理探针较宽，不能用其 18 次直接代替 strict 验收。

## 代码可证的收益漏洞

1. 上楼主速度奖励仍有 5.0 原生权重；strict speed multiplier 只在 live target 可行动且与 scanner 匹配后起效，adjacent/cycle 又必须先有严格完成。当前 strict B1→B2 几乎没有访问，因而声称“交替更赚钱”的那部分机制在实际 rollout 中几乎没有支付。尚无 matched raw-return/advantage 和 gradient 数据，不能据此断言 `RETURN_ORDERING` 已被证明。
2. 下楼 `completion_now` 只要求两后轮在 lower tread 稳定支撑，以及历史 `front_complete` latch；它**不要求当前前轮仍承重、四轮共同支撑、机身俯仰合理或后腿回到中性长度**。因此一次 rear completion 可以在全身姿态恶化时兑现。
3. completion 后 `recovery_armed` 的速度系数为 `0.70 + 0.30 * support_quality`。即使全身支撑质量差，仍至少保留 70% 速度收益；`support_quality` 只看后轮支撑和后腿长度，不看前轮载荷或俯仰。原生 forward/velocity 项继续吸引机身向前，四轮支撑下降与这一漏洞相容。它是代码层面的可能机制，不是未经匹配实验的唯一因果解释。

## 判定与下一步

本次是 **NO-GO 继续长训**。训练已停，保留 checkpoint 和事件日志。下一次修改前先用现有 checkpoint 做 16/32-step 事件窗口，逐帧核对同一物理阶的前后轮位置、四轮力、base pitch、接触高度和每项实际 reward；将“后轮碰到 lower tread”与“全身完成健康支撑转移”分开验收。上楼同时核查 B1 代理物理支撑与 strict 账本不一致的样本。未完成这些对照前，不加新 reward weight、不重启 30000 轮长训。
