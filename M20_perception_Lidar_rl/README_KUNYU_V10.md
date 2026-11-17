# 1. M20 V10 感知楼梯 PPO 版本说明

本仓库是从训练机 `192.168.1.244:/home/ubuntu/m20_perceptive_final` 直接导出的可复现代码快照，供后续优化使用。

## 1.1. 快照信息

- 上游基线提交：`a31e9cea991cb9b82c75d5f51cd874832df41388`
- 244 工作分支：`final/no-phase-perceptive-stair`
- 导出日期：2026-08-19
- 训练任务：`M20-Stair-Unified-Perceptive-v0`
- Actor observation：60D
- Action：16D（12 个腿关节位置 + 4 个轮速）
- 正式训练要求：从 0 开始，不加载旧 checkpoint

仓库保留了基线源码，并用 244 当前的 `source/rl_training/` 和 `scripts/` 覆盖。没有包含 checkpoint、TensorBoard event、训练日志、MuJoCo 输出、`analysis/` 临时目录或 Python 缓存。唯一保留的分析结果是本次 reward audit 的四个小型账单文件，位于 `docs/v10_reward_audit_symmetric_miss_seed42/`。

## 1.2. 本轮要解决的问题

旧 V10 在 FIRST 附近的专项 reward 远大于官方 locomotion reward：

- 旧 FIRST 单次约 `+8.2485`
- 旧 MISS 单次约 `-12.3728`

FIRST 降低 10 倍后，如果 MISS 不同步降低，一次 MISS 会等于约 15 次 FIRST，并接替 FIRST 成为主要高方差信号。因此当前版本把 FIRST/MISS 改成幅值相同、符号相反的教学脉冲，并在 FIRST 后额外开启一份实时 command 的线速度跟踪。

目标语义是：感知只负责教会 FIRST；FIRST 完成后，重新让官方 command-following 主导策略。

## 1.3. 当前 reward 配置

控制周期为 `0.02 s`，RewardManager 会再乘一次 `dt`。

| Reward | 配置权重 | 实际语义 |
|---|---:|---|
| `first_step_success` | `+41.2426414` | 单次 `+0.824852828` |
| `perception_miss` | `-41.2426414` | 单次 `-0.824852828` |
| `fr_clearance_progress` | `+2.0` | 保持原定义 |
| `joint_success` | `+1649.705658` | 单次约 `+32.9941`，保持原定义 |
| 官方 `track_lin_vel_xy_exp` | `+5.0` | 完全不变 |
| `post_first_track_lin_vel_xy_exp` | `+5.0` | FIRST 前关闭，FIRST 后开启 |

额外 post-FIRST tracking：

- 跟随实时 command `[vx, vy]`；
- 不依赖 `b`；
- 不依赖 first-riser rearm；
- 不使用 frozen `target_dir`；
- FIRST 后以 episode latch 保持开启；
- 只在 episode reset 清零。

本版本没有加入 stair-progress reward、倒车惩罚或 command gate，也没有修改 command sampler、Actor、PPO、官方 Rough reward、terrain distribution/curriculum、Action Manager、DelayedPD 或推理频率。

## 1.4. Reward audit 结果

审计条件：4096 env、984 control steps、PPO horizon 24、seed 42。使用旧训练早期的 `model_2000.pt` 作为固定策略，仅用于观察新 reward 账单；该模型不是当前 reward 下重新训练得到的结果。

```text
REWARD_BALANCE_AUDIT=PASS
POST_FIRST_RUNTIME_CONTRACT=PASS
```

关键账单：

| 分组 | 累计 reward | sample variance | term absolute-mass share |
|---|---:|---:|---:|
| Official total | `47633.2262` | `0.0106423` | `45.2463%` |
| FIRST（18 次） | `+14.8474` | `0.0000152` | `0.0107%` |
| MISS（121 次） | `-99.8072` | `0.0001021` | `0.0722%` |
| Clearance | `+271.5506` | `0.0001859` | `0.1965%` |
| Post-FIRST velocity | `+293.8245` | `0.0000276` | `0.2126%` |
| FULL（1 次） | `+32.9941` | `0.0013508` | `0.0239%` |
| V10-added total | `+513.4094` | `0.0017547` | `0.5146%` |

MISS variance / V10-added variance 从旧账单的 `93.38%` 降到约 `5.82%`。官方 reward 的 signed-return share 约为 `98.93%`，已经重新成为主导信号。

完整审计文件：

- `docs/v10_reward_audit_symmetric_miss_seed42/summary.json`
- `docs/v10_reward_audit_symmetric_miss_seed42/term_stats.csv`
- `docs/v10_reward_audit_symmetric_miss_seed42/ppo_batch_stats.csv`
- `docs/v10_reward_audit_symmetric_miss_seed42/episode_stats.csv`

## 1.5. 关键源码

- 环境和 reward：`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v10_unified_env_cfg.py`
- FIRST lifecycle、事件、统计：`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v10_unified.py`
- 60D Actor：`source/rl_training/rl_training/v10_unified_policy.py`
- PPO runner 与 TensorBoard 指标：`source/rl_training/rl_training/v10_unified_runner.py`
- reward audit：`scripts/tools/audit_m20_v10_reward_balance.py`

## 1.6. 从 0 开始训练

PPO 配置类仍保留早期 pilot 默认值 `max_iterations=2000`。正式 60k 实验请显式覆盖，避免误把 pilot 默认值当正式长度：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task M20-Stair-Unified-Perceptive-v0 \
  --num_envs 4096 \
  --max_iterations 60000 \
  --headless \
  --run_name v10_first_miss_symmetric_postxy_from_scratch_60k
```

不要添加 `--resume`，也不要加载此前 `model_2250.pt`。该旧模型是在 MISS 降权之前训练的，仅保留在 244 上做历史比较。

## 1.7. 后续重点观察

- FIRST / MISS 次数、累计 reward 和方差；
- `post_first_track_lin_vel_xy_exp`；
- FIRST 后沿实时 command 的速度、反向步数和强倒退比例；
- 14–17 cm、17–20 cm 的 FIRST 与 Joint/FULL 成功率；
- Stair terrain level、promotion rate；
- 转向问题需要独立验证，不能把它与本次 reward balance 实验混为一谈。

目前只能确认 reward balance 已恢复，不能宣称 FIRST 后倒退或 17 cm FULL 已经解决；这些必须由新的从零训练验证。
