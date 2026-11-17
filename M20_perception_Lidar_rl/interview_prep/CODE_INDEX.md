# 代码证据索引

> 这里记录“面试结论对应哪段代码”。后续改叙事时，优先更新本文件。

## 总入口

| 主题 | 文件/位置 | 证据 |
|---|---|---|
| V10 项目背景 | `README_KUNYU_V10.md:1` | M20 V10 感知楼梯 PPO 版本说明 |
| V10 actor/action | `README_KUNYU_V10.md:11` | Actor observation 60D，Action 16D |
| V10 reward audit 边界 | `README_KUNYU_V10.md:54`、`:114` | reward balance audit 使用固定早期模型；不能宣称 FULL 已解决 |
| V11 任务报告 | `V11_repo.md:1` | V11 仓库实现与训练报告 |
| V11 task id | `V11_repo.md:5`、`scripts/train_m20_v11_from_scratch.sh:14` | `M20-Stair-Continuous-Landing-V11-v0` |
| V11 代码职责 | `V11_repo.md:33` | V11 关键文件表 |
| V11-FD contract | `README_V11_FD.md:8` | 63D actor、signed edge、上下台阶合同 |
| V13 contract | `docs/v13_v11fd_alternation_contract.md:7` | 保留 V11-FD 63D/16D，只改 completion unit 与软交替偏好 |
| V13.2 contract | `docs/v132_v11fd_barrier_alternation_contract.md:42` | 68D actor，后 5D 是可部署有限状态 |

## 任务注册

| 任务 | 文件/位置 |
|---|---|
| V11 | `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py:88` |
| V11-FD | `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py:108` |
| V13/V13.1/V13.2 | `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py:118`、`:128`、`:148` |
| V12 | `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py:158` |

## V11 感知与状态机

| 结论 | 文件/位置 |
|---|---|
| V11 perception 参数台账 | `source/.../v11_continuous_env_cfg.py:19` |
| 默认 `front_completion_mode = alternating_risers` | `source/.../v11_continuous_env_cfg.py:22` |
| 默认 `feature_contract = v11_signed_edges_v2` | `source/.../v11_continuous_env_cfg.py:35` |
| V11 rewards 断开 V10 FR 专项项 | `source/.../v11_continuous_env_cfg.py:442` |
| V11 actor observation 替换为 `v11_actor_perception` | `source/.../v11_continuous_env_cfg.py:426` |
| V11 critic 使用 PGTT/geometry，teacher 置空 | `source/.../v11_continuous_env_cfg.py:431`、`:435`、`:439` |
| 两立面提取函数 | `source/.../mdp/perceptive_spatial.py:277` |
| V11 landing 纯几何函数 | `source/.../mdp/v11_landing.py:93`、`:173` |
| V11 simulator-only landing state | `source/.../mdp/v11_continuous.py:36` |
| V11 state 每步从 `extract_two_risers_points` 取目标 | `source/.../mdp/v11_continuous.py:1279` |
| V11 actor 6D perception | `source/.../mdp/v11_continuous.py:1400` |
| V11 landing event reward | `source/.../mdp/v11_continuous.py:1412` |
| front-riser collision termination | `source/.../mdp/v11_continuous.py:1591` |

## V11 Runner/ABI

| 结论 | 文件/位置 |
|---|---|
| RSL-RL 5.x bridge | `source/rl_training/rl_training/v11_continuous_runner.py:11` |
| 拒绝隐藏 depth-image actor input | `source/rl_training/rl_training/v11_continuous_runner.py:42` |
| actor/action 维度 gate | `source/rl_training/rl_training/v11_continuous_runner.py:117` |
| actor manifest | `source/rl_training/rl_training/v11_continuous_runner.py:126` |
| V11 PPO cfg runner class | `source/.../agents/rsl_rl_ppo_cfg.py:137` |

## V12 Teacher

| 结论 | 文件/位置 |
|---|---|
| V12 rewards 继承 V11 reward cfg | `source/.../v12_teacher_env_cfg.py:16` |
| V12 env 继承 V11 env cfg | `source/.../v12_teacher_env_cfg.py:51` |
| 固定 `bilateral_same_riser` | `source/.../v12_teacher_env_cfg.py:103` |
| 固定 positive-riser contract | `source/.../v12_teacher_env_cfg.py:105` |
| V9-style terrain distribution | `source/.../v12_teacher_env_cfg.py:147` |
| V9-style command/reset/domain randomization | `source/.../v12_teacher_env_cfg.py:164` |
| V12 map-based stair rewards | `source/.../v12_teacher_env_cfg.py:311` |
| V12 terrain curriculum | `source/.../v12_teacher_env_cfg.py:354` |
| V12 reward 函数说明 PGTT 不进 actor | `source/.../mdp/v12_teacher_rewards.py:1` |
| 4-expert soft MoE | `source/rl_training/rl_training/v12_teacher_moe.py:12`、`:52` |
| V12 runner ABI/MoE gate | `source/rl_training/rl_training/v12_teacher_runner.py:55`、`:107` |
| V12 从零训练脚本 | `scripts/train_m20_v12_from_scratch.sh:15` |
| V12 静态对齐测试 | `scripts/tools/check_m20_v12_v9_teacher_alignment.py:1` |

## 可运行的资料区脚本

```bash
python interview_prep/necessary_code/trace_contracts.py
```

它不导入 Isaac Lab，只做文本/正则扫描，适合在没有仿真环境时快速确认叙事依赖的文件和字符串还在。
