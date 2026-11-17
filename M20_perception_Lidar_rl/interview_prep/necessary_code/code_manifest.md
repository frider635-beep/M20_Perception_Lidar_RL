# 必要代码清单

这些不是复制出的第二份权威实现；权威代码仍在项目原路径。本目录只维护面试梳理所需的最小代码索引和扫描脚本。

## 核心源文件

- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py`
  - Gym task registration：V11、V11-FD、V12、V13 系列入口。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_continuous_env_cfg.py`
  - V11 perception config、reward config、observation/critic override、V10 FR terms disconnect。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/perceptive_spatial.py`
  - 高程 ray-hit 点到立面几何的提取，重点是 `extract_two_risers_points`。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_landing.py`
  - 可单测的纯几何函数：landing region、responsibility、alternation、softalt physical gate。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py`
  - V11 simulator-side state machine 和 reward/termination/curriculum metrics。
- `source/rl_training/rl_training/v11_continuous_runner.py`
  - RSL-RL 5 bridge，63D/16D ABI gate，actor manifest。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v12_teacher_env_cfg.py`
  - V12 teacher reward、terrain、reset、curriculum、positive-riser contract。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v12_teacher_rewards.py`
  - V9-style dense reward port，PGTT 只在 reward/critic 侧使用。
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v12_teacher_curriculum.py`
  - V12 teacher terrain promotion rule。
- `source/rl_training/rl_training/v12_teacher_runner.py`
  - V12 MoE runner、robust logger、ABI/MoE gate。
- `source/rl_training/rl_training/v12_teacher_moe.py`
  - 4-expert soft MoE model implementation。

## 训练与测试脚本

- `scripts/train_m20_v11_from_scratch.sh`
- `scripts/train_m20_v12_from_scratch.sh`
- `scripts/tools/test_m20_v11_fd_contract.py`
- `scripts/tools/test_m20_v12_bilateral_stair_rewards.py`
- `scripts/tools/check_m20_v12_v9_teacher_alignment.py`

## 本目录自带工具

- `trace_contracts.py`
  - 不导入 Isaac Lab。
  - 扫描任务注册、runner gate、feature contract、reward/critic actor 边界。
  - 作为路线图维护前的轻量 sanity check。
