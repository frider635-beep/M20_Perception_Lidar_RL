# 交流维护日志

## 2026-09-11 初始化

问题：围绕 `kunyu_ppo` 以及本项目 V11/V12 任务做面试用项目梳理，并新建资料文件夹。

本轮结论：

- 新增 `interview_prep/` 作为面试梳理工作区。
- 当前叙事主线采用 V10 -> V11 -> V11-FD -> V12 -> V13/V13.2 的演进关系。
- V10 定位为 first-riser perception handoff 和 reward balance 修正。
- V11 定位为两立面连续感知、63D actor、FL/FR 对称责任轮状态机。
- V12 定位为 V9 teacher 风格 MoE/dense reward/control arm，但 actor 仍受 V11 63D 可部署接口约束。

证据：

- `README_KUNYU_V10.md`
- `V11_repo.md`
- `README_V11_FD.md`
- `source/rl_training/rl_training/v11_continuous_runner.py`
- `source/rl_training/rl_training/v12_teacher_runner.py`
- `source/rl_training/rl_training/v12_teacher_moe.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v11_continuous_env_cfg.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v12_teacher_env_cfg.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/perceptive_spatial.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_landing.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v11_continuous.py`
- `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v12_teacher_rewards.py`

未验证边界：

- 本轮没有启动 Isaac Lab 训练或仿真。
- 本轮没有做 sim2sim 或真机回放。
- hub 在线职责状态因当前沙箱网络限制未能读取；不能据此断言无人并行做同主题。

下一步可维护内容：

- 继续补充你个人在项目中的具体贡献边界：你亲手做了哪些配置、reward、runner、训练、评估、部署工作。
- 把真实训练结果、TensorBoard 指标、模型 checkpoint 选择逻辑补到路线图。
- 为常见面试追问补“代码级回答 + 可验证证据”。
