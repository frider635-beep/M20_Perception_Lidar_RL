# M20 Perception LiDAR RL

M20 感知运动训练代码与独立 MuJoCo 验证程序。原始基线和冻结参考仍是 **V11.1 `model_35300.pt`**，真实训练配置在 `interview_prep/params/`。本次另发布用户指定的 Recovery 候选 **`model_36500.pt`**，其真实配置、哈希和评测记录放在 `data/m20_recovery/published_36500/`。

```text
M20_perception_Lidar_rl/       训练工程、任务定义、奖励、工具与历史设计文档
  interview_prep/             原始 35300、params、来源及 SHA-256
  data/m20_recovery/          标签回放、原模型基线、36500 配置与评测
  logs/rsl_rl/                仅发布用户指定的 model_36500.pt
  docs/                      改进分析与迁移说明
  deep_robotics_model/       官方机器人资源子模块
mujoco_validation/           验证程序、M20 资源、楼梯和平地地形
PUBLISH_MANIFEST.json         首次迁移快照的历史清单
PUBLISH_UPDATE_20261117.json  本次更新文件与权重校验记录
```

`logs` 中只上传 `M20-V11.1-AscentRecovery/2026-11-17_14-59-02_063111_guarded_30000_from35300/model_36500.pt`，其他权重、TensorBoard 事件、运行日志、录像和临时输出继续过滤；本地历史文件保留。训练目录内旧版 README 和 V13.* 报告是历史资料，最新问答及要求见 [本次交接记录](CODEX_SESSION_HANDOFF_20261117.md)。`run_mujoco_v135.sh` 沿用原文件名，默认仍加载 `35300`。

## 克隆与 MuJoCo 验证

```bash
git clone --recurse-submodules https://github.com/frider635-beep/M20_Perception_Lidar_RL.git
cd M20_Perception_Lidar_RL
python3 -m venv .venv-mujoco
source .venv-mujoco/bin/activate
python -m pip install -r mujoco_validation/deploy/mujoco_v135/requirements.txt
cd mujoco_validation
./run_mujoco_v135.sh --viewer --keyboard \
  --checkpoint ../M20_perception_Lidar_rl/interview_prep/model_35300.pt \
  --stair-height 0.16 --tread-depth 0.26
```

也可使用已安装这些依赖的环境。脚本默认使用当前环境的 `python3`；通过 `M20_PYTHON=/path/to/python` 显式指定。

Z 起立，C 进入运控，8/5 前进与后退，4/6 左右移动，7/9 转向。R 复位到起立后的静止状态，Backspace 复位到趴卧状态。到顶不会自行退出，Esc 退出。按住 8 的目标速度沿用键盘配置；测试 1 m/s 可使用固定指令 `--command 1 0 0`，此时不加 `--keyboard`。

切换模型时指定 `--checkpoint PATH`，并把该模型真实的 `params/env.yaml`、`params/agent.yaml` 放在模型旁边；也可使用 `--config-dir PATH_TO_PARAMS`。不能用另一次训练的配置替代。

验证本次发布的 `36500`，从仓库根目录执行：

```bash
cd mujoco_validation
./run_mujoco_v135.sh --viewer --keyboard \
  --checkpoint ../M20_perception_Lidar_rl/logs/rsl_rl/M20-V11.1-AscentRecovery/2026-11-17_14-59-02_063111_guarded_30000_from35300/model_36500.pt \
  --config-dir ../M20_perception_Lidar_rl/data/m20_recovery/published_36500/params \
  --stair-height 0.16 --tread-depth 0.26
```

指定配置保存在 `logs` 之外，因此上述 `--config-dir` 必须保留。该候选在 1200 次追加更新时保存，尚未通过全部保留/最终修复条件，也没有进行降噪训练；发布它不表示行为已修好。

## 训练工程与改进状态

训练需要单独安装 Isaac Sim / Isaac Lab；它们不包含在仓库内。当前代码使用 RSL-RL 5.x，目标主机需核对实际版本与资源。说明见 [迁移文档](M20_perception_Lidar_rl/docs/MIGRATION.md)和[连续训练说明](M20_perception_Lidar_rl/docs/v111_guarded_training_20261117.md)。

当前默认 **4096 环境、每 100 次 PPO 更新评测、目标追加 30000 更新**。有效行为验收失败会记录、保存权重并继续训练，同一策略和优化器不会在评测后重置。失败权重不会标为通过；运行异常、无效评测、资源不足和用户中断仍可停止。

在目标主机安装好训练依赖后，从仓库根目录进行 CPU 预检查，不启动 Isaac 或 PPO：

```bash
cd M20_perception_Lidar_rl
export M20_PYTHON=python3  # 使用目标主机已激活的 Isaac Lab 环境
bash scripts/train_m20_v111_guarded.sh --check-only \
  --total-updates 30000 --segment-updates 100 --num_envs 4096 \
  --baseline-results "$PWD/data/m20_recovery/baseline_v2/results.json" \
  --output "$PWD/artifacts/recovery_4096_preflight"
```

正式训练由用户手动启动，命令见连续训练说明。默认初始化/冻结教师仍是原始 `35300`，不会自动换成发布候选 `36500`。行为指标及限制见[下一步分析](M20_perception_Lidar_rl/docs/v111_continuous_learning_20261117.md)，后续声学方向见[降噪调研](M20_perception_Lidar_rl/docs/m20_stair_acoustic_noise_literature_20261117.md)。
