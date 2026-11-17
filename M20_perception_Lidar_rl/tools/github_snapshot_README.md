# M20 Perception LiDAR RL

M20 感知运动训练代码与独立 MuJoCo 验证程序。当前保留的基线是 **V11.1 `model_35300.pt`**，对应训练任务 `M20-V11.1-V11FD-RearPosture`，原始训练配置一同保存在 `interview_prep/params/`。

```text
M20_perception_Lidar_rl/       训练工程、任务定义、奖励、工具与历史设计文档
  interview_prep/             唯一 checkpoint、原始 params、来源及 SHA-256
  data/m20_recovery/          参考模型的 MuJoCo 观测数据，不是另一个模型
  docs/                      改进分析与迁移说明
  deep_robotics_model/       官方机器人资源子模块
mujoco_validation/           验证程序、M20 资源、楼梯和平地地形
PUBLISH_MANIFEST.json         上传文件清单、过滤规则及模型来源
```

这是当前工作文件的迁移快照。旧 checkpoint、TensorBoard/运行日志、录像、缓存和临时实验输出未上传；原机器上的任务文件和训练记录保留。旧 Git 历史不包含在此快照中。训练目录内旧版 README、V13.* 报告及部分旧启动脚本是历史资料，其 checkpoint 不随此快照分发。`run_mujoco_v135.sh` 沿用原文件名，目前默认加载这里保留的 `35300`。

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

## 训练工程与改进状态

训练需要单独安装 Isaac Sim / Isaac Lab；它们不包含在仓库内。本机环境为 Python 3.11、Isaac Sim 5.1、Isaac Lab Python 包 0.47.2、RSL-RL 5.0.1；实际读取的包版本与检查边界见 [迁移说明](M20_perception_Lidar_rl/docs/MIGRATION.md)。

**保留的模型权重未改变，仍有跳阶和平地腿部前后伸展的问题。** 已加入逐阶落脚、整机完成和参考策略约束等改进代码，尚未运行新训练，不能据此声称模型问题已解决。分析见 [上楼退化与修复说明](M20_perception_Lidar_rl/docs/v111_ascent_recovery_20261117.md)。

在目标主机安装好训练依赖后，可先执行仅 CPU 的迁移检查，它不会启动 Isaac 仿真或训练：

```bash
cd M20_perception_Lidar_rl
bash scripts/train_m20_v111_recovery.sh --check-only
```

后续用户决定训练时，入口为 `scripts/train_m20_v111_recovery.sh`。环境数和每段迭代上限从 `scripts/repair_workstation_limits.json` 读取；它们依据原主机故障记录设置，目标主机仍需重新核算资源。每段结束后运行独立 MuJoCo 检查，不会自动衔接下一段。命令与验收范围见迁移说明。
