# 仓库迁移与当前入口

## 2026-11-17 本机更新

当前入口为 `scripts/train_m20_v111_guarded.sh`：4096 环境、每 100 次更新评测、目标追加 30000 更新；有效行为失败记录后继续，同一策略和优化器持续学习。运行异常、无效评测和资源不足仍可停止。训练由用户手动启动。

本次 `logs` 仅分发用户指定的 `M20-V11.1-AscentRecovery/2026-11-17_14-59-02_063111_guarded_30000_from35300/model_36500.pt`。其真实配置、模型哈希和历史评测在 `data/m20_recovery/published_36500/`；MuJoCo 加载时显式传 `--config-dir` 指向该目录的 `params/`。原始 35300 保留为默认初始化和冻结教师。36500 尚未通过全部行为条件，没有进行降噪训练。

原模型评测的迁移基线位于 `data/m20_recovery/baseline_v2/`，包含结果与两份来源 manifest，不含完整轨迹。克隆后从仓库根目录、在相容的 Isaac Lab 环境中预检查：

```bash
git submodule update --init --recursive
python3 -m pip install -e M20_perception_Lidar_rl/source/rl_training --no-deps
cd M20_perception_Lidar_rl
export M20_PYTHON=python3
bash scripts/train_m20_v111_guarded.sh --check-only \
  --total-updates 30000 --segment-updates 100 --num_envs 4096 \
  --baseline-results "$PWD/data/m20_recovery/baseline_v2/results.json" \
  --output "$PWD/artifacts/recovery_4096_preflight"
```

最新手动命令、验证边界见 [连续训练说明](v111_guarded_training_20261117.md)，问答与任务要求见 [本机交接文件](../../CODEX_SESSION_HANDOFF_20261117.md)。当前本机记录的 Isaac Lab/isaaclab-rl 版本为 0.54.3/0.4.7，目标主机应实际核对；下面的旧版本记录不能代替环境锁。

首次快照清单 `PUBLISH_MANIFEST.json` 保留为历史记录，本次更新见根目录 `PUBLISH_UPDATE_20261117.json`。本地历史任务、权重与日志保留，提交时只选择允许发布的内容。

## 以下为原主机首次迁移记录

以下的“唯一模型”“尚未训练”“失败后停止”、256 环境与旧启动流程仅描述首次迁移时状态，已由上面的本机更新和本轮用户要求覆盖。文中 `tools/` 导出工具没有随首次快照分发，不是本次 GitHub 更新的执行入口。

GitHub 快照保留同级 `M20_perception_Lidar_rl/` 与 `mujoco_validation/`。本地任务文件、全部旧模型及训练记录没有删除；发布时通过 `tools/github_export_policy.json` 过滤。完整导出清单和哈希见快照根目录 `PUBLISH_MANIFEST.json`。旧 Git 历史不随当前文件快照上传，避免历史中的大文件重新进入新仓库。

## 唯一分发模型

`interview_prep/model_35300.pt` 来自 `M20-V11.1-V11FD-RearPosture/2026-09-26_12-13-33_v111_native_ascent_from_v11fd29999_e8192_r24`。其 SHA-256、文件大小、原始配置哈希见 `interview_prep/CHECKPOINT.json`。

`interview_prep/params/env.yaml` 和 `agent.yaml` 从 SHA-256 完全相同的原始 checkpoint 所在训练目录复制，未使用其它版本配置。不要按文件名中的迭代数判断训练版本，也不要把 `35300` 说成所有历史实验中最新的一次：它是用户明确选定的后续改进基线。

权重保持原样，仍有跳阶、平地前后伸腿和零转向指令下偏航等已观察到的问题。新增 Recovery 代码尚未训练，不能直接改善现有权重。

## 环境

MuJoCo 验证与 Isaac Lab 训练可以使用不同 Python 环境。验证依赖见同级 `mujoco_validation/deploy/mujoco_v135/requirements.txt`，推荐 Python 3.11；CPU 验证不需要启动 Isaac Sim。

本次从环境读取的版本：Python 3.11、Isaac Sim 5.1.0.0、Isaac Lab Python 包 0.47.2、isaaclab-rl 0.4.4、PyTorch 2.7.0+cu128、RSL-RL 5.0.1。完整选定包版本见 `environment_versions_20261117.json`，它是环境记录，并非完整依赖锁文件；旧 README 的 Isaac Lab 发行标签不能替代当前包版本。Isaac Lab/Sim、CUDA 驱动及 Conda 环境不打包进仓库；应在新主机安装相容环境，保留其所需的 PyTorch/CUDA 版本。当前 Recovery runner 按 RSL-RL 5.x API 编写，不支持直接更换其它主版本。

```bash
# 从快照根目录执行；普通 clone 后可单独补齐资源。
git submodule update --init --recursive

# 进入已配置好的 Isaac Lab 环境后安装本项目；不让旧 setup.py 重装整套 RL 依赖。
python -m pip install -e M20_perception_Lidar_rl/source/rl_training --no-deps
cd M20_perception_Lidar_rl
bash scripts/train_m20_v111_recovery.sh --check-only
```

检查入口只执行 CPU 文件、权重接口、执行参数、RSL 版本和参考数据检查；不构造 Isaac App，不更新策略。它不能证明训练运行时接线正确。所有启动脚本默认使用当前环境的 `python3`；需要时设置 `M20_PYTHON=/path/to/python`。

## 验证与后续训练

```bash
cd ../mujoco_validation
./run_mujoco_v135.sh --viewer --keyboard --stair-height 0.16 --tread-depth 0.26
```

Z 起立，C 运控，8546 移动，79 转向；R 恢复起立后的静止状态，Backspace 恢复趴卧状态。默认模型已指向保留的 `35300`，到顶不会退出。显式切换模型时使用 `--checkpoint`，并携带对应原始 `params/`。

改进任务为 `M20-Stair-V11.1-AscentRecovery-v0`，入口 `scripts/train_m20_v111_recovery.sh`。用户后续决定开始训练、并完成目标主机资源核算后，可执行：

```bash
cd ../M20_perception_Lidar_rl
bash scripts/train_m20_v111_recovery.sh --num_envs 256 --max_iterations 100
```

以上训练命令本次没有执行。原主机环境数及迭代限制只在 `scripts/repair_workstation_limits.json` 维护；256 环境不是对新主机稳定性的保证。入口及直接调用 `train.py` 都读取同一个限制文件，拒绝超限和分布式启动。`--max_iterations` 在该修复任务表示本段追加更新数，不能再把 50000 当作诊断短跑使用。

每段训练从 `--initial-checkpoint PATH` 加载模型与 critic，使用新的优化器状态。冻结参考默认始终为保留的 `35300`；可显式设置 `M20_RECOVERY_TEACHER`、`M20_RECOVERY_REPLAY`，须一同携带对应文件。参考观测集是该原始模型在真实 MuJoCo 物理仿真中的观测，不含拟合策略生成的数据。

训练结束后串行检查本段已保存 checkpoint，按 `scripts/repair_acceptance.json` 检查完整逐阶落脚、整段完成、无跌倒、入阶后的速度和平地姿态。没有候选通过则返回失败，不自动继续训练。该检查不覆盖所有真机条件，也未把偏航加入晋级条件；偏航仍需单独按 Z→C→8 流程验证。

本次跳过：任何 PPO 更新或短训、Isaac 环境运行时验收、改进模型行为验收、目标主机负载与真机验收。

## 重新生成上传快照

```bash
# 输出目录必须是两个源工程以外的新目录，已有目录会拒绝覆盖。
python3 tools/export_github_snapshot.py --output /path/to/new_snapshot
```

脚本保留训练任务源码与设计文档，只过滤发布产物。使用普通文件复制，不修改原文件、不删除原文件。官方机器人资源通过固定 commit 的 Git 子模块提供，不复制其它机器人私有临时资源。导出后初始化 Git，并按 `PUBLISH_MANIFEST.json` 的 `robot_submodule_commit` 登记子模块 gitlink；发布收据记录实际分支和远端 commit。
