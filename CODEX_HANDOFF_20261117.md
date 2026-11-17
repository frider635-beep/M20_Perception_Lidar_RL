# M20 项目：本次对话记录与 Codex 交接

整理日期：2026-11-17（Asia/Shanghai）。按任务推进过程整理，保留用户关键要求、模型来源、执行证据、代码入口和未完成事项。接手者应先读本文件，再查看下文列出的配置和报告。

## 1. 接手时先明确的状态

1. 后续改进基线是用户指定的 **`M20_perception_Lidar_rl/interview_prep/model_35300.pt`**。这是原始权重，尚未解决原先的跳阶和平地伸腿问题。
2. 新增 `M20-Stair-V11.1-AscentRecovery-v0` 及配套代码已经提交，但**没有运行新任务的 Isaac/PPO 训练，没有产出通过行为验收的新模型**。
3. 用户最后对训练的要求是：**“不要启动任何短训，做好改进就行，后续我要把仓库迁移到另一台主机。”** 该要求仍有效。不要因继续任务、迁移检查或所谓冒烟测试而自动启动 PPO 更新、短训、长训或离线拟合；等待用户新的明确训练指令。
4. 用户最后对整理方式的要求是：**“整理过程不需要删除任务文件，在上传时将不需要的文件进行过滤即可。”** 本地历史任务、模型和记录保留；仅发布快照过滤旧模型和生成物。
5. 仓库已上传到 **https://github.com/frider635-beep/M20_Perception_Lidar_RL**。首次发布的代码快照提交是 `6b44e803c6fd9c38b0b6b15d1e3b799632522270`；本交接文档在之后的文档提交中补充。
6. 已在独立过滤快照中通过 39 项修复 CPU 测试和 14 项 MuJoCo 接口/交互测试，共 53 项；这不代表新训练任务的运行时或修复模型行为已经通过。

## 2. 用户目标和本次对话的推进过程

### 阶段 A：阅读同事的训练仓库

用户最初提供同事的 M20 代码库，要求理解历史工作、训练记录、输入观测及上下楼奖励。仓库包含多个 V11/V11.1/V11.2/V13.* 分支方案和历史报告，不能把某篇旧 README 的接口或验收结论直接套在其它 checkpoint 上。

用户一度确认验证对象为 V13.5：

```text
M20_perception_Lidar_rl/logs/rsl_rl/M20-V13.5-Final-FromScratch/
2026-09-22_21-00-36_v135_entry_credit_resume53600_to73300_e8192_r24/model_73299.pt
```

之后用户明确选择 `interview_prep/model_35300.pt` 作为继续改进的对象。**版本号较大、训练迭代号较大、文件修改时间较新，都不能直接说明它是当前应使用的基线。** 当前选择已经明确，不应再擅自换回 V13.5 或其它“最新”模型。

### 阶段 B：迁移 MuJoCo 验证程序

本工程最初没有独立验证程序。用户允许参考并复制迁移：

```text
/home/mark/deeprobotics_lab/deploy_real_ws/rl_training/deploy
```

用户指定的地形参考文件是：

```text
/home/mark/deeprobotics_lab/deploy_real_ws/rl_training/deploy/deploy_mujoco/terrains/stairs_ascent_walls.xml
```

原主机上新建了同级的 `mujoco_validation/`，用户要求这里仅放 MuJoCo 验证需要的程序、机器人资源、地形和测试，训练源码留在 `M20_perception_Lidar_rl/`。旧机上的两个目录曾是同一 Git 仓库的不同工作树；**GitHub 迁移版将它们作为同一个仓库里的两个普通目录**，不需要复原旧工作树结构。

验证程序入口保留旧名称 `run_mujoco_v135.sh` 和 `deploy/mujoco_v135/`，当前默认已改为加载保留的 V11.1 `35300`。文件名里的 v135 不是当前模型版本选择。

### 阶段 C：按用户要求恢复交互

用户指出原键位冲突，要求移动/转向改为 **8546 / 79**，之后又要求恢复原部署程序的模式切换和复位行为：

| 按键/事件 | 当前约定 |
|---|---|
| Z | 平滑起立，完成后保持静止，等待 C |
| C | 从起立状态进入策略运控；起立过程中按 C 会等待起立结束 |
| 8 / 5 | 前进 / 后退 |
| 4 / 6 | 左移 / 右移 |
| 7 / 9 | 左转 / 右转 |
| R | 恢复最近一次 Z 起立后的静止站立状态，清零速度、指令和策略历史，等待 C |
| Backspace | 回到初始位置的趴卧状态，重新等待 Z |
| X | 当前位置平滑趴下 |
| Esc / Ctrl+C | 退出 |
| 到达楼梯顶端 | 记录终点事件，窗口继续运行；不自动退出 |

W/S/A/D/Q/E 不再发送移动指令。用户后来描述“按 W 前进向左转”，已经澄清实际是 **MuJoCo 中按 8**。不要把旧表述误解成又要恢复 W。

交互模式默认不设运行时限；批量测试可显式设置 `--duration` 或 `--stop-at-goal`。键盘 8 的速度按键盘配置执行，不能直接把它当作用户回忆的真机满杆 1 m/s 工况；要分别验证。

### 阶段 D：迁移模型加载失败和配置修复

用户将原始 `model_35300.pt` 复制到 `interview_prep/` 后运行验证，曾报错：

```text
FileNotFoundError: .../interview_prep/params/env.yaml
```

原因是移动了 checkpoint，但没有携带它真实的训练配置。最初增加了按权重 SHA-256 寻找同名原始训练文件的回退；本次发布时又把原始 `params/env.yaml`、`params/agent.yaml` 直接补齐到模型旁边。因此迁移版加载不再需要旧日志。

不要仅凭 `model_35300.pt` 这个名字选择配置：其它版本训练也可能有相同迭代号。不能借用另一个模型的 params 来绕过报错。

### 阶段 E：用户提出的行为问题

用户要求基于 `35300` 改善：

1. 高速入阶时，前腿第一次抬腿直接跳过第一级，落到第二级开始上楼。
2. 平地运动时，两前腿向前伸、两后腿向后伸。
3. 修复首阶后又出现跳过第二级、直接到第三级的问题，需要关注后续整段逐级落脚。
4. Z→C 后，按 8 前进，在零转向指令下机身向左偏转。用户确认发生在 MuJoCo；当时使用 EntryFlatRepair `model_35499.pt`。

用户对实际问题工况的回忆是：**“应该是摇杆推满 1 m/s，级高 16 cm，踏深 26 cm。”** 这是测试参考，真机实际速度和尺寸仍应以新测量为准。

用户的明确约束：**“这个减速只能在入阶时有，在后续台阶不能有。”** 不能把整段速度降下来，便宣称已经修好跳阶。

### 阶段 F：已做训练的效果和硬件故障

历史训练记录如下。这些旧 checkpoint **没有上传**，表格用于理解过程，不是新的推荐模型。

| 阶段 | 关键记录 | 结论 |
|---|---|---|
| 原始 V11.1 | `interview_prep/model_35300.pt` | 用户选定的基线；权重保持原样 |
| EntryFlatRepair | `2026-11-16_17-07-08_from35300_entry_flat_4096_trial200/model_35499.pt` | 首阶和平地姿态改善，但后续仍跳阶；不是完整解决 |
| AdjacentRepair 4096 环境试训 | `2026-11-16_17-53-11_from35499_adjacent_straight_4096_trial300/model_35750.pt` | 中断/重启；高速度跳阶、偏航和速度仍未全通过，候选未验收 |
| 后来的 256 环境长训 | `2026-11-16_19-29-54_hardware_diag_256_short/model_75350.pt` | 本次分析使用的失败末段模型；严重丧失上楼能力 |

前三类运行目录分别在 `logs/rsl_rl/M20-V11.1-EntryFlatRepair/` 和 `logs/rsl_rl/M20-V11.1-AdjacentRepair/` 下。文件名中的 `short` 不代表运行实际上短：失败长训的实际配置是 256 环境、rollout 24、固定学习率 1e-4、最大追加更新数 50000。

用户起初要求环境数不要太高，后曾允许先开 4096。出现重启后，用户明确确认：**“训练期间掉盘或自动重启了。”** 不能再将 4096 当作已验证的安全设置。现有证据不能单独证明根因是 OOM、硬盘、供电或温度；256 环境长时间运行也不等于已经证明硬件绝对安全。

用户最终评价：**“这次训练很失败，完全丧失了上楼能力，你可以自行去 mujoco 验证中采集仿真数据。仔细分析原因，并进行改进。”** 随后补充了停止新短训、准备迁移的要求。

### 阶段 G：仓库整理和发布

用户要求上传到上述个人 GitHub，checkpoint 只分发指定的 `35300`。之后澄清不删除本地任务文件，上传时过滤即可。

已执行：

- 在独立 `github_migration/` 目录建立发布副本，没有将原训练树的大量旧 Git 历史推上去。
- 发布训练源码、任务定义、工具、设计文档、独立 MuJoCo 验证程序和固定版本的官方机器人资源子模块。
- 只分发 `35300` 及其原始 params；过滤旧权重、原始日志、缓存、录像、临时实验输出等。
- 对发布前列出的 3407 个历史权重文件检查存在性，缺失数为 0；本地未因本次整理删除任务或模型文件。
- 推送 `main` 和 `agent/codex/github-migration`，确认二者在首次发布时均指向 `6b44e80`。
- 从 GitHub 重新克隆并核对 526 个源文件记录的 SHA-256、唯一模型、脚本可执行权限和干净工作树。首次快照含 534 个跟踪条目，后续补充本文会增加文档条目。

## 3. 唯一分发模型：身份、接口和配置

```text
相对仓库根目录：M20_perception_Lidar_rl/interview_prep/model_35300.pt
原始训练：M20-V11.1-V11FD-RearPosture/
  2026-09-26_12-13-33_v111_native_ascent_from_v11fd29999_e8192_r24
SHA-256：4aefe17ddc279ee809ff6257862ba4e28bb1b19ad48b11a5948b116256d61f8b
```

模型和原始配置的校验信息在 `interview_prep/CHECKPOINT.json`；配置就在旁边的 `params/`。用户没有要求把这个模型改名为“最新”，也没有验收任何新的替代模型。

当前验证器严格加载原生 **63D→512→256→128→16D ELU MLP**，不启用 actor 观测归一化；感知协议为 `v11_signed_edges_v2`。

Actor 输入顺序（Python 半开区间）：

| 下标 | 内容 |
|---|---|
| `0:3` | 机身角速度 |
| `3:6` | 机身坐标系重力投影 |
| `6:9` | 速度指令 |
| `9:25` | 相对默认关节角；其中轮子位置通道置零 |
| `25:41` | 关节速度 |
| `41:57` | 上一步动作 |
| `57:60` | 当前边缘 `[B1, H1, D1]` |
| `60:63` | 下一边缘 `[B2, H2, D2]` |

缩放和裁剪从保存配置读取。Actor 不输入实际机身线速度、完整点云或完整高程图。当前训练/预检查要求原生 568D Critic，但部署验证不使用 Critic。

输出排列：先 `fl/fr/hl/hr` 各自的 `hipx/hipy/knee`，再四个轮子的速度动作，共 16 维。关节默认角、动作缩放、PD、扭矩限幅、armature 必须匹配该模型配置。

若后续切换 checkpoint，需要匹配此接口并携带它自己的真实 params。当前验证器会拒绝不兼容的 recurrent/MoE/归一化模型，不能靠猜配置继续运行。

历史奖励说明可能对应其它版本。要回答“35300 当时训练的奖励”，优先读取该权重保存的 `params/env.yaml`；要回答“当前新任务使用的奖励”，读取当前配置类及继承链，不混为一谈。

## 4. 仿真采样得到的证据和原因分析

完整摘要：[v111_ascent_recovery_20261117.md](v111_ascent_recovery_20261117.md)。机器可读数据：[v111_ascent_benchmark_20261117.json](v111_ascent_benchmark_20261117.json)。

在相同 MuJoCo 执行接口下，对原模型及失败长训的中间/末段模型采集全程轨迹。关键判断：

- `35300` 可以到顶，但包含首阶和后续跳阶；平地轮子纵向伸展较大。
- `35499` 的首阶改善不代表后续每一级都落脚。
- `40000` 在某一个高速楼梯条件下逐阶，但后续速度显著不足，换速度或踏深也未全通过。
- `60000`、`70000` 逐渐停滞，`75350` 在代表性楼梯条件下跌倒；平地速度正常并不能说明上楼能力保留。
- 总 reward 没有正确反映这些跨仿真器行为退化；同一批 MuJoCo 观测上的 actor 输出相对参考发生明显漂移。

有代码和数据支持的风险机制：旧 overshoot 每帧持续收费，可能鼓励少动/停滞；旧 complete 只看前轮序列，不能证明后轮和机身到顶；原生进度项在部分目标速度即饱和；长时间整网更新缺少固定参考和外部行为晋级检查。

这些机制尚未通过新的训练消融确定各自独立的因果贡献。**不要将分析写成已经证明的唯一根因，也不要将已编写的修复写成行为已解决。**

旧机还保留一些离线输出层拟合实验，工具是 `scripts/tools/m20_drive_head_recovery.py`。这些实验没有产生全条件通过的候选，不是当前解决方案；实验权重未上传。当前禁止新训练的要求同样不能用“只是离线拟合”绕过。

## 5. 已准备的新 Recovery 代码及接线

下面路径均相对 `M20_perception_Lidar_rl/`：

| 文件 | 作用 |
|---|---|
| `scripts/train_m20_v111_recovery.sh` | 新任务启动/仅检查入口；启动前校验接口；每段结束跑 MuJoCo 验收 |
| `scripts/repair_workstation_limits.json` | 环境数、迭代段长度的唯一限制台账 |
| `scripts/reinforcement_learning/rsl_rl/repair_workstation_limits.py` | 启动前限制，缺失/非法配置失败拦截 |
| `scripts/reinforcement_learning/rsl_rl/train.py` | 接入限制、新任务、运行目录收据和课程步数恢复 |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v111_recovery_env_cfg.py` | 新任务奖励/课程/runner 配置；同目录 `__init__.py` 注册任务 |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/v111_recovery.py` | 有限越阶代价、机身/后轮新进展、四轮到顶事件 |
| 同一 `mdp/` 下的 `v111_first_tread.py`、`v111_adjacent.py` | 初始化并更新 Recovery 状态的调用入口 |
| `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/curriculums.py` | 新任务可选整机上楼完成的课程晋级门槛 |
| `source/rl_training/rl_training/v111_recovery_runner.py` | 固定参考策略、观测回放约束、冻结部分 actor 参数与标准差 |
| `source/rl_training/rl_training/recovery_assets.py` | 从仓库相对路径解析参考权重/回放，支持显式环境变量 |
| `data/m20_recovery/reference_observations_v1.npz` 及 `.json` | 保留的原始 35300 在真实 MuJoCo 物理仿真中的 5210 条 63D 观测及来源哈希；不是第二个模型 |
| `scripts/tools/m20_recovery_preflight.py` | CPU 文件、版本、actor/critic/执行参数/回放检查；不创建 Isaac App |
| `scripts/tools/m20_entry_flat_benchmark.py` | 完整 MuJoCo 物理轨迹、逐阶前轮支撑、速度、姿态及偏航记录 |
| `scripts/repair_acceptance.json`、`scripts/tools/m20_repair_acceptance.py` | 固定工况集合和分段候选验收；没有候选通过就停止，不自动续训 |

新任务继承 EntryFlatRepair/AdjacentRepair，保留 B1 入阶和平地轮子工作空间约束，改动主要是：

- 同一目标台阶的 overshoot 只收有限增量代价，避免停留时重复收费。
- 对机身和较落后的后轮真实向前新进展给分，来回移动不能重复刷分；B1 入阶减速区域关闭此进度奖励。
- 完成事件和上楼课程晋级要求无漏阶的前轮序列、四轮在顶面持续支撑、站立与足够机身净空。
- 后续楼梯进度目标恢复至完整速度指令，并保留独立速度验收。
- 冻结早期 actor 表征和动作标准差；在保存的 MuJoCo 观测上约束参考输出，上楼感知有效时放宽前腿抬腿关节的参考约束。
- 调整学习率和 rollout，缩短并限制每段训练；不再自动运行很长的更新段。

准确权重和超参数以代码/JSON 台账为准，不从本交接文档反向抄回配置。原主机限制当前采用保守的低环境数，不代表新主机稳定性已验证。

### 接手后必须保留的未验证项和设计风险

1. **新 Recovery 的 Isaac 运行时接线和任何 PPO 更新尚未执行。** CPU 正反例或静态接线检查不能替代运行时验收。
2. **参考模型已经统一为 35300。** 它自身存在平地伸腿和偏航；参考约束可能同时限制需要修复的输出。这是需要审查/后续实测的设计风险，不能声称“加了 teacher 就一定修好”。
3. Recovery 禁用了旧新增的 `v111_straight_stability`，当前自动晋级 JSON 也没有把偏航作为硬门槛。因此 **Z→C→8 左偏仍是明确待办**。
4. 逐阶序列门槛目前针对两只前轮，整机完成还检查四轮到顶；它没有证明两只后轮也踩过每一个台阶。
5. 减速范围必须继续保持在 B1 入阶，后续逐阶修复不能通过整段限速取得表面成功。
6. 稳定性问题没有确诊，迁移后先读取整机/GPU/磁盘余量和同机负载。不能仅凭显存就提高环境数。

## 6. 新主机的阅读顺序和可执行命令

先读取目标机器适用的 `AGENTS.md`/公司规则，再读本文件、仓库根 README、[MIGRATION.md](MIGRATION.md)、模型 `CHECKPOINT.json`、上述分析报告和实际代码。旧 README 中某些“已完成/全部修复”标题属于历史版本，不能作为当前行为验收凭据。

克隆完整发布版：

```bash
git clone --recurse-submodules https://github.com/frider635-beep/M20_Perception_Lidar_RL.git
cd M20_Perception_Lidar_RL
```

已有 clone 时先确认自己的工作树，再正常拉取并执行 `git submodule update --init --recursive`。官方资源固定在 `18192847e16ab85c056440072fcc5844cef43856`，路径为 `M20_perception_Lidar_rl/deep_robotics_model/`。

### 6.1 只运行 MuJoCo

可使用已安装依赖的环境，也可按根 README 创建单独的验证环境。进入正确环境后：

```bash
# 在仓库根目录执行
python -m pip install -r mujoco_validation/deploy/mujoco_v135/requirements.txt
cd mujoco_validation
./run_mujoco_v135.sh --viewer --keyboard \
  --checkpoint ../M20_perception_Lidar_rl/interview_prep/model_35300.pt \
  --stair-height 0.16 --tread-depth 0.26
```

Z 起立、C 运控、8546/79 操作；R/Backspace 按前文约定复位。到顶继续运行。

固定 1 m/s 指令时不加 `--keyboard`，仍由 Z→C 接管：

```bash
./run_mujoco_v135.sh --viewer --command 1 0 0 \
  --stair-height 0.16 --tread-depth 0.26
```

换模型时传 `--checkpoint PATH`，并携带对应 params；必要时显式加 `--config-dir PATH_TO_PARAMS`。默认 Python 来自当前环境 `python3`，可以设置 `M20_PYTHON=/path/to/python`。

### 6.2 CPU 迁移检查，不训练

这一步需要已有相容的 Isaac Lab Python 环境，但检查本身不启动 Isaac App：

```bash
# 在仓库根目录，激活已安装好的训练环境后执行
python -m pip install -e M20_perception_Lidar_rl/source/rl_training --no-deps
cd M20_perception_Lidar_rl
bash scripts/train_m20_v111_recovery.sh --check-only
```

**必须保留 `--check-only`。不加这个参数会走训练入口，当前没有授权执行。** `MIGRATION.md` 的正式训练命令只是后续获授权后的示例，不应在交接时照跑。

本机实读的包版本在 `environment_versions_20261117.json`。其中 Isaac Lab Python 包版本为 0.47.2、isaaclab-rl 为 0.4.4，不能只照旧 README 的发行标签装环境。它是版本记录，不是完整环境锁；目标机依赖仍需核对。

### 6.3 已执行过的 53 项测试复现入口

下面命令在仓库根目录执行，需要 pytest、RSL-RL、tensordict 及验证依赖。运行前先核算目标机资源；它包含 CPU MuJoCo 物理仿真，不包含 PPO 训练。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  M20_perception_Lidar_rl/scripts/tools/test_m20_recovery_math.py \
  M20_perception_Lidar_rl/scripts/tools/test_m20_recovery_guards.py \
  M20_perception_Lidar_rl/scripts/tools/test_m20_v111_adjacent.py \
  M20_perception_Lidar_rl/scripts/tools/test_m20_v111_entry_flat.py \
  mujoco_validation/deploy/mujoco_v135/test_contract.py
```

只采集当前基线的 MuJoCo 物理数据：

```bash
# 从仓库根目录进入训练目录；output 必须是尚未存在的新目录
cd M20_perception_Lidar_rl
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/tools/m20_entry_flat_benchmark.py \
  --checkpoint interview_prep/model_35300.pt \
  --output artifacts/baseline_35300_new_host_run01
```

脚本默认扫描平地及不同踏深/速度，保留 manifest、结果和轨迹；不启动学习。键盘流程另有 `--startup z --keyboard-profile --speeds .7`，应结合当前脚本参数查看，不能以这一速度替代所有工况。

## 7. 已验证与跳过项

已执行的事实：

- 39 项修复 CPU 测试通过；14 项 MuJoCo 合约/交互测试通过。
- 原始 35300、配套 params、参考观测集的文件/接口/有限值检查通过。
- 在不含旧训练日志的过滤快照中再次执行上述 53 项，全部通过。
- 官方资源从上游拉取并检出固定 commit。
- 发布副本的 shell 入口执行过 1.2 秒 headless 物理仿真，完成加载和启动；这只是入口检查。
- GitHub `main` 与任务分支推送成功；重新克隆后核对模型唯一性、文件哈希与启动脚本可执行权限。

明确跳过/未完成：

- 新 Recovery 的 Isaac 环境运行时验收、任何 PPO 更新或短训。
- 新改进 checkpoint 的行为验收；当前不存在已通过的新 checkpoint。
- 新主机负载、磁盘/GPU/供电稳定性验收及真机运行。
- 偏航的完整修复验收。

全量首次快照的空白格式扫描发现未改历史文件中的旧格式问题；未批量格式化这些历史文件。已修改代码的相应检查通过。不要把“53 项测试通过”表述为仓库所有历史测试、所有任务、所有真机条件均已通过。

## 8. 推荐继续的任务顺序

1. 确认 clone 的实际 HEAD、目录结构、模型哈希、params 和子模块，完成仅 CPU 的迁移检查。
2. 审查新 Recovery 的调用接线及未验证风险，特别是原始 35300 的参考约束是否会妨碍平地姿态/偏航修复；先改进代码和必要的正反例测试。
3. 在资源允许时用 MuJoCo 重现 35300 的完整楼梯、平地和 Z→C→8 轨迹，区分键盘指令、坐标变换、观测与策略输出的来源。
4. 补齐偏航验收，并检查完整逐阶条件、入阶后的实际速度和四轮整机完成证据。仅入阶成功、只到顶、只低速通过或 reward 上涨都不够。
5. 用户明确授权训练后，才进行目标机资源核算、有限段训练和逐段外部验收；没有候选通过就保持失败状态，不自动续训，不替换原始基线。

## 9. 旧主机路径和 Git 注意事项

以下路径仅用于追溯旧机器，不能复制成新主机的固定配置：

```text
外层工作区：/media/mark/2T_SSD/M20_perception_Lidar_rl_2
原训练树：  外层工作区/M20_perception_Lidar_rl
原验证树：  外层工作区/mujoco_validation
发布副本：  外层工作区/github_migration
旧训练 Python：/home/mark/miniconda3/envs/isaaclab-5.1/bin/python
```

旧外层工作区本身不是 Git 仓库；`github_migration/` 才是上传用的独立仓库。新主机普通 clone 后，克隆目录本身就是 Git 根目录，两个工程目录在其中。

旧原训练树的历史 remotes 读取失败，且没有 `hub` remote；本次用户明确指定个人 GitHub，已按此目标发布，不要往旧的同事仓库误推。发布副本的远端名称为 `user-github`，任务分支为 `agent/codex/github-migration`；新机器正常 clone 后远端通常叫 `origin`。

旧原训练树在整理前已有较大的 `artifacts/v112_route_repair_20260928/train_fromscratch.log` 修改及 `deep_robotics_model` 未跟踪内容，这些保留在旧机，没有混入公开快照。不要为了“弄干净”而对原工程做 `reset --hard`、`clean`、删除任务/模型或覆盖这些用户文件。

后续提交按目标机器的适用规则执行，先拉取、使用任务分支、记录测试和推送结果。只改文档不需要重跑训练或全仓库仿真。不要把另一窗口的消息当成本窗口用户授权。

## 10. 给下一台 Codex 的启动说明

用户可以把下面一段直接发给新会话：

> 请先阅读 `M20_perception_Lidar_rl/docs/CODEX_HANDOFF_20261117.md`、根 README 和 `MIGRATION.md`，核对实际代码与保存配置后继续 M20 上楼/平地姿态/偏航修复。基线只用用户选定的 `interview_prep/model_35300.pt`。目前只做代码改进、CPU 检查和必要的 MuJoCo 数据分析，不要启动任何短训、正式训练、PPO 更新或离线拟合；等待我新的训练指令。保留本地任务文件，发布时过滤旧模型和运行产物。先说明已确认的状态和还未验证的事项，再继续工作。
