# M20 MuJoCo 验证

本目录只包含 MuJoCo 验证程序、机器人资源、地形和测试。训练源码留在同级 `M20_perception_Lidar_rl/`。程序名沿用 v135，支持符合原生 63D→16D 接口的所选模型。

默认权重是用户指定的：

```text
M20_perception_Lidar_rl/interview_prep/model_35300.pt
```

实际加载 checkpoint 的 MLP actor，严格使用该模型保存的 `params/env.yaml`、`params/agent.yaml`。配置来源会打印并写入运行记录。无须启动 Isaac Sim 或 ROS。

使用已安装 `requirements.txt` 依赖的 Python 环境。启动脚本默认使用当前环境的 `python3`，也可设置 `M20_PYTHON=/path/to/python`。切换模型时传入 `--checkpoint PATH`，并提供它真实的配套配置；不能借用另一次训练的 params。

## 运行

在 `mujoco_validation/` 下执行：

```bash
# 带窗口固定指令：初始趴卧，先 Z 起立，再 C 执行该指令。
./run_mujoco_v135.sh --viewer --command 0.5 0 0

# 键盘操作：Z 起立，C 运控；默认持续运行，Esc 退出。
./run_mujoco_v135.sh --viewer --keyboard

# 无窗口运行，未达到终点时返回非零。
./run_mujoco_v135.sh --headless --duration 30 --require-goal

# 下楼。
./run_mujoco_v135.sh --viewer \
  --terrain-xml deploy/mujoco_v135/terrains/stairs_descent_walls.xml

# 平地对照。
./run_mujoco_v135.sh --viewer \
  --terrain-xml deploy/mujoco_v135/terrains/flat.xml

# 保留碰撞墙，仅通过透明度改善视野；支持改变台阶参数。
./run_mujoco_v135.sh --viewer --stair-height 0.16 --tread-depth 0.26 --wall-alpha 0.25

# 录像。文件名必须尚未存在。
./run_mujoco_v135.sh --headless --duration 30 \
  --video /tmp/m20_v135_ascent.mp4
```

窗口启动后机器人处于趴卧状态，策略不会自行接管。

| 按键 | 行为 |
|---|---|
| Z | 两阶段平滑起立，完成后保持静止，等待 C |
| C | 起立完成后进入运控；起立过程中按 C 会等待起立完成后接管；趴卧时按 C 不直接接管 |
| R | 复位至最近一次完成 Z 起立时的位置和站立姿态，清零速度、指令及策略历史，保持静止等待 C；还未完成过起立时，恢复出生位置的默认站立状态 |
| Backspace | 回到出生位置、初始趴卧状态，清除运控状态，重新等待 Z |
| 8 / 5 | 运控模式下前进 / 后退 |
| 4 / 6 | 运控模式下左移 / 右移 |
| 7 / 9 | 运控模式下左转 / 右转 |
| X | 在当前位置平滑趴下 |
| Esc / Ctrl+C | 退出 |

支持主键盘数字行和数字小键盘；终端输入使用数字小键盘时请开启 NumLock。原 W/S/A/D/Q/E 不再发送移动指令。窗口失去焦点时停止键盘运动指令。未使用 `--keyboard` 时，C 后执行 `--command vx vy wz` 的固定指令。鼠标可旋转、缩放跟随视角。

到达楼梯顶端仅记录 `[GOAL]`，**不会自动退出**，也不会自动改变当前指令。窗口模式默认无限时；如需限时可显式传入 `--duration 秒数`。跌倒后窗口也保留，可按 R 或 Backspace 恢复。批量评估需要到顶即退出时，显式使用 `--stop-at-goal`；无窗口模式默认按指定时长运行，遇跌倒停止，可用 `--no-stop-on-fall` 更改。

默认使用本机 `~/miniconda3/envs/isaaclab-5.1/bin/python`。更换解释器：

```bash
M20_PYTHON=/path/to/python ./run_mujoco_v135.sh --viewer
```

依赖见 `requirements.txt`；本机已有依赖，未修改已有 Python 环境。替换 checkpoint 时使用 `--checkpoint /path/to/model.pt`。配置默认读取相邻 `params/`；如果 checkpoint 被单独复制，则在本地训练日志 `logs/rsl_rl/<experiment>/<run>/` 中按同名文件的完整 SHA-256 找回原配置。仅文件名或网络形状相同不能证明配置匹配；找不到或配置存在歧义会报错。相邻 `params/` 不完整也会报错，不会自动混用其它目录。

模型放到其他机器或路径后，可明确指定其原训练配置（同时需要 `env.yaml` 和 `agent.yaml`）：

```bash
./run_mujoco_v135.sh --viewer --keyboard \
  --checkpoint /path/to/model.pt \
  --config-dir /path/to/original/run/params
```

`--config-dir` 是用户明确选择的配置，程序仍校验网络与观测/动作契约，但无法仅凭 YAML 证明它和权重的训练来源一致；应使用该 checkpoint 原训练保存的文件。不猜测缺失的默认关节角或动作尺度。`--robot-xml` 可显式替换机器人，默认固定为本目录迁入的部署 MJCF。

## 实现与证据

- `run.py`：真实 `mj_step` 物理闭环、按名称映射关节/电机、窗口、录像及结果。
- `interaction.py`：趴卧、起立、静止等待、策略接管、两种复位和姿态快照。
- `policy.py`、`saved_config.py`：读取原始 actor 和保存配置，检查输入、动作、默认角、增益、限幅和 armature。
- `perception.py`：从真实碰撞地形射线得到局部点阵，提取 signed B/H/D，保留训练时的确认、保持和延迟；不上报地形模板中的阶数或高度作为感知输入。
- `terrain_loader.py`、`reset_placement.py`、`keyboard_input.py`、`teleop.py`：从用户指定部署目录迁移的地形、落地放置和交互实现。逐文件来源及哈希见 `MIGRATION.json`。
- `assets/`：部署机器人 MJCF 及其实际引用的网格；未保留源目录的无关网格和旧场景。
- `terrains/`：上楼、下楼、平地；默认上楼模板与用户提供文件一致。

默认 `clean` 模式关闭感知随机噪声，但保留时间过滤；`--perception noisy` 按保存配置加入高度/距离噪声和随机保持。当前未加入本体观测噪声、随机 PD 延迟或训练域随机化；这些差异写入每次 manifest。

每次运行输出到 `runs/<时间>/`：`manifest.json`（源与运行参数）、`summary.json`（结果、状态转换及终点事件）、`trajectory.npz`（控制步的状态、模式、63D 输入、动作、指令和接触）。长会话定期保存 `trajectory_00000.npz` 等分段文件，避免无限累积内存；最后一段仍保存为 `trajectory.npz`。`--output` 可指定新的结果目录，已有目录拒绝覆盖。复位后的时间线用 `episode` 区分；`--duration` 按整个会话的累计控制步计时。

终点判据检查四轮越过末级、位于路线内、连续的向上支撑接触及无检测到的绕行/跌倒。它不替代逐阶净空、交替顺序、冲击或多种子统计验收。`timeout` 表示到了指定时长，不能理解为通过或失败。

## 测试

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest \
  -o cache_dir=/tmp/m20_v135_pytest_cache deploy/mujoco_v135/test_contract.py -q
```

测试使用已安装 RSL-RL 作独立 actor 对照，并与 `reference_risers.py` 中冻结的训练几何函数对拍；RSL-RL、tensordict、pytest 仅用于测试，不是运行器依赖。关闭第三方 pytest 插件自动加载是为了隔离本机已有的 typeguard/typing_extensions 冲突。

实际执行结果、资源、未验证边界及 Git/Hub 状态见 `VALIDATION.md`。
