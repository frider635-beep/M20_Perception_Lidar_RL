# V11.1 逐级支撑与零转向偏航修复

## 对象与证据

用户实际运行的是验证 worktree 中的 `run_mujoco_v135.sh`，显式传入 EntryFlatRepair 的 `model_35499.pt`，按 Z 起立、C 切入、8 前进。脚本名称不代表 checkpoint 是 V13.5；本次沿用 V11.1 的 63 维观测与 16 维动作契约。

复查上一轮保存的真实 MuJoCo 接触轨迹，用户主场景中左前轮稳定支撑序列为 `1→3→4→6→8→10→12`，右前轮为 `1→3→6→10→11→12`。首阶已支持，后续仍漏踩。上一轮只验首个落脚，不能据此宣布逐级上楼通过，原报告已补充范围修正。

新的键盘复现使用验证程序原有 `Interaction`、`TeleopKeyboard` 与 8 键速度斜坡，模拟的是键盘读取接口，没有向桌面发送按键。Z 起立后切入 C，保持零指令，再按住 8。原 35499 的平地转向指令一直为零，但 C 后静止偏航约 10.51°，之后前进最大追加偏航约 2.58°。因此不能把问题解释成 W/8 冲突，也不能只检查输入值便认定无偏航。

## 实现

新任务 `M20-Stair-V11.1-AdjacentRepair-v0` 继承上一轮 EntryFlatRepair，初始化自用户实际使用的 35499。独立配置台账为 `M20V111AdjacentCfg`，新增运行脚本 `scripts/train_m20_v111_adjacent.sh`，仍复用原 launcher 的用户环境数量上限检查。

逐级支撑状态由现有 `v11_state → prepare_first_tread/finish_first_tread → finish_sequence` 接入，奖励通过新任务配置注册。每只前轮分别记录最高已支撑级号，依据真实楼梯几何、轮底高度、踏面内位置与接触力持续确认下一踏面；不规定左右前轮先后。稳定支撑下一踏面得到一次事件奖励，漏过任意中间踏面产生一次失败成本，过长摆腿有密集越界成本。漏踩历史持续保留，后来回踩不能洗掉失败；失败后下一目标随真实支撑位置重同步，继续学习剩余台阶，但整段无跳级奖励不再发放。B1 的已有事件奖励不重复支付。

本模块没有后续速度目标、速度乘数、等待时限、双轮同踏面停留或转向指令。首阶入口速度项继承上一轮的局部范围；真实 Isaac 对三个速度包装器做原值比较，入口和平地门控关闭时逐值等于原 V11.1。

零转向稳定性在几何确认的平地、零横移及零转向指令下，惩罚机身偏航角速度和横移速度，并增加训练中直行与静止指令的采样覆盖。显式转向与侧移指令不进入该门控。不更改 MuJoCo 的键位、动作混合或输出指令，也不通过部署侧姿态补偿掩盖策略问题。角速度奖励的行为效果必须通过新权重闭环比较，不能仅靠奖励代码判定通过。

## 资源与验证

完整来源哈希、实时资源余量、预算依据和进程上限保存在 `artifacts/v111_adjacent_repair_20261116/execution.json`。4096 是用户本机本轮指定的上限，不能据此推断任意任务都不会掉盘。初始权重与上一轮候选不覆盖。

17 项 CPU 测试通过，覆盖完整双前轮支撑、任意领前轮、不可重复奖励、单轮跳级后回踩不能洗掉历史、失败后继续学习、接触/几何假阳性、重置、非有限值、B1 速度范围及 benchmark 检测能力。

真实 Isaac 冻结策略检查通过：保存 ABI、观测/动作/执行器与旧任务一致，唯一有意改变的指令字段是直行和静止采样比例；旧任务原生落脚状态保持；后续速度包装器逐值一致。实跑识别到逐级支撑事件，奖励均有限且策略权重不变。数据与日志分别存入 execution 及 runtime_contract.log。

评测器现在记录每只前轮整段稳定支撑序列、所有前向跳过的踏面、整段完成状态、平地伸展、实际速度和零转向下的静止/运动偏航。到顶只在有限离线测试中结束，交互程序仍保持运行。未走完楼梯、只有 B1 通过或后来回踩都不能作为整段通过。原始物理轨迹保存在本地 NPZ，统计 JSON 纳入提交。

本机 live 知识库、Hub Space 工具及 hub Git 远端缺失，使用已读取的归档准则/技能和本地记录。Hub 空间建立与独立复核跳过；不能将本地记录称作 Hub 验收。origin fetch 为 HTTP 403，github 为 Repository not found，未向 GitHub 写入。原有日志、子模块内容、文件权限修改不纳入本次提交。

## 执行命令

```bash
cd /media/mark/2T_SSD/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl
# 有限训练；模型另存到 M20-V11.1-AdjacentRepair
bash scripts/train_m20_v111_adjacent.sh --num_envs 4096 --max_iterations 300

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  scripts/tools/test_m20_v111_adjacent.py scripts/tools/test_m20_v111_entry_flat.py
PYTHONPATH="$PWD/source/rl_training" python scripts/tools/test_m20_v111_entry_flat_runtime.py \
  --headless --adjacent --checkpoint /path/to/checkpoint.pt
python scripts/tools/m20_entry_flat_benchmark.py \
  --checkpoint /path/to/checkpoint.pt --output /path/to/new_results
python scripts/tools/m20_entry_flat_benchmark.py \
  --checkpoint /path/to/checkpoint.pt --startup z --keyboard-profile \
  --speeds .7 --treads .26 --output /path/to/new_keyboard_results
```

## 行为结果

**当前没有通过完整验收的替换权重。** 本轮计划 300 次更新，训练日志最后写于本地 18:05:49，最后记录迭代为 35794；机器于 18:06:49 重新启动。进程已不存在，没有正常结束的计时/内存 receipt，也没有最后的 35798 权重。原因尚不清楚，已向用户核实是否手动重启或训练期间故障；不能由日志缺失推定硬盘故障。原因确认前未重开 GPU 训练。

最后有效保存的是 `logs/rsl_rl/M20-V11.1-AdjacentRepair/2026-11-16_17-53-11_from35499_adjacent_straight_4096_trial300/model_35750.pt`，仍只是未通过候选。CPU MuJoCo 完整复测如下：

| 检查 | 原 35499 | 候选 35750 |
|---|---|---|
| 16/26 cm，0.5、0.7 m/s，双前轮逐级支撑完整楼梯 | 均有漏踩 | 两组均完整 1～12 |
| 16/26 cm，1 m/s，漏踩级号 FL / FR | [2,5,7,9,11] / [2,4,5,7,8,9] | [3] / [2,3] |
| 16/26 cm，1 m/s，后续实际速度 | 约 0.947 m/s | 约 0.820 m/s |
| Z→C 静止，偏航变化 | 约 10.51° | 约 3.00° |
| 随后按 8，平地额外最大偏航 | 约 2.58° | 约 13.81° |

30 cm 踏深的低/中速同样完整通过，但 1 m/s 右前轮仍漏第二级。候选平地姿态没有恢复原始 35300 的大幅外伸，但伸展指标也比 35499 增大。键盘 8 的实际稳态命令为 0.7 m/s，与独立的 1 m/s 测试分别记录。以上不是通过结果，不能只挑低速完整、静止偏航改善或总奖励增加来推荐模型。代码没有新增后续减速项，但实测后续速度下降，同样必须视为未满足用户要求。

比较明细、所有支撑序列和模型哈希保存在 `comparison.json`、各 benchmark 子目录的结果及 `execution.json`。较早的中间权重测试也保留，不冒充最终验收。接下来仍需解决高速漏踩、后续速度和直行偏航，并重新确认训练期间本机稳定性；真机本次未运行。
