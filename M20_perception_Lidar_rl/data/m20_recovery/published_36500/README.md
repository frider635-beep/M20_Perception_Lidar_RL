# 用户指定发布的 Recovery 36500

权重保留在训练目录中的原路径：

```text
logs/rsl_rl/M20-V11.1-AscentRecovery/2026-11-17_14-59-02_063111_guarded_30000_from35300/model_36500.pt
SHA-256: 76d06d6daa19693ebdbbb2d3619a7dc6fd5ada2a9fb5a0daf3b345bdecc4f9c5
```

按用户要求，Git 的 `logs` 子树只包含这一份权重。这里的 `params/env.yaml`、`params/agent.yaml` 从该次训练真实保存的配置原样复制，未借用原始 35300 或其他训练配置；详细哈希见 `CHECKPOINT.json`。

从仓库根目录加载该候选：

```bash
cd mujoco_validation
./run_mujoco_v135.sh --viewer --keyboard \
  --checkpoint ../M20_perception_Lidar_rl/logs/rsl_rl/M20-V11.1-AscentRecovery/2026-11-17_14-59-02_063111_guarded_30000_from35300/model_36500.pt \
  --config-dir ../M20_perception_Lidar_rl/data/m20_recovery/published_36500/params \
  --stair-height 0.16 --tread-depth 0.26
```

必须显式指定 `--config-dir`，因为配置没有放进 `logs`。保存配置中原主机的绝对路径用于历史追溯，目标主机训练时应使用当前代码的资产解析和实际环境路径。

`evaluation/` 原样保留该候选在追加 1200 更新后的结果和报告。报告内原主机路径是历史记录，当前路径使用本目录文件。候选当时 `retention_passed=false`（站立平地 1 m/s 的侧移退化），`repair_passed=false`；它不是通过全部行为条件或已经降噪的模型。

默认训练初始化和冻结教师继续采用 `interview_prep/model_35300.pt`。发布候选不自动切换初始化、教师或优化器恢复配置。
