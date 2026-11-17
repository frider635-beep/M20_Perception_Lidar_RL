# M20_perception_Lidar_rl

这是从 `2026-09-02_17-15-45_v11_fd_from_scratch` 整理出的独立 V11-FD 训练项目。
项目包含训练源码、M20 模型与工具、原始运行配置、301 个 checkpoint，以及完整 30000 iteration TensorBoard 日志。

## 固定训练记录

- 任务：`M20-Stair-Flat-Descent-V11FD-v0`
- 实验：`M20-V11-Flat-Descent`
- seed：42
- 环境数：8192
- 最终 checkpoint：`model_29999.pt`
- 特征契约：`v11_signed_edges_v2`
- Actor ABI：`obs[1,63] -> actions[1,16]`

日志位于：

`logs/rsl_rl/M20-V11-Flat-Descent/2026-09-02_17-15-45_v11_fd_from_scratch`

## 使用

```bash
./verify_project.sh
./tensorboard.sh
./train_from_scratch.sh
./resume_from_29999.sh
```

训练资源路径可以通过 `M20_TRAIN_ENV`、`M20_CONDA_SH`、`M20_ISAACLAB_SH`、
`M20_NUM_ENVS` 和 `M20_MAX_ITERATIONS` 覆盖。部署与 ONNX 导出位于同级目录
`../M20_perception_deploy_63d`。

## 可复现性边界

checkpoint、TensorBoard、运行期 YAML 和导出 ONNX 已完整保存并经过校验。原 V11-FD
核心源码在训练启动时未纳入 Git，且训练期间源工作树发生过变化，因此无法证明这里的源码与
启动进程逐字节一致；详细证据见 `provenance/RUN_PROVENANCE.md`。
