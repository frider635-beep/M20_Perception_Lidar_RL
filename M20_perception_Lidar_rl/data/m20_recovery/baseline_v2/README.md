# 原始 35300 的固定评测基线

这里原样复制了 `artifacts/guarded_30000_baseline_20261117_v2/` 的 `results.json`、`standing/manifest.json` 和 `keyboard/manifest.json`，供另一台主机进行连续训练预检查和保留能力比较。复制来源、模型身份和文件哈希见 `PROVENANCE.json`。

没有复制原始完整轨迹或其他运行产物。manifest 中原主机绝对路径是历史来源；预检查使用实际 `--baseline-results` 路径，并读取同目录两份 manifest 的原始模型 SHA-256，不依赖这些历史路径存在。

此基线覆盖 12 个平地/上楼工况，不包含下楼、声学或真机验收。与新候选比较不意味着原始模型本身已解决漏阶、伸腿或偏航问题。
