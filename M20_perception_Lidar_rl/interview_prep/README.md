# M20 雷达感知步态面试梳理

这个目录用于把 `kunyu_ppo` / M20 雷达感知楼梯步态项目整理成面试时可讲、可追问、可维护的一套资料。

## 入口

- `TECHNICAL_ROADMAP.md`：主技术路线图，从 V10 到 V11/V11-FD/V12/V13 系列的演进脉络。
- `CODE_INDEX.md`：关键代码证据索引，记录“哪段结论对应哪个文件/函数”。
- `INTERVIEW_STORY.md`：面试表达稿，适合压缩成 2 分钟、5 分钟或深挖版。
- `DISCUSSION_LOG.md`：后续我和你继续交流时维护的决策/补充/待验证日志。
- `diagrams/pipeline.mmd`：可以渲染成图的 Mermaid 架构图。
- `necessary_code/trace_contracts.py`：纯文本静态扫描脚本，用来快速回读 V11/V12 的任务、ABI 和契约证据。

## 当前证据边界

本目录基于本仓库当前文件静态阅读整理，未重新启动 Isaac Lab 训练，也未做 sim2sim 或真机验证。已有文档本身明确：

- V10 当前只确认 reward balance 已恢复，不能宣称 FIRST 后倒退或 17 cm FULL 已解决。
- V11 短训/冒烟主要证明状态机与奖励接线可达，不等同于候选模型验收。
- V11-FD 完整 30000 iteration、sim2sim 行为验收和真机验收在其 README 中标为未运行。

## 后续维护规则

交流中如果新增结论，请同时更新：

1. `TECHNICAL_ROADMAP.md`：放进路线图或待验证问题。
2. `CODE_INDEX.md`：给出至少一个文件/函数证据。
3. `DISCUSSION_LOG.md`：记录日期、问题、结论、证据、未验证边界。

更新前后可运行：

```bash
python interview_prep/necessary_code/trace_contracts.py
```

如果脚本输出 `M20_INTERVIEW_TRACE=FAIL`，说明某些路线图依赖的代码契约已经变了，需要先回读代码再改叙事。
