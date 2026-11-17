"""Reproducible training manifest and evidence-limited post-run report."""
from __future__ import annotations
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

PREFIX = "Diag/V135Behavior/"
RATIOS = {
    "首摆腿落稳 B1/首摆事件": ("Ascent/first_lift_b1_event", "Ascent/first_lift_event"),
    "首摆空抬返回/首摆事件": ("Ascent/first_lift_empty_event", "Ascent/first_lift_event"),
    "首摆跳级/首摆事件": ("Ascent/first_lift_skip_event", "Ascent/first_lift_event"),
    "完整严格交替链/完整起点 episode": ("Ascent/full_chain_episodes", "Ascent/full_episodes"),
    "严格前缀长度/楼梯总级数": ("Ascent/strict_prefix_sum", "Ascent/flight_length_sum"),
    "成功 B1 后到 B2 条件比例": ("Ascent/B2_episodes", "Ascent/B1_episodes"),
    "上楼 B1 到达率": ("Ascent/B1_episodes", "Ascent/full_episodes"),
    "上楼 B2 严格前缀到达率": ("Ascent/B2_episodes", "Ascent/full_episodes"),
    "上楼 B3 严格前缀到达率": ("Ascent/B3_episodes", "Ascent/full_episodes"),
    "相邻异侧事件/合法完成": ("Ascent/ordered_events", "Ascent/valid_events"),
    "跳级事件/完整起点上楼 episode": ("Ascent/skip_events", "Ascent/full_episodes"),
    "双跳事件/完整起点上楼 episode": ("Ascent/dual_skip_events", "Ascent/full_episodes"),
    "严格平台事件/完整起点上楼 episode（账本代理）": ("Ascent/strict_platform_events", "Ascent/full_episodes"),
    "上楼平均路径速度 m/s": ("Ascent/speed_sum", "Ascent/full_frames"),
    "局部起点帧/完整起点帧": ("Ascent/partial_frames", "Ascent/full_frames"),
    "下楼平均路径速度 m/s": ("Descent/speed_sum", "Descent/frames"),
    "下楼轮地滑移 m/s": ("Descent/slip_sum", "Descent/frames"),
    "下楼地形相对俯仰误差 rad": ("Descent/terrain_pitch_error_sum", "Descent/frames"),
    "下楼有效地面证据比例": ("Descent/ground_evidence_sum", "Descent/frames"),
    "下楼后腿超长帧比例": ("Descent/extension_frames", "Descent/frames"),
    "下楼后腿折叠帧比例": ("Descent/folding_frames", "Descent/frames"),
    "下楼后腿后伸帧比例": ("Descent/backward_reach_frames", "Descent/frames"),
    "下楼异常持续超阈值帧比例": ("Descent/prolonged_defect_frames", "Descent/frames"),
    "出口后成功恢复延迟 s（仅成功样本）": ("Flat/recovery_seconds_sum", "Flat/recoveries"),
    "出口后未恢复即结束比例": ("Flat/censored_exits", "Flat/geometric_exits"),
}


def write_manifest(log_dir, env, runner, config):
    root = Path(__file__).resolve().parents[3]
    paths = list((root / "source/rl_training/rl_training").rglob("v135*.py"))
    paths += [root / "source/rl_training/rl_training/v11_continuous_runner.py",
              root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py",
              root / "scripts/reinforcement_learning/rsl_rl/train.py"]
    paths += [root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp" / name
              for name in ("events.py", "commands.py")]
    raw = env.unwrapped
    classes = {}
    for name, obj in (("runner", runner), ("ppo", runner.alg),
                      ("actor", runner.alg.actor), ("critic", runner.alg.critic)):
        classes[name] = {"class": type(obj).__module__ + ":" + type(obj).__name__,
                         "source": inspect.getfile(type(obj))}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                    classes=classes, num_envs=raw.num_envs, step_dt=raw.step_dt,
                    actor_dim=runner.alg.actor.obs_dim, critic_dim=runner.alg.critic.obs_dim,
                    action_dim=raw.action_manager.total_action_dim,
                    stage_starts_enabled=raw.cfg.v135_stage_starts.enabled,
                    rollout=config.num_steps_per_env, max_iterations=config.max_iterations,
                    ascent_tread_widths=raw.cfg.v135_riser_identity.ascent_tread_widths,
                    metric_command_bins=raw._v135_behavior_metrics.command_bins,
                    metric_terrain_levels=raw._v135_behavior_metrics.terrain_levels,
                    reward_weights={name: raw.reward_manager.get_term_cfg(name).weight
                                    for name in raw.reward_manager.active_terms},
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in paths if p.is_file()})
    manifest["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                           capture_output=True, text=True).stdout.strip()
    path = Path(log_dir) / "v135_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def build_report(run_dir, status="manual_snapshot"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    run = Path(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(run)
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    data = {}
    nonfinite = {}
    reward_trends = {}
    for tag in accumulator.Tags().get("scalars", []):
        # RSL versions may prepend Episode/ to extra log keys.
        if "Episode_Reward/" in tag or tag == "Train/mean_reward":
            reward_values = [float(event.value) for event in accumulator.Scalars(tag) if math.isfinite(event.value)]
            width = max(1, len(reward_values) // 5)
            if reward_values:
                reward_trends[tag] = {"first20pct": sum(reward_values[:width]) / width,
                                      "last20pct": sum(reward_values[-width:]) / width}
        if PREFIX not in tag:
            continue
        values = {}
        for event in accumulator.Scalars(tag):
            if math.isfinite(event.value):
                values[event.step] = float(event.value)
            else:
                nonfinite[tag] = nonfinite.get(tag, 0) + 1
        data[tag.split(PREFIX, 1)[1]] = values
    steps = sorted({step for values in data.values() for step in values})
    split = max(1, len(steps) // 5)
    windows = {"all": set(steps), "first20pct": set(steps[:split]), "last20pct": set(steps[-split:])}
    ratios = dict(RATIOS)
    for kind, title in (("AscentByLevel", "terrain level"), ("AscentByCommand", "command bin")):
        prefixes = sorted({tag.rsplit("/", 1)[0] for tag in data if tag.startswith(kind + "/")})
        for prefix in prefixes:
            token = prefix.rsplit("/", 1)[-1]
            for label, numerator, denominator in (("交替/合法完成", "ordered_events", "valid_events"),
                                                   ("跳级/有效帧", "skip_events", "frames"),
                                                   ("路径速度 m/s", "speed_sum", "frames")):
                ratios[f"{title} {token}：{label}"] = (prefix + "/" + numerator, prefix + "/" + denominator)
    for prefix in sorted({tag.rsplit("/", 1)[0] for tag in data if tag.startswith("AscentByTread/")}):
        token = prefix.rsplit("/", 1)[-1]
        for label, num, den in (("完整严格链", "full_chain_episodes", "episodes"),
                                ("交替/合法完成", "ordered_events", "valid_events"),
                                ("跳级/有效帧", "skip_events", "frames")):
            ratios[f"tread {token}：{label}"] = (prefix + "/" + num, prefix + "/" + den)
    ratios["B1 入阶滚转绝对值 rad"] = ("AscentEntry/B1_roll_sum", "AscentEntry/B1_frames")
    ratios["上楼质心投影越出支撑区域帧比例"] = ("AscentEntry/outside_support_frames", "AscentSupport/frames")
    ratios["合法落阶承重结算质量"] = ("AscentEntry/settlement_quality_sum", "AscentEntry/settlements")
    metrics = {}
    for label, (num, den) in ratios.items():
        metrics[label] = {}
        for window, selected in windows.items():
            # Missing tags remain unknown; they must never silently become zero.
            available = num in data and den in data
            common = selected & set(data.get(num, {})) & set(data.get(den, {}))
            numerator = sum(data[num][step] for step in common) if available else None
            denominator = sum(data[den][step] for step in common) if available else None
            metrics[label][window] = {"numerator": numerator, "denominator": denominator,
                                      "value": numerator / denominator if denominator else None,
                                      "matched_logging_points": len(common)}
    manifest_path = run / "v135_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    result = {"status": status, "run_dir": str(run.resolve()), "manifest": manifest,
              "iteration_range": [steps[0], steps[-1]] if steps else None,
              "logging_points": len(steps), "metrics": metrics, "nonfinite_samples": nonfinite,
              "reward_component_trends": reward_trends,
              "checkpoints": sorted(p.name for p in run.glob("model_*.pt")),
              "unverified": ["固定 terrain/command/seed 的独立冻结评估", "matched return / raw advantage / PPO gradient attribution",
                             "逐个物理阶真值复核与 detector disagreement", "真实成功前缀恢复后的动态稳定性"],
              "verdict": "行为验收未完成；训练日志不能单独证明目标策略已收敛"}
    def fmt(value):
        return "未采集/分母为零" if value is None else f"{value:.4f}"
    lines = ["# V13.5 训练后行为报告", "", f"运行目录：`{run.resolve()}`", f"结束状态：`{status}`",
             f"日志 iteration 范围：{result['iteration_range']}；有效记录点：{len(steps)}", "",
             "本报告由真实 TensorBoard 计数生成。前后窗口各取日志点的 20%，采用总分子/总分母，"
             "不是各环境比率的平均。训练地形课程变化会影响比较，不能据此断言因果。", ""]
    if manifest:
        classes = manifest.get("classes", {})
        lines += ["运行配置："
                  f"`{manifest.get('num_envs')} env × {manifest.get('rollout')} rollout`，"
                  f"计划最大 {manifest.get('max_iterations')} iterations；"
                  f"Actor/Critic/Action={manifest.get('actor_dim')}/{manifest.get('critic_dim')}/{manifest.get('action_dim')}；"
                  f"stage starts={'on' if manifest.get('stage_starts_enabled') else 'off'}。",
                  f"实际 PPO：`{classes.get('ppo', {}).get('class', 'unknown')}`；"
                  f"Actor：`{classes.get('actor', {}).get('class', 'unknown')}`。",
                  f"Git HEAD：`{manifest.get('git_head', 'unknown')}`；完整源码哈希和奖励权重见 `v135_manifest.json`。",
                  f"Checkpoint：{', '.join(result['checkpoints']) if result['checkpoints'] else '无'}。", ""]
    else:
        lines += ["未找到 `v135_manifest.json`，实际训练配置和源码版本未核实。", ""]
    lines += ["| 指标 | 前20% | 后20% | 全程 | 全程分母 |", "|---|---:|---:|---:|---:|"]
    for label, values in metrics.items():
        lines.append(f"| {label} | {fmt(values['first20pct']['value'])} | {fmt(values['last20pct']['value'])} | "
                     f"{fmt(values['all']['value'])} | {fmt(values['all']['denominator'])} |")
    lines += ["", "## 奖励分项趋势（日志原始单位）", "", "| 分项 | 前20%均值 | 后20%均值 |", "|---|---:|---:|"]
    lines += [f"| {name} | {fmt(values['first20pct'])} | {fmt(values['last20pct'])} |"
              for name, values in sorted(reward_trends.items())]
    lines += ["", f"非有限值记录：{nonfinite or '无'}", "", "## 判读边界", "", "- 同时看上楼 B2/B3、跳级和速度，不能只看总 reward。",
              "- 下楼超长、折叠、后伸是几何代理且可能重叠；不能当作互斥类别或动态稳定性证明。",
              "- 恢复延迟只统计成功样本；必须同时看未恢复退出比例，避免幸存者偏差。",
              "- 严格平台事件不是独立物理验收；任意楼梯级数的完整链仍需冻结轨迹评估。",
              "- 局部起点样本不计入完整起点到达率。中断时尚未结束的 episode 未计入 episode 分母。",
              "", "## 未完成验证", ""] + ["- " + item for item in result["unverified"]]
    lines += ["", "## 下次改动前", "", "固定同一批 checkpoint、terrain、command、seed，复核最差行为窗口；"
              "先确认是哪一段指标恶化，再决定改奖励、检测器或采样。当前不自动给出 reward-ranking 根因。", ""]
    (run / "v135_training_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    path = run / "V135_TRAINING_REPORT.md"
    path.write_text("\n".join(lines))
    return path
