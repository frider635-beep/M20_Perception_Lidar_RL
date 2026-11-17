"""Verify count-denominator report math and missing-data behavior with real event files."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from torch.utils.tensorboard import SummaryWriter
import importlib.util
_module_path = Path(__file__).resolve().parents[2] / "source/rl_training/rl_training/v135_training_report.py"
_spec = importlib.util.spec_from_file_location("v135_training_report", _module_path)
_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_report)
build_report = _report.build_report
PREFIX = _report.PREFIX

with TemporaryDirectory() as folder:
    writer = SummaryWriter(folder)
    for step, numerator, denominator in ((1, 2, 10), (2, 8, 20), (3, 3, 10), (4, 9, 20), (5, 4, 10)):
        writer.add_scalar(PREFIX + 'AscentByTread/T45/full_chain_episodes', numerator, step)
        writer.add_scalar(PREFIX + 'AscentByTread/T45/episodes', denominator, step)
        writer.add_scalar(PREFIX + 'Ascent/B1_episodes', numerator, step)
        writer.add_scalar(PREFIX + 'Ascent/full_episodes', denominator, step)
        writer.add_scalar('Episode_Reward/track_lin_vel_xy_exp', step, step)
        writer.add_scalar(PREFIX + 'AscentByLevel/L2/valid_events', denominator, step)
        writer.add_scalar(PREFIX + 'AscentByLevel/L2/ordered_events', numerator, step)
        writer.add_scalar(PREFIX + 'AscentByLevel/L2/frames', denominator * 10, step)
        writer.add_scalar(PREFIX + 'AscentByLevel/L2/skip_events', 1, step)
        writer.add_scalar(PREFIX + 'AscentByLevel/L2/speed_sum', denominator * 5, step)
    writer.flush(); writer.close()
    path = build_report(folder, 'interrupted')
    report = json.loads(Path(folder, 'v135_training_report.json').read_text())
    result = report['metrics']['上楼 B1 到达率']['all']
    assert abs(report['metrics']['tread T45：完整严格链']['all']['value'] - 26/70) < 1e-7
    assert result['numerator'] == 26 and result['denominator'] == 70
    assert abs(result['value'] - 26/70) < 1e-7
    assert report['metrics']['上楼 B2 严格前缀到达率']['all']['value'] is None
    assert report['status'] == 'interrupted' and report['logging_points'] == 5
    assert abs(report['metrics']['terrain level L2：交替/合法完成']['all']['value'] - 26/70) < 1e-7
    assert '未采集/分母为零' in path.read_text()
    print('V135_TRAINING_REPORT: PASS')

# Installed RSL Logger averages per-step extras. Check the exact count conversion.
from types import SimpleNamespace
import importlib.util
import torch
module_path = Path(__file__).resolve().parents[2] / 'source/rl_training/rl_training/v135_behavior_metrics.py'
spec = importlib.util.spec_from_file_location('v135_behavior_metrics', module_path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
env = SimpleNamespace(num_envs=2, device='cpu', extras={'log': {}},
    cfg=SimpleNamespace(v135_longtrain=SimpleNamespace(rollout_steps_per_update=24),
        scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=SimpleNamespace(num_rows=2)))))
metrics = module.V135BehaviorMetrics(env)
per_step = []
for event_count in [2] + [0] * 22 + [1]:
    metrics.sums = {}
    metrics.add('Ascent/skip_events', torch.tensor([event_count]))
    env.cfg.v135_riser_identity = SimpleNamespace(ascent_tread_widths=(.30, .30, .25, .35, .40, .45))
    metrics.publish()
    per_step.append(float(env.extras['log'][module.V135BehaviorMetrics.PREFIX + 'Ascent/skip_events']))
assert sum(per_step) / 24 == 3
assert module.V135BehaviorMetrics.PREFIX + 'Ascent/full_episodes' in env.extras['log']
print('V135_LOGGER_COUNT_CONVERSION: PASS')
