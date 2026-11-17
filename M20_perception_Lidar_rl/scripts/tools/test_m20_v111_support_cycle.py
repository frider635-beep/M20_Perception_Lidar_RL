"""CPU positive/negative trajectories for the actual V11.1 timing functions."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity'
spec = importlib.util.spec_from_file_location('cycle', BASE / 'mdp/v111_support_cycle.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tree = ast.parse((BASE / 'config/wheeled/deeprobotics_m20/v111_v11fd_rear_posture_env_cfg.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'M20V111RearPostureCfg')
cfg = NS(**{n.target.id: ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.AnnAssign)})

def step(s, upper, lower, active=True, dt=.02):
    return m.cycle_step(s, torch.tensor([upper]), torch.tensor([lower]), torch.tensor([active]), dt, cfg)

for side in (0, 1):
    s = {}; up = [False, False]; down = [True, True]
    up[side] = True; down[side] = False
    total = 0.
    for _ in range(8):
        step(s, up, down); total += s['cycle_delta'].item()
    assert s['cycle_ready'].item() and not s['cycle_done'].item()
    assert abs(total - .35) < 1e-6
    # The follower may swing only after the leader has dwelled.
    for _ in range(4): step(s, up, [False, False])
    for i in range(6):
        step(s, [True, True], [False, False]); total += s['cycle_delta'].item()
        assert s['cycle_done'].item() == (i == 5)
    assert abs(total - 1.) < 1e-6
    for _ in range(30):
        step(s, [True, True], [False, False])
        assert s['cycle_delta'].item() == 0 and not s['cycle_event'].item()

# Near-simultaneous contact cannot substitute for a leader dwell.
s = {}; step(s, [True, False], [False, True])
for _ in range(30): step(s, [True, True], [False, False])
assert s['cycle_bad'].item() and not s['cycle_done'].item()
# Simultaneous landing: no arbitrary side selection.
s = {}
for _ in range(30): step(s, [True, True], [False, False])
assert s['cycle_leader'].item() == -1 and s['cycle_best'].item() == 0
assert s['cycle_bad'].item(), 'simultaneous pair must permit failed-cycle physical resync'
# Leader leaves while follower swings: cannot settle later by returning.
s = {}
for _ in range(8): step(s, [True, False], [False, True])
for _ in range(2): step(s, [False, False], [False, False])
for _ in range(20): step(s, [True, True], [False, False])
assert s['cycle_bad'].item() and not s['cycle_done'].item()
# Waiting cannot farm a reward or indefinitely suspend speed-deficit cost.
s = {}
total = 0
for _ in range(100):
    step(s, [True, False], [False, True]); total += s['cycle_delta'].item()
assert total <= .350001 and not s['cycle_holding'].item()
# Timing is in seconds: a different control rate requires more frames.
s = {}
for _ in range(15): step(s, [True, False], [False, True], dt=.01)
assert not s['cycle_ready'].item()
step(s, [True, False], [False, True], dt=.01)
assert s['cycle_ready'].item()

# Plane evidence must reject stairs, missing rays and slopes, independent of pose.
wheel = torch.tensor([[[.3,.2,.09],[.3,-.2,.09],[-.3,.2,.09],[-.3,-.2,.09]]])
root = torch.tensor([[0.,0.,.4]])
hits = torch.tensor([[[.3,.2,0.],[.3,-.2,0.],[-.3,.2,0.],[-.3,-.2,0.],[.05,0.,0.],[.55,0.,0.]]])
forward = torch.tensor([[1.,0.]])
def flat(h, no=True): return m.flat_geometry(wheel,root,h,forward,torch.tensor([no]),cfg)[0].item()
assert flat(hits) and not flat(hits,False)
missing = hits.clone(); missing[:] = torch.nan
assert not flat(missing)
stepped = hits.clone(); stepped[:,0:2,2] = .18
assert not flat(stepped)
slope = hits.clone(); slope[...,2] = slope[...,0] * .2
assert not flat(slope)
cost = m.flat_posture_cost(torch.tensor([.4,.30,.4,.4]),torch.tensor([0.,0.,.2,0.]),torch.tensor([0.,0.,0.,1.3]),cfg)
assert cost[0] == 0 and (cost[1:] > 0).all()
print('V111_TIMED_CYCLE_CPU: PASS (left/right, early follow, simultaneous, lost support, bounded credit, dt)')
print('V111_LEVEL_POSTURE_CPU: PASS (flat, stair, slope, missing rays, crouch, tilt, rocking)')
