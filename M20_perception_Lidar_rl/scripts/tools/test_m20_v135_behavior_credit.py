"""Adversarial payment checks, independent of Isaac imports."""
from pathlib import Path
import importlib.util
import torch
ROOT = Path(__file__).resolve().parents[2] / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp'
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module
credit = load('v135_forward_credit').bounded_forward_credit
quality = load('v135_whole_body_descent').workspace_quality
# An arbitrarily long episode cannot mine more than one tread of distance.
best = torch.zeros(4); remaining = torch.full((4,), .3); paid = torch.zeros(4)
for t in range(200):
    x = torch.tensor([t*.01, 0., .1 if t % 2 else 0., t*.01])
    q = torch.tensor([1., 1., 1., 0.])
    best, remaining, amount = credit(x,best,remaining,q,torch.ones(4,dtype=torch.bool),speed=torch.ones(4),dt=.02)
    paid += amount*.02
assert torch.allclose(paid, torch.tensor([.3,0.,.02,0.]),atol=1e-5), paid
# Bad gait spent its budget; changing quality later cannot reclaim history.
_, _, retry = credit(best,best,remaining,torch.ones(4),torch.ones(4,dtype=torch.bool),speed=torch.ones(4),dt=.02)
assert not retry.any()
# Same leg length, wrong direction: old scalar-length test cannot distinguish this.
neutral = torch.tensor([[[0.,0.,-.4],[0.,0.,-.4]]])
normal = quality(neutral,neutral,free=.35,scale=.35)
extended = quality(torch.tensor([[[-.4,0.,0.],[.4,0.,0.]]]),neutral,free=.35,scale=.35)
folded = quality(neutral*.1,neutral,free=.35,scale=.35)
assert (normal == 1).all() and (extended < .1).all() and (folded < .1).all()
assert torch.isfinite(extended).all()
print('V135_BEHAVIOR_CREDIT: PASS (bounded, stationary, backtrack, bad-gait budget, signed workspace)')

exit_test = load('v135_exit_geometry').platform_exit_evidence
wheels = torch.tensor([[[.3,0.],[.3,0.],[.2,0.],[.2,0.]]]).repeat(5,1,1)
wheels[1,2,0] = -.1
hits = torch.tensor([[[.4,0.,0.],[.8,0.,0.]]]).repeat(5,1,1)
hits[2] = float('nan')
hits[3,1,2] = .2
seen = torch.tensor([True,True,True,True,False])
result = exit_test(wheels,torch.zeros(5,2),seen,torch.tensor([[.3,0.]]).repeat(5,1),
    torch.tensor([[1.,0.]]).repeat(5,1),hits,torch.ones(5,dtype=torch.bool),
    radius=.1,plane_tolerance=.05,lookahead=.6,half_width=.25)
assert result.tolist() == [True,False,False,False,False], result
print('V135_EXIT_GEOMETRY: PASS (platform, rear still on stair, missing rays, next riser, no flight)')
