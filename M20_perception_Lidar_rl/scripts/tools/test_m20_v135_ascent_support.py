"""Three-wheel support and lateral edge identity regression, CPU only."""
import importlib.util
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2]/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp'
def load(name):
 spec=importlib.util.spec_from_file_location(name,root/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
q=load('v135_ascent_support').three_wheel_quality
f=torch.tensor([[0.,30.,30.,30.],[30.,0.,30.,30.],[0.,30.,0.,30.],[0.,0.,30.,30.]])
z=torch.zeros(4);side=torch.tensor([0,1,0,0]);valid=torch.ones(4,dtype=torch.bool)
a,b=q(f,torch.ones_like(f),side,z,z,z,z,valid)
assert torch.allclose(a,torch.tensor([1.,1.,0.,0.]))
assert q(f,torch.ones_like(f),side,z,z,z,z,~valid)[0].eq(0).all()
work=torch.ones_like(f);work[0,0]=0 # swing workspace should not be a support condition
assert q(f,work,side,z,z,z,z,valid)[0][0]==1
work[0,2]=.1
assert q(f,work,side,z,z,z,z,valid)[0][0]<.2
m=load('v135_riser_identity')
for forward in ([1.,0.],[0.,1.],[-1.,0.],[0.,-1.]):
 fw=torch.tensor([forward]);lat=torch.stack((-fw[:,1],fw[:,0]),1)
 target=fw*1.2;edges=target[:,None,:]+lat[:,None,:]*.4+fw[:,None,:]*torch.tensor([[[0.],[.3],[.6]]])
 kw=dict(step_width=.3,longitudinal_tolerance=.08,lateral_tolerance=.15)
 args=(edges,torch.ones(1,3,dtype=torch.bool),torch.zeros(1,2),target,fw)
 assert not m.v135_match_locked_target_edges(*args,**kw)['valid'].any()
 out=m.v135_match_locked_target_edges(*args,**kw,face_origin_xy=torch.zeros(1,2),face_half_width=torch.tensor([[1.2,1.5]]))
 assert out['valid'].all() and out['slot'].tolist()==[[0,1]]
 out=m.v135_match_locked_target_edges(edges+lat[:,None,:]*2,*args[1:],**kw,face_origin_xy=torch.zeros(1,2),face_half_width=torch.tensor([[1.2,1.5]]))
 assert not out['valid'].any()
print('V135_ASCENT_THREE_WHEEL_SUPPORT_AND_EDGE_MATCH: PASS')

nonfinite=torch.full((4,),float("nan"))
assert q(f,torch.ones_like(f),side,nonfinite,z,z,z,valid)[0].eq(0).all()
print("V135_ASCENT_NONFINITE_NO_CREDIT: PASS")
