"""Reward discrimination on real frozen trajectories plus geometry counterexamples."""
import ast,importlib.util,sys,types,json
from pathlib import Path
import torch,numpy as np
root=Path(__file__).resolve().parents[2];mdp=root/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp'
pkg=types.ModuleType('workspace_test');pkg.__path__=[str(mdp)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location('workspace_test.v112_flat_posture',mdp/'v112_flat_posture.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
p=root/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/v112_variable_tread_env_cfg.py';tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='M20V112HandoffCfg');cfg=types.SimpleNamespace(**{n.target.id:ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.AnnAssign)})
reach=torch.zeros(2,4);delta=torch.zeros(2,8);assert torch.equal(mod.workspace_cost(reach,delta,cfg),torch.zeros(2))
reach[1,0]=.20;assert mod.workspace_cost(reach,delta,cfg)[1]>1
# Equal displacements of root and all bodies must not change body-relative cost.
q=torch.tensor([[1.,0,0,0]]);hip=torch.tensor([[[.314,.2,0],[.314,-.2,0],[-.314,.2,0],[-.314,-.2,0]]]);wheel=hip+torch.tensor([[[0.,0,-.48]]]);positions=torch.cat((wheel,hip),1)
robot=types.SimpleNamespace(data=types.SimpleNamespace(root_quat_w=q,body_pos_w=positions,joint_pos=torch.zeros(1,8),default_joint_pos=torch.zeros(1,8)))
base={'step':0,'cost':torch.zeros(1),'blend':torch.ones(1),'active':torch.ones(1,dtype=torch.bool)}
mod.base_flat_state=lambda env:base
env=types.SimpleNamespace(common_step_counter=0,cfg=types.SimpleNamespace(v112_handoff=cfg),scene={'robot':robot},_v112_workspace_ids=([0,1,2,3],[4,5,6,7],list(range(8))))
assert mod.level_posture_penalty(env)==0
robot.data.body_pos_w[:,0,0]+=.2;env.common_step_counter+=1;base['step']+=1
bad=mod.level_posture_penalty(env).clone();assert bad < -1
robot.data.body_pos_w+=100;env.common_step_counter+=1;base['step']+=1;assert torch.allclose(mod.level_posture_penalty(env),bad,atol=1e-4)
base['blend']=torch.zeros(1);env.common_step_counter+=1;base['step']+=1;assert mod.level_posture_penalty(env)==0,'stair pose constrained despite disabled flat gate'
base['blend']=torch.ones(1)
for step in (100,10000,1000000):env.common_step_counter=step;assert mod.level_posture_penalty(env)<-1
# Real recorded forward rollout: new term must recognize extension old height missed.
a=root/'artifacts/v112_crawl_diagnosis_20260927';d=np.load(a/'native_forward.npz');meta=json.loads((a/'native_forward.json').read_text());mask=(d['terrain_type']<8)&(d['episode_length']>100)&(d['command'][...,0]>.6)
q=torch.from_numpy(d['quat'][mask]);f=torch.stack((1-2*(q[:,2]**2+q[:,3]**2),2*(q[:,1]*q[:,2]+q[:,0]*q[:,3]),2*(q[:,1]*q[:,3]-q[:,0]*q[:,2])),1)
wi=[meta['body_names'].index(l+'_wheel') for l in ('fl','fr','hl','hr')];hi=[meta['body_names'].index(l+'_hipy') for l in ('fl','fr','hl','hr')];b=torch.from_numpy(d['body'][mask]);reach=((b[:,wi]-b[:,hi])*f[:,None]).sum(-1)
ji=[1,2,4,5,7,8,10,11];delta=torch.from_numpy(d['q'][mask][:,ji])-torch.tensor(meta['nominal'])[ji]
cost=mod.workspace_cost(reach,delta,cfg);assert cost.mean()>.1
report={'real_forward_samples':len(cost),'workspace_cost_mean':float(cost.mean()),'workspace_cost_positive_fraction':float((cost>0).float().mean()),'old_height_penalty_zero_fraction':float((d['clearance'][mask]>=.4).mean())};print(report)
(root/'artifacts/v112_pose_repair_20260927/workspace_discrimination.json').write_text(json.dumps(report,indent=2))
print('WORKSPACE_NORMAL_EXTENSION_ORIGIN_GATE_PERSISTENCE_REAL_TRAJECTORY: PASS')

cfg.flat_workspace_enabled=False
for key in ("workspace_cost","reach_max","joint_deviation","extension_rate"):
 assert mod.level_metric(env,slice(None),key)==0
cfg.flat_workspace_enabled=True
try:mod.workspace_cost(torch.full((1,4),float("nan")),torch.zeros(1,8),cfg)
except RuntimeError:pass
else:raise AssertionError("invalid pose silently accepted")
print("DISABLED_METRICS_AND_NONFINITE_FAIL_CLOSED: PASS")
