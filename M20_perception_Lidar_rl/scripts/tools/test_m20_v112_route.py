"""CPU reward-conflict and credit-replay controls; production helper functions."""
import ast, importlib.util, sys, types
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2]
mdp=root/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp'
pkg=types.ModuleType('route_test');pkg.__path__=[str(mdp)];sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location('route_test.v112_handoff',mdp/'v112_handoff.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
tree=ast.parse((mdp.parent/'config/wheeled/deeprobotics_m20/v112_variable_tread_env_cfg.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='M20V112HandoffCfg');cfg=types.SimpleNamespace(**{n.target.id:ast.literal_eval(n.value) for n in cls.body if isinstance(n,ast.AnnAssign)})
q=m.route_quality(torch.tensor([0.,.2,.5,1.]),torch.zeros(4),cfg)
assert q[0]==q[1]==1 and 0<q[2]<1 and q[3]==0
best=torch.tensor([-float('inf')]);yes=torch.tensor([True]);no=~yes
paid,best=m.once_per_riser(torch.tensor([1.]),yes,no,best,no,.08);assert paid
paid,best=m.once_per_riser(torch.tensor([1.]),yes,no,best,no,.08);assert not paid
paid,best=m.once_per_riser(torch.tensor([1.2]),yes,yes,best,no,.08);assert not paid
paid,best=m.once_per_riser(torch.tensor([1.2]),yes,no,best,no,.08);assert paid
paid,best=m.once_per_riser(torch.tensor([1.]),yes,no,best,yes,.08);assert paid
# Full route lifecycle: missing stair detections cannot release route quality;
# only platform confirmation AND rear wheels beyond the mesh's last edge can.
native=types.ModuleType('route_test.v11_continuous');sys.modules[native.__name__]=native
geo=types.ModuleType('route_test.v135_terrain_geometry');sys.modules[geo.__name__]=geo
geo.v135_terrain_geometry=lambda env:dict(radii=torch.ones(1,2),flight_riser_count=torch.tensor([4]),step_width=torch.tensor([.3]),ascent=torch.tensor([True]))
landing=dict(platform_confirmed=torch.tensor([False]),bilateral_landing_event=torch.tensor([False]),cycle_event=torch.tensor([False]),adjacent_credit_blocked=torch.tensor([False]),reference_riser_world_xy=torch.tensor([[1.,0.]]))
native.v11_state=lambda env:{'landing':landing}
native._v11_ascending_stair_forward_mask=lambda env:torch.tensor([True])
class Scene(dict): pass
scene=Scene(robot=types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.tensor([[0.,1.,0.]]),root_quat_w=torch.tensor([[1.,0,0,0.]]),body_pos_w=torch.zeros(1,2,3))))
scene.env_origins=torch.zeros(1,3)
env=types.SimpleNamespace(common_step_counter=1,num_envs=1,device='cpu',episode_length_buf=torch.tensor([1]),_v11_rear_body_ids=[0,1],scene=scene,cfg=types.SimpleNamespace(v112_handoff=cfg,v11_perception=types.SimpleNamespace(encounter_riser_switch_distance=.08)),command_manager=types.SimpleNamespace(get_term=lambda _:types.SimpleNamespace(v11_path_heading_w=torch.zeros(1),v11_path_origin_w=torch.zeros(1,2))))
assert m.route_state(env)['quality']==0
landing['platform_confirmed'][:]=True;env.common_step_counter+=1
assert m.route_state(env)['quality']==0, 'mid-flight false platform released route'
scene['robot'].data.body_pos_w[:,:,0]=2.;env.common_step_counter+=1
assert m.route_state(env)['quality']==0, 'offroute bypass released route at endpoint'
scene['robot'].data.root_pos_w[:,1]=0.;env.common_step_counter+=1
assert m.route_state(env)['quality']==1, 'physical final exit never released'
scene['robot'].data.root_pos_w[:,1]=1.;env.common_step_counter+=1
assert m.route_state(env)['quality']==1, 'flat turn reactivated completed stair route'
landing['platform_confirmed'][:]=False;env.common_step_counter+=1;env.episode_length_buf.zero_()
assert m.route_state(env)['quality']==0, 'reset retained completion'
# Execute reward wrappers with native helpers isolated; test command untouched.
rewards=types.ModuleType('route_test.rewards');sys.modules[rewards.__name__]=rewards
native=types.ModuleType('route_test.v11_continuous');sys.modules[native.__name__]=native
v=types.SimpleNamespace(root_lin_vel_b=torch.tensor([[0.,0.,0.],[.7,0.,0.]]))
command=torch.tensor([[.7,0.,0.],[.7,0.,0.]])
env=types.SimpleNamespace(scene={'robot':types.SimpleNamespace(data=v)})
rewards.track_lin_vel_xy_exp=lambda env,**kw:torch.exp(-(command[:,:2]-v.root_lin_vel_b[:,:2]).square().sum(1)/kw['std']**2)
native.v11_world_forward_progress_reward=lambda env,**kw:torch.tensor([0.,1.])
m.route_state=lambda env:{'quality':torch.ones(2)}
m._factors=lambda env:(torch.ones(2),torch.zeros(2,dtype=torch.bool))
assert m.track_velocity(env,.5,'base_velocity')[1]>m.track_velocity(env,.5,'base_velocity')[0]
m._factors=lambda env:(torch.full((2,),.25),torch.ones(2,dtype=torch.bool))
assert m.track_velocity(env,.5,'base_velocity')[0]>m.track_velocity(env,.5,'base_velocity')[1], 'dwell still asks for forward speed'
assert torch.equal(m.forward_progress(env),torch.zeros(2))
m.route_state=lambda env:{'quality':torch.zeros(2)}
assert torch.equal(m.track_velocity(env,.5,'base_velocity'),torch.zeros(2))
assert torch.equal(command,torch.tensor([[.7,0.,0.],[.7,0.,0.]]))
print('V112_ROUTE_NORMAL_BYPASS_HOLD_FORWARD_CONFLICT_AND_ONCE_CREDIT: PASS')
