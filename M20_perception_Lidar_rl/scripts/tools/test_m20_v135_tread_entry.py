"""Mixed-tread identity, measured swing, COM support and no-reward-farming contracts."""
import importlib.util
import sys
import types
from copy import deepcopy
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[2]
mdp = root / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp'
pkg = types.ModuleType('tread_fixture'); pkg.__path__ = [str(mdp)]; sys.modules[pkg.__name__] = pkg
def load(name):
    spec = importlib.util.spec_from_file_location('tread_fixture.' + name, mdp / (name + '.py'))
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module
geo = load('v135_terrain_geometry'); contract = load('v135_ascent_contract')
ledger = load('v135_riser_identity'); support = load('v135_ascent_support')
NS = types.SimpleNamespace
class Cfg(NS):
    def copy(self): return deepcopy(self)
stair = Cfg(proportion=.3, step_width=.3, border_width=1., platform_width=3., holes=False,
            function=NS(__name__='inverted_pyramid_stairs_terrain'))
cfg = Cfg(size=(8.,8.), curriculum=True, num_cols=20,
    sub_terrains={'flat':Cfg(proportion=.2), 'pyramid_stairs':Cfg(proportion=.2,step_width=.3),
                 'pyramid_stairs_inv':stair, 'other':Cfg(proportion=.3)})
before=geo.column_names(cfg)
widths=(.3,.3,.25,.35,.4,.45)
geo.configure_ascent_treads(cfg,widths)
after=geo.column_names(cfg)
assert before[:8] == after[:8] and before[14:] == after[14:]
env=NS(device='cpu',cfg=NS(scene=NS(terrain=NS(terrain_generator=cfg))),
       scene=NS(terrain=NS(terrain_types=torch.arange(20))))
g=geo.v135_terrain_geometry(env)
assert torch.allclose(g['step_width'][8:14],torch.tensor(widths))
assert g['ascent'].nonzero().flatten().tolist()==list(range(8,14))
assert g['flight_riser_count'][8:14].tolist()==[7,7,8,6,5,5]
# Mixed widths in the SAME vector batch, with all four stair headings.
width=torch.tensor(widths).repeat_interleave(4)
n=len(width); heading=torch.arange(4).repeat(6)*(.5*torch.pi)
radii=torch.stack([torch.tensor(contract.inverted_flight_layout((8,8),1,3,float(w))[0]) for w in width])
origin=torch.zeros(n,2); root_xy=torch.zeros(n,2)
anchor,fw=contract.flight_anchor(root_xy,origin,heading,radii)
edges=anchor[:,None]+torch.arange(3)[None,:,None]*width[:,None,None]*fw[:,None]
match=ledger.v135_match_locked_target_edges(edges,torch.ones(n,3,dtype=torch.bool),root_xy,anchor,fw,
    step_width=width,longitudinal_tolerance=.08,lateral_tolerance=.15)
assert match['valid'].all() and match['slot'].tolist()==[[0,1]]*n
# At all widths a wheel fully inside B1 may settle; beyond B1 must never settle.
x=(.1+(.5*(width-.2)).clamp(max=.05)+.005)[:,None].expand(-1,2)
z=torch.zeros_like(x); active=torch.ones(n,dtype=torch.bool)
kw=dict(wheel_radius=.1,front_margin=.05,step_width=width,clearance_margin=.02,
        support_tolerance=.04,contact_threshold=20.,lateral_tolerance=.5,confirmation_frames=3)
_,_,ready=ledger.v135_locked_target_completion_step(torch.ones_like(x,dtype=torch.bool),
    torch.full_like(x,2,dtype=torch.long),x,x,z,z,torch.full_like(x,30),torch.zeros(n),active,**kw)
assert ready.all(), 'short tread has an empty support interval'
_,_,ready=ledger.v135_locked_target_completion_step(torch.ones_like(x,dtype=torch.bool),
    torch.full_like(x,2,dtype=torch.long),x,width[:,None].expand(-1,2),z,z,torch.full_like(x,30),torch.zeros(n),active,**kw)
assert not ready.any(), 'B2 aliased to B1'
# Geometry depends on column, not stale env initialization or curriculum level.
env.scene.terrain.terrain_types=torch.tensor([13,8,10])
assert torch.allclose(geo.v135_terrain_geometry(env)['step_width'],torch.tensor([.45,.3,.25]))
# Equal progress / no swing must never invent a FL support set.
contact=torch.tensor([[1,1,1,1],[0,1,1,1],[1,0,1,1],[0,0,1,1]],dtype=torch.bool)
bottom=torch.tensor([[0.,0.,0.,0.],[.1,0.,0.,0.],[0.,.1,0.,0.],[.1,.1,0.,0.]])
side=support.measured_swing_side(contact,bottom,torch.zeros(4),torch.full((4,),-1),torch.zeros(4,dtype=torch.bool),lift_min=.03)
assert side.tolist()==[-1,0,1,-1]
# Same contacts but COM outside => reduced quality; cardinal rotation invariant.
points=torch.tensor([[[.4,.25],[.4,-.25],[-.4,.25],[-.4,-.25]]]).repeat(3,1,1)
contacts=torch.ones(3,4,dtype=torch.bool)
com=torch.tensor([[0.,0.],[0.,.5],[.0,.0]])
contacts[2,:2]=False
margin,valid=support.support_margin(points,contacts,com)
assert margin[0]>0 and margin[1]<0 and valid.tolist()==[True,True,False]
rot=torch.tensor([[0.,-1.],[1.,0.]])
m2,v2=support.support_margin(points@rot,contacts,com@rot)
assert torch.allclose(margin,m2) and torch.equal(valid,v2)
# Three collinear contacts cannot certify balance.
line=torch.zeros(1,4,2);line[0,:,0]=torch.arange(4)
assert not support.support_margin(line,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,2))[1].any()
print('V135_MIXED_TREAD_AND_ENTRY_CONTRACT: PASS')

# Run the actual installed mesh generator: inspect physical vertical faces,
# independently of the ledger's layout arithmetic (not a mirrored oracle).
import numpy as np
import inspect
import __editable___isaaclab_0_54_4_finder as finder
mesh_dir=Path(finder.MAPPING['isaaclab'])/'terrains/trimesh'
mesh_pkg=types.ModuleType('actual_isaac_mesh');mesh_pkg.__path__=[str(mesh_dir)];sys.modules[mesh_pkg.__name__]=mesh_pkg
spec=importlib.util.spec_from_file_location('actual_isaac_mesh.mesh_terrains',mesh_dir/'mesh_terrains.py')
actual_mesh=importlib.util.module_from_spec(spec);sys.modules[spec.name]=actual_mesh;spec.loader.exec_module(actual_mesh)
for width in sorted(set(widths)):
    mesh_cfg=NS(size=(8.,8.),border_width=1.,platform_width=3.,step_width=width,
                step_height_range=(.10,.24),holes=False)
    meshes, center=actual_mesh.inverted_pyramid_stairs_terrain(.5,mesh_cfg)
    physical=[]
    # Union-of-box surface sampled along the midline, directly from mesh bounds.
    samples=np.linspace(center[0],7.9,10001)
    surface=np.full_like(samples,-np.inf)
    for mesh in meshes:
        bounds=mesh.bounds
        inside=(samples>=bounds[0,0])&(samples<=bounds[1,0])&(center[1]>=bounds[0,1])&(center[1]<=bounds[1,1])
        surface[inside]=np.maximum(surface[inside],bounds[1,2])
    changes=np.where(np.diff(surface)>.05)[0]
    edges=(samples[changes]+samples[changes+1])*.5-center[0]
    rad,count=contract.inverted_flight_layout(mesh_cfg.size,1,3,width)
    assert len(edges)==count, (width,edges,count)
    assert np.allclose(edges,rad[0]+np.arange(count)*width,atol=.001), (width,edges,rad)
print('V135_ACTUAL_MESH_BOUNDARIES: PASS',inspect.getfile(actual_mesh.inverted_pyramid_stairs_terrain))

# Full support-state calls: cache/idempotency, preparation, measured swing,
# acceptance and target/reset isolation without invoking a second gait FSM.
whole_mod=types.ModuleType('tread_fixture.v135_whole_body_descent')
whole=dict(bodies=[0,1,2,3],contacts=[0,1,2,3],workspace=torch.ones(1,4),
           pitch_error=torch.zeros(1),ground_valid=torch.ones(1,dtype=torch.bool))
whole_mod.v135_whole_body_descent_state=lambda e,sensor_cfg=None:whole
sys.modules[whole_mod.__name__]=whole_mod
snapshot=dict(target_id=torch.ones(1,dtype=torch.long),target_valid=torch.ones(1,dtype=torch.bool),
              target_height_valid=torch.ones(1,dtype=torch.bool),target_lower_z=torch.zeros(1),
              first_swing_side=torch.tensor([-1]),seeded=torch.tensor([False]),
              forward_world_xy=torch.tensor([[1.,0.]]),locked_crossed=torch.zeros(1,2,dtype=torch.bool),locked_support_steps=torch.zeros(1,2,dtype=torch.long))
ledger.v135_riser_identity_state=lambda e,sensor_cfg=None:dict(reward_snapshot=snapshot)
class Scene(dict):pass
scene=Scene()
scene.terrain=NS(terrain_types=torch.tensor([8]))
forces=torch.zeros(1,4,3);forces[:,:,2]=30
scene.sensors={'contact_forces':NS(data=NS(net_forces_w=forces))}
wheels=torch.cat((points[:1],torch.full((1,4,1),.1)),2)
data=NS(body_pos_w=wheels,body_lin_vel_w=torch.zeros(1,4,3),body_ang_vel_w=torch.zeros(1,4,3),
        body_com_pos_w=torch.tensor([[[0.,0.,.4]]]),root_quat_w=torch.tensor([[1.,0.,0.,0.]]),
        root_ang_vel_w=torch.zeros(1,3))
scene['robot']=NS(data=data,root_physx_view=NS(get_masses=lambda:torch.tensor([[40.]])))
env=NS(num_envs=1,device='cpu',common_step_counter=0,episode_length_buf=torch.zeros(1,dtype=torch.long),scene=scene,
       cfg=NS(scene=NS(terrain=NS(terrain_generator=cfg)),v135_riser_identity=NS(support_preparation_frames=5,
           support_margin_scale=.1,first_swing_lift_min=.03),v11_perception=NS(contact_force_threshold=20.,
           wheel_radius=.1,front_support_confirmation_frames=3)))
def tick():
    env.common_step_counter+=1;env.episode_length_buf+=1
    return support.v135_ascent_support_state(env)
for _ in range(5):state=tick()
assert state['side'].item()==-1 and state['prep_count'].item()==5 and state['settlement_quality'].item()==0
forces[0,0,2]=0;wheels[0,0,2]=.2;snapshot['first_swing_side'][0]=0
state=tick();assert state['swing'].item() and state['side'].item()==0
count=state['swing_count'].clone()
assert support.v135_ascent_support_state(env) is state and torch.equal(state['swing_count'],count)
# Double-front lift is poor support, not an omitted frame.
forces[0,1,2]=0;wheels[0,1,2]=.2
state=tick();assert state['swing'].item() and state['quality'].item()==0
forces[:,:,2]=30;wheels[:,:,2]=.1;snapshot['locked_crossed'][0,0]=True;snapshot['locked_support_steps'][0,0]=1
for _ in range(3):state=tick()
q=state['settlement_quality'].clone();assert 0<q.item()<1
for _ in range(20):state=tick()
assert torch.equal(state['settlement_quality'],q), 'waiting diluted poor swing quality'
snapshot['target_id']+=1;snapshot['seeded'][0]=True;snapshot['locked_crossed'][:]=False
state=tick();assert state['side'].item()==-1 and state['settlement_quality'].item()==0
print('V135_ENTRY_STATE_IDEMPOTENCY_AND_NO_WAIT_FARMING: PASS')

# Preparation predates strict acquisition; acquisition must not erase it.
del env._v135_ascent_support
snapshot['target_id'][:]=0;snapshot['seeded'][:]=False
snapshot['target_valid'][:]=False;snapshot['target_height_valid'][:]=False
snapshot['first_swing_side'][:]=-1;snapshot['locked_support_steps'][:]=0
forces[:,:,2]=30;wheels[:,:,2]=.1
for _ in range(5): state=tick()
assert state['prep_count'].item()==5
snapshot['target_id'][:]=1;snapshot['target_valid'][:]=True;snapshot['target_height_valid'][:]=True
forces[0,0,2]=0;wheels[0,0,2]=.2
state=tick();assert state['prep_quality'].item()>0
# Legal alternating landing: landed FL bears load while FR begins its swing.
forces[0,0,2]=30;forces[0,1,2]=0
wheels[0,0,2]=.1;wheels[0,1,2]=.2
snapshot['locked_support_steps'][0,0]=1
state=tick()
assert state['acceptance'].item()
assert state['settlement_evidence'].item()
assert state['settlement_quality'].item()>0
# Missing preparation is explicit and cannot earn certified quality.
del env._v135_ascent_support
forces[0,0,2]=0;forces[0,1,2]=30
wheels[0,0,2]=.2;wheels[0,1,2]=.1
snapshot['locked_support_steps'][:]=0
state=tick();assert state['missing_preparation'].item()
assert not state['settlement_evidence'].item()
assert state['settlement_quality'].item()==0
print('V135_LATE_ACQUISITION_AND_ALTERNATING_ACCEPTANCE: PASS')
