"""Real ledger/shaping wiring with deterministic physical event fixtures."""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
import torch

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"
package = types.ModuleType("ascent_fixture")
package.__path__ = [str(MDP)]
sys.modules[package.__name__] = package
def load(name):
    path = MDP / (name + ".py")
    spec = importlib.util.spec_from_file_location("ascent_fixture." + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

contract = load("v135_ascent_contract")
ledger = load("v135_riser_identity")
shaping = load("v135_final_ascent_reward")
v11 = types.ModuleType("ascent_fixture.v11_continuous")
v11.v11_state = lambda env, **kw: env.base
sys.modules[v11.__name__] = v11

# Regression: far-away lift and near-edge vertical-only lift receive no credit.
kw = dict(crossing_span=.1, step_width=.3, wheel_radius=.09,
          lateral_tolerance=.15, confirmation_frames=3)
z = torch.zeros(1, 2)
for x, advance in ((-.7, .1), (-.02, 0.)):
    p = shaping.v135_locked_front_potential(torch.ones(1, 2), z.bool(), z.long(),
        torch.full((1, 2), x), z, torch.tensor([True]), swing_advance=torch.full((1, 2), advance), **kw)
    assert p.eq(0).all(), (x, p)

def fixture(size=(8., 8.)):
    stair = NS(proportion=1., border_width=.5, platform_width=3., step_width=.3, holes=False,
               function=NS(__name__="inverted_pyramid_stairs_terrain"))
    terrain = NS(num_cols=2, curriculum=True, size=size, sub_terrains={"pyramid_stairs_inv": stair})
    radii, count = contract.inverted_flight_layout(terrain.size, .5, 3., .3)
    radius = radii[0]
    pc = NS(ascending_stair_terrain_type_ids=(1,), forward_command_threshold=.1,
            wheel_radius=.09, front_margin=.01, clearance_margin=.05,
            upper_support_tolerance=.04, contact_force_threshold=10., front_support_confirmation_frames=3)
    ic = NS(step_width=.3, longitudinal_tolerance=.08, lateral_tolerance=.15,
            b1_world_stability_tolerance=.03, b1_confirmation_frames=3,
            irrecoverable_behind_distance=-.3, overstep_margin=.12,
            ordered_speed_multiplier=1.2, ordinary_speed_multiplier=1., skip_speed_multiplier=.85,
            first_swing_lift_min=.03, first_swing_advance_min=.02)
    data = NS(root_pos_w=torch.tensor([[radius-.7, 0., .5]]),
              body_pos_w=torch.tensor([[[radius-.25, .226576, .09], [radius-.25, -.226576, .09]]]))
    sensor = NS(data=NS(net_forces_w=torch.tensor([[[0.,0.,50.], [0.,0.,50.]]])))
    class Scene(dict):
        pass
    scene=Scene(robot=NS(data=data));scene.sensors={"contact_forces":sensor}
    scene.terrain=NS(terrain_types=torch.tensor([1]));scene.env_origins=torch.zeros(1,3)
    command=NS(v11_path_heading_w=torch.tensor([.2])) # jitter must not rotate the ledger
    env=NS(num_envs=1, device="cpu", common_step_counter=0, episode_length_buf=torch.zeros(1,dtype=torch.long),
           step_dt=.02, cfg=NS(v11_perception=pc,v135_riser_identity=ic,scene=NS(terrain=NS(terrain_generator=terrain))),
           scene=scene, _v11_front_body_ids=[0,1], _v11_front_contact_ids=[0,1],
           command_manager=NS(get_command=lambda n:torch.tensor([[1.,0.,0.]]),get_term=lambda n:command))
    def target(k):
        edge=radius+(k-1)*.3
        pool=dict(valid=torch.ones(1,3,dtype=torch.bool),riser_height=torch.full((1,3),.15),
                  riser_lower_world=torch.tensor([[(k-1)*.15,k*.15,(k+1)*.15]]))
        env.base=dict(riser_pool=pool,riser_pool_world_xy=torch.tensor([[[edge,0.],[edge+.3,0.],[edge+.6,0.]]]),
                      landing=dict(target_active=torch.tensor([True]),reference_riser_world_xy=torch.tensor([[edge,0.]]),
                                   landing_event=torch.tensor([False]),candidate_ready=torch.zeros(1,2,dtype=torch.bool),
                                   responsible=torch.tensor([0])))
        return edge
    def tick():
        env.common_step_counter+=1;env.episode_length_buf+=1
        state=ledger.v135_riser_identity_state(env)
        progress=shaping.v135_final_ascent_state(env)
        return state,progress
    target(1)
    return env,target,tick,radius,count

for first, size in ((0,(8.,8.)), (1,(8.,8.)), (0,(6.,6.)), (1,(10.,10.))):
    env,target,tick,radius,count=fixture(size)
    for _ in range(3): s,p=tick()
    assert s["target_valid"].item() and torch.allclose(s["forward_world_xy"],torch.tensor([[1.,0.]]))
    for k in range(1,count+1):
        side=(first+k-1)%2;edge=target(k);wheel=env.scene["robot"].data.body_pos_w
        force=env.scene.sensors["contact_forces"].data.net_forces_w
        env.scene["robot"].data.root_pos_w[0,0]=edge-.5
        wheel[0,side]=torch.tensor([edge-.25, .226576 if side==0 else -.226576,(k-1)*.15+.09]);force[0,side,2]=50
        tick() # grounded reference in the new target coordinates
        wheel[0,side,0]=edge-.18;wheel[0,side,2]=(k-1)*.15+.18;force[0,side,2]=0
        s,p=tick()
        if k==1: assert s["first_lift_event"].item()
        wheel[0,side,0]=edge+.02;wheel[0,side,2]=k*.15+.15
        tick()
        wheel[0,side,0]=edge+.12;wheel[0,side,2]=k*.15+.09;force[0,side,2]=50
        for _ in range(3): s,p=tick()
        assert s["valid"].item(), (k,s["locked_support_steps"],s["first_swing_side"])
        assert s["target_id"].item()==k+1 and s["expected_side"].item()==1-side
        assert s["reward_snapshot"]["target_id"].item()==k
        assert abs(s["reward_snapshot"]["target_lower_z"].item()-(k-1)*.15)<1e-5
        assert p["target_id_last"].item()==k and p["best_potential"][0,side]>.8
        assert s["episode_ordered_valid"].item()
        if k < count:
            env.base["landing"]["landing_event"][:] = True
            env.base["landing"]["responsible"][:] = side
            s,p=tick()
            assert not s["duplicate"].item() and s["episode_ordered_valid"].item(), "late native touchdown"
            env.base["landing"]["landing_event"][:] = False
    assert s["flight_complete"].item() and s["strict_prefix_depth"].item()==count
    assert not s["target_valid"].item()

# B2 cannot be relabelled B1 if it is the first visible edge.
env,target,tick,radius,count=fixture();target(2)
for _ in range(5):s,p=tick()
assert not s["target_valid"].item() and s["b1_identity_rejected"].item()

# First swing returns to the lower floor: failure remains after later steps.
env,target,tick,radius,count=fixture()
for _ in range(3):tick()
wheel=env.scene["robot"].data.body_pos_w;force=env.scene.sensors["contact_forces"].data.net_forces_w
wheel[0,0,2]=.19;force[0,0,2]=0
tick();wheel[0,0,2]=.09;force[0,0,2]=50
for _ in range(3):s,p=tick()
assert s["first_lift_empty_event"].item() and not s["episode_ordered_valid"].item()
for _ in range(4):s,p=tick()
assert s["first_swing_failed"].item() and s["first_swing_side"].item()==0
# Simultaneous first swings cannot qualify as a strict prefix.
env,target,tick,radius,count=fixture()
for _ in range(3):tick()
wheel=env.scene["robot"].data.body_pos_w;force=env.scene.sensors["contact_forces"].data.net_forces_w
wheel[0,:,2]=.19;force[0,:,2]=0
s,p=tick()
assert s["first_swing_failed"].item() and not s["episode_ordered_valid"].item()

# First touchdown on B2 is a skip, not a B1 completion.
env,target,tick,radius,count=fixture()
for _ in range(3):tick()
wheel=env.scene["robot"].data.body_pos_w;force=env.scene.sensors["contact_forces"].data.net_forces_w
wheel[0,0,2]=.19;force[0,0,2]=0
tick()
wheel[0,0,0]=radius+.42;wheel[0,0,2]=.39;force[0,0,2]=50
for _ in range(3):s,p=tick()
assert s["first_lift_skip_event"].item() and not s["valid"].item()
assert not s["episode_ordered_valid"].item()
# Reset must clear the failed attempt, rather than polluting the next episode.
env.episode_length_buf.zero_();s,p=tick()
assert not s["first_swing_failed"].item() and s["first_swing_side"].item()==-1
assert s["episode_ordered_valid"].item()
print("V135_FIRST_SWING_CONTRACT: PASS (FL/FR, multiple flight lengths, B2-as-B1, pure empty lift, simultaneous, skip, reset, reward snapshot)")

# Physical transverse bounds rotate with the flight, independent of scanner lane.
for heading in (0., torch.pi/2, torch.pi, -torch.pi/2):
    origin=torch.tensor([[2.,-3.]])
    _,forward=contract.flight_anchor(origin,origin,torch.tensor([heading]),(1.4,1.4))
    lateral=torch.stack((-forward[:,1],forward[:,0]),1)
    for k in (1,3):
        bound=1.4+(k-1)*.3-.09
        wheels=origin[:,None,:]+lateral[:,None,:]*torch.tensor([[[.226576],[bound+.01]]])
        y,half=contract.tread_lateral_geometry(wheels,origin,forward,(1.4,1.4),torch.tensor([k]),.3,.09)
        assert (y.abs()<=half).tolist()==[[True,False]]
print("V135_PHYSICAL_TREAD_LATERAL: PASS")
