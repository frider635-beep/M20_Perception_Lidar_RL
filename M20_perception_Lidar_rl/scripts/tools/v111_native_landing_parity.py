from types import SimpleNamespace as NS
from unittest.mock import patch
import copy
import torch

def check_native_landing(cfg):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import v11_continuous as m
    names = ['fl_wheel','fr_wheel','hl_wheel','hr_wheel','fl_hipy','fl_knee','fr_hipy','fr_knee']
    def find(name): return [names.index(name)], [name]
    def fixture():
        asset = NS(find_bodies=find, data=NS(root_pos_w=torch.zeros(1,3),
            root_quat_w=torch.tensor([[1.,0.,0.,0.]]), body_pos_w=torch.zeros(1,8,3),
            body_lin_vel_w=torch.zeros(1,8,3), body_ang_vel_w=torch.zeros(1,8,3)))
        forces = torch.zeros(1,8,3); forces[...,2] = 100
        env = NS(num_envs=1,device='cpu',cfg=cfg,step_dt=.02,episode_length_buf=torch.ones(1,dtype=torch.long),
            scene=NS(sensors={'contact_forces':NS(find_bodies=find,data=NS(net_forces_w=forces))}),
            command_manager=NS(get_term=lambda _:NS(v11_path_heading_w=torch.zeros(1))))
        def frame(x,z,contact=(True,True),edge=1.):
            env.episode_length_buf += 1
            asset.data.body_pos_w[:,:2,0] = torch.tensor(x)
            asset.data.body_pos_w[:,:2,2] = torch.tensor(z) + cfg.v11_perception.wheel_radius
            forces[:,:2,2] = torch.tensor(contact).float() * 100
            risers = dict(valid=torch.tensor([[True,True]]),riser_x=torch.tensor([[edge,edge+.3]]),
                riser_y=torch.zeros(1,2),riser_height=torch.full((1,2),.18),
                riser_lower_world=torch.tensor([[round((edge-1)/.3)*.18,round((edge-1)/.3)*.18+.18]]))
            return m._v11_bilateral_landing_state(env,asset,risers,torch.tensor([True]))
        return env, frame

    candidate,cf=fixture(); baseline,bf=fixture()
    candidate.cfg=copy.deepcopy(cfg); candidate.cfg.v111_rear_posture.physical_first_tread=False
    baseline.cfg=copy.deepcopy(cfg);del baseline.cfg.v111_rear_posture
    with patch.object(m,'wheel_positions_yaw',lambda a,ids:a.data.body_pos_w[:,ids]), patch.object(m,'classify_vertical_riser_contact',lambda *a,**k:torch.zeros(1,2,dtype=torch.bool)):
        frames=[([.8,.8],[0.,0.]),([.9,.9],[.25,.25]),([1.01,1.01],[.25,.25]),([1.15,1.01],[.18,.25])]+[([1.15,1.15],[.18,.18])]*12
        for x,z in frames:
            actual,expected=cf(x,z),bf(x,z)
            for key,value in expected.items():
                if isinstance(value,torch.Tensor):torch.testing.assert_close(actual[key],value,equal_nan=True,msg=lambda msg:f'{key}: {msg}')
    print('V111_POST_ENTRY_LANDING_STATE_EQUALS_NATIVE: PASS',flush=True)
