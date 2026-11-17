"""Synthetic contacts through the production bilateral state, including a legacy counterexample."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch


def check_cycle_integration(cfg):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import v11_continuous as m
    names = ['fl_wheel','fr_wheel','hl_wheel','hr_wheel','fl_hipy','fl_knee','fr_hipy','fr_knee']
    def find(name): return [names.index(name)], [name]
    old_first = cfg.v111_rear_posture.physical_first_tread
    cfg.v111_rear_posture.physical_first_tread = False
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
    with patch.object(m,'wheel_positions_yaw',lambda a,ids:a.data.body_pos_w[:,ids]), \
         patch.object(m,'classify_vertical_riser_contact',lambda *a,**k:torch.zeros(1,2,dtype=torch.bool)):
        # Same physical trajectory: old handoff pays near-synchronous support,
        # timed handoff must withhold that completion (RED baseline -> PASS).
        for enabled in (False,True):
            cfg.v111_rear_posture.timed_support_cycle = enabled
            env, frame = fixture()
            frame([.8,.8],[0.,0.]); frame([.9,.9],[.25,.25])
            frame([1.01,1.01],[.25,.25])
            frame([1.15,1.01],[.18,.25])
            paid = False
            for _ in range(5):
                s=frame([1.15,1.15],[.18,.18]); paid |= bool(s['bilateral_landing_event'].item())
            assert paid == (not enabled), (enabled, paid)
        # Correct dwell, both mirror images, scanner promotion while holding.
        for side in (0,1):
            env, frame = fixture()
            x=[.8,.8]; z=[0.,0.]; contact=[True,True]
            frame(x,z)
            x[side]=.9;z[side]=.25;contact[side]=False;frame(x,z,contact)
            x[side]=1.01;frame(x,z,contact)
            x[side]=1.15;z[side]=.18;contact[side]=True
            for _ in range(10): s=frame(x,z,contact)
            assert s['cycle_ready'].item() and not s['cycle_done'].item()
            mate=1-side;x[mate]=.9;z[mate]=.25;contact[mate]=False;frame(x,z,contact,edge=1.3)
            x[mate]=1.01;frame(x,z,contact,edge=1.3)
            x[mate]=1.15;z[mate]=.18;contact[mate]=True
            for i in range(8):
                s=frame(x,z,contact,edge=1.3)
                assert s['bilateral_landing_event'].item() == (i == 7), (i,s['cycle_pair_steps'])
            assert s['cycle_done'].item()
            # Per-episode reset must remove timing and payment history.
            env.episode_length_buf.zero_();s=frame([.8,.8],[0.,0.])
            assert s['cycle_leader'].item() == -1 and not s['cycle_done'].item()
        # A safe physical support below the desired crossing clearance margin
        # earns graded shaping, but cannot earn strict completion credit.
        env, frame = fixture()
        frame([.8,.8],[0.,0.])
        frame([.9,.8],[.25,0.],(False,True))
        frame([1.01,.8],[.21,0.],(False,True))
        partial = 0.
        for _ in range(10):
            s=frame([1.15,.8],[.18,0.]);partial += s['cycle_delta'].item()
        assert 0 < partial < .35 and not s['candidate_ready'].all()
        assert not s['bilateral_landing_event'].item()
        # B1 physical release: synchronized landing is failed-but-resyncable;
        # a valid pair still dwelling must remain locked until cycle_done.
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_first_tread as ft
        env, frame = fixture()
        s = frame([.8,.8],[0.,0.])
        asset = NS(data=NS(body_pos_w=torch.zeros(1,8,3)))
        env._v111_first_tread = dict(pending=torch.tensor([True]),height_valid=torch.tensor([True]),
            anchor=torch.tensor([[1.,0.]]),forward=torch.tensor([[1.,0.]]),tread=torch.tensor([.3]),
            skip_steps=torch.zeros(1,dtype=torch.long),skipped=torch.tensor([False]),done=torch.tensor([False]))
        s['bilateral_confirmed'][:] = True
        s['cycle_done'][:] = False; s['cycle_bad'][:] = False
        ft.finish_first_tread(env,asset,s)
        assert not env._v111_first_tread['done'].item(), 'B1 released before pair dwell'
        s['cycle_bad'][:] = True
        ft.finish_first_tread(env,asset,s)
        assert env._v111_first_tread['done'].item(), 'synchronous B1 cannot resync'
        env._v111_first_tread['done'][:] = False
        s['cycle_bad'][:] = False;s['cycle_done'][:] = True
        ft.finish_first_tread(env,asset,s)
        assert env._v111_first_tread['done'].item(), 'completed dwell cannot release B1'
    cfg.v111_rear_posture.physical_first_tread = old_first
    print('V111_CYCLE_PRODUCTION_TRAJECTORIES: PASS (legacy counterexample rejected, mirror, dwell, scanner, reset)',flush=True)
