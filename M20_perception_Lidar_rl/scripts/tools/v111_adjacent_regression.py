"""Synthetic trajectories through the actual bilateral landing implementation."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch


def check_adjacent_landing(cfg):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import v11_continuous as m
    names=['fl_wheel','fr_wheel','hl_wheel','hr_wheel','fl_hipy','fl_knee','fr_hipy','fr_knee']
    def find(name): return [names.index(name)],[name]
    asset=NS(find_bodies=find,data=NS(root_pos_w=torch.zeros(1,3),
        root_quat_w=torch.tensor([[1.,0.,0.,0.]]),body_pos_w=torch.zeros(1,8,3)))
    forces=torch.zeros(1,8,3);forces[:,:,2]=100.
    sensor=NS(find_bodies=find,data=NS(net_forces_w=forces))
    env=NS(num_envs=1,device='cpu',cfg=cfg,episode_length_buf=torch.ones(1,dtype=torch.long),
           scene=NS(sensors={'contact_forces':sensor}),
           command_manager=NS(get_term=lambda _:NS(v11_path_heading_w=torch.zeros(1))))
    timed_enabled=cfg.v111_rear_posture.timed_support_cycle
    cfg.v111_rear_posture.timed_support_cycle=False
    handoff_enabled=cfg.v111_rear_posture.same_tread_handoff
    cfg.v111_rear_posture.same_tread_handoff=False
    first_enabled=cfg.v111_rear_posture.physical_first_tread
    cfg.v111_rear_posture.physical_first_tread=False
    r=cfg.v11_perception.wheel_radius
    rise=.18
    def frame(edge,x,bottom,next_valid=True):
        env.episode_length_buf+=1
        asset.data.body_pos_w[:,0:2,0]=torch.as_tensor(x)
        asset.data.body_pos_w[:,0:2,2]=torch.as_tensor(bottom)+r
        risers={'valid':torch.tensor([[True,next_valid]]),
                'riser_x':torch.tensor([[edge,edge+.3]]), 'riser_y':torch.zeros(1,2),
                'riser_height':torch.full((1,2),rise),
                'riser_lower_world':torch.tensor([[round((edge-1.)/.3)*rise,round((edge-1.)/.3)*rise+rise]])}
        return m._v11_bilateral_landing_state(env,asset,risers,torch.tensor([True]))
    with patch.object(m,'wheel_positions_yaw',lambda a,ids:a.data.body_pos_w[:,ids]), \
         patch.object(m,'classify_vertical_riser_contact',lambda *a,**k:torch.zeros(1,2,dtype=torch.bool)):
        frame(1.,.8,.25)
        frame(1.,.9,.25)
        s=frame(1.,1.01,.25)
        assert s['crossing'].all()
        # Promote B2 after skipping B1: B1 crossing must not survive.
        s=frame(1.3,1.1,.43)
        assert s['adjacent_credit_blocked'].all() and not s['crossing'].any()
        frame(1.3,1.2,.43)
        frame(1.3,1.31,.43)
        for _ in range(3): s=frame(1.3,1.45,.36)
        assert s['candidate_ready'].all(), 'physical resync must stay reachable'
        assert not s['bilateral_landing_event'].any() and s['clearance_settlement'].sum()==0
        # Normal B3 must be rewarded after local resync, not permanently dead.
        frame(1.6,1.5,.61)
        frame(1.6,1.55,.61)
        frame(1.6,1.61,.61)
        for _ in range(3): s=frame(1.6,1.75,.54)
        assert s['bilateral_landing_event'].all(), 'normal next tread credit unreachable'
        # Reset must erase blocked state and old approach/support.
        env.episode_length_buf.zero_()
        s=frame(1.,1.15,.18)
        assert not s['crossing'].any(), 'spawn above tread must not create crossing'
        assert not s['adjacent_credit_blocked'].any()
        # A wheel contacting beyond B2 cannot count as B1 support.
        env.episode_length_buf.zero_();frame(1.,.8,.25)
        frame(1.,.9,.25)
        frame(1.,1.01,.25)
        for _ in range(3): s=frame(1.,1.45,.18)
        assert not s['support'].any()
        cfg.v111_rear_posture.physical_first_tread=first_enabled
        from rl_training.tasks.manager_based.locomotion.velocity.mdp import v111_first_tread as ft
        env.scene.env_origins=torch.zeros(1,3)
        geometry=dict(ascent=torch.tensor([True]),radii=torch.tensor([[1.,1.]]),step_width=torch.tensor([.3]))
        with patch.object(ft,'v135_terrain_geometry',lambda _:geometry):
            del env._v11_landing
            # First scan B2 must not initialize a B1 height or pay completion.
            frame(1.3,1.1,.43)
            assert not env._v111_first_tread['height_valid'].any()
            for _ in range(3): s=frame(1.3,1.45,.36)
            assert env._v111_first_tread['skipped'].all()
            assert not s['bilateral_landing_event'].any()
            # Genuine B1 remains locked when the scanner promotes B2 midair.
            env.episode_length_buf.zero_()
            frame(1.,.8,.25);frame(1.,.9,.25)
            s=frame(1.3,1.01,.25)
            torch.testing.assert_close(s['reference_riser_world_xy'][:,0],torch.tensor([1.]))
            for _ in range(3): s=frame(1.3,1.15,.18)
            assert s['bilateral_landing_event'].all()
            assert env._v111_first_tread['done'].all()
            assert not env._v111_first_tread['skipped'].any()
            # Empty swing cannot settle the first tread.
            env.episode_length_buf.zero_()
            frame(1.,.8,.25);frame(1.,.9,.25)
            for _ in range(3): s=frame(1.,.9,0.)
            assert not env._v111_first_tread['done'].any()
            assert not s['landing_event'].any()
        cfg.v111_rear_posture.physical_first_tread=False
        cfg.v111_rear_posture.same_tread_handoff=handoff_enabled
        # Both orderings: old left/right support on different frames cannot
        # create a bilateral completion while the leader is on the next tread.
        for lead in (0,1):
            env.episode_length_buf.zero_();forces[:,:,2]=100.
            frame(1.,.8,.25);frame(1.,.9,.25)
            x=[.9,.9];z=[.25,.25];x[lead]=1.01
            frame(1.,x,z);x[lead]=1.15;z[lead]=.18
            for _ in range(3): frame(1.,x,z)
            x[lead]=1.45;z[lead]=.36;x[1-lead]=1.01
            frame(1.,x,z);x[1-lead]=1.15;z[1-lead]=.18
            for _ in range(3): s=frame(1.,x,z)
            assert s['candidate_ready'].all(), 'historical candidates should still be visible'
            assert not s['bilateral_confirmed'].any()
            assert not s['bilateral_landing_event'].any()
            # Support overlap is now real; one and only one event can settle.
            for _ in range(3): s=frame(1.,1.15,.18)
            assert s['bilateral_landing_event'].all()
            s=frame(1.,1.15,.18)
            assert not s['bilateral_landing_event'].any()
        # Simultaneous landing with no prior single-leg support is not a handoff.
        env.episode_length_buf.zero_();forces[:,:,2]=100.
        frame(1.,.8,.25);frame(1.,.9,.25);frame(1.,1.01,.25)
        for _ in range(3): s=frame(1.,1.15,.18)
        assert s['bilateral_confirmed'].all() and (s['handoff_leader'] == -1).all()
        assert not s['bilateral_landing_event'].any()
        # Sustained double-front airborne motion cannot earn clean landing credit.
        env.episode_length_buf.zero_();forces[:,:,2]=100.
        frame(1.,.8,0.)
        forces[:,0:2,2]=0.
        frame(1.,.85,.25);frame(1.,.9,.25);frame(1.,1.01,.25)
        forces[:,0:2,2]=100.
        for _ in range(3): s=frame(1.,1.15,.18)
        assert s['bilateral_confirmed'].all(), 'physical completion must permit local resync'
        assert s['unsupported_handoff'].all()
        assert not s['bilateral_landing_event'].any() and s['clearance_settlement'].sum()==0
        env.episode_length_buf.zero_();s=frame(1.,.8,0.)
        assert not s['unsupported_handoff'].any() and not s['bilateral_confirmed'].any()
        cfg.v111_rear_posture.physical_first_tread=first_enabled
    cfg.v111_rear_posture.timed_support_cycle=timed_enabled
    print('V111_SAME_TREAD_HANDOFF: PASS',flush=True)
    print('V111_ADJACENT_TRAJECTORIES: PASS',flush=True)
    print('V111_PHYSICAL_B1_ENTRY: PASS',flush=True)
