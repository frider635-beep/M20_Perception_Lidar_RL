"""Reward-only physical B1 lock for V11.1; native bilateral behavior after entry."""
import torch
from .v135_ascent_contract import flight_anchor
from .v135_terrain_geometry import v135_terrain_geometry


def prepare_first_tread(env, asset, risers):
    geometry = v135_terrain_geometry(env)
    heading = env.command_manager.get_term('base_velocity').v11_path_heading_w
    anchor, forward = flight_anchor(asset.data.root_pos_w[:, :2],
                                   env.scene.env_origins[:, :2], heading, geometry['radii'])
    if not hasattr(env, '_v111_first_tread'):
        env._v111_first_tread = dict(last=env.episode_length_buf.clone(),
            done=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool),
            height_valid=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool),
            lower=torch.zeros(env.num_envs,device=env.device),
            height=torch.zeros(env.num_envs,device=env.device),
            skip_steps=torch.zeros(env.num_envs,device=env.device,dtype=torch.long),
            skipped=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool))
    s=env._v111_first_tread
    reset=env.episode_length_buf < s['last']
    for key in ('done','height_valid','lower','height','skip_steps','skipped'):
        s[key][reset]=0
    if hasattr(env.cfg, 'v111_entry_flat'):
        from .v111_entry_flat import reset_entry
        reset_entry(s, reset, env)
    if hasattr(env.cfg, 'v111_adjacent'):
        from .v111_adjacent import reset_sequence
        reset_sequence(env, reset)
    if hasattr(env.cfg, 'v111_recovery'):
        from .v111_recovery import reset_recovery
        reset_recovery(env, reset)
    s['last']=env.episode_length_buf.clone()
    pending=geometry['ascent'] & ~s['done']
    q=asset.data.root_quat_w
    yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    c,sn=yaw.cos(),yaw.sin()
    world=torch.stack((asset.data.root_pos_w[:,0:1]+c[:,None]*risers['riser_x']-sn[:,None]*risers['riser_y'],
                       asset.data.root_pos_w[:,1:2]+sn[:,None]*risers['riser_x']+c[:,None]*risers['riser_y']),-1)
    distance=((world-anchor[:,None])*forward[:,None]).sum(-1).abs()
    match=risers['valid'] & (risers['riser_height']>0) & (distance < .08)
    slot=distance.masked_fill(~match,torch.inf).argmin(1)[:,None]
    acquire=pending & ~s['height_valid'] & match.any(1)
    for key,source in (('lower','riser_lower_world'),('height','riser_height')):
        s[key]=torch.where(acquire,risers[source].gather(1,slot).squeeze(1),s[key])
    s['height_valid'] |= acquire
    projected={k:v.clone() for k,v in risers.items()}
    edges=anchor[:,None]+torch.arange(2,device=env.device)[None,:,None]*geometry['step_width'][:,None,None]*forward[:,None]
    delta=edges-asset.data.root_pos_w[:,None,:2]
    values={'riser_x':c[:,None]*delta[:,:,0]+sn[:,None]*delta[:,:,1],
            'riser_y':-sn[:,None]*delta[:,:,0]+c[:,None]*delta[:,:,1],
            'riser_lower_world':s['lower'][:,None].expand(-1,2),
            'riser_height':s['height'][:,None].expand(-1,2),
            'valid':s['height_valid'][:,None].expand(-1,2)}
    for key,value in values.items():
        projected[key]=torch.where(pending[:,None],value,projected[key])
    s.update(pending=pending,anchor=anchor,forward=forward,tread=geometry['step_width'])
    return projected


def finish_first_tread(env, asset, landing):
    if hasattr(env.cfg, 'v111_entry_flat'):
        from .v111_entry_flat import finish_entry
        finish_entry(env, asset, landing)
        if hasattr(env.cfg, 'v111_adjacent'):
            from .v111_adjacent import finish_sequence
            finish_sequence(env, asset)
        return
    s=env._v111_first_tread
    completed=s['pending'] & (landing['bilateral_confirmed'] if env.cfg.v111_rear_posture.same_tread_handoff else landing['candidate_ready'].all(1)) & s['height_valid']
    if getattr(env.cfg.v111_rear_posture, 'timed_support_cycle', False):
        # Keep B1 until dwell completes. Failed cycles may physically resync;
        # a successful but still settling pair must not release its target.
        completed &= landing.get('cycle_done', torch.zeros_like(completed)) | landing.get('cycle_bad', torch.zeros_like(completed))
    wheel=asset.data.body_pos_w[:,env._v11_front_body_ids,:2]
    travel=((wheel-s['anchor'][:,None])*s['forward'][:,None]).sum(-1)
    force=env.scene.sensors['contact_forces'].data.net_forces_w[:,env._v11_front_contact_ids,2]
    # Supported beyond B2 is a failed entry, never a B1 completion. Release
    # the entry lock afterwards so later native stair learning remains live.
    skip=(s['pending'] & ~completed & (travel >= s['tread'][:,None]+env.cfg.v11_perception.wheel_radius).all(1)
          & (force > env.cfg.v11_perception.contact_force_threshold).all(1))
    s['skip_steps']=torch.where(skip,s['skip_steps']+1,torch.zeros_like(s['skip_steps']))
    failed=s['skip_steps']>=3
    s['skipped'] |= failed
    s['done'] |= completed | failed
    if failed.any():
        from .v11_continuous import _v11_reset_landing
        _v11_reset_landing(landing,failed,reset_credit_ledger=False)
        if env.cfg.v111_rear_posture.adjacent_landing_credit:
            landing['adjacent_credit_blocked'] |= failed
