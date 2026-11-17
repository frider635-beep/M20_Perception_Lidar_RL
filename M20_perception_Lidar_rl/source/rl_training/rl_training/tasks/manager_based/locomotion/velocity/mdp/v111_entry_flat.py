"""Reward-side repair for the selected V11.1 checkpoint; actor ABI unchanged."""
import math
import torch


def reset_entry(s, reset, env):
    if 'entry_certified' not in s:
        for key in ('entry_certified', 'entry_approached'):
            s[key] = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool)
        for key in ('entry_support_steps', 'entry_skip_steps'):
            s[key] = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.long)
        s['entry_failed'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for key in ('entry_certified', 'entry_approached', 'entry_support_steps',
                'entry_skip_steps', 'entry_failed'):
        s[key][reset] = 0


def entry_step(certified, approached, support_steps, skip_steps, failed,
               x, bottom, force_z, lower, rise, tread, pending, cfg, dt):
    """Per-wheel B1 evidence. One wheel skipping cannot be hidden by the other.

    Success requires approach followed by sustained support inside B1 at its
    physical height. A skip is latched for the episode, including after release
    of the original target lock. Neither airborne crossing nor contact at B2
    pays a B1 success event. No ordering between FL and FR is imposed.
    """
    finite = torch.isfinite(torch.cat((x, bottom, force_z, lower[:, None], rise[:, None], tread[:, None]), 1)).all()
    if not finite:
        raise RuntimeError('Nonfinite first-tread reward geometry')
    margin = torch.minimum(torch.full_like(tread, cfg.entry_edge_margin), tread*cfg.entry_edge_fraction)
    end = tread-margin
    contact = force_z >= cfg.entry_contact_force
    approached = approached | (pending[:, None] & contact & (x < 0)
        & ((bottom-lower[:, None]).abs() <= cfg.entry_plane_tolerance))
    top = lower+rise
    support = (pending[:, None] & approached & contact & (x >= margin[:, None])
        & (x <= end[:, None]) & ((bottom-top[:, None]).abs() <= cfg.entry_plane_tolerance))
    support_steps = torch.where(support, support_steps+1, 0)
    ready = support_steps >= math.ceil(cfg.entry_support_seconds/dt)
    event = ready & ~certified & ~failed[:, None]
    certified = certified | (ready & ~failed[:, None])
    skipped = (pending[:, None] & approached & ~certified & contact & (x >= end[:, None])
        & (bottom > top[:, None]+cfg.entry_plane_tolerance))
    skip_steps = torch.where(skipped, skip_steps+1, 0)
    failure = (skip_steps >= math.ceil(cfg.entry_skip_seconds/dt)).any(1)
    fail_event = failure & ~failed
    failed = failed | failure
    # Dense overshoot cost provides a gradient-bearing learning signal before
    # touchdown, rather than only removing completion credit after the skip.
    airborne = bottom > lower[:, None]+cfg.entry_lift_min
    overshoot = ((x-end[:, None]).clamp_min(0)/tread[:, None]).square().clamp_max(4)
    overshoot *= pending[:, None] & ~certified & airborne
    return dict(entry_certified=certified, entry_approached=approached,
        entry_support_steps=support_steps, entry_skip_steps=skip_steps,
        entry_failed=failed, entry_event=event, entry_fail_event=fail_event,
        entry_overshoot=overshoot.amax(1))


def finish_entry(env, asset, landing):
    s, cfg = env._v111_first_tread, env.cfg.v111_entry_flat
    wheel = asset.data.body_pos_w[:, env._v11_front_body_ids]
    x = ((wheel[..., :2]-s['anchor'][:, None])*s['forward'][:, None]).sum(-1)
    force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v11_front_contact_ids, 2]
    valid = s['pending'] & s['height_valid']
    s.update(entry_step(s['entry_certified'], s['entry_approached'], s['entry_support_steps'],
        s['entry_skip_steps'], s['entry_failed'], x,
        wheel[..., 2]-env.cfg.v11_perception.wheel_radius, force,
        s['lower'], s['height'], s['tread'], valid, cfg, float(env.step_dt)))
    complete = s['entry_certified'].all(1) & ~s['entry_failed']
    s['entry_complete_event'] = complete & ~s['done']
    # Once a failed entry is physically beyond B1, later stair learning may
    # resynchronize. Failure remains a metric/event, never a speed restriction
    # on subsequent stairs.
    s['done'] |= complete | s['entry_failed']
    s['skipped'] |= s['entry_failed']
    s['entry_near'] = valid & ~complete & ~s['entry_failed'] & (x.amax(1) >= -cfg.entry_approach_distance)
    s['entry_speed'] = (asset.data.root_lin_vel_w[:, :2]*s['forward']).sum(1)
    command = env.command_manager.get_command('base_velocity')[:, 0].abs()
    limit = torch.minimum(command, torch.full_like(command, cfg.entry_speed_limit))
    s['entry_speed_cost'] = (((s['entry_speed']-limit).clamp_min(0)/cfg.entry_speed_scale).square()
        .clamp_max(4)*s['entry_near'])


def entry_state(env):
    from .v11_continuous import v11_state
    v11_state(env)
    return env._v111_first_tread


def entry_penalty(env):
    s, cfg = entry_state(env), env.cfg.v111_entry_flat
    if hasattr(env.cfg, 'v111_recovery'):
        # v11_state has already updated finish_recovery. Convert the one-off
        # event to a rate because RewardManager multiplies rewards by step_dt.
        overshoot = env._v111_recovery['entry_cost_event']/float(env.step_dt)
    else:
        overshoot = s['entry_overshoot']
    return -(overshoot + cfg.entry_speed_cost_scale*s['entry_speed_cost'])


def entry_support_reward(env):
    s = entry_state(env)
    return (s['entry_event'].float().sum(1) + s['entry_complete_event'].float())/float(env.step_dt)


def entry_skip_penalty(env):
    return -entry_state(env)['entry_fail_event'].float()/float(env.step_dt)


def flat_state(env):
    from .v111_support_cycle import flat_state as planar_state
    from .v112_flat_posture import workspace_cost
    base = planar_state(env)
    robot, cfg = env.scene['robot'], env.cfg.v111_entry_flat
    if not hasattr(env, '_v111_entry_flat_ids'):
        legs = ('fl', 'fr', 'hl', 'hr')
        wheel = [robot.find_bodies(leg+'_wheel')[0][0] for leg in legs]
        hip = [robot.find_bodies(leg+'_hipy')[0][0] for leg in legs]
        joint = [robot.joint_names.index(leg+'_'+kind+'_joint') for leg in legs for kind in ('hipy', 'knee')]
        env._v111_entry_flat_ids = wheel, hip, joint
    wheel, hip, joint = env._v111_entry_flat_ids
    q = robot.data.root_quat_w
    forward = torch.stack((1-2*(q[:, 2].square()+q[:, 3].square()),
        2*(q[:, 1]*q[:, 2]+q[:, 0]*q[:, 3]), 2*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])), 1)
    reach = ((robot.data.body_pos_w[:, wheel]-robot.data.body_pos_w[:, hip])*forward[:, None]).sum(-1)
    delta = robot.data.joint_pos[:, joint]-robot.data.default_joint_pos[:, joint]
    cost = workspace_cost(reach, delta, cfg)
    return dict(base, workspace_cost=cost, reach_max=reach.abs().amax(1))


def flat_penalty(env):
    s = flat_state(env)
    cost = s['workspace_cost']+s['cost']
    if hasattr(env.cfg, 'v111_recovery'):
        # A soft cap keeps useful differences near the stretched baseline;
        # unlike an exponential it does not rapidly flatten those differences.
        cap = env.cfg.v111_recovery.flat_cost_cap
        cost = cap*cost/(cap+cost)
    return -cost*s['blend']


def drive_multiplier(env):
    if hasattr(env.cfg, 'v111_recovery'):
        # Preserve the native locomotion objective independently of shape.
        return torch.ones(env.num_envs, device=env.device)
    cfg, flat = env.cfg.v111_entry_flat, flat_state(env)
    # Planar-ground posture only. B1 failure is penalized at the event;
    # retaining its history must not suppress speed on any later stair.
    return torch.exp(-cfg.flat_drive_cost_scale*flat['workspace_cost']*flat['blend'])


def track_velocity(env, std, command_name):
    from .v111_rear_posture import v111_track_lin_vel_with_descent_support
    return v111_track_lin_vel_with_descent_support(env, std, command_name)*drive_multiplier(env)


def forward_progress(env, target_speed_fraction=.5):
    from .v111_rear_posture import v111_world_forward_with_descent_support
    return v111_world_forward_with_descent_support(env, target_speed_fraction)*drive_multiplier(env)


def forward_deficit(env, min_speed_fraction=.5, speed_scale=.3):
    from .v111_rear_posture import v111_world_forward_deficit_with_descent_support
    native = v111_world_forward_deficit_with_descent_support(env, min_speed_fraction, speed_scale)
    # Do not punish a controlled deceleration into B1 at a high command speed.
    return torch.where(entry_state(env)['entry_near'], torch.zeros_like(native), native)


def metric(env, env_ids, key):
    if key.startswith('flat_'):
        s = flat_state(env)
        if key == 'flat_active':
            return s['active'][env_ids].float().mean()
        value = s[key.removeprefix('flat_')][env_ids]
        mask = s['active'][env_ids].float()
        return (value*mask).sum()/mask.sum().clamp_min(1)
    s = entry_state(env)
    value = s[key][env_ids].float()
    return value.mean()
