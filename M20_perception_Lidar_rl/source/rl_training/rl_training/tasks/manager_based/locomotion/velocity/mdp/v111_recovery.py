"""Finite swing cost and whole-body forward evidence for the recovery task."""
import math
import torch


def budget_step(target, peak, expected, overshoot):
    prior = torch.where(target == expected, peak, 0.)
    current = overshoot.clamp(0, 1)
    return expected.clone(), torch.maximum(prior, current), (current-prior).clamp_min(0).sum(1)


def progress_step(best, position, active):
    prior = torch.where(torch.isfinite(best), best, position)
    delta = (position-prior).clamp_min(0)*active
    return torch.where(active, torch.maximum(prior, position), position), delta


def reset_recovery(env, reset):
    if not hasattr(env, '_v111_recovery'):
        env._v111_recovery = dict(target=torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.long),
            peak=torch.zeros((env.num_envs, 2), device=env.device),
            root_best=torch.full((env.num_envs,), torch.nan, device=env.device),
            rear_best=torch.full((env.num_envs,), torch.nan, device=env.device),
            top_frames=torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
            whole_flight=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    s = env._v111_recovery
    for key in ('target', 'peak', 'top_frames', 'whole_flight'):
        s[key][reset] = 0
    for key in ('root_best', 'rear_best'):
        s[key][reset] = torch.nan


def finish_recovery(env, robot, geometry):
    entry, adjacent, s = env._v111_first_tread, env._v111_adjacent, env._v111_recovery
    c = env.cfg.v111_recovery
    s['target'], s['peak'], s['cost_event'] = budget_step(s['target'], s['peak'],
        adjacent['overshoot_target'], adjacent['overshoot_side'])
    if not hasattr(env, '_v111_recovery_wheels'):
        legs = ('fl', 'fr', 'hl', 'hr')
        env._v111_recovery_wheels = [robot.find_bodies(leg+'_wheel')[0][0] for leg in legs]
        sensor = env.scene.sensors['contact_forces']
        env._v111_recovery_contacts = [sensor.find_bodies(leg+'_wheel')[0][0] for leg in legs]
    wheels = robot.data.body_pos_w[:, env._v111_recovery_wheels]
    wheel_x = ((wheels[..., :2]-entry['anchor'][:, None])*entry['forward'][:, None]).sum(-1)
    root_x = ((robot.data.root_pos_w[:, :2]-entry['anchor'])*entry['forward']).sum(-1)
    course = geometry['ascent'] & entry['height_valid'] & (root_x < geometry['flight_riser_count']*entry['tread'])
    # This new positive progress term is disabled during the inherited B1-only
    # deceleration scope. It cannot demand extra speed while entering B1.
    command = env.command_manager.get_command('base_velocity')[:, 0]
    active = course & ~entry['entry_near'] & (command > env.cfg.v11_perception.forward_command_threshold)
    s['root_best'], root_delta = progress_step(s['root_best'], root_x, active)
    s['rear_best'], rear_delta = progress_step(s['rear_best'], wheel_x[:, 2:].amin(1), active)
    s['progress'] = ((root_delta+c.rear_progress_scale*rear_delta)/entry['tread'].clamp_min(1.e-6)).clamp_max(1)
    top = entry['lower']+geometry['flight_riser_count']*entry['height']
    bottom = wheels[..., 2]-env.cfg.v11_perception.wheel_radius
    force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v111_recovery_contacts, 2]
    support = (force >= c.contact_force) & ((bottom-top[:, None]).abs() <= c.plane_tolerance)
    candidate = (geometry['ascent'] & adjacent['complete'] & support.all(1)
        & (wheel_x >= (geometry['flight_riser_count']-1)[:, None]*entry['tread'][:, None]).all(1)
        & (-robot.data.projected_gravity_b[:, 2] >= c.minimum_upright_cosine)
        & (robot.data.root_pos_w[:, 2]-top >= c.minimum_top_clearance))
    s['top_frames'] = torch.where(candidate, s['top_frames']+1, 0)
    ready = s['top_frames'] >= math.ceil(c.top_support_seconds/float(env.step_dt))
    s['whole_event'] = ready & ~s['whole_flight']
    s['whole_flight'] |= ready


def state(env):
    from .v11_continuous import v11_state
    v11_state(env)
    return env._v111_recovery


def overshoot_cost(env):
    return -state(env)['cost_event']/float(env.step_dt)


def body_progress(env):
    return state(env)['progress']/float(env.step_dt)


def whole_completion(env):
    return state(env)['whole_event'].float()/float(env.step_dt)


def metric(env, env_ids, key):
    return state(env)[key][env_ids].float().mean()
