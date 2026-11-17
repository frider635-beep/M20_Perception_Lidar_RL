"""V11.2 tread-relative support and reward-only sequential handoff helpers."""
import torch


def stair_coordinates(env, asset, risers):
    """Rotate scanner coordinates into the generated flight's cardinal frame."""
    q = asset.data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2].square() + q[:, 3].square()))
    heading = env.command_manager.get_term('base_velocity').v11_path_heading_w
    side = torch.round(heading / (torch.pi / 2)).long() % 4
    forward = q.new_tensor(((1., 0.), (0., 1.), (-1., 0.), (0., -1.)))[side]
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    x, y = risers['riser_x'], risers['riser_y']
    world_delta = torch.stack((yaw.cos()[:, None] * x - yaw.sin()[:, None] * y,
                               yaw.sin()[:, None] * x + yaw.cos()[:, None] * y), -1)
    out = dict(risers)
    out['riser_x'] = (world_delta * forward[:, None]).sum(-1)
    out['riser_y'] = (world_delta * lateral[:, None]).sum(-1)
    return out, forward


def wheels_in_stair_frame(asset, ids, forward):
    delta = asset.data.body_pos_w[:, ids] - asset.data.root_pos_w[:, None]
    lateral = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    return torch.stack(((delta[..., :2] * forward[:, None]).sum(-1),
                        (delta[..., :2] * lateral[:, None]).sum(-1), delta[..., 2]), -1)


def support_bounds(edge, width, cfg):
    # Bound the load-bearing contact position, not the wheel's full projected
    # diameter. Contact force/height/slip and vertical-face collision are still
    # checked independently. This remains nonempty at the shortest tread.
    inset = torch.minimum(torch.full_like(width, cfg.contact_edge_margin),
                          width * cfg.contact_edge_fraction)
    valid = torch.isfinite(width) & (width > 2 * inset) & (inset > 0)
    return edge + inset, edge + width - inset, valid


def certify_cycle(landing):
    """A timed pair is not a completion without the original safe crossing."""
    safe = landing['candidate_ready'].all(1) & ~landing['unsupported_handoff']
    rejected = landing['cycle_event'] & ~safe
    landing['cycle_bad'] |= rejected
    landing['cycle_done'] &= ~rejected
    landing['cycle_event'] &= safe


def reward_factor(cost, blend, scale):
    return 1.0 - blend + blend * torch.exp(-scale * cost)


def _factors(env):
    from .v111_support_cycle import cycle_drive
    from .v112_flat_posture import flat_state
    from .v135_descent_support_transfer import v135_descent_velocity_modulation
    flat = flat_state(env)
    _, holding = cycle_drive(env)
    # Only actual support transfer reduces drive; seeing a target is not a cost.
    drive = torch.where(holding, env.cfg.v111_rear_posture.hold_drive_scale, 1.0)
    descent = v135_descent_velocity_modulation(env)
    blend = flat['blend']
    # Confirmed planar geometry overrides stale stair shaping, but never
    # changes the actor action or the velocity command.
    stair = drive * descent
    posture = reward_factor(flat['cost'], blend, env.cfg.v112_handoff.flat_drive_scale)
    return ((1.0 - blend) * stair + blend) * posture, holding & (blend < 1.0)


def track_velocity(env, std, command_name):
    from .rewards import track_lin_vel_xy_exp
    factors, holding = _factors(env)
    native = track_lin_vel_xy_exp(env, std=std, command_name=command_name)
    # During the bounded, physically certified dwell do not ask for forward
    # velocity at the same time. This changes reward only, never actor commands.
    quiet = torch.exp(-env.scene['robot'].data.root_lin_vel_b[:, :2].square().sum(1) / std**2)
    return torch.where(holding, quiet, native) * factors * route_state(env)['quality']


def forward_progress(env, target_speed_fraction=.5):
    from .v11_continuous import v11_world_forward_progress_reward
    factors, holding = _factors(env)
    native = v11_world_forward_progress_reward(env, target_speed_fraction=target_speed_fraction)
    return torch.where(holding, 0., native) * factors * route_state(env)['quality']


def forward_deficit(env, min_speed_fraction=.5, speed_scale=.3):
    from .v111_rear_posture import v111_world_forward_deficit_with_descent_support
    from .v11_continuous import v11_world_forward_speed_deficit
    from .v112_flat_posture import flat_state
    value = v111_world_forward_deficit_with_descent_support(env, min_speed_fraction, speed_scale)
    native = v11_world_forward_speed_deficit(env, min_speed_fraction=min_speed_fraction, speed_scale=speed_scale)
    blend = flat_state(env)['blend']
    value = (1.0 - blend) * value + blend * native
    return torch.where(_factors(env)[1], torch.zeros_like(value), value)


def tread_metric(env, env_ids, width, key):
    from .v135_terrain_geometry import v135_terrain_geometry
    from .v11_continuous import v11_state
    geometry = v135_terrain_geometry(env)
    mask = geometry['ascent'][env_ids] & torch.isclose(
        geometry['step_width'][env_ids], torch.tensor(width, device=env.device))
    if key == 'samples':
        return mask.float().sum()
    state = v11_state(env)['landing']
    value = state.get(key, torch.zeros(env.num_envs, device=env.device))[env_ids].float()
    return (value * mask).sum() / mask.float().sum().clamp_min(1)


def route_quality(lateral, yaw_error, cfg):
    """Task route is independent of riser detection and current posture."""
    excess = (lateral.abs() - cfg.route_lateral_free).clamp_min(0)
    quality = torch.exp(-(excess / cfg.route_lateral_scale).square()
                        - (yaw_error / cfg.route_heading_scale).square())
    return torch.where(lateral.abs() >= cfg.route_lateral_limit, 0., quality)


def route_state(env):
    from .v11_continuous import v11_state, _v11_ascending_stair_forward_mask
    step = int(env.common_step_counter)
    cached = getattr(env, '_v112_route', None)
    if cached is not None and cached['step'] == step:
        return cached
    robot = env.scene['robot']
    command = env.command_manager.get_term('base_velocity')
    heading = command.v11_path_heading_w
    left = torch.stack((-heading.sin(), heading.cos()), 1)
    lateral = ((robot.data.root_pos_w[:, :2] - command.v11_path_origin_w) * left).sum(1)
    q = robot.data.root_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2].square() + q[:, 3].square()))
    error = torch.atan2(torch.sin(yaw - heading), torch.cos(yaw - heading))
    # Platform confirmation comes from the physical landing lifecycle, not
    # an empty scan or a flat patch reached by sidestepping the staircase.
    from .v135_terrain_geometry import v135_terrain_geometry
    geometry = v135_terrain_geometry(env)
    side = torch.round(heading / (torch.pi / 2)).long() % 4
    forward = q.new_tensor(((1., 0.), (0., 1.), (-1., 0.), (0., -1.)))[side]
    end = geometry['radii'].gather(1, (side % 2)[:, None]).squeeze(1) + (geometry['flight_riser_count'] - 1) * geometry['step_width']
    ids = env._v11_rear_body_ids
    rear = ((robot.data.body_pos_w[:, ids, :2] - env.scene.env_origins[:, None, :2]) * forward[:, None]).sum(-1)
    landing = v11_state(env)['landing']
    exited = (landing['platform_confirmed'] & (rear >= end[:, None]).all(1)
              & (lateral.abs() <= env.cfg.v112_handoff.route_lateral_free))
    if cached is None:
        cached = {key: torch.zeros(env.num_envs, device=env.device) for key in
                  ('physical_pairs', 'style_cycles', 'route_frames', 'offroute_frames')}
        cached['last'] = env.episode_length_buf.clone()
        cached['completed'] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    reset = env.episode_length_buf < cached['last']
    for key in ('physical_pairs', 'style_cycles', 'route_frames', 'offroute_frames'):
        cached[key][reset] = 0
    cached['completed'][reset] = False
    cached['completed'] |= exited
    active = _v11_ascending_stair_forward_mask(env) & ~cached['completed']
    quality = torch.where(active, route_quality(lateral, error, env.cfg.v112_handoff), 1.)
    cached['physical_pairs'] += landing['bilateral_landing_event'].float()
    cached['style_cycles'] += paid_cycle_event(env).float()
    cached['route_frames'] += active.float()
    cached['offroute_frames'] += (active & (lateral.abs() >= env.cfg.v112_handoff.route_lateral_limit)).float()
    cached.update(step=step, last=env.episode_length_buf.clone(), active=active, quality=quality, lateral=lateral)
    env._v112_route = cached
    return cached


def track_yaw(env, std, command_name):
    from .rewards import track_ang_vel_z_exp
    return track_ang_vel_z_exp(env, std=std, command_name=command_name) * route_state(env)['quality']


def once_per_riser(position, event, blocked, best, reset, tolerance):
    best = torch.where(reset, -torch.inf, best)
    paid = event & ~blocked & (position > best + tolerance)
    return paid, torch.where(paid, position, best)


def paid_cycle_event(env):
    from .v11_continuous import v11_state
    step = int(env.common_step_counter)
    cache = getattr(env, '_v112_cycle_payment', None)
    if cache is not None and cache['step'] == step:
        return cache['paid']
    state = v11_state(env)['landing']
    heading = env.command_manager.get_term('base_velocity').v11_path_heading_w
    forward = torch.stack((heading.cos(), heading.sin()), 1)
    position = (state['reference_riser_world_xy'] * forward).sum(1)
    if cache is None:
        cache = dict(best=torch.full_like(position, -torch.inf), last=env.episode_length_buf.clone())
    paid, best = once_per_riser(position, state['cycle_event'], state['adjacent_credit_blocked'],
        cache['best'], env.episode_length_buf < cache['last'], env.cfg.v11_perception.encounter_riser_switch_distance)
    cache.update(step=step, best=best, paid=paid, last=env.episode_length_buf.clone())
    env._v112_cycle_payment = cache
    return paid


def cycle_completion(env):
    return paid_cycle_event(env).float() / float(env.step_dt)


def route_metric(env, env_ids, key):
    state = route_state(env)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    from .v135_terrain_geometry import v135_terrain_geometry
    ids = ids[v135_terrain_geometry(env)['ascent'][ids]]
    if key == 'samples':
        return torch.tensor(float(ids.numel()), device=env.device)
    if ids.numel() == 0:
        return torch.zeros((), device=env.device)
    if key == 'offroute_fraction':
        return (state['offroute_frames'][ids] / state['route_frames'][ids].clamp_min(1)).mean()
    return state[key][ids].float().mean()
