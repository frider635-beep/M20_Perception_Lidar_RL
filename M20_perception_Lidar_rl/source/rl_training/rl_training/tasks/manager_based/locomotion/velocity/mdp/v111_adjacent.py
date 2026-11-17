"""Physical per-front-wheel adjacent treads; no speed shaping after B1."""
import math
import torch


def reset_sequence(env, reset):
    if not hasattr(env, '_v111_adjacent'):
        shape = (env.num_envs, 2)
        env._v111_adjacent = dict(
            last_index=torch.zeros(shape, device=env.device, dtype=torch.long),
            observed_index=torch.full(shape, -1, device=env.device, dtype=torch.long),
            support_frames=torch.zeros(shape, device=env.device, dtype=torch.long),
            ever_skipped=torch.zeros(shape, device=env.device, dtype=torch.bool),
            missed_count=torch.zeros(shape, device=env.device, dtype=torch.long),
            complete=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    state = env._v111_adjacent
    for key in ('last_index', 'support_frames', 'ever_skipped', 'missed_count', 'complete'):
        state[key][reset] = 0
    state['observed_index'][reset] = -1


def sequence_step(state, x, lateral, bottom, force_z, lower, rise, tread,
                  half_width, riser_count, active, cfg, dt):
    """Confirm actual support on each successive tread, independently for FL/FR.

    A skip pays one failure event, is retained in the episode history, and
    resynchronizes the next physical target so later learning is still possible.
    Holding or revisiting a tread never earns another reward. No time budget,
    prescribed lead leg, bilateral wait, velocity multiplier or yaw command.
    """
    if not all(torch.isfinite(v).all() for v in (x, lateral, bottom, force_z, lower, rise, tread, half_width)):
        raise RuntimeError('Nonfinite adjacent-tread geometry')
    if ((active.any(1) & ((rise <= 0) | (tread <= 0)))).any():
        raise ValueError('Active adjacent treads require positive rise and tread')
    previous = state['last_index']
    expected = previous+1
    height_index = torch.round((bottom-lower[:, None])/rise[:, None].clamp_min(1.e-6)).long()
    plane = lower[:, None]+height_index*rise[:, None]
    margin = torch.minimum(torch.full_like(tread, cfg.edge_margin), tread*cfg.edge_fraction)[:, None]
    start = (height_index-1)*tread[:, None]+margin
    end = height_index*tread[:, None]-margin
    width = half_width[:, None]+(height_index-1).clamp_min(0)*tread[:, None]-cfg.lateral_margin
    surface = (active & (height_index >= 1) & (height_index <= riser_count[:, None])
        & (force_z >= cfg.contact_force) & ((bottom-plane).abs() <= cfg.plane_tolerance)
        & (x >= start) & (x <= end) & (lateral.abs() <= width))
    frames = torch.where(surface,
        torch.where(height_index == state['observed_index'], state['support_frames']+1, 1), 0)
    observed = torch.where(surface, height_index, -1)
    stable = surface & (frames >= math.ceil(cfg.support_seconds/dt))
    advancing = stable & (height_index > previous)
    skipped = (height_index-expected).clamp_min(0)*advancing
    # B1 already has its own contact reward; new credit begins at B2.
    event = advancing & (height_index == expected) & (expected >= 2)
    last = torch.where(advancing, height_index, previous)
    ever_skipped = state['ever_skipped'] | (skipped > 0)
    complete = (last >= riser_count[:, None]).all(1) & ~ever_skipped.any(1) & active.all(1)
    next_plane_lower = lower[:, None]+(expected-1)*rise[:, None]
    airborne = bottom > next_plane_lower+cfg.lift_min
    overshoot = ((x-(expected*tread[:, None]-margin)).clamp_min(0)
        / tread[:, None].clamp_min(1.e-6)).square().clamp_max(4)
    overshoot *= active & airborne & (expected >= 2) & (expected <= riser_count[:, None])
    return dict(last_index=last, observed_index=observed, support_frames=frames,
        ever_skipped=ever_skipped, missed_count=state['missed_count']+skipped,
        complete=complete, support_event=event, skip_event=skipped,
        complete_event=complete & ~state['complete'], overshoot=overshoot.amax(1),
        overshoot_side=overshoot, overshoot_target=expected)


def finish_sequence(env, asset):
    from .v135_terrain_geometry import v135_terrain_geometry
    s, cfg = env._v111_first_tread, env.cfg.v111_adjacent
    geometry = v135_terrain_geometry(env)
    wheel = asset.data.body_pos_w[:, env._v11_front_body_ids]
    x = ((wheel[..., :2]-s['anchor'][:, None])*s['forward'][:, None]).sum(-1)
    left = torch.stack((-s['forward'][:, 1], s['forward'][:, 0]), 1)
    lateral = ((wheel[..., :2]-env.scene.env_origins[:, None, :2])*left[:, None]).sum(-1)
    half_width = (left.abs()*geometry['radii']).sum(-1)-env.cfg.v11_perception.wheel_radius
    force = env.scene.sensors['contact_forces'].data.net_forces_w[:, env._v11_front_contact_ids, 2]
    active = (geometry['ascent'] & s['height_valid'])[:, None] & s['entry_approached']
    env._v111_adjacent.update(sequence_step(env._v111_adjacent, x, lateral,
        wheel[..., 2]-env.cfg.v11_perception.wheel_radius, force, s['lower'], s['height'],
        s['tread'], half_width, geometry['flight_riser_count'], active, cfg, float(env.step_dt)))
    if hasattr(env.cfg, 'v111_recovery'):
        from .v111_recovery import finish_recovery
        finish_recovery(env, asset, geometry)


def state(env):
    from .v11_continuous import v11_state
    v11_state(env)
    return env._v111_adjacent


def support_reward(env):
    return state(env)['support_event'].float().sum(1)/float(env.step_dt)


def skip_penalty(env):
    return -state(env)['skip_event'].float().sum(1)/float(env.step_dt)


def overshoot_penalty(env):
    return -state(env)['overshoot']


def complete_reward(env):
    return state(env)['complete_event'].float()/float(env.step_dt)


def straight_stability_penalty(env):
    from .v111_entry_flat import flat_state
    cfg = env.cfg.v111_adjacent
    robot = env.scene['robot']
    cmd = env.command_manager.get_command('base_velocity')
    straight = (cmd[:, 2].abs() < cfg.straight_yaw_deadband) & (cmd[:, 1].abs() < cfg.straight_lateral_deadband)
    yaw = ((robot.data.root_ang_vel_b[:, 2]-cmd[:, 2])/cfg.straight_yaw_scale).square().clamp_max(4)
    lateral = ((robot.data.root_lin_vel_b[:, 1]-cmd[:, 1])/cfg.straight_lateral_scale).square().clamp_max(4)
    return -(yaw+lateral)*straight*flat_state(env)['blend']


def metric(env, env_ids, key):
    return state(env)[key][env_ids].float().mean()
