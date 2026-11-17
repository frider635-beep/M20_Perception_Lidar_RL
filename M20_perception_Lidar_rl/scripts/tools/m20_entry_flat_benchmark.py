"""Physical MuJoCo benchmark: complete front-wheel tread sequences and flat reach.

Uses the validation runner's real robot, sensor, PD and saved checkpoint contract.
Contact labels are evaluation outputs only; they never enter policy observations.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def parse_args():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=repo/'interview_prep/model_35300.pt')
    p.add_argument('--validation-root', type=Path, default=repo.parent/'mujoco_validation')
    p.add_argument('--config-dir', type=Path)
    p.add_argument('--speeds', type=float, nargs='+', default=[.5, .7, 1.])
    p.add_argument('--treads', type=float, nargs='+', default=[.26, .30])
    p.add_argument('--height', type=float, default=.16)
    p.add_argument('--startup', choices=('standing', 'z'), default='standing')
    p.add_argument('--perception', choices=('clean', 'noisy'), default='clean')
    p.add_argument('--stair-seconds', type=float, default=14., help='Motion duration for the complete-flight support sequence')
    p.add_argument('--keyboard-profile', action='store_true', help='Z, C, two seconds zero command, then held 8 using the production command ramp; requires --startup z --speeds .7')
    p.add_argument('--no-trajectories', action='store_true', help='Keep manifest and results without NPZ trajectories')
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args()


def update_touchdown(supported, prior, frames, first):
    import numpy as np
    frames = np.where(supported > 0, np.where(supported == prior, frames+1, 1), 0)
    first = np.where((first == 0) & (frames >= 3), supported, first)
    return frames, first


def support_sequence_report(labels, frames_required=3):
    """Retain every forward gap; touching a skipped tread later cannot erase it."""
    sequences, gaps = [], []
    for side in range(labels.shape[1]):
        last, frames, high = -1, 0, 0
        sequence, missing = [], []
        for value in labels[:, side]:
            index = int(value)
            frames = frames+1 if index == last else 1
            last = index
            if index <= 0 or frames != frames_required:
                continue
            if sequence and sequence[-1] == index:
                continue
            sequence.append(index)
            if index > high:
                missing.extend(range(high+1, index))
                high = index
        sequences.append(sequence)
        gaps.append(missing)
    return dict(front_support_sequences=sequences, skipped_treads=gaps,
                no_observed_tread_skip=bool(all(sequence for sequence in sequences) and not any(gaps)))


def top_support_candidate(wheel_position, wheel_force, root_position, upright, *,
                          top_height, goal_x, wheel_radius, width):
    """Four supported wheels and an upright, clear body, not an airborne crossing."""
    import numpy as np
    wheel_position, wheel_force, root_position = map(np.asarray, (wheel_position, wheel_force, root_position))
    if not all(np.isfinite(v).all() for v in (wheel_position, wheel_force, root_position, upright)):
        return False
    return bool((wheel_position[:, 0] > goal_x + wheel_radius).all()
        and (np.abs(wheel_position[:, 1]) <= width / 2).all()
        and (wheel_force[:, 2] >= 15.).all()
        and (np.abs(wheel_position[:, 2] - wheel_radius - top_height) <= .035).all()
        and upright >= .8 and root_position[2] - top_height >= .30)


def route_progress_metrics(times, positions, rear_positions, active, command_speed, tread):
    """Measure world-route progress over windows; rotation and wheel spin do not count."""
    import numpy as np
    times, positions, rear_positions, active = map(np.asarray, (times, positions, rear_positions, active))
    window_seconds = max(.8, 2.5 * (tread or .26) / max(command_speed, .2))
    ratios, rear_ratios, stalled = [], [], []
    for end in range(len(times)):
        start = int(np.searchsorted(times, times[end] - window_seconds, side='left'))
        elapsed = times[end] - times[start]
        if elapsed < window_seconds - .03 or not active[start:end+1].all():
            continue
        ratio = float((positions[end] - positions[start]) / (elapsed * command_speed))
        rear_ratio = float((rear_positions[end] - rear_positions[start]) / (elapsed * command_speed))
        ratios.append(ratio)
        rear_ratios.append(rear_ratio)
        stalled.append(ratio < .2 and rear_ratio < .2)
    return dict(progress_window_seconds=window_seconds,
        minimum_route_speed_ratio=min(ratios) if ratios else None,
        minimum_rear_route_speed_ratio=min(rear_ratios) if rear_ratios else None,
        stall_detected=bool(any(stalled)), stall_window_count=sum(stalled),
        progress_windows=len(ratios))


def run(a):
    import numpy as np
    import torch
    import mujoco
    sys.path.insert(0, str(a.validation_root.resolve()/'deploy/mujoco_v135'))
    from policy import SavedPolicy
    from run import Robot
    from interaction import Interaction
    from teleop import TeleopKeyboard
    from unittest.mock import patch
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    saved = SavedPolicy(a.checkpoint, a.config_dir)
    base = a.validation_root.resolve()/'deploy/mujoco_v135'
    manifest = dict(saved.metadata, schema_version=2, profile='z_c_8' if a.keyboard_profile else a.startup,
                    phase_labels={'0': 'nonpolicy', '1': 'policy_idle', '2': 'moving', '3': 'after_motion'}, speeds=a.speeds, treads=a.treads, height=a.height,
                    startup=a.startup, perception=a.perception,
                    physics_dt=.001, finish_support_seconds=.06, finish_min_upright=.8, finish_min_clearance=.30, support_force=15., support_normal_min=.9,
                    contact_plane_tolerance=.025, support_frames=3,
                    flat_measurement_window='motion start +2s through +12s; motion start is 1s standing, 5.4s Z, or 7.4s Z+C+8',
                    first_landing='first top-surface support persisting 3 policy frames per front wheel',
                    stair_motion_seconds=a.stair_seconds,
                    sequence_criterion='each front wheel supports every tread in ascending order; later contact cannot erase a gap',
                    keyboard_profile=a.keyboard_profile,
                    seed=42, physics_enabled=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = []
    for terrain, tread in [('flat', None), *(('ascent', w) for w in a.treads)]:
        for speed in a.speeds:
            args = SimpleNamespace(robot_xml=base/'assets/m20/mjcf/M20.xml',
                terrain_xml=base/'terrains'/('flat.xml' if terrain == 'flat' else 'stairs_ascent_walls.xml'),
                physics_dt=.001, stair_height=a.height if tread else None,
                tread_depth=tread, stair_count=None, stair_start=None, spawn_xy=None,
                wall_alpha=.25, perception=a.perception, seed=42)
            r = Robot(saved, args)
            keyboard = None
            if a.keyboard_profile:
                if a.startup != 'z' or speed != .7:
                    raise ValueError('Keyboard profile requires --startup z --speeds .7 (the saved 8 binding)')
                class Keys:
                    held = set()
                    def focused(self): return True
                    def pressed(self): return self.held
                    def close(self): pass
                # Exercise production command generation without sending input
                # to the desktop or relying on a real focused X11 window.
                with patch('teleop.X11WindowKeys.open', return_value=None):
                    keyboard = TeleopKeyboard()
                keyboard._window_keys = Keys()
            session = Interaction(r, keyboard=keyboard) if a.startup == 'z' else None
            if session:
                session.event('z')
            scheduled_policy_start = 5.4 if session else 1.
            policy_start = None if session else scheduled_policy_start
            motion_start = None if session else policy_start
            hip_ids = [r.model.body(f'{leg}_hipy').id for leg in ('fl', 'fr', 'hl', 'hr')]
            last, frames, first = np.zeros(16), np.zeros(4, dtype=int), np.zeros(4, dtype=int)
            prior = np.zeros(4, dtype=int)
            top_frames, maximum_top_frames, bypass = 0, 0, False
            records = {k: [] for k in ('time', 'position', 'quaternion', 'velocity', 'angular_velocity', 'reach', 'joint_pos', 'joint_vel', 'observation', 'action', 'wheel_position', 'wheel_force', 'torque', 'supported_tread', 'command', 'policy_active', 'phase')}
            fell = False
            termination = 'duration'
            motion_seconds = 12. if terrain == 'flat' else a.stair_seconds
            duration = scheduled_policy_start+(2. if keyboard else 0.)+motion_seconds+(2. if terrain == 'flat' else 0.)+3.
            for step in range(round(duration/saved.period)):
                t = float(r.data.time)
                if session:
                    session.update()
                    if t >= scheduled_policy_start and session.mode == 'ready':
                        session.event('c')
                        policy_start = t
                        motion_start = t+(2. if keyboard else 0.)
                active = session.mode == 'policy' if session else t >= motion_start
                moving_now = active and motion_start is not None and motion_start <= t < motion_start+motion_seconds
                if motion_start is not None and t >= motion_start+motion_seconds+(2. if terrain == 'flat' else 0.):
                    break
                phase = (2 if moving_now else 1 if active and t < motion_start else 3 if active else 0)
                command = np.asarray([speed if moving_now else 0., 0., 0.])
                if keyboard:
                    keyboard._window_keys.held = {'8'} if command[0] else set()
                    command = keyboard.velocity_command()
                obs = r.observe(command, last)
                action = session.action(obs) if session else saved(obs)
                for _ in range(round(saved.period/args.physics_dt)):
                    if not active:
                        r.control_pose(session.pose_target() if session else saved.nominal)
                    else:
                        r.control(action)
                    mujoco.mj_step(r.model, r.data)
                mujoco.mj_forward(r.model, r.data)
                last = action if active else np.zeros(16)
                if session:
                    session.previous = last.copy()
                supported = np.full(4, -1, dtype=int)
                wheel_force = np.zeros((4, 3))
                for i, contact in enumerate(r.data.contact):
                    geom1, geom2 = map(int, contact.geom)
                    terrain_geom, robot_geom = ((geom1, geom2) if geom1 in r.terrain_geoms else (geom2, geom1))
                    if terrain_geom not in r.terrain_geoms or robot_geom < 0:
                        continue
                    body = r.model.geom_bodyid[robot_geom]
                    wheel_ids = np.flatnonzero(r.wheels == body)
                    force = np.zeros(6)
                    mujoco.mj_contactForce(r.model, r.data, i, force)
                    # Contact frame's first axis points geom1 -> geom2.
                    # The normal force on geom2 has the positive direction.
                    if len(wheel_ids):
                        sign = 1 if robot_geom == geom2 else -1
                        wheel_force[wheel_ids[0]] += sign*(contact.frame.reshape(3, 3).T@force[:3])
                    if abs(contact.frame[2]) < .9:
                        continue
                    sides = np.flatnonzero(r.wheels == body)
                    if not len(sides):
                        continue
                    height = a.height if tread else 1.
                    index = round(contact.pos[2]/height)
                    if force[0] >= 15. and abs(contact.pos[2]-index*height) < .025:
                        supported[sides[0]] = index
                frames, first = update_touchdown(supported, prior, frames, first)
                prior = supported.copy()
                forward = r.data.xmat[r.base_id].reshape(3, 3)[:, 0]
                reach = (r.data.xpos[r.wheels]-r.data.xpos[hip_ids]) @ forward
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(r.model, r.data, mujoco.mjtObj.mjOBJ_BODY, r.base_id, velocity, 1)
                for key, value in dict(time=t, position=r.data.xpos[r.base_id].copy(),
                    quaternion=r.data.xquat[r.base_id].copy(), velocity=velocity[3:].copy(),
                    angular_velocity=velocity[:3].copy(), reach=reach, joint_pos=r.data.qpos[r.qids].copy(),
                    joint_vel=r.data.qvel[r.vids].copy(), observation=obs.copy(), action=action.copy(),
                    wheel_position=r.data.xpos[r.wheels].copy(), wheel_force=wheel_force,
                    torque=r.data.ctrl[r.aids].copy(),
                    supported_tread=supported, command=command, policy_active=active, phase=phase).items():
                    records[key].append(value)
                if active and (r.data.xpos[r.base_id, 2]-r.ground() < .22 or r.data.xmat[r.base_id].reshape(3, 3)[2, 2] < .35):
                    fell = True
                    termination = 'fall'
                    break
                bypass |= bool(active and abs(r.data.xpos[r.base_id, 1]) > r.terrain.course_width / 2
                               and r.data.xpos[r.base_id, 0] < r.terrain.goal_x)
                if tread:
                    count = int(r.terrain.parameters['step_count'][0])
                    supported_top = top_support_candidate(r.data.xpos[r.wheels], wheel_force, r.data.xpos[r.base_id],
                        r.data.xmat[r.base_id].reshape(3, 3)[2, 2], top_height=count*a.height,
                        goal_x=r.terrain.goal_x, wheel_radius=saved.perception['wheel_radius'], width=r.terrain.course_width)
                    top_frames = top_frames+1 if supported_top and not bypass else 0
                    maximum_top_frames = max(maximum_top_frames, top_frames)
                    if top_frames*saved.period >= .06:
                        termination = 'flight_complete'
                        break
            arrays = {k: np.asarray(v) for k, v in records.items()}
            policy_entered = policy_start is not None
            if not policy_entered:
                termination = 'startup_failed'
                policy_start = motion_start = duration
            window = (arrays['time'] >= motion_start+2) & (arrays['time'] < motion_start+12)
            reach = arrays['reach'][window]
            q = arrays['quaternion']
            yaw = np.unwrap(np.arctan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2)))
            initial_yaw = yaw[np.searchsorted(arrays['time'], motion_start).clip(0, len(yaw)-1)]
            idle_start_yaw = yaw[np.searchsorted(arrays['time'], policy_start).clip(0, len(yaw)-1)]
            moving = (arrays['time'] >= motion_start) & (arrays['time'] < motion_start+motion_seconds)
            later = ((arrays['position'][:, 0] > r.terrain.parameters['start_x'][0]+2*tread)
                & (arrays['position'][:, 0] < r.terrain.goal_x)) if tread else np.zeros(len(window), dtype=bool)
            route_mask = later & moving if tread else window
            progress = route_progress_metrics(arrays['time'], arrays['position'][:, 0],
                arrays['wheel_position'][:, 2:, 0].min(1), route_mask, speed, tread)
            motion_index = np.searchsorted(arrays['time'], motion_start).clip(0, len(yaw)-1)
            lateral = arrays['position'][:, 1]-arrays['position'][motion_index, 1]
            idle = (arrays['time'] >= policy_start) & (arrays['time'] < motion_start)
            result = dict(schema_version=2, profile='z_c_8' if keyboard else a.startup,
                policy_start=policy_start if policy_entered else None, motion_start=motion_start if policy_entered else None,
                policy_entered=policy_entered, terrain=terrain, tread=tread, command_speed=speed, fell=fell, termination=termination,
                observed_bypass=bypass, whole_body_top_supported=termination == 'flight_complete',
                maximum_top_support_seconds=maximum_top_frames*saved.period,
                after_c_idle_yaw_max_change_deg=float(np.degrees(np.abs(yaw[idle]-idle_start_yaw).max())) if idle.any() else 0.,
                forward_lateral_max_abs=float(np.abs(lateral[moving]).max()) if moving.any() else None,
                route_mean_velocity_x=float((arrays['position'][window][-1, 0]-arrays['position'][window][0, 0])
                    /(arrays['time'][window][-1]-arrays['time'][window][0])) if window.sum()>1 else None,
                **progress,
                first_supported_tread=first[:2].tolist() if tread else None,
                both_front_first_tread=bool((first[:2] == 1).all()) if tread else None,
                any_front_skipped=bool((first[:2] > 1).any()) if tread else None,
                simulated_seconds=float(r.data.time), final_position=r.data.xpos[r.base_id].tolist(),
                mean_velocity_x=float(arrays['velocity'][window, 0].mean()) if window.any() else None,
                later_stairs_mean_velocity_x=float(arrays['velocity'][later, 0].mean()) if later.any() else None,
                later_stairs_route_velocity_x=float(np.gradient(arrays['position'][:, 0], arrays['time'])[later].mean()) if later.any() else None,
                mean_reach_per_leg=reach.mean(0).tolist() if len(reach) else None,
                p95_max_abs_reach=float(np.quantile(np.abs(reach).max(1), .95)) if len(reach) else None)
            result.update(command_yaw_max_abs=float(np.abs(arrays['command'][:, 2]).max()),
                after_c_idle_yaw_change_deg=float(np.degrees(initial_yaw-idle_start_yaw)),
                forward_yaw_max_change_deg=float(np.degrees(np.abs(yaw[moving]-initial_yaw).max())) if moving.any() else None,
                mean_yaw_rate=float(arrays['angular_velocity'][window, 2].mean()) if window.any() else None)
            if tread:
                result.update(support_sequence_report(arrays['supported_tread'][:, :2]))
                rear_report = support_sequence_report(arrays['supported_tread'][:, 2:])
                result['rear_support_sequences'] = rear_report['front_support_sequences']
                result['rear_skipped_treads'] = rear_report['skipped_treads']
                result['rear_no_observed_tread_skip'] = rear_report['no_observed_tread_skip']
                count = int(r.terrain.parameters['step_count'][0])
                result['all_treads_supported'] = (termination == 'flight_complete'
                    and result['no_observed_tread_skip']
                    and all(set(range(1, count+1)).issubset(sequence) for sequence in result['front_support_sequences']))
            tag = f'{terrain}_{tread}_{speed}'
            if not a.no_trajectories:
                np.savez_compressed(a.output/(tag+'.npz'), **arrays)
            results.append(result)
            print(json.dumps(result), flush=True)
            (a.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    run(parse_args())
