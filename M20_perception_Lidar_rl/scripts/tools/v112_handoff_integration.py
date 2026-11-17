"""Execute production landing state on all tread widths; no optimizer updates."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import copy
import math
from itertools import product
import torch


def check_handoff(cfg):
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import v11_continuous as m
    from rl_training.tasks.manager_based.locomotion.velocity.mdp import v135_terrain_geometry as geo
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.v112_handoff import support_bounds, reward_factor
    cfg = copy.deepcopy(cfg)
    cfg.v111_rear_posture.physical_first_tread = False
    names = ['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel', 'fl_hipy', 'fl_knee', 'fr_hipy', 'fr_knee']
    def find(name): return [names.index(name)], [name]

    def fixture(width, yaw=.2617993878, course=0.):
        body_yaw = course + yaw
        asset = NS(find_bodies=find, data=NS(root_pos_w=torch.zeros(1, 3),
            root_quat_w=torch.tensor([[math.cos(body_yaw / 2), 0., 0., math.sin(body_yaw / 2)]]), body_pos_w=torch.zeros(1, 8, 3),
            body_lin_vel_w=torch.zeros(1, 8, 3), body_ang_vel_w=torch.zeros(1, 8, 3)))
        forces = torch.zeros(1, 8, 3); forces[..., 2] = 100
        env = NS(num_envs=1, device='cpu', cfg=cfg, step_dt=.02,
            episode_length_buf=torch.ones(1, dtype=torch.long),
            scene=NS(sensors={'contact_forces': NS(find_bodies=find, data=NS(net_forces_w=forces))}),
            command_manager=NS(get_term=lambda _: NS(v11_path_heading_w=torch.tensor([body_yaw]))))
        def frame(x, z, contact=(True, True), edge=1., slip=0.):
            env.episode_length_buf += 1
            lateral = torch.tensor([.2, -.2])
            asset.data.body_pos_w[:, :2, 0] = math.cos(course) * torch.tensor(x) - math.sin(course) * lateral
            asset.data.body_pos_w[:, :2, 1] = math.sin(course) * torch.tensor(x) + math.cos(course) * lateral
            asset.data.body_pos_w[:, :2, 2] = torch.tensor(z) + cfg.v11_perception.wheel_radius
            asset.data.body_lin_vel_w[:, :2, 0] = slip
            forces[:, :2, 2] = torch.tensor(contact).float() * 100
            level = round((edge - 1) / width) * .18
            edges = torch.tensor([[edge, edge + width]])
            risers = dict(valid=torch.tensor([[True, True]]), riser_x=math.cos(yaw) * edges,
                riser_y=-math.sin(yaw) * edges, riser_height=torch.full((1, 2), .18),
                riser_lower_world=torch.tensor([[level, level + .18]]))
            with patch.object(geo, 'v135_terrain_geometry', lambda _: {'step_width': torch.tensor([width])}):
                return m._v11_bilateral_landing_state(env, asset, risers, torch.tensor([True]))
        return env, frame

    with patch.object(m, 'classify_vertical_riser_contact', lambda *a, **k: torch.zeros(1, 2, dtype=torch.bool)):
        for width in cfg.v112_tread_widths:
            low, high, valid = support_bounds(torch.tensor([1.]), torch.tensor([width]), cfg.v112_handoff)
            assert valid.item() and high.item() > low.item()
            for side, course in product((0, 1), (0., math.pi / 2, math.pi, -math.pi / 2)):
                env, frame = fixture(width, course=course)
                # Two consecutive risers with the SAME leader: alternation is not required.
                for stair in range(2):
                    edge, floor = 1. + stair * width, stair * .18
                    x, z, c = [edge - width / 2] * 2, [floor] * 2, [True, True]
                    frame(x, z, c, edge)
                    x[side] = edge - .01; z[side] = floor + .25; c[side] = False; frame(x, z, c, edge)
                    x[side] = edge + .01; frame(x, z, c, edge)
                    x[side] = edge + width / 2; z[side] = floor + .18; c[side] = True
                    for _ in range(12): state = frame(x, z, c, edge)
                    assert state['cycle_ready'].item(), (width, side, stair, 'no leader dwell',
                        {k: state[k].tolist() for k in ('cycle_leader', 'cycle_bad', 'cycle_lead_steps',
                         'support_steps', 'crossing', 'candidate_ready', 'wheel_yaw', 'upper_z', 'target_active')})
                    mate = 1 - side
                    x[mate] = edge - .01; z[mate] = floor + .25; c[mate] = False
                    frame(x, z, c, edge + width)  # scanner advances, physical target must stay
                    x[mate] = edge + .01; frame(x, z, c, edge + width)
                    x[mate] = edge + width / 2; z[mate] = floor + .18; c[mate] = True
                    paid = 0
                    # Use original scan after completing to avoid advancing into a third target.
                    for _ in range(12):
                        state = frame(x, z, c, edge)
                        paid += int(state['cycle_event'])
                    assert paid == 1 and state['cycle_done'].item(), (width, side, stair, paid)
                    for _ in range(10):
                        state = frame(x, z, c, edge)
                        assert state['cycle_delta'].item() == 0 and not state['bilateral_landing_event'].item()
                skipped_edge = 1. + 3 * width
                state = frame(x, z, c, skipped_edge)
                assert state['adjacent_credit_blocked'].item(), 'scanner skipped a whole tread without blocking credit'
                assert not state['bilateral_landing_event'].item()
                env.episode_length_buf.zero_()
                state = frame([.8, .8], [0., 0.])
                assert state['cycle_leader'].item() == -1

            # Simultaneous arrival, premature following, leader skipping B2, slip.
            for mode in ('simultaneous', 'early', 'skip', 'slip'):
                env, frame = fixture(width)
                frame([.8, .8], [0., 0.])
                if mode == 'simultaneous':
                    frame([.99, .99], [.25, .25], (False, False))
                    frame([1.01, 1.01], [.25, .25], (False, False))
                    events = physical = 0
                    for _ in range(20):
                        state = frame([1 + width / 2] * 2, [.18, .18])
                        events += int(state['cycle_event'])
                        physical += int(state['bilateral_landing_event'])
                    assert physical == 1, 'physical completion erased by style failure'
                    assert state['candidate_ready'].all(), 'negative control did not reach safe crossing'
                    assert state['cycle_bad'].item() and events == 0
                    state = frame([1 + width / 2] * 2, [.18, .18], edge=1 + width)
                    assert not state['adjacent_credit_blocked'].item(), 'physical step treated as missed due to failed style'
                    continue
                frame([.99, .8], [.25, 0.], (False, True))
                frame([1.01, .8], [.25, 0.], (False, True))
                x = [1 + width / 2, .8]
                for _ in range(4 if mode == 'early' else 12):
                    state = frame(x, [.18, 0.], slip=.5 if mode == 'slip' else 0.)
                # The follower always has a safe swing/crossing; the only
                # invalid conditions are timing, leader departure, or slip.
                x[1] = .99
                frame(x, [.18, .25], (True, False), slip=.5 if mode == 'slip' else 0.)
                x[1] = 1.01
                frame(x, [.18, .25], (True, False), slip=.5 if mode == 'slip' else 0.)
                if mode == 'skip':
                    x[0] = 1 + width + width / 2
                x[1] = 1 + width / 2
                events = 0
                for _ in range(15):
                    state = frame(x, [.18, .18], slip=.5 if mode == 'slip' else 0.)
                    events += int(state['cycle_event'])
                assert state['candidate_ready'].all(), (width, mode, 'unrelated safety gate masked negative')
                assert events == 0, (width, mode, 'invalid completion paid')
    factor = reward_factor(torch.tensor([0., 1., 2.]), torch.ones(3), cfg.v112_handoff.flat_drive_scale)
    assert factor[0] == 1 and factor[2] < factor[1] < 1
    assert torch.equal(reward_factor(torch.tensor([0., 1., 2.]), torch.zeros(3), 1.), torch.ones(3))
    print('V112_PRODUCTION_HANDOFF_7_WIDTHS_BOTH_LEADERS_REPEAT_LEADER_NEGATIVES: PASS', flush=True)
