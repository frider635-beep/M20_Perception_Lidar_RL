"""V11.1 exit visibility: ignore passed edges, never missing forward geometry."""
import torch


def riser_masks(edge_xy, valid, wheel_xy, root_xy, forward, radius):
    edge_s = ((edge_xy - root_xy[:, None]) * forward[:, None]).sum(-1)
    wheel_s = ((wheel_xy - root_xy[:, None]) * forward[:, None]).sum(-1)
    # A riser underneath any wheel still prevents flat recovery. Forward
    # coverage and platform coplanarity are checked separately by V13.5.
    relevant = valid & (edge_s >= wheel_s.amin(1)[:, None] - radius)
    upcoming = valid & (edge_s > radius)
    return ~relevant.any(1), upcoming.any(1)


def directional_riser_masks(env, base):
    robot = env.scene['robot']
    heading = env.command_manager.get_term('base_velocity').v11_path_heading_w
    forward = torch.stack((heading.cos(), heading.sin()), 1)
    cfg = env.cfg.v11_perception
    pool = base['riser_pool']
    valid = pool['valid'] & (pool['riser_height'].abs() <= cfg.max_step_height + cfg.height_valid_tolerance)
    return riser_masks(base['riser_pool_world_xy'], valid,
                       robot.data.body_pos_w[:, env._v135_wheel_body_ids, :2],
                       robot.data.root_pos_w[:, :2], forward, float(cfg.wheel_radius))
