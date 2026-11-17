"""Positive platform evidence; missing edges alone never imply stair exit."""
import torch


def platform_exit_evidence(wheel_xy, edge_xy, edge_seen, root_xy, forward, hits,
                           no_riser, *, radius, plane_tolerance, lookahead, half_width):
    beyond = (((wheel_xy - edge_xy[:, None]) * forward[:, None]).sum(-1) > radius).all(1)
    offset = hits[..., :2] - root_xy[:, None]
    ahead = (offset * forward[:, None]).sum(-1)
    side = torch.stack((-forward[:, 1], forward[:, 0]), 1)
    lateral = (offset * side[:, None]).sum(-1)
    corridor = (torch.isfinite(hits).all(-1) & (ahead >= 0) & (ahead <= lookahead)
                & (lateral.abs() < half_width))
    zmax = hits[..., 2].masked_fill(~corridor, -torch.inf).amax(1)
    zmin = hits[..., 2].masked_fill(~corridor, torch.inf).amin(1)
    coverage = ((corridor & (ahead > .75 * lookahead)).any(1)
                & (corridor & (ahead < .25 * lookahead)).any(1))
    return edge_seen & beyond & no_riser & coverage & ((zmax - zmin) <= plane_tolerance)
