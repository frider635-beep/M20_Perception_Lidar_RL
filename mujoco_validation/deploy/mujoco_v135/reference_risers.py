"""Frozen pure geometry reference for differential validation; no Isaac dependency."""
from __future__ import annotations
import torch

def extract_two_risers_points(
    points_yaw: torch.Tensor,
    *,
    corridor_half_width: float,
    h_min: float,
    d_end: float,
    min_separation: float,
    signed_edges: bool = False,
) -> dict[str, torch.Tensor]:
    """Return the current and next same-sign edge in the approach corridor.

    Outputs have shape ``[N, 2]`` ordered along +x.  The first slot retains a
    just-passed edge down to ``d_end``; the second slot must be farther forward
    by ``min_separation`` and have the same sign.  With ``signed_edges=False``
    this preserves the historical positive-riser contract.  With it enabled,
    positive heights denote an ascent and negative heights a descent.
    """
    if points_yaw.shape[1:] != (21, 15, 3):
        raise ValueError(f"expected points [N,21,15,3], got {tuple(points_yaw.shape)}")
    p = points_yaw.flip(1)
    finite = torch.isfinite(p[..., 2])
    low, up = p[:, :-1], p[:, 1:]
    pair_valid = finite[:, :-1] & finite[:, 1:]
    signed_height = up[..., 2] - low[..., 2]
    height = signed_height if signed_edges else torch.relu(signed_height)
    magnitude = height.abs() if signed_edges else height
    edge_x = 0.5 * (low[..., 0] + up[..., 0])
    edge_y = 0.5 * (low[..., 1] + up[..., 1])
    lower_z = torch.minimum(low[..., 2], up[..., 2])
    candidate = pair_valid & (edge_y.abs() <= float(corridor_half_width)) & (magnitude >= float(h_min))

    # Collapse each x row to one robust lateral peak so one physical riser cannot
    # occupy both output slots merely because several y rays hit it.
    masked_height = torch.where(candidate, magnitude, torch.full_like(magnitude, float("-inf")))
    y_index = masked_height.argmax(dim=2)
    row_valid = candidate.any(dim=2)
    gather = y_index.unsqueeze(-1)
    row_height = height.gather(2, gather).squeeze(-1)
    row_x = edge_x.gather(2, gather).squeeze(-1)
    row_y = edge_y.gather(2, gather).squeeze(-1)
    row_lower = lower_z.gather(2, gather).squeeze(-1)
    row_valid &= row_x > float(d_end)

    n = points_yaw.shape[0]
    batch = torch.arange(n, device=points_yaw.device)
    first_score = torch.where(row_valid, row_x.abs(), torch.full_like(row_x, float("inf")))
    first_index = first_score.argmin(dim=1)
    first_valid = row_valid.any(dim=1)
    first_x = row_x[batch, first_index]
    first_height = row_height[batch, first_index]
    same_sign = (row_height * first_height[:, None]) > 0.0 if signed_edges else torch.ones_like(row_valid)
    second_candidate = (
        row_valid & same_sign
        & (row_x > first_x[:, None] + float(min_separation))
        & first_valid[:, None]
    )
    second_score = torch.where(second_candidate, row_x, torch.full_like(row_x, float("inf")))
    second_index = second_score.argmin(dim=1)
    second_valid = second_candidate.any(dim=1)

    def pair(values: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
        a = torch.where(first_valid, values[batch, first_index], torch.full_like(first_x, fill))
        b = torch.where(second_valid, values[batch, second_index], torch.full_like(first_x, fill))
        return torch.stack((a, b), dim=1)

    return {
        "valid": torch.stack((first_valid, second_valid), dim=1),
        "riser_height": pair(row_height),
        "riser_distance": pair(row_x),
        "riser_x": pair(row_x),
        "riser_y": pair(row_y),
        "riser_lower_world": pair(row_lower),
    }
