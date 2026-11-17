"""V13.5-Final Actor Perception: strict P1/P2 for ascent, native signed edge for descent.

Key fix: V13.5 previously replaced all 6D perception with strict target, causing descent blindness.
Final uses:
- Ascent: strict P1/P2 from v135_target_actor_perception (already correct)
- Descent: native V11 signed edge (H<0)
- Flat: naturally zeroed

This allows:
- Ascent: strict riser identity and target tracking
- Descent: see actual downward edges
- GRU gate: H1>0 for ascent, H1<0 naturally disables
"""

from __future__ import annotations

import torch


def v135_final_actor_perception(env, sensor_cfg=None) -> torch.Tensor:
    """6D perception: ascent=strict, descent=native, flat=zero.

    Returns:
        [num_envs, 6]: [P1_feature_0, P1_feature_1, P1_feature_2,
                        P2_feature_0, P2_feature_1, P2_feature_2]

    Where each P_feature is [valid, height, distance]:
        - Ascent: from v135_target_actor_perception (strict P1/P2)
        - Descent: from V11 native signed edge (H<0)
        - Flat: ~[0, 0, 0]
    """
    from .v11_continuous import v11_state
    from .v135_riser_identity import v135_target_actor_perception

    base = v11_state(env) if sensor_cfg is None else v11_state(env, sensor_cfg=sensor_cfg)

    # Get strict P1/P2 from existing correct function
    strict_6d = v135_target_actor_perception(env, sensor_cfg)

    # Get V11 native perception (works for descent/flat)
    native_6d = base["actor_feature"]  # [num_envs, 6]

    # Determine terrain type
    terrain_types = env.scene.terrain.terrain_types
    ascending_ids = torch.tensor(
        tuple(int(v) for v in env.cfg.v11_perception.ascending_stair_terrain_type_ids),
        device=env.device, dtype=terrain_types.dtype,
    )
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    ascent = (terrain_types[:, None] == ascending_ids[None, :]).any(1) & (
        command_x > float(env.cfg.v11_perception.forward_command_threshold)
    )

    # Select: ascent uses strict, descent/flat use native
    perception = torch.where(
        ascent[:, None],
        strict_6d,
        native_6d,
    )

    return perception


__all__ = ["v135_final_actor_perception"]
