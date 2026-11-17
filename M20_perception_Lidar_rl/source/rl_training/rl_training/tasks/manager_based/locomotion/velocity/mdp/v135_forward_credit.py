"""Finite, current-riser forward credit; no held historical gait dividend."""
from __future__ import annotations

import torch


def bounded_forward_credit(position, best, remaining, quality, active, *, speed, dt):
    """Consume first-visit distance even when gait is bad; never bank it for later."""
    distance = (position - best).clamp_min(0)
    eligible = torch.minimum(distance, remaining.clamp_min(0)) * active.float()
    updated_best = torch.maximum(best, position)
    remaining = (remaining - eligible).clamp_min(0)
    fraction = (eligible / (speed.clamp_min(0.1) * dt)).clamp(0, 1)
    return updated_best, remaining, fraction * quality.clamp(0, 1)


def v135_current_riser_speed_factor(env, sensor_cfg=None):
    from .v135_riser_identity import v135_riser_identity_state
    from .v135_final_ascent_reward import v135_final_ascent_state

    strict = v135_riser_identity_state(env, sensor_cfg)["reward_snapshot"]
    progress = v135_final_ascent_state(env, sensor_cfg)
    step = int(env.common_step_counter)
    if not hasattr(env, "_v135_forward_credit"):
        env._v135_forward_credit = {
            "step": -1, "episode": env.episode_length_buf.clone(),
            "target": torch.full_like(strict["target_id"], -1),
            "best": torch.zeros(env.num_envs, device=env.device),
            "remaining": torch.zeros(env.num_envs, device=env.device),
            "factor": torch.ones(env.num_envs, device=env.device),
        }
    s = env._v135_forward_credit
    if s["step"] == step:
        return s["factor"]
    cfg = env.cfg.v135_riser_identity
    position = ((env.scene["robot"].data.root_pos_w[:, :2] - strict["target_world_xy"])
                * strict["forward_world_xy"]).sum(1)
    reset = env.episode_length_buf < s["episode"]
    changed = reset | (s["target"] != strict["target_id"])
    s["best"] = torch.where(changed, position, s["best"])
    s["remaining"] = torch.where(changed, strict["step_width"], s["remaining"])
    delta = torch.stack((progress["fl_delta"], progress["fr_delta"]), dim=1)
    total = delta.sum(1)
    leader = delta.amax(1)
    expected = delta.gather(1, strict["expected_side"].clamp(0, 1)[:, None]).squeeze(1)
    quality = torch.where(strict["seeded"], expected / total.clamp_min(1e-6),
                          (delta[:, 0] - delta[:, 1]).abs() / total.clamp_min(1e-6))
    # A past good stride alone never authorizes credit on a later ordinary stride.
    quality *= (leader > 0).float()
    terrain = env.scene.terrain.terrain_types
    ascent = torch.zeros_like(strict["target_valid"])
    for type_id in env.cfg.v11_perception.ascending_stair_terrain_type_ids:
        ascent |= terrain == int(type_id)
    speed = env.command_manager.get_command("base_velocity")[:, 0]
    active = ascent & strict["target_valid"] & strict["target_height_valid"] & (speed > 0)
    invalid = strict["wrong_side"] | strict["duplicate"] | strict["skip"] | strict["dual_skip"]
    quality *= (~invalid & ~changed).float()
    s["best"], s["remaining"], credit = bounded_forward_credit(
        position, s["best"], s["remaining"], quality, active, speed=speed, dt=float(env.step_dt))
    from .v135_descent_stance import v135_post_stair_flat_state
    blend = v135_post_stair_flat_state(env, sensor_cfg)["flat_blend"]
    from .v135_ascent_support import v135_ascent_support_state
    support = v135_ascent_support_state(env, sensor_cfg)
    # Preserve all base locomotion reward and at least half of existing ascent
    # bonus. No stationary payoff: credit already requires new forward distance.
    credit *= 0.5 + 0.5 * support["quality"]
    s["factor"] = 1.0 + (float(cfg.ordered_speed_multiplier) - 1.0) * credit * (1.0 - blend)
    s["factor"] = torch.where(active, s["factor"], torch.ones_like(s["factor"]))
    s["episode"] = env.episode_length_buf.clone()
    s["target"] = strict["target_id"].clone()
    s["step"] = step
    return s["factor"]
