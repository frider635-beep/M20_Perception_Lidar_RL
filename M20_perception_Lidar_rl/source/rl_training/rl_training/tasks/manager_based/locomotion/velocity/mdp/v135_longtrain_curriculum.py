"""Shared schedules for the V13.5 uninterrupted long-training task."""

from __future__ import annotations

import torch


def v135_longtrain_update(env) -> float:
    """Return the completed PPO-update equivalent from control-step time."""
    cfg = env.cfg.v135_longtrain
    steps = max(1, int(cfg.rollout_steps_per_update))
    return float(getattr(env, "common_step_counter", 0)) / float(steps)


def v135_linear_fraction(env) -> float:
    """0 at initialization and 1 after the configured anneal horizon."""
    horizon = max(1.0, float(env.cfg.v135_longtrain.anneal_updates))
    return min(max(v135_longtrain_update(env) / horizon, 0.0), 1.0)


def v135_early_mixture_probability(env) -> float:
    cfg = env.cfg.v135_longtrain
    return float(cfg.early_mixture_probability) * (1.0 - v135_linear_fraction(env))


def v135_ascent_error_scale(env) -> float:
    """Delay full wrong/skip pressure until after useful interaction emerges."""
    cfg = env.cfg.v135_longtrain
    update = v135_longtrain_update(env)
    start = float(cfg.ascent_penalty_ramp_start_update)
    end = max(start + 1.0, float(cfg.anneal_updates))
    fraction = min(max((update - start) / (end - start), 0.0), 1.0)
    return float(cfg.initial_penalty_scale) + (1.0 - float(cfg.initial_penalty_scale)) * fraction


def v135_descent_error_scale(env) -> float:
    cfg = env.cfg.v135_longtrain
    fraction = v135_linear_fraction(env)
    return float(cfg.initial_penalty_scale) + (1.0 - float(cfg.initial_penalty_scale)) * fraction


def v135_descent_velocity_baseline(env) -> float:
    cfg = env.cfg.v135_longtrain
    fraction = v135_linear_fraction(env)
    return float(cfg.descent_velocity_baseline_initial) + fraction * (
        float(cfg.descent_velocity_baseline_final) - float(cfg.descent_velocity_baseline_initial)
    )


def v135_longtrain_metric(env, env_ids, key: str):
    """Expose the shared schedule without feeding it to policy or value inputs."""
    del env_ids
    if key == "fraction":
        value = v135_linear_fraction(env)
    elif key == "early_mixture_probability":
        value = v135_early_mixture_probability(env)
    elif key == "ascent_penalty_scale":
        value = v135_ascent_error_scale(env)
    elif key == "descent_penalty_scale":
        value = v135_descent_error_scale(env)
    elif key == "descent_velocity_baseline":
        value = v135_descent_velocity_baseline(env)
    else:
        raise ValueError(f"Unknown V13.5 LongTrain metric: {key}")
    # ``episode_length_buf`` is an integer tensor.  ``new_tensor(value)``
    # therefore truncated every fractional schedule value (0.70, 0.25,
    # 0.85, ...) to zero before the later ``.float()`` conversion.  Build
    # the scalar with an explicit floating dtype so TensorBoard reports the
    # same schedule used by reset, command, and reward code.
    return torch.tensor(value, device=env.device, dtype=torch.float32)


__all__ = [name for name in globals() if name.startswith("v135_")]
