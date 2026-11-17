"""One authoritative geometry lookup for V135 ascent variants (no new FSM)."""
from __future__ import annotations
import torch
from .v135_ascent_contract import inverted_flight_layout


def ascent_name(name):
    return name == "pyramid_stairs_inv" or name.startswith(("v135_ascent_tread_", "v112_ascent_tread_"))


def column_names(cfg):
    names = tuple(cfg.sub_terrains)
    weights = [float(cfg.sub_terrains[n].proportion) for n in names]
    total = sum(weights)
    if total <= 0 or not cfg.curriculum:
        raise ValueError("V135 geometry requires the curriculum generator's fixed column mapping")
    cumulative, value = [], 0.
    for weight in weights:
        value += weight / total
        cumulative.append(value)
    return tuple(next(n for n, p in zip(names, cumulative) if c / cfg.num_cols + .001 < p)
                 for c in range(cfg.num_cols))


def configure_ascent_treads(cfg, widths):
    """Split only the ascent columns, preserving every other column and mixture.

    The default six ascent columns contain two 30cm anchors and four variants.
    Each flight has constant tread, independently of the sampled rise/level.
    """
    old = column_names(cfg)
    columns = [i for i, n in enumerate(old) if n == "pyramid_stairs_inv"]
    if len(widths) != len(columns):
        raise ValueError(f"Expected {len(columns)} ascent tread widths, got {len(widths)}")
    source = cfg.sub_terrains["pyramid_stairs_inv"]
    if any(w <= .20 for w in widths):
        raise ValueError("Tread must accommodate the configured wheel/support landing window")
    items = {}
    for name, item in cfg.sub_terrains.items():
        if name != "pyramid_stairs_inv":
            items[name] = item
            continue
        for i, width in enumerate(widths):
            variant = item.copy()
            variant.step_width = float(width)
            variant.proportion = float(source.proportion) / len(widths)
            items[name if i == 0 else f"v135_ascent_tread_{i}"] = variant
    cfg.sub_terrains = items
    new = column_names(cfg)
    if any(a != b and not (ascent_name(a) and ascent_name(b)) for a, b in zip(old, new)):
        raise ValueError("Tread mixture changed a frozen non-ascent column")


def v135_terrain_geometry(env):
    """Gather per-env geometry by actual terrain column; curriculum row changes safe."""
    if not hasattr(env, "_v135_geometry_table"):
        cfg = env.cfg.scene.terrain.terrain_generator
        widths, radii, counts, ascent = [], [], [], []
        for name in column_names(cfg):
            sub = cfg.sub_terrains[name]
            is_up = ascent_name(name)
            if is_up:
                if sub.function.__name__ != "inverted_pyramid_stairs_terrain" or sub.holes:
                    raise ValueError("Unverified V135 ascent mesh")
                rad, count = inverted_flight_layout(cfg.size, float(sub.border_width),
                                                   float(sub.platform_width), float(sub.step_width))
            else:
                rad, count = (0., 0.), 0
            widths.append(float(getattr(sub, "step_width", .30)))
            radii.append(rad); counts.append(count); ascent.append(is_up)
        env._v135_geometry_table = dict(
            step_width=torch.tensor(widths, device=env.device),
            radii=torch.tensor(radii, device=env.device),
            flight_riser_count=torch.tensor(counts, device=env.device, dtype=torch.long),
            ascent=torch.tensor(ascent, device=env.device, dtype=torch.bool),
            max_risers=max(counts))
    table = env._v135_geometry_table
    ids = env.scene.terrain.terrain_types
    return {k: v[ids] if isinstance(v, torch.Tensor) else v for k, v in table.items()}
