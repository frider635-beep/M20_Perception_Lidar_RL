"""Discrete, constant-per-flight widths with verified terrain column coverage."""
from .v135_terrain_geometry import column_names


def configure_treads(cfg, widths):
    widths = tuple(float(w) for w in widths)
    if not widths or min(widths) <= 0 or len(set(widths)) != len(widths):
        raise ValueError("Tread widths must be positive and distinct")
    items = {}
    for name, source in cfg.sub_terrains.items():
        if name not in ("pyramid_stairs", "pyramid_stairs_inv"):
            items[name] = source
            continue
        direction = "ascent" if name == "pyramid_stairs_inv" else "descent"
        for index, width in enumerate(widths):
            variant = source.copy()
            variant.step_width = width
            variant.proportion = float(source.proportion) / len(widths)
            items[name if index == 0 else f"v112_{direction}_tread_{index}"] = variant
    cfg.sub_terrains = items
    names = column_names(cfg)
    for direction in ("ascent", "descent"):
        original = "pyramid_stairs_inv" if direction == "ascent" else "pyramid_stairs"
        actual = {items[name].step_width for name in names
                  if name == original or name.startswith(f"v112_{direction}_")}
        if actual != set(widths):
            raise ValueError(f"Insufficient terrain columns for {direction}: {actual}")
    stairs = tuple(i for i, name in enumerate(names)
                   if name.startswith("v112_") or name in ("pyramid_stairs", "pyramid_stairs_inv"))
    ascent = tuple(i for i, name in enumerate(names)
                   if name == "pyramid_stairs_inv" or name.startswith("v112_ascent_"))
    return stairs, ascent
