"""Find saved configuration for original or relocated checkpoints without guessing."""
from pathlib import Path
import hashlib


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_configs(directory):
    missing = [name for name in ("env.yaml", "agent.yaml") if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved training configuration in {directory}: {', '.join(missing)}. "
            "Keep both files in the checkpoint's params/ directory, or pass "
            "--config-dir /path/to/original/run/params. Do not substitute another model's configuration.")
    return directory


def resolve_configs(checkpoint, config_dir=None, *, search_roots=None):
    checkpoint = Path(checkpoint).resolve(strict=True)
    if config_dir is not None:
        return require_configs(Path(config_dir).resolve()), "explicit", None
    adjacent = checkpoint.parent / "params"
    if adjacent.exists():
        return require_configs(adjacent), "adjacent", None
    if search_roots is None:
        # Only search the known training-log layout, never a recursive disk scan.
        search_roots = set()
        for start in (checkpoint.parent, Path(__file__).resolve().parent):
            for parent in (start, *start.parents):
                for suffix in ("logs/rsl_rl", "M20_perception_Lidar_rl/logs/rsl_rl"):
                    root = parent / suffix
                    if root.is_dir():
                        search_roots.add(root.resolve())
    target_digest = digest(checkpoint)
    matches = {}
    for root in sorted(map(Path, search_roots)):
        for candidate in sorted(root.glob(f"*/*/{checkpoint.name}")):
            if not candidate.is_file() or candidate.stat().st_size != checkpoint.stat().st_size:
                continue
            if digest(candidate) != target_digest:
                continue
            params = candidate.parent / "params"
            if all((params / name).is_file() for name in ("env.yaml", "agent.yaml")):
                identity = tuple(digest(params / name) for name in ("env.yaml", "agent.yaml"))
                matches.setdefault(identity, (params, candidate.resolve()))
    if len(matches) > 1:
        paths = ", ".join(str(value[0]) for value in matches.values())
        raise ValueError(f"Identical checkpoint has conflicting saved configurations: {paths}. "
                         "Select the original training configuration with --config-dir.")
    if matches:
        params, source = next(iter(matches.values()))
        return params, "checkpoint_sha256", source
    raise FileNotFoundError(
        f"No saved training configuration for {checkpoint}. Expected {adjacent}; "
        "no byte-identical checkpoint with complete params/ was found in local training logs. "
        "Pass --config-dir /path/to/original/run/params containing env.yaml and agent.yaml.")
