"""Canonical paths for per-episode training artifacts.

New runs keep monitor and reward streams below ``episodes/``.  Readers retain
fallback support for older runs that wrote ``episodes*.monitor.csv`` directly
in the run directory.
"""
from pathlib import Path


def episodes_directory(output):
    return Path(output) / "episodes"


def monitor_directory(output):
    return episodes_directory(output) / "monitors"


def reward_directory(output):
    return episodes_directory(output) / "rewards"


def ensure_episode_directories(output):
    monitor_directory(output).mkdir(parents=True, exist_ok=True)
    reward_directory(output).mkdir(parents=True, exist_ok=True)


def monitor_name(n_envs, rank=None):
    if n_envs == 1:
        if rank is not None:
            raise ValueError("rank is only valid for multi-environment runs")
        return "episodes.monitor.csv"
    if rank is None or rank < 0 or rank >= n_envs:
        raise ValueError("rank must identify a worker in a multi-environment run")
    return f"episodes.worker-{rank:03d}.monitor.csv"


def monitor_path(output, n_envs, rank=None):
    """Return the path used by a new run when writing a Monitor CSV."""
    return monitor_directory(output) / monitor_name(n_envs, rank)


def monitor_paths(output, n_envs):
    """Return Monitor paths, preferring the new layout and reading legacy runs."""
    output = Path(output)
    ranks = [None] if n_envs == 1 else range(n_envs)
    paths = []
    for rank in ranks:
        name = monitor_name(n_envs, rank)
        current = monitor_directory(output) / name
        legacy = output / name
        paths.append(current if current.is_file() or not legacy.is_file() else legacy)
    return paths


def discover_monitor_files(output):
    """Discover both layouts without returning duplicate filenames."""
    output = Path(output)
    found = {}
    for directory in (output / "episodes" / "monitors", output):
        for path in directory.glob("episodes*.monitor.csv"):
            found.setdefault(path.name, path)
    return [found[name] for name in sorted(found)]


def reward_log_path(monitor_file):
    """Map a Monitor path to its dedicated reward-stream path.

    Legacy Monitor paths keep their historical adjacent ``.rewards.jsonl``
    location so old tools and archived runs remain readable.
    """
    monitor_file = Path(monitor_file)
    if monitor_file.parent.name == "monitors" and monitor_file.parent.parent.name == "episodes":
        return monitor_file.parent.parent / "rewards" / f"{monitor_file.name}.rewards.jsonl"
    return Path(f"{monitor_file}.rewards.jsonl")
