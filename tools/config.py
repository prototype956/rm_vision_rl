"""Configuration loading and validation for the standalone training tools."""
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_CONFIG_ROOT = ROOT / "config" / "tools"

_SCHEMAS = {
    "view": {
        "mode", "checkpoint", "replay", "device", "output_dir", "viewer_binary", "label",
        "open_view", "scene_count", "scene_seeds", "workers", "scene_seed",
    },
    "analysis": {"run_dir", "device", "evaluation_scenes", "viewer_binary", "open_view"},
    "compare": {"checkpoint", "baseline_checkpoint", "output_dir", "device", "workers"},
    "manual": {"mode", "config", "checkpoint", "episode_steps", "output_dir", "viewer_binary",
                "scene_seed"},
    "diagnose": {"checkpoint", "rollouts", "device", "output_dir"},
    "benchmark": {"config", "output_dir", "n_envs", "repeats", "rollout_mode", "timesteps", "updates"},
}

_PATH_FIELDS = {
    "view": {"checkpoint", "replay", "output_dir", "viewer_binary"},
    "analysis": {"run_dir", "viewer_binary"},
    "compare": {"checkpoint", "baseline_checkpoint", "output_dir"},
    "manual": {"config", "checkpoint", "output_dir", "viewer_binary"},
    "diagnose": {"checkpoint", "output_dir"},
    "benchmark": {"config", "output_dir"},
}

_DEFAULTS = {
    "view": {"mode": "checkpoint", "replay": None, "device": "cpu", "output_dir": "artifacts/evaluation",
             "viewer_binary": None, "label": None, "open_view": True, "scene_count": 1,
             "scene_seeds": [], "workers": 4, "scene_seed": None},
    "analysis": {"device": "cpu", "evaluation_scenes": 5, "viewer_binary": None, "open_view": False},
    "compare": {"output_dir": "artifacts/comparison", "device": "cpu", "workers": 4},
    "manual": {"mode": "checkpoint", "config": None, "episode_steps": None,
                "output_dir": "artifacts/manual", "viewer_binary": None, "scene_seed": None},
    "diagnose": {"rollouts": 3, "device": "cpu", "output_dir": "artifacts/diagnostics"},
    "benchmark": {"n_envs": [1, 2, 4], "repeats": 2, "rollout_mode": "total",
                   "timesteps": 2048, "updates": None},
}


def _resolve_paths(tool, config):
    for key in _PATH_FIELDS[tool]:
        value = config.get(key)
        if value is not None:
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError(f"tools.{tool}.{key} must be a nonempty path string or null")
            path = Path(value).expanduser()
            config[key] = str((path if path.is_absolute() else ROOT / path).resolve())


def _validate(tool, config):
    if not isinstance(config, dict):
        raise ValueError(f"tools.{tool} configuration must be a JSON object")
    unknown = set(config) - _SCHEMAS[tool]
    if unknown:
        raise ValueError(f"tools.{tool} has unknown fields: {sorted(unknown)}")
    config = {**copy.deepcopy(_DEFAULTS[tool]), **copy.deepcopy(config)}
    _resolve_paths(tool, config)

    if tool == "view":
        mode = config.get("mode", "checkpoint")
        if mode not in ("checkpoint", "replay", "live"):
            raise ValueError("tools.view.mode must be checkpoint, replay, or live")
        if config.get("checkpoint") and config.get("replay"):
            raise ValueError("tools.view.checkpoint and replay are mutually exclusive")
        if type(config.get("scene_count", 1)) is not int or config.get("scene_count", 1) < 1:
            raise ValueError("tools.view.scene_count must be positive")
        if type(config.get("workers", 4)) is not int or config.get("workers", 4) < 1:
            raise ValueError("tools.view.workers must be positive")
        seeds = config.get("scene_seeds", [])
        if not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("tools.view.scene_seeds must be a list of nonnegative integers")
        scene_seed = config.get("scene_seed")
        if scene_seed is not None and (type(scene_seed) is not int or scene_seed < 0):
            raise ValueError("tools.view.scene_seed must be a nonnegative integer or null")
        if seeds and config.get("scene_count", 1) != 1:
            raise ValueError("tools.view.scene_seeds and scene_count are mutually exclusive")
        if mode == "checkpoint" and not config.get("checkpoint"):
            raise ValueError("tools.view.checkpoint is required in checkpoint mode")
        if mode == "replay" and not config.get("replay"):
            raise ValueError("tools.view.replay is required in replay mode")
        if mode == "live" and (not config.get("checkpoint") or seeds
                                or config.get("scene_count", 1) != 1):
            raise ValueError("tools.view live mode does not support offline scene selection")
    elif tool == "analysis":
        if not config.get("run_dir"):
            raise ValueError("tools.analysis.run_dir is required")
        if type(config.get("evaluation_scenes", 5)) is not int or config.get("evaluation_scenes", 5) < 2:
            raise ValueError("tools.analysis.evaluation_scenes must be at least 2")
    elif tool == "compare":
        for key in ("checkpoint", "baseline_checkpoint"):
            if not config.get(key):
                raise ValueError(f"tools.compare.{key} is required")
        if type(config.get("workers", 4)) is not int or config.get("workers", 4) < 1:
            raise ValueError("tools.compare.workers must be positive")
    elif tool == "manual":
        if config.get("mode", "checkpoint") not in ("config", "checkpoint"):
            raise ValueError("tools.manual.mode must be config or checkpoint")
        if config.get("mode") == "config" and not config.get("config"):
            raise ValueError("tools.manual.config is required in config mode")
        if config.get("mode") == "checkpoint" and not config.get("checkpoint"):
            raise ValueError("tools.manual.checkpoint is required in checkpoint mode")
        if config.get("config") and config.get("checkpoint"):
            raise ValueError("tools.manual.config and checkpoint are mutually exclusive")
    elif tool == "diagnose":
        if not config.get("checkpoint"):
            raise ValueError("tools.diagnose.checkpoint is required")
        if type(config.get("rollouts", 3)) is not int or config.get("rollouts", 3) < 1:
            raise ValueError("tools.diagnose.rollouts must be positive")
    elif tool == "benchmark":
        if not config.get("config"):
            raise ValueError("tools.benchmark.config is required")
        if config.get("output_dir") is None:
            raise ValueError("tools.benchmark.output_dir is required")
        if config.get("rollout_mode", "total") not in ("total", "per-env"):
            raise ValueError("tools.benchmark.rollout_mode must be total or per-env")
        n_envs = config.get("n_envs")
        if (not isinstance(n_envs, list) or not n_envs
                or any(type(value) is not int or value < 1 for value in n_envs)):
            raise ValueError("tools.benchmark.n_envs must be a nonempty list of positive integers")
        if type(config.get("repeats", 2)) is not int or config.get("repeats", 2) < 1:
            raise ValueError("tools.benchmark.repeats must be positive")
        if config.get("timesteps") is not None and config.get("updates") is not None:
            raise ValueError("tools.benchmark.timesteps and updates are mutually exclusive")
        if config.get("timesteps") is None and config.get("updates") is None:
            raise ValueError("tools.benchmark requires timesteps or updates")
        for key in ("timesteps", "updates"):
            value = config.get(key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"tools.benchmark.{key} must be positive")
    return config


def load_tool_config(tool, path=None, cli_overrides=None):
    """Load, merge, resolve, and validate one tool configuration."""
    if tool not in _SCHEMAS:
        raise ValueError(f"unknown tool configuration: {tool}")
    config_path = Path(path) if path is not None else TOOLS_CONFIG_ROOT / f"{tool}.json"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        value = json.loads(config_path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"tool configuration not found: {config_path}") from error
    if cli_overrides:
        value.update(cli_overrides)
    return _validate(tool, value)


def write_effective_config(path, config):
    """Write the resolved tool configuration beside a run's artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
