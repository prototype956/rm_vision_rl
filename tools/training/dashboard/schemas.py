"""Validation and normalization for dashboard requests.

The dashboard deliberately exposes a small set of training overrides.  The
environment and reward contract remains owned by the selected preset or by a
checkpoint when resuming.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from src.training.config import ROOT, validate_config
from src.training.models import read_checkpoint_metadata


class RequestError(ValueError):
    """Raised when a dashboard request is invalid or unsafe."""


NEW_FIELDS = {
    "preset", "budget_mode", "budget_value", "n_envs", "device", "seed",
    "checkpoint_updates", "auto_analysis",
}
RESUME_FIELDS = {
    "checkpoint", "budget_mode", "budget_value", "n_envs", "device", "auto_analysis",
}

# Tool requests are deliberately kept separate from the standalone tools'
# complete configuration schemas.  The dashboard accepts only values that a
# user can safely select in the browser and fills the remaining values from
# config/tools/<tool>.json.  In particular, no command, module, or arbitrary
# filesystem path is accepted from the request body.
TOOL_FIELDS = {
    "analysis": {"run_id", "run_dir", "device", "evaluation_scenes", "open_view"},
    "compare": {"checkpoint", "baseline_checkpoint", "output_dir", "device", "workers"},
    "diagnose": {"checkpoint", "rollouts", "device", "output_dir"},
    "benchmark": {
        "config", "preset", "output_dir", "n_envs", "repeats", "rollout_mode",
        "timesteps", "updates", "budget_mode", "budget_value",
    },
    "view": {
        "mode", "checkpoint", "replay", "output_dir", "device", "scene_count",
        "scene_seeds", "workers", "scene_seed", "open_view",
    },
    "manual": {
        "mode", "config", "preset", "checkpoint", "episode_steps", "output_dir", "scene_seed",
    },
}

TOOL_NAMES = frozenset(TOOL_FIELDS)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise RequestError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise RequestError(f"{name} must be boolean")
    return value


def _safe_relative_path(root: Path, value: Any, *, name: str, must_exist: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RequestError(f"{name} must be a nonempty path string")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    root = root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RequestError(f"{name} must stay below {root}") from error
    if must_exist and not resolved.is_file():
        raise RequestError(f"{name} does not point to a regular file: {resolved}")
    return resolved


def training_config_paths(root: Path = ROOT) -> list[Path]:
    """Return available training presets below the repository config root."""
    directory = (root / "config" / "training").resolve()
    return sorted(path for path in directory.glob("*.json") if path.is_file())


def load_preset(preset: Any, root: Path = ROOT) -> tuple[str, dict[str, Any]]:
    """Load a preset while rejecting paths outside ``config/training``."""
    config_root = (root / "config" / "training").resolve()
    if isinstance(preset, str) and preset.startswith("config/training/"):
        preset = preset[len("config/training/"):]
    path = _safe_relative_path(config_root, preset, name="preset", must_exist=True)
    if path.suffix != ".json":
        raise RequestError("preset must be a JSON file")
    try:
        config = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RequestError(f"unable to read preset {path.name}: {error}") from error
    try:
        config = validate_config(config)
    except (TypeError, ValueError) as error:
        raise RequestError(f"invalid training preset {path.name}: {error}") from error
    return path.name, config


def _budget(config: dict[str, Any], mode: Any, value: Any) -> tuple[int, int]:
    if mode not in ("timesteps", "updates"):
        raise RequestError("budget_mode must be timesteps or updates")
    value = _positive_int(value, "budget_value")
    rollout = config["ppo"]["n_steps"] * config["n_envs"]
    requested = value if mode == "timesteps" else value * rollout
    actual = ((requested + rollout - 1) // rollout) * rollout
    return requested, actual


def _apply_runtime(config: dict[str, Any], payload: dict[str, Any], *, resume: bool) -> None:
    if "n_envs" in payload and payload["n_envs"] is not None:
        config["n_envs"] = _positive_int(payload["n_envs"], "n_envs")
    if "device" in payload and payload["device"] is not None:
        if not isinstance(payload["device"], str) or not payload["device"]:
            raise RequestError("device must be a nonempty string")
        config["device"] = payload["device"]
    if not resume:
        if "seed" in payload and payload["seed"] is not None:
            seed = payload["seed"]
            if type(seed) is not int or seed < 0 or seed >= 2**32:
                raise RequestError("seed must be an integer in [0, 2**32)")
            config["seed"] = seed
        if "checkpoint_updates" in payload and payload["checkpoint_updates"] is not None:
            config["checkpoint_updates"] = _positive_int(
                payload["checkpoint_updates"], "checkpoint_updates")


def validate_new_payload(payload: Any, root: Path = ROOT) -> dict[str, Any]:
    """Return an effective new-training request with a validated config."""
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    unknown = set(payload) - NEW_FIELDS
    if unknown:
        raise RequestError(f"unknown training fields: {sorted(unknown)}")
    preset_name, config = load_preset(payload.get("preset"), root)
    _apply_runtime(config, payload, resume=False)
    try:
        config = validate_config(config)
    except (TypeError, ValueError) as error:
        raise RequestError(str(error)) from error
    requested, actual = _budget(config, payload.get("budget_mode"), payload.get("budget_value"))
    auto_analysis = payload.get("auto_analysis", False)
    if "auto_analysis" in payload:
        auto_analysis = _boolean(auto_analysis, "auto_analysis")
    return {
        "preset": preset_name, "config": config, "requested_timesteps": requested,
        "actual_timesteps": actual, "auto_analysis": auto_analysis,
    }


def validate_resume_payload(payload: Any, root: Path = ROOT) -> dict[str, Any]:
    """Return a validated resume request and checkpoint metadata."""
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    unknown = set(payload) - RESUME_FIELDS
    if unknown:
        raise RequestError(f"unknown resume fields: {sorted(unknown)}")
    checkpoint_root = (root / "artifacts" / "training").resolve()
    checkpoint_value = payload.get("checkpoint")
    if isinstance(checkpoint_value, str) and checkpoint_value.startswith("artifacts/training/"):
        checkpoint_value = checkpoint_value[len("artifacts/training/"):]
    checkpoint = _safe_relative_path(checkpoint_root, checkpoint_value,
                                     name="checkpoint", must_exist=True)
    if checkpoint.suffix != ".zip":
        raise RequestError("checkpoint must be a ZIP file")
    try:
        metadata = read_checkpoint_metadata(checkpoint)
        config = validate_config(copy.deepcopy(metadata["config"]))
    except Exception as error:
        raise RequestError(f"invalid checkpoint: {error}") from error
    _apply_runtime(config, payload, resume=True)
    try:
        config = validate_config(config)
    except (TypeError, ValueError) as error:
        raise RequestError(str(error)) from error
    requested, actual = _budget(config, payload.get("budget_mode"), payload.get("budget_value"))
    auto_analysis = payload.get("auto_analysis", False)
    if "auto_analysis" in payload:
        auto_analysis = _boolean(auto_analysis, "auto_analysis")
    return {
        "checkpoint": checkpoint, "metadata": metadata, "config": config,
        "requested_timesteps": requested, "actual_timesteps": actual,
        "auto_analysis": auto_analysis,
    }


def _tool_path(root: Path, value: Any, name: str, *, base: Path,
               must_exist: bool = False, directory: bool = False) -> Path:
    """Resolve a dashboard tool path below ``base`` and reject traversal."""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RequestError(f"{name} must be a nonempty path string")
    base = base.resolve()
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        # Accept both repository-relative paths (``artifacts/training/x``)
        # and UI-friendly paths relative to the scoped root (``x`` or
        # ``run-id/final.zip``).
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise RequestError(f"{name} must stay below {base}") from error
    if must_exist:
        if directory and not resolved.is_dir():
            raise RequestError(f"{name} does not point to a directory: {resolved}")
        if not directory and not resolved.is_file():
            raise RequestError(f"{name} does not point to a regular file: {resolved}")
    return resolved


def _tool_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise RequestError(f"{name} must be an integer >= {minimum}")
    return value


def _tool_device(value: Any) -> str:
    if value is None:
        return "cpu"
    if not isinstance(value, str) or not value or len(value) > 64:
        raise RequestError("device must be a nonempty string")
    return value


def validate_tool_payload(tool: Any, payload: Any, root: Path = ROOT) -> dict[str, Any]:
    """Validate and normalize one safe web request for a standalone tool.

    The returned dictionary is ready to be written as a tool configuration;
    all path values are absolute paths constrained to repository roots.  The
    tool entry points still perform their own domain validation when they
    consume the generated configuration.
    """
    if tool not in TOOL_NAMES:
        raise RequestError(f"unknown tool: {tool}")
    if not isinstance(payload, dict):
        raise RequestError("request body must be a JSON object")
    unknown = set(payload) - TOOL_FIELDS[tool]
    if unknown:
        raise RequestError(f"unknown {tool} fields: {sorted(unknown)}")
    root = Path(root).resolve()
    training_root = (root / "artifacts" / "training").resolve()
    artifacts_root = (root / "artifacts").resolve()
    config_root = (root / "config" / "training").resolve()
    tool_root = (root / "config" / "tools").resolve()
    defaults_path = tool_root / f"{tool}.json"
    try:
        defaults = json.loads(defaults_path.read_text())
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RequestError(f"unable to read {defaults_path}: {error}") from error
    if not isinstance(defaults, dict):
        raise RequestError(f"invalid tool defaults: {defaults_path}")
    values = {**defaults, **copy.deepcopy(payload)}
    # `preset` is a UI-friendly alias for the benchmark/manual training config.
    # Compare resolved paths only when both fields were explicitly supplied;
    # otherwise the web-selected preset intentionally overrides the standalone
    # config/tools default.
    if payload.get("preset") is not None:
        if payload.get("config") is not None:
            preset_path = _tool_path(root, payload["preset"], "preset",
                                     base=config_root, must_exist=True)
            config_path = _tool_path(root, payload["config"], "config",
                                     base=config_root, must_exist=True)
            if preset_path != config_path:
                raise RequestError("preset and config must refer to the same training preset")
        values["config"] = payload["preset"]
        values.pop("preset", None)
    values.pop("preset", None)

    if tool == "analysis" and payload.get("run_dir") is None and payload.get("run_id") is not None:
        values["run_dir"] = payload["run_id"]
    if tool == "benchmark" and values.get("budget_mode") is not None:
        mode, amount = values.get("budget_mode"), values.get("budget_value")
        if mode not in ("timesteps", "updates"):
            raise RequestError("budget_mode must be timesteps or updates")
        if amount is None:
            raise RequestError("budget_value is required with budget_mode")
        if payload.get("timesteps") is not None or payload.get("updates") is not None:
            raise RequestError("budget_mode cannot be combined with timesteps or updates")
        values["timesteps"] = amount if mode == "timesteps" else None
        values["updates"] = amount if mode == "updates" else None

    if tool == "analysis":
        values["run_dir"] = str(_tool_path(root, values.get("run_dir"), "run_dir",
                                             base=training_root, must_exist=True, directory=True))
        values["device"] = _tool_device(values.get("device"))
        values["evaluation_scenes"] = _tool_integer(values.get("evaluation_scenes", 5),
                                                      "evaluation_scenes", minimum=2)
        values["open_view"] = False
        values.pop("viewer_binary", None)
    elif tool == "compare":
        for key in ("checkpoint", "baseline_checkpoint"):
            values[key] = str(_tool_path(root, values.get(key), key, base=training_root,
                                         must_exist=True))
            if Path(values[key]).suffix != ".zip":
                raise RequestError(f"{key} must be a ZIP checkpoint")
        values["output_dir"] = str(_tool_path(root, values.get("output_dir"), "output_dir",
                                               base=artifacts_root))
        values["device"] = _tool_device(values.get("device"))
        values["workers"] = _tool_integer(values.get("workers", 4), "workers")
    elif tool == "diagnose":
        values["checkpoint"] = str(_tool_path(root, values.get("checkpoint"), "checkpoint",
                                               base=training_root, must_exist=True))
        if Path(values["checkpoint"]).suffix != ".zip":
            raise RequestError("checkpoint must be a ZIP checkpoint")
        values["rollouts"] = _tool_integer(values.get("rollouts", 3), "rollouts")
        values["device"] = _tool_device(values.get("device"))
        values["output_dir"] = str(_tool_path(root, values.get("output_dir"), "output_dir",
                                               base=artifacts_root))
    elif tool == "benchmark":
        values["config"] = str(_tool_path(root, values.get("config"), "config",
                                           base=config_root, must_exist=True))
        values["output_dir"] = str(_tool_path(root, values.get("output_dir"), "output_dir",
                                               base=artifacts_root))
        envs = values.get("n_envs")
        if not isinstance(envs, list) or not envs:
            raise RequestError("n_envs must be a nonempty list")
        values["n_envs"] = [_tool_integer(item, "n_envs") for item in envs]
        if len(set(values["n_envs"])) != len(values["n_envs"]):
            raise RequestError("n_envs values must be unique")
        values["repeats"] = _tool_integer(values.get("repeats", 2), "repeats")
        if values.get("rollout_mode", "total") not in ("total", "per-env"):
            raise RequestError("rollout_mode must be total or per-env")
        timesteps, updates = values.get("timesteps"), values.get("updates")
        if timesteps is not None and updates is not None:
            raise RequestError("timesteps and updates are mutually exclusive")
        if timesteps is None and updates is None:
            raise RequestError("benchmark requires timesteps or updates")
        if timesteps is not None:
            values["timesteps"] = _tool_integer(timesteps, "timesteps")
        if updates is not None:
            values["updates"] = _tool_integer(updates, "updates")
    elif tool == "view":
        mode = values.get("mode", "checkpoint")
        if mode not in ("checkpoint", "replay"):
            raise RequestError("web view supports checkpoint or replay mode")
        values["mode"] = mode
        if mode == "checkpoint":
            values["checkpoint"] = str(_tool_path(root, values.get("checkpoint"), "checkpoint",
                                                   base=training_root, must_exist=True))
            if Path(values["checkpoint"]).suffix != ".zip":
                raise RequestError("checkpoint must be a ZIP checkpoint")
            values["replay"] = None
        else:
            values["replay"] = str(_tool_path(root, values.get("replay"), "replay",
                                               base=artifacts_root, must_exist=True))
            values["checkpoint"] = None
        values["output_dir"] = str(_tool_path(root, values.get("output_dir"), "output_dir",
                                               base=artifacts_root))
        values["device"] = _tool_device(values.get("device"))
        values["scene_count"] = _tool_integer(values.get("scene_count", 1), "scene_count")
        values["workers"] = _tool_integer(values.get("workers", 4), "workers")
        seeds = values.get("scene_seeds", [])
        if not isinstance(seeds, list) or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise RequestError("scene_seeds must be a list of nonnegative integers")
        if seeds and values["scene_count"] != 1:
            raise RequestError("scene_seeds and scene_count are mutually exclusive")
        values["scene_seeds"] = seeds
        for key in ("scene_seed",):
            if values.get(key) is not None:
                values[key] = _tool_integer(values[key], key, minimum=0)
        values["open_view"] = False
        values.pop("viewer_binary", None)
    elif tool == "manual":
        if payload.get("preset") is not None and payload.get("checkpoint") is None:
            values["mode"] = "config"
        mode = values.get("mode", "checkpoint")
        if mode not in ("config", "checkpoint"):
            raise RequestError("manual mode must be config or checkpoint")
        values["mode"] = mode
        if mode == "checkpoint":
            values["checkpoint"] = str(_tool_path(root, values.get("checkpoint"), "checkpoint",
                                                   base=training_root, must_exist=True))
            if Path(values["checkpoint"]).suffix != ".zip":
                raise RequestError("checkpoint must be a ZIP checkpoint")
            values["config"] = None
        else:
            values["config"] = str(_tool_path(root, values.get("config"), "config",
                                               base=config_root, must_exist=True))
            values["checkpoint"] = None
        values["output_dir"] = str(_tool_path(root, values.get("output_dir"), "output_dir",
                                               base=artifacts_root))
        if values.get("episode_steps") is not None:
            values["episode_steps"] = _tool_integer(values["episode_steps"], "episode_steps")
        if values.get("scene_seed") is not None:
            values["scene_seed"] = _tool_integer(values["scene_seed"], "scene_seed", minimum=0)
    # Preserve only fields understood by tools.config; the generated config is
    # never allowed to carry dashboard-only aliases or arbitrary keys.
    allowed = {
        "analysis": {"run_dir", "device", "evaluation_scenes", "viewer_binary", "open_view"},
        "compare": {"checkpoint", "baseline_checkpoint", "output_dir", "device", "workers"},
        "diagnose": {"checkpoint", "rollouts", "device", "output_dir"},
        "benchmark": {"config", "output_dir", "n_envs", "repeats", "rollout_mode", "timesteps", "updates"},
        "view": {"mode", "checkpoint", "replay", "device", "output_dir", "viewer_binary", "label",
                 "open_view", "scene_count", "scene_seeds", "workers", "scene_seed"},
        "manual": {"mode", "config", "checkpoint", "episode_steps", "output_dir", "viewer_binary", "scene_seed"},
    }[tool]
    return {key: values.get(key) for key in allowed if key in values}
