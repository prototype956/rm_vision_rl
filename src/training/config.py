"""读取和校验 MLP 训练配置，与 Gym 环境及通信配置分开管理。"""
import copy
import json
import math
from pathlib import Path
from src.policy.decision_clock import validate_clock

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/training/static_fire_ppo.json"
ENV_PATHS = ("simulator_root", "vision_root", "simulator_binary", "bridge_binary", "simulator_config")


def validate_config(value):
    """在启动仿真进程前检查策略类型和配置字段，拒绝不支持的配置。"""
    config = copy.deepcopy(value)
    required = {"version", "algorithm", "policy_kind", "seed", "device", "torch_threads",
                "total_timesteps", "checkpoint_updates", "environment", "ppo"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"training config requires exactly: {sorted(required)}")
    if config["version"] != 1 or (config["algorithm"], config["policy_kind"]) != ("maskable_ppo", "mlp"):
        raise ValueError("only version 1 maskable_ppo/mlp is implemented; recurrent policies need a separate trainer")

    def integer(name, number, minimum=1):
        if type(number) is not int or number < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    for name in ("torch_threads", "total_timesteps", "checkpoint_updates"):
        integer(name, config[name])
    integer("seed", config["seed"], 0)
    if config["seed"] >= 2**32:
        raise ValueError("PPO seed must be less than 2**32")
    if not isinstance(config["device"], str) or not config["device"]:
        raise ValueError("device must be a nonempty PyTorch device string")
    env = config["environment"]
    if (not isinstance(env, dict) or not {"episode_steps", "scene_seed", "scenario"} <= env.keys()
            or set(env) - {"episode_steps", "scene_seed", "scenario", "decision_clock", *ENV_PATHS}):
        raise ValueError("invalid environment config keys")
    integer("episode_steps", env["episode_steps"])
    integer("scene_seed", env["scene_seed"], 0)
    if not isinstance(env["scenario"], dict):
        raise ValueError("environment.scenario must be an object")
    if "decision_clock" in env:
        if env["decision_clock"] is None:
            raise ValueError("omit decision_clock to disable it")
        env["decision_clock"] = validate_clock(env["decision_clock"])
    if "unlimited_heat" in env["scenario"] and type(env["scenario"]["unlimited_heat"]) is not bool:
        raise ValueError("environment.scenario.unlimited_heat must be a boolean")
    for name in ENV_PATHS:
        if name in env:
            if not isinstance(env[name], str) or not env[name]:
                raise ValueError(f"environment.{name} must be a path string")
            path = Path(env[name]).expanduser()
            env[name] = str((path if path.is_absolute() else ROOT / path).resolve())
    ppo = config["ppo"]
    fields = {"n_steps", "batch_size", "n_epochs", "learning_rate", "gamma", "gae_lambda",
              "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "net_arch"}
    if not isinstance(ppo, dict) or set(ppo) != fields:
        raise ValueError(f"ppo requires exactly: {sorted(fields)}")
    for name in ("n_steps", "batch_size", "n_epochs"):
        integer(name, ppo[name], 2 if name != "n_epochs" else 1)
    if ppo["batch_size"] > ppo["n_steps"] or ppo["n_steps"] % ppo["batch_size"]:
        raise ValueError("batch_size must divide n_steps for the single-environment trainer")
    for name in fields - {"n_steps", "batch_size", "n_epochs", "net_arch"}:
        number = ppo[name]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("learning_rate", "clip_range", "max_grad_norm"):
        if ppo[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 < ppo["gamma"] <= 1 or not 0 <= ppo["gae_lambda"] <= 1:
        raise ValueError("gamma must be in (0,1] and gae_lambda in [0,1]")
    if not isinstance(ppo["net_arch"], dict) or set(ppo["net_arch"]) != {"pi", "vf"}:
        raise ValueError("net_arch requires pi and vf layer lists")
    for layers in ppo["net_arch"].values():
        if not isinstance(layers, list) or not layers:
            raise ValueError("network layer lists must be nonempty")
        for width in layers:
            integer("layer width", width)
    return config


def read_config(path=DEFAULT_CONFIG):
    return validate_config(json.loads(Path(path).read_text()))
