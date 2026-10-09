"""读取和校验 MLP 训练配置，与 Gym 环境及通信配置分开管理。"""
import copy
import json
import math
from pathlib import Path
from src.policy.decision_clock import validate_clock
from src.environment.rotation_fire import rotation_scenario, validate_speed_range

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/training/rotation_joint_shared_score_ppo.json"
ENV_PATHS = ("simulator_root", "vision_root", "simulator_binary", "bridge_binary", "simulator_config")


def validate_config(value):
    """在启动仿真进程前检查策略类型和配置字段，拒绝不支持的配置。"""
    config = copy.deepcopy(value)
    required = {"version", "algorithm", "policy_kind", "seed", "device", "torch_threads",
                "total_timesteps", "checkpoint_updates", "environment", "ppo"}
    if (not isinstance(config, dict) or not required <= config.keys()
            or set(config) - required - {"n_envs", "reward_scale", "switch_penalty", "selection_reward"}):
        raise ValueError(f"training config requires {sorted(required)}; optional: n_envs, reward_scale, switch_penalty")
    config.setdefault("n_envs", 1)
    config.setdefault("reward_scale", 1.0)
    scale = config["reward_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("reward_scale must be finite and positive")
    if (config["version"] != 1 or config["algorithm"] != "maskable_ppo"
            or config["policy_kind"] not in ("mlp", "factorized_mlp", "shared_score_mlp")):
        raise ValueError("only version 1 maskable_ppo with mlp/factorized_mlp/shared_score_mlp is implemented")

    def integer(name, number, minimum=1):
        if type(number) is not int or number < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    for name in ("torch_threads", "total_timesteps", "checkpoint_updates", "n_envs"):
        integer(name, config[name])
    integer("seed", config["seed"], 0)
    if config["seed"] >= 2**32:
        raise ValueError("PPO seed must be less than 2**32")
    if not isinstance(config["device"], str) or not config["device"]:
        raise ValueError("device must be a nonempty PyTorch device string")
    env = config["environment"]
    if (not isinstance(env, dict) or not {"episode_steps", "scene_seed", "scenario"} <= env.keys()
            or set(env) - {"episode_steps", "scene_seed", "scenario", "decision_clock", "task",
                           "angular_speed_range_rad_s", "action_mode", "selection_memory", *ENV_PATHS}):
        raise ValueError("invalid environment config keys")
    integer("episode_steps", env["episode_steps"])
    integer("scene_seed", env["scene_seed"], 0)
    if not isinstance(env["scenario"], dict):
        raise ValueError("environment.scenario must be an object")
    if env.get("action_mode", "fire_only") not in ("fire_only", "joint"):
        raise ValueError("environment.action_mode must be fire_only or joint")
    if config["policy_kind"] in ("factorized_mlp", "shared_score_mlp") and env.get("action_mode") != "joint":
        raise ValueError(f"{config['policy_kind']} requires joint action mode")
    if "selection_memory" in env and type(env["selection_memory"]) is not bool:
        raise ValueError("selection_memory must be boolean")
    if env.get("selection_memory", False) and env.get("action_mode") != "joint":
        raise ValueError("selection_memory requires joint action mode")
    if "selection_reward" in config:
        reward = config["selection_reward"]
        fields = {"switch_cost", "return_cost", "return_window_s", "facing_rate_per_s", "facing_margin"}
        if not isinstance(reward, dict) or set(reward) != fields:
            raise ValueError("selection_reward requires exactly " + str(sorted(fields)))
        for key, number in reward.items():
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError("selection_reward values must be finite and nonnegative")
        if reward["return_window_s"] <= 0 or reward["facing_margin"] >= 1:
            raise ValueError("return_window_s must be positive; facing_margin must be < 1")
        if not env.get("selection_memory", False) or env.get("action_mode") != "joint":
            raise ValueError("selection_reward requires joint mode with selection_memory")
        if "switch_penalty" in config:
            raise ValueError("selection_reward replaces switch_penalty; cannot combine them")
    if "switch_penalty" in config:
        penalty = config["switch_penalty"]
        if not isinstance(penalty, dict) or set(penalty) != {"c0", "c1"}:
            raise ValueError("switch_penalty requires exactly c0 and c1; omit to disable")
        for key, number in penalty.items():
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError(f"switch_penalty.{key} must be finite and nonnegative")
        if env.get("action_mode", "fire_only") != "joint":
            raise ValueError("switch_penalty requires joint action mode")
    task = env.get("task", "static_fire")
    if task not in ("static_fire", "random_rotation_fire"):
        raise ValueError("environment.task must be static_fire or random_rotation_fire")
    if task == "random_rotation_fire":
        env["angular_speed_range_rad_s"] = validate_speed_range(
            env.get("angular_speed_range_rad_s", [1, 7]))
        rotation_scenario({"scenario": env["scenario"]})
    elif "angular_speed_range_rad_s" in env:
        raise ValueError("angular_speed_range_rad_s requires random_rotation_fire")
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
    rollout_size = ppo["n_steps"] * config["n_envs"]
    if ppo["batch_size"] > rollout_size or rollout_size % ppo["batch_size"]:
        raise ValueError("batch_size must divide n_steps * n_envs")
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
