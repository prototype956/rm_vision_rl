"""创建模型并管理包含配置元数据的检查点，供训练和评估入口复用。

当前仅支持无循环状态的 MaskablePPO/MLP。接入循环策略时，还需实现序列采样、
隐藏状态复位和动作掩码，不能只增加 GRU 层。
"""
import copy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.policy.observations import SCHEMA, VERSION
from src.policy.decision_clock import CONTRACT as CLOCK_CONTRACT
from src.training.config import ENV_PATHS, validate_config


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_contract(env):
    """生成观测与动作契约，保留历史时间轴；展平操作由策略特征提取器负责。"""
    contract = {
        "version": VERSION, "schema_sha256": _digest(SCHEMA),
        "spaces": {name: {"shape": list(space.shape), "dtype": str(space.dtype)}
                   for name, space in env.observation_space.spaces.items()},
        "action_version": 1, "actions": ["track", "request_fire"],
        "action_count": int(env.action_space.n),
    }
    if "decision_clock" in env.observation_space.spaces:
        contract.update(action_version=2, decision_clock=copy.deepcopy(CLOCK_CONTRACT))
    return contract


def training_metadata(config, env):
    """在启动仿真前记录实际路径、配置指纹和观测契约。"""
    config = validate_config(config)
    wrapped = env.envs[0] if isinstance(env, DummyVecEnv) else env
    base = wrapped.unwrapped
    for name in ENV_PATHS:
        config["environment"][name] = str(getattr(base, name))
    config["environment"]["episode_steps"] = base.episode_steps
    config["environment"]["scene_seed"] = wrapped.get_wrapper_attr("scene_seed")
    config["environment"]["scenario"] = base._scenario({"scenario": wrapped.get_wrapper_attr("scenario")})
    if config["environment"].get("task") == "random_rotation_fire":
        config["environment"]["angular_speed_range_rad_s"] = list(base.angular_speed_range_rad_s)
    modules = base.vision_root / "src/config/modules"
    files = sorted(modules.rglob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no vision configuration in {modules}")
    return {
        "format_version": 1,
        "config": config,
        "observation": observation_contract(env),
        "fingerprints": {
            "environment": _digest(config["environment"]),
            "simulator_config": _file_digest(base.simulator_config),
            "vision_config": {str(path.relative_to(modules)): _file_digest(path) for path in files},
        },
        "dependencies": {name: version(name) for name in
                         ("torch", "stable-baselines3", "sb3-contrib", "gymnasium", "numpy", "tensorboard")},
    }


def build_model(config, env):
    config = validate_config(config)
    params = copy.deepcopy(config["ppo"])
    architecture = params.pop("net_arch")
    torch.set_num_threads(config["torch_threads"])
    return MaskablePPO(
        "MultiInputPolicy", env, seed=config["seed"], device=config["device"],
        policy_kwargs={"net_arch": architecture, "activation_fn": torch.nn.Tanh},
        **params,
    )


def read_checkpoint_metadata(path):
    """先读取 JSON 元数据，再选择加载器或反序列化模型及优化器。"""
    with ZipFile(path) as archive:
        try:
            metadata = json.loads(archive.read("rmvision.json"))
        except KeyError as error:
            raise ValueError("checkpoint lacks RM training metadata; use a checkpoint from this trainer") from error
    if metadata.get("format_version") != 1:
        raise ValueError("unsupported checkpoint metadata version")
    validate_config(metadata["config"])
    for name in ("num_timesteps", "completed_updates"):
        if type(metadata.get(name)) is not int or metadata[name] < 0:
            raise ValueError(f"checkpoint has invalid {name}")
    return metadata


def load_model(path, env, *, device=None):
    """恢复策略与优化器，并在下次 learn/reset 时开始新的仿真回合。"""
    metadata = read_checkpoint_metadata(path)
    config = copy.deepcopy(metadata["config"])
    if device is not None:
        config["device"] = device
    current = training_metadata(config, env)
    if current["observation"] != metadata["observation"]:
        raise ValueError("checkpoint observation/action contract differs from the current environment")
    if current["fingerprints"] != metadata["fingerprints"]:
        raise ValueError("environment or vision configuration changed since the checkpoint; restore its configuration")
    torch.set_num_threads(config["torch_threads"])
    model = MaskablePPO.load(path, env=env, device=config["device"], force_reset=True)
    if model.num_timesteps != metadata["num_timesteps"]:
        raise ValueError("checkpoint timestep metadata disagrees with saved model")
    return model, metadata


def save_checkpoint(model, metadata, path, completed_updates):
    """将权重、优化器和元数据保存到同一个 ZIP 文件，并原子替换目标文件。"""
    path = Path(path)
    data = copy.deepcopy(metadata)
    data.update(num_timesteps=int(model.num_timesteps), completed_updates=completed_updates)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".zip", delete=False) as file:
        temporary = Path(file.name)
    try:
        model.save(temporary)
        with ZipFile(temporary, "a") as archive:
            archive.writestr("rmvision.json", json.dumps(data, indent=2, allow_nan=False))
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_latest(checkpoint, destination):
    """将已完成的检查点原子复制到目标路径，避免保存未完成的优化更新。"""
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".latest-", suffix=".zip", delete=False) as file:
        temporary = Path(file.name)
    try:
        shutil.copyfile(checkpoint, temporary)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
