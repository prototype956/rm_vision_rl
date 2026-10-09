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
from time import perf_counter

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecEnv

from src.policy.observations import SCHEMA, JOINT_SCHEMA, VERSION
from src.policy.decision_clock import CONTRACT as CLOCK_CONTRACT
from src.training.config import validate_config
from src.training.parallel import sampling_metadata


class MeasuredMaskablePPO(MaskablePPO):
    """保持 PPO 行为不变，分别记录采样（含自动 reset）和参数更新的墙钟时间。"""

    def collect_rollouts(self, *args, **kwargs):
        started = perf_counter()
        result = super().collect_rollouts(*args, **kwargs)
        self.logger.record("time/sampling_seconds", perf_counter() - started)
        return result

    def train(self):
        started = perf_counter()
        super().train()
        self.logger.record("time/optimization_seconds", perf_counter() - started)


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
    if env.action_space.n == 9:
        contract.update(version=2, schema_sha256=_digest(JOINT_SCHEMA), action_version=3,
                        actions=["hold"] + [f"{kind}_slot_{slot}" for slot in range(4)
                                             for kind in ("track", "request_fire")])
    if "selection_memory" in env.observation_space.spaces:
        contract.update(version=3, selection_memory={
            "slots": [-1, 0, 1, 2, 3], "encoding": "one_hot",
            "meaning": "previous_distinct_confirmed_slot",
            "reset": "episode_generation_missing_callback_unreliable_or_invalid"})
    return contract


def training_metadata(config, env):
    """在启动仿真前记录实际路径、配置指纹和观测契约。"""
    config = validate_config(config)
    settings = (env.env_method("training_settings", indices=0)[0] if isinstance(env, VecEnv)
                else env.get_wrapper_attr("training_settings")())
    config["environment"].update(settings)
    modules = Path(settings["vision_root"]) / "src/config/modules"
    files = sorted(modules.rglob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no vision configuration in {modules}")
    result = {
        "format_version": 1,
        "config": config,
        "observation": observation_contract(env),
        "sampling": sampling_metadata(config),
        "fingerprints": {
            "environment": _digest(config["environment"]),
            "simulator_config": _file_digest(Path(settings["simulator_config"])),
            "vision_config": {str(path.relative_to(modules)): _file_digest(path) for path in files},
        },
        "dependencies": {name: version(name) for name in
                         ("torch", "stable-baselines3", "sb3-contrib", "gymnasium", "numpy", "tensorboard")},
    }

    if config["policy_kind"] in ("factorized_mlp", "shared_score_mlp"):
        result["policy_contract"] = copy.deepcopy(_policy_contract(config["policy_kind"]))
    return result


def _policy_contract(policy_kind):
    """Return the architecture contract for custom actor policies."""
    if policy_kind == "factorized_mlp":
        from src.training.factorized_policy import CONTRACT
        return CONTRACT
    if policy_kind == "shared_score_mlp":
        from src.training.shared_score_policy import CONTRACT
        return CONTRACT
    return None


def _policy_class(policy_kind):
    """Return the exact custom policy class, or None for the SB3 MLP policy."""
    if policy_kind == "factorized_mlp":
        from src.training.factorized_policy import FactorizedActorCriticPolicy
        return FactorizedActorCriticPolicy
    if policy_kind == "shared_score_mlp":
        from src.training.shared_score_policy import SharedScoreActorCriticPolicy
        return SharedScoreActorCriticPolicy
    return None


def _policy_matches(model_policy, policy_kind):
    expected = _policy_class(policy_kind)
    if expected is not None:
        return type(model_policy) is expected
    from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
    return type(model_policy) is MaskableMultiInputActorCriticPolicy


def _check_policy_contract(config, metadata):
    expected = _policy_contract(config["policy_kind"])
    if metadata.get("policy_contract") != expected:
        raise ValueError("checkpoint policy contract differs; train a new model for a different policy kind")


def build_model(config, env):
    config = validate_config(config)
    params = copy.deepcopy(config["ppo"])
    architecture = params.pop("net_arch")
    torch.set_num_threads(config["torch_threads"])
    policy = "MultiInputPolicy"
    policy_class = _policy_class(config["policy_kind"])
    if policy_class is not None:
        policy = policy_class
    return MeasuredMaskablePPO(
        policy, env, seed=config["seed"], device=config["device"],
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
    _check_policy_contract(metadata["config"], metadata)
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
    model = MeasuredMaskablePPO.load(path, env=env, device=config["device"], force_reset=True)
    if not _policy_matches(model.policy, config["policy_kind"]):
        raise ValueError("saved policy class disagrees with checkpoint policy_kind")
    if model.num_timesteps != metadata["num_timesteps"]:
        raise ValueError("checkpoint timestep metadata disagrees with saved model")
    return model, metadata


def save_checkpoint(model, metadata, path, completed_updates):
    """将权重、优化器和元数据保存到同一个 ZIP 文件，并原子替换目标文件。"""
    path = Path(path)
    data = copy.deepcopy(metadata)
    _check_policy_contract(data["config"], data)
    if not _policy_matches(model.policy, data["config"]["policy_kind"]):
        raise ValueError("policy class and metadata policy_kind disagree")
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
