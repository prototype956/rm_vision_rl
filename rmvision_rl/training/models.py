"""Model factories and self-describing checkpoints, independent of the training CLI.

Only stateless MaskablePPO/MLP is implemented. A future recurrent integration must
also supply sequence rollouts, hidden-state resets and masking, not just a GRU layer.
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

from rmvision_rl.policy.observations import SCHEMA, VERSION
from rmvision_rl.training.config import ENV_PATHS, validate_config


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_contract(env):
    """Keep the history axes intact; flattening is a policy extractor responsibility."""
    return {
        "version": VERSION, "schema_sha256": _digest(SCHEMA),
        "spaces": {name: {"shape": list(space.shape), "dtype": str(space.dtype)}
                   for name, space in env.observation_space.spaces.items()},
        "action_version": 1, "actions": ["track", "request_fire"],
        "action_count": int(env.action_space.n),
    }


def training_metadata(config, env):
    """Snapshot effective paths and config fingerprints before starting any simulation."""
    config = validate_config(config)
    wrapped = env.envs[0] if isinstance(env, DummyVecEnv) else env
    base = wrapped.unwrapped
    for name in ENV_PATHS:
        config["environment"][name] = str(getattr(base, name))
    config["environment"]["episode_steps"] = base.episode_steps
    config["environment"]["scene_seed"] = wrapped.get_wrapper_attr("scene_seed")
    config["environment"]["scenario"] = base._scenario({"scenario": wrapped.get_wrapper_attr("scenario")})
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
    """Read plain JSON before choosing a loader or deserializing model/optimizer objects."""
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
    """Restore policy and optimizer, but start a fresh simulator episode on next learn/reset."""
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
    """Commit weights, optimizer and metadata as one atomically replaced ZIP file."""
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
    """Only publish an already completed checkpoint; never reserialize a partial update."""
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
