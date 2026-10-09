"""提供单/多环境 PPO 训练入口，模型创建和持久化由 models.py 负责。"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import tempfile
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from src.training.config import DEFAULT_CONFIG, ROOT, read_config, validate_config
from src.training.parallel import make_training_vector
from src.training.models import (
    build_model, load_model, publish_latest, read_checkpoint_metadata,
    save_checkpoint, training_metadata,
)


class SamplingGuard(BaseCallback):
    """检查动作掩码和时间截断信息是否正确传递，保持环境输出的训练奖励不变。"""

    def _on_step(self):
        for info in self.locals["infos"]:
            if info["action_masked"]:
                raise RuntimeError("PPO submitted a masked action; abort instead of learning from a substituted action")
            if info.get("TimeLimit.truncated") and "terminal_observation" not in info:
                raise RuntimeError("time truncation lost its terminal observation")
        return True


def _write_manifest(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".run-", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _log_update(model, completed_updates, start_steps, started):
    # SB3 在优化结束后记录训练统计，因此在此处写出日志，
    # 使最后一次更新的损失也能写入 CSV 和 TensorBoard。
    for key, value in model.logger.name_to_value.items():
        if key.startswith("train/") and ("loss" in key or key == "train/approx_kl"):
            if not np.isfinite(value):
                raise FloatingPointError(f"non-finite PPO metric: {key}")
    if any(not torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
        raise FloatingPointError("non-finite policy parameters")
    episodes = list(model.ep_info_buffer)
    for output, key in (("ep_rew_mean", "r"), ("ep_len_mean", "l"),
                        ("ep_damage_mean", "episode_damage"), ("actual_shots_mean", "actual_shots"),
                        ("slot_switches_mean", "slot_switches"),
                        ("hit_reward_mean", "episode_hit_reward"),
                        ("switch_penalty_mean", "episode_switch_penalty"),
                        ("base_switch_penalty_mean", "episode_base_switch_penalty"),
                        ("return_penalty_mean", "episode_return_penalty"),
                        ("facing_penalty_mean", "episode_facing_penalty"),
                        ("reward_penalty_mean", "episode_reward_penalty")):
        # 旧检查点可能保存了尚无选板统计的回合缓存，不把缺失值算作零次切板。
        values = [episode[key] for episode in episodes if key in episode]
        if values:
            model.logger.record("rollout/" + output, float(np.mean(values)))
    elapsed = perf_counter() - started
    model.logger.record("time/fps", (model.num_timesteps - start_steps) / max(elapsed, 1e-9))
    model.logger.record("time/time_elapsed", elapsed)
    model.logger.record("time/total_timesteps", model.num_timesteps)
    model.logger.record("time/completed_updates", completed_updates)
    model.logger.dump(step=model.num_timesteps)


def run_training(config=None, *, resume=None, output_dir=None, timesteps=None, seed=None,
                 device=None, n_envs=None):
    """按完整采样批次训练；恢复时的预算表示新增步数。

    Args:
        config: 新训练使用的配置；恢复训练时使用检查点内的配置。
        resume: 待恢复的检查点路径，与 config 和 seed 互斥。
        output_dir: 新运行目录，必须尚不存在。
        timesteps: 本次新增采样步数，向上取整到完整 rollout。
        seed: 新模型的随机种子，不影响场景种子。
        device: 覆盖配置中的 PyTorch 计算设备。
        n_envs: 覆盖独立环境数量，恢复时也允许；不改变每环境 n_steps。

    Returns:
        本次运行的输出目录，包含日志和完整更新后保存的模型。
    """
    previous = None
    if resume is not None:
        if config is not None or seed is not None:
            raise ValueError("resume uses its saved configuration and seed; only budget, device, n_envs and output may change")
        resume = Path(resume).resolve()
        previous = read_checkpoint_metadata(resume)
        config = validate_config(previous["config"])
    else:
        config = read_config() if config is None else validate_config(config)
        if seed is not None:
            config["seed"] = seed
    if device is not None:
        config["device"] = device
    if timesteps is not None:
        config["total_timesteps"] = timesteps
    if n_envs is not None:
        config["n_envs"] = n_envs
    config = validate_config(config)
    # 显式请求 CUDA 时应检查可用性，避免静默退回 CPU。
    requested_device = torch.device(config["device"])
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch installation; use --device cpu")
    requested = config["total_timesteps"]
    rollout_steps = config["ppo"]["n_steps"] * config["n_envs"]
    actual = ((requested + rollout_steps - 1) // rollout_steps) * rollout_steps
    if output_dir is None:
        parent = ROOT / "artifacts/training"
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-")
        output = Path(tempfile.mkdtemp(prefix=stamp, dir=parent))
    else:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}\nRequested additional samples: {requested}; complete-rollout budget: {actual}", flush=True)
    print(f"Environments: {config['n_envs']}; steps per environment: {config['ppo']['n_steps']}; "
          f"samples per update: {rollout_steps}", flush=True)
    print(f"Training reward: damage * {config['reward_scale']:g} "
          f"(20 damage = {20 * config['reward_scale']:g} reward)", flush=True)
    print(f"Selection reward: {config.get('selection_reward', 'disabled')}", flush=True)
    print(f"Switch penalty: {config.get('switch_penalty', 'disabled')}", flush=True)
    unlimited_heat = config["environment"]["scenario"].get("unlimited_heat", False)
    print(f"Heat system: {'disabled (unlimited heat)' if unlimited_heat else 'enabled'}", flush=True)
    manifest = {"status": "starting", "requested_timesteps": requested, "rounded_timesteps": actual,
                "resume_from": str(resume) if resume else None, "config": config}
    _write_manifest(output / "run.json", manifest)
    try:
        with ExitStack() as stack:
            setup_started = perf_counter()
            vec = make_training_vector(config, output)
            stack.callback(vec.close)
            metadata = training_metadata(config, vec)
            config = metadata["config"]
            if resume is None:
                model = build_model(config, vec)
                completed_updates = 0
            else:
                model, _ = load_model(resume, vec, device=config["device"])
                completed_updates = previous["completed_updates"]
            logger = configure(str(output / "logs"), ["stdout", "csv", "tensorboard"])
            stack.callback(logger.close)
            model.set_logger(logger)
            start_steps = model.num_timesteps
            target_steps = start_steps + actual
            manifest.update(metadata=metadata, config=config, actual_device=str(model.device),
                            start_timesteps=start_steps, last_completed_timesteps=start_steps,
                            completed_updates=completed_updates, status="running")
            _write_manifest(output / "run.json", manifest)
            guard = SamplingGuard()
            started = perf_counter()
            manifest["setup_seconds"] = started - setup_started
            while model.num_timesteps < target_steps:
                # 学习率和裁剪阈值固定，因此每个完整 rollout 可单独调用 learn()，
                # 保留观测和优化器状态；仅在 train() 完成后保存检查点。
                before = model.num_timesteps
                learn_started = perf_counter()
                model.learn(total_timesteps=rollout_steps, reset_num_timesteps=False,
                            log_interval=None, callback=guard)
                model.logger.record("time/learn_seconds", perf_counter() - learn_started)
                model.logger.record("time/n_envs", config["n_envs"])
                model.logger.record("time/rollout_samples", rollout_steps)
                if model.num_timesteps != before + rollout_steps:
                    raise RuntimeError("PPO did not complete the expected rollout")
                completed_updates += 1
                _log_update(model, completed_updates, start_steps, started)
                if completed_updates % config["checkpoint_updates"] == 0:
                    checkpoint = output / f"checkpoint_{model.num_timesteps}.zip"
                    save_checkpoint(model, metadata, checkpoint, completed_updates)
                    publish_latest(checkpoint, output / "latest.zip")
                manifest.update(last_completed_timesteps=model.num_timesteps,
                                completed_updates=completed_updates)
                _write_manifest(output / "run.json", manifest)
            manifest["training_seconds"] = perf_counter() - started
            save_checkpoint(model, metadata, output / "final.zip", completed_updates)
            publish_latest(output / "final.zip", output / "latest.zip")
        manifest["status"] = "complete"
        _write_manifest(output / "run.json", manifest)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        _write_manifest(output / "run.json", manifest)
        # 异常时可能只完成部分采样或优化更新，不能将其保存为完整检查点。
        raise
    print(f"Completed {actual} samples; lifetime steps {model.num_timesteps}; model: {output / 'final.zip'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Train a fire-control MaskablePPO/MLP policy")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help=f"JSON config (default: {DEFAULT_CONFIG})")
    source.add_argument("--resume", type=Path, help="checkpoint ZIP; continue from a fresh simulator episode")
    parser.add_argument("--timesteps", type=int, help="additional samples, rounded up to complete rollouts")
    parser.add_argument("--seed", type=int, help="new model RNG seed; does not change the scene seed")
    parser.add_argument("--device", help="PyTorch device, defaults to config (cpu)")
    parser.add_argument("--n-envs", type=int, help="independent sampling environments (default: saved config or 1)")
    parser.add_argument("--output-dir", type=Path, help="new run directory; must not already exist")
    args = parser.parse_args()
    if args.resume and args.seed is not None:
        parser.error("--seed cannot override the saved seed on --resume")
    config = None if args.resume else read_config(args.config or DEFAULT_CONFIG)

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        run_training(config, resume=args.resume, output_dir=args.output_dir,
                     timesteps=args.timesteps, seed=args.seed, device=args.device, n_envs=args.n_envs)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
