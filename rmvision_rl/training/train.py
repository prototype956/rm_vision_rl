"""Single-environment PPO entry point; model selection and persistence live in models.py."""
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

from rmvision_rl.training.config import DEFAULT_CONFIG, ROOT, read_config, validate_config
from rmvision_rl.training.environment import make_environment, vectorize
from rmvision_rl.training.models import (
    build_model, load_model, publish_latest, read_checkpoint_metadata,
    save_checkpoint, training_metadata,
)


class SamplingGuard(BaseCallback):
    """Detect lost mask/timeout plumbing without changing environment rewards."""

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
    # SB3 records training statistics after optimization. Dump here, not at rollout end,
    # so the final update's losses also reach CSV and TensorBoard.
    for key, value in model.logger.name_to_value.items():
        if key.startswith("train/") and ("loss" in key or key == "train/approx_kl"):
            if not np.isfinite(value):
                raise FloatingPointError(f"non-finite PPO metric: {key}")
    if any(not torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
        raise FloatingPointError("non-finite policy parameters")
    episodes = list(model.ep_info_buffer)
    for output, key in (("ep_rew_mean", "r"), ("ep_len_mean", "l"),
                        ("ep_damage_mean", "episode_damage"), ("actual_shots_mean", "actual_shots")):
        if episodes:
            model.logger.record("rollout/" + output, float(np.mean([episode[key] for episode in episodes])))
    elapsed = perf_counter() - started
    model.logger.record("time/fps", (model.num_timesteps - start_steps) / max(elapsed, 1e-9))
    model.logger.record("time/time_elapsed", elapsed)
    model.logger.record("time/total_timesteps", model.num_timesteps)
    model.logger.record("time/completed_updates", completed_updates)
    model.logger.dump(step=model.num_timesteps)


def run_training(config=None, *, resume=None, output_dir=None, timesteps=None, seed=None, device=None):
    """Run complete rollouts; resume budgets are additional samples, never lifetime targets."""
    previous = None
    if resume is not None:
        if config is not None or seed is not None:
            raise ValueError("resume uses its saved configuration and seed; only budget, device and output may change")
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
    config = validate_config(config)
    # Explicit CUDA requests must not silently become CPU runs.
    requested_device = torch.device(config["device"])
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this PyTorch installation; use --device cpu")
    requested = config["total_timesteps"]
    rollout_steps = config["ppo"]["n_steps"]
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
    manifest = {"status": "starting", "requested_timesteps": requested, "rounded_timesteps": actual,
                "resume_from": str(resume) if resume else None, "config": config}
    _write_manifest(output / "run.json", manifest)
    try:
        with ExitStack() as stack:
            env = make_environment(config, output / "environment", output / "episodes.monitor.csv")
            stack.callback(env.close)
            vec = vectorize(env)
            stack.callback(vec.close)
            metadata = training_metadata(config, env)
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
            while model.num_timesteps < target_steps:
                # Constant LR/clip permit one learn() per full rollout without resetting
                # observations or optimizer state. Saving happens only AFTER train() returns.
                before = model.num_timesteps
                model.learn(total_timesteps=rollout_steps, reset_num_timesteps=False,
                            log_interval=None, callback=guard)
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
            save_checkpoint(model, metadata, output / "final.zip", completed_updates)
            publish_latest(output / "final.zip", output / "latest.zip")
        manifest["status"] = "complete"
        _write_manifest(output / "run.json", manifest)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        _write_manifest(output / "run.json", manifest)
        # No error-path save: model may contain a partial rollout or optimizer update.
        raise
    print(f"Completed {actual} samples; lifetime steps {model.num_timesteps}; model: {output / 'final.zip'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Train the static-target MaskablePPO/MLP policy")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help=f"JSON config (default: {DEFAULT_CONFIG})")
    source.add_argument("--resume", type=Path, help="checkpoint ZIP; continue from a fresh simulator episode")
    parser.add_argument("--timesteps", type=int, help="additional samples, rounded up to complete rollouts")
    parser.add_argument("--seed", type=int, help="new model RNG seed; does not change the fixed scene seed")
    parser.add_argument("--device", help="PyTorch device, defaults to config (cpu)")
    parser.add_argument("--output-dir", type=Path, help="new run directory; must not already exist")
    parser.add_argument("--no-view", action="store_true", help="skip automatic checkpoint evaluation and replay")
    parser.add_argument("--viewer-binary", type=Path, help="optional training_preview executable")
    args = parser.parse_args()
    if args.resume and args.seed is not None:
        parser.error("--seed cannot override the saved seed on --resume")
    config = None if args.resume else read_config(args.config or DEFAULT_CONFIG)

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        output = run_training(config, resume=args.resume, output_dir=args.output_dir,
                              timesteps=args.timesteps, seed=args.seed, device=args.device)
        if not args.no_view:
            from rmvision_rl.training.view import evaluate_checkpoint, open_replay
            try:
                replay = evaluate_checkpoint(output / "final.zip", device=args.device)
                open_replay(replay, viewer_binary=args.viewer_binary)
            except (Exception, KeyboardInterrupt) as error:
                # Training already succeeded. A separate viewing failure must not change its status.
                print(f"Training is complete; evaluation/replay stopped: {type(error).__name__}: {error}",
                      flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
