"""Evaluate one checkpoint, then replay recorded truth without running another physical world."""
import argparse
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile


REPLAY_VERSION = 1


def _atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".replay-",
                                     suffix=".json", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, allow_nan=False, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate_checkpoint(checkpoint, *, device=None, output_dir=None):
    # Replay-only use does not import PyTorch or require training dependencies.
    from gymnasium.utils.seeding import np_random
    from rmvision_rl.environment.evaluation import EvaluationConfig, EvaluationSession
    from rmvision_rl.environment.spawn import reset_spawn
    from rmvision_rl.policy.static_fire import StaticFirePolicy
    from rmvision_rl.training.environment import make_environment
    from rmvision_rl.training.models import load_model, read_checkpoint_metadata
    from rmvision_rl.transport.processes import training_worker
    from rmvision_rl.transport.vision_bridge import vision_worker

    checkpoint = Path(checkpoint).resolve()
    # latest.zip may be atomically replaced by a training task during evaluation.
    # Bind metadata, loaded weights and replay identity to one immutable byte snapshot.
    checkpoint_data = checkpoint.read_bytes()
    metadata = read_checkpoint_metadata(BytesIO(checkpoint_data))
    config = metadata["config"]
    settings = config["environment"]
    evaluation_config = EvaluationConfig(window_ms=settings["episode_steps"] * 10)
    parent = Path(output_dir).resolve() if output_dir else checkpoint.parent / "evaluation"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="eval-", dir=parent))
    frames, attempts = [], []
    raw_damage = 0.0
    with ExitStack() as stack:
        # This unstarted environment supplies the same strict checkpoint contract as training.
        env = make_environment(config, output / "environment")
        stack.callback(env.close)
        model, metadata = load_model(BytesIO(checkpoint_data), env, device=device)
        model.policy.set_training_mode(False)
        base = env.unwrapped
        client = stack.enter_context(training_worker(
            base.simulator_root, base.simulator_binary, output / "simulator.log", base.simulator_config))
        bridge = stack.enter_context(vision_worker(base.bridge_binary, base.vision_root, output / "bridge"))

        def predict(obs):
            action, _ = model.predict(obs, deterministic=True,
                                      action_masks=obs["action_mask"].astype(bool, copy=True))
            return int(action.item())

        policy = StaticFirePolicy(predict)
        session = EvaluationSession(client, bridge, evaluation_config,
                                    policy=policy, policy_mode="fire_only")
        rng, _ = np_random(settings["scene_seed"])
        scenario = base._scenario({"scenario": settings["scenario"]})
        reset_spawn(session.reset, rng, scenario, attempts)
        initial_scene = session.response["data"]["evaluation"]["scenario"]

        def capture(response):
            data = response["data"]
            score = session.score.summary() if session.score else None
            stats = session.info()
            metrics = {
                "raw_damage": raw_damage,
                "eligible_damage": score["eligible_damage_observed"] if score else 0,
                "official_damage": score["official_damage"] if score else None,
                "shots": score["eligible_shots"] if score else 0,
                "hits": score["damaging_projectiles"] if score else 0,
                "excluded_shots": score["excluded_own_shots"] if score else 0,
                "evaluation_time_ns": stats["evaluation_time_ns"] or 0,
                "settlement_time_ns": stats["settlement_time_ns"] or 0,
                "end_reason": session.end_reason,
            }
            snapshot = {
                "time_ns": response["sim_time_ns"], "phase": session.status,
                "metrics": metrics,
                "data": {
                    "feedback": {k: data["feedback"][k] for k in ("yaw_rad", "pitch_rad")},
                    "evaluation": {k: data["evaluation"][k]
                                   for k in ("robots", "projectiles", "controlled_muzzle")},
                    "events": data["events"],
                },
            }
            if frames and frames[-1]["time_ns"] == snapshot["time_ns"]:
                # EndWindow has no physical step. Keep its events on the final physical frame.
                previous = frames.pop()["data"]["events"]
                unique = {e["event_id"]: e for e in previous + data["events"]}
                snapshot["data"]["events"] = list(unique.values())
            frames.append(snapshot)

        capture(session.response)
        print(f"Evaluating {checkpoint.name}: fixed scene {settings['scene_seed']}, "
              f"{evaluation_config.window_ms / 1000:g}s, deterministic masked policy", flush=True)
        ticks = 0
        while session.status in ("warming", "evaluating", "settling"):
            phase = session.status
            result = session.advance()
            if phase == "evaluating":
                raw_damage += float(result["responses"][0]["data"]["reward_damage"])
            for response in result["responses"]:
                capture(response)
            ticks += 1
            if ticks % 100 == 0:
                info = session.info()
                print(f"  {session.status}: evaluation {(info['evaluation_time_ns'] or 0)/1e9:.2f}s, "
                      f"settlement {(info['settlement_time_ns'] or 0)/1e9:.2f}s, "
                      f"raw damage {raw_damage:g}", flush=True)
        if session.status == "warmup_timed_out":
            raise TimeoutError("checkpoint evaluation warmup timed out")
        if session.status not in ("complete", "settlement_timed_out"):
            raise RuntimeError(f"checkpoint evaluation failed: {session.info()}")
        score = session.score.summary()
        record = {
            "version": REPLAY_VERSION,
            "model": {"name": checkpoint.name, "path": str(checkpoint),
                      "sha256": hashlib.sha256(checkpoint_data).hexdigest(),
                      "num_timesteps": metadata["num_timesteps"],
                      "algorithm": config["algorithm"], "policy_kind": config["policy_kind"]},
            "fingerprints": metadata["fingerprints"],
            "render": {"simulator_root": str(base.simulator_root),
                       "config": str(base.simulator_config), "assets": str(base.simulator_root / "assets")},
            "scene_seed": settings["scene_seed"], "reset_attempts": attempts,
            "scenario": initial_scene, "step_ns": 10_000_000,
            "summary": {"status": session.status, "end_reason": session.end_reason,
                        "raw_damage": raw_damage, "score": score,
                        "hit_rate": score["damaging_projectiles"] / score["eligible_shots"]
                        if score["eligible_shots"] else None},
            "frames": frames,
        }
    # Publish only after inference AND both worker shutdowns succeeded.
    path = output / "replay.json"
    _atomic_json(path, record)
    rate = record["summary"]["hit_rate"]
    rate_text = f"{rate:.1%}" if rate is not None else "—"
    print(f"Evaluation {session.status}: raw damage={raw_damage:g}, "
          f"window damage={score['official_damage']}, shots={score['eligible_shots']}, "
          f"hit rate={rate_text}\nReplay: {path}", flush=True)
    return path


def open_replay(path, *, viewer_binary=None):
    path = Path(path).resolve()
    with path.open() as file:
        replay = json.load(file)
    if replay.get("version") != REPLAY_VERSION or not replay.get("frames"):
        raise ValueError("unsupported or empty replay")
    render = replay["render"]
    root = Path(render["simulator_root"])
    binary = Path(viewer_binary).resolve() if viewer_binary else root / "target/release/examples/training_preview"
    command = [str(binary), "--replay", str(path), "--config", render["config"], "--assets", render["assets"]]
    manual = f"python -m rmvision_rl.training.view --replay {shlex.quote(str(path))}"
    if viewer_binary:
        manual += f" --viewer-binary {shlex.quote(str(binary))}"
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print(f"No graphical session; replay retained. Open on the desktop:\n{manual}", flush=True)
        return False
    if not binary.is_file():
        print(f"Replay viewer unavailable: {binary}\nBuild training_preview, then run:\n{manual}", flush=True)
        return False
    digest = hashlib.sha256(Path(render["config"]).read_bytes()).hexdigest()
    if digest != replay["fingerprints"]["simulator_config"]:
        raise ValueError("simulator config differs from the recorded replay; restore it before viewing")
    env = os.environ.copy()
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    rustlib = subprocess.check_output(["rustc", "--print", "target-libdir"], text=True).strip()
    env["LD_LIBRARY_PATH"] = ":".join([str(binary.parent / "deps"), str(binary.parent.parent / "deps"),
                                        str(Path(sysroot) / "lib"), rustlib, env.get("LD_LIBRARY_PATH", "")])
    process = subprocess.Popen(command, cwd=root, env=env)
    try:
        if process.wait() != 0:
            raise RuntimeError(f"replay viewer failed; replay retained at {path}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--device", help="inference device; default from checkpoint")
    parser.add_argument("--output-dir", type=Path, help="parent directory for a new evaluation run")
    parser.add_argument("--viewer-binary", type=Path)
    args = parser.parse_args()
    if args.replay and (args.device is not None or args.output_dir is not None):
        parser.error("--device and --output-dir apply only to --checkpoint")

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        path = args.replay or evaluate_checkpoint(args.checkpoint, device=args.device, output_dir=args.output_dir)
        open_replay(path, viewer_binary=args.viewer_binary)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
