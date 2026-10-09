"""评估检查点、生成离线回放，或实时运行已保存策略。"""
import argparse
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import tempfile

from tools.config import load_tool_config, write_effective_config


REPLAY_VERSION = 1


def _random_scene_seeds(count):
    """Generate unique 32-bit scene seeds for one view invocation."""
    if type(count) is not int or count < 1:
        raise ValueError("scene count must be a positive integer")
    used = set()
    seeds = []
    while len(seeds) < count:
        seed = secrets.randbits(32)
        if seed not in used:
            used.add(seed)
            seeds.append(seed)
    return seeds


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


def evaluate_checkpoint(checkpoint, *, device=None, output_dir=None, scene_override=None,
                        scene_seed=None):
    """Evaluate fixed weights and record a deterministic replay."""
    # 仅播放已有回放时延迟加载训练模块，避免依赖 PyTorch。
    from src.environment.evaluation import EvaluationConfig, EvaluationSession
    from src.environment.spawn import reset_spawn
    from src.policy.decision import make_policy
    from src.training.environment import make_environment
    from src.training.models import load_model, read_checkpoint_metadata
    from src.transport.processes import training_worker
    from src.transport.vision_bridge import vision_worker

    checkpoint = Path(checkpoint).resolve()
    # 评估期间，训练任务可能原子替换 latest.zip。
    # 使用固定字节副本，使元数据、加载权重和回放标识保持一致。
    checkpoint_data = checkpoint.read_bytes()
    metadata = read_checkpoint_metadata(BytesIO(checkpoint_data))
    config = metadata["config"]
    settings = config["environment"]
    effective_scene_seed = settings["scene_seed"] if scene_seed is None else int(scene_seed)
    evaluation_config = EvaluationConfig(window_ms=settings["episode_steps"] * 10)
    parent = Path(output_dir).resolve() if output_dir else checkpoint.parent / "evaluation"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="eval-", dir=parent))
    frames, attempts = [], []
    raw_damage = 0.0
    last_control, last_action = {}, None
    with ExitStack() as stack:
        # 用尚未启动的环境生成与训练一致的检查点契约。
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

        policy = make_policy(settings.get("action_mode", "fire_only"), predict,
                             decision_clock=settings.get("decision_clock"),
                             selection_memory=settings.get("selection_memory", False))
        session = EvaluationSession(client, bridge, evaluation_config,
                                    policy=policy, policy_mode=policy.bridge_mode)
        if scene_override is None:
            scenario, rng, scene_sample = base.prepare_scene(
                seed=effective_scene_seed, options={"scenario": settings["scenario"]})
            reset_spawn(session.reset, rng, scenario, attempts)
        else:
            # 对照场景已完成出生检查，双方必须使用同一个 seed；这里不再重试换场。
            scenario = scene_override["scenario"]
            scene_sample = {"comparison_scene": scene_override["scene_id"]}
            if "sampling_seed" in scene_override:
                scene_sample["sampling_seed"] = scene_override["sampling_seed"]
            attempts.append({"seed": scene_override["spawn_seed"]})
            session.reset(scene_override["spawn_seed"], scenario)
        initial_scene = session.response["data"]["evaluation"]["scenario"]
        replay_scene_seed = (scene_override.get("sampling_seed", effective_scene_seed)
                             if scene_override is not None else effective_scene_seed)

        def capture(response):
            data = response["data"]
            score = session.score.summary() if session.score else None
            stats = session.info()
            metrics = {
                **stats["selection"], "action_mode": settings.get("action_mode", "fire_only"),
                "wire_action": last_action, "control": last_control,
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
            if policy.clock.enabled:
                metrics["decision_clock"] = policy.clock.info()
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
                # EndWindow 不推进物理，将其事件合并到相同时间戳的最后一帧。
                previous = frames.pop()["data"]["events"]
                unique = {e["event_id"]: e for e in previous + data["events"]}
                snapshot["data"]["events"] = list(unique.values())
            frames.append(snapshot)

        capture(session.response)
        print(f"Evaluating {checkpoint.name}: reference scene {settings['scene_seed']}, "
              f"{evaluation_config.window_ms / 1000:g}s, deterministic masked policy", flush=True)
        ticks = 0
        while session.status in ("warming", "evaluating", "settling"):
            phase = session.status
            result = session.advance()
            if phase == "evaluating":
                last_control = result["vision"]["control"]
                last_action = (result["vision"].get("policy") or {}).get("action", 0)
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
            "scene_seed": replay_scene_seed, "reset_attempts": attempts,
            "scene_sample": {**scene_sample, "spawn_seed": attempts[-1]["seed"]},
            "decision_clock": settings.get("decision_clock"),
            "action_mode": settings.get("action_mode", "fire_only"),
            "scene_override": scene_override,
            "scenario": initial_scene, "step_ns": 10_000_000,
            "summary": {"status": session.status, "end_reason": session.end_reason,
                        "raw_damage": raw_damage, "score": score,
                        "slot_switches": session.selection["slot_switches"],
                        "hit_rate": score["damaging_projectiles"] / score["eligible_shots"]
                        if score["eligible_shots"] else None},
            "frames": frames,
        }
    # 仅在推理完成且仿真与桥接进程均成功退出后写出回放。
    path = output / "replay.json"
    _atomic_json(path, record)
    rate = record["summary"]["hit_rate"]
    rate_text = f"{rate:.1%}" if rate is not None else "—"
    print(f"Evaluation {session.status}: raw damage={raw_damage:g}, "
          f"window damage={score['official_damage']}, shots={score['eligible_shots']}, "
          f"hit rate={rate_text}\nReplay: {path}", flush=True)
    return path


def _prepare_scene_overrides(config, output, seeds):
    """Resolve reproducible scene and spawn data shared by multiple replays."""
    from src.environment.spawn import reset_spawn
    from src.training.environment import make_environment
    from src.transport.processes import training_worker

    settings = config["environment"]
    env = make_environment(config, output / "scene-generation")
    base = env.unwrapped
    scenes = []
    try:
        with training_worker(base.simulator_root, base.simulator_binary,
                             output / "scene-generation.log", base.simulator_config) as client:
            for index, seed in enumerate(seeds):
                scenario, rng, sample = base.prepare_scene(
                    seed=int(seed), options={"scenario": settings["scenario"]})
                attempts = []
                initial = reset_spawn(client.reset, rng, scenario, attempts)
                scenes.append({
                    "scene_id": index,
                    "sampling_seed": int(seed),
                    "scene_sample": sample,
                    "scenario": scenario,
                    "spawn_seed": attempts[-1]["seed"],
                    "resolved_scenario": initial["data"]["evaluation"]["scenario"],
                    "reset_attempts": attempts,
                })
    finally:
        env.close()
    _atomic_json(output / "evaluation_scenes.json", scenes)
    return scenes


def evaluate_checkpoint_scenes(checkpoint, *, seeds, device=None, output_dir=None, workers=1):
    """Evaluate one checkpoint on explicitly selected, shared random scenes."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    checkpoint = Path(checkpoint).resolve()
    checkpoint_data = checkpoint.read_bytes()
    from src.training.models import read_checkpoint_metadata

    metadata = read_checkpoint_metadata(BytesIO(checkpoint_data))
    parent = Path(output_dir).resolve() if output_dir else checkpoint.parent / "evaluation"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="eval-scenes-", dir=parent))
    scenes = _prepare_scene_overrides(metadata["config"], output, seeds)
    tasks = [(checkpoint, scene, device, output) for scene in scenes]
    workers = min(workers, len(tasks))
    pool = multiprocessing.get_context("spawn").Pool(workers)
    try:
        results = pool.map(_evaluate_scene_worker, tasks)
        pool.close()
        pool.join()
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    return [(path, f"scene seed {seed}") for _, seed, path in sorted(results)]


def _evaluate_scene_worker(task):
    """Run one independent scene evaluation in its own simulator/bridge pair."""
    checkpoint, scene, device, output = task
    path = evaluate_checkpoint(
        checkpoint, device=device,
        output_dir=output / f"scene-{scene['scene_id']:02d}",
        scene_override={key: scene[key] for key in
                        ("scene_id", "scenario", "spawn_seed", "sampling_seed")})
    return scene["scene_id"], scene["sampling_seed"], path


def write_replay_playlist(replays, path):
    """Write one viewer manifest for multiple independently generated replays."""
    path = Path(path).resolve()
    if not replays:
        raise ValueError("cannot create an empty replay playlist")
    entries = []
    common = None
    first = None
    for index, (replay_path, label) in enumerate(replays):
        replay_path = Path(replay_path).resolve()
        record = json.loads(replay_path.read_text())
        if record.get("version") != REPLAY_VERSION or not record.get("frames"):
            raise ValueError(f"unsupported or empty replay: {replay_path}")
        render = record.get("render")
        fingerprints = record.get("fingerprints")
        if not isinstance(render, dict) or not isinstance(fingerprints, dict):
            raise ValueError(f"replay lacks render metadata: {replay_path}")
        signature = (render.get("simulator_root"), render.get("config"), render.get("assets"),
                     fingerprints.get("simulator_config"))
        if any(value is None for value in signature):
            raise ValueError(f"replay lacks complete render metadata: {replay_path}")
        if common is None:
            common = signature
            first = record
        elif signature != common:
            raise ValueError("replays in one playlist use different simulator/render contracts")
        entries.append({
            "index": index,
            "path": os.path.relpath(replay_path, path.parent),
            "label": label or f"scene {index + 1}",
            "scene_seed": record.get("scene_seed"),
        })
    manifest = {
        "version": REPLAY_VERSION,
        "kind": "playlist",
        "render": first["render"],
        "fingerprints": {"simulator_config": first["fingerprints"]["simulator_config"]},
        "model": first.get("model", {}),
        "replays": entries,
    }
    _atomic_json(path, manifest)
    return path


def _start_replay(path, *, viewer_binary=None, label=None):
    path = Path(path).resolve()
    with path.open() as file:
        replay = json.load(file)
    if replay.get("kind") == "playlist":
        if replay.get("version") != REPLAY_VERSION or not replay.get("replays"):
            raise ValueError("unsupported or empty replay playlist")
    elif replay.get("version") != REPLAY_VERSION or not replay.get("frames"):
        raise ValueError("unsupported or empty replay")
    render = replay["render"]
    root = Path(render["simulator_root"])
    binary = Path(viewer_binary).resolve() if viewer_binary else root / "target/release/examples/training_preview"
    command = [str(binary), "--replay", str(path), "--config", render["config"], "--assets", render["assets"]]
    manual = f"python -m tools.training.view --replay {shlex.quote(str(path))}"
    if label:
        command += ["--label", label]
        manual += f" --label {shlex.quote(label)}"
    if viewer_binary:
        manual += f" --viewer-binary {shlex.quote(str(binary))}"
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print(f"No graphical session; replay retained. Open on the desktop:\n{manual}", flush=True)
        return None
    if not binary.is_file():
        print(f"Replay viewer unavailable: {binary}\nBuild training_preview, then run:\n{manual}", flush=True)
        return None
    digest = hashlib.sha256(Path(render["config"]).read_bytes()).hexdigest()
    if digest != replay["fingerprints"]["simulator_config"]:
        raise ValueError("simulator config differs from the recorded replay; restore it before viewing")
    env = os.environ.copy()
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    rustlib = subprocess.check_output(["rustc", "--print", "target-libdir"], text=True).strip()
    env["LD_LIBRARY_PATH"] = ":".join([str(binary.parent / "deps"), str(binary.parent.parent / "deps"),
                                        str(Path(sysroot) / "lib"), rustlib, env.get("LD_LIBRARY_PATH", "")])
    return subprocess.Popen(command, cwd=root, env=env)


def open_replays(replays, *, viewer_binary=None):
    """先启动所有回放窗口，再等待退出；关闭一个窗口不影响其他窗口。

    收集普通启动或退出错误，使其他有效回放仍可打开。
    收到中断时先回收本函数拥有的全部进程，再向上抛出中断。
    """
    processes, errors = [], []
    try:
        for path, label in replays:
            try:
                process = _start_replay(path, viewer_binary=viewer_binary, label=label)
                if process is not None:
                    processes.append((process, path))
            except Exception as error:
                errors.append(f"{path}: {type(error).__name__}: {error}")
                print(f"Replay launch failed: {errors[-1]}", flush=True)
        for process, path in processes:
            if process.wait() != 0:
                errors.append(f"replay viewer failed; replay retained at {path}")
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, _ in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if errors:
        raise RuntimeError("; ".join(errors))
    return bool(processes)


def open_replay(path, *, viewer_binary=None, label=None):
    return open_replays([(path, label)], viewer_binary=viewer_binary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--replay", type=Path)
    parser.add_argument("--config", type=Path, help="tool configuration JSON")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, help="parent directory for a new evaluation run")
    parser.add_argument("--viewer-binary", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--no-view", action="store_true", default=None)
    parser.add_argument("--live", action="store_true",
                        help="run checkpoint inference in the live Gym viewer; ↑ changes to a new random seed")
    parser.add_argument("--scene-seed", type=int)
    parser.add_argument("--scene-seeds", type=int, nargs="+")
    parser.add_argument("--scene-count", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    overrides = {}
    if args.checkpoint is not None:
        overrides.update(mode="checkpoint", checkpoint=str(args.checkpoint), replay=None)
    if args.replay is not None:
        overrides.update(mode="replay", replay=str(args.replay), checkpoint=None)
    if args.live:
        overrides.update(mode="live", replay=None)
    for key in ("device", "output_dir", "viewer_binary", "label", "scene_seed", "scene_seeds",
                "scene_count", "workers"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    if args.no_view is not None:
        overrides["open_view"] = not args.no_view
    config = load_tool_config("view", args.config, overrides)
    mode = config["mode"]

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        if mode == "live":
            from tools.training.manual import run_manual
            scene_seed = (config["scene_seed"] if config["scene_seed"] is not None
                          else _random_scene_seeds(1)[0])
            output = run_manual(checkpoint=config["checkpoint"], output_dir=config["output_dir"],
                                viewer_binary=config["viewer_binary"], scene_seed=scene_seed)
            write_effective_config(output / "tool_config.json",
                                   {**config, "scene_seed": scene_seed})
            return 1 if json.loads((output / "run.json").read_text())["status"] == "failed" else 0
        if mode == "replay":
            replay_path = Path(config["replay"]).resolve()
            write_effective_config(replay_path.parent / "tool_config.json", config)
            if config.get("open_view", True):
                open_replay(replay_path, viewer_binary=config["viewer_binary"], label=config["label"])
            return 0
        if config.get("scene_seeds") or config.get("scene_count", 1) > 1:
            seeds = list(config["scene_seeds"]) or _random_scene_seeds(config["scene_count"])
            replays = evaluate_checkpoint_scenes(
                config["checkpoint"], seeds=seeds, device=config["device"], output_dir=config["output_dir"],
                workers=config["workers"])
            output = Path(replays[0][0]).parents[2]
            effective_config = {**config, "scene_count": len(seeds), "scene_seeds": seeds}
            write_effective_config(output / "tool_config.json", effective_config)
            if config.get("open_view", True):
                if len(replays) == 1:
                    open_replay(replays[0][0], viewer_binary=config["viewer_binary"], label=config["label"])
                else:
                    playlist = write_replay_playlist(replays, output / "playlist.json")
                    open_replay(playlist, viewer_binary=config["viewer_binary"], label=config["label"])
            elif len(replays) > 1:
                write_replay_playlist(replays, output / "playlist.json")
            return 0
        scene_seed = (config["scene_seed"] if config["scene_seed"] is not None
                      else _random_scene_seeds(1)[0])
        path = evaluate_checkpoint(config["checkpoint"], device=config["device"],
                                   output_dir=config["output_dir"], scene_seed=scene_seed)
        write_effective_config(path.parent / "tool_config.json",
                               {**config, "scene_seed": scene_seed})
        if config.get("open_view", True):
            open_replay(path, viewer_binary=config["viewer_binary"], label=config["label"])
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
