"""通过原生窗口按键操作真实 PPO Gym 环境，支持按住按键持续请求射击。"""
import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading

from src.training.config import DEFAULT_CONFIG, ROOT, read_config, validate_config
from src.training.environment import make_environment
from tools.training.view import _atomic_json


VERSION = 1
CSV_FIELDS = ("episode", "step", "time_s", "action", "fire_legal_before", "fire_legal_next",
              "action_masked", "shot_requested", "shot_accepted", "reject_reason", "actual_shots",
              "reward", "cumulative_reward", "last_nonzero_reward", "last_nonzero_step",
              "terminated", "truncated", "end_reason", "decision_due_before", "decision_wait_ms",
              "decision_due_next", "clock_episode_stream", "physical_fire_legal_before", "clock_masked",
              "executed_action", "wire_action", "selected_slot", "slot_switched", "slot_switches", "mask_reason")


class ManualSession:
    """每个协议请求恰好执行一次 Gym 操作，重置必须由调用方显式请求。"""

    def __init__(self, env, writer, file, *, original_episode_steps, override=False):
        self.env, self.writer, self.file = env, writer, file
        self.original_episode_steps, self.override = original_episode_steps, override
        self.episode, self.step, self.total = 0, 0, 0.0
        self.last_reward, self.last_step = None, None
        self.done, self.obs = True, None

    def fire_legal(self):
        indices = [1] if self.env.action_space.n == 2 else [2, 4, 6, 8]
        return bool(self.obs["action_mask"][indices].any())

    def execute(self, request):
        operation = request["op"]
        if operation == "reset":
            self.obs, info = self.env.reset()
            self.episode += 1
            self.step, self.total = 0, 0.0
            self.last_reward, self.last_step = None, None
            self.done = False
            reward, action, before, terminated, truncated = 0.0, None, None, False, False
        elif operation == "step":
            if self.done:
                raise ValueError("episode ended or not started; reset before stepping")
            action = request.get("action")
            if type(action) is not int or not self.env.action_space.contains(action):
                raise ValueError("manual action must be in the configured action space")
            before = self.fire_legal()
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            reward = float(reward)
            self.step += 1
            self.total += reward
            if reward != 0:
                self.last_reward, self.last_step = reward, self.step
            self.done = terminated or truncated
        else:
            raise ValueError(f"unknown manual operation: {operation}")
        metrics = dict(episode=self.episode, step=self.step, time_s=info["episode_time_s"], action=action,
                       fire_legal_before=before, fire_legal_next=self.fire_legal(),
                       action_masked=info["action_masked"], shot_requested=info["shot_requested"],
                       shot_accepted=info["shot_accepted"], reject_reason=info["reject_reason"],
                       actual_shots=info["actual_shots"], reward=reward, cumulative_reward=self.total,
                       last_nonzero_reward=self.last_reward, last_nonzero_step=self.last_step,
                       terminated=bool(terminated), truncated=bool(truncated), end_reason=info["end_reason"])
        clock = info.get("decision_clock") or {}
        before_clock = info.get("decision_clock_before") or {}
        metrics.update(decision_due_before=before_clock.get("decision_due"),
                       decision_wait_ms=clock.get("wait_ms"), decision_due_next=clock.get("decision_due"),
                       clock_episode_stream=clock.get("episode_stream"),
                       physical_fire_legal_before=info.get("physical_fire_legal_before"),
                       clock_masked=info.get("clock_masked", False))
        metrics.update({key: info.get(key) for key in
                        ("executed_action", "wire_action", "selected_slot", "slot_switched", "slot_switches", "mask_reason")})
        if operation == "step":
            self.writer.writerow(metrics)
            self.file.flush()
        metrics.update(action_mode=self.env.unwrapped.action_mode, action_mask=self.obs["action_mask"].tolist(),
                       episode_steps=self.env.unwrapped.episode_steps,
                       original_episode_steps=self.original_episode_steps, length_overridden=self.override)
        return {**self.env.unwrapped.debug_snapshot(), "metrics": metrics}


def _viewer_environment(binary):
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    rustlib = subprocess.check_output(["rustc", "--print", "target-libdir"], text=True).strip()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([str(binary.parent / "deps"), str(binary.parent.parent / "deps"),
                                        str(Path(sysroot) / "lib"), rustlib, env.get("LD_LIBRARY_PATH", "")])
    return env


def run_manual(config=None, *, checkpoint=None, episode_steps=None, output_dir=None, viewer_binary=None):
    from src.training.models import read_checkpoint_metadata, training_metadata

    if config is not None and checkpoint is not None:
        raise ValueError("choose config or checkpoint")
    saved = read_checkpoint_metadata(checkpoint) if checkpoint is not None else None
    config = validate_config(saved["config"]) if saved else (read_config() if config is None else validate_config(config))
    original_steps = config["environment"]["episode_steps"]
    if episode_steps is not None:
        if type(episode_steps) is not int or episode_steps < 1:
            raise ValueError("episode_steps must be a positive integer")
    root = Path(config["environment"].get("simulator_root", ROOT.parent / "rm_simulator_2027"))
    binary = Path(viewer_binary).resolve() if viewer_binary else root / "target/release/examples/training_preview"
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("manual mode requires a graphical desktop (DISPLAY or WAYLAND_DISPLAY)")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"build training_preview before manual debugging: {binary}")
    viewer_env = _viewer_environment(binary)
    # 启动仿真与桥接进程前，先检查查看器版本是否兼容。
    help_result = subprocess.run([str(binary), "--help"], env=viewer_env, capture_output=True, text=True, timeout=15)
    if help_result.returncode or "--manual" not in help_result.stdout:
        raise RuntimeError("viewer lacks --manual support; rebuild training_preview")
    parent = Path(output_dir).resolve() if output_dir else ROOT / "artifacts/manual"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="manual-", dir=parent))
    state = {"version": VERSION, "status": "starting", "checkpoint": str(Path(checkpoint).resolve()) if saved else None,
             "original_episode_steps": original_steps, "episode_steps_override": episode_steps, "config": config,
             "input_mode": "realtime_hold_fire"}
    manifest = output / "run.json"
    _atomic_json(manifest, state)
    print(f"Manual debug: {output}\nJoint: 1–4 select plate | Hold F: fire | Release F: track | Space: pause/resume | R: restart (runs after warmup)", flush=True)
    closing = threading.Event()
    viewer = watcher = None
    handlers = {}

    class ViewerClosed(Exception):
        pass

    def stopped(signum, frame):
        if closing.is_set():
            return
        if signum == signal.SIGUSR1:
            raise ViewerClosed()
        raise KeyboardInterrupt(f"signal {signum}")

    try:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            handlers[sig] = signal.signal(sig, stopped)
        with ExitStack() as stack:
            # 先校验检查点原始契约，再应用显式指定的回合长度覆盖。
            contract_env = make_environment(config, output / "environment")
            stack.callback(contract_env.close)
            metadata = training_metadata(config, contract_env)
            if saved and any(metadata[key] != saved[key] for key in ("fingerprints", "observation")):
                raise ValueError("checkpoint environment/observation contract changed; restore its configuration")
            config = metadata["config"]
            if episode_steps is not None:
                config["environment"]["episode_steps"] = episode_steps
            env = make_environment(config, output / "environment", output / "episodes.monitor.csv")
            stack.callback(env.close)
            state.update(config=config, status="ready")
            _atomic_json(manifest, state)
            log = stack.enter_context((output / "viewer.log").open("w"))
            csv_file = stack.enter_context((output / "steps.csv").open("w", newline=""))
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            csv_file.flush()
            session = ManualSession(env, writer, csv_file, original_episode_steps=original_steps,
                                    override=episode_steps is not None)
            base = env.unwrapped
            viewer = subprocess.Popen([str(binary), "--manual", "--config", str(base.simulator_config),
                                       "--assets", str(base.simulator_root / "assets")], cwd=base.simulator_root,
                                      env=viewer_env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=log, text=True, bufsize=1)

            def watch_exit():
                viewer.wait()
                if not closing.is_set():
                    os.kill(os.getpid(), signal.SIGUSR1)

            watcher = threading.Thread(target=watch_exit, name="manual-viewer-exit", daemon=True)
            watcher.start()
            previous_id, fault = 0, None
            while True:
                line = viewer.stdout.readline()
                if not line:
                    break
                request_id = None
                try:
                    request = json.loads(line)
                    request_id = request.get("id")
                    if (request.get("version") != VERSION or type(request_id) is not int
                            or request_id != previous_id + 1):
                        raise ValueError("invalid manual protocol version or request sequence")
                    previous_id = request_id
                    if fault:
                        raise RuntimeError(f"session faulted: {fault}")
                    state["status"] = "warming" if request["op"] == "reset" else "running"
                    response = session.execute(request)
                    state.update(status="episode_complete" if session.done else "ready",
                                 episode=session.episode, step=session.step, cumulative_reward=session.total)
                    reply = {"version": VERSION, "id": request_id, "ok": True, **response}
                except ViewerClosed:
                    raise
                except Exception as error:
                    fault = f"{type(error).__name__}: {error}"
                    state.update(status="failed", error=fault)
                    env.close()
                    reply = {"version": VERSION, "id": request_id, "ok": False, "error": fault}
                    print(f"Manual session stopped: {fault}", flush=True)
                _atomic_json(manifest, state)
                viewer.stdin.write(json.dumps(reply, allow_nan=False, separators=(",", ":")) + "\n")
                viewer.stdin.flush()
    except (ViewerClosed, BrokenPipeError):
        if viewer is not None and viewer.poll() not in (0, None):
            state.update(status="failed", error=f"viewer exited with {viewer.returncode}; see viewer.log")
        elif state["status"] != "failed":
            state["status"] = "closed"
    except KeyboardInterrupt:
        state["status"] = "interrupted"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        closing.set()
        if viewer is not None:
            if viewer.poll() is None:
                viewer.terminate()
            try:
                viewer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                viewer.kill()
                viewer.wait()
            for pipe in (viewer.stdin, viewer.stdout):
                pipe.close()
        if watcher is not None:
            watcher.join()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if state["status"] not in ("failed", "interrupted"):
            state["status"] = "closed"
        _atomic_json(manifest, state)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path)
    source.add_argument("--checkpoint", type=Path, help="use saved environment contract; no policy inference")
    parser.add_argument("--episode-steps", type=int, help="explicit episode length override (10 ms per step)")
    parser.add_argument("--output-dir", type=Path, help="parent directory for a unique debug session")
    parser.add_argument("--viewer-binary", type=Path)
    args = parser.parse_args()
    output = run_manual(None if args.checkpoint else read_config(args.config or DEFAULT_CONFIG),
                        checkpoint=args.checkpoint, episode_steps=args.episode_steps,
                        output_dir=args.output_dir, viewer_binary=args.viewer_binary)
    return 1 if json.loads((output / "run.json").read_text())["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
