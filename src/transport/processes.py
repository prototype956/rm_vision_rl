"""管理训练子进程及其 socket，正常退出或检查失败时均回收资源。"""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import tempfile
import time

from src.transport.client import TrainingClient


@contextmanager
def training_worker(sim_root, binary, log_path, config_path=None, *, physics_step_us=None):
    binary = binary.resolve()
    env = os.environ.copy()
    sysroot = subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip()
    rustlib = subprocess.check_output(["rustc", "--print", "target-libdir"], text=True).strip()
    env["LD_LIBRARY_PATH"] = ":".join([
        str(binary.parent / "deps"), str(Path(sysroot) / "lib"), rustlib,
        env.get("LD_LIBRARY_PATH", ""),
    ])
    with tempfile.TemporaryDirectory(prefix="rm-rl-scenarios-") as directory, log_path.open("w") as log:
        path = Path(directory) / "training.sock"
        command = [
            str(binary), "--socket", str(path), "--config", str(config_path or sim_root / "config.toml"),
            "--assets", str(sim_root / "assets"),
        ]
        if physics_step_us is not None:
            if physics_step_us not in (500, 1000, 2000):
                raise ValueError("physics step must be 500, 1000 or 2000 microseconds")
            command.extend(["--physics-step-us", str(physics_step_us)])
        process = subprocess.Popen(command, cwd=sim_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        client = None
        try:
            deadline = time.monotonic() + 30
            while not path.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"training process exited; see {log_path}")
                if time.monotonic() > deadline:
                    raise TimeoutError("training socket startup timed out")
                time.sleep(0.05)
            client = TrainingClient(path)
            yield client
            client.close()
            if process.wait(timeout=10) != 0:
                raise RuntimeError("training process failed during Close")
        finally:
            if client is not None:
                client.disconnect()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
