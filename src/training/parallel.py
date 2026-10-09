"""独立环境进程的创建、随机流派生与有界回收；沿用 SB3 的采样和截断语义。"""
import copy
from functools import partial
from multiprocessing.util import Finalize
import os
import signal
from time import monotonic

from stable_baselines3.common.vec_env import SubprocVecEnv

from src.training.environment import make_environment, vectorize
from src.training.paths import ensure_episode_directories, monitor_path


def worker_config(config, rank):
    """rank 0 保持历史单环境随机流；其他 rank 使用独立、可复现的种子。

    场景 seed 支持非负任意精度整数，决策时钟 seed 限定为 uint32。
    配置保持副本隔离；恢复训练重新从各 worker 的首个场景开始。
    """
    result = copy.deepcopy(config)
    settings = result["environment"]
    settings["scene_seed"] += rank
    if "decision_clock" in settings:
        settings["decision_clock"]["seed"] = (settings["decision_clock"]["seed"] + rank) % 2**32
    return result


def sampling_metadata(config):
    workers = []
    for rank in range(config["n_envs"]):
        settings = worker_config(config, rank)["environment"]
        workers.append(dict(rank=rank, scene_seed=settings["scene_seed"],
                            decision_clock_seed=settings.get("decision_clock", {}).get("seed")))
    return dict(version=1, n_envs=config["n_envs"],
                start_method="spawn" if config["n_envs"] > 1 else "inline", workers=workers)


def _create_worker(config, directory, monitor):
    # 每个 Python worker 独立进程组；原生子进程继承该组，异常时可一起回收。
    os.setsid()

    def stop(signum, frame):
        raise KeyboardInterrupt(f"environment worker signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    env = make_environment(config, directory, monitor)
    # SB3 worker 在 EOF/异常退出时不保证调用 env.close；进程正常清理时补齐。
    Finalize(None, env.close, exitpriority=10)
    return env


class TrainingSubprocVecEnv(SubprocVecEnv):
    """使用 SB3 原生 worker；关闭时不因失联 worker 而卡住或遗漏其同伴。"""

    def __init__(self, factories):
        try:
            super().__init__(factories, start_method="spawn")
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, "closed", False):
            return
        self.closed = True
        # 即使一条管道损坏，也通知其余 worker。正在执行的 step 后会处理 close。
        for remote in getattr(self, "remotes", ()):
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = monotonic() + 20
        for process in getattr(self, "processes", ()):
            process.join(timeout=max(0, deadline - monotonic()))
        for process in getattr(self, "processes", ()):
            # 即使 Python worker 已崩溃，也回收仍在其进程组内的原生子进程。
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                if process.is_alive():
                    process.terminate()
        deadline = monotonic() + 5
        for process in getattr(self, "processes", ()):
            process.join(timeout=max(0, deadline - monotonic()))
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if process.is_alive():
                    process.kill()
            if process.is_alive():
                process.join(timeout=5)
        for remote in getattr(self, "remotes", ()):
            remote.close()
        self.waiting = False


def make_training_vector(config, output):
    """每个实例独占 socket、原生进程、monitor 和日志。"""
    ensure_episode_directories(output)
    if config["n_envs"] == 1:
        env = make_environment(config, output / "environment", monitor_path(output, 1))
        try:
            return vectorize(env)
        except BaseException:
            env.close()
            raise
    factories = []
    for rank in range(config["n_envs"]):
        directory = output / "environment" / f"worker-{rank:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        monitor = monitor_path(output, config["n_envs"], rank)
        factories.append(partial(_create_worker, worker_config(config, rank), directory, monitor))
    return TrainingSubprocVecEnv(factories)
