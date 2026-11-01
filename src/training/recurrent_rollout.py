"""Explicit-reset workers and complete episodes; paused workers never auto-reset."""
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import torch

from src.training.environment import make_environment
from src.training.parallel import worker_config
from src.training.paths import monitor_path
from src.training.recurrent_policy import tensor_observation, combine_states, select_state


def _worker(connection, config, directory, monitor):
    env = None
    try:
        if os.name == "posix":
            os.setsid()
        env = make_environment(config, directory, monitor)
        env.unwrapped._admission_progress = lambda value: connection.send(("admission_retry", value))
        connection.send((True, None))
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            if command == "reset":
                result = env.reset(options={"curriculum_scene": value}) if value is not None else env.reset()
            elif command == "step":
                result = env.step(value)
            else:
                raise ValueError("unknown worker command")
            if command == "step" and config.get("runtime", {}).get("bounded_logging") and not (result[2] or result[3]):
                obs,reward,terminated,truncated,info = result
                keys = ("executed_action", "wire_action", "shot_requested", "shot_accepted", "reject_reason",
                        "simulator_actual_shots", "simulator_rejected_requests", "action_masked")
                result = obs,reward,terminated,truncated,{k:info.get(k) for k in keys}
            connection.send((True, result))
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class EpisodeWorkers:
    def __init__(self, config, output):
        self.connections, self.processes = [], []
        self.admission_limit = config.get("curriculum", {}).get("translation_admission_attempts",32)
        ctx = mp.get_context("spawn")
        try:
            for rank in range(config["n_envs"]):
                parent, child = ctx.Pipe()
                directory = Path(output) / "environment" / f"worker-{rank:03d}"
                monitor = monitor_path(Path(output), config["n_envs"], rank if config["n_envs"] > 1 else None)
                process = ctx.Process(target=_worker, args=(child, worker_config(config, rank), directory, monitor))
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for rank in range(len(self.connections)):
                self.receive(rank)
        except BaseException:
            self.close()
            raise

    def send(self, rank, command, value=None):
        self.connections[rank].send((command, value))

    def receive(self, rank):
        retries = 0
        while True:
            if not self.connections[rank].poll(60):
                raise TimeoutError(f"environment {rank} did not respond in 60 s")
            success, value = self.connections[rank].recv()
            if success == "admission_retry":
                retries += 1
                if retries > self.admission_limit:
                    raise RuntimeError(f"environment {rank} exceeded {self.admission_limit} admission retries")
                print(f"ADMISSION_RETRY environment={rank} attempt={value['attempt']} "
                      f"seed={value['seed']} reason={value['kind']}", flush=True)
                continue
            if not success:
                raise RuntimeError(f"environment {rank}:\n{value}")
            return value

    def close(self):
        for conn in self.connections:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic() + 10
        for p in self.processes:
            p.join(max(0, deadline - time.monotonic()))
        for p in self.processes:
            if os.name == "posix":
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif p.is_alive():
                p.terminate()
        for p in self.processes:
            p.join(2)
            if os.name == "posix":
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif p.is_alive():
                p.kill()
            p.join(2)
        for conn in self.connections:
            conn.close()
        self.connections, self.processes = [], []


def advantages(rewards, values, bootstrap, gamma, gae_lambda):
    """One complete episode, with bootstrap already zeroed for true termination."""
    result = np.empty(len(rewards), np.float32)
    carry, following = 0.0, float(bootstrap)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * following - values[t]
        carry = delta + gamma * gae_lambda * carry
        result[t], following = carry, values[t]
    return result, result + np.asarray(values, np.float32)


@torch.no_grad()
def collect_episodes(policy, workers, config, curriculum=None):
    policy.eval()
    count = len(workers.connections)
    device = next(policy.parameters()).device
    def reset(rank):
        descriptor = None if curriculum is None else curriculum.next_scene(
            config["environment"]["scenario"]["duel"]["controlled"]["preset"], rank)
        workers.send(rank, "reset", descriptor)
    for rank in range(count):
        reset(rank)
    observations = [workers.receive(rank)[0] for rank in range(count)]
    states = [policy.initial_state(1) for _ in range(count)]
    buffers = [dict(observations=[], actions=[], log_probs=[], values=[], rewards=[],
                    terminated=[], truncated=[], execution=[]) for _ in range(count)]
    steps = [0] * count
    active = list(range(count))
    complete = []
    last_progress = time.monotonic()
    while active:
        batch = tensor_observation([observations[r] for r in active], device)
        distribution, values, next_state = policy(batch, combine_states([states[r] for r in active]))
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        actions_np, logp_np, values_np = (x.cpu().numpy() for x in (actions, log_probs, values))
        for i, rank in enumerate(active):
            b = buffers[rank]
            b["observations"].append(observations[rank])
            for key, value in (("actions", actions_np[i]), ("log_probs", logp_np[i]), ("values", values_np[i])):
                b[key].append(value)
            states[rank] = select_state(next_state, slice(i, i + 1))
            workers.send(rank, "step", int(actions_np[i]))
        remaining = []
        resets = []
        for rank in active:
            obs, reward, terminated, truncated, info = workers.receive(rank)
            b = buffers[rank]
            b["rewards"].append(reward)
            b["terminated"].append(terminated)
            b["truncated"].append(truncated)
            b["execution"].append({k: info.get(k) for k in (
                "executed_action", "wire_action", "shot_requested", "shot_accepted", "reject_reason",
                "simulator_actual_shots", "simulator_rejected_requests", "action_masked")})
            steps[rank] += 1
            observations[rank] = obs
            if terminated or truncated:
                bootstrap = 0.0
                if truncated and not terminated:
                    final = tensor_observation([obs], device)
                    value, _ = policy.critic(final, states[rank][1], torch.ones_like(final["dt_s"]))
                    bootstrap = value.item()
                b["observations"].append(obs)
                b["observations"] = {key: np.stack([o[key] for o in b["observations"]]) for key in obs}
                for key in ("actions", "log_probs", "values", "rewards", "terminated", "truncated"):
                    b[key] = np.asarray(b[key])
                b["advantages"], b["returns"] = advantages(
                    b["rewards"], b["values"], bootstrap, config["ppo"]["gamma"], config["ppo"]["gae_lambda"])
                b.update(bootstrap=bootstrap, info=info, worker=rank)
                complete.append(b)
                if steps[rank] >= config["ppo"]["n_steps"]:
                    # Leave this process at its completed episode, until the next batch.
                    continue
                buffers[rank] = dict(observations=[], actions=[], log_probs=[], values=[], rewards=[],
                                     terminated=[], truncated=[], execution=[])
                states[rank] = policy.initial_state(1)
                reset(rank)
                resets.append(rank)
            remaining.append(rank)
        for rank in resets:
            observations[rank] = workers.receive(rank)[0]
        active = remaining
        if time.monotonic()-last_progress >= 15 and "runtime" in config:
            print(f"SAMPLING steps={sum(steps)} completed_episodes={len(complete)} active={len(active)} min_worker_steps={min(steps)} max_worker_steps={max(steps)}",flush=True)
            last_progress = time.monotonic()
    return complete
