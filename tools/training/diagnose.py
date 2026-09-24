"""记录真实 MaskablePPO 更新过程，保持原有采样和优化行为。"""
import argparse
from contextlib import ExitStack, contextmanager
import copy
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import signal
import tempfile

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from src.training.config import ROOT
from src.training.environment import make_environment, vectorize
from src.training.models import load_model, read_checkpoint_metadata
from tools.training.view import _atomic_json


def gae(rewards, values, starts, last_value, last_done, gamma, lam):
    """用 float64 独立重算 GAE；rewards 已包含时间截断的价值自举补偿。"""
    rewards, values = np.asarray(rewards, dtype=float), np.asarray(values, dtype=float)
    starts = np.asarray(starts, dtype=float)
    advantages, deltas = np.zeros_like(rewards), np.zeros_like(rewards)
    carry = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        continuation = 1.0 - (last_done if i == len(rewards) - 1 else starts[i + 1])
        successor = last_value if i == len(rewards) - 1 else values[i + 1]
        deltas[i] = rewards[i] + gamma * successor * continuation - values[i]
        carry = deltas[i] + gamma * lam * continuation * carry
        advantages[i] = carry
    return advantages, deltas


def reward_contributions(raw, rewards, values, starts, last_value, last_done, gamma, lam):
    baseline, _ = gae(rewards, values, starts, last_value, last_done, gamma, lam)
    indices = np.flatnonzero(raw)
    contributions = np.zeros((len(indices), len(raw)))
    max_error = 0.0
    for row, index in enumerate(indices):
        removed = np.array(rewards, dtype=float, copy=True)
        removed[index] -= raw[index]
        modified, _ = gae(removed, values, starts, last_value, last_done, gamma, lam)
        contributions[row] = baseline - modified
        expected = np.zeros(len(raw))
        expected[index] = raw[index]
        for previous in range(index - 1, -1, -1):
            expected[previous] = expected[previous + 1] * gamma * lam * (1 - starts[previous + 1])
        max_error = max(max_error, float(np.max(np.abs(expected - contributions[row]))))
    return indices, contributions, max_error


def value_metrics(values, targets):
    variance = float(np.var(targets))
    return {"mse": float(np.mean((values - targets) ** 2)),
            "explained_variance": None if variance == 0 else float(1 - np.var(targets - values) / variance)}


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


class CsvLog:
    def __init__(self, path, *, append=False):
        fields = None
        if append and path.exists() and path.stat().st_size:
            with path.open(newline="") as file:
                fields = next(csv.reader(file))
        self.file = path.open("a" if append else "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields) if fields else None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


class EventCapture(gym.Wrapper):
    """在 DummyVecEnv 自动重置已结束回合前读取物理事件。"""
    def __init__(self, env):
        super().__init__(env)
        self.episode = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.episode += 1
        return result

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        snapshot = self.unwrapped.debug_snapshot()
        info = dict(info, diagnostic={"episode": self.episode, "events": snapshot["data"]["events"],
                                     "projectiles": snapshot["data"]["evaluation"]["projectiles"],
                                     "terminated": terminated, "truncated": truncated})
        return obs, reward, terminated, truncated, info


@contextmanager
def local_method(instance, name, replacement):
    """临时替换单个实例的方法，退出后恢复，不修改全局或第三方类。"""
    present, previous = name in instance.__dict__, instance.__dict__.get(name)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if present:
            setattr(instance, name, previous)
        else:
            delattr(instance, name)


class UpdateTrace:
    def __init__(self, model, writer, rollout, base_step):
        self.model, self.writer, self.rollout, self.base_step = model, writer, rollout, base_step
        self.indices, self.orders, self.rows = None, [], []
        self.pending = []
        self.batch = 0

    def flush(self, executed):
        for row in self.pending:
            row["optimizer_executed"] = executed
            self.writer.write(row)
            self.rows.append(row)
        self.pending = []

    @contextmanager
    def install(self):
        buffer, policy = self.model.rollout_buffer, self.model.policy
        original_get, original_evaluate = buffer._get_samples, policy.evaluate_actions

        def samples(indices, *args, **kwargs):
            self.flush(False)
            self.indices = indices.copy()
            self.orders.append(indices.tolist())
            self.batch += 1
            return original_get(indices, *args, **kwargs)

        def evaluate(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            with torch.no_grad():
                ids = self.indices
                raw = torch.as_tensor(buffer.advantages[ids].flatten(), device=self.model.device)
                used = (raw - raw.mean()) / (raw.std() + 1e-8) if self.model.normalize_advantage else raw
                old_log = torch.as_tensor(buffer.log_probs[ids].flatten(), device=self.model.device)
                ratios = torch.exp(result[1] - old_log)
                clip = self.model.clip_range(self.model._current_progress_remaining)
                for j, index in enumerate(ids):
                    a, ratio = float(used[j]), float(ratios[j])
                    self.pending.append({
                        "rollout": self.rollout, "epoch": (self.batch - 1) // (buffer.buffer_size // self.model.batch_size) + 1,
                        "minibatch": self.batch, "sample_index": int(index), "training_step": self.base_step + int(index) + 1,
                        "action": int(buffer.actions[index, 0]), "fire_legal": bool(buffer.action_masks[index, fire_indices(self.model.action_space.n)].any()),
                        "raw_advantage": float(raw[j]), "normalized_advantage": a,
                        "ratio": ratio, "ratio_outside_clip": abs(ratio - 1) > clip,
                        "surrogate_clipped": (a >= 0 and ratio > 1 + clip) or (a < 0 and ratio < 1 - clip),
                        "old_log_prob": float(old_log[j]), "log_prob": float(result[1][j]),
                        "value": float(result[0].flatten()[j]), "optimizer_executed": False})
            return result

        with ExitStack() as stack:
            stack.enter_context(local_method(buffer, "_get_samples", samples))
            stack.enter_context(local_method(policy, "evaluate_actions", evaluate))
            handle = policy.optimizer.register_step_post_hook(lambda *args: self.flush(True))
            stack.callback(handle.remove)
            try:
                yield self
            finally:
                self.flush(False)


def fire_indices(action_count):
    return [1] if action_count == 2 else [2, 4, 6, 8]


def probabilities(model, observations, masks):
    with torch.no_grad():
        tensors = {key: torch.as_tensor(value, device=model.device) for key, value in observations.items()}
        distribution = model.policy.get_distribution(tensors, action_masks=masks)
        return (distribution.distribution.probs[:, fire_indices(model.action_space.n)].sum(dim=1).cpu().numpy().copy(),
                model.policy.predict_values(tensors).flatten().cpu().numpy().copy())


class CaptureRollout(BaseCallback):
    def __init__(self, output, step_writer, event_file):
        super().__init__()
        self.output, self.step_writer, self.event_file = output, step_writer, event_file
        self.rollout = 0
        self.all_steps, self.all_events, self.summaries = [], [], []
        self.requests, self.shots = {}, {}

    def _on_rollout_start(self):
        self.rollout += 1
        self.rows = []
        self.base_step = self.model.num_timesteps

    def _on_step(self):
        info = self.locals["infos"][0]
        if info["action_masked"]:
            raise RuntimeError("PPO submitted a masked action")
        capture = info["diagnostic"]
        row = {"rollout": self.rollout, "sample_index": len(self.rows), "episode": capture["episode"],
               "episode_step": info["episode_steps"], "training_step": self.model.num_timesteps,
               "time_s": info["episode_time_s"], "physical_time_ns": info["physical_time_ns"],
               "action": int(self.locals["actions"][0]),
               "fire_legal": bool(self.locals["action_masks"][0, fire_indices(self.model.action_space.n)].any()),
               "fire_action": int(self.locals["actions"][0]) in fire_indices(self.model.action_space.n),
               "selected_slot": info["selected_slot"], "slot_switched": info["slot_switched"],
               "shot_requested": info["shot_requested"], "shot_accepted": info["shot_accepted"],
               "action_masked": info["action_masked"], "reject_reason": info["reject_reason"],
               "actual_shots": info["actual_shots"], "raw_reward": float(self.locals["rewards"][0]),
               "terminated": capture["terminated"], "truncated": capture["truncated"],
               "end_reason": info["end_reason"], "active_projectiles": len(capture["projectiles"])}
        clock_before = info.get("decision_clock_before") or {}
        row.update(decision_due=clock_before.get("decision_due"),
                   decision_wait_ms=clock_before.get("wait_ms"),
                   clock_episode_stream=clock_before.get("episode_stream"),
                   physical_fire_legal=info.get("physical_fire_legal_before"))
        damage = 0.0
        for event in capture["events"]:
            data = event["data"]
            item = dict(event, episode=capture["episode"], rollout=self.rollout,
                        training_step=self.model.num_timesteps, episode_step=info["episode_steps"])
            request_key = (capture["episode"], data.get("request_id"))
            projectile_key = (capture["episode"], data.get("projectile_id"))
            if event["kind"] == "fire_requested" and data.get("robot_id") == 1:
                self.requests[request_key] = item
            if event["kind"] == "shot_fired" and data.get("robot_id") == 1:
                self.shots[projectile_key] = item
            request, shot = self.requests.get(request_key), self.shots.get(projectile_key)
            item.update(request_step=request["training_step"] if request else None,
                        shot_step=shot["training_step"] if shot else None,
                        association="request_id" if request else "unassociated")
            item["cross_rollout"] = bool(request and request["rollout"] != self.rollout)
            # 用回合限定标识符，避免关联重置后复用的编号。
            item["cross_episode"] = False
            if event["kind"] == "damage_applied" and data.get("shooter") == 1:
                damage += float(data["actual"])
            self.event_file.write(json.dumps(item, allow_nan=False) + "\n")
            self.all_events.append(item)
        self.event_file.flush()
        row["event_damage"] = damage
        self.rows.append(row)
        # 先保存原始状态转换，避免后续采样失败导致 GAE 计算前的数据丢失。
        with (self.output / "sampling.jsonl").open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        return True

    def _on_rollout_end(self):
        b = self.model.rollout_buffer
        self.observations = {key: value[:, 0].copy() for key, value in b.observations.items()}
        self.arrays = {key: getattr(b, key)[:, 0].copy() for key in
                       ("actions", "rewards", "values", "log_probs", "episode_starts", "advantages", "returns", "action_masks")}
        arrays = self.arrays
        arrays["raw_rewards"] = np.array([r["raw_reward"] for r in self.rows])
        self.last_value = float(self.locals["values"].flatten()[0])
        self.last_done = bool(self.locals["dones"][0])
        self.before_probability, self.before_value = probabilities(self.model, self.observations, arrays["action_masks"])
        ref, deltas = gae(arrays["rewards"], arrays["values"], arrays["episode_starts"], self.last_value,
                          self.last_done, self.model.gamma, self.model.gae_lambda)
        indices, contributions, removal_error = reward_contributions(
            arrays["raw_rewards"], arrays["rewards"], arrays["values"], arrays["episode_starts"],
            self.last_value, self.last_done, self.model.gamma, self.model.gae_lambda)
        self.reference, self.deltas = ref, deltas
        self.removal_error = removal_error
        np.savez_compressed(self.output / f"rollout-{self.rollout:02d}.npz", **arrays,
                            **{"obs_" + key: val for key, val in self.observations.items()},
                            last_value=self.last_value, last_done=self.last_done, td_error=deltas,
                            reference_advantages=ref, reward_indices=indices, reward_contributions=contributions,
                            fire_probability_before=self.before_probability)
        if self.rollout == 1:
            self.parity_model = BytesIO()
            self.model.save(self.parity_model)
            self.parity_buffer = copy.deepcopy(b)
            # 复制后的实例适配器仍通过闭包引用正在记录的缓冲区，
            # 因此对照更新必须使用类上的原始实现。
            self.parity_buffer.__dict__.pop("_get_samples", None)
            self.parity_rng = rng_state()

    def finish(self, trace):
        p_after, v_after = probabilities(self.model, self.observations, self.arrays["action_masks"])
        b = self.arrays
        groups = {}
        for i, row in enumerate(self.rows):
            group = "masked" if not row["fire_legal"] else "fire" if row["fire_action"] else "track"
            row.update(group=group, buffer_reward=float(b["rewards"][i]),
                       timeout_bootstrap=float(b["rewards"][i] - row["raw_reward"]),
                       value_before=float(b["values"][i]), value_after=float(v_after[i]),
                       old_log_prob=float(b["log_probs"][i]), old_action_probability=float(np.exp(b["log_probs"][i])),
                       td_error=float(self.deltas[i]), advantage=float(b["advantages"][i]), target=float(b["returns"][i]),
                       fire_probability_before=float(self.before_probability[i]), fire_probability_after=float(p_after[i]),
                       probability_change=float(p_after[i] - self.before_probability[i]))
            self.step_writer.write(row)
            groups.setdefault(group, []).append(row)
        self.all_steps.extend(self.rows)
        normalized = {}
        for group, rows in groups.items():
            ids = {r["sample_index"] for r in rows}
            records = [r for r in trace.rows if r["sample_index"] in ids and r["optimizer_executed"]]
            normalized[group] = {"samples": len(rows), "mean_advantage": float(np.mean([r["advantage"] for r in rows])),
                                 "mean_probability_before": float(np.mean([r["fire_probability_before"] for r in rows])),
                                 "mean_probability_after": float(np.mean([r["fire_probability_after"] for r in rows])),
                                 "mean_probability_change": float(np.mean([r["probability_change"] for r in rows])),
                                 "normalized_advantage_mean": float(np.mean([r["normalized_advantage"] for r in records])) if records else None,
                                 "normalized_positive_fraction": float(np.mean([r["normalized_advantage"] > 0 for r in records])) if records else None,
                                 "surrogate_clipped_fraction": float(np.mean([r["surrogate_clipped"] for r in records])) if records else None}
        summary = {"rollout": self.rollout, "steps": len(self.rows),
                   "reward": sum(r["raw_reward"] for r in self.rows), "nonzero_rewards": int(np.count_nonzero(b["raw_rewards"])),
                   "reward_event_max_error": max(abs(r["event_damage"] - r["raw_reward"]) for r in self.rows),
                   "gae_max_error": float(np.max(np.abs(self.reference - b["advantages"]))),
                   "gae_matches": bool(np.allclose(self.reference, b["advantages"], rtol=2e-5, atol=2e-4)),
                   "reward_removal_max_error": self.removal_error, "groups": normalized,
                   "value_before": value_metrics(self.before_value, b["returns"]),
                   "value_after": value_metrics(v_after, b["returns"]),
                   "target_mean": float(np.mean(b["returns"])), "target_std": float(np.std(b["returns"])),
                   "optimizer_steps": len({r["minibatch"] for r in trace.rows if r["optimizer_executed"]}),
                   "minibatch_rows": len(trace.rows), "last_value": self.last_value, "last_done": self.last_done,
                   "episode_boundaries": sum(r["terminated"] or r["truncated"] for r in self.rows),
                   "pending_at_episode_end": sum(r["active_projectiles"] for r in self.rows if r["terminated"] or r["truncated"]),
                   "timeout_compensation_only_at_truncation": all(r["timeout_bootstrap"] == 0 or r["truncated"] for r in self.rows)}
        self.summaries.append(summary)
        np.savez_compressed(self.output / f"update-{self.rollout:02d}.npz", fire_probability_after=p_after, value_after=v_after)
        return summary


def compare_trees(a, b):
    if isinstance(a, torch.Tensor):
        return float(torch.max(torch.abs(a - b)).item()) if a.numel() else 0.0
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("state dictionary keys differ")
        return max((compare_trees(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("state lengths differ")
        return max((compare_trees(x, y) for x, y in zip(a, b)), default=0.0)
    if a != b:
        raise AssertionError(f"state scalar differs: {a} != {b}")
    return 0.0


def verify_parity(model, callback, trace):
    """仅重放优化器更新以核对诊断影响，不重放环境，结束后恢复随机数状态。"""
    post_rng = rng_state()
    try:
        callback.parity_model.seek(0)
        control = MaskablePPO.load(callback.parity_model, device=model.device)
        control.rollout_buffer = callback.parity_buffer
        control._current_progress_remaining = model._current_progress_remaining
        control.set_logger(configure(folder=None, format_strings=[]))
        orders = []
        original = control.rollout_buffer._get_samples

        def samples(indices, *args, **kwargs):
            orders.append(indices.tolist())
            return original(indices, *args, **kwargs)

        restore_rng(callback.parity_rng)
        with local_method(control.rollout_buffer, "_get_samples", samples):
            control.train()
        untraced_rng = rng_state()
        rng_equal = (post_rng[0] == untraced_rng[0] and np.array_equal(post_rng[1][1], untraced_rng[1][1])
                     and post_rng[1][2:] == untraced_rng[1][2:] and torch.equal(post_rng[2], untraced_rng[2])
                     and (post_rng[3] is None or all(torch.equal(a, b) for a, b in zip(post_rng[3], untraced_rng[3]))))
        weight_error = compare_trees(model.policy.state_dict(), control.policy.state_dict())
        optimizer_error = compare_trees(model.policy.optimizer.state_dict(), control.policy.optimizer.state_dict())
        return {"status": "passed" if orders == trace.orders and rng_equal and max(weight_error, optimizer_error) <= 1e-7 else "failed",
                "minibatch_orders_equal": orders == trace.orders, "rng_equal": rng_equal,
                "weight_max_error": weight_error, "optimizer_max_error": optimizer_error, "extra_environment_steps": 0}
    finally:
        restore_rng(post_rng)
        del callback.parity_buffer, callback.parity_model


def diagnose(checkpoint, *, rollouts=3, device=None, output_dir=None):
    if type(rollouts) is not int or rollouts < 1:
        raise ValueError("rollouts must be a positive integer")
    checkpoint = Path(checkpoint).resolve()
    content = checkpoint.read_bytes()
    metadata = read_checkpoint_metadata(BytesIO(content))
    config = copy.deepcopy(metadata["config"])
    if device is not None:
        config["device"] = device
    if torch.device(config["device"]).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    parent = Path(output_dir).resolve() if output_dir else ROOT / "artifacts/diagnostics"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="diagnose-", dir=parent))
    snapshot = output / "source.zip"
    snapshot.write_bytes(content)
    state = {"version": 1, "status": "starting", "created_at": datetime.now(timezone.utc).isoformat(),
             "source": str(checkpoint), "source_sha256": hashlib.sha256(content).hexdigest(),
             "config": config, "checkpoint_metadata": metadata, "requested_rollouts": rollouts,
             "start_timesteps": metadata["num_timesteps"], "completed_rollouts": 0, "summaries": [],
             "report": str(output / "report.html")}
    _atomic_json(output / "run.json", state)
    print(f"Diagnostic output: {output}", flush=True)
    callback = None
    old_term = signal.getsignal(signal.SIGTERM)

    def interrupt(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupt)
    try:
        with ExitStack() as stack:
            env = EventCapture(make_environment(config, output / "environment", output / "episodes.monitor.csv"))
            stack.callback(env.close)
            vec = vectorize(env)
            stack.callback(vec.close)
            model, _ = load_model(snapshot, vec, device=config["device"])
            logger = configure(str(output / "logs"), ["csv"])
            stack.callback(logger.close)
            model.set_logger(logger)
            step_writer, batch_writer = CsvLog(output / "steps.csv"), CsvLog(output / "minibatches.csv")
            stack.callback(step_writer.close)
            stack.callback(batch_writer.close)
            events = stack.enter_context((output / "events.jsonl").open("w"))
            callback = CaptureRollout(output, step_writer, events)
            state.update(status="running", actual_device=str(model.device))
            for number in range(1, rollouts + 1):
                print(f"Sampling rollout {number}/{rollouts}: {config['ppo']['n_steps']} steps", flush=True)
                trace = UpdateTrace(model, batch_writer, number, model.num_timesteps)
                # 适配器仅记录 learn() 内部调用的上游 train()，不额外执行更新。
                with trace.install():
                    model.learn(total_timesteps=config["ppo"]["n_steps"], reset_num_timesteps=False,
                                log_interval=None, callback=callback)
                if number == 1:
                    state["parity"] = verify_parity(model, callback, trace)
                    _atomic_json(output / "parity.json", state["parity"])
                summary = callback.finish(trace)
                state.update(completed_rollouts=number, summaries=callback.summaries,
                             last_timesteps=model.num_timesteps)
                _atomic_json(output / "run.json", state)
                model.logger.dump(step=model.num_timesteps)
                print(f"Rollout {number}: reward={summary['reward']}, GAE error={summary['gae_max_error']:.3g}, "
                      f"fire={summary['groups'].get('fire', {})}", flush=True)
                if state.get("parity", {}).get("status") != "passed":
                    raise RuntimeError("tracing parity failed; remaining rollouts cancelled")
            state["status"] = "complete"
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        print(state["error"], flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        try:
            state["source_sha256_after"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state["source_unchanged"] = state["source_sha256_after"] == state["source_sha256"]
        except OSError as error:
            state["source_unchanged"] = None
            state["source_check_error"] = str(error)
        from tools.training.diagnostic_report import write_report
        try:
            write_report(output, state, callback.all_steps if callback else [], callback.all_events if callback else [])
            state["report_status"] = "complete"
        except Exception as error:
            state.update(report_status="failed", report_error=f"{type(error).__name__}: {error}")
        _atomic_json(output / "run.json", state)
    print(f"Status: {state['status']}; report: {output / 'report.html'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description="Trace the reward → GAE → PPO probability learning signal")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, help="parent directory for a unique diagnostic session")
    args = parser.parse_args()
    try:
        output = diagnose(**vars(args))
        return 0 if json.loads((output / "run.json").read_text())["status"] == "complete" else 1
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Diagnosis failed: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
