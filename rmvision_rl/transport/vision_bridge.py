"""Own a detector-frame-only C++ bridge; actual simulator publication is acknowledged separately."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import select
import subprocess


class VisionBridge:
    def __init__(self, process):
        self.process = process
        self._prepared = None

    def exchange(self, value):
        self.process.stdin.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 30)[0]:
            raise TimeoutError("vision bridge response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("vision bridge exited without a response")
        result = json.loads(line)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result

    def reset(self, round_id):
        self.cancel()
        return self.exchange(dict(op="reset", round_id=round_id))

    def prepare(self, response, *, external=True):
        """Stop at the policy callback, or cache a no-callback command; never publish here."""
        if self._prepared is not None:
            raise RuntimeError("submit/ack or cancel the prepared cycle first")
        # Explicit allowlist: evaluation truth, events, rewards, scenario and seed stay outside C++.
        data = response["data"]
        result = self.exchange(dict(op="step_policy" if external else "step",
                                    round_id=response["round_id"], sim_time_ns=response["sim_time_ns"],
                                    visual_frames=data["visual_frames"], feedback=data["feedback"],
                                    self_referee=data["self_referee"]))
        self._prepared = result
        return copy.deepcopy(result)

    def submit(self, action):
        """Resume exactly the prepared callback with its original correlation token."""
        result = self._prepared
        if result is None or result.get("kind") != "policy_observation":
            raise RuntimeError("no policy callback awaiting an action")
        if type(action) is not int or not 0 <= action <= 8:
            raise ValueError("policy must return a Python integer action in [0,8]")
        if not result["observation"]["action_mask"][action]:
            raise ValueError("masked policy action")
        self._prepared = self.exchange(dict(op="policy_action", token=result["token"], action=action))
        return copy.deepcopy(self._prepared)

    def cancel(self):
        """Discard an unsubmitted cycle. C++ requires Reset afterwards, without publication."""
        if self._prepared is None:
            return
        if self._prepared.get("kind") == "policy_observation":
            request = dict(op="policy_cancel", token=self._prepared["token"])
        else:
            request = dict(op="cancel")
        self.exchange(request)
        self._prepared = None

    def step(self, response, policy=None):
        result = self.prepare(response, external=policy is not None)
        if result.get("kind") == "policy_observation":
            try:
                from rmvision_rl.policy.observations import TensorPolicy
                if isinstance(policy, TensorPolicy):
                    policy.set_generation(result["metadata"]["track_generation"])
                # The caller sees a copy of the semantic whitelist, never tokens or experiment metadata.
                action = policy(copy.deepcopy(result["observation"]))
                return self.submit(action)
            except Exception:
                # A partial callback cannot be resumed as a new cycle or silently replaced with WAIT.
                self.process.terminate()
                self.process.wait(timeout=5)
                raise
        return result

    def ack(self, success):
        result = self.exchange(dict(op="ack", success=success))
        self._prepared = None
        return result

    def begin_training(self, round_id, start_ns):
        return self.exchange(dict(op="begin_training", round_id=round_id, start_ns=start_ns))

    def begin_evaluation(self, round_id, start_ns, end_ns, policy_mode="rule"):
        request = dict(op="begin_evaluation", round_id=round_id, start_ns=start_ns, end_ns=end_ns)
        if policy_mode != "rule":
            request["policy_mode"] = policy_mode
        return self.exchange(request)


@contextmanager
def vision_worker(binary, vision_root, output, *, diagnostics=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = output / "logger.json"
    config.write_text(json.dumps(dict(schema_version=1, log_dir="logs", level="warn", console=False)))
    with (output / "stderr.log").open("w") as log:
        command = [str(binary.resolve()), str(vision_root / "src/config/modules"), str(config)]
        if diagnostics is not None:
            command.append(str(diagnostics.resolve()))
        process = subprocess.Popen(command,
                                   cwd=output, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   text=True, bufsize=1)
        try:
            bridge = VisionBridge(process)
            yield bridge
            bridge.cancel()
            process.stdin.close()
            if process.wait(timeout=10) != 0:
                raise RuntimeError("vision bridge failed; see " + str(output / "stderr.log"))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
            if not process.stdin.closed:
                process.stdin.close()
