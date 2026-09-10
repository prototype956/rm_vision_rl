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
        return self.exchange(dict(op="reset", round_id=round_id))

    def step(self, response, policy=None):
        # Explicit allowlist: evaluation truth, events, rewards, scenario and seed stay outside C++.
        data = response["data"]
        result = self.exchange(dict(op="step" if policy is None else "step_policy",
                                    round_id=response["round_id"], sim_time_ns=response["sim_time_ns"],
                                    visual_frames=data["visual_frames"], feedback=data["feedback"],
                                    self_referee=data["self_referee"]))
        if result.get("kind") == "policy_observation":
            try:
                # The caller sees a copy of the semantic whitelist, never tokens or experiment metadata.
                action = policy(copy.deepcopy(result["observation"]))
                if type(action) is not int or not 0 <= action <= 8:
                    raise ValueError("policy must return a Python integer action in [0,8]")
                if not result["observation"]["action_mask"][action]:
                    raise ValueError("masked policy action")
                return self.exchange(dict(op="policy_action", token=result["token"], action=action))
            except Exception:
                # A partial callback cannot be resumed as a new cycle or silently replaced with WAIT.
                self.process.terminate()
                self.process.wait(timeout=5)
                raise
        return result

    def ack(self, success):
        return self.exchange(dict(op="ack", success=success))

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
            yield VisionBridge(process)
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
