"""管理接收二维检测帧的 C++ 桥接，仿真命令发布后需单独确认。"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import select
import subprocess


def control_referee(data, now_ns):
    """合并低频裁判规则与最新的自身武器机械禁射状态。

    热量、生命和弹量沿用裁判采样的时间戳与有效性；第 5–7 位
    （射击间隔、供弹、枪口状态）来自自身武器反馈，不使用评估真值。
    旧仿真器缺少此通道时保留原有的保守处理。
    """
    referee = copy.deepcopy(data["self_referee"])
    if "self_weapon" not in data:
        return referee
    weapon = data["self_weapon"]
    if not isinstance(weapon, dict) or type(weapon.get("version")) is not int or weapon["version"] != 1:
        raise ValueError("unsupported self_weapon feedback version")
    sample = weapon.get("sample_ns")
    blocks = weapon.get("fire_blocks")
    if (type(sample) is not int or sample < 0 or type(blocks) is not int or
            blocks < 0 or blocks & ~0xe0 or type(weapon.get("valid")) is not bool):
        raise ValueError("invalid self_weapon feedback")
    if not weapon["valid"] or not 0 <= now_ns - sample <= 10_000_000:
        # 自身机械反馈缺失或过期时，不能解除已有禁射条件。
        referee["valid"] = False
        referee["fire_permitted"] = False
        return referee
    old_blocks = referee["fire_blocks"]
    referee["fire_blocks"] = (old_blocks & ~0xe0) | blocks
    # 保留原因位无法解释的显式禁射；其他情况根据合并后的原因重新判定。
    referee["fire_permitted"] = bool(
        (referee["fire_permitted"] or old_blocks != 0) and referee["fire_blocks"] == 0)
    return referee


class VisionBridge:
    """管理 C++ 桥接的同步 JSONL 请求及当前待提交周期。"""
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
        """准备当前控制周期，在策略回调处暂停或缓存无需回调的命令。

        本方法不发布命令或推进物理；后续必须提交并确认，或取消该周期。

        Args:
            response: 当前仿真响应，仅提取控制所需字段。
            external: 是否使用外部策略回调；False 表示规则控制。

        Returns:
            策略观测或已计算的控制结果副本。

        Raises:
            RuntimeError: 上一周期尚未处理，或桥接返回错误。
            TimeoutError: 等待桥接响应超时。
        """
        if self._prepared is not None:
            raise RuntimeError("submit/ack or cancel the prepared cycle first")
        # 仅转发明确列出的控制字段，评估真值、事件、奖励、场景和种子不进入 C++。
        data = response["data"]
        result = self.exchange(dict(op="step_policy" if external else "step",
                                    round_id=response["round_id"], sim_time_ns=response["sim_time_ns"],
                                    visual_frames=data["visual_frames"], feedback=data["feedback"],
                                    self_referee=control_referee(data, response["sim_time_ns"])))
        self._prepared = result
        return copy.deepcopy(result)

    def submit(self, action):
        """使用原关联 token 提交动作，恢复同一次暂停的控制计算。

        不向仿真发布命令或推进物理，调用前必须先获得待决策观测。

        Args:
            action: [0, 8] 内的 Python 整数，且必须被当前动作掩码允许。

        Returns:
            包含待发布控制命令的桥接响应副本。

        Raises:
            RuntimeError: 当前没有等待动作的策略回调，或桥接返回错误。
            ValueError: 动作类型、范围或掩码检查失败。
            TimeoutError: 等待桥接响应超时。
        """
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
        """取消未提交周期，不发布命令；取消后 C++ 桥接必须重新 Reset。"""
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
                from src.policy.observations import TensorPolicy
                if isinstance(policy, TensorPolicy):
                    policy.set_generation(result["metadata"]["track_generation"])
                # 策略仅接收语义观测的副本，不接收 token 或实验元数据。
                action = policy(copy.deepcopy(result["observation"]))
                return self.submit(action)
            except Exception:
                # 回调失败后不能作为新周期继续，也不能静默替换为 WAIT。
                self.process.terminate()
                self.process.wait(timeout=5)
                raise
        return result

    def ack(self, success):
        """将仿真命令发布结果反馈给 C++，成功交换后清除本地待处理周期。"""
        result = self.exchange(dict(op="ack", success=success))
        self._prepared = None
        return result

    def begin_training(self, round_id, start_ns, *, policy_mode="fire_only"):
        return self.exchange(dict(op="begin_training", round_id=round_id, start_ns=start_ns,
                                  policy_mode=policy_mode))

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
