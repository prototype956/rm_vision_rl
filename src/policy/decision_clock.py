"""在 10 ms 控制时间轴上生成独立于策略动作的射击决策机会。"""
import copy

import numpy as np


FEATURES = ["decision_due", "wait_s", "min_interval_s", "max_interval_s", "episode_interval_s"]
CONTRACT = {"version": 1, "features": FEATURES, "control_step_ms": 10,
            "first_decision": "episode_start", "missed_opportunity": "discard",
            "distribution": "discrete_uniform_inclusive"}


def validate_clock(value):
    """校验可复现的决策时钟配置；未提供配置时关闭时钟。"""
    if value is None:
        return None
    fields = {"min_interval_ms", "max_interval_ms", "resample", "seed"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"decision_clock requires exactly: {sorted(fields)}")
    for name in ("min_interval_ms", "max_interval_ms"):
        if type(value[name]) is not int or not 10 <= value[name] <= 60_000 or value[name] % 10:
            raise ValueError(f"decision_clock.{name} must be a multiple of 10 ms in [10,60000]")
    if value["max_interval_ms"] < value["min_interval_ms"]:
        raise ValueError("decision_clock minimum exceeds maximum")
    if value["resample"] not in ("decision", "episode"):
        raise ValueError("decision_clock.resample must be decision or episode")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**32:
        raise ValueError("decision_clock.seed must be an integer in [0,2**32)")
    return copy.deepcopy(value)


class DecisionClock:
    """仅在完成控制步后推进决策时钟，被屏蔽或 LOST 的周期也会消耗到期机会。

    每回合的随机数流独立于场景和 actor；观测历史或目标跟踪重置不应调用
    start_episode。新实例从回合流 0 开始，使检查点评估共享相同的首回合机会时间表。
    """

    def __init__(self, config):
        self.config = validate_clock(config)
        self.episode = -1
        self.step = self.next_step = self.decisions = 0
        self.interval_steps = 0
        self.rng = None

    @property
    def enabled(self):
        return self.config is not None

    @property
    def due(self):
        return not self.enabled or self.step == self.next_step

    def start_episode(self):
        """在预热结束后启动新的回合随机数流，并在首个控制步提供决策机会。"""
        if not self.enabled:
            return
        self.episode += 1
        self.rng = np.random.default_rng(np.random.SeedSequence(
            [self.config["seed"], self.episode, 0x53484f54]))
        self.step = self.next_step = self.decisions = 0
        self.interval_steps = self._draw() if self.config["resample"] == "episode" else 0

    def _draw(self):
        return int(self.rng.integers(self.config["min_interval_ms"] // 10,
                                     self.config["max_interval_ms"] // 10 + 1))

    def complete_step(self):
        """在成功推进物理后更新时钟；到期机会无论是否开火均被消耗。"""
        if not self.enabled:
            return
        if self.rng is None:
            raise RuntimeError("start the decision clock after warmup before stepping")
        if self.due:
            if self.config["resample"] == "decision":
                self.interval_steps = self._draw()
            self.next_step = self.step + self.interval_steps
            self.decisions += 1
        self.step += 1

    def info(self):
        if not self.enabled:
            return None
        return {"episode_stream": self.episode, "step": self.step,
                "decision_due": self.due, "wait_ms": (self.next_step - self.step) * 10,
                "next_decision_step": self.next_step, "decisions_completed": self.decisions,
                "sampled_interval_ms": self.interval_steps * 10 if self.interval_steps else None}

    def observation(self):
        return np.asarray([self.due, (self.next_step - self.step) * .01,
                           self.config["min_interval_ms"] / 1000,
                           self.config["max_interval_ms"] / 1000,
                           self.interval_steps * .01 if self.config["resample"] == "episode" else 0],
                          dtype=np.float32)
