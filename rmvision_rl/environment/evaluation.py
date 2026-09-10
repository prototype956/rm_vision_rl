"""Bounded rule-baseline evaluation: warm up, run a fixed window, close and naturally settle."""
from dataclasses import asdict, dataclass

from rmvision_rl.environment.warmup import WarmupSession
from rmvision_rl.policy.observations import TensorPolicy
from rmvision_rl.scoring.window import WindowScore, integer


@dataclass(frozen=True)
class EvaluationConfig:
    window_ms: int = 30_000
    max_settle_steps: int = 1000
    target_hp: int = 100_000

    def __post_init__(self):
        integer(self.window_ms, "window_ms", 10)
        integer(self.max_settle_steps, "max_settle_steps", 1)
        integer(self.target_hp, "target_hp", 1)
        if self.window_ms > 120_000 or self.window_ms % 10 or self.max_settle_steps > 6000 or self.target_hp > 1_000_000:
            raise ValueError("invalid evaluation duration, settlement bound or target HP")


class EvaluationSession:
    """The experiment owner; all timing and score fields stay outside estimator/policy input.

    Once this object owns the pair, do not advance the client or bridge independently.
    Transport ambiguity and failed publication acknowledgements latch fault, requiring explicit
    recovery/reset. A timeout never silently starts another episode or publishes a complete score.
    """

    def __init__(self, client, bridge, config=None, warmup_config=None, *, policy=None, policy_mode="rule"):
        if policy_mode not in ("rule", "nine", "fire_only") or ((policy is None) != (policy_mode == "rule")):
            raise ValueError("rule mode requires no callback; nine/fire_only require a policy callback")
        if policy is not None and not callable(policy):
            raise TypeError("policy must be callable")
        self.client, self.bridge = client, bridge
        self.policy, self.policy_mode = policy, policy_mode
        self.config = config or EvaluationConfig()
        self.warmup = WarmupSession(client, bridge, warmup_config)
        self.status = "idle"
        self.response = None
        self.score = None
        self.error = None

    def reset(self, seed, scenario=None):
        self.status, self.error, self.score = "fault", None, None
        scene = dict(scenario or {})
        scene.setdefault("target_hp", self.config.target_hp)
        if isinstance(self.policy, TensorPolicy):
            self.policy.reset()
        self.warmup.reset(seed, scene)
        self.response = self.warmup.response
        self.status = "warming"
        return self.info()

    def info(self):
        now = self.response["sim_time_ns"] if self.response else 0
        start = self.score.start_ns if self.score else None
        result = dict(status=self.status, config=asdict(self.config), error=self.error, physical_time_ns=now,
                    evaluation_time_ns=min(now-start, self.config.window_ms*1_000_000) if start is not None else None,
                    settlement_time_ns=max(0, now-self.score.end_ns) if self.score else 0,
                    score=self.score.summary() if self.score else None)
        if self.policy_mode != "rule":
            result["policy_mode"] = self.policy_mode
        return result

    def _close(self):
        self.response = self.client.end_window(self.config.max_settle_steps)
        self.score.ingest(self.response)
        self.status = "settling"
        self._finish_if_terminal()

    def _finish_if_terminal(self):
        state = self.response["data"]["settlement"]
        if state is not None and state["status"] in ("complete", "timed_out"):
            self.score.finish(self.response)
            self.status = "complete" if state["status"] == "complete" else "settlement_timed_out"

    def advance(self):
        if self.status not in ("warming", "evaluating", "settling"):
            raise RuntimeError("evaluation is not active; reset required")
        result = None
        responses = []
        try:
            if self.status == "warming":
                result = self.warmup.advance()["vision"]
                self.response = self.warmup.response
                responses.append(self.response)
                if self.warmup.status == "ready":
                    start = self.warmup.evaluation_start_ns
                    self.score = WindowScore(self.response["round_id"], start,
                                             start+self.config.window_ms*1_000_000)
                    self.score.ingest(self.response)
                    if self.policy_mode == "rule":
                        self.bridge.begin_evaluation(self.response["round_id"], start, self.score.end_ns)
                    else:
                        self.bridge.begin_evaluation(self.response["round_id"], start, self.score.end_ns,
                                                     policy_mode=self.policy_mode)
                    self.warmup.status = "handed_off"
                    self.status = "evaluating"
                elif self.warmup.status == "timed_out":
                    self.status = "warmup_timed_out"
            elif self.status == "evaluating":
                before = self.response["sim_time_ns"]
                if before >= self.score.end_ns:
                    raise RuntimeError("evaluation advanced beyond its fixed window")
                if isinstance(self.policy, TensorPolicy):
                    self.policy.begin_step()
                result = (self.bridge.step(self.response) if self.policy is None else
                          self.bridge.step(self.response, self.policy))
                self.response = self.client.advance(**result["command"])
                responses.append(self.response)
                self.bridge.ack(True)
                if self.response["sim_time_ns"] != before+10_000_000:
                    raise RuntimeError("evaluation control step changed")
                self.score.ingest(self.response)
                if self.response["sim_time_ns"] == self.score.end_ns:
                    self._close()
                    responses.append(self.response)
            else:
                self.response = self.client.settle()
                responses.append(self.response)
                self.score.ingest(self.response)
                self._finish_if_terminal()
            return dict(evaluation=self.info(), vision=result, responses=responses)
        except Exception as error:
            self.status, self.error = "fault", str(error)
            if self.score:
                self.score.invalidate(error)
            raise
