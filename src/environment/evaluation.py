"""管理规则或回调策略的限时评估：预热、正式窗口、关窗及自然结算。"""
from dataclasses import asdict, dataclass

from src.environment.warmup import WarmupSession
from src.policy.observations import TensorPolicy
from src.policy.decision import DecisionPolicy
from src.scoring.window import WindowScore, integer


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
    """独占仿真与桥接的评估流程，计时和计分信息不进入估计器或策略输入。

    会话接管后，调用方不得单独推进仿真或桥接。通信结果不确定或发布确认失败时
    进入 fault 状态，需要显式恢复或重置；超时不会自动开始新回合或提供完整分数。
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
        self.end_reason = None

    def reset(self, seed, scenario=None):
        self.status, self.error, self.score = "fault", None, None
        self.end_reason = None
        self.selection = dict(selected_slot=-1, slot_switched=False, slot_switches=0)
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
        result = dict(status=self.status, config=asdict(self.config), error=self.error,
                    end_reason=self.end_reason, physical_time_ns=now,
                    selection=getattr(self, "selection", {}),
                    evaluation_time_ns=min(now, self.score.end_ns)-start if start is not None else None,
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
                    if isinstance(self.policy, DecisionPolicy):
                        self.policy.clock.start_episode()
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
                clock_before = (self.policy.clock.info()
                                if isinstance(self.policy, DecisionPolicy) else None)
                result = (self.bridge.step(self.response) if self.policy is None else
                          self.bridge.step(self.response, self.policy))
                self.response = self.client.advance(**result["command"])
                responses.append(self.response)
                self.bridge.ack(True)
                if self.response["sim_time_ns"] != before+10_000_000:
                    raise RuntimeError("evaluation control step changed")
                if isinstance(self.policy, DecisionPolicy):
                    self.policy.clock.complete_step()
                    if clock_before is not None:
                        result["decision_clock_before"] = clock_before
                        result["physical_fire_legal"] = self.policy.physical_fire_action is not None
                        result["decision_clock_after"] = self.policy.clock.info()
                control = result['control']
                self.selection = dict(selected_slot=control['selected_slot'],
                                      slot_switched=control.get('slot_switched', False),
                                      slot_switches=self.selection['slot_switches'] + int(control.get('slot_switched', False)))
                self.score.ingest(self.response)
                robots = {r["robot_id"]: r for r in self.response["data"]["evaluation"]["robots"]}
                death = ("controlled_dead" if robots[1]["hp"] <= 0 else
                         "target_destroyed" if robots[2]["hp"] <= 0 else None)
                if death or self.response["sim_time_ns"] == self.score.end_ns:
                    self.end_reason = death or "time_limit"
                    if death:
                        self.score.shorten(self.response["sim_time_ns"])
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
