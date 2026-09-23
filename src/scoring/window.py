"""按实际出膛时间在 [start, end) 内的弹丸统计独立评估伤害。

不修改物理奖励和事件。账本不完整或校验失败时不提供正式分数；
完全一致的事件重传不会重复累计伤害。
"""
import json


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an integer >= " + str(minimum))
    return value


class WindowScore:
    """按出膛窗口关联弹丸事件，校验完整性并汇总正式评估分数。

    窗口采用纳秒时间戳和左闭右开区间，仅完整自然结算后提供 official_damage。
    """
    MAX_EVENTS = 100_000
    MAX_SHOTS = 10_000

    def __init__(self, round_id, start_ns, end_ns, robot_id=1):
        self.round_id = integer(round_id, "round_id", 1)
        self.start_ns = integer(start_ns, "start_ns")
        self.end_ns = integer(end_ns, "end_ns", start_ns+1)
        self.robot_id = integer(robot_id, "robot_id", 1)
        self.events, self.shots = {}, {}
        self.closed = False
        self.status = "collecting"
        self.error = None
        self.latest_time_ns = start_ns

    def invalidate(self, reason):
        self.status, self.error = "invalid", str(reason)

    def shorten(self, end_ns):
        """因机器人死亡提前关闭评估窗口，仍将结束时刻作为出膛归属的右开边界。"""
        end_ns = integer(end_ns, "end_ns", self.start_ns + 1)
        if self.closed or self.status != "collecting" or end_ns > self.end_ns:
            raise ValueError("cannot shorten this evaluation window")
        self.end_ns = end_ns
        for shot in self.shots.values():
            if shot["classification"] == "eligible" and shot["launched_at_ns"] >= end_ns:
                shot["classification"] = "at_or_after_end"

    def ingest(self, response):
        """接收同一回合的事件，去重并关联出膛、伤害和弹丸终结。

        校验失败会将账本置为 invalid 并抛出异常，之后必须开始新的评估账本。
        """
        if self.status == "invalid":
            raise RuntimeError("score ledger is invalid; start a new round")
        try:
            if not response["ok"] or response["round_id"] != self.round_id:
                raise ValueError("foreign/failed response cannot enter this score ledger")
            now = integer(response["sim_time_ns"], "sim_time_ns")
            for event in response["data"]["events"]:
                ident = integer(event["event_id"], "event_id", 1)
                canonical = json.dumps(event, sort_keys=True, allow_nan=False, separators=(",", ":"))
                if event["round_id"] != self.round_id:
                    raise ValueError("foreign round event")
                if ident in self.events:
                    if self.events[ident] != canonical:
                        raise ValueError("conflicting duplicate event")
                    continue
                if self.status != "collecting":
                    raise ValueError("new event after score finalization")
                if ident != len(self.events)+1 or ident > self.MAX_EVENTS:
                    raise ValueError("event gap/order error or ledger capacity exceeded")
                at = integer(event["time_ns"], "event time")
                if at > now:
                    raise ValueError("future event")
                kind, data = event["kind"], event["data"]
                if kind == "shot_fired":
                    pid = integer(data["projectile_id"], "projectile_id", 1)
                    request = integer(data["request_id"], "request_id", 1)
                    owner = integer(data["robot_id"], "robot_id", 1)
                    if pid in self.shots or len(self.shots) >= self.MAX_SHOTS:
                        raise ValueError("duplicate launch or shot capacity exceeded")
                    if any(s["request_id"] == request for s in self.shots.values()):
                        raise ValueError("one request launched multiple projectiles")
                    classification = ("other_robot" if owner != self.robot_id else
                                      "before_window" if at < self.start_ns else
                                      "at_or_after_end" if at >= self.end_ns else "eligible")
                    self.shots[pid] = dict(projectile_id=pid, request_id=request, robot_id=owner,
                                          launched_at_ns=at, classification=classification,
                                          damage=0, damage_time_ns=None, ended_at_ns=None, end_reason=None)
                elif kind in ("damage_applied", "projectile_ended"):
                    pid = integer(data["projectile_id"], "projectile_id", 1)
                    if pid not in self.shots:
                        raise ValueError("outcome has no actual launch record")
                    shot = self.shots[pid]
                    if data["request_id"] != shot["request_id"] or at < shot["launched_at_ns"]:
                        raise ValueError("outcome request/time does not match launch")
                    if shot["ended_at_ns"] is not None:
                        raise ValueError("duplicate terminal or damage after terminal")
                    if kind == "damage_applied":
                        if data["shooter"] != shot["robot_id"] or shot["damage_time_ns"] is not None:
                            raise ValueError("mismatched shooter or repeated projectile damage")
                        shot["damage"] = integer(data["actual"], "actual damage")
                        shot["damage_time_ns"] = at
                    else:
                        if shot["damage_time_ns"] is not None and at < shot["damage_time_ns"]:
                            raise ValueError("terminal predates damage")
                        if data["reason"] in ("reset", "truncated_by_reset"):
                            raise ValueError("reset truncation is not natural settlement")
                        shot["ended_at_ns"], shot["end_reason"] = at, data["reason"]
                elif kind == "evaluation_window_closed":
                    if self.closed or at != self.end_ns or data["window_end_ns"] != self.end_ns:
                        raise ValueError("wrong or repeated cutoff boundary")
                    self.closed = True
                self.events[ident] = canonical
            self.latest_time_ns = max(self.latest_time_ns, now)
        except Exception as error:
            self.invalidate(error)
            raise

    def finish(self, response):
        """核对物理计数器和自然结算状态，返回评估汇总。

        结算超时标记为 incomplete；只有完整结算且账本一致才提供正式分数。
        """
        self.ingest(response)
        try:
            state = response["data"]["settlement"]
            if (not self.closed or state is None or state["window_end_ns"] != self.end_ns
                    or state["status"] not in ("complete", "timed_out")):
                raise ValueError("physical settlement has not reached a terminal boundary")
            pending = any(s["ended_at_ns"] is None for s in self.shots.values())
            own = next(r for r in response["data"]["evaluation"]["robots"] if r["robot_id"] == self.robot_id)
            launches = [s for s in self.shots.values() if s["robot_id"] == self.robot_id]
            if own["actual_shots"] != len(launches) or own["damage_dealt"] != sum(s["damage"] for s in launches):
                raise ValueError("physical counters disagree with the event ledger")
            if state["status"] == "complete":
                if pending or any(state["remaining"].values()):
                    raise ValueError("physical completion is missing projectile terminal events")
                self.status = "complete"
            else:
                self.status = "incomplete"
            return self.summary()
        except Exception as error:
            self.invalidate(error)
            raise

    def summary(self):
        """返回当前账本摘要；尚未完整结算时 official_damage 为 None。"""
        shots = list(self.shots.values())
        eligible = [s for s in shots if s["classification"] == "eligible"]
        damage = sum(s["damage"] for s in eligible)
        excluded = [s for s in shots if s["robot_id"] == self.robot_id and s not in eligible]
        return dict(score_version=1, interval="[start_ns,end_ns)", round_id=self.round_id,
                    start_ns=self.start_ns, end_ns=self.end_ns, status=self.status, error=self.error,
                    official_damage=damage if self.status == "complete" else None,
                    eligible_damage_observed=damage, eligible_shots=len(eligible),
                    damaging_projectiles=sum(s["damage"] > 0 for s in eligible),
                    eligible_unresolved=sum(s["ended_at_ns"] is None for s in eligible),
                    excluded_own_shots=len(excluded), excluded_own_damage=sum(s["damage"] for s in excluded),
                    own_physical_damage=sum(s["damage"] for s in shots if s["robot_id"] == self.robot_id),
                    event_count=len(self.events), closed=self.closed,
                    projectiles=sorted(shots, key=lambda s:s["projectile_id"]))
