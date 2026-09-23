"""在禁射状态下建立稳定跟踪，并连续切换到正式运行阶段。

生命周期元数据仅供会话管理使用，不进入策略观测或动作掩码。
就绪判断仅使用检测、估计结果和自身遥测。
"""
from dataclasses import asdict, dataclass


STEP_NS = 10_000_000
EPOCH_NS = 10**18


@dataclass(frozen=True)
class WarmupConfig:
    confirmation_frames: int = 5
    max_frame_age_ms: int = 100
    timeout_ms: int = 15_000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int:
                raise ValueError(key + " must be an integer")
        if not 1 <= self.confirmation_frames <= 30:
            raise ValueError("confirmation_frames must be in [1, 30]")
        if not 10 <= self.max_frame_age_ms <= 500:
            raise ValueError("max_frame_age_ms must be in [10, 500]")
        if not 10 <= self.timeout_ms <= 120_000 or self.timeout_ms % 10:
            raise ValueError("timeout_ms must be a multiple of 10 in [10, 120000]")


class TrackingConfirmation:
    """按连续且新鲜的不同图像帧确认跟踪，不按 100 Hz 控制调用次数计数。"""

    def __init__(self, config):
        self.config = config
        self.count = 0
        self.sequence = None
        self.capture_ns = None
        self.reason = "no_frame"

    def observe(self, now, feedback, result):
        control_ok = (result["command"]["valid"] and not result["control"]["search"]
                      and result["control"]["selected_slot"] >= 0)
        feedback_ok = (feedback["valid"] and
                       0 <= EPOCH_NS + now - feedback["timestamp_ns"] <= STEP_NS)
        max_age = self.config.max_frame_age_ms * 1_000_000
        for frame in result["frames"]:
            sequence, capture = frame["sequence"], frame["capture_time_ns"]
            if self.sequence is not None and sequence <= self.sequence:
                raise ValueError("duplicate/out-of-order confirmation frame")
            if self.sequence is not None and sequence != self.sequence + 1:
                self.count = 0
            self.sequence, self.capture_ns = sequence, capture
            good = (frame["tracker_state"] == "tracking" and frame["pnp"]
                    and 0 <= now - capture <= max_age and control_ok and feedback_ok)
            self.count = self.count + 1 if good else 0
        if self.capture_ns is None:
            self.reason = "no_frame"
        elif not 0 <= now - self.capture_ns <= max_age:
            self.reason = "stale_frame"
        elif result["tracker_state"] != "tracking":
            self.reason = "not_tracking"
        elif not feedback_ok:
            self.reason = "invalid_feedback"
        elif not control_ok:
            self.reason = "no_tracking_command"
        else:
            self.reason = "confirming" if self.count < self.config.confirmation_frames else "confirmed"
            return self.reason == "confirmed"
        self.count = 0
        return False


class WarmupSession:
    """管理一对仿真与桥接从重置到预热就绪或超时的状态转换。

    就绪时保留物理时间、估计器与 MPC 历史，以及命令延迟队列。
    通过本类继续跟踪时仍保持禁射。发布结果不确定时，应先处理客户端待确认请求，
    再调用 reset() 重置两端，不能沿用部分更新的预热状态。
    """

    def __init__(self, client, bridge, config=None):
        self.client, self.bridge = client, bridge
        self.config = config or WarmupConfig()
        self.status = "idle"
        self.response = None
        self.evaluation_start_ns = None
        self.confirmation = TrackingConfirmation(self.config)

    def reset(self, seed, scenario=None):
        scenario = dict(scenario or {})
        scenario.setdefault("measurements", {})
        if scenario["measurements"] is None:
            raise ValueError("warmup requires detector measurements")
        self.status = "fault"
        self.evaluation_start_ns = None
        self.confirmation = TrackingConfirmation(self.config)
        self.response = self.client.reset(seed, scenario)
        if not self.response["data"]["capabilities"]["visual_measurements"]:
            raise RuntimeError("simulator did not enable measurements")
        self.bridge.reset(self.response["round_id"])
        self.status = "warming"
        return self.info()

    def info(self):
        now = self.response["sim_time_ns"] if self.response else 0
        return dict(status=self.status, config=asdict(self.config),
                    round_id=self.response["round_id"] if self.response else None,
                    physical_time_ns=now, confirmed_frames=self.confirmation.count,
                    last_frame_sequence=self.confirmation.sequence,
                    last_capture_time_ns=self.confirmation.capture_ns,
                    reason=self.confirmation.reason,
                    evaluation_start_ns=self.evaluation_start_ns,
                    evaluation_time_ns=(now - self.evaluation_start_ns
                                        if self.evaluation_start_ns is not None else None))

    def _publish(self, result):
        if result["command"]["fire"]:
            raise RuntimeError("the warmup bridge must supply an acknowledged disabled-fire command")
        self.response = self.client.advance(**result["command"])
        self.bridge.ack(True)

    def advance(self):
        if self.status != "warming":
            raise RuntimeError("advance requires an active warmup")
        try:
            now = self.response["sim_time_ns"]
            result = self.bridge.step(self.response)
            ready = self.confirmation.observe(now, self.response["data"]["feedback"], result)
            self._publish(result)
            # 只有最后一次禁射命令发布得到确认后，就绪状态才生效。
            # 恰好到达超时边界时，已完成确认的就绪结果优先。
            if ready:
                self.status = "ready"
                self.evaluation_start_ns = self.response["sim_time_ns"]
            elif self.response["sim_time_ns"] >= self.config.timeout_ms * 1_000_000:
                self.status = "timed_out"
            return dict(warmup=self.info(), vision=result)
        except Exception:
            self.status = "fault"
            self.evaluation_start_ns = None
            raise

    def advance_tracking(self):
        """从就绪状态继续禁射跟踪，以预热完成时刻为计时起点，保留估计历史。"""
        if self.status != "ready":
            raise RuntimeError("tracking evaluation requires a ready warmup")
        try:
            result = self.bridge.step(self.response)
            self._publish(result)
            # 正式阶段丢失目标属于评估结果，不能重新预热或重置计时。
            return dict(warmup=self.info(), vision=result)
        except Exception:
            self.status = "fault"
            self.evaluation_start_ns = None
            raise
