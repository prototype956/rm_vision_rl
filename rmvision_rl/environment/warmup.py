"""Disabled-fire warmup and a continuous handoff to tracking evaluation.

Lifecycle metadata is for the experiment owner, never a policy observation or action mask.
Only detector/estimator results and own telemetry can decide readiness.
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
    """Count distinct consecutive fresh tracking frames, not 100 Hz control ticks."""

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
    """Own the reset → warmup → ready/timed_out boundary for one simulator/bridge pair.

    A READY handoff retains physical time, estimator/MPC history and command delay queues.
    This adapter still forces fire=False during subsequent tracking evaluation.
    On an ambiguous publication failure, resolve the client's pending transport request,
    then explicitly reset both owners through reset(); never continue partial warmup state.
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
            # Readiness takes effect only after the final disabled-fire publication is acknowledged.
            # At the exact timeout boundary, an already confirmed publication wins the tie.
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
        """Continue a READY round with zeroed evaluation time, without restarting estimation."""
        if self.status != "ready":
            raise RuntimeError("tracking evaluation requires a ready warmup")
        try:
            result = self.bridge.step(self.response)
            self._publish(result)
            # Later target loss belongs to evaluation; it must not restart warmup or its clock.
            return dict(warmup=self.info(), vision=result)
        except Exception:
            self.status = "fault"
            self.evaluation_start_ns = None
            raise
