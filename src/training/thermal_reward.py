"""Physical thermal event deltas, independent of sampled Actor telemetry."""
import math


DEFAULT = dict(ordinary_onset=2.0, round_onset=10.0, locked_per_s=0.2)


def validate(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULT):
        raise ValueError("thermal_reward requires ordinary_onset, round_onset, locked_per_s")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in value.values()):
        raise ValueError("thermal reward coefficients must be finite and nonnegative")
    return dict(value)


def costs(previous, current, config):
    """Counters start at policy takeover. Co-triggered onsets charge only round cost."""
    fields = ("ordinary_lock_count", "round_lock_count", "joint_lock_count", "locked_s")
    if previous is None or any(k not in current or k not in previous for k in fields):
        raise ValueError("thermal reward requires complete physical counters")
    delta = {k: current[k] - previous[k] for k in fields}
    if any(not math.isfinite(v) or v < -1e-9 for v in delta.values()):
        raise ValueError("thermal counters must be finite and monotonic within an episode")
    ordinary = delta["ordinary_lock_count"] - delta["joint_lock_count"]
    if ordinary < 0 or delta["joint_lock_count"] > delta["round_lock_count"]:
        raise ValueError("invalid co-triggered heat lock counters")
    return dict(ordinary_heat_penalty=config["ordinary_onset"] * ordinary,
                round_heat_penalty=config["round_onset"] * delta["round_lock_count"],
                locked_time_penalty=config["locked_per_s"] * max(0., delta["locked_s"]))
