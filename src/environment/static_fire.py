"""保留原静止靶场景及 Gym 接口。"""
import copy

from src.environment.fire import FireEnv


class StaticFireEnv(FireEnv):
    """静止靶任务；省略目标距离时保持旧的 4 米默认值。"""

    @staticmethod
    def _scenario(options):
        options = {} if options is None else copy.deepcopy(options)
        if not isinstance(options, dict) or set(options) - {"scenario"}:
            raise ValueError("reset options only supports 'scenario'")
        scenario = options.get("scenario", {})
        if not isinstance(scenario, dict):
            raise ValueError("scenario must be a dictionary")
        scenario.setdefault("motion", {"kind": "static"})
        if scenario["motion"] != {"kind": "static"}:
            raise ValueError("StaticFireEnv requires static motion")
        scenario.setdefault("target_distance_m", 4.0)
        scenario.setdefault("target_hp", 100000)
        if "unlimited_heat" in scenario and type(scenario["unlimited_heat"]) is not bool:
            raise ValueError("scenario.unlimited_heat must be a boolean")
        measurements = scenario.setdefault("measurements", {})
        if not isinstance(measurements, dict):
            raise ValueError("static target training requires detector measurements")
        for key, value in {"noise_std_px": 0.25, "latency_ms": 20,
                           "dropout_probability": 0.0}.items():
            measurements.setdefault(key, value)
        return scenario
