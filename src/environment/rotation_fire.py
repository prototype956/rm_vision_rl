"""可复现的随机原地旋转靶；出生几何仍由仿真器验证。"""
import copy
import math

import numpy as np

from src.environment.fire import FireEnv


def validate_speed_range(value):
    """校验角速度大小范围，单位 rad/s；符号由采样器等概率生成。"""
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
            or not 0 < value[0] <= value[1] <= 7):
        raise ValueError("angular_speed_range_rad_s requires 0 < min <= max <= 7")
    return list(value)


def rotation_scenario(options):
    """返回未采样模板；不补齐位置或朝向，避免覆盖仿真器随机出生。"""
    options = {} if options is None else copy.deepcopy(options)
    if not isinstance(options, dict) or set(options) - {"scenario"}:
        raise ValueError("reset options only supports 'scenario'")
    scenario = options.get("scenario", {})
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be a dictionary")
    if "motion" in scenario:
        raise ValueError("random_rotation_fire samples motion; omit scenario.motion")
    scenario.setdefault("target_hp", 100000)
    if "unlimited_heat" in scenario and type(scenario["unlimited_heat"]) is not bool:
        raise ValueError("scenario.unlimited_heat must be a boolean")
    measurements = scenario.setdefault("measurements", {})
    if not isinstance(measurements, dict):
        raise ValueError("rotation target training requires detector measurements")
    for key, value in {"noise_std_px": 0.25, "latency_ms": 20,
                       "dropout_probability": 0.0}.items():
        measurements.setdefault(key, value)
    return scenario


class RotationFireEnv(FireEnv):
    """每回合独立采样匀速旋转，正角速度按 Bevy 世界 +Y 右手方向。

    Gym 随机流每回合只生成一个 episode_seed。角速度与出生重试使用该种子的
    独立子流，出生拒绝次数不会改变角速度或下一回合的采样结果。
    """

    def __init__(self, *, angular_speed_range_rad_s=(1, 7), **kwargs):
        self.angular_speed_range_rad_s = validate_speed_range(angular_speed_range_rad_s)
        super().__init__(**kwargs)

    @staticmethod
    def _scenario(options):
        return rotation_scenario(options)

    def _sample_scene(self, scenario):
        episode_seed = int(self.np_random.integers(0, 2**63))
        # 固定域编号是采样契约，后续新增随机参数时不得复用这两个流。
        motion_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 0]))
        spawn_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 1]))
        low, high = self.angular_speed_range_rad_s
        speed = float(motion_rng.uniform(low, high))
        speed *= 1 if motion_rng.integers(2) else -1
        scenario["motion"] = {"kind": "rotation", "angular_speed_rad_s": speed}
        return scenario, spawn_rng, {"episode_seed": episode_seed,
                                     "angular_speed_rad_s": speed}
