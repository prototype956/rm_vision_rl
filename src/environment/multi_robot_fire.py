"""联盟赛一对一多车型火控任务；车型和运动真值只供场景生成与评估。"""
import copy

import numpy as np

from src.environment.fire import FireEnv, simulator_root_path

PROFILES = ("standard", "wheel_leg", "low", "folding")
PRESETS = ("infantry_health_cooling", "infantry_health_burst",
           "infantry_power_cooling", "infantry_power_burst", "sentry")
KINDS = ("static", "rotation", "translation", "combined", "lift", "climb", "tunnel", "jump")


def multi_robot_scenario(options):
    """保留未采样模板；新任务禁止高血量覆盖与无限资源。"""
    options = {} if options is None else copy.deepcopy(options)
    if not isinstance(options, dict) or set(options) - {"scenario"}:
        raise ValueError("reset options only supports scenario")
    scenario = options.get("scenario", {})
    if not isinstance(scenario, dict) or set(scenario) - {"duel", "measurements", "unlimited_heat"}:
        raise ValueError("multi_robot_fire supports duel/measurements and real resources only")
    if scenario.get("unlimited_heat", False) is not False:
        raise ValueError("multi_robot_fire requires normal heat")
    scenario["unlimited_heat"] = False
    duel = scenario.setdefault("duel", {})
    if not isinstance(duel, dict) or set(duel) - {"controlled", "target"}:
        raise ValueError("duel supports controlled and target")
    for name in ("controlled", "target"):
        robot = duel.setdefault(name, {})
        if (not isinstance(robot, dict) or set(robot) -
                {"profile", "preset", "motion", "position_xz_m", "yaw_rad", "surface"}):
            raise ValueError("invalid duel robot fields")
        allowed = PROFILES if name == "controlled" else (*PROFILES, "heavy_target")
        if "profile" in robot and robot["profile"] not in allowed:
            raise ValueError("invalid profile; heavy_target is enemy-only")
        if "preset" in robot and robot["preset"] not in PRESETS:
            raise ValueError("invalid robot resource preset")
        if robot.get("profile") == "heavy_target" and "preset" in robot:
            raise ValueError("heavy_target uses its own 450HP preset")
        if "motion" in robot and (not isinstance(robot["motion"], dict) or
                                  robot["motion"].get("kind", "static") not in KINDS):
            raise ValueError("invalid motion kind")
    measurements = scenario.setdefault("measurements", {})
    if not isinstance(measurements, dict):
        raise ValueError("measurements must be an object")
    for key, value in {"noise_std_px": 0.25, "latency_ms": 20,
                       "dropout_probability": 0.0}.items():
        measurements.setdefault(key, value)
    return scenario


def _kinds(profile):
    result = list(KINDS[:4])
    if profile == "wheel_leg":
        result.extend(("lift", "climb"))
    if profile in ("low", "folding"):
        result.append("tunnel")
    return result


class MultiRobotFireEnv(FireEnv):
    """每回合独立采样车型、角色和合法运动；观测及九动作接口保持原样。"""

    finite_episode = True

    def __init__(self, **kwargs):
        kwargs.setdefault("episode_steps", 3000)
        kwargs.setdefault("simulator_config", simulator_root_path(kwargs.get("simulator_root")) /
                          "configs/rmul_2027_multirobot_infantry.toml")
        if kwargs["episode_steps"] > 3000 or (kwargs.get("warmup_config") is not None and kwargs["warmup_config"].timeout_ms > 15000):
            raise ValueError("multi_robot_fire validates up to 30s episodes plus 15s warmup")
        super().__init__(**kwargs)

    @staticmethod
    def _scenario(options):
        return multi_robot_scenario(options)

    def _sample_scene(self, scenario):
        episode_seed = int(self.np_random.integers(0, 2**63))
        kind_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 2]))
        motion_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 3]))
        spawn_rng = np.random.default_rng(np.random.SeedSequence([episode_seed, 4]))
        own, enemy = scenario["duel"]["controlled"], scenario["duel"]["target"]
        own.setdefault("profile", PROFILES[int(kind_rng.integers(4))])
        own.setdefault("preset", "infantry_health_cooling")
        if "profile" not in enemy:
            # Eight infantry/sentry variants and one heavy proxy have equal probability.
            count = 4 if "preset" in enemy else 9
            index = int(kind_rng.integers(count))
            enemy["profile"] = "heavy_target" if index == 8 else PROFILES[index % 4]
            if index != 8:
                enemy.setdefault("preset", "sentry" if index >= 4 else "infantry_health_cooling")
        elif enemy["profile"] != "heavy_target":
            enemy.setdefault("preset", "sentry" if kind_rng.integers(2) else "infantry_health_cooling")
        explicit = ["motion" in own, "motion" in enemy]
        for index, robot in enumerate((own, enemy)):
            if "motion" not in robot:
                choices = _kinds(robot["profile"])
                other = (enemy, own)[index]
                if explicit[1-index] and other["motion"].get("kind", "static") not in ("static", "rotation", "lift"):
                    choices = [k for k in choices if k not in ("climb", "tunnel", "jump")]
                robot["motion"] = {"kind": choices[int(motion_rng.integers(len(choices)))]}
            motion = robot["motion"]
            motion.setdefault("amplitude_m", float(motion_rng.uniform(0.2, 0.5)))
            motion.setdefault("speed_m_s", float(motion_rng.uniform(0.2, 0.7)))
            motion.setdefault("angular_speed_rad_s", float(motion_rng.uniform(0.5, 3.0)) *
                              (1 if motion_rng.integers(2) else -1))
        # Complex templates use a validated observer anchor. Do not randomly give that
        # observer a translation that leaves the boundary or another complex template.
        special = [r["motion"]["kind"] in ("climb", "tunnel", "jump") for r in (own, enemy)]
        if any(special):
            chosen = 0 if special[0] else 1
            other = 1 - chosen
            robots = (own, enemy)
            if special[other] and explicit[other] and explicit[chosen]:
                # Both explicitly requested: simulator validates the complete joint path.
                pass
            elif not explicit[other]:
                robots[other]["motion"]["kind"] = "static"
        return scenario, spawn_rng, {"episode_seed": episode_seed,
                                     "sampling_revision": 1, "task": "multi_robot_fire"}
