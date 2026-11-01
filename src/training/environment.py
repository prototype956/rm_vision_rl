"""按任务选择固定场景或随机场景序列，独立于 SB3 的模型种子。"""
import copy
import json
import math
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.static_fire import StaticFireEnv
from src.environment.rotation_fire import RotationFireEnv
from src.environment.multi_robot_fire import MultiRobotFireEnv
from src.environment.warmup import TrackingInitializationTimeout
from src.training.config import ENV_PATHS
from src.training.paths import reward_log_path
from src.training.selection_reward import selection_costs


def switch_cost(info, config):
    """返回正的惩罚及原因；pi/4 为固定软阈值，不限制切板动作或射击时钟。"""
    if config is None:
        return 0.0, "disabled"
    if not info.get("slot_switched", False):
        return 0.0, "not_switched"
    context = info.get("switch_reward_context")
    if context is None:
        return 0.0, "no_policy_observation"
    if context["previous_slot"] < 0:
        return 0.0, "first_selection"
    if context["track_generation"] != info.get("track_generation"):
        return 0.0, "generation_changed"
    if not context["previous_valid"]:
        return 0.0, "previous_invalid"
    omega, age = context["yaw_velocity_rad_s"], context["selected_slot_age_s"]
    if not math.isfinite(omega) or not math.isfinite(age) or age < 0:
        raise ValueError("invalid pre-action switch reward context")
    progress = min(1.0, max(0.0, abs(omega) * age / (math.pi / 4)))
    return config["c0"] + config["c1"] * (1.0 - progress) ** 2, "charged"


class TrainingReward(gym.RewardWrapper):
    """在 Monitor 和价值自举之前计算训练奖励；物理伤害和独立评估不变。

    命中奖励先乘 scale，再减切板成本。省略 penalty 时兼容旧检查点。
    逐步日志与回合统计保留分项，避免净奖励掩盖真实伤害变化。
    """

    def __init__(self, env, scale, penalty=None, reward_log=None, selection_reward=None, thermal_reward=None):
        super().__init__(env)
        self.thermal_reward = copy.deepcopy(thermal_reward)
        self._selection_total = 0.0
        self._thermal_previous = None
        self._thermal_totals = {}
        self.scale = scale
        self.penalty = copy.deepcopy(penalty)
        self.selection_reward = copy.deepcopy(selection_reward)
        self.reward_log = Path(reward_log) if reward_log else None
        self._reward_file = None
        self._hit_total = self._penalty_total = 0.0
        self._components = dict(base_switch_penalty=0.0, return_penalty=0.0, facing_penalty=0.0)

    def reward(self, reward):
        return float(reward) * self.scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._selection_total = 0.0
        self._thermal_previous = copy.deepcopy(info.get("physical_thermal"))
        self._thermal_baseline = copy.deepcopy(self._thermal_previous)
        self._thermal_totals = dict(ordinary_heat_penalty=0., round_heat_penalty=0., locked_time_penalty=0.)
        if self.thermal_reward is not None and self._thermal_previous is None:
            raise ValueError("simulator does not expose physical thermal counters")
        self._hit_total = self._penalty_total = 0.0
        self._components = dict(base_switch_penalty=0.0, return_penalty=0.0, facing_penalty=0.0)
        return obs, dict(info, episode_hit_reward=0.0, episode_switch_penalty=0.0,
                         episode_training_reward=0.0, episode_reward_penalty=0.0,
                         **{"episode_" + k: v for k, v in self._components.items()})

    def step(self, action):
        obs, damage, terminated, truncated, info = self.env.step(action)
        hit = self.reward(damage)
        penalty, reason = switch_cost(info, self.penalty)
        parts = dict(base_switch_penalty=0.0, return_penalty=0.0, facing_penalty=0.0)
        if self.selection_reward is not None:
            parts, details = selection_costs(info, self.selection_reward)
            info = dict(info, **details)
            penalty = sum(parts.values())
            reason = "selection_reward"
        for key, value in parts.items():
            self._components[key] += value
        selection_penalty = penalty
        self._selection_total += selection_penalty
        thermal = dict(ordinary_heat_penalty=0., round_heat_penalty=0., locked_time_penalty=0.)
        if self.thermal_reward is not None:
            from src.training.thermal_reward import costs
            thermal = costs(self._thermal_previous, info["physical_thermal"], self.thermal_reward)
            self._thermal_previous = copy.deepcopy(info["physical_thermal"])
        for key, value in thermal.items():
            self._thermal_totals[key] += value
        info = dict(info, **thermal, **{"episode_"+k:v for k,v in self._thermal_totals.items()})
        if self.thermal_reward is not None:
            info["episode_ordinary_heat_locks"] = self._thermal_previous["ordinary_lock_count"]-self._thermal_baseline["ordinary_lock_count"]
            info["episode_round_heat_locks"] = self._thermal_previous["round_lock_count"]-self._thermal_baseline["round_lock_count"]
            info["episode_heat_locked_s"] = self._thermal_previous["locked_s"]-self._thermal_baseline["locked_s"]
        penalty += sum(thermal.values())
        reward = hit - penalty
        self._hit_total += hit
        self._penalty_total += penalty
        info = dict(info, **parts, reward_penalty=penalty,
                    **{"episode_" + k: v for k, v in self._components.items()},
                    episode_reward_penalty=self._penalty_total,
                    hit_reward=hit, switch_penalty=selection_penalty,
                    switch_penalty_reason=reason, training_reward=reward,
                    episode_hit_reward=self._hit_total, episode_switch_penalty=self._selection_total,
                    episode_training_reward=self._hit_total-self._penalty_total)
        if self.reward_log is not None and (self.penalty is not None or self.selection_reward is not None):
            if self._reward_file is None:
                self.reward_log.parent.mkdir(parents=True, exist_ok=True)
                self._reward_file = self.reward_log.open("w")
            fields = ("episode_steps", "physical_time_ns", "action", "executed_action",
                      "selected_slot", "slot_switched", "track_generation", "switch_reward_context",
                      "hit_reward", "switch_penalty", "switch_penalty_reason", "training_reward",
                      "base_switch_penalty", "return_penalty", "facing_penalty",
                      "facing_active", "facing_best", "facing_selected", "facing_gap")
            row = {key: info.get(key) for key in fields}
            row["episode_index"] = info.get("scene", {}).get("episode_index")
            self._reward_file.write(json.dumps(row, allow_nan=False) + "\n")
            self._reward_file.flush()
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            if self._reward_file is not None:
                self._reward_file.close()
                self._reward_file = None
        finally:
            self.env.close()


class FixedScenario(gym.Wrapper):
    """使用独立于模型的场景种子，每次重置重复相同场景。

    测量噪声也会重复，适合验证基础学习流程，不能据此判断泛化能力。
    底层 Gym 环境仍保留原有的 seed 接口。
    """

    def __init__(self, env, scene_seed, scenario):
        super().__init__(env)
        self.scene_seed = scene_seed
        self.scenario = copy.deepcopy(scenario)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("training reset options are fixed by its saved configuration")
        return self.env.reset(seed=self.scene_seed, options={"scenario": self.scenario})

    def action_masks(self):
        return self.env.action_masks()

    def training_settings(self):
        """返回可序列化的环境契约，不启动仿真，供父进程记录检查点元数据。"""
        base = self.unwrapped
        settings = {name: str(getattr(base, name)) for name in ENV_PATHS}
        settings.update(episode_steps=base.episode_steps, scene_seed=self.scene_seed,
                        scenario=base._scenario({"scenario": self.scenario}))
        if isinstance(base, RotationFireEnv):
            settings["angular_speed_range_rad_s"] = list(base.angular_speed_range_rad_s)
        return settings


class RandomScenario(FixedScenario):
    """首次以 scene_seed 初始化，后续 reset 连续采样；忽略模型传入的 seed。

    恢复训练创建新实例，因此从保存种子的首个场景重新开始。
    """

    def __init__(self, env, scene_seed, scenario):
        super().__init__(env, scene_seed, scenario)
        self._started = False

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("training reset options are fixed by its saved configuration")
        result = self.env.reset(seed=None if self._started else self.scene_seed,
                                options={"scenario": self.scenario})
        self._started = True
        return result


class CurriculumScenario(FixedScenario):
    """Only the parent scheduler supplies curriculum descriptors at reset boundaries."""
    def reset(self, *, seed=None, options=None):
        from src.training.curriculum import descriptor_scene
        if not options or set(options) != {"curriculum_scene"}:
            raise ValueError("curriculum reset requires a parent scene descriptor")
        descriptor = copy.deepcopy(options["curriculum_scene"])
        if type(descriptor.get("version")) is not int or descriptor["version"] not in (2, 3):
            raise ValueError("only curriculum v2/v3 scene descriptors are supported")
        rejections = []
        attempts = 128 if descriptor["version"] == 3 and (descriptor["course"] in (7,8) or descriptor.get("release_revision",1) >= 2) else 32
        for attempt in range(1 if descriptor.get("admitted") or getattr(self.unwrapped, "replay_recording", None) else attempts):
            candidate = copy.deepcopy(descriptor) if descriptor.get("admitted") or getattr(self.unwrapped, "replay_recording", None) else descriptor_scene(descriptor, self.scenario["duel"]["controlled"]["preset"], attempt)
            if candidate.get("release_revision",1) >= 3 and candidate["scenario"].get("visibility_admission") != "full_path_clear":
                raise ValueError("revision 3 scene must explicitly request full_path_clear")
            try:
                from src.policy.decision_clock import DecisionClock
                clock = copy.deepcopy(self.unwrapped._history.clock.config)
                if clock is not None:
                    clock["seed"] = (descriptor["seed"] ^ 0x53484F54) % (2**32)
                    self.unwrapped._history.clock = DecisionClock(clock)
                obs, info = self.env.reset(seed=descriptor["seed"], options={"scenario":candidate["scenario"]})
                from src.training.scene_admission import check_fireability, check_trackability
                probe = self.unwrapped._client.inspect()["data"]["evaluation"]["fireability"]
                check_fireability(probe)
                trackability = check_trackability(candidate)
                info = dict(info, fireability=probe, trackability=trackability)
                if candidate.get("release_revision",1) >= 3:
                    from src.training.curriculum_v2 import digest
                    from src.training.models import _file_digest
                    proof=copy.deepcopy(self.unwrapped._response["data"]["evaluation"]["scenario"].get("visibility_admission"))
                    if not proof or proof.get("policy") != "full_path_clear" or proof.get("duration_ms") != 45000 or proof.get("min_visible_armor_count",0) < 1:
                        raise RuntimeError("invalid explicit duel: missing full_path_clear certificate")
                    if not hasattr(self,"_visibility_binary_hash"):
                        self._visibility_binary_hash=_file_digest(self.unwrapped.simulator_binary)
                        self._visibility_config_hash=_file_digest(self.unwrapped.simulator_config)
                    certificate=dict(proof=proof,scenario_sha256=digest(candidate["scenario"]),native_sha256=self._visibility_binary_hash,
                        simulator_config_sha256=self._visibility_config_hash)
                    candidate["visibility_certificate"]=certificate
                    info["visibility_admission"]=certificate
                directory = getattr(self.unwrapped, "log_dir", None)
                if directory is not None:
                    from tools.training.view import _atomic_json
                    _atomic_json(Path(directory) / "last-admitted-scene.json", candidate)
                self.descriptor = candidate
                self.admission_rejections = rejections
                return obs, dict(info, curriculum_scene=candidate, admission_rejections=copy.deepcopy(rejections))
            except (RuntimeError, TrackingInitializationTimeout) as error:
                # reset includes visual warmup, before policy takeover. An actual
                # collision here invalidates this placement just like a path
                # rejection. Frozen cases must fail visibly, and step() errors
                # must never be resampled or converted to successful episodes.
                tracking_timeout = isinstance(error, TrackingInitializationTimeout)
                if tracking_timeout and descriptor.get("release_revision",1)>=3:
                    # Geometry has its own controller-independent certificate.
                    # A clear case with failed estimator warmup is a real
                    # evaluation failure, not permission to select an easier pose.
                    raise
                if descriptor.get("admitted") or not (tracking_timeout or str(error).startswith(("invalid seed ", "invalid explicit duel", "runtime collision:"))):
                    raise
                rejection = dict(attempt=attempt, seed=descriptor["seed"],
                                 course=descriptor["course"], case=descriptor["case"],
                                 scenario=candidate["scenario"], error=str(error),
                                 kind="tracking_initialization_timeout" if tracking_timeout else "invalid_placement")
                if tracking_timeout:
                    rejection["warmup"] = error.info
                    rejection["phase"] = error.phase
                rejections.append(rejection)
                # Retain failed admissions even if all retries fail or the worker
                # is interrupted. These are never counted as policy episodes.
                directory = getattr(self.unwrapped, "log_dir", None)
                if directory is not None:
                    path = Path(directory) / "admission-rejections.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(rejection, allow_nan=False) + "\n")
                progress = getattr(self.unwrapped, "_admission_progress", None)
                if progress is not None:
                    progress({k: rejection[k] for k in ("attempt", "seed", "kind")})
        raise RuntimeError(f"no legal curriculum placement: {descriptor}; rejections={rejections}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, dict(info, curriculum_scene=self.descriptor,
            admission_rejections=copy.deepcopy(self.admission_rejections))


def make_environment(config, log_dir, monitor_file=None, *, controller="policy"):
    """创建尚未启动的任务环境，每个实例独占一对仿真与视觉桥接进程。"""
    settings = config["environment"]
    kwargs = {key: settings[key] for key in ENV_PATHS if key in settings}
    kwargs["observation_mode"] = settings.get("observation_mode", "legacy")
    kwargs["action_mode"] = settings.get("action_mode", "fire_only")
    kwargs["selection_memory"] = settings.get("selection_memory", False)
    if "decision_clock" in settings:
        kwargs["decision_clock"] = settings["decision_clock"]
    rotation = settings.get("task", "static_fire") == "random_rotation_fire"
    if rotation:
        kwargs["angular_speed_range_rad_s"] = settings.get("angular_speed_range_rad_s", [1, 7])
    multi = settings.get("task") == "multi_robot_fire"
    base_class = MultiRobotFireEnv if multi else RotationFireEnv if rotation else StaticFireEnv
    if controller == "traditional":
        if not multi:
            raise ValueError("traditional evaluation requires multi_robot_fire")
        from src.environment.rule_fire import RuleEnv
        from src.training.config import ROOT
        base_class = RuleEnv
        kwargs["bridge_binary"] = ROOT / "artifacts/build/vision-bridge/rmvision-rule-bridge"
    elif controller != "policy":
        raise ValueError("unknown controller")
    wrapper = CurriculumScenario if "curriculum" in config else RandomScenario if rotation or multi else FixedScenario
    base = base_class(episode_steps=settings["episode_steps"], log_dir=log_dir, **kwargs)
    if config.get("runtime", {}).get("bounded_logging"):
        base.detailed_logging = False
    if "curriculum" in config:
        # The curriculum owns the 32 pose retries. Repeating native seeds cannot
        # repair the same explicit invalid pose and would hide its rejection.
        base._spawn_attempts = 1
    env = wrapper(base, settings["scene_seed"], settings["scenario"])
    reward_log = str(reward_log_path(monitor_file)) if monitor_file else None
    if config.get("runtime", {}).get("bounded_logging"):
        reward_log = None
    if reward_log:
        Path(reward_log).parent.mkdir(parents=True, exist_ok=True)
        Path(monitor_file).parent.mkdir(parents=True, exist_ok=True)
    env = TrainingReward(env, config.get("reward_scale", 1.0),
                         config.get("switch_penalty"), reward_log, config.get("selection_reward"), config.get("thermal_reward"))
    return Monitor(env, filename=str(monitor_file) if monitor_file else None,
                   info_keywords=("episode_damage", "actual_shots", "end_reason", "slot_switches",
                                  "episode_hit_reward", "episode_switch_penalty", "episode_training_reward",
                                  "episode_base_switch_penalty", "episode_return_penalty",
                                  "episode_facing_penalty", "episode_reward_penalty"))


def vectorize(env):
    # DummyVecEnv 在自动重置前提供 terminal_observation 和 TimeLimit.truncated。
    # 时间截断的价值自举由 MaskablePPO 执行，此处不能重复追加奖励。
    return DummyVecEnv([lambda: env])
