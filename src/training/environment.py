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

    def __init__(self, env, scale, penalty=None, reward_log=None, selection_reward=None):
        super().__init__(env)
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
        reward = hit - penalty
        self._hit_total += hit
        self._penalty_total += penalty
        info = dict(info, **parts, reward_penalty=penalty,
                    **{"episode_" + k: v for k, v in self._components.items()},
                    episode_reward_penalty=self._penalty_total,
                    hit_reward=hit, switch_penalty=penalty,
                    switch_penalty_reason=reason, training_reward=reward,
                    episode_hit_reward=self._hit_total, episode_switch_penalty=self._penalty_total,
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


def make_environment(config, log_dir, monitor_file=None):
    """创建尚未启动的任务环境，每个实例独占一对仿真与视觉桥接进程。"""
    settings = config["environment"]
    kwargs = {key: settings[key] for key in ENV_PATHS if key in settings}
    kwargs["action_mode"] = settings.get("action_mode", "fire_only")
    kwargs["selection_memory"] = settings.get("selection_memory", False)
    if "decision_clock" in settings:
        kwargs["decision_clock"] = settings["decision_clock"]
    rotation = settings.get("task", "static_fire") == "random_rotation_fire"
    if rotation:
        kwargs["angular_speed_range_rad_s"] = settings.get("angular_speed_range_rad_s", [1, 7])
    base_class = RotationFireEnv if rotation else StaticFireEnv
    wrapper = RandomScenario if rotation else FixedScenario
    base = base_class(episode_steps=settings["episode_steps"], log_dir=log_dir, **kwargs)
    env = wrapper(base, settings["scene_seed"], settings["scenario"])
    reward_log = str(reward_log_path(monitor_file)) if monitor_file else None
    if reward_log:
        Path(reward_log).parent.mkdir(parents=True, exist_ok=True)
        Path(monitor_file).parent.mkdir(parents=True, exist_ok=True)
    env = TrainingReward(env, config.get("reward_scale", 1.0),
                         config.get("switch_penalty"), reward_log, config.get("selection_reward"))
    return Monitor(env, filename=str(monitor_file) if monitor_file else None,
                   info_keywords=("episode_damage", "actual_shots", "end_reason", "slot_switches",
                                  "episode_hit_reward", "episode_switch_penalty", "episode_training_reward",
                                  "episode_base_switch_penalty", "episode_return_penalty",
                                  "episode_facing_penalty", "episode_reward_penalty"))


def vectorize(env):
    # DummyVecEnv 在自动重置前提供 terminal_observation 和 TimeLimit.truncated。
    # 时间截断的价值自举由 MaskablePPO 执行，此处不能重复追加奖励。
    return DummyVecEnv([lambda: env])
