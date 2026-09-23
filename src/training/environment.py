"""固定训练场景配置，使 SB3 自动重置回合时仍使用相同场景。"""
import copy

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.static_fire import StaticFireEnv
from src.training.config import ENV_PATHS


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


def make_environment(config, log_dir, monitor_file=None):
    """创建尚未启动的固定场景环境，每个实例独占一对仿真与视觉桥接进程。"""
    settings = config["environment"]
    kwargs = {key: settings[key] for key in ENV_PATHS if key in settings}
    if "decision_clock" in settings:
        kwargs["decision_clock"] = settings["decision_clock"]
    base = StaticFireEnv(episode_steps=settings["episode_steps"], log_dir=log_dir, **kwargs)
    env = FixedScenario(base, settings["scene_seed"], settings["scenario"])
    return Monitor(env, filename=str(monitor_file) if monitor_file else None,
                   info_keywords=("episode_damage", "actual_shots", "end_reason"))


def vectorize(env):
    # DummyVecEnv 在自动重置前提供 terminal_observation 和 TimeLimit.truncated。
    # 时间截断的价值自举由 MaskablePPO 执行，此处不能重复追加奖励。
    return DummyVecEnv([lambda: env])
