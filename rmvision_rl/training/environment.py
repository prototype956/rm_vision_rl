"""Fixed reset configuration for training, including SB3 automatic episode resets."""
import copy

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rmvision_rl.environment.static_fire import StaticFireEnv
from rmvision_rl.training.config import ENV_PATHS


class FixedScenario(gym.Wrapper):
    """Scene randomness is independent of the learner seed; every reset repeats this scene.

    This deliberately repeats measurement noise too. It is a simple learning fixture,
    not a claim of generalization. The base Gym environment keeps its original seed API.
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
    """Create an unstarted fixed-scene env; each instance owns its simulator pair."""
    settings = config["environment"]
    kwargs = {key: settings[key] for key in ENV_PATHS if key in settings}
    base = StaticFireEnv(episode_steps=settings["episode_steps"], log_dir=log_dir, **kwargs)
    env = FixedScenario(base, settings["scene_seed"], settings["scenario"])
    return Monitor(env, filename=str(monitor_file) if monitor_file else None,
                   info_keywords=("episode_damage", "actual_shots", "end_reason"))


def vectorize(env):
    # DummyVecEnv supplies terminal_observation and TimeLimit.truncated before auto-reset.
    # MaskablePPO performs the timeout bootstrap; this layer must not add another reward.
    return DummyVecEnv([lambda: env])
