"""Shared Gym/evaluation encoding for rule-selected tracking and single-shot requests."""
import numpy as np

from rmvision_rl.policy.observations import TensorPolicy


class StaticFirePolicy(TensorPolicy):
    def __init__(self, actor=None):
        self.binary_actor = actor
        super().__init__(self._act)

    def reset(self):
        super().reset()
        self.track_action, self.fire_action = 0, None

    def begin_step(self):
        super().begin_step()
        self.track_action, self.fire_action = 0, None

    def observation(self):
        return {
            "features": np.array(self.rows, dtype=np.float32),
            "valid": np.array(self.valid, dtype=np.int8),
            "action_mask": np.array([True, self.fire_action is not None], dtype=np.int8),
        }

    def _act(self, encoded):
        mask = encoded["action_mask"]
        self.track_action = next((i for i in (1, 3, 5, 7) if mask[i]), 0)
        if not mask[self.track_action]:
            raise RuntimeError("fire-only observation has no legal tracking/WAIT action")
        if self.track_action and mask[self.track_action + 1]:
            self.fire_action = self.track_action + 1
        obs = self.observation()
        if self.binary_actor is None:
            return obs
        action = self.binary_actor(obs)
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or action not in (0, 1) or not obs["action_mask"][action]):
            raise ValueError("evaluation policy returned an invalid or masked action")
        return self.fire_action if action == 1 else self.track_action
