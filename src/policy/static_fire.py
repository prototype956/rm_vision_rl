"""为 Gym 和独立评估统一编码规则选板下的跟踪与单发请求。"""
import numpy as np

from src.policy.observations import TensorPolicy
from src.policy.decision_clock import DecisionClock


class StaticFirePolicy(TensorPolicy):
    """将规则选板的九动作观测适配为跟踪与单发两个动作。

    观测与动作编码由 Gym 和评估共用。只有物理掩码允许且决策时钟到期时，
    才开放单发动作；不提供 actor 时返回观测，由 Gym 调用方提交动作。
    """
    def __init__(self, actor=None, *, decision_clock=None):
        self.binary_actor = actor
        self.clock = DecisionClock(decision_clock)
        super().__init__(self._act)

    def reset(self):
        super().reset()
        self.track_action, self.fire_action = 0, None
        self.physical_fire_action = None

    def begin_step(self):
        super().begin_step()
        self.track_action, self.fire_action = 0, None
        self.physical_fire_action = None

    def observation(self):
        """返回独立 NumPy 观测：float32[8, 90] 特征、int8[8] 有效位及 int8[2] 掩码。

        启用决策时钟时额外包含 float32[5] 的 decision_clock 特征。
        """
        result = {
            "features": np.array(self.rows, dtype=np.float32),
            "valid": np.array(self.valid, dtype=np.int8),
            "action_mask": np.array([True, self.fire_action is not None], dtype=np.int8),
        }
        if self.clock.enabled:
            result["decision_clock"] = self.clock.observation()
        return result

    def _act(self, encoded):
        mask = encoded["action_mask"]
        self.track_action = next((i for i in (1, 3, 5, 7) if mask[i]), 0)
        if not mask[self.track_action]:
            raise RuntimeError("fire-only observation has no legal tracking/WAIT action")
        if self.track_action and mask[self.track_action + 1]:
            self.physical_fire_action = self.track_action + 1
            if self.clock.due:
                self.fire_action = self.physical_fire_action
        obs = self.observation()
        if self.binary_actor is None:
            return obs
        action = self.binary_actor(obs)
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or action not in (0, 1) or not obs["action_mask"][action]):
            raise ValueError("evaluation policy returned an invalid or masked action")
        return self.fire_action if action == 1 else self.track_action
