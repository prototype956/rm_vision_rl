"""训练、Gym 和评估共用的离散策略生命周期与射击机会约束。"""
import numpy as np

from src.policy.decision_clock import DecisionClock
from src.policy.observations import TensorPolicy


class DecisionPolicy(TensorPolicy):
    """每周期编码一次估计观测；时钟仅由环境在完成物理步后推进。

    子类负责动作空间到桥接九动作的映射。目标代次重置只清除历史和掩码，
    不重新采样射击时钟；start_episode 仅在完成回合预热后调用。
    """

    action_count = 9
    bridge_mode = "nine"
    fire_actions = (2, 4, 6, 8)

    def __init__(self, actor=None, *, decision_clock=None, version=2):
        self.decision_actor = actor
        self.clock = DecisionClock(decision_clock)
        super().__init__(self._act, version=version)

    def _clear_actions(self):
        self.mask = np.zeros(self.action_count, dtype=bool)
        self.mask[0] = True
        self.physical_fire_action = None
        self.track_action, self.fire_action = 0, None

    def reset(self):
        super().reset()
        self._clear_actions()

    def begin_step(self):
        super().begin_step()
        self._clear_actions()

    def observation(self):
        result = dict(features=np.array(self.rows, dtype=np.float32),
                      valid=np.array(self.valid, dtype=np.int8),
                      action_mask=self.mask.astype(np.int8, copy=True))
        if self.clock.enabled:
            result['decision_clock'] = self.clock.observation()
        return result

    def requests_fire(self, action):
        return action in self.fire_actions

    def resolve_action(self, action):
        """返回实际策略动作和九动作指令；非法射击仅可降级为同板跟踪或等待。"""
        executed = int(action)
        if not self.mask[executed]:
            track = executed - 1 if self.requests_fire(executed) else 0
            executed = track if self.mask[track] else 0
        return executed, executed

    def _update_mask(self, wire_mask):
        self.mask = np.array(wire_mask, dtype=bool)
        self.physical_fire_action = next((i for i in self.fire_actions if self.mask[i]), None)
        if not self.clock.due:
            self.mask[list(self.fire_actions)] = False

    def _act(self, encoded):
        self._update_mask(encoded['action_mask'])
        obs = self.observation()
        if self.decision_actor is None:
            return obs
        action = self.decision_actor(obs)
        if (isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer))
                or not 0 <= action < self.action_count or not self.mask[action]):
            raise ValueError('evaluation policy returned an invalid or masked action')
        return self.resolve_action(action)[1]


class JointFirePolicy(DecisionPolicy):
    """四槽位直接选择与单发请求；不调用规则选板器。"""


def make_policy(action_mode='fire_only', actor=None, *, decision_clock=None):
    if action_mode == 'joint':
        return JointFirePolicy(actor, decision_clock=decision_clock)
    if action_mode == 'fire_only':
        from src.policy.static_fire import StaticFirePolicy
        return StaticFirePolicy(actor, decision_clock=decision_clock)
    raise ValueError('action_mode must be fire_only or joint')
