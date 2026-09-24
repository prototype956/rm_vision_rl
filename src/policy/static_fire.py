"""为 Gym 和独立评估统一编码规则选板下的跟踪与单发请求。"""
from src.policy.decision import DecisionPolicy


class StaticFirePolicy(DecisionPolicy):
    """保持旧 v1 特征和两动作语义，选板由 C++ 规则适配器完成。"""

    action_count = 2
    bridge_mode = 'fire_only'
    fire_actions = (1,)

    def __init__(self, actor=None, *, decision_clock=None):
        super().__init__(actor, decision_clock=decision_clock, version=1)

    def _update_mask(self, mask):
        self.track_action = next((i for i in (1, 3, 5, 7) if mask[i]), 0)
        if not mask[self.track_action]:
            raise RuntimeError('fire-only observation has no legal tracking/WAIT action')
        if self.track_action and mask[self.track_action + 1]:
            self.physical_fire_action = self.track_action + 1
            if self.clock.due:
                self.fire_action = self.physical_fire_action
        self.mask[1] = self.fire_action is not None

    def resolve_action(self, action):
        executed = int(action) if self.mask[action] else 0
        return executed, self.fire_action if executed == 1 else self.track_action
