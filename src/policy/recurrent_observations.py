"""Single-step v5 observation adapter; recurrent memory belongs to the policy runner.

All sensor features reuse the v4 whitelist. Identity numbers only produce reset
events; they are never neural features. candidate_map maps new slots to old slots.
"""
from collections import deque
import copy
import numpy as np

from src.policy.asymmetric_observations import SCHEMA as V4_SCHEMA, NAMES
from src.policy.decision import DecisionPolicy

MODE = "asymmetric_gru_v1"
SCHEMA = dict(copy.deepcopy(V4_SCHEMA), version=5, name=MODE, history=1,
              dt_s="elapsed physical/monotonic seconds; first step zero; network dt/(0.1+dt)")
ACTOR_KEYS = ("features", "valid", "action_mask", "selection_memory", "decision_clock",
              "dt_s", "episode_start", "target_reset", "candidate_map")


class RecurrentObservationPolicy(DecisionPolicy):
    """Reuse selection/mask semantics, exposing only the newest encoded row."""

    def __init__(self, actor=None, *, decision_clock=None, selection_memory=True):
        super().__init__(actor, decision_clock=decision_clock, version=4,
                         selection_memory=selection_memory)

    def reset(self):
        super().reset()
        self.rows = deque([[0.0] * self.feature_count], maxlen=1)
        self.valid = deque([False], maxlen=1)
        self.last_now_ns = None
        self.last_present = False
        self.last_reset_count = None
        self.elapsed = 0.0
        self.episode_reset = True
        self.reset_target = True
        self.slot_map = np.arange(4, dtype=np.int64)

    def begin_step(self):
        super().begin_step()
        self.reset_target = False
        self.slot_map = np.arange(4, dtype=np.int64)

    def set_generation(self, generation):
        first = self.generation is None
        super().set_generation(generation)
        self.reset_target |= first or self.generation_changed

    def set_tracking_metadata(self, metadata):
        if "track_generation" in metadata:
            self.set_generation(metadata["track_generation"])
        count = metadata.get("track_reset_count")
        if count is not None:
            self.reset_target |= self.last_reset_count is not None and count != self.last_reset_count
            self.last_reset_count = count
        if "candidate_map" in metadata:
            mapping = np.asarray(metadata["candidate_map"], dtype=np.int64)
            if mapping.shape != (4,) or sorted(mapping.tolist()) != list(range(4)):
                self.reset_target = True
                self.return_slot = self._last_selected = -1
            else:
                self.slot_map = mapping
                inverse = np.argsort(mapping)
                if self.return_slot >= 0:
                    self.return_slot = int(inverse[self.return_slot])
                if self._last_selected >= 0:
                    self._last_selected = int(inverse[self._last_selected])

    def __call__(self, observation):
        now = observation["now_ns"]
        self.episode_reset = self.last_now_ns is None
        if self.last_now_ns is not None and now <= self.last_now_ns:
            raise ValueError("recurrent observation requires strictly increasing control timestamps")
        self.elapsed = 0.0 if self.episode_reset else (now-self.last_now_ns)*1e-9
        self.last_now_ns = now
        present = observation["estimate"]["state"] != "lost"
        self.reset_target |= self.episode_reset or (self.last_present and not present)
        self.last_present = present
        return super().__call__(observation)

    def observation(self):
        old = super().observation()
        return dict(features=old["features"][-1].copy(), valid=old["valid"][-1:].copy(),
                    action_mask=old["action_mask"],
                    selection_memory=old.get("selection_memory", np.array([1,0,0,0,0], dtype=np.int8)),
                    decision_clock=old.get("decision_clock", np.zeros(5, dtype=np.float32)),
                    dt_s=np.array([self.elapsed], dtype=np.float32),
                    episode_start=np.array([self.episode_reset], dtype=np.int8),
                    target_reset=np.array([self.reset_target], dtype=np.int8),
                    candidate_map=self.slot_map.copy())
