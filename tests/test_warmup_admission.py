"""Only explicit visual admission failures may resample an unfrozen training pose."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym

from src.environment.warmup import TrackingInitializationTimeout
from src.training.curriculum_v2 import scene
from src.training.environment import CurriculumScenario
from src.training.recurrent_rollout import EpisodeWorkers


class Backend(gym.Env):
    def __init__(self, errors, directory):
        self.errors = iter(errors)
        self.calls = 0
        self.log_dir = directory
        self._history = SimpleNamespace(clock=SimpleNamespace(config=None))
        self._client = Mock()
        self._client.inspect.return_value = {"data": {"evaluation": {"fireability": {}}}}
        self._admission_progress = Mock()

    def reset(self, **kwargs):
        self.calls += 1
        error = next(self.errors, None)
        if error is not None:
            raise error
        return {}, {}

    def step(self, action):
        raise TrackingInitializationTimeout({"reason": "policy_failure"})


class WarmupAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.descriptor = scene(8, 3399570263033475680, "infantry_health_cooling", 752915)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        for name in ("check_fireability", "check_trackability"):
            patcher = patch("src.training.scene_admission." + name, return_value={})
            patcher.start()
            self.addCleanup(patcher.stop)

    def timeout(self):
        return TrackingInitializationTimeout({"reason": "not_tracking", "physical_time_ns": 15000000000})

    def wrapper(self, errors):
        backend = Backend(errors, self.directory)
        return CurriculumScenario(backend, 17, self.descriptor["scenario"]), backend

    def test_random_retry_changes_only_pose_and_keeps_durable_evidence(self):
        env, backend = self.wrapper([self.timeout()])
        _, info = env.reset(options={"curriculum_scene": self.descriptor})
        admitted = info["curriculum_scene"]
        self.assertEqual(backend.calls, 2)
        self.assertEqual(admitted["attempt"], 1)
        for key in ("seed", "course", "group", "case", "strata", "advanced_factors"):
            self.assertEqual(admitted[key], self.descriptor[key])
        self.assertEqual(admitted["scenario"]["duel"]["target"]["motion"], self.descriptor["scenario"]["duel"]["target"]["motion"])
        rows = [json.loads(line) for line in (self.directory / "admission-rejections.jsonl").read_text().splitlines()]
        self.assertEqual(rows, info["admission_rejections"])
        self.assertEqual(rows[0]["kind"], "tracking_initialization_timeout")
        self.assertEqual(rows[0]["warmup"]["reason"], "not_tracking")
        backend._admission_progress.assert_called_once()
        with self.assertRaises(TrackingInitializationTimeout):
            env.step(0)
        self.assertEqual(backend.calls, 2)

    def test_frozen_case_is_not_replaced_or_retried(self):
        env, backend = self.wrapper([self.timeout()])
        with self.assertRaises(TrackingInitializationTimeout):
            env.reset(options={"curriculum_scene": dict(self.descriptor, admitted=True)})
        self.assertEqual(backend.calls, 1)
        backend._admission_progress.assert_not_called()

    def test_revision_three_does_not_select_an_easier_pose_after_warmup_failure(self):
        from src.training.curriculum_v3 import scene as clear_scene
        self.descriptor=clear_scene(8,17,"infantry_health_cooling",release_revision=3)
        for frozen in (False,True):
            env,backend=self.wrapper([self.timeout()])
            descriptor=dict(self.descriptor,admitted=True) if frozen else self.descriptor
            with self.assertRaises(TrackingInitializationTimeout):
                env.reset(options={"curriculum_scene":descriptor})
            self.assertEqual(backend.calls,1)
            backend._admission_progress.assert_not_called()

    def test_transport_and_generic_timeouts_still_fail(self):
        for error in (TimeoutError("socket read timed out"), RuntimeError("bridge disconnected")):
            env, backend = self.wrapper([error])
            with self.assertRaises(type(error)):
                env.reset(options={"curriculum_scene": self.descriptor})
            self.assertEqual(backend.calls, 1)

    def test_all_32_failed_admissions_remain_an_error(self):
        env, backend = self.wrapper([self.timeout() for _ in range(32)])
        with self.assertRaisesRegex(RuntimeError, "no legal curriculum placement"):
            env.reset(options={"curriculum_scene": self.descriptor})
        self.assertEqual(backend.calls, 32)
        self.assertEqual(len((self.directory / "admission-rejections.jsonl").read_text().splitlines()), 32)

    def test_parent_waits_only_while_bounded_admission_makes_progress(self):
        workers = EpisodeWorkers.__new__(EpisodeWorkers)
        workers.admission_limit = 32
        conn = Mock()
        progress = {"attempt": 0, "seed": 12, "kind": "tracking_initialization_timeout"}
        conn.recv.side_effect = [("admission_retry", progress), (True, "ready")]
        workers.connections = [conn]
        self.assertEqual(workers.receive(0), "ready")
        self.assertEqual(conn.poll.call_args_list[0].args, (60,))
        conn.poll.return_value = False
        with self.assertRaisesRegex(TimeoutError, "did not respond"):
            workers.receive(0)


if __name__ == "__main__":
    unittest.main()
