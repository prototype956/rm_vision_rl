"""Selective repair must preserve accepted evidence and report real failures."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.training.curriculum_v2 import digest
from src.training.curriculum_v3 import scene
from src.training.scene_placement import candidate_scene
from tools.training.curriculum_v3 import read, write
from tools.training.repair_scene_admissions import repair


class RepairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.binary = self.root / "native"
        self.binary.write_bytes(b"unchanged native")
        self.config = self.root / "simulation"
        self.config.write_bytes(b"unchanged config")
        self.native = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        self.simulation = hashlib.sha256(self.config.read_bytes()).hexdigest()
        self.good = self.certify(scene(7, 87000, "sentry", 0, release_revision=3))
        self.bad = scene(7, 87001, "sentry", 1, release_revision=3)
        # Reproduce the historical blocked manifest's stale flag. The failure
        # list must override it, while the accepted descriptor remains intact.
        self.bad["admitted"] = True
        self.frozen = dict(version=3, calibration=[[] for _ in range(12)],
                           validation=[[] for _ in range(12)], test=[[] for _ in range(12)])
        self.frozen["calibration"][7] = [self.good, self.bad]
        self.frozen["sha256"] = digest(self.frozen)
        folder = self.root / "input/sentry"
        write(folder / "curriculum/scenarios.json", self.frozen)
        write(folder / "config.json", dict(curriculum=dict(release_revision=3)))
        write(folder / "preflight.json", dict(passed=False,
            failures=[dict(split="calibration", course=7, case=1, error="invalid explicit duel")]))
        self.args = SimpleNamespace(scenarios=self.root / "input", output=self.root / "output",
                                    roles=["sentry"], workers=1)
        probe = SimpleNamespace(unwrapped=SimpleNamespace(simulator_binary=self.binary,
            simulator_config=self.config), close=lambda: None)
        patcher = patch("src.training.environment.make_environment", return_value=probe)
        patcher.start()
        self.addCleanup(patcher.stop)

    def certify(self, value):
        value = copy.deepcopy(value)
        value["admitted"] = True
        value["visibility_certificate"] = dict(scenario_sha256=digest(value["scenario"]),
            native_sha256=self.native, simulator_config_sha256=self.simulation,
            proof=dict(version=1, policy="full_path_clear", duration_ms=45000,
                max_interval_ms=10, min_interval_ms=1, min_visible_armor_count=1,
                algorithm_fingerprint_fnv64="fixture", policy_inputs=False))
        return value

    def test_only_failed_case_runs_and_accepted_descriptor_is_identical(self):
        def evaluate(config, scenes, output, workers, **kwargs):
            self.assertEqual([d["case"] for d in scenes], [1])
            self.assertNotIn("admitted", scenes[0])
            return [dict(case=1, admitted_scene=self.certify(candidate_scene(scenes[0], 0)))]
        with patch("tools.training.curriculum_v3.evaluate", side_effect=evaluate) as evaluator:
            repair(self.args)
        evaluator.assert_called_once()
        actual = read(self.args.output / "sentry/curriculum/scenarios.json")
        self.assertEqual(actual["calibration"][7][0], self.good)
        self.assertNotEqual(actual["calibration"][7][1]["scenario"], self.bad["scenario"])
        self.assertEqual(read(self.args.output / "repair-summary.json")["repaired"], 1)
        self.assertEqual(read(self.root / "input/sentry/curriculum/scenarios.json"), self.frozen)

    def test_warmup_failure_is_retained_and_blocks_ready_manifest(self):
        with patch("tools.training.curriculum_v3.evaluate", return_value=[dict(case=1,
                failure="TrackingInitializationTimeout: no tracking")]) as evaluator:
            with self.assertRaisesRegex(RuntimeError, "no release"):
                repair(self.args)
        evaluator.assert_called_once()
        self.assertEqual(read(self.args.output / "report.json")["status"], "blocked")
        actual = read(self.args.output / "sentry/curriculum/scenarios.json")
        self.assertEqual(actual["calibration"][7][0], self.good)
        self.assertFalse(actual["calibration"][7][1]["admitted"])

    def test_invalid_accepted_certificate_blocks_before_any_simulation(self):
        altered = copy.deepcopy(self.frozen)
        altered["calibration"][7][0]["visibility_certificate"]["native_sha256"] = "different"
        altered.pop("sha256")
        altered["sha256"] = digest(altered)
        write(self.root / "input/sentry/curriculum/scenarios.json", altered)
        with patch("tools.training.curriculum_v3.evaluate") as evaluator:
            with self.assertRaisesRegex(ValueError, "certificate"):
                repair(self.args)
        evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
