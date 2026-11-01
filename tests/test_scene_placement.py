"""Placement preserves task geometry, independent RNG and historical descriptors."""
import copy
import math
import unittest

import numpy as np

from src.training.curriculum import descriptor_scene
from src.training.curriculum_v3 import scene
from src.training.scene_admission import difficulty_signature
from src.training.scene_placement import SAMPLER, _intersects, candidate_scene, identity


class PlacementTests(unittest.TestCase):
    def candidate(self, course=7, seed=87022):
        value = scene(course, seed, "infantry_health_cooling", 22, release_revision=3)
        value["placement_search"] = dict(sampler=SAMPLER, hints_sha256=identity())
        return value

    def test_segment_screen_catches_crossings_parallel_and_tangent(self):
        start = np.array([[-2., 0.], [-2., 2.], [0., -2.], [-2., 1.]])
        end = np.array([[2., 0.], [2., 2.], [0., 2.], [2., 1.]])
        np.testing.assert_array_equal(_intersects(start, end, np.array([-1., -1.]),
                                                 np.array([1., 1.])), [True, False, True, True])

    def test_candidates_preserve_distance_angle_direction_and_all_task_fields(self):
        for course in (7, 8):
            source = self.candidate(course)
            reference = copy.deepcopy(source)
            poses = []
            for attempt in (0, 1, 31, 127):
                value = descriptor_scene(source, "infantry_health_cooling", attempt)
                self.assertEqual(difficulty_signature(source), difficulty_signature(value))
                duel = value["scenario"]["duel"]
                delta = np.array(duel["controlled"]["position_xz_m"]) - duel["target"]["position_xz_m"]
                self.assertAlmostEqual(np.linalg.norm(delta), source["initial_distance_m"], places=12)
                heading = duel["target"]["motion"]["translation_heading_rad"]
                actual = math.remainder(heading - math.atan2(-delta[1], delta[0]), math.tau)
                expected = math.radians(source["translation_angle_deg"]) * source["strata"]["lateral_side"]
                expected += math.pi if source["strata"]["direction"] < 0 else 0.
                self.assertAlmostEqual(math.remainder(actual - expected, math.tau), 0., places=12)
                poses.append(duel["target"]["position_xz_m"])
            self.assertEqual(source, reference)
            self.assertEqual(len({tuple(p) for p in poses}), 4)

    def test_repeatability_does_not_consume_global_random_stream(self):
        source = self.candidate()
        before = np.random.get_state()
        first = candidate_scene(source, 0)
        second = candidate_scene(source, 0)
        after = np.random.get_state()
        self.assertEqual(first, second)
        for a, b in zip(before, after):
            np.testing.assert_equal(a, b)
        self.assertEqual(first["placement_search"]["coarse_pool_size"], 32768)

    def test_explicit_opt_in_and_attempt_bounds(self):
        source = self.candidate()
        for attempt in (-1, 128):
            with self.assertRaises(ValueError):
                candidate_scene(source, attempt)
        source["placement_search"]["hints_sha256"] = "old"
        with self.assertRaises(ValueError):
            candidate_scene(source, 0)
        # Existing frozen banks, including their old retry generation, keep
        # exactly the same path unless this explicit sampler marker is present.
        for revision in (1, 2, 3):
            original = scene(7, 87022, "sentry", 22, release_revision=revision)
            self.assertEqual(descriptor_scene(original, "sentry", 4),
                             scene(7, 87022, "sentry", 22, attempt=4,
                                   strata=original["strata"], release_revision=revision))


if __name__ == "__main__":
    unittest.main()
