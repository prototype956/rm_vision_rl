"""验证模式隔离、动作降级与时钟不受目标代次影响；不模拟火控算法。"""
import copy
import unittest
import numpy as np

from src.environment.rotation_fire import RotationFireEnv
from src.policy.decision import make_policy
from src.policy.observations import SCHEMA, JOINT_SCHEMA
from src.training.config import read_config, validate_config
from src.training.models import observation_contract

CLOCK = dict(min_interval_ms=50, max_interval_ms=100, resample='decision', seed=17)


class JointContractTest(unittest.TestCase):
    def test_old_config_and_schema_stay_frozen(self):
        old = read_config('config/training/rotation_fire_ppo.json')
        self.assertNotIn('action_mode', old['environment'])
        self.assertEqual(len(SCHEMA['features']), 90)
        self.assertEqual(JOINT_SCHEMA['features'][:90], SCHEMA['features'])
        for mode, count, shape, version in [('fire_only', 2, (8, 90), 1), ('joint', 9, (8, 107), 2)]:
            with RotationFireEnv(action_mode=mode, decision_clock=CLOCK) as env:
                self.assertEqual(env.action_space.n, count)
                self.assertEqual(env.observation_space['features'].shape, shape)
                self.assertEqual(observation_contract(env)['version'], version)

    def test_bad_modes_and_clock_ranges_rejected(self):
        for mode in (None, False, 'nine', 'rule'):
            config = read_config(); config['environment']['action_mode'] = mode
            with self.assertRaises(ValueError): validate_config(config)
        for bounds in ((0, 7), (1, 8), (7, 1), (1, float('nan'))):
            config = read_config(); config['environment']['angular_speed_range_rad_s'] = bounds
            with self.assertRaises(ValueError): validate_config(config)

    def test_mask_and_same_slot_fallback(self):
        p = make_policy('joint', decision_clock=CLOCK); p.clock.start_episode()
        p.begin_step(); p._update_mask([True] * 9)
        for action in range(9): self.assertEqual(p.resolve_action(action), (action, action))
        p.clock.complete_step(); p.begin_step(); p._update_mask([True] * 9)
        self.assertEqual(np.flatnonzero(p.mask).tolist(), [0, 1, 3, 5, 7])
        for action in (2, 4, 6, 8): self.assertEqual(p.resolve_action(action), (action-1, action-1))
        p.begin_step(); p._update_mask([True] + [False] * 8)
        for action in range(9): self.assertEqual(p.resolve_action(action), (0, 0))

    def test_generation_reset_and_skipped_ticks_do_not_resample_clock(self):
        a, b = [make_policy('joint', decision_clock=CLOCK) for _ in range(2)]
        for p in (a, b): p.clock.start_episode()
        due = []
        for step in range(200):
            a.begin_step(); a.set_generation(step)
            a.resolve_action(0)  # No callback / LOST: still consumes due opportunities.
            self.assertEqual(a.clock.info(), b.clock.info())
            if a.clock.due: due.append(step)
            a.clock.complete_step(); b.clock.complete_step()
        self.assertTrue(all(5 <= y-x <= 10 for x, y in zip(due, due[1:])))

    def test_fire_only_mapping_unchanged(self):
        p = make_policy('fire_only', decision_clock=CLOCK); p.clock.start_episode(); p.begin_step()
        mask = [True] + [False] * 8; mask[5] = mask[6] = True
        p._update_mask(mask)
        self.assertEqual(p.resolve_action(0), (0, 5))
        self.assertEqual(p.resolve_action(1), (1, 6))
        p.clock.complete_step(); p.begin_step(); p._update_mask(mask)
        self.assertEqual(p.resolve_action(1), (0, 5))

    def test_sampling_sequence_is_independent_of_actions(self):
        with RotationFireEnv(action_mode='joint') as a, RotationFireEnv() as b:
            scenes = []
            for i in range(4):
                sa, ra, ia = a.prepare_scene(seed=17 if i == 0 else None)
                sb, rb, ib = b.prepare_scene(seed=17 if i == 0 else None)
                self.assertEqual((sa, ia), (sb, ib)); self.assertEqual(ra.integers(2**63), rb.integers(2**63))
                scenes.append(copy.deepcopy(sa))
            self.assertNotEqual(scenes[0], scenes[1])
            self.assertEqual(a.prepare_scene(seed=17)[0], scenes[0])


if __name__ == '__main__':
    unittest.main()
