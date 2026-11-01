"""Window-owned settlement and evaluation-only tracking metrics."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from src.policy.asymmetric_observations import NAMES
from src.training.offline_evaluation import evaluate_scene
from src.training.curriculum_v3 import scene
from src.training.curriculum_metrics import TrackingMetrics


class FakeEnvironment:
    def __init__(self, timed_out=False):
        self.unwrapped=self
        self.episode_steps=1
        self.timed_out=timed_out
        self._actor_semantic={"thermal":{}}
        self._bridge=SimpleNamespace(event_history={},record_snapshot=lambda _:None,close_recording=lambda _:None)
        self._client=SimpleNamespace(end_window=self.end_window,settle=self.settle)
        self._last_replay_command={"yaw":0}
        self.closed=False
        self._response=self.response(0,[],0,False,None)
    def event(self,ident,at,kind,data):
        return dict(event_id=ident,round_id=1,time_ns=at,kind=kind,data=data)
    def response(self,now,events,damage,dead,state):
        return dict(ok=True,round_id=1,sim_time_ns=now,data=dict(events=events,settlement=state,
            evaluation={"robots":[dict(robot_id=1,actual_shots=int(now>0),damage_dealt=damage,
                resources={"round_locked":False}),dict(robot_id=2,hp=0 if dead else 20)]}))
    def observation(self):
        features=np.zeros(len(NAMES));features[NAMES.index("target.state.tracking")]=1
        return {"features":features,"dt_s":np.array([.01]),"target_reset":np.array([False]),"action_mask":np.ones(9)}
    def reset(self,options):
        self.descriptor=options['curriculum_scene']
        return self.observation(),{"curriculum_scene":self.descriptor}
    def step(self,action):
        launch=self.event(1,5_000_000,"shot_fired",dict(projectile_id=1,request_id=1,robot_id=1))
        self._response=self.response(10_000_000,[launch],0,False,None)
        return self.observation(),0,False,True,dict(episode_steps=1,episode_damage=0,actual_shots=1,
            executed_action=0,selected_slot=0,shot_requested=True,shot_accepted=True,
            slot_switches=0,episode_training_reward=0,end_reason="time_limit")
    def end_window(self,max_settle_steps):
        assert max_settle_steps==1000
        status="timed_out" if self.timed_out else "settling"
        state=dict(window_end_ns=10_000_000,status=status,remaining={"projectiles":1})
        cutoff=self.event(2,10_000_000,"evaluation_window_closed",dict(window_end_ns=10_000_000))
        return self.response(10_000_000,[cutoff],0,False,state)
    def settle(self):
        damage=self.event(3,30_000_000,"damage_applied",dict(projectile_id=1,request_id=1,shooter=1,actual=20))
        ended=self.event(4,30_000_000,"projectile_ended",dict(projectile_id=1,request_id=1,reason="armor"))
        state=dict(window_end_ns=10_000_000,status="complete",remaining={"projectiles":0})
        return self.response(30_000_000,[damage,ended],20,True,state)
    def close(self):self.closed=True


class OfflineEvaluationTests(unittest.TestCase):
    def run_case(self,root,record=False,timeout=False):
        env=FakeEnvironment(timeout)
        with patch('src.training.environment.make_environment',return_value=env),patch('src.training.offline_evaluation.predictor',return_value=lambda obs:0):
            # The fake bridge records callbacks, not MCAP bytes; HUD file processing
            # has its own real-MCAP tests and is outside this settlement fixture.
            row=evaluate_scene({"curriculum":{"version":3}},scene(8,17,"infantry_health_cooling"),root,record_mcap=record,combat_hud=False)
        self.assertTrue(env.closed)
        return row
    def test_delayed_kill_belongs_to_launch_window_and_recording_is_passive(self):
        with tempfile.TemporaryDirectory() as tmp:
            off=self.run_case(Path(tmp)/'off');on=self.run_case(Path(tmp)/'on',record=True)
        self.assertEqual(off['end_reason'],'target_destroyed')
        self.assertEqual(off['damaging_hits'],1)
        self.assertEqual(off['actual_shots'],1)
        self.assertAlmostEqual(off['kill_time_s'],.03)
        self.assertEqual(off['summary']['window']['episode_damage'],0)
        self.assertEqual(off['summary']['settlement']['official_damage'],20)
        self.assertEqual(off['episode_return'],0)
        self.assertEqual(off['summary'],on['summary'])
    def test_timeout_never_claims_final_damage_or_kill(self):
        with tempfile.TemporaryDirectory() as tmp:row=self.run_case(Path(tmp)/'timeout',timeout=True)
        self.assertEqual(row['status'],'settlement_timed_out')
        self.assertIsNone(row['damage'])
        self.assertIsNone(row['summary']['settlement']['official_damage'])
        self.assertEqual(row['end_reason'],'time_limit')


def packet(tracking, reset=False, fire=False):
    features = np.zeros(len(NAMES), np.float32)
    features[NAMES.index("target.state.tracking")] = tracking
    return dict(features=features, target_reset=[reset], dt_s=[.01],
                action_mask=[True, False, fire, False, fire, False, fire, False, fire])


class TrackingMetricsTests(unittest.TestCase):
    def test_recovery_and_right_censored_loss_are_separate(self):
        metrics = TrackingMetrics(packet(True))
        for obs in [packet(False), packet(False), packet(True), packet(False, reset=True)]:
            metrics.observe(obs)
        result = metrics.result()
        self.assertAlmostEqual(result["recovery_times_s"][0], .02)
        self.assertEqual(result["target_resets"], 1)
        self.assertEqual(result["missing_steps"], 3)
        self.assertEqual(result["unrecovered_loss_s"], 0.)

    def test_untracked_fire_permission_is_visible_to_validation(self):
        metrics = TrackingMetrics(packet(True))
        metrics.observe(packet(False, fire=True))
        self.assertEqual(metrics.result()["unsafe_fire_mask_steps"], 1)


if __name__ == "__main__":
    unittest.main()
