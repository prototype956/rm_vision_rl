"""V2 scheduling, admission, information boundaries and bounded recurrent replay."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.training.curriculum_v2 import (
    Curriculum, DEFAULT, STAGES, GROUPS, PROFILES, scene, assess, bank, validate, validate_update,
)
from src.training.thermal_reward import costs, DEFAULT as THERMAL


class CurriculumV2Tests(unittest.TestCase):
    def test_warmup_collision_retries_only_unfrozen_admission(self):
        import gymnasium as gym
        from types import SimpleNamespace
        from src.training.environment import CurriculumScenario

        class Backend(gym.Env):
            def __init__(self, error):
                self.calls = 0
                self.error = error
                self._history = SimpleNamespace(clock=SimpleNamespace(config=None))
                self._client = Mock()
                self._client.inspect.return_value = {"data":{"evaluation":{"fireability":{}}}}
            def reset(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError(self.error)
                return {}, {}
            def step(self, action):
                raise RuntimeError("runtime collision: during policy")

        descriptor = scene(0,4063201340366790718,"infantry_health_cooling",814727)
        with patch("src.training.scene_admission.check_fireability"), patch("src.training.scene_admission.check_trackability",return_value={}):
            backend = Backend('runtime collision: blocked by "HighlandCurb"')
            env = CurriculumScenario(backend,17,descriptor['scenario'])
            _,info = env.reset(options={"curriculum_scene":descriptor})
            self.assertEqual(backend.calls,2)
            self.assertEqual(info['curriculum_scene']['attempt'],1)
            self.assertEqual(info['admission_rejections'][0]['attempt'],0)
            self.assertIn('HighlandCurb',info['admission_rejections'][0]['error'])
            with self.assertRaisesRegex(RuntimeError,"during policy"):
                env.step(0)
            self.assertEqual(backend.calls,2)
            frozen = Backend('runtime collision: blocked by "HighlandCurb"')
            with self.assertRaisesRegex(RuntimeError,"HighlandCurb"):
                CurriculumScenario(frozen,17,descriptor['scenario']).reset(
                    options={"curriculum_scene":dict(descriptor,admitted=True)})
            self.assertEqual(frozen.calls,1)
            broken = Backend('transport disconnected')
            with self.assertRaisesRegex(RuntimeError,"transport disconnected"):
                CurriculumScenario(broken,17,descriptor['scenario']).reset(options={"curriculum_scene":descriptor})
            self.assertEqual(broken.calls,1)

    def test_active_stages_and_independent_balanced_geometry(self):
        self.assertEqual(len(STAGES),12)
        self.assertNotIn("C4",STAGES)
        self.assertNotIn("C6",STAGES)
        for c,rows in enumerate(bank("sentry",DEFAULT)):
            profiles = [r["scenario"]["duel"]["target"]["profile"] for r in rows]
            self.assertEqual([profiles.count(p) for p in PROFILES],[10]*4)
            self.assertEqual(set(r["group"] for r in rows),set(GROUPS[c]))
            for p in PROFILES:
                distances = [r["initial_distance_m"] for r in rows if r["scenario"]["duel"]["target"]["profile"] == p]
                self.assertGreater(max(distances)-min(distances),.2)
            for row in rows:
                duel = row["scenario"]["duel"]
                self.assertNotIn("policy_start",row["scenario"])
                self.assertNotIn(duel["target"]["motion"]["kind"],("climb","tunnel","lift"))
                self.assertLessEqual(len(row["advanced_factors"]),2)
                self.assertEqual(duel["controlled"]["profile"],"standard")
            if c >= 5:
                self.assertEqual(sum(r["scenario"]["duel"]["target"]["preset"] == "sentry" for r in rows),20)

    def test_retry_only_changes_placement_and_resume_replays_rng(self):
        for row in bank("sentry",DEFAULT)[11]:
            other = scene(row["course"],row["seed"],"sentry",row["case"],1,row["strata"])
            self.assertEqual(row["initial_distance_m"],other["initial_distance_m"])
            self.assertEqual(row["scenario"]["duel"]["target"]["motion"],other["scenario"]["duel"]["target"]["motion"])
        a = Curriculum(DEFAULT)
        a.stage = 6
        a.next_scene("sentry",7)
        b = Curriculum(DEFAULT,state=a.state())
        for rank in (0,127,7,7):
            self.assertEqual(a.next_scene("sentry",rank),b.next_scene("sentry",rank))
        with self.assertRaises(ValueError):
            Curriculum(dict(version=1),state=a.state())

    def test_weights_and_thermal_do_not_block_promotion(self):
        c = Curriculum(DEFAULT)
        c.stage = 3
        c.reports = {str(i):dict(passed=True,deficit=0.) for i in range(3)}
        c.reports["0"]["deficit"] = .8
        w = c.weights()
        self.assertAlmostEqual(w[3],.7)
        self.assertAlmostEqual(w[:3].sum(),.3)
        self.assertTrue((w[:3] >= .15/3).all())
        self.assertFalse(w[4:].any())
        rows = [dict(r,actual_shots=20,damaging_hits=20,end_reason="target_destroyed",
                     ordinary_heat_locks=1) for r in bank("sentry",DEFAULT)[3]]
        report = assess(3,rows)
        self.assertTrue(report["passed"])
        self.assertFalse(report["thermal"]["passed"])
        for _ in range(3):c.observe(report)
        self.assertTrue(c.promote())

    def test_long_run_review_and_regression_recovery(self):
        c = Curriculum(DEFAULT)
        c.stage,c.mastered,c.last_full_review_steps = 11,True,1000
        c.reports = {str(i):dict(passed=True,deficit=0.) for i in range(12)}
        evaluator = Mock()
        evaluator.evaluate.return_value = dict(passed=True,deficit=0.)
        validate_update(c,evaluator,None,2_000_000,11)
        evaluator.evaluate.assert_not_called()
        def evaluate(actor,index,update):return dict(passed=index!=2,deficit=.5 if index==2 else 0.)
        evaluator.evaluate.side_effect = evaluate
        validate_update(c,evaluator,None,5_001_000,12)
        self.assertEqual(c.weak_streaks,{"2":0})
        evaluator.evaluate.side_effect = None
        for update in (13,14,15):validate_update(c,evaluator,None,5_001_000+update,update)
        self.assertEqual(c.weak_streaks,{})
        self.assertEqual(c.stage,11)

    def test_resource_configuration_and_remaining_budget(self):
        from src.training.config import read_config
        for role in ("infantry","sentry"):
            c = read_config(Path(__file__).parents[1]/f"config/training/rmul_2027_{role}_curriculum_v2.json")
            self.assertEqual(c["total_timesteps"],1_000_000_000)
            self.assertEqual(c["n_envs"],128)
            self.assertEqual(c["ppo"]["n_steps"],512)
            self.assertEqual(c["ppo"]["batch_size"],64)
            self.assertEqual(c["thermal_reward"],THERMAL)
            self.assertTrue(c["runtime"]["bounded_logging"])


    def test_streak_and_old_course_regression_block_promotion(self):
        c = Curriculum(DEFAULT)
        for _ in range(3):
            c.observe(dict(passed=True, deficit=0.))
        self.assertTrue(c.promote())
        for _ in range(2):
            c.observe(dict(passed=True, deficit=0.))
        self.assertFalse(c.promote())
        c.observe(dict(passed=False, deficit=.2))
        self.assertEqual(c.streak, 0)
        for _ in range(3):
            c.observe(dict(passed=True, deficit=0.))
        c.reports["0"] = dict(passed=False, deficit=.1)
        self.assertFalse(c.promote())
        c.reports["0"] = dict(passed=True, deficit=0.)
        self.assertTrue(c.promote())

    def test_model_checkpoint_preserves_curriculum_and_budget(self):
        import tempfile
        from pathlib import Path
        from src.training.config import read_config
        from src.training.recurrent_ppo import RecurrentPPO
        from src.training.models import _digest
        from src.policy.recurrent_observations import SCHEMA
        config = read_config(Path(__file__).resolve().parents[1]/"config/training/rmul_2027_sentry_curriculum_v2.json")
        config["device"] = "cpu"
        model = RecurrentPPO(config)
        curriculum = Curriculum(config["curriculum"])
        curriculum.stage = 5
        curriculum.next_scene("sentry", 2)
        model.curriculum_state = curriculum.state()
        model.num_timesteps, model.completed_updates = 24500, 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"checkpoint.zip"
            model.save(path, dict(format_version=1, observation=dict(version=5, schema_sha256=_digest(SCHEMA))))
            restored = RecurrentPPO.load(path, device="cpu")
        self.assertEqual(restored.num_timesteps, 24500)
        self.assertEqual(restored.config["total_timesteps"]-restored.num_timesteps, 999975500)
        resumed = Curriculum(config["curriculum"], state=restored.curriculum_state)
        self.assertEqual(curriculum.next_scene("sentry", 2), resumed.next_scene("sentry", 2))

    def test_no_shot_or_missing_group_cannot_pass(self):
        rows = [dict(group="rotation", actual_shots=0, damaging_hits=0, end_reason="time_limit")]*40
        self.assertFalse(assess(2, rows)["passed"])

    def test_native_pose_error_reaches_curriculum_retry_boundary(self):
        from src.environment.spawn import reset_spawn
        calls = []
        def reject(seed, scenario):
            calls.append(seed)
            raise RuntimeError("invalid seed 1: initial view blocked")
        with self.assertRaisesRegex(RuntimeError, "^invalid seed"):
            reset_spawn(reject, np.random.default_rng(17), {}, [], max_attempts=1)
        self.assertEqual(len(calls), 1)

    def test_export_only_resume_collects_no_training_rollout(self):
        from unittest.mock import Mock, patch
        from src.training.recurrent_ppo import export_reference
        env = Mock()
        env.reset.return_value = ({"features":np.array([1., 2.])}, {})
        env.step.return_value = ({"features":np.array([3., 4.])}, 0., False, False, {})
        descriptor = dict(admitted=True)
        with patch("src.training.environment.make_environment", return_value=env):
            reference = export_reference({}, "unused", descriptor)
        env.reset.assert_called_once_with(options={"curriculum_scene":descriptor})
        self.assertEqual(reference["features"].shape, (3, 2))
        self.assertEqual(env.step.call_count, 2)
        env.close.assert_called_once()

    def test_legal_scene_tracking_timeout_is_a_failure_not_an_engine_error(self):
        from unittest.mock import Mock, patch
        from pathlib import Path
        from src.training.config import read_config
        from src.training.curriculum_evaluation import _case
        config = read_config(Path(__file__).resolve().parents[1]/"config/training/rmul_2027_sentry_curriculum_v2.json")
        env = Mock()
        env.reset.side_effect = TimeoutError("fire-target warmup timed out: legal scene")
        with patch("src.training.environment.make_environment", return_value=env):
            row = _case((config, scene(0, 20000, "sentry"), None, "unused", True))
        self.assertIn("failure", row)
        self.assertNotIn("error", row)
        self.assertEqual(row["end_reason"], "tracking_initialization_timeout")
        self.assertFalse(assess(0, [row])["passed"])
        env.close.assert_called_once()

    def test_curriculum_and_future_events_cannot_enter_observation_encoders(self):
        from src.policy.asymmetric_observations import _example, adapt_observation, encode
        from src.policy.privileged import encode as encode_critic
        # Deterministic minimal inputs replace a captured simulator snapshot.
        raw = copy.deepcopy(_example)
        raw["estimate"]["state"] = "tracking"
        now_ns = 300_000_000
        packet = dict(version=1,imu=raw["imu"],thermal=raw["thermal"])
        hidden = dict(curriculum_id=11,sampling_probability=.7,future_blackouts_ms=[[250,1450]],
                      future_hit=True,future_motion=dict(kind="combined",amplitude_m=1.5))
        altered_raw = {**copy.deepcopy(raw),**hidden}
        altered_packet = {**copy.deepcopy(packet),**hidden}
        altered_packet["thermal"].update(hidden)
        a = encode(adapt_observation(raw,packet,now_ns))
        b = encode(adapt_observation(altered_raw,altered_packet,now_ns))
        self.assertEqual(a[0],b[0])
        np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(a[2],b[2])
        resource_keys = ("hp max_hp heat heat_limit base_cooling_per_second "
                         "effective_cooling_per_second allowance_remaining cooling_locked "
                         "round_locked cooling_active cooling_remaining_s defense_active "
                         "defense_remaining_s fold lift_m fire_ready requested_fold stable_s "
                         "requested_lift_m lift_velocity_m_s").split()
        robots,chassis,muzzles,armors = [],[],[],[]
        for owner in (1,2):
            robot = dict.fromkeys(resource_keys,0.)
            robot.update(robot_id=owner,position_bevy_m=[owner,0,0],rotation_bevy_xyzw=[0,0,0,1],
                         velocity_bevy_m_s=[0]*3,angular_velocity_bevy_rad_s=[0]*3,
                         blocked=None,region=None,tunnel_progress=None,pending_shot_s=None,last_shot_age_s=None)
            robots.append(robot)
            chassis.append(dict(owner=owner,yaw_rad=0.,yaw_velocity_rad_s=0.,rotation_bevy_xyzw=[0,0,0,1]))
            muzzles.append(dict(owner=owner,position_bevy_m=[0]*3,rotation_bevy_xyzw=[0,0,0,1]))
            armors.extend(dict(owner=owner,position_ros_m=[0]*3,rotation_ros_xyzw=[0,0,0,1],
                               corners_ros_m=[[0]*3 for _ in range(4)]) for _ in range(4))
        actuator = dict.fromkeys(("yaw_rad pitch_rad yaw_velocity_rad_s pitch_velocity_rad_s "
                                 "target_yaw_rad target_pitch_rad yaw_acceleration_rad_s2 "
                                 "pitch_acceleration_rad_s2 command_valid saturation_flags").split(),0.)
        truth = dict(version=1,sample_ns=now_ns,robots=robots,actuator=actuator,
                     geometry=dict(chassis=chassis,muzzles=muzzles,armors=armors),
                     projectiles=[],pending_commands=[],pending_requests=[])
        altered_truth = {**copy.deepcopy(truth),**hidden}
        for robot in altered_truth["robots"]:
            robot.update(hidden)
        np.testing.assert_array_equal(encode_critic(truth,now_ns),encode_critic(altered_truth,now_ns))

    def test_retired_curriculum_versions_are_rejected_before_reset(self):
        from src.training.config import read_config, validate_config
        from src.training.environment import CurriculumScenario
        import gymnasium as gym
        from types import SimpleNamespace
        config = read_config(Path(__file__).parents[1]/"config/training/rmul_2027_sentry_curriculum_v2.json")
        for version in (1,4):
            with self.assertRaises(ValueError):
                validate(dict(version=version))
            with self.assertRaises(ValueError):
                validate_config(dict(config,curriculum=dict(version=version)))
        class Backend(gym.Env):
            def __init__(self):
                self.calls = 0
                self._history = SimpleNamespace(clock=SimpleNamespace(config=None))
            def reset(self,**kwargs):
                self.calls += 1
                return {},{}
        backend = Backend()
        descriptor = scene(0,20000,"sentry")
        with self.assertRaisesRegex(ValueError,"only curriculum v2"):
            CurriculumScenario(backend,17,descriptor["scenario"]).reset(
                options={"curriculum_scene":dict(descriptor,version=1)})
        self.assertEqual(backend.calls,0)


class ThermalRewardTests(unittest.TestCase):
    def test_union_time_and_joint_event_priority(self):
        a = dict(ordinary_lock_count=0,round_lock_count=0,joint_lock_count=0,locked_s=0.)
        b = dict(ordinary_lock_count=1,round_lock_count=1,joint_lock_count=1,locked_s=.01)
        r = costs(a,b,THERMAL)
        self.assertEqual(r["ordinary_heat_penalty"],0.)
        self.assertEqual(r["round_heat_penalty"],10.)
        self.assertAlmostEqual(r["locked_time_penalty"],.002)
        self.assertEqual(sum(costs(b,b,THERMAL).values()),0.)
        with self.assertRaises(ValueError):costs(b,a,THERMAL)

    def test_time_integral_and_takeover_baseline(self):
        a = dict(ordinary_lock_count=5,round_lock_count=0,joint_lock_count=0,locked_s=3.)
        b = dict(a,ordinary_lock_count=6,locked_s=3.7)
        c = dict(b,locked_s=4.9)
        split = sum(costs(a,b,THERMAL).values())+sum(costs(b,c,THERMAL).values())
        self.assertAlmostEqual(split,2+.2*1.9)
        self.assertAlmostEqual(split,sum(costs(a,c,THERMAL).values()))

    def test_reward_wrapper_adds_costs_before_bootstrap_without_observation_changes(self):
        import gymnasium as gym
        from src.training.environment import TrainingReward
        class Env(gym.Env):
            def reset(self,**kwargs):
                return {"actor":1},dict(physical_thermal=dict(ordinary_lock_count=7,round_lock_count=0,joint_lock_count=0,locked_s=1.))
            def step(self,action):
                return {"actor":1},20.,False,True,dict(physical_thermal=dict(ordinary_lock_count=8,round_lock_count=0,joint_lock_count=0,locked_s=1.1))
        env = TrainingReward(Env(),.05,thermal_reward=THERMAL)
        before,_ = env.reset()
        after,r,_,truncated,info = env.step(0)
        self.assertEqual(before,after)
        self.assertTrue(truncated)
        self.assertAlmostEqual(r,1-2-.02)
        self.assertAlmostEqual(info["episode_training_reward"],r)
        self.assertEqual(info["switch_penalty"],0.)


class StreamingTests(unittest.TestCase):
    def test_restored_optimizer_continues_identically_to_uninterrupted_training(self):
        from tests.test_recurrent_contract import RecurrentContractTest
        from src.training.config import read_config
        from src.training.recurrent_ppo import RecurrentPPO
        from src.training.models import _digest
        from src.policy.recurrent_observations import SCHEMA
        for role in ("infantry","sentry"):
            config = read_config(Path(__file__).parents[1]/f"config/training/rmul_2027_{role}_curriculum_v2.json")
            config["device"] = "cpu"
            config["ppo"].update(sequence_length=5,burn_in=3,batch_size=2,n_epochs=1)
            original = RecurrentPPO(config)
            original.update(RecurrentContractTest().episodes(original.policy))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"saved.zip"
                original.save(path,dict(format_version=1,observation=dict(version=5,schema_sha256=_digest(SCHEMA))))
                restored = RecurrentPPO.load(path,device="cpu")
            fresh = RecurrentContractTest().episodes(original.policy)
            original.update(fresh)
            restored.update(fresh)
            self.assertEqual(original.num_timesteps,restored.num_timesteps)
            self.assertEqual(original.completed_updates,restored.completed_updates)
            for key,value in original.policy.state_dict().items():
                torch.testing.assert_close(value,restored.policy.state_dict()[key],atol=0,rtol=0)

    def test_validation_jobs_use_bytes_without_shared_tensor_descriptors(self):
        import contextlib
        import io
        from src.training.config import read_config
        from src.training.curriculum_evaluation import CurriculumEvaluator
        from src.policy.recurrent_actor import RecurrentActor
        config = read_config(Path(__file__).parents[1]/"config/training/rmul_2027_infantry_curriculum_v2.json")
        actor = RecurrentActor()
        with tempfile.TemporaryDirectory() as directory:
            evaluator = CurriculumEvaluator(config,directory)
            evaluator.pool.shutdown()
            captured = []
            class Pool:
                def map(self,fn,jobs):
                    captured.extend(jobs)
                    return [dict(group=d["group"],profile=d["scenario"]["duel"]["target"]["profile"],
                        actual_shots=20,damaging_hits=20,end_reason="target_destroyed") for _,d,_,_,_ in jobs]
                def shutdown(self,**kwargs):pass
            evaluator.pool = Pool()
            with contextlib.redirect_stdout(io.StringIO()):evaluator.evaluate(actor,0,1)
            self.assertEqual(len(captured),40)
            payload = captured[0][2]
            self.assertIsInstance(payload,bytes)
            restored = torch.load(io.BytesIO(payload),map_location="cpu",weights_only=True)
            for key,value in actor.state_dict().items():torch.testing.assert_close(value,restored[key])
            evaluator.close()

    def test_v2_export_uses_deployment_only_configuration(self):
        import json
        from src.policy.recurrent_actor import RecurrentActor
        from src.policy.recurrent_deployment import export_actor,load_actor
        from src.training.config import read_config
        config = read_config(Path(__file__).parents[1]/"config/training/rmul_2027_infantry_curriculum_v2.json")
        with tempfile.TemporaryDirectory() as directory:
            export_actor(RecurrentActor(),directory,config=config,training_steps=77)
            _,manifest = load_actor(directory)
        self.assertEqual(manifest["training_context"]["curriculum_version"],2)
        self.assertEqual(manifest["training_steps"],77)
        self.assertFalse({"curriculum","thermal_reward","runtime"} & manifest["simulation_config"].keys())
        self.assertEqual(manifest["simulation_config"]["environment"]["scenario"]["duel"]["controlled"]["motion"]["kind"],"static")

    def test_native_logs_rotate_and_preserve_latest_bytes(self):
        import os
        import subprocess
        import sys
        from unittest.mock import patch
        from src.transport.native_log import NativeLog
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{"RM_BOUNDED_NATIVE_LOGS":"1"}):
            path = Path(directory)/"native.log"
            with NativeLog(path,1024) as log:
                process = subprocess.Popen([sys.executable,"-c","import sys;sys.stdout.write('x'*20000+'TAIL')"],stdout=log.output)
                log.spawned()
                self.assertEqual(process.wait(10),0)
            self.assertTrue(path.read_bytes().endswith(b"TAIL"))
            self.assertLessEqual(path.stat().st_size,1024)
            self.assertLessEqual(path.with_suffix(".log.1").stat().st_size,1024)

    def test_cpu_replay_cache_matches_full_batch(self):
        from tests.test_recurrent_contract import RecurrentContractTest
        from src.training.recurrent_policy import RecurrentPolicy
        from src.training.recurrent_ppo import pack_episodes,state_cache,segments
        from src.training.sequence_storage import replay_cache,window
        policy = RecurrentPolicy()
        episodes = RecurrentContractTest().episodes(policy)
        chunks = segments(episodes,5,3)
        old = state_cache(policy,pack_episodes(episodes,"cpu"),episodes,chunks)
        new = replay_cache(policy,episodes,chunks,1)
        for key in old:
            for a,b in zip((*old[key][0],old[key][1]),(*new[key][0],new[key][1])):
                torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
                self.assertEqual(b.storage_offset(),0)
        obs,targets,sizes,burn_sizes,burn = window(episodes,chunks[-2:],"cpu",0.,1.)
        self.assertLessEqual(len(obs["features"]),8)
        for col,(e,start,_,size) in enumerate(chunks[-2:]):
            np.testing.assert_array_equal(obs["features"][burn:burn+size,col].numpy(),episodes[e]["observations"]["features"][start:start+size])
            np.testing.assert_array_equal(targets["log_probs"][:size,col].numpy(),episodes[e]["log_probs"][start:start+size])


if __name__ == "__main__":
    unittest.main()
