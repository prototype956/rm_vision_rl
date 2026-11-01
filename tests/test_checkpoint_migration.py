"""Checkpoint compatibility policies and immutable training payloads."""
import copy
import unittest
from tools.training.checkpoint_migration import check_clearance
from tools.training.checkpoint_migration import WARMUP_SOURCES, check_warmup
from tools.training.checkpoint_migration import check_curriculum


class ClearanceMigrationTests(unittest.TestCase):
    def metadata(self):
        return dict(observation={"version":5},policy_contract={"kind":"asymmetric_gru"},dependencies={"torch":"same"},
            config={"device":"cuda:0","total_timesteps":1000000000,"n_envs":128,"curriculum":{"version":2},
                    "ppo":{"gamma":0.999},"thermal_reward":{"ordinary_onset":2},
                    "environment":{"scene_seed":17,"simulator_root":"old","bridge_binary":"old"}},
            fingerprints={"environment":"old","native_training":"old","training_contract":"old",
                          "reward":"same","curriculum":"same","python_sources":{"unchanged":"hash"},"vision_bridge":"same"})

    def test_only_native_binary_and_runtime_locations_may_change(self):
        before=self.metadata();after=copy.deepcopy(before)
        after['config']['environment'].update(simulator_root='new',bridge_binary='new')
        for key in ('environment','native_training','training_contract'):after['fingerprints'][key]='new'
        check_clearance(before,after)

    def test_budget_reward_curriculum_and_ppo_changes_are_rejected(self):
        before=self.metadata()
        for group,key,value in ((None,'total_timesteps',1003800921),(None,'n_envs',4),
                                ('thermal_reward','ordinary_onset',0),('curriculum','version',3),('ppo','gamma',.99)):
            with self.subTest(group=group,key=key):
                after=copy.deepcopy(before)
                target=after['config'] if group is None else after['config'][group]
                target[key]=value
                with self.assertRaisesRegex(ValueError,'training configuration'):check_clearance(before,after)

    def test_network_observation_python_and_vision_changes_are_rejected(self):
        before=self.metadata()
        for key in ('observation','policy_contract','dependencies'):
            after=copy.deepcopy(before);after[key]={'changed':True}
            with self.assertRaises(ValueError):check_clearance(before,after)
        for key in ('python_sources','vision_bridge','reward','curriculum'):
            after=copy.deepcopy(before);after['fingerprints'][key]='changed'
            with self.assertRaisesRegex(ValueError,'fingerprints'):check_clearance(before,after)


class WarmupMigrationTests(unittest.TestCase):
    def pair(self):
        old=dict(observation={},policy_contract={},dependencies={},config={
            'device':'cuda:0','total_timesteps':1000000000,'environment':{'simulator_root':'old'},
            'ppo':{'learning_rate':.0003},'reward_scale':.05,'curriculum':{'version':2}},
            fingerprints={'environment':'old','training_contract':'old','native_training':'same',
                          'vision_bridge':'same','reward':'same','v2_sources':{'recurrent_ppo.py':'same'},
                          'python_sources':{k:'old' for k in WARMUP_SOURCES}|{'other.py':'same'}})
        new=copy.deepcopy(old)
        new['config']['environment']['simulator_root']='new'
        new['fingerprints']['environment']='new'
        new['fingerprints']['training_contract']='new'
        new['fingerprints']['python_sources'].update({k:'new' for k in WARMUP_SOURCES})
        return old,new

    def test_only_the_four_reviewed_admission_sources_are_allowed(self):
        old,new=self.pair()
        self.assertEqual(set(check_warmup(old,new)),WARMUP_SOURCES)

    def test_unrelated_python_change_is_rejected(self):
        old,new=self.pair()
        new['fingerprints']['python_sources']['other.py']='changed'
        with self.assertRaisesRegex(ValueError,'unexpected Python'):
            check_warmup(old,new)

    def test_budget_and_optimizer_configuration_are_preserved(self):
        for target,key,value in (('config','total_timesteps',1007652379),('ppo','learning_rate',.001)):
            old,new=self.pair()
            (new['config'] if target=='config' else new['config']['ppo'])[key]=value
            with self.assertRaisesRegex(ValueError,'training configuration'):
                check_warmup(old,new)

    def test_native_vision_reward_and_observations_are_preserved(self):
        for key in ('native_training','vision_bridge','reward','v2_sources'):
            old,new=self.pair()
            new['fingerprints'][key]='changed'
            with self.assertRaises(ValueError):check_warmup(old,new)
        old,new=self.pair()
        new['observation']={'different':True}
        with self.assertRaises(ValueError):check_warmup(old,new)


def metadata():
    return dict(observation={"version":5},policy_contract={"version":5},dependencies={"torch":"same"},
        config=dict(device="cuda:0",n_envs=128,total_timesteps=1_000_000_000,
            curriculum={"version":2},ppo={"learning_rate":.0003},
            environment={"simulator_root":"/old","episode_steps":3000,"scenario":{"same":True}}),
        fingerprints=dict(simulator_config="same",vision_config={"x":"same"},reward="same",
            python_sources={"training/curriculum_v2.py":"old","policy/recurrent_actor.py":"unchanged"}))


class CurriculumMigrationTests(unittest.TestCase):
    def test_only_versioned_curriculum_and_paths_may_change(self):
        old=metadata();new=copy.deepcopy(old)
        new["config"]["curriculum"]={"version":3}
        new["config"]["environment"]["simulator_root"]="/new"
        new["fingerprints"]["python_sources"]["training/curriculum_v3.py"]="new"
        check_curriculum(old,new)
    def test_learning_budget_rewards_and_observations_cannot_change(self):
        old=metadata()
        for field,value in (("n_envs",64),("total_timesteps",2_000_000_000),("ppo",{"learning_rate":.001})):
            new=copy.deepcopy(old);new['config'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):check_curriculum(old,new)
        for field in ('observation','dependencies','policy_contract'):
            new=copy.deepcopy(old);new[field]={"different":True}
            with self.subTest(field=field),self.assertRaises(ValueError):check_curriculum(old,new)
        new=copy.deepcopy(old);new['fingerprints']['reward']='different'
        with self.assertRaises(ValueError):check_curriculum(old,new)
        new=copy.deepcopy(old);new['fingerprints']['python_sources']['policy/recurrent_actor.py']='different'
        with self.assertRaises(ValueError):check_curriculum(old,new)


class MigrationReviewGateTests(unittest.TestCase):
    def test_environment_or_incomplete_review_never_writes_checkpoint(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import Mock,patch
        import numpy as np
        from src.policy.recurrent_actor import CONTRACT
        from tools.training.checkpoint_migration import migrate_curriculum
        for field in ("environment_errors","incomplete_cases"):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                old=metadata();old["policy_contract"]=CONTRACT;old["config"]["seed"]=17;old["completed_updates"]=107
                old["curriculum_state"]=dict(stage=0,counts={"0":12},rng=np.random.default_rng(17).bit_generator.state)
                new=copy.deepcopy(old);new["config"]["curriculum"]={"version":3}
                evaluator=Mock()
                evaluator.evaluate.return_value=dict(passed=False,deficit=1.,**{field:1})
                source=Path(tmp)/"source.zip";source.write_bytes(b"unchanged checkpoint placeholder")
                destination=Path(tmp)/"migrated.zip"
                with patch('src.training.models.read_checkpoint_metadata',return_value=old),patch('src.training.models.training_metadata',return_value=new),patch('src.training.environment.make_environment'),patch('src.training.recurrent_ppo.RecurrentPPO.load',return_value=SimpleNamespace(policy=SimpleNamespace(actor=object()))),patch('src.training.curriculum_evaluation.CurriculumEvaluator',return_value=evaluator):
                    with self.assertRaisesRegex(RuntimeError,"migration review"):
                        migrate_curriculum(source,destination,new["config"],{},Path(tmp)/"review")
                self.assertFalse(destination.exists())
                self.assertTrue((Path(tmp)/"review/migration-blocked.json").exists())
                self.assertEqual(source.read_bytes(),b"unchanged checkpoint placeholder")
                evaluator.close.assert_called_once()


class CheckpointArchiveTests(unittest.TestCase):
    def test_payload_is_opaque_and_source_cannot_be_overwritten(self):
        import json
        import tempfile
        from pathlib import Path
        from zipfile import ZipFile
        from tools.training.checkpoint_migration import read_payload, write_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp)/"source.zip", Path(tmp)/"target.zip"
            payload = b"weights\x00optimizer\xffrng"
            write_checkpoint(source, {"step": 12}, payload)
            original = source.read_bytes()
            write_checkpoint(target, {"migrated": True}, read_payload(source))
            self.assertEqual(source.read_bytes(), original)
            with ZipFile(target) as archive:
                self.assertEqual(archive.read("recurrent.pt"), payload)
                self.assertTrue(json.loads(archive.read("rmvision.json"))["migrated"])
            with self.assertRaises(FileExistsError):
                write_checkpoint(target, {}, b"changed")
            self.assertEqual(read_payload(target), payload)

    def test_extra_or_duplicate_archive_members_are_rejected(self):
        import tempfile
        import warnings
        from pathlib import Path
        from zipfile import ZipFile
        from tools.training.checkpoint_migration import read_payload
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"source.zip"
            for extra in ("unexpected", "recurrent.pt"):
                with self.subTest(extra=extra):
                    with ZipFile(path, "w") as archive, warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        archive.writestr("rmvision.json", "{}")
                        archive.writestr("recurrent.pt", b"original")
                        archive.writestr(extra, b"unexpected")
                    with self.assertRaisesRegex(ValueError, "archive members"):
                        read_payload(path)


if __name__ == "__main__":
    unittest.main()
