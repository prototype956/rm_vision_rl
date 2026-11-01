"""Default preset routing and manual reset boundaries without native processes."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.training.config import DEFAULT_CONFIG, ROOT, read_config
from src.training.train import run_training
from tools.config import load_tool_config
from tools.training.manual import ManualSession


class TrainingPresetsTests(unittest.TestCase):
    def test_role_specific_versioned_presets_keep_v2_default(self):
        expected = {f"rmul_2027_{role}_curriculum_v{version}.json" for role in ("infantry","sentry") for version in (2,3)}
        self.assertEqual({p.name for p in (ROOT/"config/training").glob("*.json")},expected)
        self.assertEqual(DEFAULT_CONFIG.name,"rmul_2027_infantry_curriculum_v2.json")
        for role in ("infantry","sentry"):
            config = read_config(ROOT/f"config/training/rmul_2027_{role}_curriculum_v2.json")
            self.assertEqual(config["policy_kind"],"asymmetric_gru")
            self.assertEqual(config["curriculum"]["version"],2)
            self.assertEqual(config["environment"]["scenario"]["duel"]["controlled"]["preset"],
                             "sentry" if role == "sentry" else "infantry_health_cooling")
        self.assertEqual(Path(load_tool_config("benchmark")["config"]),DEFAULT_CONFIG)

    def test_default_training_dispatches_to_recurrent_curriculum(self):
        destination = Path("unused-output")
        with patch("src.training.recurrent_ppo.run_recurrent_training",return_value=destination) as run:
            self.assertEqual(run_training(output_dir=destination),destination)
        config = run.call_args.args[0]
        self.assertEqual(config["curriculum"]["version"],2)
        self.assertEqual(config["environment"]["scenario"]["duel"]["controlled"]["preset"],
                         "infantry_health_cooling")
        self.assertEqual(config["total_timesteps"],1_000_000_000)
        self.assertEqual(run.call_args.kwargs,dict(resume=None,output_dir=destination))

    def test_cli_without_config_selects_infantry_v2(self):
        from src.training.train import main
        with patch("sys.argv",["train","--output-dir","unused-output"]), patch("src.training.train.run_training") as run:
            main()
        config = run.call_args.args[0]
        self.assertEqual(config["curriculum"]["version"],2)
        self.assertEqual(config["policy_kind"],"asymmetric_gru")
        self.assertEqual(config["n_envs"],128)

    def test_manual_course_resets_get_distinct_c0_scenes(self):
        env = Mock()
        env.action_space = SimpleNamespace(n=9)
        env.unwrapped = SimpleNamespace(action_mode="joint",episode_steps=3000,
                                       debug_snapshot=lambda:{"data":{"events":[]}})
        info = dict(episode_time_s=0.,action_masked=False,shot_requested=False,shot_accepted=False,
                    reject_reason=None,actual_shots=0,episode_damage=0.,end_reason=None)
        env.reset.return_value = ({"action_mask":np.ones(9,dtype=bool)},info)
        session = ManualSession(env,Mock(),Mock(),original_episode_steps=3000,
                                curriculum_role="sentry",scene_seed=17)
        for episode in (1,2):
            result = session.execute({"op":"reset"})
            self.assertEqual(result["metrics"]["episode"],episode)
        descriptors = [call.kwargs["options"]["curriculum_scene"] for call in env.reset.call_args_list]
        self.assertEqual([d["seed"] for d in descriptors],[17,18])
        for descriptor in descriptors:
            self.assertEqual(descriptor["stage_id"],"C0")
            self.assertEqual(descriptor["version"],2)
            self.assertEqual(descriptor["scenario"]["duel"]["controlled"]["preset"],"sentry")
        env.step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
