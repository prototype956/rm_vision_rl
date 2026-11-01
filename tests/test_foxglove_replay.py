"""Regression tests for the replay's model/scenario and diagnostic isolation boundaries."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.config import _validate
from tools.training.foxglove_replay import select_scenes
from src.training.curriculum_v2 import digest
from src.transport.vision_bridge import VisionBridge

class ReplayContracts(unittest.TestCase):
    def test_exact_frozen_selection_and_checksum(self):
        scene = dict(version=2, course=0, case=3, seed=17, scenario={"pose":"original"})
        manifest = dict(version=2, validation=[[scene]], test=[[scene]])
        manifest["sha256"] = digest(manifest)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"scenarios.json"
            path.write_text(json.dumps(manifest))
            _, rows = select_scenes(path,"validation",0,[3])
            self.assertEqual(rows[0]["scenario"],scene["scenario"])
            self.assertTrue(rows[0]["admitted"])
            with self.assertRaises(ValueError):select_scenes(path,"validation",0,[0])
            manifest["validation"][0][0]["seed"] = 99
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):select_scenes(path,"validation",0,[3])

    def test_explicit_model_and_scene_contract(self):
        good = dict(mode="actor",actor="artifacts/actor",record_mcap=True,
                    scenarios="artifacts/scenarios.json",course=0,cases=[0])
        self.assertEqual(_validate("view",good)["mode"],"actor")
        self.assertTrue(_validate("view",good)["combat_hud"])
        self.assertFalse(_validate("view",{**good,"combat_hud":False})["combat_hud"])
        with self.assertRaises(ValueError): _validate("view",{**good,"combat_hud":"false"})
        for changes in ({"checkpoint":"a.zip"},{"record_mcap":False},{"cases":[0,0]},
                        {"scenarios":None},{"course":-1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _validate("view",{**good,**changes})

    def test_actor_identity_keeps_exported_training_steps(self):
        from tools.training.foxglove_replay import snapshot_model
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);actor=root/"actor";actor.mkdir();output=root/"output";output.mkdir()
            (actor/"actor.pt").write_bytes(b"fixed actor snapshot")
            (actor/"manifest.json").write_text("{}")
            manifest=dict(sha256="actor-hash",training_steps=31583059,simulation_config={})
            with patch("src.policy.recurrent_deployment.load_actor",return_value=(None,manifest)):
                snapshot,_,model,_=snapshot_model(None,actor,output)
            self.assertEqual(model["num_timesteps"],31583059)
            self.assertEqual((snapshot/"actor.pt").read_bytes(),(actor/"actor.pt").read_bytes())

    def test_truth_uses_separate_post_computation_message(self):
        bridge = VisionBridge(None)
        bridge.recording=True
        sent=[]
        bridge.exchange=lambda value: sent.append(copy.deepcopy(value)) or {"ok":True}
        response=dict(ok=True,round_id=1,sim_time_ns=10,data=dict(
            evaluation={"robots":[]},events=[],self_referee={},visual_frames=[],
            privileged_state={"secret_truth":1},actor_feedback={"legitimate":1},reward_damage=99))
        bridge.record_snapshot(response)
        self.assertEqual(sent[0]["op"],"record_snapshot")
        self.assertEqual(set(sent[0]["response"]["data"]),
            {"evaluation","events","self_referee","visual_frames"})
        self.assertNotIn("privileged_state",sent[0]["response"]["data"])

    def test_dashboard_actor_paths_and_frozen_fields(self):
        from tools.training.dashboard.schemas import validate_tool_payload, RequestError
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"config/tools").mkdir(parents=True)
            (root/"config/tools/view.json").write_text(json.dumps(dict(mode="checkpoint",scene_count=1,scene_seeds=[],workers=1,output_dir="artifacts/replays")))
            actor=root/"artifacts/actor";actor.mkdir(parents=True)
            scenes=root/"artifacts/scenes.json";scenes.write_text("{}")
            request=dict(mode="actor",actor=str(actor),record_mcap=True,scenarios=str(scenes),course=0,cases=[0,1])
            values=validate_tool_payload("view",request,root)
            self.assertTrue(values["record_mcap"])
            self.assertTrue(values["combat_hud"])
            self.assertFalse(validate_tool_payload("view",{**request,"combat_hud":False},root)["combat_hud"])
            with self.assertRaises(RequestError):
                validate_tool_payload("view",{**request,"combat_hud":"false"},root)
            self.assertIsNone(values["checkpoint"])
            self.assertEqual(values["cases"],[0,1])
            with self.assertRaises(RequestError):
                validate_tool_payload("view",{**request,"actor":str(root.parent)},root)

if __name__=="__main__":unittest.main()
