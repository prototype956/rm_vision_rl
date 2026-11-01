"""Explicit fixed-model evaluations recorded through the native Foxglove diagnostics."""
import copy
import hashlib
import io
import json
import tempfile
from pathlib import Path


def select_scenes(path, split, course, cases):
    """Select exact frozen descriptors; never substitute or retry another scene."""
    from src.training.curriculum_v2 import digest
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") not in (2, 3):
        raise ValueError("only frozen curriculum v2/v3 manifests are supported")
    if manifest.get("sha256") != digest({k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("frozen scene manifest checksum mismatch")
    if type(course) is not int or course < 0 or not cases or len(set(cases)) != len(cases):
        raise ValueError("select a course and unique cases explicitly")
    bank = manifest[split]
    if course >= len(bank):
        raise ValueError("course is outside the frozen scene bank")
    by_case = {d["case"]: d for d in bank[course]}
    rows = []
    for case in cases:
        if type(case) is not int or case not in by_case:
            raise ValueError(f"unknown case {case} in course {course}")
        scene = copy.deepcopy(by_case[case])
        if scene["course"] != course:
            raise ValueError("scene course mismatch")
        # Admission has one attempt using this exact recorded pose, not a derived scene.
        scene["admitted"] = True
        rows.append(scene)
    return manifest, rows


def snapshot_model(checkpoint, actor, output):
    if bool(checkpoint) == bool(actor):
        raise ValueError("provide exactly one Checkpoint or Actor")
    if checkpoint:
        from src.training.models import read_checkpoint_metadata
        data = Path(checkpoint).read_bytes()
        metadata = read_checkpoint_metadata(io.BytesIO(data))
        if metadata["config"]["policy_kind"] != "asymmetric_gru":
            raise ValueError("native MCAP model evaluation currently requires a GRU checkpoint")
        snapshot = output / "snapshot.zip"
        snapshot.write_bytes(data)
        model = dict(kind="checkpoint", sha256=hashlib.sha256(data).hexdigest(),
                     num_timesteps=metadata["num_timesteps"], fingerprints=metadata["fingerprints"],
                     observation_contract=metadata["observation"])
        return snapshot, metadata["config"], model, metadata
    from src.policy.recurrent_deployment import load_actor
    source = Path(actor)
    snapshot = output / "actor"
    snapshot.mkdir()
    for name in ("actor.pt", "manifest.json"):
        (snapshot / name).write_bytes((source / name).read_bytes())
    _, manifest = load_actor(snapshot)
    model = dict(kind="actor", sha256=manifest["sha256"], num_timesteps=manifest.get("training_steps"),
                 manifest_sha256=hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest())
    return snapshot, manifest["simulation_config"], model, manifest


def make_predictor(path, kind, device):
    import numpy as np
    import torch
    if kind == "actor":
        if device not in (None, "cpu"):
            raise ValueError("exported Actor evaluations use the existing CPU deployment runner")
        from src.policy.recurrent_deployment import ActorRunner
        runner = ActorRunner(path)
        return lambda obs: runner.predict(obs)[0]
    from src.training.recurrent_ppo import RecurrentPPO
    from src.policy.recurrent_observations import ACTOR_KEYS
    model = RecurrentPPO.load(path, device=device or "cpu")
    actor = model.policy.actor.eval()
    state = actor.initial_state(1)
    @torch.inference_mode()
    def predict(obs):
        nonlocal state
        tensors = {k: torch.as_tensor(np.asarray(obs[k])[None], device=model.device) for k in ACTOR_KEYS}
        _, _, actions, state = actor(tensors, state)
        return int(actions.item())
    return predict


def evaluate_scene(snapshot, config, model, descriptor, directory, *, device="cpu", record_mcap=True, combat_hud=True):
    from src.training.offline_evaluation import evaluate_scene as evaluate
    row = evaluate(config, descriptor, directory, kind=model["kind"], source=snapshot,
                   model=model, record_mcap=record_mcap, device=device, combat_hud=combat_hud)
    return dict(status=row["status"], scene=descriptor,
                mcap="replay.mcap" if record_mcap else None, summary=row["summary"])


def run(config):
    from tools.training.view import _atomic_json
    from src.training.curriculum_v2 import DEFAULT
    manifest, scenes = select_scenes(config["scenarios"], config.get("split", "validation"),
                                     config.get("course"), config.get("cases"))
    output_parent = Path(config["output_dir"])
    output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="foxglove-", dir=output_parent))
    snapshot, settings, model, metadata = snapshot_model(config.get("checkpoint"), config.get("actor"), output)
    role = settings["environment"]["scenario"]["duel"]["controlled"]["preset"]
    if any(d["scenario"]["duel"]["controlled"]["preset"] != role for d in scenes):
        raise ValueError("frozen scenes and model robot role differ")
    simulator_config = settings["environment"].get("simulator_config")
    for key in ("simulator_root", "vision_root", "simulator_binary", "bridge_binary", "simulator_config"):
        # Resolve portable replay paths against this checkout, retaining the model's
        # algorithm/scenario settings. Training machine absolute paths aren't portable.
        settings["environment"].pop(key, None)
    if simulator_config:
        original = Path(simulator_config)
        from src.environment.fire import simulator_root_path
        target = simulator_root_path() / "configs" / original.name
        if not target.is_file():
            raise FileNotFoundError(f"restore the model simulator configuration: {target}")
        settings["environment"]["simulator_config"] = str(target)
    if "curriculum" not in settings:
        from src.training.curriculum import implementation
        settings["curriculum"] = copy.deepcopy(implementation(manifest).DEFAULT)
    index = dict(version=1, source="synthetic_measurements", model=model,
                 scenarios_sha256=manifest["sha256"], split=config.get("split", "validation"),
                 status="running", scenes=[])
    _atomic_json(output / "replay_index.json", index)
    print(json.dumps(dict(output=str(output), status="running")), flush=True)
    failed = False
    for descriptor in scenes:
        name = f'C{descriptor["course"]}-case-{descriptor["case"]:02d}'
        try:
            row = evaluate_scene(snapshot, settings, model, descriptor, output / name,
                                 device=config.get("device", "cpu"), combat_hud=config.get("combat_hud", True))
            failed |= row["status"] != "complete"
        except KeyboardInterrupt:
            index["status"] = "failed"
            index["scenes"].append(dict(scene=descriptor,directory=name,status="interrupted"))
            _atomic_json(output / "replay_index.json", index)
            raise
        except Exception as error:
            failed = True
            row = dict(scene=descriptor, status="failed", error=f"{type(error).__name__}: {error}")
            (output / name).mkdir(exist_ok=True)
            _atomic_json(output / name / "failure.json", row)
        row["directory"] = name
        index["scenes"].append(row)
        _atomic_json(output / "replay_index.json", index)
        print(json.dumps(dict(directory=name, status=row["status"], error=row.get("error"))), flush=True)
    index["status"] = "failed" if failed else "complete"
    _atomic_json(output / "replay_index.json", index)
    return 1 if failed else 0
