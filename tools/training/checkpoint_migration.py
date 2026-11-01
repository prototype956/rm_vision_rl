"""Audited checkpoint migrations; each mode keeps its own strict compatibility policy.

Only metadata changes. The recurrent payload (weights, optimizer and RNG state)
is copied byte-for-byte. Curriculum migration additionally re-evaluates frozen scenes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


CLEARANCE_NATIVE_FILES = {"training/environment.rs", "training/mod.rs",
    "training/gimbal_clearance.rs", "robots.rs", "gimbal_actuator.rs"}
WARMUP_SOURCES = {"environment/warmup.py", "environment/fire.py",
    "training/environment.py", "training/recurrent_rollout.py"}


def _check_contract(old, new, *, allow_curriculum=False, label="migration"):
    from src.training.config import ENV_PATHS
    for key in ("observation", "policy_contract", "dependencies"):
        if old[key] != new[key]:
            raise ValueError(label + " changed " + key)
    def normalized(config):
        result = copy.deepcopy(config)
        result.pop("device", None)
        if allow_curriculum:
            result.pop("curriculum", None)
        for key in ENV_PATHS:
            result["environment"].pop(key, None)
        return result
    if normalized(old["config"]) != normalized(new["config"]):
        raise ValueError(label + " changed training configuration, PPO, budget or reward")


def _metadata(checkpoint, destination, config, probe_directory=None, *, previous=None):
    from src.training.environment import make_environment
    from src.training.models import read_checkpoint_metadata, training_metadata
    if destination.exists():
        raise FileExistsError(destination)
    if previous is None:
        previous = read_checkpoint_metadata(checkpoint)
    probe = make_environment(config, probe_directory or destination.parent / "metadata-probe")
    try:
        return previous, training_metadata(config, probe)
    finally:
        probe.close()


def read_payload(checkpoint):
    with ZipFile(checkpoint) as source:
        if sorted(source.namelist()) != ["recurrent.pt", "rmvision.json"]:
            raise ValueError("unexpected checkpoint archive members")
        return source.read("recurrent.pt")


def write_checkpoint(destination, metadata, payload):
    """Write metadata atomically and verify the unchanged opaque training payload."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".new")
    with ZipFile(temporary, "x") as archive:
        archive.writestr("rmvision.json", json.dumps(metadata, indent=2, allow_nan=False))
        archive.writestr("recurrent.pt", payload)
    with ZipFile(temporary) as archive:
        if archive.read("recurrent.pt") != payload:
            raise AssertionError("migration modified recurrent tensor payload")
    temporary.replace(destination)


def _updated_metadata(previous, current, fields=("config", "fingerprints", "sampling", "dependencies")):
    metadata = copy.deepcopy(previous)
    for key in fields:
        metadata[key] = current[key]
    return metadata


def _native_changes(previous, current):
    def hashes(metadata):
        root = Path(metadata["config"]["environment"]["simulator_root"]) / "src"
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*.rs")}
    before, after = hashes(previous), hashes(current)
    return {key: dict(previous_sha256=before.get(key), sha256=after.get(key))
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}


def check_clearance(old, new):
    _check_contract(old, new)
    allowed = {"environment", "native_training", "training_contract"}
    before = {k:v for k,v in old["fingerprints"].items() if k not in allowed}
    after = {k:v for k,v in new["fingerprints"].items() if k not in allowed}
    if before != after:
        raise ValueError("migration changed Python/vision/reward/curriculum fingerprints")


def check_warmup(old, new):
    _check_contract(old, new, label="warmup migration")
    allowed = {"environment", "training_contract", "python_sources"}
    before = {k:v for k,v in old["fingerprints"].items() if k not in allowed}
    after = {k:v for k,v in new["fingerprints"].items() if k not in allowed}
    if before != after:
        raise ValueError("warmup migration changed native/vision/reward/curriculum/optimizer source")
    before, after = old["fingerprints"]["python_sources"], new["fingerprints"]["python_sources"]
    changed = {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    if changed != WARMUP_SOURCES:
        raise ValueError("unexpected Python source changes: " + str(sorted(changed)))
    return {key: dict(previous_sha256=before[key], sha256=after[key]) for key in sorted(changed)}


def check_curriculum(old, new):
    _check_contract(old, new, allow_curriculum=True)
    for key in ("simulator_config","vision_config","reward"):
        if old["fingerprints"][key] != new["fingerprints"][key]:
            raise ValueError("migration changed "+key)
    allowed_python={"training/curriculum.py","training/curriculum_v3.py","training/offline_evaluation.py",
        "training/config.py","training/environment.py","training/models.py","training/recurrent_ppo.py",
        "training/curriculum_evaluation.py","training/recurrent_rollout.py","training/scene_admission.py","environment/rule_fire.py",
        "environment/fire.py","transport/vision_bridge.py","policy/recurrent_deployment.py",
        "training/curriculum_v2.py","training/train.py","environment/multi_robot_fire.py","training/scene_placement.py"}
    before=old["fingerprints"]["python_sources"]
    after=new["fingerprints"]["python_sources"]
    unexpected=[k for k in set(before)|set(after) if before.get(k)!=after.get(k) and k not in allowed_python]
    if unexpected:raise ValueError("unapproved Python changes: "+str(sorted(unexpected)))


def migrate_fix(kind, checkpoint, destination, config):
    """Apply a reviewed native-clearance or visual-warmup fix without retraining."""
    if kind not in ("clearance", "warmup"):
        raise ValueError("unknown fix migration")
    checkpoint, destination = Path(checkpoint).resolve(), Path(destination).resolve()
    previous, current = _metadata(checkpoint, destination, config)
    if kind == "clearance":
        check_clearance(previous, current)
        changes = _native_changes(previous, current)
        unexpected = changes.keys() - CLEARANCE_NATIVE_FILES
        if unexpected:
            raise ValueError("unapproved native source change: " + str(sorted(unexpected)))
        if "training/gimbal_clearance.rs" not in changes:
            raise ValueError("expected native clearance fix not found")
        changed = ["native swept-clearance diagnostics and physical articulated joint contact limit",
                   "actual actuator contact feedback and physical muzzle ordering",
                   "runtime file locations", "native binary fingerprint"]
        receipt_kind = "native_gimbal_clearance_fix_v1"
    else:
        changes = check_warmup(previous, current)
        changed = ["typed visual-initialization timeout",
                   "bounded unfrozen admission retries and durable diagnostics",
                   "worker admission progress messages", "runtime paths"]
        receipt_kind = "visual_admission_timeout_fix_v1"
    payload = read_payload(checkpoint)
    metadata = _updated_metadata(previous, current)
    receipt = dict(kind=receipt_kind, source=str(checkpoint), destination=str(destination),
        source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        recurrent_payload_sha256=hashlib.sha256(payload).hexdigest(), source_changes=changes,
        num_timesteps=previous["num_timesteps"], completed_updates=previous["completed_updates"],
        curriculum_stage=previous["curriculum_state"]["stage"],
        budget=current["config"]["total_timesteps"],
        remaining_budget=current["config"]["total_timesteps"]-previous["num_timesteps"],
        preserved=["actor", "critic", "optimizer", "torch_rng", "cuda_rng", "ppo_rng",
                   "curriculum_rng", "worker_episode_counts", "frozen_scenarios"], changed=changed)
    metadata[kind + "_migration"] = receipt
    write_checkpoint(destination, metadata, payload)
    receipt["destination_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".migration.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def migrate_curriculum(checkpoint, destination, config, frozen, output):
    from src.training.models import read_checkpoint_metadata
    from src.training.recurrent_ppo import RecurrentPPO
    from src.training.curriculum_evaluation import CurriculumEvaluator
    from src.training.curriculum_v3 import Curriculum
    from tools.training.curriculum_v3 import write
    checkpoint,destination,output=Path(checkpoint),Path(destination),Path(output)
    if destination.exists():
        raise FileExistsError(destination)
    previous=read_checkpoint_metadata(checkpoint)
    if previous["config"]["curriculum"]["version"] != 2:
        raise ValueError("migration requires a v2 checkpoint")
    previous,current=_metadata(checkpoint,destination,config,output/"metadata-probe",previous=previous)
    from src.policy.recurrent_actor import CONTRACT
    current["policy_contract"]=CONTRACT
    check_curriculum(previous,current)
    native_changes={key:dict(before=value["previous_sha256"],after=value["sha256"])
                    for key,value in _native_changes(previous,current).items()}
    if set(native_changes)-{"robot_motion.rs","training/environment.rs","training/measurements.rs","training/duel.rs","training/gimbal_clearance.rs","training/scenario.rs","training/mod.rs","training/optical_geometry.rs","training/visibility_admission.rs"}:
        raise ValueError("unapproved native changes: "+str(sorted(native_changes)))
    model=RecurrentPPO.load(checkpoint,device="cpu")
    curriculum=Curriculum(config["curriculum"],config["seed"])
    curriculum.counts=copy.deepcopy(previous["curriculum_state"]["counts"])
    curriculum.rng.bit_generator.state=copy.deepcopy(previous["curriculum_state"]["rng"])
    evaluator=CurriculumEvaluator(config,output,frozen)
    try:
        old_stage=previous["curriculum_state"]["stage"]
        reports={str(c):evaluator.evaluate(model.policy.actor,c,previous["completed_updates"]) for c in range(old_stage+1)}
        if any(r.get("environment_errors") or r.get("incomplete_cases") or r.get("tracking_initialization_failures") for r in reports.values()):
            write(output/"migration-blocked.json",dict(status="blocked",reports=reports))
            raise RuntimeError("migration review has environment or incomplete-settlement failures")
        curriculum.stage=next((c for c in range(old_stage+1) if not reports[str(c)]["passed"]),old_stage)
        curriculum.reports=reports
        curriculum.observe(reports[str(curriculum.stage)])
        current["curriculum_scenarios"]=evaluator.manifest
    finally:evaluator.close()
    tensors=read_payload(checkpoint)
    receipt=dict(version=3,source=str(checkpoint.resolve()),source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        recurrent_payload_sha256=hashlib.sha256(tensors).hexdigest(),
        num_timesteps=previous["num_timesteps"],completed_updates=previous["completed_updates"],
        old_stage=old_stage,new_stage=curriculum.stage,calibration_sha256=config["curriculum"]["calibration"]["sha256"],
        scenarios_sha256=current["curriculum_scenarios"]["sha256"],
        budget=config["total_timesteps"],remaining_budget=config["total_timesteps"]-previous["num_timesteps"],
        preserved=["actor","critic","optimizer","torch_rng","cuda_rng","ppo_rng","worker_episode_counts"],
        changed=["versioned curriculum","scene manifests","baseline thresholds","initial review history"],
        native_source_changes=native_changes,
        diagnostic_compatibility="includes previously authorized replay-only measurement/event plumbing; no Actor schema change",
        source_fingerprints=previous["fingerprints"],destination_fingerprints=current["fingerprints"])
    metadata=_updated_metadata(previous,current,("config","fingerprints","sampling","dependencies","curriculum_scenarios"))
    metadata.update(curriculum_state=curriculum.state(),pending_validation=False,curriculum_migration=receipt)
    write_checkpoint(destination,metadata,tensors)
    receipt["destination_sha256"]=hashlib.sha256(destination.read_bytes()).hexdigest()
    write(destination.with_suffix(".migration.json"),receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("clearance", "warmup", "curriculum"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, help="required for curriculum migration")
    parser.add_argument("--output", type=Path, help="new review directory for curriculum migration")
    args = parser.parse_args()
    from src.training.config import read_config
    if args.kind == "curriculum":
        if args.scenarios is None or args.output is None:
            parser.error("curriculum migration requires --scenarios and --output")
        from tools.training.curriculum_v3 import manifest
        args.output.mkdir(parents=True, exist_ok=False)
        receipt = migrate_curriculum(args.checkpoint, args.destination, read_config(args.config),
                                     manifest(args.scenarios), args.output)
    else:
        if args.scenarios is not None or args.output is not None:
            parser.error("--scenarios and --output apply only to curriculum migration")
        receipt = migrate_fix(args.kind, args.checkpoint, args.destination, read_config(args.config))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
