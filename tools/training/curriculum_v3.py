"""V3 scene admission/repair, traditional calibration and qualified training."""
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def write(path, value):
    from tools.training.view import _atomic_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, value)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest(path):
    from src.training.curriculum_v2 import digest
    value = read(path)
    if value.get("version") != 3 or value.get("sha256") != digest({k:v for k,v in value.items() if k != "sha256"}):
        raise ValueError("invalid frozen v3 scene manifest")
    return value


def _job(job):
    config, descriptor, kind, source, output, preflight = job
    if preflight:
        from src.training.curriculum_evaluation import _case
        row = _case((config, descriptor, None, output, True))
    else:
        from src.training.offline_evaluation import evaluate_scene
        try:
            row = evaluate_scene(config, descriptor, output, kind=kind, source=source)
        except Exception as error:
            row = dict(course=descriptor["course"], group=descriptor["group"], case=descriptor["case"],
                seed=descriptor["seed"], profile=descriptor["scenario"]["duel"]["target"]["profile"],
                motion_type=descriptor.get("motion_type"), status="failed", actual_shots=0,
                damaging_hits=0, end_reason="environment_error", error=f"{type(error).__name__}: {error}")
    write(Path(output)/"result.json", row)
    return row


def evaluate(config, scenes, output, workers, *, kind="traditional", source=None, preflight=False):
    jobs = [(config,d,kind,source,Path(output)/f"case-{d['case']:03d}",preflight) for d in scenes]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for row in pool.map(_job, jobs):
            rows.append(row)
            write(Path(output)/"results.json", rows)
            print(json.dumps(dict(phase="preflight" if preflight else kind, course=row["course"],
                case=row["case"], completed=len(rows), total=len(jobs), error=(row.get("error",row.get("failure")) or "")[:240])),flush=True)
    return rows


def preflight(args):
    from src.training.config import ROOT, read_config
    from src.training.curriculum_v3 import bank, TRANSLATION, digest, scene
    all_failures = {}
    for role in args.roles:
        config = read_config(ROOT/f"config/training/rmul_2027_{role}_curriculum_v3.json")
        data = dict(version=3, **{s:bank(config["environment"]["scenario"]["duel"]["controlled"]["preset"],
            config["curriculum"],split=s) for s in ("calibration","validation","test")})
        if args.legacy_scenarios:
            legacy = read(Path(args.legacy_scenarios)/role/"curriculum/scenarios.json")
            if legacy.get("sha256") != digest({k:v for k,v in legacy.items() if k != "sha256"}):
                raise ValueError("legacy scene manifest checksum mismatch")
            for split in ("validation", "test"):
                for course in range(len(data[split])):
                    if course not in TRANSLATION or legacy.get("version")==3:
                        data[split][course] = [dict(copy.deepcopy(d),version=3,release_revision=config["curriculum"]["release_revision"],admitted=True) for d in legacy[split][course]]
                        for descriptor in data[split][course]:
                            descriptor.pop("visibility_certificate",None)
                            if config["curriculum"]["release_revision"]>=3:descriptor["scenario"]["visibility_admission"]="full_path_clear"
            if legacy.get("version")==3 and "calibration" in legacy:
                data["calibration"]=copy.deepcopy(legacy["calibration"])
                for scenes in data["calibration"]:
                    for descriptor in scenes:
                        descriptor.update(release_revision=config["curriculum"]["release_revision"],admitted=True)
                        descriptor.pop("visibility_certificate",None)
                        if config["curriculum"]["release_revision"]>=3:descriptor["scenario"]["visibility_admission"]="full_path_clear"
        failures = []
        historical = []
        for split in ("calibration", "validation", "test"):
            for course, scenes in enumerate(data[split]):
                if not scenes:
                    continue
                if args.reuse_admissions:
                    from src.training.curriculum_v3 import scene
                    for i,descriptor in enumerate(scenes):
                        cached=args.reuse_admissions/role/split/f"C{course}"/f"case-{descriptor['case']:03d}"/"result.json"
                        if not cached.exists():continue
                        previous=read(cached).get("admitted_scene")
                        if previous is None:continue
                        expected=scene(course,descriptor["seed"],config["environment"]["scenario"]["duel"]["controlled"]["preset"],
                            descriptor["case"],previous["attempt"],strata=descriptor["strata"],release_revision=config["curriculum"]["release_revision"])
                        if expected["scenario"]==previous["scenario"]:
                            scenes[i]=dict(expected,admitted=True)
                rows = evaluate(config, scenes, args.output/role/split/f"C{course}", args.workers, preflight=True)
                renew=[]
                if args.renew_invalid_placements:
                    for descriptor,row in zip(scenes,rows):
                        reason=row.get("error",row.get("failure",""))
                        placement_error=("invalid explicit duel:" in reason or "invalid seed " in reason)
                        if config["curriculum"].get("release_revision",1)<3:
                            placement_error=placement_error or "tracking initialization" in reason.lower() or "warmup timed out" in reason.lower()
                        if descriptor.get("admitted") and row.get("phase")=="reset" and placement_error:
                            historical.append(dict(split=split,course=course,case=descriptor["case"],
                                descriptor=copy.deepcopy(descriptor),error=reason))
                            renewed=copy.deepcopy(descriptor);renewed.pop("admitted",None)
                            renew.append(renewed)
                    if renew:
                        replacements=evaluate(config,renew,args.output/role/split/"renewed"/f"C{course}",args.workers,preflight=True)
                        by_case={r["case"]:r for r in replacements}
                        rows=[by_case.get(row["case"],row) for row in rows]
                for i,row in enumerate(rows):
                    if "error" in row or "failure" in row:
                        scenes[i]=dict(scenes[i],admitted=False)
                        scenes[i].pop("visibility_certificate",None)
                        failures.append(dict(split=split,course=course,case=row["case"],error=(row.get("error",row.get("failure")) or "")[:240]))
                    else:
                        scenes[i] = dict(row["admitted_scene"], admitted=True)
                for item in historical:
                    if item["split"]==split and item["course"]==course:
                        replacement=next((s for s in scenes if s["case"]==item["case"] and s.get("admitted")),None)
                        if replacement is not None:
                            from src.training.scene_admission import difficulty_signature
                            if difficulty_signature(item["descriptor"])!=difficulty_signature(replacement):
                                raise RuntimeError("refreeze changed task difficulty")
                            item.update(replacement=copy.deepcopy(replacement),difficulty_unchanged=True)
                write(args.output/role/"preflight.json",dict(passed=not failures,failures=failures,historical_frozen_failures=historical))
        all_failures[role] = failures
        data["sha256"] = digest(data)
        write(args.output/role/"curriculum/scenarios.json",data)
        write(args.output/role/"config.json",config)
    write(args.output/"report.json",dict(status="complete" if not any(all_failures.values()) else "blocked",roles=all_failures))
    if any(all_failures.values()):raise RuntimeError("v3 scene admission failed; exact frozen failures retained")


def provenance(config, frozen):
    from src.training.environment import make_environment
    from src.training.models import training_metadata
    from src.training.config import ROOT
    env = make_environment(config, ROOT/"artifacts/metadata-probe")
    try:
        fp = training_metadata(config,env)["fingerprints"]
    finally:
        env.close()
    keys = ("simulator_config","vision_config","native_training","vision_bridge","python_sources")
    binary = ROOT/"artifacts/build/vision-bridge/rmvision-rule-bridge"
    result=dict(scenarios_sha256=frozen["sha256"],
        fingerprints={k:fp[k] for k in keys}, rule_bridge_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    if config["curriculum"].get("release_revision",1)>=3:result.update(visibility_admission="full_path_clear",release_revision=3)
    return result


def calibrate(args):
    from src.training.curriculum_v3 import TRANSLATION, build_calibration, assess
    reports = {}
    for role in args.roles:
        config = read(args.scenarios/role/"config.json")
        frozen = manifest(args.scenarios/role/"curriculum/scenarios.json")
        identity=provenance(config,frozen)
        if config["curriculum"].get("release_revision",1)>=3:
            from src.training.scene_admission import validate_visibility_certificate
            for split in ("calibration","validation","test"):
                for scenes in frozen[split]:
                    for descriptor in scenes:validate_visibility_certificate(descriptor,identity["fingerprints"]["native_training"],identity["fingerprints"]["simulator_config"])
        if not all(d.get("admitted") for c in TRANSLATION for d in frozen["calibration"][c]+frozen["validation"][c]):
            reports[role]=dict(passed=False,error="translation scenes have not passed admission")
            write(args.output/"report.json",dict(status="blocked",roles=reports))
            continue
        role_name = config["environment"]["scenario"]["duel"]["controlled"]["preset"]
        rows = {str(c):evaluate(config,frozen["calibration"][c],args.output/role/"calibration"/f"C{c}",args.workers) for c in TRANSLATION}
        try:
            artifact = build_calibration(role_name,rows,identity,cases_per_bucket=config["curriculum"]["translation_cases"]//12)
            write(args.output/role/"candidate-calibration.json",artifact)
            validation = {}
            for course in TRANSLATION:
                values = evaluate(config,frozen["validation"][course],args.output/role/"validation"/f"C{course}",args.workers)
                validation[str(course)] = assess(course,values,artifact)
            passed = all(r["passed"] for r in validation.values())
            reports[role] = dict(passed=passed, validation=validation)
            if passed:
                config["curriculum"]["calibration"] = artifact
                write(args.output/role/"calibration.json",artifact)
                write(args.output/role/"config.json",config)
        except Exception as error:
            reports[role] = dict(passed=False,error=f"{type(error).__name__}: {error}")
        write(args.output/"report.json",dict(status="complete" if len(reports)==2 and all(r["passed"] for r in reports.values()) else "blocked",roles=reports))
    if not all(r["passed"] for r in reports.values()):
        raise RuntimeError("traditional calibration or validation failed; no release")


def calibrated_config(args, role):
    from src.training.curriculum_v3 import validate_calibration
    config = read(args.calibration/role/"config.json")
    preflight_report = read(args.scenarios/"report.json")
    if preflight_report["status"] != "complete" or set(preflight_report["roles"]) != {"infantry","sentry"}:
        raise ValueError("full v3 preflight must pass before training")
    artifact = config["curriculum"]["calibration"]
    validate_calibration(artifact)
    frozen = manifest(args.scenarios/role/"curriculum/scenarios.json")
    if config["curriculum"].get("release_revision",1)>=3:
        from src.training.scene_admission import validate_visibility_certificate
        for split in ("calibration","validation","test"):
            for scenes in frozen[split]:
                for descriptor in scenes:validate_visibility_certificate(descriptor,artifact["provenance"]["fingerprints"]["native_training"],artifact["provenance"]["fingerprints"]["simulator_config"])
    if artifact["role"] != config["environment"]["scenario"]["duel"]["controlled"]["preset"] or artifact["provenance"] != provenance(config,frozen):
        raise ValueError("calibration source/config/binary/scenario identity differs")
    report = read(args.calibration/"report.json")
    if report["status"] == "approved_exception":
        from tools.training.curriculum_v3_release import validate_exception_report
        artifacts={r:read(args.calibration/r/"calibration.json") for r in ("infantry","sentry")}
        if artifacts[role] != artifact:
            raise ValueError("config calibration differs from approved baseline artifact")
        validate_exception_report(report,artifacts)
    elif report["status"] != "complete":
        raise ValueError("both roles must pass frozen traditional validation")
    return config,frozen


def check_release(args):
    collision = read(args.collision_report) if args.collision_report else {}
    qualification = read(args.qualification/"status.json") if args.qualification else {}
    if collision.get("passed") is not True or collision.get("original_reproduced") is not True or qualification.get("status") != "complete":
        raise ValueError("run requires successful original/fixed collision regression and both-role qualification")
    if any(qualification.get("roles",{}).get(r,{}).get("new_updates") != 2 or not qualification["roles"][r].get("checkpoint_restored") or not qualification["roles"][r].get("actor_exported") for r in ("infantry","sentry")):
        raise ValueError("both roles must restore/export after exactly two new updates")
    if any(qualification["roles"][r].get("validation_coverage_verified") is not True
           or qualification["roles"][r].get("validation_environment_errors") != 0
           or qualification["roles"][r].get("validation_incomplete_cases") != 0 for r in ("infantry","sentry")):
        raise ValueError("qualification must verify full validation coverage without environment errors")
    if qualification.get("calibration_report_sha256") != hashlib.sha256((args.calibration/"report.json").read_bytes()).hexdigest():
        raise ValueError("qualification calibration differs")
    from src.training.curriculum_v2 import digest
    for role in ("infantry","sentry"):
        config,frozen=calibrated_config(args,role)
        if collision.get("native_training_sha256") != config["curriculum"]["calibration"]["provenance"]["fingerprints"]["native_training"]:
            raise ValueError("collision regression binary differs from calibration")
        receipt=qualification["roles"][role]
        if receipt.get("config_sha256") != digest(config) or receipt.get("scenarios_sha256") != frozen["sha256"]:
            raise ValueError("qualification config or scenarios differ")
    from src.training.models import read_checkpoint_metadata
    metadata=read_checkpoint_metadata(args.checkpoint)
    checkpoint_hash=metadata.get("runtime_relocation",{}).get("source_sha256") or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if qualification.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("qualification checkpoint differs")


def train(args):
    from src.training.recurrent_ppo import run_recurrent_training, RecurrentPPO
    from src.training.models import read_checkpoint_metadata
    if args.mode == "run":check_release(args)
    roles = {}
    for role in ("infantry","sentry"):
        config,frozen = calibrated_config(args,role)
        checkpoint = args.checkpoint if role == "infantry" else None
        if checkpoint:
            metadata = read_checkpoint_metadata(checkpoint)
            if metadata["config"]["curriculum"] != config["curriculum"] or metadata.get("curriculum_migration",{}).get("version") != 3:
                raise ValueError("explicit matching v3 migration checkpoint is required")
        write(args.output/"status.json",dict(status="running",active_role=role,mode=args.mode,roles=roles))
        directory = run_recurrent_training(config,resume=checkpoint,output_dir=args.output/role,
            frozen_scenarios=frozen,stop_after_updates=2 if args.mode == "benchmark" else None)
        run = read(directory/"run.json")
        if args.mode == "benchmark":
            model = RecurrentPPO.load(directory/"paused.zip")
            initial_updates = metadata["completed_updates"] if checkpoint else 0
            if model.completed_updates-initial_updates != 2 or model.num_timesteps != run["last_completed_timesteps"]:
                raise RuntimeError("qualification did not restore exactly two new updates")
            from src.training.curriculum_v2 import digest
            roles[role] = dict(status="complete", steps=model.num_timesteps, updates=model.completed_updates,
                config_sha256=digest(config),scenarios_sha256=frozen["sha256"],
                new_updates=2,checkpoint_restored=True,actor_exported=(directory/"actor/actor.pt").is_file())
            if not roles[role]["actor_exported"]:
                raise RuntimeError("qualification Actor export missing")
            from tools.training.curriculum_v3_release import audit_qualification
            roles[role].update(audit_qualification(directory,initial_updates,model.completed_updates,frozen))
        else:
            roles[role] = dict(status=run["status"],steps=run["last_completed_timesteps"])
            if run["status"] != "complete":
                raise RuntimeError("formal role did not complete its cumulative budget")
        write(args.output/"status.json",dict(status="complete" if len(roles)==2 else "running",roles=roles,
            calibration_report_sha256=hashlib.sha256((args.calibration/"report.json").read_bytes()).hexdigest(),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("preflight","repair","calibrate","benchmark","run"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scenarios",type=Path)
    parser.add_argument("--legacy-scenarios",type=Path)
    parser.add_argument("--reuse-admissions",type=Path)
    parser.add_argument("--renew-invalid-placements",action="store_true")
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--qualification",type=Path)
    parser.add_argument("--collision-report",type=Path)
    parser.add_argument("--roles",nargs="+",choices=("infantry","sentry"),default=("infantry","sentry"))
    parser.add_argument("--workers",type=int,default=16)
    parser.add_argument("--frozen-runtime",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    if args.mode != "preflight" and not args.scenarios:
        parser.error("--scenarios is required")
    if args.mode in ("benchmark","run") and (not args.calibration or not args.checkpoint):
        parser.error("--calibration and --checkpoint are required")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key]="1"
    os.environ["RM_BOUNDED_NATIVE_LOGS"]="1"
    os.environ["RM_TRAINING_THREADS"]="1"
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=args.frozen_runtime)
    try:
        if args.mode == "preflight":preflight(args)
        elif args.mode == "repair":
            from tools.training.repair_scene_admissions import repair
            repair(args)
        elif args.mode == "calibrate":calibrate(args)
        else:
            if args.mode == "run":
                from tools.training.curriculum_v3_release import freeze,verify_and_relocate
                check_release(args)
                if not args.frozen_runtime:freeze(args)
                verify_and_relocate(args)
            train(args)
        if args.mode in ("preflight","repair"):write(args.output/"status.json",dict(status="complete"))
    except BaseException as error:
        write(args.output/"failure.json",dict(status="failed",error=f"{type(error).__name__}: {error}"))
        if args.mode in ("benchmark","run"):
            write(args.output/"status.json",dict(status="failed",error=f"{type(error).__name__}: {error}"))
        raise

if __name__ == "__main__":main()
