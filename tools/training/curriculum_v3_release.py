"""Validate release evidence and freeze accepted v3 inputs before a formal run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tools.training.checkpoint_migration import read_payload, write_checkpoint


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def bridge_libraries(binaries):
    """Resolve non-system ELF dependencies without trusting build-tree RUNPATHs."""
    result={}
    for binary in binaries:
        output=subprocess.check_output(['ldd',str(binary)],text=True)
        if 'not found' in output:raise ValueError('bridge shared library is missing')
        for line in output.splitlines():
            if ' => ' not in line:continue
            path=Path(line.split(' => ',1)[1].split(' (',1)[0].strip())
            if not path.is_absolute():continue
            resolved=path.resolve()
            if any(resolved.is_relative_to(Path(root).resolve()) for root in ('/lib','/usr/lib')):
                continue
            if path.name in result and sha(result[path.name])!=sha(path):
                raise ValueError('conflicting bridge shared library names')
            result[path.name]=path
    return result


def verify_bridge_libraries(binaries, directory, expected):
    actual=bridge_libraries(binaries)
    if set(actual)!=set(expected):raise ValueError('frozen bridge dependencies differ')
    for name,path in actual.items():
        if path.resolve().parent!=directory.resolve() or sha(path)!=expected[name]:
            raise ValueError('bridge still resolves outside its frozen library directory')


def freeze(args):
    from src.training.config import ROOT
    from tools.training.curriculum_v3 import calibrated_config,write
    if os.name != "posix":raise RuntimeError("formal v3 training requires Linux")
    for role in ("infantry","sentry"):calibrated_config(args,role)
    workspace=ROOT.parent
    runtime=args.output.with_name(args.output.name+"-runtime")
    runtime.mkdir(parents=True,exist_ok=False)
    files={}
    def save(source,target):
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        files[target.relative_to(runtime).as_posix()]=sha(target)
    for repo,folders in {"rm_vision_rl":("src","tools","native","config","tests","docs"),
                         "rm_simulator_2027":("src","crates","configs","tests"),
                         "rm_vision_2027":("src","docs")}.items():
        for folder in folders:
            for source in (workspace/repo/folder).rglob("*"):
                if source.is_file() and "__pycache__" not in source.parts and source.suffix != ".pyc":
                    save(source,runtime/source.relative_to(workspace))
        for name in ("Cargo.toml","Cargo.lock","CMakeLists.txt","config.toml","README.md","requirements-training.txt"):
            source=workspace/repo/name
            if source.is_file():save(source,runtime/source.relative_to(workspace))
    sim=workspace/"rm_simulator_2027"
    shutil.copytree(sim/"assets",runtime/"rm_simulator_2027/assets")
    for p in (runtime/"rm_simulator_2027/assets").rglob("*"):
        if p.is_file():files[p.relative_to(runtime).as_posix()]=sha(p)
    for source in [sim/"target/release/daedalus_training",*sorted((sim/"target/release/deps").glob("*.so"))]:
        save(source,runtime/source.relative_to(workspace))
    libdir=Path(subprocess.check_output(["rustc","--print","target-libdir"],text=True).strip())
    for source in libdir.glob("*.so"):save(source,runtime/"rm_simulator_2027/target/release/deps"/source.name)
    bridge_binaries=[]
    for name in ("rmvision-rl-bridge","rmvision-rule-bridge"):
        source=ROOT/"artifacts/build/vision-bridge"/name
        bridge_binaries.append(source)
        save(source,runtime/source.relative_to(workspace))
    library_dir=runtime/'rm_vision_rl/artifacts/build/vision-bridge/lib'
    libraries=bridge_libraries(bridge_binaries)
    for name,source in libraries.items():save(source,library_dir/name)
    os.environ['LD_LIBRARY_PATH']=str(library_dir)+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    library_hashes={name:sha(path) for name,path in libraries.items()}
    verify_bridge_libraries([runtime/b.relative_to(workspace) for b in bridge_binaries],library_dir,library_hashes)
    shutil.copytree(Path(sys.prefix),runtime/".venv-gru",symlinks=True)
    for role in ("infantry","sentry"):
        for name in ("config.json","calibration.json"):
            save(args.calibration/role/name,runtime/"inputs/calibration"/role/name)
        for name in ("config.json","curriculum/scenarios.json"):
            save(args.scenarios/role/name,runtime/"inputs/scenarios"/role/name)
    save(args.scenarios/"report.json",runtime/"inputs/scenarios/report.json")
    save(args.calibration/"report.json",runtime/"inputs/calibration/report.json")
    save(args.qualification/"status.json",runtime/"inputs/qualification/status.json")
    save(args.collision_report,runtime/"inputs/collision-report.json")
    save(args.checkpoint,runtime/"inputs/checkpoint.zip")
    release=dict(version=3,status="frozen",runtime_root=str(runtime),files=files,
        bridge_libraries=library_hashes,
        source_checkpoint_sha256=sha(args.checkpoint),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        python_version=sys.version,source_python=sys.executable)
    write(runtime/"release.json",release)
    write(args.output.with_suffix(".sources.json"),release)
    command=[str(runtime/".venv-gru/bin/python"),"-u","-m","tools.training.curriculum_v3","run",
        "--output",str(args.output),"--scenarios",str(runtime/"inputs/scenarios"),
        "--calibration",str(runtime/"inputs/calibration"),"--checkpoint",str(runtime/"inputs/checkpoint.zip"),
        "--qualification",str(runtime/"inputs/qualification"),"--collision-report",str(runtime/"inputs/collision-report.json"),
        "--frozen-runtime"]
    os.chdir(runtime/"rm_vision_rl")
    os.execv(command[0],command)


def verify_and_relocate(args):
    from src.training.config import ROOT,ENV_PATHS
    from src.training.environment import make_environment
    from src.training.models import training_metadata,read_checkpoint_metadata
    from tools.training.curriculum_v3 import read,write
    runtime=ROOT.parent
    release=read(runtime/"release.json")
    if release.get("status") != "frozen" or release.get("runtime_root") != str(runtime):
        raise ValueError("formal runtime has no matching freeze receipt")
    for name,expected in release["files"].items():
        if sha(runtime/name) != expected:raise ValueError("frozen runtime changed: "+name)
    library_dir=ROOT/'artifacts/build/vision-bridge/lib'
    os.environ['LD_LIBRARY_PATH']=str(library_dir)+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    verify_bridge_libraries([ROOT/'artifacts/build/vision-bridge'/name for name in
        ('rmvision-rl-bridge','rmvision-rule-bridge')],library_dir,release.get('bridge_libraries',{}))
    previous=read_checkpoint_metadata(args.checkpoint)
    config=read(args.calibration/"infantry/config.json")
    for key in ENV_PATHS:config["environment"].pop(key,None)
    probe=make_environment(config,args.output/"relocation-probe")
    try:current=training_metadata(config,probe)
    finally:probe.close()
    before,after=copy.deepcopy(previous["config"]),copy.deepcopy(current["config"])
    for value in (before,after):
        for key in ENV_PATHS:value["environment"].pop(key,None)
    if before != after or previous["observation"] != current["observation"] or previous["dependencies"] != current["dependencies"]:
        raise ValueError("runtime relocation changed more than paths")
    a,b=previous["fingerprints"],current["fingerprints"]
    if {k:v for k,v in a.items() if k not in ("environment","training_contract")} != {k:v for k,v in b.items() if k not in ("environment","training_contract")}:
        raise ValueError("runtime relocation changed source/binary/config fingerprints")
    payload=read_payload(args.checkpoint)
    metadata=copy.deepcopy(previous)
    for key in ("config","fingerprints","sampling","dependencies"):metadata[key]=current[key]
    metadata["runtime_relocation"]=dict(source_sha256=sha(args.checkpoint),
        recurrent_payload_sha256=hashlib.sha256(payload).hexdigest(),runtime_root=str(runtime))
    destination=args.output/"relocated-checkpoint.zip"
    write_checkpoint(destination,metadata,payload)
    write(args.output/"relocation.json",metadata["runtime_relocation"])
    args.checkpoint=destination


AUTHORIZED_GATE_SHA256 = "cfbcdc562cfa61eec9494206c4637e78e23f1b43493f7cad400bb86e75ba9dfc"
AUTHORIZED_CALIBRATION_SHA256 = "ccd95cbc2fe9411fab1e1ba0310194aea197fc9db581d97199d5ab2754ba7f01"
AUTHORIZED_BUCKET = ("infantry", "8", "wheel_leg/oblique")
SCOPE = "one-time traditional independent validation release exception"


def validate_exception_report(report, artifacts):
    from src.training.curriculum_v3 import bucket_keys, validate_calibration
    from src.training.curriculum_v2 import digest
    """Reject changed identities, extra failures, incomplete cases or new limits."""
    if report.get("status") != "approved_exception":
        raise ValueError("explicit approved baseline exception required")
    approval = report.get("approval", {})
    if (approval.get("approved") is not True or approval.get("scope") != SCOPE
            or not approval.get("authorization_text") or not approval.get("authorization_source")
            or approval.get("sha256") != digest({k:v for k,v in approval.items() if k != "sha256"})):
        raise ValueError("baseline exception lacks explicit bound authorization")
    original_json = report.get("original_gate_json", "")
    original_sha = hashlib.sha256(original_json.encode()).hexdigest()
    if original_sha != AUTHORIZED_GATE_SHA256 or approval.get("gate_report_sha256") != original_sha:
        raise ValueError("exception does not match the authorized original gate")
    original = json.loads(original_json)
    if original.get("passed") is not False or report.get("roles") != original.get("reports"):
        raise ValueError("original failed baseline results must remain unchanged")
    if set(artifacts) != {"infantry", "sentry"} or set(report["roles"]) != set(artifacts):
        raise ValueError("both role baselines required")
    if artifacts["infantry"]["sha256"] != AUTHORIZED_CALIBRATION_SHA256:
        raise ValueError("authorized calibration identity differs")
    failures = []
    for role, artifact in artifacts.items():
        validate_calibration(artifact)
        if approval.get("calibration_sha256", {}).get(role) != artifact["sha256"]:
            raise ValueError("exception calibration checksum differs")
        stages = report["roles"][role].get("validation", {})
        if set(stages) != {"7", "8"}:
            raise ValueError("complete independent translation validation required")
        for course, stage in stages.items():
            if stage.get("calibration_sha256") != artifact["sha256"]:
                raise ValueError("validation does not belong to the calibrated baseline")
            if set(stage.get("metrics", {})) != set(bucket_keys()):
                raise ValueError("all twelve validation buckets required")
            for bucket, value in stage["metrics"].items():
                limits = artifact["thresholds"][course][bucket]
                if (value.get("cases") != 32 or value.get("invalid_cases") != 0
                        or value.get("actual_shots", 0) <= 0
                        or value.get("baseline") != limits["baseline"]
                        or value.get("required_hit_efficiency") != limits["hit_efficiency"]
                        or value.get("required_kill_rate") != limits["kill_rate"]):
                    raise ValueError("incomplete validation or changed promotion threshold")
                passed = (value["hit_efficiency"] >= limits["hit_efficiency"] - 1e-12
                          and value["kill_rate"] >= limits["kill_rate"] - 1e-12)
                if bool(value["passed"]) != passed:
                    raise ValueError("baseline pass flags disagree with measured scores")
                if not passed:
                    failures.append((role, course, bucket))
                    if value["kill_rate"] < limits["kill_rate"]:
                        raise ValueError("kill-rate failures are not authorized")
    if failures != [AUTHORIZED_BUCKET]:
        raise ValueError("only the single authorized hit-efficiency failure is allowed")
    return approval


def audit_qualification(directory, initial_updates, completed_updates, frozen):
    directory=Path(directory)
    if completed_updates-initial_updates!=2:
        raise ValueError('qualification must contain exactly two new updates')
    checked=0
    for update in range(initial_updates+1,completed_updates+1):
        folder=directory/'curriculum'/f'update-{update:04d}'
        summaries=sorted(folder.glob('C*/summary.json'))
        if not summaries:raise ValueError('qualification update has no validation evidence')
        for path in summaries:
            report=json.loads(path.read_text())
            if (report.get('environment_errors')!=0 or report.get('incomplete_cases')!=0
                    or report.get('tracking_initialization_failures')!=0):
                raise ValueError('qualification validation has environment or settlement failures')
            course=report['course']
            if report['update']!=update or path.parent.name!=f'C{course}':
                raise ValueError('qualification validation identity differs')
            expected={(d['case'],d['seed']) for d in frozen['validation'][course]}
            rows=[json.loads(line) for line in (path.parent/'episodes.jsonl').read_text().splitlines()]
            actual={(r['case'],r['seed']) for r in rows}
            if (not expected or len(expected)!=report['cases'] or len(rows)!=len(expected)
                    or actual!=expected or any(r.get('status')!='complete' or 'error' in r or 'failure' in r for r in rows)):
                raise ValueError('qualification validation coverage is incomplete or invalid')
            checked+=len(rows)
    return dict(validation_environment_errors=0,validation_incomplete_cases=0,
        validation_coverage_verified=True,validation_cases_checked=checked)
