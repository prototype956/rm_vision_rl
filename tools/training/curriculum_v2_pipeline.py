"""Durable server pipeline: qualify, freeze sources, train both roles, compare and report."""
import argparse
import hashlib
import json
import os
import shutil
import signal
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def atomic(path,value):
    temporary = path.with_suffix(".new")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification",type=Path,required=True)
    parser.add_argument("--scenarios",type=Path,required=True)
    parser.add_argument("--loss-regression",type=Path,required=True)
    parser.add_argument("--rule-diagnostics",type=Path,required=True)
    parser.add_argument("--tunnel",type=Path,required=True)
    parser.add_argument("--ready",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    def stop(signum,frame):raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    args = parser.parse_args()
    for name in vars(args):
        if isinstance(getattr(args,name),Path):setattr(args,name,getattr(args,name).resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    status_path = output.with_suffix(".pipeline.json")
    if output.exists() or status_path.exists():raise FileExistsError("pipeline output already exists")
    status = dict(status="waiting_for_qualification",output=str(output),qualification=str(args.qualification.resolve()))
    atomic(status_path,status)
    python = sys.executable
    runtime_cwd = None
    def command(*parts):
        subprocess.run([python,*map(str,parts)],check=True,cwd=runtime_cwd)
    try:
        while True:
            file = args.qualification/"status.json"
            qualification = json.loads(file.read_text()) if file.exists() else {}
            if qualification.get("status") == "failed":raise RuntimeError("qualification failed; formal training not started")
            if qualification.get("status") == "complete" and args.ready.exists():break
            time.sleep(15)
        if any(not qualification["roles"].get(role,{}).get("checkpoint_restored") or qualification["roles"][role]["updates"] < 2 for role in ("infantry","sentry")):
            raise RuntimeError("both roles must pass two complete updates and checkpoint restore")
        loss = json.loads((args.loss_regression/"results.json").read_text())
        if not loss["passed"] or len(loss["cases"]) != 16:raise RuntimeError("physical blackout regression incomplete")
        for directory,count in ((args.rule_diagnostics,96),(args.tunnel,40)):
            rows = json.loads((directory/"results.json").read_text())
            if len(rows) != count or any("error" in row for row in rows):
                raise RuntimeError("diagnostic environment error; formal training not started")
        ready = json.loads(args.ready.read_text())
        workspace_for_check = Path(__file__).resolve().parents[3]
        for relative,expected in ready["file_hashes"].items():
            if hashlib.sha256((workspace_for_check/relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("source changed after qualification release: "+relative)
        status.update(status="final_regressions")
        atomic(status_path,status)
        command("-m","unittest","discover","-s","tests","-v")
        # Only source/configuration directories are archived; no credentials, models or logs.
        root = Path(__file__).resolve().parents[2]
        workspace = root.parent
        git_sources = {}
        for repo in ("rm_vision_rl","rm_simulator_2027","rm_vision_2027"):
            branch = subprocess.check_output(["git","-C",str(workspace/repo),"branch","--show-current"],text=True).strip()
            if branch != "RL_fire":raise RuntimeError(f"{repo} is on {branch}, expected RL_fire")
            git_sources[repo] = dict(branch=branch,head=subprocess.check_output(["git","-C",str(workspace/repo),"rev-parse","HEAD"],text=True).strip())
        source_files = []
        for repo,folders in (("rm_vision_rl",("src","native","config","tools","tests","docs")),
                             ("rm_simulator_2027",("src","configs","tests","examples","crates")),
                             ("rm_vision_2027",("src","docs"))):
            for folder in folders:
                source_files.extend(p for p in (workspace/repo/folder).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
            for name in ("Cargo.toml","Cargo.lock","CMakeLists.txt","README.md","requirements-training.txt","config.toml"):
                p = workspace/repo/name
                if p.is_file():source_files.append(p)
        hashes = {str(p.relative_to(workspace)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(source_files))}
        freeze = dict(source_git=git_sources,source_sha256=hashes,configuration={role:json.loads((root/f"config/training/rmul_2027_{role}_curriculum_v2.json").read_text()) for role in ("infantry","sentry")},
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),qualification=qualification)
        atomic(output.with_suffix(".sources.json"),freeze)
        with tarfile.open(output.with_suffix(".sources.tar.gz"),"w:gz") as archive:
            for p in sorted(set(source_files)):archive.add(p,arcname=str(p.relative_to(workspace)))
        # Run from a frozen copy so later edits/rebuilds in the shared workspace do
        # not change multi-week experiments. Existing working trees remain editable.
        runtime_root = output.with_name(output.name+"-runtime")
        runtime_root.mkdir(exist_ok=False)
        for p in sorted(set(source_files)):
            target = runtime_root/p.relative_to(workspace)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
        sim_source = workspace/"rm_simulator_2027"
        sim_target = runtime_root/"rm_simulator_2027"
        shutil.copytree(sim_source/"assets",sim_target/"assets")
        native_files = [sim_source/"target/release/daedalus_training",*sorted((sim_source/"target/release/deps").glob("*.so"))]
        for p in native_files:
            target = runtime_root/p.relative_to(workspace)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
        rust_libdir = Path(subprocess.check_output(["rustc","--print","target-libdir"],text=True).strip())
        freeze["rust_shared_libraries"] = {}
        for library in rust_libdir.glob("*.so"):
            shutil.copy2(library,sim_target/"target/release/deps"/library.name)
            freeze["rust_shared_libraries"][library.name] = hashlib.sha256(library.read_bytes()).hexdigest()
        for variant in ("vision-bridge","rule-comparison"):
            source = root/f"artifacts/build/{variant}/rmvision-rl-bridge"
            target = runtime_root/source.relative_to(workspace)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            native_files.append(source)
        shutil.copytree(root/".venv-gru",runtime_root/".venv-gru",symlinks=True)
        python = str(runtime_root/".venv-gru/bin/python")
        runtime_cwd = runtime_root/"rm_vision_rl"
        freeze["runtime_root"] = str(runtime_root)
        freeze["native_sha256"] = {str(p.relative_to(workspace)):hashlib.sha256(p.read_bytes()).hexdigest() for p in native_files}
        freeze["assets_sha256"] = {str(p.relative_to(workspace)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((sim_source/"assets").rglob("*")) if p.is_file()}
        # Select a currently idle device without taking another task's reserved GPU.
        preferred = os.environ.get("CUDA_VISIBLE_DEVICES","")
        while True:
            raw = subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True)
            available = [fields[0].strip() for line in raw.splitlines() if len(fields:=line.split(",")) == 3
                         and int(fields[1]) < 512 and int(fields[2]) < 5]
            if available:break
            status.update(status="waiting_for_free_gpu")
            atomic(status_path,status)
            time.sleep(15)
        os.environ["CUDA_VISIBLE_DEVICES"] = preferred if preferred in available else available[0]
        freeze["cuda_visible_devices"] = os.environ["CUDA_VISIBLE_DEVICES"]
        freeze["runtime_python"] = python
        atomic(output.with_suffix(".sources.json"),freeze)
        command("-m","tools.training.curriculum_v2","check-snapshot","--scenarios",args.scenarios,
                "--output",output.with_name(output.name+"-snapshot-check"))
        status.update(status="training",source_manifest=str(output.with_suffix(".sources.json")))
        atomic(status_path,status)
        command("-u","-m","tools.training.curriculum_v2","run","--scenarios",args.scenarios,"--output",output)
        result = json.loads((output/"status.json").read_text())
        if result["status"] != "complete":raise RuntimeError("training paused/failed; retain checkpoint and actual budget")
        command("-m","tools.training.curriculum_report","--run",output)
        status.update(status="final_comparison")
        atomic(status_path,status)
        command("-u","-m","tools.training.curriculum_v2_diagnostics","--scenarios",args.scenarios,
                "--output",output/"comparison","--checkpoint-root",output,"--all","--record","--workers","8")
        command("-m","tools.training.curriculum_report","--run",output)
        status.update(status="server_complete",nuc_deployment="pending local delivery",budget_per_role=1_000_000_000)
    except BaseException as error:
        status.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic(status_path,status)


if __name__ == "__main__":main()
