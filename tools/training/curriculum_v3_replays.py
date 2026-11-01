"""Finite paired T1/T2 exports: first frozen validation case per profile/motion bucket."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tools.training.curriculum_v3 import read,write,manifest


def job(value):
    config,descriptor,kind,checkpoint,output,model,combat_hud=value
    from src.training.offline_evaluation import evaluate_scene
    try:
        if not descriptor.get("admitted"):
            raise ValueError("selected frozen scene has not passed admission")
        row=evaluate_scene(config,descriptor,output,kind=kind,source=checkpoint,
            model=model,record_mcap=True,combat_hud=combat_hud)
        return dict(status=row["status"],controller=kind,scene=descriptor,path=str(output/"replay.mcap"),summary=row["summary"])
    except Exception as error:
        row=dict(status="failed",controller=kind,scene=descriptor,error=f"{type(error).__name__}: {error}")
        write(output/"failure.json",row)
        return row


def main():
    from src.training.curriculum_v3 import TRANSLATION,PROFILES,MOTIONS
    from src.training.models import read_checkpoint_metadata
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--combat-hud",action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
    a.output.mkdir(parents=True,exist_ok=False)
    snapshot=a.output/"snapshot.zip";snapshot.write_bytes(a.checkpoint.read_bytes())
    metadata=read_checkpoint_metadata(snapshot)
    role="sentry" if metadata["config"]["environment"]["scenario"]["duel"]["controlled"]["preset"]=="sentry" else "infantry"
    frozen=manifest(a.scenarios/role/"curriculum/scenarios.json")
    config=read(a.scenarios/role/"config.json")
    model=dict(kind="checkpoint",sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
               fingerprints=metadata["fingerprints"],num_timesteps=metadata["num_timesteps"])
    jobs=[]
    for course in TRANSLATION:
        for profile in PROFILES:
            for motion in MOTIONS:
                d=next(d for d in frozen["validation"][course] if d["scenario"]["duel"]["target"]["profile"]==profile and d["motion_type"]==motion)
                for kind in ("checkpoint","traditional"):
                    identity=model if kind=="checkpoint" else dict(kind="traditional",controller="native RulePolicy + shared DecisionClock")
                    jobs.append((config,d,kind,snapshot,a.output/f"C{course}-{profile}-{motion}-{kind}",identity,a.combat_hud))
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context("spawn")) as pool:
        for row in pool.map(job,jobs):
            rows.append(row)
            write(a.output/"replay_index.json",dict(status="running",scenarios_sha256=frozen["sha256"],scenes=rows))
            print(json.dumps(dict(completed=len(rows),total=len(jobs),status=row["status"])),flush=True)
    passed=all(r["status"]=="complete" for r in rows)
    write(a.output/"replay_index.json",dict(status="complete" if passed else "failed",scenarios_sha256=frozen["sha256"],scenes=rows))
    if not passed:raise RuntimeError("paired replay export incomplete")

if __name__ == "__main__":main()
