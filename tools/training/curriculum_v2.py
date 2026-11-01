"""Server-only v2 preflight, 128-environment qualification, and sequential training."""
import argparse
import json
import os
import signal
from pathlib import Path
from time import perf_counter


def run(args):
    from src.training.config import ROOT,read_config
    from src.training.curriculum_evaluation import CurriculumEvaluator
    from src.training.recurrent_ppo import run_recurrent_training,RecurrentPPO
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(mode=args.mode,status="running",roles={})
    def publish():
        temporary = args.output/"status.new"
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        temporary.replace(args.output/"status.json")
    publish()
    try:
        for role in ("infantry","sentry"):
            config = read_config(ROOT/f"config/training/rmul_2027_{role}_curriculum_v2.json")
            started = perf_counter()
            report["active_role"] = role
            publish()
            if args.mode == "preflight":
                evaluator = CurriculumEvaluator(config,args.output/role)
                try:
                    results = evaluator.preflight()
                    report["roles"][role] = dict(passed=all(r["passed"] for r in results),seconds=perf_counter()-started)
                finally:
                    evaluator.close()
            else:
                if args.scenarios is None:
                    raise ValueError("benchmark/run require the admitted preflight directory")
                frozen = json.loads((args.scenarios/role/"curriculum/scenarios.json").read_text())
                if frozen["version"] != 2 or not all(d.get("admitted") for bank in (frozen["validation"],frozen["test"]) for stage in bank for d in stage):
                    raise ValueError("v2 scenario bank has not passed full preflight")
                directory = run_recurrent_training(config,output_dir=args.output/role,
                    stop_after_updates=2 if args.mode == "benchmark" else None,frozen_scenarios=frozen)
                metadata = json.loads((directory/"run.json").read_text())
                report["roles"][role] = dict(status=metadata["status"],steps=metadata["last_completed_timesteps"],
                    seconds=perf_counter()-started,updates=metadata["completed_updates"])
                if args.mode == "benchmark":
                    model = RecurrentPPO.load(directory/"paused.zip")
                    if model.num_timesteps != metadata["last_completed_timesteps"] or model.completed_updates < 2:
                        raise RuntimeError("benchmark checkpoint counters did not restore")
                    report["roles"][role]["checkpoint_restored"] = True
                    report["roles"][role]["remaining_budget"] = config["total_timesteps"]-model.num_timesteps
                    del model
                elif metadata["status"] != "complete":
                    report["status"] = metadata["status"]
                    publish()
                    return
            publish()
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        publish()


def check_snapshot(args):
    """Smoke-check frozen paths, physical clock and initial heat without training."""
    from src.training.config import ROOT, read_config
    from src.training.environment import make_environment
    args.output.mkdir(parents=True,exist_ok=False)
    rows = []
    for role in ("infantry","sentry"):
        config = read_config(ROOT/f"config/training/rmul_2027_{role}_curriculum_v2.json")
        scenes = json.loads((args.scenarios/role/"curriculum/scenarios.json").read_text())["validation"]
        for course in (0,10):
            env = make_environment(config,args.output/f"{role}-{course}")
            try:
                obs,info = env.reset(options={"curriculum_scene":scenes[course][0]})
                for name in ("simulator_root","vision_root","simulator_binary","bridge_binary","simulator_config"):
                    if not Path(getattr(env.unwrapped,name)).is_relative_to(ROOT.parent):
                        raise AssertionError("runtime path escaped frozen snapshot: "+name)
                initial = info["physical_time_ns"]
                for _ in range(2):obs,_,_,_,info = env.step(0)
                if info["physical_time_ns"] != initial+20_000_000:raise AssertionError("control clock changed")
                if info["physical_thermal"]["ordinary_lock_count"] != 0:raise AssertionError("unexpected initial thermal event")
                rows.append(dict(role=role,course=course,passed=True))
            finally:env.close()
    (args.output/"results.json").write_text(json.dumps(dict(passed=True,cases=rows),indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("preflight","benchmark","run","check-snapshot"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scenarios",type=Path)
    args = parser.parse_args()
    # These must be set before importing numpy/torch or spawning native workers.
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["RM_TRAINING_THREADS"] = "1"
    os.environ["RUST_LOG"] = "error"
    os.environ["RM_BOUNDED_NATIVE_LOGS"] = "1"
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if args.mode == "check-snapshot":
        if args.scenarios is None:
            parser.error("check-snapshot requires --scenarios")
        check_snapshot(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
