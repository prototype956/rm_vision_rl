"""Traditional baseline diagnostics and independent tunnel exit replay.

Uses a frozen scene bank; controller performance never edits or filters scenes.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from src.environment.rule_fire import RuleEnv
from src.policy.recurrent_actor import RecurrentActor
from src.policy.recurrent_observations import ACTOR_KEYS
from src.training.config import ROOT,ENV_PATHS,read_config
from src.training.environment import CurriculumScenario,FixedScenario,TrainingReward
from src.training.curriculum_v2 import assess
from src.training.models import read_checkpoint_metadata


def tunnel_bank(role):
    """Five legal diagonal exit anchors per compact profile and controlled role."""
    rows = []
    # Observe the left exit; initialize on this visible side, wait through the far-side traversal.
    anchors = ((-1.4,3.20),(-1.2,3.25),(-1.,3.30),(-.8,3.35),(-.6,3.40))
    for profile in ("low","folding"):
        for index,position in enumerate(anchors):
            scenario = dict(duel=dict(controlled=dict(profile="standard",preset=role,
                position_xz_m=list(position),motion=dict(kind="static")),
                target=dict(profile=profile,preset="infantry_health_cooling",position_xz_m=[2.1,3.46],yaw_rad=0.,
                    motion=dict(kind="tunnel",amplitude_m=.5,speed_m_s=.5,angular_speed_rad_s=0.))),
                unlimited_heat=False,measurements=dict(noise_std_px=.25,latency_ms=20,dropout_probability=0.),
                policy_start=dict(minimum_warmup_ms=4500))
            # Standalone terrain diagnostics have no curriculum stage or version.
            rows.append(dict(course=None,group="tunnel_exit",case=len(rows),
                seed=81000+len(rows),scenario=scenario,scarce=False,damage_capacity=None))
    return rows


def evaluate(job):
    config,descriptor,controller,checkpoint,directory,record = job
    torch.set_num_threads(1)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    settings = config["environment"]
    bridge = ROOT/"artifacts/build/rule-comparison/rmvision-rl-bridge"
    kwargs = {k:settings[k] for k in ENV_PATHS if k in settings}
    kwargs.update(bridge_binary=bridge,episode_steps=3000,observation_mode="asymmetric_gru_v1",
                  action_mode="joint",selection_memory=True,decision_clock=settings["decision_clock"],log_dir=directory/"environment")
    if descriptor["group"] == "tunnel_exit":
        kwargs["decision_clock"] = copy.deepcopy(kwargs["decision_clock"])
        kwargs["decision_clock"]["seed"] = (descriptor["seed"] ^ 0x53484F54) % (2**32)
    training_steps = 0
    checkpoint_sha256 = None
    if controller == "traditional":
        base = RuleEnv(**kwargs)
        actor = state = None
    else:
        from src.environment.multi_robot_fire import MultiRobotFireEnv
        from zipfile import ZipFile
        from io import BytesIO
        base = MultiRobotFireEnv(**kwargs)
        with ZipFile(checkpoint) as archive:
            payload = torch.load(BytesIO(archive.read("recurrent.pt")),map_location="cpu",weights_only=True)
        training_steps = read_checkpoint_metadata(checkpoint)["num_timesteps"]
        checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        actor = RecurrentActor().eval()
        actor.load_state_dict({k.removeprefix("actor."):v for k,v in payload["policy"].items() if k.startswith("actor.")})
        state = actor.initial_state(1)
    warmup_probes = []
    if descriptor["group"] == "tunnel_exit":
        original_start = base._start
        def start_with_probes():
            original_start()
            original_advance = base._client.advance
            def advance_with_probe(**command):
                result = original_advance(**command)
                if result["sim_time_ns"] % 100_000_000 == 0:
                    from src.training.scene_admission import check_fireability
                    physical = base._client.inspect()["data"]["evaluation"]["fireability"]
                    try:check_fireability(physical); clear=True
                    except RuntimeError:clear=False
                    target = next(r for r in result["data"]["evaluation"]["robots"] if r["robot_id"] == 2)
                    warmup_probes.append(dict(round_time_s=result["sim_time_ns"]*1e-9,
                        common_visible_ballistic_clear=clear,target_bevy_m=target["position_bevy_m"]))
                return result
            base._client.advance = advance_with_probe
        base._start = start_with_probes
    base.detailed_logging = False
    base._spawn_attempts = 1
    # Independent tunnel probes use fixed scenarios; training scenes retain v2 admission.
    wrapper = FixedScenario(base,descriptor["seed"],descriptor["scenario"]) if descriptor["group"] == "tunnel_exit" else CurriculumScenario(base,settings["scene_seed"],settings["scenario"])
    env = TrainingReward(wrapper,config["reward_scale"],
        selection_reward=config["selection_reward"],thermal_reward=config["thermal_reward"])
    frames,probes = [],[]
    row = dict(role="sentry" if settings["scenario"]["duel"]["controlled"]["preset"] == "sentry" else "infantry",
        seed=descriptor["seed"],stage_id=descriptor.get("stage_id","tunnel_exit"),
        controller=controller,course=descriptor["course"],group=descriptor["group"],
        case=descriptor["case"],profile=descriptor["scenario"]["duel"]["target"]["profile"])
    try:
        options = None if descriptor["group"] == "tunnel_exit" else {"curriculum_scene":descriptor}
        obs,info = env.reset(options=options)
        initial = base.debug_snapshot()
        start = initial["time_ns"]
        scenario = copy.deepcopy(initial["data"]["evaluation"]["scenario"])
        row["initial_state_sha256"] = hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
        row["training_steps"] = training_steps
        row["checkpoint_sha256"] = checkpoint_sha256
        opportunities = []
        shot_requests = 0
        first_exit_clear = first_exit_tracking = first_exit_hit = None
        previous_hits = 0
        with torch.inference_mode():
            for step in range(3000):
                if base._history.clock.due:
                    opportunities.append(base._response["sim_time_ns"]-start)
                action = None
                if actor is not None:
                    tensors = {k:torch.as_tensor(np.asarray(obs[k])[None]) for k in ACTOR_KEYS}
                    _,_,actions,state = actor(tensors,state)
                    action = int(actions.item())
                obs,_,term,trunc,info = env.step(action)
                shot_requests += int(info.get("shot_requested",False))
                snapshot = base.debug_snapshot()
                own = next(r for r in snapshot["data"]["evaluation"]["robots"] if r["robot_id"] == 1)
                elapsed = (snapshot["time_ns"]-start)*1e-9
                if descriptor["group"] == "tunnel_exit" and step % 10 == 0:
                    from src.training.scene_admission import check_fireability
                    probe = base._client.inspect()["data"]["evaluation"]["fireability"]
                    try:check_fireability(probe); clear=True
                    except RuntimeError:clear=False
                    probes.append(dict(elapsed_s=elapsed,common_visible_ballistic_clear=clear,
                        tracking=base._actor_semantic["estimate"]["state"]))
                    # Record the real shared visible/shootable window after crossing the bridge.
                    if snapshot["time_ns"] >= 3_000_000_000 and clear and first_exit_clear is None:
                        first_exit_clear = elapsed
                if first_exit_clear is not None and first_exit_tracking is None and base._actor_semantic["estimate"]["state"] == "tracking":
                    first_exit_tracking = elapsed
                hits = own["resources"]["damaging_hits"]
                if first_exit_clear is not None and hits > previous_hits and first_exit_hit is None:
                    first_exit_hit = elapsed
                previous_hits = hits
                if record:
                    snapshot.update(phase="complete" if term or trunc else "evaluating",metrics=dict(
                        raw_damage=info["episode_damage"],eligible_damage=info["episode_damage"],shots=info["actual_shots"],
                        hits=hits,selected_slot=info.get("selected_slot",-1),slot_switches=info["slot_switches"],
                        shot_requests=shot_requests,target_tracking=base._actor_semantic["estimate"]["state"],
                        evaluation_time_ns=snapshot["time_ns"]-start,settlement_time_ns=0,end_reason=info["end_reason"]))
                    frames.append(snapshot)
                if term or trunc:break
        resources = own["resources"]
        row.update(damage=info["episode_damage"],actual_shots=info["actual_shots"],damaging_hits=resources["damaging_hits"],
            end_reason=info["end_reason"],shot_requests=shot_requests,slot_switches=info["slot_switches"],
            ordinary_heat_locks=resources["ordinary_heat_lock_count"],round_heat_lock_count=resources["round_heat_lock_count"],
            heat_locked_s=own["heat_locked_s"],episode_return=info["episode_training_reward"],
            kill_time_s=info["episode_time_s"] if info["end_reason"] == "target_destroyed" else None,
            opportunities=opportunities,
            episode_return_basis="full training reward" if actor is not None else "damage and thermal only; selection context unavailable",
            first_exit_clear_s=first_exit_clear,first_exit_tracking_s=first_exit_tracking,first_exit_hit_s=first_exit_hit)
        if record:
            from tools.training.view import _atomic_json
            _atomic_json(directory/"replay.json",dict(version=1,model=dict(name=controller,num_timesteps=training_steps),
                fingerprints={},render=dict(simulator_root=str(base.simulator_root),config=str(base.simulator_config),assets=str(base.simulator_root/"assets")),
                scene_seed=descriptor["seed"],scenario=scenario,step_ns=10_000_000,frames=frames,
                summary=dict(status="complete",end_reason=row["end_reason"],raw_damage=row["damage"],
                    score=dict(official_damage=row["damage"],eligible_shots=row["actual_shots"],damaging_projectiles=row["damaging_hits"])) ))
            (directory/"fireability.json").write_text(json.dumps(dict(warmup=warmup_probes,policy=probes),indent=2))
            row["warmup_physical_occlusion_observed"] = any(not p["common_visible_ballistic_clear"]
                and 2.15 <= p["target_bevy_m"][0] <= 2.85 for p in warmup_probes)
            row["takeover_round_s"] = start*1e-9
    except Exception as error:
        row.update(error=f"{type(error).__name__}: {error}",actual_shots=0,damaging_hits=0,end_reason="environment_error")
    finally:
        env.close()
    (directory/"result.json").write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tunnel",action="store_true")
    parser.add_argument("--checkpoint-root",type=Path)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--record",action="store_true",help="record the first fixed case per stage, or all tunnel cases")
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    jobs = []
    for role in ("infantry","sentry"):
        config = read_config(ROOT/f"config/training/rmul_2027_{role}_curriculum_v2.json")
        if args.tunnel:
            stages = [tunnel_bank("sentry" if role == "sentry" else "infantry_health_cooling")]
        else:
            if args.scenarios is None:parser.error("--scenarios required")
            stages = json.loads((args.scenarios/role/"curriculum/scenarios.json").read_text())["test" if args.all else "validation"]
        for c,stage in enumerate(stages):
            selected = stage if args.tunnel or args.all else [next(d for d in stage if d["scenario"]["duel"]["target"]["profile"] == p)
                for p in ("standard","wheel_leg","low","folding")]
            for descriptor in selected:
                for mode in (("traditional","gru") if args.checkpoint_root else ("traditional",)):
                    checkpoint = args.checkpoint_root/role/"final.zip" if args.checkpoint_root else None
                    jobs.append((config,descriptor,mode,checkpoint,args.output/role/f"stage-{c:02d}-case-{descriptor['case']:02d}"/mode,args.tunnel or (args.record and descriptor["case"] == stage[0]["case"])))
    (args.output/"scenarios.json").write_text(json.dumps([dict(role="sentry" if job[0]["environment"]["scenario"]["duel"]["controlled"]["preset"] == "sentry" else "infantry",scene=job[1],controller=job[2]) for job in jobs],indent=2))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn")) as pool:
        rows = list(pool.map(evaluate,jobs))
    (args.output/"results.json").write_text(json.dumps(rows,indent=2))
    if args.checkpoint_root and not any("error" in r for r in rows):
        for i in range(0,len(rows),2):
            a,b = rows[i:i+2]
            if a["initial_state_sha256"] != b["initial_state_sha256"]:
                raise RuntimeError("paired initial physical states differ")
            count = min(len(a["opportunities"]),len(b["opportunities"]))
            if a["opportunities"][:count] != b["opportunities"][:count]:
                raise RuntimeError("paired firing clocks differ")
    if any("error" in r for r in rows):raise SystemExit("diagnostic environment errors; inspect results.json")


if __name__ == "__main__":
    main()
