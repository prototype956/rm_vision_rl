"""Independent deterministic curriculum validation and frozen scenario manifests."""
from concurrent.futures import ProcessPoolExecutor
import copy
import io
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch

from src.training.curriculum_v2 import bank, assess, digest


def _case(job):
    config, descriptor, weights, directory, preflight = job
    if config["curriculum"]["version"] == 3 and not preflight:
        from src.training.offline_evaluation import evaluate_scene
        try:
            return evaluate_scene(config, descriptor, directory, weights=weights)
        except Exception as error:
            return dict(course=descriptor["course"], group=descriptor["group"], seed=descriptor["seed"],
                case=descriptor["case"], profile=descriptor["scenario"]["duel"]["target"]["profile"],
                motion_type=descriptor.get("motion_type"), actual_shots=0, damaging_hits=0,
                status="failed", end_reason="environment_error", error=f"{type(error).__name__}: {error}")
    from src.training.environment import make_environment
    from src.policy.recurrent_actor import RecurrentActor
    from src.policy.recurrent_observations import ACTOR_KEYS
    from src.policy.asymmetric_observations import NAMES
    from src.training.curriculum_metrics import TrackingMetrics
    torch.set_num_threads(1)
    actor = None
    if not preflight:
        actor = RecurrentActor().eval()
        if isinstance(weights,bytes):
            weights = torch.load(io.BytesIO(weights),map_location="cpu",weights_only=True)
        actor.load_state_dict(weights)
        state = actor.initial_state(1)
    config = copy.deepcopy(config)
    config["environment"]["decision_clock"]["seed"] = descriptor["seed"] % (2**32)
    env = make_environment(config, directory)
    row = {k: descriptor[k] for k in ("course", "group", "seed", "case", "scarce", "damage_capacity")}
    row["profile"] = descriptor["scenario"]["duel"]["target"]["profile"]
    row["phase"]="reset"
    try:
        obs, initial = env.reset(options={"curriculum_scene": descriptor})
        row["admitted_scene"] = initial["curriculum_scene"]
        if "visibility_admission" in initial:row["visibility_admission"]=initial["visibility_admission"]
        row["fireability"] = initial.get("fireability")
        row["trackability"] = initial.get("trackability")
        tracking_metrics = TrackingMetrics(obs)
        original = env.unwrapped._response["data"]["evaluation"]["scenario"]
        positions = [np.asarray(original[f"{name}_position_bevy_m"])[[0,2]] for name in ("controlled", "target")]
        current_robots = {r["robot_id"]:r for r in env.unwrapped._response["data"]["evaluation"]["robots"]}
        takeover = [np.asarray(current_robots[i]["position_bevy_m"])[[0,2]] for i in (1, 2)]
        row.update(initial_horizontal_distance_m=float(np.linalg.norm(positions[0]-positions[1])),
                   takeover_horizontal_distance_m=float(np.linalg.norm(takeover[0]-takeover[1])))
        lost_steps = requests = 0
        initial_thermal = copy.deepcopy(env.unwrapped._actor_semantic["thermal"])
        base_cooling = 30 if config["environment"]["scenario"]["duel"]["controlled"]["preset"] == "sentry" else 24
        if initial_thermal["effective_cooling_per_second"] != base_cooling:
            raise RuntimeError("effective cooling differs from initialized rule state")
        terminal = False
        limit = 50 if preflight else 3000
        row["phase"]="control"
        with torch.inference_mode():
            for _ in range(limit):
                if actor is None:
                    action = 0
                else:
                    tensors = {k: torch.as_tensor(np.asarray(obs[k])[None]) for k in ACTOR_KEYS}
                    _, _, actions, state = actor(tensors, state)
                    action = int(actions.item())
                # The request belongs to the pre-action observation, whereas
                # tracking/state recovery below belongs to the returned packet.
                request_without_tracking = action in (2, 4, 6, 8) and not tracking_metrics.tracking(obs)
                obs, _, terminated, truncated, info = env.step(action)
                tracking_metrics.observe(obs)
                tracking_metrics.missing_requests += int(request_without_tracking and info.get("shot_requested", False))
                requests += int(info.get("shot_requested", False))
                lost_steps += int(obs["features"][NAMES.index("target.state.tracking")] < .5)
                terminal = terminated or truncated
                if terminal:
                    break
        snapshot = env.unwrapped.debug_snapshot()["data"]
        own = next(r for r in snapshot["evaluation"]["robots"] if r["robot_id"] == 1)
        resource = own["resources"]
        # Heat locks are evaluation-only, separate from the actor thermal packet.
        row.update(damage=info["episode_damage"], actual_shots=info["actual_shots"],
                   damaging_hits=resource["damaging_hits"], end_reason=info["end_reason"],
                   slot_switches=info["slot_switches"], episode_return=info["episode_training_reward"],
                   steps=info["episode_steps"], shot_requests=requests, missing_steps=lost_steps,
                   kill_time_s=info["episode_time_s"] if info["end_reason"] == "target_destroyed" else None,
                   tracking=tracking_metrics.result(),
                   ordinary_heat_locks=resource.get("ordinary_heat_lock_count", resource.get("heat_lock_count", 0)),
                   round_heat_lock_count=resource.get("round_heat_lock_count", int(resource["round_locked"])),
                   round_heat_lock=resource["round_locked"], heat_locked_s=own.get("heat_locked_s", 0.), preflight=preflight,
                   initial_actor_feedback=dict(thermal=initial_thermal))
    except Exception as error:
        tracking_timeout = isinstance(error, TimeoutError) and str(error).startswith(
            ("fire-target warmup timed out", "curriculum phase warmup timed out"))
        row.update(damage=0, actual_shots=0, damaging_hits=0,
                   end_reason="tracking_initialization_timeout" if tracking_timeout else "environment_error")
        row["failure" if tracking_timeout else "error"] = f"{type(error).__name__}: {error}"
    finally:
        env.close()
    return row


class CurriculumEvaluator:
    def __init__(self, config, output, frozen=None):
        self.config = copy.deepcopy(config)
        self.output = Path(output)/"curriculum"
        self.output.mkdir(parents=True, exist_ok=True)
        role = config["environment"]["scenario"]["duel"]["controlled"]["preset"]
        from src.training.curriculum import implementation
        self.module = implementation(config["curriculum"])
        self.validation = self.module.bank(role, config["curriculum"])
        self.test = self.module.bank(role, config["curriculum"], test=True)
        self.calibration_scenes = self.module.bank(role, config["curriculum"], split="calibration") if self.module.__name__.endswith("v3") else None
        if frozen is not None:
            if frozen.get("version") != config["curriculum"]["version"] or len(frozen["validation"]) != len(self.validation):
                raise ValueError("frozen scene bank curriculum version differs")
            payload = {k:v for k,v in frozen.items() if k != "sha256"}
            if digest(payload) != frozen["sha256"]:
                raise ValueError("frozen scene manifest checksum mismatch")
            self.validation, self.test = frozen["validation"], frozen["test"]
            self.calibration_scenes = frozen.get("calibration", self.calibration_scenes)
        self.write_manifest()
        worker_cap = 32 if config["curriculum"].get("release_revision",1) >= 2 else 16
        self.pool = ProcessPoolExecutor(max_workers=min(worker_cap, max(config["n_envs"], (os.cpu_count() or 4)//4)), mp_context=mp.get_context("spawn"))
        self.cache = {}

    def write_manifest(self):
        manifest = dict(version=self.config["curriculum"]["version"], validation=self.validation, test=self.test)
        if self.calibration_scenes is not None:
            manifest["calibration"] = self.calibration_scenes
        manifest["sha256"] = digest(manifest)
        self.manifest = copy.deepcopy(manifest)
        (self.output/"scenarios.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def evaluate(self, actor, course, update, test=False, preflight=False):
        key = (course, update, test, preflight)
        if key in self.cache:
            return self.cache[key]
        directory = self.output/("preflight-test" if preflight and test else "preflight-validation" if preflight else "test" if test else f"update-{update:04d}")/f"C{course}"
        directory.mkdir(parents=True, exist_ok=True)
        weights = None
        if not preflight:
            # Bytes bypass torch multiprocessing storage/FD sharing, which exhausts
            # descriptors with 128 standing workers plus queued validation jobs.
            stream = io.BytesIO()
            torch.save({k:v.detach().cpu() for k,v in actor.state_dict().items()},stream)
            weights = stream.getvalue()
        scenes = (self.test if test else self.validation)[course]
        jobs = [(self.config, d, weights, directory/f"case-{i:03d}", preflight) for i,d in enumerate(scenes)]
        rows = []
        for row in self.pool.map(_case, jobs):
            rows.append(row)
            with (directory/"episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        if preflight and not any("error" in r or "failure" in r for r in rows):
            for i, row in enumerate(rows):
                scenes[i] = dict(row["admitted_scene"], admitted=True)
            self.write_manifest()
        report = (self.module.assess(course, rows, self.config["curriculum"].get("calibration"))
                  if self.config["curriculum"]["version"] == 3 else assess(course, rows))
        if preflight:
            report["passed"] = not any("error" in r or "failure" in r for r in rows)
            report["criterion"] = "environment and initial feedback contract; no policy shooting"
        report.update(incomplete_cases=sum(r.get("status") != "complete" for r in rows) if self.config["curriculum"]["version"] == 3 and not preflight else 0,
                      environment_errors=sum("error" in r for r in rows),
                      tracking_initialization_failures=sum("failure" in r for r in rows), update=update, course=course,
                      cases=len(rows), test=test, preflight=preflight)
        (directory/"summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("CURRICULUM "+json.dumps(report), flush=True)
        self.cache[key] = report
        # A billion-step run must not retain every historical report in RAM.
        while len(self.cache) > 48:
            self.cache.pop(next(iter(self.cache)))
        return report

    def preflight(self):
        reports = [self.evaluate(None, course, 0, test=test, preflight=True) for test in (False, True) for course in range(len(self.validation))]
        if any(not r["passed"] for r in reports):
            raise RuntimeError("curriculum preflight failed; inspect separately recorded environment errors and tracking failures")
        return reports

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
