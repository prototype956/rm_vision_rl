"""Stratified translation curriculum; all descriptors and baselines are evaluation-only."""
import copy
import math
import numpy as np
from src.training import curriculum_v2 as v2
from src.training.curriculum_v2 import digest, STAGES, PROFILES

TRANSLATION = (7, 8)
MOTIONS = ("radial", "oblique", "lateral")
MOTION_WEIGHTS = (.2, .3, .5)
DEFAULT = {**v2.DEFAULT, "version": 3, "translation_cases": 96,
           "calibration_seed": 80000, "translation_admission_attempts": 128, "release_revision": 1, "calibration": None}
REVISION_2 = {**DEFAULT, "release_revision": 2, "translation_cases": 384}
CURRENT = {**REVISION_2, "release_revision": 3}
SCORING = "window-owned damaging_projectiles / eligible_shots; natural settlement; 30s window; 1000 settle steps"


def validate(value):
    if not isinstance(value, dict) or set(value) - DEFAULT.keys():
        raise ValueError("invalid curriculum v3 configuration")
    expected = {1:DEFAULT,2:REVISION_2,3:CURRENT}.get(value.get("release_revision",1))
    if expected is None:raise ValueError("unknown v3 release revision")
    result = {**expected, **copy.deepcopy(value)}
    if {k:v for k,v in result.items() if k != "calibration"} != {k:v for k,v in expected.items() if k != "calibration"}:
        raise ValueError("v3 scene and promotion defaults are frozen")
    if result["calibration"] is not None:
        validate_calibration(result["calibration"])
        if result["release_revision"] >= 3 and result["calibration"]["provenance"].get("visibility_admission") != "full_path_clear":
            raise ValueError("revision 3 requires an independently recalibrated full_path_clear baseline")
    return result


def classify(angle_degrees):
    if not 0 <= angle_degrees <= 90:
        raise ValueError("translation acute angle must be in [0,90]")
    return "radial" if angle_degrees < 20 else "oblique" if angle_degrees < 60 else "lateral"


def scene(course, seed, role, case=0, attempt=0, strata=None, release_revision=1):
    strata = copy.deepcopy(strata or {})
    result = v2.scene(course, seed, role, case, attempt, strata)
    result["version"] = 3
    result["release_revision"] = release_revision
    if release_revision >= 3:
        result["scenario"]["visibility_admission"]="full_path_clear"
    if course not in TRANSLATION:
        return result
    rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    motion_type = strata.get("motion_type") or str(rng.choice(MOTIONS, p=MOTION_WEIGHTS))
    if motion_type not in MOTIONS:
        raise ValueError("invalid translation motion_type")
    lo, hi = {"radial":(0.,20.), "oblique":(20.,60.), "lateral":(60.,90.)}[motion_type]
    angle = float(rng.uniform(lo, hi))
    side = strata.get("lateral_side", 1 if rng.integers(2) else -1)
    direction = strata.get("direction", 1 if rng.integers(2) else -1)
    placement = np.random.default_rng(np.random.SeedSequence([seed,306,attempt]))
    own = result["scenario"]["duel"]["controlled"]
    enemy = result["scenario"]["duel"]["target"]
    delta = np.asarray(own["position_xz_m"]) - enemy["position_xz_m"]
    heading = math.atan2(-delta[1], delta[0]) + side*math.radians(angle)
    heading += math.pi if direction < 0 else 0.
    enemy["motion"]["translation_heading_rad"] = math.remainder(heading, math.tau)
    if course == 7:
        enemy["motion"]["amplitude_m"] = float(rng.uniform(1.5, 2.))
    # Place the unchanged relative task inside the 12x8 m league floor. The
    # real collider/support checks still decide admission; this only avoids
    # wasting attempts outside the arena. No battle outcome enters sampling.
    axis = np.array([math.cos(heading),-math.sin(heading)])
    center = np.asarray(enemy["position_xz_m"])
    reach = axis*enemy["motion"]["amplitude_m"]
    points = np.array([own["position_xz_m"],center-reach,center+reach])
    lower = np.array([-5.4,-3.4])-points.min(axis=0)
    upper = np.array([5.4,3.4])-points.max(axis=0)
    offset = placement.uniform(lower,upper) if np.all(lower<=upper) else np.zeros(2)
    for robot in (own,enemy):
        robot["position_xz_m"] = (np.asarray(robot["position_xz_m"])+offset).tolist()
    result.update(motion_type=motion_type, translation_angle_deg=angle)
    result["strata"].update(motion_type=motion_type, lateral_side=int(side), direction=int(direction))
    return result


def bank(role, config, test=False, *, split=None):
    config = validate(config)
    split = split or ("test" if test else "validation")
    if split not in ("calibration", "validation", "test"):
        raise ValueError("invalid scene split")
    result = v2.bank(role, v2.DEFAULT, test=split == "test")
    for rows in result:
        for row in rows:
            row["version"] = 3
            row["release_revision"] = config["release_revision"]
            if config["release_revision"] >= 3:
                row["scenario"]["visibility_admission"]="full_path_clear"
    base = config[{"calibration":"calibration_seed", "validation":"validation_seed", "test":"test_seed"}[split]]
    for course in range(len(STAGES)):
        if course not in TRANSLATION:
            if split == "calibration":
                result[course] = []
            continue
        rows = []
        for replicate in range(config["translation_cases"]//96):
            for profile in PROFILES:
                for motion_type in MOTIONS:
                    for target_role in ("infantry_health_cooling", "sentry"):
                        for direction in (-1, 1):
                            for side in (-1, 1):
                                case = len(rows)
                                strata = dict(profile=profile, motion_type=motion_type,
                                    target_role=target_role, direction=direction, lateral_side=side)
                                rows.append(scene(course, base+course*1000+case, role, case,
                                    strata=strata,release_revision=config["release_revision"]))
        result[course] = rows
    return result


def fingerprint(config):
    value = dict(config=validate(config), stages=STAGES, motions=MOTIONS,
        weights=MOTION_WEIGHTS, scoring=SCORING, legacy=v2.fingerprint(v2.DEFAULT))
    if config.get("release_revision", 1) >= 3:
        from src.training.scene_placement import identity
        value["random_placement_sampler_sha256"] = identity()
    return digest(value)


def bucket_keys():
    return [f"{profile}/{motion}" for profile in PROFILES for motion in MOTIONS]


def bucket_metrics(rows):
    result = {}
    for key in bucket_keys():
        profile, motion = key.split("/")
        samples = [r for r in rows if r.get("profile") == profile and r.get("motion_type") == motion]
        shots = sum(r.get("actual_shots", 0) for r in samples)
        hits = sum(r.get("damaging_hits", 0) for r in samples)
        kills = sum(r.get("end_reason") == "target_destroyed" for r in samples)
        invalid = sum(r.get("status") != "complete" or "error" in r or "failure" in r for r in samples)
        result[key] = dict(cases=len(samples), actual_shots=shots, damaging_hits=hits,
            kills=kills, hit_efficiency=hits/shots if shots else 0.,
            kill_rate=kills/len(samples) if samples else 0., invalid_cases=invalid)
    return result


def build_calibration(role, course_rows, provenance, cases_per_bucket=8):
    if cases_per_bucket not in (8,32):
        raise ValueError("unsupported calibration sample count")
    thresholds = {}
    for course in TRANSLATION:
        metrics = bucket_metrics(course_rows[str(course)])
        for key, value in metrics.items():
            if value["cases"] != cases_per_bucket or value["invalid_cases"] or not all(value[k] > 0 for k in ("actual_shots", "damaging_hits", "kills")):
                raise ValueError(f"unqualified traditional calibration: {STAGES[course]} {key}: {value}")
        thresholds[str(course)] = {key:dict(baseline=value,
            hit_efficiency=.9*value["hit_efficiency"],
            kill_rate=min(.8, max(0., value["kill_rate"]-.1))) for key,value in metrics.items()}
    artifact = dict(version=1, status="ready", role=role, scoring=SCORING, cases_per_bucket=cases_per_bucket,
                    thresholds=thresholds, provenance=copy.deepcopy(provenance))
    artifact["sha256"] = digest(artifact)
    return artifact


def validate_calibration(artifact):
    if not isinstance(artifact, dict) or artifact.get("version") != 1 or artifact.get("status") != "ready" or artifact.get("scoring") != SCORING:
        raise ValueError("ready v3 calibration is required")
    if artifact.get("sha256") != digest({k:v for k,v in artifact.items() if k != "sha256"}):
        raise ValueError("calibration checksum mismatch")
    if set(artifact.get("thresholds", {})) != {"7", "8"}:
        raise ValueError("calibration must contain T1 and T2")
    cases_per_bucket=artifact.get("cases_per_bucket",8)
    if cases_per_bucket not in (8,32):raise ValueError("invalid calibration sample count")
    for limits in artifact["thresholds"].values():
        if set(limits) != set(bucket_keys()):
            raise ValueError("calibration buckets are incomplete")
        for value in limits.values():
            baseline = value["baseline"]
            if baseline["cases"] != cases_per_bucket or baseline["invalid_cases"] or not all(baseline[k] > 0 for k in ("actual_shots", "damaging_hits", "kills")):
                raise ValueError("calibration contains a degenerate baseline")
            if value["hit_efficiency"] != .9*baseline["hit_efficiency"] or value["kill_rate"] != min(.8,max(0.,baseline["kill_rate"]-.1)):
                raise ValueError("calibration threshold formula differs")


def assess(course, rows, calibration=None):
    invalid = any(r.get("status") != "complete" or "error" in r or "failure" in r for r in rows)
    if course not in TRANSLATION:
        report = v2.assess(course, rows)
        if invalid:
            report.update(passed=False, combat_passed=False, deficit=1.)
        return report
    if calibration is None:
        return dict(passed=False, combat_passed=False, deficit=1., metrics=bucket_metrics(rows), reason="calibration_missing")
    validate_calibration(calibration)
    metrics, deficits = bucket_metrics(rows), []
    expected_cases=calibration.get("cases_per_bucket",8)
    for key, value in metrics.items():
        limit = calibration["thresholds"][str(course)][key]
        value.update(required_hit_efficiency=limit["hit_efficiency"], required_kill_rate=limit["kill_rate"], baseline=limit["baseline"])
        deficit = max(0., 1-value["hit_efficiency"]/limit["hit_efficiency"], 1-value["kill_rate"]/limit["kill_rate"] if limit["kill_rate"] > 0 else 0.)
        value["passed"] = value["cases"] == expected_cases and not value["invalid_cases"] and deficit <= 1e-9
        value["failure_reasons"] = [] if value["passed"] else ["incomplete_or_below_baseline_threshold"]
        deficits.append(deficit if value["passed"] or value["cases"] == expected_cases and not value["invalid_cases"] else 1.)
    thermal = v2.assess(course, rows)["thermal"]
    passed = all(m["passed"] for m in metrics.values()) and not invalid
    return dict(passed=passed, combat_passed=passed, deficit=1. if invalid else max(deficits), metrics=metrics,
                thermal=thermal, calibration_sha256=calibration["sha256"])


class Curriculum(v2.Curriculum):
    def __init__(self, config, seed=17, state=None):
        super().__init__(v2.DEFAULT, seed)
        self.config = validate(config)
        self.histories = {}
        if state is not None:
            if state.get("version") != 3 or state.get("fingerprint") != fingerprint(self.config):
                raise ValueError("v3 curriculum fingerprint changed; explicit migration required")
            for key in ("stage", "streak", "reports", "counts", "mastered", "last_full_review_steps", "weak_streaks", "histories"):
                setattr(self, key, copy.deepcopy(state[key]))
            self.rng.bit_generator.state = state["rng"]

    def next_scene(self, role, worker):
        course = int(self.rng.choice(len(STAGES), p=self.weights()))
        seed = int(self.rng.integers(0, 2**62))
        key = str(worker)
        index = self.counts.get(key, 0)
        self.counts[key] = index+1
        result = scene(course, seed, role, int(self.rng.integers(0, 1000000)), release_revision=self.config["release_revision"])
        if self.config["release_revision"] >= 3 and course in TRANSLATION:
            from src.training.scene_placement import SAMPLER, identity
            result["placement_search"] = dict(sampler=SAMPLER, hints_sha256=identity())
        result["episode_index"] = index
        return result

    def observe(self, report):
        super().observe(report)
        key = str(self.stage)
        self.histories[key] = (self.histories.get(key, [])+[bool(report["passed"])])[-3:]

    def promotion_ready(self):
        if self.stage not in TRANSLATION:
            return super().promotion_ready()
        history = self.histories.get(str(self.stage), [])
        return len(history) == 3 and history[-1] and sum(history) >= 2

    def promote(self):
        previous = self.stage
        promoted = super().promote()
        if self.stage != previous:
            self.histories.pop(str(self.stage), None)
        return promoted

    def state(self):
        value = super().state()
        value.update(version=3, fingerprint=fingerprint(self.config), histories=copy.deepcopy(self.histories))
        return value


def validate_update(curriculum, evaluator, actor, steps, update):
    c = curriculum
    previous = c.stage
    def evaluate(index, history=False):
        report = evaluator.evaluate(actor,index,update)
        c.reports[str(index)] = report
        if history:
            key = str(index)
            c.histories[key] = (c.histories.get(key, [])+[bool(report["passed"])])[-3:]
        return report
    if not c.mastered:
        c.observe(evaluate(c.stage))
        if c.promotion_ready() or update % c.config["review_every_updates"] == 0:
            for index in range(c.stage):evaluate(index)
        c.promote()
        if c.mastered:c.last_full_review_steps = steps
        return c.stage != previous
    due = steps-c.last_full_review_steps >= c.config["mastered_review_steps"]
    full = due or bool(c.weak_streaks) and update % c.config["review_every_updates"] == 0
    indices = range(len(STAGES)) if full else [int(k) for k in c.weak_streaks]
    for index in indices:
        report = evaluate(index,history=True)
        key = str(index)
        if not report["passed"]:c.weak_streaks[key] = 0
        elif key in c.weak_streaks:c.weak_streaks[key] += 1
    if full:c.last_full_review_steps = steps
    def stable(key):
        h = c.histories.get(key, [])
        return len(h) == 3 and h[-1] and sum(h) >= (2 if int(key) in TRANSLATION else 3)
    if c.weak_streaks and all(stable(k) for k in c.weak_streaks):
        for index in range(len(STAGES)):
            if not evaluate(index)["passed"]:c.weak_streaks[str(index)] = 0
        if all(r["passed"] for r in c.reports.values()):c.weak_streaks.clear()
        c.last_full_review_steps = steps
    return False
