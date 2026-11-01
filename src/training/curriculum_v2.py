"""Frozen v2 combat stages; descriptors never enter policy observations."""
import copy
import hashlib
import json
import math

import numpy as np

DEFAULT = dict(version=2, validation_cases=40, review_every_updates=5,
               consecutive_passes=3, current_fraction=.7, uniform_review_fraction=.5,
               validation_seed=20000, test_seed=50000, mastered_review_steps=5_000_000)
STAGES = ("C0", "C1", "C2", "C3", "C5", "C8-R1", "C8-R2", "C8-T1",
          "C8-T2", "C8-D1", "C8-D2", "C8-M")
PROFILES = ("standard", "wheel_leg", "low", "folding")
GROUPS = (("static",), ("slow_rotation",), ("rotation", "translation"), ("combined",),
          ("rotation", "translation", "combined"), ("rotation",), ("rotation",),
          ("translation",), ("translation",), ("near", "far"), ("far",),
          ("spin_translation", "self_fast", "distance_fast", "self_distance"))
THRESHOLDS = ((.9,.8), (.9,.7), (.85,.7), (.8,.65), (.8,.65),
              (.8,.65), (.8,.65), (.8,.65), (.8,.65), (.8,.65), (.8,.65), (.75,.6))


def validate(value):
    if not isinstance(value, dict) or set(value) - DEFAULT.keys():
        raise ValueError("invalid curriculum v2 configuration")
    result = {**DEFAULT, **value}
    if result != DEFAULT:
        raise ValueError("v2 defaults are frozen; change version for a new curriculum")
    return result


def digest(value):
    """Stable checksum for frozen scene manifests and training contracts."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fingerprint(config):
    return digest(dict(config=config, stages=STAGES, groups=GROUPS, thresholds=THRESHOLDS))


def scene(course, seed, role, case=0, attempt=0, strata=None):
    if not 0 <= course < len(STAGES):
        raise ValueError("invalid v2 stage")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    pose = np.random.default_rng(np.random.SeedSequence([seed, 897, attempt]))
    strata = copy.deepcopy(strata or {})
    group = strata.get("group", GROUPS[course][case % len(GROUPS[course])])
    profile = strata.get("profile", PROFILES[int(rng.integers(4))])
    target_role = strata.get("target_role", "sentry" if rng.integers(2) else "infantry_health_cooling") if course >= 5 else "infantry_health_cooling"
    own = dict(profile="standard", preset=role, motion=dict(kind="static"))
    enemy = dict(profile=profile, preset=target_role, motion=dict(kind="static"))

    def motion(kind, angular=(1.,3.), speed=(.2,1.), amplitude=(.5,1.), controlled=False):
        sign = strata.get("own_direction" if controlled else "direction", 1 if rng.integers(2) else -1)
        return dict(kind=kind, amplitude_m=float(rng.uniform(*amplitude)),
                    speed_m_s=float(rng.uniform(*speed)),
                    angular_speed_rad_s=float(rng.uniform(*angular))*sign)

    def own_motion(kind):
        return motion(kind, angular=(.3,1.), controlled=True)

    def slow_target():
        kind = strata.get("distance_target", "rotation" if rng.integers(2) else "static")
        return motion(kind, angular=(.5,1.))

    distance = float(rng.uniform(3.,5.))
    advanced = []
    if course == 1:
        enemy["motion"] = motion("rotation", angular=(.5,1.))
    elif course == 2:
        enemy["motion"] = motion(group)
    elif course == 3:
        enemy["motion"] = motion("combined")
    elif course == 4:
        own["motion"] = own_motion(group)
        kinds = ("static",)*4 + ("rotation", "translation", "combined", "combined")
        enemy["motion"] = motion(strata.get("target_kind", kinds[int(rng.integers(8))]))
    elif course in (5,6):
        enemy["motion"] = motion("rotation", angular=(3.,5.) if course == 5 else (5.,7.))
        advanced = ["fast_rotation"]
    elif course in (7,8):
        enemy["motion"] = motion("translation", speed=(1.,1.5) if course == 7 else (1.5,2.), amplitude=(1.,1.5))
        advanced = ["fast_translation"]
    elif course in (9,10):
        bounds = (6.5,8.) if course == 10 else (2.,3.) if group == "near" else (5.,6.5)
        distance = float(rng.uniform(*bounds))
        enemy["motion"] = slow_target()
        advanced = ["extended_distance"]
    elif course == 11:
        fast_kind = strata.get("fast_kind", "rotation" if rng.integers(2) else "translation")
        self_kind = strata.get("self_kind", ("rotation", "translation", "combined")[int(rng.integers(3))])
        if group == "spin_translation":
            enemy["motion"] = motion("combined", angular=(3.,7.), speed=(1.,2.), amplitude=(1.,1.5))
            advanced = ["fast_rotation", "fast_translation"]
        elif group in ("self_fast", "distance_fast"):
            enemy["motion"] = motion(fast_kind, angular=(3.,7.), speed=(1.,2.), amplitude=(1.,1.5))
            advanced = ["fast_"+fast_kind]
        else:
            enemy["motion"] = motion(strata.get("base_kind", ("static", "rotation", "translation", "combined")[int(rng.integers(4))]))
        if group in ("self_fast", "self_distance"):
            own["motion"] = own_motion(self_kind)
            advanced.append("self_motion")
        if group in ("distance_fast", "self_distance"):
            bounds = (2.,3.) if strata.get("distance_side", int(rng.integers(2))) == 0 else (5.,8.)
            distance = float(rng.uniform(*bounds))
            advanced.append("extended_distance")
    jitter = float(pose.uniform(-.3,.3))
    # A random bearing avoids coupling long distances to one map diagonal.
    bearing = float(pose.uniform(-math.pi, math.pi))
    direction = np.array([math.sin(bearing), math.cos(bearing)]) * distance/2
    own["position_xz_m"] = (direction + [jitter,0.]).tolist()
    enemy["position_xz_m"] = (-direction + [jitter,0.]).tolist()
    enemy["yaw_rad"] = float(pose.uniform(-math.pi, math.pi))
    scenario = dict(duel=dict(controlled=own, target=enemy), unlimited_heat=False,
                    measurements=dict(noise_std_px=.25, latency_ms=20, dropout_probability=0.))
    return dict(version=2, course=course, stage_id=STAGES[course], group=group, seed=int(seed),
                case=int(case), attempt=attempt, strata=strata, scenario=scenario,
                scarce=False, damage_capacity=None, initial_distance_m=distance,
                advanced_factors=advanced)


def bank(role, config, test=False):
    config = validate(config)
    base = config["test_seed" if test else "validation_seed"]
    result = []
    for c in range(len(STAGES)):
        rng = np.random.default_rng(np.random.SeedSequence([base,c,410]))
        n = config["validation_cases"]
        dimensions = dict(group=GROUPS[c], profile=PROFILES,
                          target_role=("infantry_health_cooling","sentry"), direction=(-1,1),
                          own_direction=(-1,1), distance_target=("static","rotation"),
                          target_kind=("static",)*4+("rotation","translation","combined","combined"),
                          fast_kind=("rotation","translation"), self_kind=("rotation","translation","combined"),
                          base_kind=("static","rotation","translation","combined"), distance_side=(0,1))
        columns = {key: rng.permutation([values[i % len(values)] for i in range(n)]).tolist()
                   for key,values in dimensions.items()}
        result.append([scene(c,base+c*1000+i,role,i,strata={k:v[i] for k,v in columns.items()}) for i in range(n)])
    return result


def assess(course, rows):
    kill_limit, hit_limit = THRESHOLDS[course]
    metrics, deficits = {}, []
    # Both primary tasks and each physical profile must pass.
    strata = [(g,[r for r in rows if r["group"] == g]) for g in GROUPS[course]]
    strata += [("profile/"+p,[r for r in rows if r.get("profile", r.get("scenario",{}).get("duel",{}).get("target",{}).get("profile")) == p]) for p in PROFILES]
    for group,samples in strata:
        shots = sum(r.get("actual_shots",0) for r in samples)
        hit = sum(r.get("damaging_hits",0) for r in samples)/shots if shots else 0.
        kill = sum(r.get("end_reason") == "target_destroyed" for r in samples)/len(samples) if samples else 0.
        deficit = max(0.,1-hit/hit_limit,1-kill/kill_limit)
        if not samples or any("error" in r for r in samples):
            deficit = 1.
        metrics[group] = dict(hit_efficiency=hit,kill_rate=kill,cases=len(samples))
        deficits.append(deficit)
    ordinary = sum(r.get("ordinary_heat_locks",0)>0 for r in rows)
    permanent = sum(r.get("round_heat_lock_count",int(r.get("round_heat_lock",False)))>0 for r in rows)
    thermal = dict(passed=bool(rows) and permanent == 0 and ordinary/len(rows) <= .05,
                   locked_episodes=ordinary, locked_episode_fraction=ordinary/max(1,len(rows)),
                   permanent_locked_episodes=permanent)
    deficit = float(max(deficits))
    return dict(passed=deficit <= 1e-9, combat_passed=deficit <= 1e-9,
                deficit=deficit, metrics=metrics, thermal=thermal)


class Curriculum:
    def __init__(self, config, seed=17, state=None):
        self.config = validate(config)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed,781]))
        self.stage, self.streak, self.reports, self.counts = 0,0,{},{}
        self.mastered, self.last_full_review_steps = False,0
        self.weak_streaks = {}
        if state is not None:
            if state.get("version") != 2 or state.get("fingerprint") != fingerprint(self.config):
                raise ValueError("curriculum version/fingerprint changed; start a new experiment")
            for k in ("stage","streak","reports","counts","mastered","last_full_review_steps","weak_streaks"):
                setattr(self,k,copy.deepcopy(state[k]))
            self.rng.bit_generator.state = state["rng"]

    def weights(self):
        weights = np.zeros(len(STAGES))
        if self.stage == 0:
            weights[0] = 1.
            return weights
        gaps = np.array([self.reports.get(str(c),{}).get("deficit",0.) for c in range(self.stage)])
        weak = gaps/gaps.sum() if gaps.sum() else np.full(self.stage,1/self.stage)
        weights[:self.stage] = .15/self.stage + .15*weak
        weights[self.stage] = .7
        return weights

    def next_scene(self, role, worker):
        course = int(self.rng.choice(len(STAGES),p=self.weights()))
        seed = int(self.rng.integers(0,2**62))
        key = str(worker)
        case = self.counts.get(key,0)
        self.counts[key] = case+1
        result = scene(course,seed,role,int(self.rng.integers(0,1000000)))
        result["episode_index"] = case
        return result

    def observe(self, report):
        self.reports[str(self.stage)] = report
        self.streak = self.streak+1 if report["passed"] else 0

    def promotion_ready(self):
        return self.streak >= self.config["consecutive_passes"]

    def promote(self):
        if not self.promotion_ready() or any(not self.reports.get(str(c),{}).get("passed",False) for c in range(self.stage)):
            return False
        if self.stage == len(STAGES)-1:
            self.mastered = True
            return False
        self.stage += 1
        self.streak = 0
        return True

    def state(self):
        return copy.deepcopy(dict(version=2,fingerprint=fingerprint(self.config),stage=self.stage,
            streak=self.streak,reports=self.reports,counts=self.counts,mastered=self.mastered,
            last_full_review_steps=self.last_full_review_steps,weak_streaks=self.weak_streaks,
            rng=self.rng.bit_generator.state))


def validate_update(curriculum, evaluator, actor, steps, update):
    """Boundary-only validation. Thermal status never enters promotion/review deficits."""
    c = curriculum
    previous_stage = c.stage
    def evaluate(index):
        report = evaluator.evaluate(actor,index,update)
        c.reports[str(index)] = report
        return report
    if not c.mastered:
        c.observe(evaluate(c.stage))
        if c.promotion_ready() or update % c.config["review_every_updates"] == 0:
            for index in range(c.stage):
                evaluate(index)
        c.promote()
        if c.mastered:
            c.last_full_review_steps = steps
        return c.stage != previous_stage
    due = steps-c.last_full_review_steps >= c.config["mastered_review_steps"]
    full = due or bool(c.weak_streaks) and update % c.config["review_every_updates"] == 0
    indices = range(len(STAGES)) if full else [int(k) for k in c.weak_streaks]
    for index in indices:
        report = evaluate(index)
        key = str(index)
        if not report["passed"]:
            c.weak_streaks[key] = 0
        elif key in c.weak_streaks:
            c.weak_streaks[key] += 1
    if full:
        c.last_full_review_steps = steps
    recovered = c.weak_streaks and all(v >= c.config["consecutive_passes"] for v in c.weak_streaks.values())
    if recovered:
        for index in range(len(STAGES)):
            report = evaluate(index)
            if not report["passed"]:
                c.weak_streaks[str(index)] = 0
        if all(r["passed"] for r in c.reports.values()):
            c.weak_streaks.clear()
        c.last_full_review_steps = steps
    return False
