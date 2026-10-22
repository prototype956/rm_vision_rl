"""Restricted actor contract v4. Only explicit sensor/estimator fields enter the tensor.

The estimator's origin follows the gimbal; its axes are IMU world axes. Encoding
rotates into the current heading frame (X forward, Y left, Z up). No navigation
position or chassis velocity is used. Softsign scaling retains over-limit values.
"""
import copy
import math
import numpy as np

VERSION = 4
MODE = "asymmetric_v1"


def bounded(value, scale=1.0):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite observation")
    return value / (scale + abs(value))


def quat_product(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def empty_observation():
    return dict(estimate=dict(state="lost", label=None, armor_type=None,
                             center_world=[0]*3, velocity_world=[0]*3,
                             orientation_xyzw=[0, 0, 0, 1], radii_m=[0, 0],
                             height_offset_m=0, armor_tilt_rad=0, yaw_velocity_rad_s=0,
                             center_covariance_world=[[0]*3 for _ in range(3)],
                             yaw_variance_rad2=0),
                feedback=dict(valid=False, yaw_rad=0, pitch_rad=0),
                candidates=[dict(valid=False, yaw_rad=0, pitch_rad=0, distance_m=0,
                                 fly_time_s=0, prediction_horizon_s=0,
                                 facing_now_rad=0, facing_impact_rad=0,
                                 detected=False, last_seen_age_s=None) for _ in range(4)],
                previous_slot=-1, selected_slot_age_s=0, prediction_age_s=None,
                action_mask=[True]+[False]*8)


def adapt_observation(raw, packet, now_ns, *, last_request_ns=None, request_previous=False):
    """Whitelist wire fields. Referee flags/mechanical state never enter this adapter."""
    result = empty_observation()
    if raw is not None:
        for key in result:
            if key == "estimate":
                result[key] = {name: copy.deepcopy(raw[key][name]) for name in result[key]}
            elif key == "candidates":
                result[key] = [{name: copy.deepcopy(candidate[name]) for name in template}
                               for candidate, template in zip(raw[key], result[key], strict=True)]
            elif key == "feedback":
                result[key] = {name: raw[key][name] for name in result[key]}
            else:
                result[key] = copy.deepcopy(raw[key])
    if packet["version"] != 1:
        raise ValueError("unsupported actor sensor packet")
    result["imu"] = copy.deepcopy(packet["imu"])
    result["thermal"] = copy.deepcopy(packet["thermal"])
    result["now_ns"] = now_ns  # Used only to derive sensor ages, never encoded as absolute time.
    result["since_request_s"] = (None if last_request_ns is None else
                                 (now_ns-last_request_ns)*1e-9)
    result["request_previous"] = request_previous
    return result


def encode(o):
    names, values = [], []

    def add(name, value, scale=None):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite feature: " + name)
        names.append(name)
        values.append(value if scale is None else bounded(value, scale))

    def vec(name, data, scale=None):
        for i, value in enumerate(data):
            add(f"{name}.{i}", value, scale)

    e, f, imu, thermal = o["estimate"], o["feedback"], o["imu"], o["thermal"]
    q = imu["orientation_xyzw"]
    x, y, z, w = q
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotate = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
    unyaw = [0, 0, -math.sin(yaw/2), math.cos(yaw/2)]
    present = e["state"] != "lost"
    add("target.valid", present)
    for state in ("lost", "detecting", "tracking", "temp_lost"):
        add("target.state."+state, e["state"] == state)
    for label in range(9):
        add(f"target.label.{label}", present and e["label"] == label)
    for kind in range(2):
        add(f"target.armor_type.{kind}", present and e["armor_type"] == kind)
    vec("target.position_heading_m", rotate @ e["center_world"] if present else [0]*3, 10)
    vec("target.relative_velocity_heading_mps", rotate @ e["velocity_world"] if present else [0]*3, 10)
    tq = quat_product(unyaw, e["orientation_xyzw"])
    vec("target.orientation_heading_xyzw", np.array(tq)*(1 if tq[3] >= 0 else -1) if present else [0]*4)
    vec("target.radii_m", e["radii_m"] if present else [0]*2, .5)
    for key, scale in (("height_offset_m", .5), ("armor_tilt_rad", 1),
                       ("yaw_velocity_rad_s", 10), ("yaw_variance_rad2", 1)):
        add("target."+key, e[key] if present else 0, scale)
    cov = rotate @ np.asarray(e["center_covariance_world"]) @ rotate.T
    vec("target.position_covariance_m2", [cov[i, j] if present else 0
                                        for i, j in ((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))], 1)
    add("target.age_s", 1 if o["prediction_age_s"] is None else o["prediction_age_s"], .1)
    iq = quat_product(unyaw, q)
    vec("imu.orientation_heading_xyzw", np.array(iq)*(1 if iq[3] >= 0 else -1))
    vec("imu.gyro_gimbal_rad_s", imu["gyro_rad_s"], 10)
    add("imu.valid", imu["valid"])
    add("imu.age_s", (o["now_ns"]-imu["sample_ns"])*1e-9, .1)
    for slot in range(-1, 4):
        add(f"selection.previous.{slot}", o["previous_slot"] == slot)
    add("selection.age_s", o["selected_slot_age_s"] if o["previous_slot"] >= 0 else 0, 1)
    for slot, candidate in enumerate(o["candidates"]):
        p = f"candidate.{slot}."
        valid = candidate["valid"]
        add(p+"ballistic_valid", valid)
        add(p+"detected", candidate["detected"])
        add(p+"predicted", present)
        age = candidate["last_seen_age_s"]
        add(p+"seen_before", age is not None)
        add(p+"last_seen_age_s", 10 if age is None else age, 1)
        yaw_error = candidate["yaw_rad"]-f["yaw_rad"]
        add(p+"yaw_error_sin", math.sin(yaw_error) if valid else 0)
        add(p+"yaw_error_cos", math.cos(yaw_error) if valid else 0)
        add(p+"pitch_error_rad", candidate["pitch_rad"]-f["pitch_rad"] if valid else 0, 1)
        for key, scale in (("distance_m",10), ("fly_time_s",.5), ("prediction_horizon_s",.5)):
            add(p+key, candidate[key] if valid else 0, scale)
        for phase in ("facing_now_rad", "facing_impact_rad"):
            add(p+phase+".sin", math.sin(candidate[phase]) if valid else 0)
            add(p+phase+".cos", math.cos(candidate[phase]) if valid else 0)
    add("fire.request_previous", o["request_previous"])
    add("fire.request_age_known", o["since_request_s"] is not None)
    add("fire.since_request_s", 10 if o["since_request_s"] is None else o["since_request_s"], 1)
    add("fire.thermal_valid", thermal["valid"])
    add("fire.thermal_age_s", (o["now_ns"]-thermal["sample_ns"])*1e-9, .1)
    for key, scale in (("heat",400), ("heat_limit",400), ("base_cooling_per_second",100),
                       ("allowance_remaining",200), ("cooling_remaining_s",60),
                       ("effective_cooling_per_second",100)):
        add("fire."+key, thermal[key], scale)
    add("fire.cooling_active", thermal["cooling_active"])
    mask = o["action_mask"]
    if len(mask) != 9 or any(type(v) is not bool for v in mask):
        raise ValueError("nine boolean masks required")
    return names, np.asarray(values, dtype=np.float32).tolist(), list(mask)


_example = empty_observation()
_example.update(imu=dict(orientation_xyzw=[0,0,0,1], gyro_rad_s=[0]*3, valid=False, sample_ns=0),
                thermal=dict(valid=False, sample_ns=0, heat=0, heat_limit=0,
                             base_cooling_per_second=0, allowance_remaining=0,
                             cooling_remaining_s=0, effective_cooling_per_second=0,
                             cooling_active=False), now_ns=0, since_request_s=None,
                request_previous=False)
NAMES = encode(_example)[0]
SCHEMA = dict(version=VERSION, name=MODE, features=NAMES, history=8,
              normalization="signed softsign x/(scale+abs(x)); booleans/quaternions/trig unchanged",
              coordinates="gimbal-relative origin; heading frame X-forward Y-left Z-up")
FIRE_FEATURES = [i for i,n in enumerate(NAMES) if n.startswith("fire.")]
TARGET_FEATURES = [i for i,n in enumerate(NAMES)
                   if n.startswith(("target.", "candidate.", "selection."))]
