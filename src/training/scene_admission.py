"""Control-plane checks: physical fireability is never an Actor action mask."""
import numpy as np


def difficulty_signature(descriptor):
    """Pose retries may change bearings, never task parameters or motion strata."""
    import copy
    value=copy.deepcopy(descriptor)
    for key in ("attempt","admitted","visibility_certificate","placement_search"):
        value.pop(key,None)
    for robot in value["scenario"]["duel"].values():
        robot.pop("position_xz_m",None)
        robot.pop("yaw_rad",None)
        # World heading follows the resampled sight bearing; its acute-angle
        # stratum, angle and direction remain explicitly bound below.
        robot["motion"].pop("translation_heading_rad",None)
    return value


def validate_visibility_certificate(descriptor, native_sha256, simulator_config_sha256):
    """Release gate binds all frozen cases to the actual admitted task/binary."""
    from src.training.curriculum_v2 import digest
    if descriptor.get("release_revision",1)<3:
        return
    certificate=descriptor.get("visibility_certificate",{})
    proof=certificate.get("proof",{})
    if (descriptor["scenario"].get("visibility_admission")!="full_path_clear"
        or not descriptor.get("admitted")
        or certificate.get("scenario_sha256")!=digest(descriptor["scenario"])
        or certificate.get("native_sha256")!=native_sha256
        or certificate.get("simulator_config_sha256")!=simulator_config_sha256
        or proof.get("version")!=1 or proof.get("policy")!="full_path_clear"
        or proof.get("duration_ms")!=45000 or proof.get("max_interval_ms")!=10
        or proof.get("min_interval_ms")!=1 or proof.get("min_visible_armor_count",0)<1
        or not proof.get("algorithm_fingerprint_fnv64") or proof.get("policy_inputs") is not False):
        raise ValueError("frozen full_path_clear certificate missing or mismatched")


def check_fireability(probe):
    visible = [p for p in probe["camera_view"]["plates"] if p["visibility"] == "visible"]
    for path in probe["probes"]:
        if not path["path_clear"]:
            continue
        x,y,z = path["target_bevy_m"]
        ros = np.array([-z,-x,y])
        if any(np.linalg.norm(ros-np.asarray(p["center_world"])) < .02 for p in visible):
            return
    raise RuntimeError("invalid explicit duel: no common camera-visible and ballistic-clear armor")


def check_trackability(descriptor):
    """Full 30 s center-bearing screen using the same prescribed sinusoidal motion.

    Armor spin does not equal required gimbal yaw rate. This checks the chassis
    bearing relative to our scripted yaw; estimator/plate selection is evaluated
    by the real controller separately. No future plan is exposed to either network.
    """
    duel = descriptor["scenario"]["duel"]
    time = np.arange(0.,30.001,.01)
    def path(robot):
        start = np.asarray(robot["position_xz_m"],float)
        motion = robot["motion"]
        yaw = robot.get("yaw_rad",0.)
        translation_yaw = motion.get("translation_heading_rad", yaw)
        values = np.repeat(start[None],len(time),axis=0)
        if motion["kind"] in ("translation","combined"):
            displacement = motion["amplitude_m"]*np.sin(time*motion["speed_m_s"]/motion["amplitude_m"])
            values += displacement[:,None]*np.array([np.cos(translation_yaw),-np.sin(translation_yaw)])[None]
        angles = yaw + (time*motion["angular_speed_rad_s"] if motion["kind"] in ("rotation","combined") else 0.)
        return values,angles
    own,own_yaw = path(duel["controlled"])
    target,_ = path(duel["target"])
    delta = target-own
    distance = np.linalg.norm(delta,axis=1)
    bearing = np.unwrap(np.arctan2(-delta[:,0],-delta[:,1]))-own_yaw
    rates = np.abs(np.gradient(bearing,.01))
    feasible = (rates <= 3.) & (distance >= .6)
    longest = current = 0
    for valid in feasible:
        current = current+1 if valid else 0
        longest = max(longest,current)
    if longest < 100:
        raise RuntimeError("invalid explicit duel: no one-second center-bearing window within gimbal yaw speed")
    return dict(max_center_yaw_rate_rad_s=float(rates.max()),
                longest_rate_feasible_window_s=longest*.01,
                minimum_horizontal_distance_m=float(distance.min()), sampled_step_s=.01,
                criterion="chassis center bearing only; plate tracking/ballistics verified separately")
