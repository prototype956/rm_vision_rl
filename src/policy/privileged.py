"""Versioned critic-only current-state whitelist; never flatten the evaluation dictionary.

Bevy coordinates are X/Z horizontal and Y up; armor geometry is explicitly ROS.
Fixed-size collections carry validity bits; overflow fails, it is never silently dropped.
Absolute timestamps/IDs are used only for pairing, ordering and deriving intervals.
"""
import numpy as np
from src.policy.asymmetric_observations import bounded

MAX_PROJECTILES = 256
MAX_COMMANDS = 64
MAX_REQUESTS = 16
CONTRACT = dict(version=1, robot_ids=[1, 2], armors_per_robot=4,
                projectiles=MAX_PROJECTILES, pending_commands=MAX_COMMANDS,
                pending_requests=MAX_REQUESTS, normalization="softsign",
                overflow="error", time="same decision tick; remaining/elapsed intervals only")


def encode(state, now_ns):
    if state["version"] != 1 or state["sample_ns"] != now_ns:
        raise ValueError("critic truth must match the decision timestamp")
    values = []

    def add(value, scale=1):
        values.append(bounded(value, scale))

    def vec(data, scale=1):
        for value in data:
            add(value, scale)

    def collection(data, capacity, width, encoder):
        if len(data) > capacity:
            raise ValueError("critic collection overflow; increase schema capacity and retrain")
        for item in data:
            start = len(values)
            add(1)
            encoder(item)
            if len(values)-start != width:
                raise ValueError("critic collection schema width mismatch")
        values.extend([0.] * (capacity-len(data))*width)

    robots = {r["robot_id"]: r for r in state["robots"]}
    geometry = state["geometry"]
    if set(robots) != {1, 2}:
        raise ValueError("critic v1 requires exactly own and opponent robots")
    for owner in (1, 2):
        r = robots[owner]
        vec(r["position_bevy_m"], 10)
        vec(r["rotation_bevy_xyzw"])
        vec(r["velocity_bevy_m_s"], 10)
        vec(r["angular_velocity_bevy_rad_s"], 10)
        for key, scale in (("hp",400),("max_hp",400),("heat",400),("heat_limit",400),
                           ("base_cooling_per_second",100),("effective_cooling_per_second",100),
                           ("allowance_remaining",200),("cooling_locked",1),("round_locked",1),
                           ("cooling_active",1),("cooling_remaining_s",60),
                           ("defense_active",1),("defense_remaining_s",2),
                           ("fold",1),("lift_m",.5),("fire_ready",1),("blocked",1),
                           ("requested_fold",1),("stable_s",.1),
                           ("requested_lift_m",.5),("lift_velocity_m_s",1)):
            add(r[key] is not None if key == "blocked" else r[key],scale)
        add(-1 if r["region"] is None else r["region"], 16)
        vec(r["tunnel_progress"] if r["tunnel_progress"] is not None else [-1,0], 3)
        for key in ("pending_shot_s", "last_shot_age_s"):
            add(r[key] is not None)
            add(0 if r[key] is None else r[key], .1)
        ch = next(c for c in geometry["chassis"] if c["owner"] == owner)
        add(np.sin(ch["yaw_rad"])); add(np.cos(ch["yaw_rad"]))
        add(ch["yaw_velocity_rad_s"],10)
        vec(ch["rotation_bevy_xyzw"])
        gun = next(c for c in geometry["muzzles"] if c["owner"] == owner)
        vec(gun["position_bevy_m"],10); vec(gun["rotation_bevy_xyzw"])
        plates = [p for p in geometry["armors"] if p["owner"] == owner]
        if len(plates) != 4:
            raise ValueError("critic requires four armor geometries per robot")
        for plate in plates:
            vec(plate["position_ros_m"],10); vec(plate["rotation_ros_xyzw"])
            vec(np.asarray(plate["corners_ros_m"]).flatten(),10)
    a = state["actuator"]
    for key in ("yaw_rad", "pitch_rad", "yaw_velocity_rad_s", "pitch_velocity_rad_s",
                "target_yaw_rad", "target_pitch_rad", "yaw_acceleration_rad_s2",
                "pitch_acceleration_rad_s2", "command_valid", "saturation_flags"):
        add(a[key],10)

    def projectile(p):
        add(p["owner"] == 1); vec(p["position_bevy_m"],10)
        vec(p["velocity_bevy_m_s"],30); add(p["remaining_s"],5)
    collection(state["projectiles"], MAX_PROJECTILES, 9, projectile)

    def command(c):
        add(c["fire_level"])
        add(c["stage"] == "inbox")
        epoch = 1_000_000_000_000_000_000
        add((now_ns-(c["command_timestamp_ns"]-epoch))*1e-9,.1)
        activation = c["activate_timestamp_ns"]
        add(activation is not None)
        add(0 if activation is None else max(0,activation-epoch-now_ns)*1e-9,.1)
        add(c["yaw_rad"],np.pi); add(c["pitch_rad"],np.pi)
    collection(state["pending_commands"], MAX_COMMANDS, 8, command)

    def request(r):
        add(r["robot_id"] == 1); add((now_ns-r["requested_at_ns"])*1e-9,.1)
    collection(state["pending_requests"], MAX_REQUESTS, 3, request)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (FEATURE_COUNT,):
        raise ValueError(f"critic schema mismatch: {result.shape}, expected {FEATURE_COUNT}")
    return result


# Per robot: 13 pose/motion + 21 resources/mechanisms + 3 region/progress + 4 pending/last
# + 7 chassis + 7 muzzle + four (3+4+12) armor geometries.
FEATURE_COUNT = 2*(13+21+3+4+7+7+4*19)+10+MAX_PROJECTILES*9+MAX_COMMANDS*8+MAX_REQUESTS*3
CONTRACT["feature_count"] = FEATURE_COUNT
