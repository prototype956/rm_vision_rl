"""CPU Actor export and stateful loading; no Critic or training imports required."""
import copy
import hashlib
import json
from pathlib import Path
from time import monotonic_ns

import numpy as np
import torch

from src.policy.recurrent_actor import CONTRACT
from src.policy.recurrent_observations import ACTOR_KEYS, SCHEMA, RecurrentObservationPolicy


def _field_contract(name):
    """Physical units before the inherited v4 encoder; every entry has an explicit scale."""
    scale, unit = None, "dimensionless"
    scales = (("position_heading_m.", 10, "m"), ("relative_velocity_heading_mps.", 10, "m/s"),
              ("radii_m.", .5, "m"), ("height_offset_m", .5, "m"), ("armor_tilt_rad", 1, "rad"),
              ("yaw_velocity_rad_s", 10, "rad/s"), ("yaw_variance_rad2", 1, "rad^2"),
              ("position_covariance_m2.", 1, "m^2"), ("gyro_gimbal_rad_s.", 10, "rad/s"),
              ("pitch_error_rad", 1, "rad"), ("distance_m", 10, "m"),
              ("fly_time_s", .5, "s"), ("prediction_horizon_s", .5, "s"),
              ("heat_limit", 400, "heat"), ("fire.heat", 400, "heat"),
              ("cooling_per_second", 100, "heat/s"), ("allowance_remaining", 200, "projectiles"),
              ("cooling_remaining_s", 60, "s"), ("since_request_s", 1, "s"))
    for match, value, physical_unit in scales:
        if match in name:
            scale, unit = value, physical_unit
            break
    if "age_s" in name:
        unit = "s"
        scale = .1 if name in ("target.age_s", "imu.age_s", "fire.thermal_age_s") else 1
    return dict(name=name, unit=unit, scale=scale,
                normalization="identity" if scale is None else "x/(scale+abs(x))")


def export_actor(actor, directory, *, config=None, training_steps=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    module = torch.jit.script(copy.deepcopy(actor).cpu().eval())
    model_file = directory / "actor.pt"
    torch.jit.save(module, str(model_file))
    manifest = dict(format_version=1, kind="asymmetric_gru_actor", contract=CONTRACT,
        observation=SCHEMA, features=[_field_contract(n) for n in SCHEMA["features"]],
        input_keys=list(ACTOR_KEYS),
        inputs={"features": ["B", len(SCHEMA["features"])], "valid": ["B", 1],
                "action_mask": ["B", 9], "selection_memory": ["B", 5], "decision_clock": ["B", 5],
                "dt_s": ["B", 1], "episode_start": ["B", 1], "target_reset": ["B", 1],
                "candidate_map": ["B", 4]},
        state_shapes=[["B",128], ["B",4,64], ["B",64], ["B",64]],
        outputs=["selection_probabilities[B,5]", "conditional_fire_probabilities[B,4]", "action[B]", "state"],
        dt="physical or monotonic seconds since previous observation, first=0; dt/(0.1+dt)",
        resets="episode clears all; target clears target/candidates; candidate_map=new_to_old",
        decision_clock="[due, wait_s, configured_min_interval_s, configured_max_interval_s, episode_fixed_interval_s]; x/(1+abs(x))",
        selection_memory="one-hot previous distinct confirmed slot: [-1,0,1,2,3]",
        sha256=hashlib.sha256(model_file.read_bytes()).hexdigest(), training_steps=training_steps)
    if config is not None:
        deployment = copy.deepcopy(config)
        for key in ("simulator_root", "vision_root", "simulator_binary", "bridge_binary", "simulator_config"):
            deployment["environment"].pop(key, None)
        if "curriculum" in config:
            # NUC needs the stable v5 Actor API, not server-only curriculum/reward/logging.
            manifest["training_context"] = dict(curriculum_version=config["curriculum"]["version"], seed=config["seed"],
                                                thermal_reward=config.get("thermal_reward"))
            for key in ("curriculum", "runtime", "thermal_reward"):
                deployment.pop(key, None)
            duel = deployment["environment"]["scenario"]["duel"]
            duel["controlled"]["motion"] = dict(kind="static")
            duel["target"] = dict(profile="standard", preset="infantry_health_cooling",
                motion=dict(kind="combined", amplitude_m=.5, speed_m_s=.5, angular_speed_rad_s=1.))
        manifest["simulation_config"] = deployment
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return directory


def load_actor(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "asymmetric_gru_actor" or manifest.get("contract") != CONTRACT or manifest.get("observation") != SCHEMA:
        raise ValueError("Actor observation/network contract mismatch")
    path = directory / "actor.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Actor checksum mismatch")
    return torch.jit.load(str(path), map_location="cpu").eval(), manifest


@torch.inference_mode()
def verify_export(actor, directory, observations):
    """Replay a real episode through eager and exported CPU Actor, without Critic."""
    reference = copy.deepcopy(actor).cpu().eval()
    deployed, manifest = load_actor(directory)
    expected_state, actual_state = reference.initial_state(1), deployed.initial_state(1)
    steps = len(observations["features"])
    max_error = 0.0
    for t in range(steps):
        obs = {k: torch.as_tensor(observations[k][t:t+1]) for k in ACTOR_KEYS}
        expected = reference(obs, expected_state)
        actual = deployed(obs, actual_state)
        expected_state, actual_state = expected[3], actual[3]
        for a, b in zip(expected[:3], actual[:3]):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        for a, b in zip(expected_state, actual_state):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        max_error = max(max_error, (expected[0]-actual[0]).abs().max().item(),
                        (expected[1]-actual[1]).abs().max().item())
    report = dict(steps=steps, deterministic_action_mismatches=0, max_probability_error=max_error,
                  actor_sha256=manifest["sha256"], critic_input_used=False,
                  reference="trained eager Actor with identical parameters on CPU")
    (Path(directory) / "export-verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class ActorRunner:
    """One runner per environment/session; each observation must be supplied once."""
    def __init__(self, directory, *, threads=1):
        torch.set_num_threads(threads)
        self.model, self.manifest = load_actor(directory)
        self.pending = None
        self.reset()

    def reset(self):
        if self.pending is not None:
            self.model, self.manifest = self.pending
            self.pending = None
        self.state = self.model.initial_state(1)
        self.started = False

    def request_model(self, directory):
        # Validate before scheduling; a running episode always retains its old model.
        self.pending = load_actor(directory)

    @torch.inference_mode()
    def predict(self, observation):
        for key, shape in self.manifest["inputs"].items():
            array = np.asarray(observation[key])
            if array.shape != tuple(shape[1:]) or not np.isfinite(array).all():
                raise ValueError(f"invalid Actor input: {key}")
        if sorted(np.asarray(observation["candidate_map"]).tolist()) != [0, 1, 2, 3]:
            raise ValueError("candidate_map must map four new slots to four distinct old slots")
        mask = np.asarray(observation["action_mask"], dtype=bool)
        if not mask[[0,1,3,5,7]].any() or (mask[[2,4,6,8]] & ~mask[[1,3,5,7]]).any():
            raise ValueError("invalid selection/conditional firing mask")
        if observation["dt_s"][0] < 0 or (observation["episode_start"][0] and observation["dt_s"][0] != 0):
            raise ValueError("dt_s must be nonnegative and zero at an episode start")
        if bool(observation["episode_start"][0]):
            self.reset()
        elif not self.started:
            raise ValueError("a deployment session must start at an episode boundary")
        obs = {k: torch.as_tensor(np.asarray(observation[k])[None]) for k in ACTOR_KEYS}
        selection, fire, action, self.state = self.model(obs, self.state)
        self.started = True
        return int(action.item()), selection[0].numpy(), fire[0].numpy()


class LiveObservationAdapter(RecurrentObservationPolicy):
    """Real-device API boundary. Sensor sample_ns must use the same monotonic clock.

    The caller supplies estimated target data, sensor feedback and tracker events;
    physical serial transport is deliberately outside this adapter.
    When using DecisionClock, the caller starts it after reset/warmup and calls
    clock.complete_step() after each completed 10 ms control cycle, as FireEnv does.
    """
    def observe(self, raw, packet, metadata, *, last_request_ns=None, request_previous=False, now_ns=None):
        from src.policy.asymmetric_observations import adapt_observation
        now_ns = monotonic_ns() if now_ns is None else now_ns
        self.begin_step()
        self.set_tracking_metadata(metadata)
        self(adapt_observation(raw, packet, now_ns, last_request_ns=last_request_ns,
                               request_previous=request_previous))
        return self.observation()
