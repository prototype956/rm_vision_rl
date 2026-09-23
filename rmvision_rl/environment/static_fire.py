"""Gymnasium sampling of static targets through the real estimator and control bridge."""
from contextlib import ExitStack
import copy
from pathlib import Path
import sys
import tempfile

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from rmvision_rl.environment.warmup import STEP_NS, WarmupConfig, WarmupSession
from rmvision_rl.policy.observations import FEATURE_NAMES, HISTORY
from rmvision_rl.policy.static_fire import StaticFirePolicy
from rmvision_rl.environment.spawn import reset_spawn
from rmvision_rl.transport.processes import training_worker
from rmvision_rl.transport.vision_bridge import vision_worker


class StaticFireEnv(gym.Env):
    """One simulator/bridge pair, one 10 ms transition per step, no automatic Reset.

    A returned observation owns a paused C++ callback (or a cached no-callback command).
    Only step submits that cycle. Preparation never advances the physical world.
    Time limits cut a continuing task; delayed outcomes are estimated by the learner's
    terminal-observation bootstrap, not by folding evaluation settlement into rewards.
    """

    metadata = {"render_modes": []}

    def __init__(self, *, simulator_root=None, vision_root=None, simulator_binary=None,
                 bridge_binary=None, simulator_config=None, log_dir=None,
                 episode_steps=3000, warmup_config=None, render_mode=None):
        super().__init__()
        if render_mode is not None:
            raise ValueError("StaticFireEnv only supports render_mode=None")
        if type(episode_steps) is not int or episode_steps < 1:
            raise ValueError("episode_steps must be a positive integer")
        root = Path(__file__).resolve().parents[2]
        self.simulator_root = Path(simulator_root or root.parent / "rm_simulator_2027").resolve()
        self.vision_root = Path(vision_root or root.parent / "rm_vision_2027").resolve()
        self.simulator_binary = Path(
            simulator_binary or self.simulator_root / "target/release/daedalus_training").resolve()
        self.bridge_binary = Path(
            bridge_binary or root / "artifacts/build/vision-bridge/rmvision-rl-bridge").resolve()
        self.simulator_config = Path(
            simulator_config or self.simulator_root / "config.toml").resolve()
        self.log_dir = Path(log_dir or root / "artifacts/gym").resolve()
        self.episode_steps = episode_steps
        self.warmup_config = warmup_config or WarmupConfig()
        self.render_mode = None
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict({
            "features": spaces.Box(-1.0, 1.0, (HISTORY, len(FEATURE_NAMES)), np.float32),
            "valid": spaces.MultiBinary(HISTORY),
            "action_mask": spaces.MultiBinary(2),
        })
        self._history = StaticFirePolicy()
        self._stack = None
        self._client = self._bridge = None
        self._prepared = self._response = self._obs = None
        self._status = "idle"

    def _start(self):
        if self._stack is not None:
            return
        for path in (self.simulator_binary, self.bridge_binary, self.simulator_config):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._output = Path(tempfile.mkdtemp(prefix="env-", dir=self.log_dir))
        self._stack = ExitStack()
        self._client = self._stack.enter_context(training_worker(
            self.simulator_root, self.simulator_binary, self._output / "simulator.log",
            self.simulator_config))
        self._bridge = self._stack.enter_context(vision_worker(
            self.bridge_binary, self.vision_root, self._output / "bridge"))

    def _fail(self):
        """Unwind worker contexts with the original error, preserving it for the caller."""
        error = sys.exc_info()
        self._status = "fault"
        stack, self._stack = self._stack, None
        try:
            if stack is not None:
                stack.__exit__(*error)
        except Exception as cleanup_error:
            if hasattr(error[1], "add_note"):
                error[1].add_note(f"worker cleanup: {cleanup_error}")
        finally:
            self._client = self._bridge = self._prepared = None

    @staticmethod
    def _scenario(options):
        options = {} if options is None else copy.deepcopy(options)
        if not isinstance(options, dict) or set(options) - {"scenario"}:
            raise ValueError("reset options only supports 'scenario'")
        scenario = options.get("scenario", {})
        if not isinstance(scenario, dict):
            raise ValueError("scenario must be a dictionary")
        scenario.setdefault("motion", {"kind": "static"})
        if scenario["motion"] != {"kind": "static"}:
            raise ValueError("StaticFireEnv requires static motion")
        scenario.setdefault("target_distance_m", 4.0)
        scenario.setdefault("target_hp", 100000)
        measurements = scenario.setdefault("measurements", {})
        if not isinstance(measurements, dict):
            raise ValueError("static target training requires detector measurements")
        for key, value in {"noise_std_px": 0.25, "latency_ms": 20,
                           "dropout_probability": 0.0}.items():
            measurements.setdefault(key, value)
        return scenario

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = self._scenario(options)
        self._status = "resetting"
        self._steps = 0
        self._damage = 0.0
        self._last_request = {"action": None, "action_masked": False,
                              "shot_requested": False, "shot_accepted": False,
                              "reject_reason": None}
        self._history.reset()
        self._reset_attempts = []
        try:
            self._start()
            self._bridge.cancel()
            warmup = WarmupSession(self._client, self._bridge, self.warmup_config)
            reset_spawn(warmup.reset, self.np_random, scenario, self._reset_attempts)
            while warmup.status == "warming":
                warmup.advance()
            if warmup.status != "ready":
                raise TimeoutError(f"static-target warmup timed out: {warmup.info()}")
            self._response = warmup.response
            self._start_ns = self._response["sim_time_ns"]
            self._warmup_info = warmup.info()
            # A process-local transport counter is not reproducible episode information.
            self._warmup_info.pop("round_id")
            self._bridge.begin_training(self._response["round_id"], self._start_ns)
            self._prepare()
            self._status = "ready"
            return self._observation(), self._info(0.0, None)
        except Exception:
            self._fail()
            raise

    def _prepare(self):
        self._prepared = self._bridge.prepare(self._response)
        self._history.begin_step()
        if self._prepared.get("kind") == "policy_observation":
            self._history.set_generation(self._prepared["metadata"]["track_generation"])
            self._history(self._prepared["observation"])
        else:
            # A skipped callback still occupies one history row and one physical step.
            if self._prepared["command"]["fire"]:
                raise RuntimeError("a skipped policy callback must not issue fire")
        self._track_action, self._fire_action = self._history.track_action, self._history.fire_action
        self._obs = self._history.observation()

    def _observation(self):
        return {key: value.copy() for key, value in self._obs.items()}

    def action_masks(self):
        if self._status != "ready":
            raise RuntimeError("reset required before querying action masks")
        return self._obs["action_mask"].astype(bool, copy=True)

    def _robots(self):
        return {r["robot_id"]: r for r in self._response["data"]["evaluation"]["robots"]}

    def _info(self, reward, reason):
        own = self._robots()[1]
        return {"damage": reward, "episode_damage": self._damage,
                "actual_shots": own["actual_shots"], "episode_steps": self._steps,
                "episode_time_s": self._steps * 0.01,
                "physical_time_ns": self._response["sim_time_ns"],
                "end_reason": reason, **self._last_request,
                "reset_attempts": copy.deepcopy(self._reset_attempts),
                "warmup": copy.deepcopy(self._warmup_info)}

    def step(self, action):
        if self._status != "ready":
            raise RuntimeError("reset required before step (episode ended or environment not ready)")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be an integer in {0,1}")
        if not self.action_space.contains(action):
            raise ValueError("action must be an integer in {0,1}")
        action = int(action)
        try:
            masked = action == 1 and self._fire_action is None
            wire_action = self._fire_action if action == 1 and not masked else self._track_action
            result = (self._bridge.submit(wire_action)
                      if self._prepared.get("kind") == "policy_observation" else self._prepared)
            before = self._response["sim_time_ns"]
            self._response = self._client.advance(**result["command"])
            self._bridge.ack(True)
            if self._response["sim_time_ns"] != before + STEP_NS:
                raise RuntimeError("Gym transition did not advance exactly 10 ms")
            self._steps += 1
            reward = float(self._response["data"]["reward_damage"])
            self._damage += reward
            control = result["control"]
            self._last_request = {
                "action": action, "action_masked": masked,
                "shot_requested": control.get("shot_requested", False),
                "shot_accepted": control.get("shot_accepted", False),
                "reject_reason": control.get("reject_reason_name"),
            }
            robots = self._robots()
            reason = ("controlled_dead" if robots[1]["hp"] <= 0 else
                      "target_destroyed" if robots[2]["hp"] <= 0 else None)
            terminated = reason is not None
            truncated = not terminated and self._steps >= self.episode_steps
            if truncated:
                reason = "time_limit"
            # Prepare the actual next observation even at the time limit. Cancel without
            # submitting it afterwards; no new action or physical step is taken at the boundary.
            self._prepare()
            if terminated or truncated:
                self._bridge.cancel()
                self._prepared = None
                self._status = "complete"
            return self._observation(), reward, terminated, truncated, self._info(reward, reason)
        except Exception:
            self._fail()
            raise

    def close(self):
        stack, self._stack = self._stack, None
        try:
            if stack is not None:
                stack.close()
        finally:
            self._client = self._bridge = self._prepared = None
            self._status = "closed"
