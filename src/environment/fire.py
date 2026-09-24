"""通过真实视觉估计与控制桥接，为火控任务提供 Gymnasium 采样环境。"""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys
import tempfile

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.environment.warmup import STEP_NS, WarmupConfig, WarmupSession
from src.policy.observations import HISTORY
from src.policy.decision import make_policy
from src.environment.spawn import reset_spawn
from src.transport.processes import training_worker
from src.transport.vision_bridge import vision_worker


class FireEnv(gym.Env):
    """管理一对仿真与桥接进程，每次 step 推进 10 ms，回合结束后需显式重置。

    返回观测时，C++ 回调处于暂停状态，或已缓存无需回调的控制命令。
    只有 step 才会提交该周期；准备观测不会推进物理世界。
    时间上限截断持续任务，延迟收益由训练器利用末观测进行价值自举，
    不将独立评估的尾部结算伤害追加为训练奖励。
    """

    metadata = {"render_modes": []}

    def __init__(self, *, simulator_root=None, vision_root=None, simulator_binary=None,
                 bridge_binary=None, simulator_config=None, log_dir=None,
                 episode_steps=3000, warmup_config=None, render_mode=None, decision_clock=None,
                 action_mode="fire_only"):
        super().__init__()
        if render_mode is not None:
            raise ValueError("FireEnv only supports render_mode=None")
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
        self.action_mode = action_mode
        self._history = make_policy(action_mode, decision_clock=decision_clock)
        self.action_space = spaces.Discrete(self._history.action_count)
        self.observation_space = spaces.Dict({
            "features": spaces.Box(-1.0, 1.0, (HISTORY, self._history.feature_count), np.float32),
            "valid": spaces.MultiBinary(HISTORY),
            "action_mask": spaces.MultiBinary(int(self.action_space.n)),
        })
        if self._history.clock.enabled:
            self.observation_space.spaces["decision_clock"] = spaces.Box(
                np.zeros(5, dtype=np.float32), np.array([1, 60, 60, 60, 60], dtype=np.float32))
        self._stack = None
        self._clock_log = None
        self._scene_log = None
        self._action_log = None
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
        self._scene_log = self._stack.enter_context((self._output / "scenes.jsonl").open("w"))
        self._action_log = self._stack.enter_context((self._output / "actions.jsonl").open("w"))
        if self._history.clock.enabled:
            self._clock_log = self._stack.enter_context((self._output / "decision-clock.jsonl").open("w"))
        self._client = self._stack.enter_context(training_worker(
            self.simulator_root, self.simulator_binary, self._output / "simulator.log",
            self.simulator_config))
        self._bridge = self._stack.enter_context(vision_worker(
            self.bridge_binary, self.vision_root, self._output / "bridge"))

    def _fail(self):
        """携带原异常退出工作进程上下文，保留原错误供调用方处理。"""
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

    def _scenario(self, options):
        """校验并补齐未采样的场景配置，由具体任务实现。"""
        raise NotImplementedError

    def _sample_scene(self, scenario):
        """静止任务直接沿用 Gym 出生随机流，保留旧 seed 的行为。"""
        return scenario, self.np_random, {}

    def prepare_scene(self, *, seed=None, options=None):
        """仅采样一次回合配置，供 Gym reset 与独立评估共用，不启动工作进程。

        显式 seed 重置场景序列；省略时继续。返回实际场景、独立出生流及复现信息。
        真值和采样信息仅供日志使用，不能加入策略观测。
        """
        scenario = self._scenario(options)
        super().reset(seed=seed)
        if seed is not None or not hasattr(self, "_scene_index"):
            self._scene_index = 0
        scenario, spawn_rng, sample = self._sample_scene(scenario)
        sample = {"episode_index": self._scene_index, **sample}
        self._scene_index += 1
        return scenario, spawn_rng, sample

    def reset(self, *, seed=None, options=None):
        """重置仿真和视觉状态，完成禁射预热后准备首个决策观测。

        Args:
            seed: Gym 随机种子，用于派生出生种子，与仿真种子不直接等同。
            options: 可选场景配置，仅支持 scenario 字段，运动约束由具体任务定义。

        Returns:
            (obs, info)，其中 obs 为独立数组副本，准备观测不额外推进物理。

        Raises:
            TimeoutError: 在规定时间内未完成预热。
        """
        scenario, spawn_rng, self._scene_sample = self.prepare_scene(seed=seed, options=options)
        self._status = "resetting"
        self._steps = 0
        self._damage = 0.0
        self._switches = 0
        self._last_request = {"action": None, "action_masked": False,
                              "shot_requested": False, "shot_accepted": False,
                              "reject_reason": None, "executed_action": None, "wire_action": None,
                              "selected_slot": -1, "slot_switched": False, "mask_reason": None}
        self._history.reset()
        self._reset_attempts = []
        try:
            self._start()
            self._bridge.cancel()
            warmup = WarmupSession(self._client, self._bridge, self.warmup_config)
            reset_spawn(warmup.reset, spawn_rng, scenario, self._reset_attempts)
            self._scene_info = {
                **self._scene_sample, "spawn_seed": self._reset_attempts[-1]["seed"],
                "scenario": copy.deepcopy(warmup.response["data"]["evaluation"]["scenario"]),
            }
            self._scene_log.write(json.dumps(
                {**self._scene_info, "reset_attempts": self._reset_attempts}, allow_nan=False) + "\n")
            self._scene_log.flush()
            while warmup.status == "warming":
                warmup.advance()
            if warmup.status != "ready":
                raise TimeoutError(f"fire-target warmup timed out: {warmup.info()}")
            self._response = warmup.response
            self._start_ns = self._response["sim_time_ns"]
            self._warmup_info = warmup.info()
            # 进程内通信计数器不能作为可复现的回合信息。
            self._warmup_info.pop("round_id")
            self._bridge.begin_training(self._response["round_id"], self._start_ns,
                                        policy_mode=self._history.bridge_mode)
            self._history.clock.start_episode()
            self._prepare()
            self._status = "ready"
            return self._observation(), self._info(0.0, None)
        except BaseException:
            # 中断也必须携带异常回收进程，不能对尚未确认的请求发送正常 Close。
            self._fail()
            raise

    def _prepare(self):
        self._prepared = self._bridge.prepare(self._response)
        self._history.begin_step()
        if self._prepared.get("kind") == "policy_observation":
            self._history.set_generation(self._prepared["metadata"]["track_generation"])
            self._history(self._prepared["observation"])
        else:
            # 跳过策略回调的周期仍占用一行历史，并推进一个物理步。
            if self._prepared["command"]["fire"]:
                raise RuntimeError("a skipped policy callback must not issue fire")
        self._obs = self._history.observation()

    def _observation(self):
        return {key: value.copy() for key, value in self._obs.items()}

    def action_masks(self):
        """返回当前动作空间布尔掩码的副本，必须先完成 reset。"""
        if self._status != "ready":
            raise RuntimeError("reset required before querying action masks")
        return self._obs["action_mask"].astype(bool, copy=True)

    def debug_snapshot(self):
        """返回独立的显示用真值副本，不推进仿真或准备策略观测。"""
        if self._response is None:
            raise RuntimeError("reset required before reading debug state")
        data = self._response["data"]
        return copy.deepcopy({
            "time_ns": self._response["sim_time_ns"],
            "data": {
                "feedback": {key: data["feedback"][key] for key in ("yaw_rad", "pitch_rad")},
                "evaluation": {key: data["evaluation"][key]
                               for key in ("scenario", "robots", "projectiles", "controlled_muzzle")},
                "events": data["events"],
            },
        })

    def _robots(self):
        return {r["robot_id"]: r for r in self._response["data"]["evaluation"]["robots"]}

    def _info(self, reward, reason):
        own = self._robots()[1]
        result = {"damage": reward, "episode_damage": self._damage,
                "action_mode": self.action_mode, "slot_switches": self._switches,
                "actual_shots": own["actual_shots"], "episode_steps": self._steps,
                "episode_time_s": self._steps * 0.01,
                "physical_time_ns": self._response["sim_time_ns"],
                "end_reason": reason, **self._last_request,
                "reset_attempts": copy.deepcopy(self._reset_attempts),
                "warmup": copy.deepcopy(self._warmup_info),
                "scene": copy.deepcopy(self._scene_info)}
        if self._history.clock.enabled:
            result["decision_clock"] = self._history.clock.info()
            result["physical_fire_legal_next"] = self._history.physical_fire_action is not None
        return result

    def step(self, action):
        """执行一个策略决策，推进 10 ms 并准备下一观测。

        Args:
            action: 整数动作，语义由 action_mode 确定；非法射击只降级为同板跟踪或等待。

        Returns:
            (obs, reward, terminated, truncated, info)。reward 为本步实际伤害；
            死亡或击毁目标标记 terminated，达到时限标记 truncated。
            回合结束后需显式 reset，不追加独立评估的尾部结算奖励。
        """
        if self._status != "ready":
            raise RuntimeError("reset required before step (episode ended or environment not ready)")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError(f"action must be an integer in [0,{self.action_space.n - 1}]")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0,{self.action_space.n - 1}]")
        action = int(action)
        try:
            clock_before = self._history.clock.info()
            physical_fire_legal = self._history.physical_fire_action is not None
            executed, wire_action = self._history.resolve_action(action)
            masked = executed != action
            clock_masked = (self._history.requests_fire(action) and clock_before is not None
                            and not clock_before["decision_due"])
            mask_reason = ("decision_clock" if clock_masked else "fire_control") if masked else None
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
            switched = control.get("slot_switched", False)
            self._switches += int(switched)
            self._last_request = {
                "action": action, "executed_action": executed, "wire_action": wire_action,
                "action_masked": masked, "mask_reason": mask_reason,
                "selected_slot": control["selected_slot"], "slot_switched": switched,
                "track_generation": control.get("track_generation"),
                "shot_requested": control.get("shot_requested", False),
                "shot_accepted": control.get("shot_accepted", False),
                "reject_reason": control.get("reject_reason_name"),
            }
            if clock_before is not None:
                self._last_request.update(
                    decision_clock_before=clock_before, physical_fire_legal_before=physical_fire_legal,
                    clock_masked=clock_masked)
            self._action_log.write(json.dumps({"episode_index": self._scene_info["episode_index"],
                                              "physical_time_ns": before, **self._last_request},
                                             allow_nan=False) + "\n")
            self._action_log.flush()
            # 完成控制步后即消耗到期机会，TRACK 或 LOST 周期也不例外。
            # 读取观测、跳过回调或重置目标代次均不触发随机抽样。
            self._history.clock.complete_step()
            if clock_before is not None and clock_before["decision_due"] and self._clock_log is not None:
                record = {"clock": self._history.clock.config, "before": clock_before,
                          "after": self._history.clock.info(), "physical_time_ns": before,
                          "action": action, "physical_fire_legal": physical_fire_legal,
                          "shot_requested": self._last_request["shot_requested"],
                          "shot_accepted": self._last_request["shot_accepted"],
                          "reject_reason": self._last_request["reject_reason"]}
                self._clock_log.write(json.dumps(record, allow_nan=False) + "\n")
                self._clock_log.flush()
            robots = self._robots()
            reason = ("controlled_dead" if robots[1]["hp"] <= 0 else
                      "target_destroyed" if robots[2]["hp"] <= 0 else None)
            terminated = reason is not None
            truncated = not terminated and self._steps >= self.episode_steps
            if truncated:
                reason = "time_limit"
            # 到达时间上限时仍准备真实末观测，供训练器进行价值自举。
            # 随后取消该周期，不提交新动作或额外推进物理。
            self._prepare()
            if terminated or truncated:
                self._bridge.cancel()
                self._prepared = None
                self._status = "complete"
            return self._observation(), reward, terminated, truncated, self._info(reward, reason)
        except BaseException:
            self._fail()
            raise

    def close(self):
        """取消待处理周期并回收本环境拥有的进程，可重复调用。"""
        stack, self._stack = self._stack, None
        try:
            if stack is not None:
                stack.close()
        finally:
            self._client = self._bridge = self._prepared = None
            self._status = "closed"
