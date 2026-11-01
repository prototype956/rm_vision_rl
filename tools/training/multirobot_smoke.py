"""NUC 单环境验收，不创建或更新模型，不操作桌面。"""
import argparse
import json
import re
from pathlib import Path
import tempfile

import numpy as np

from src.environment.multi_robot_fire import MultiRobotFireEnv
from src.environment.static_fire import StaticFireEnv
from src.training.config import read_config, ROOT


def action_for(obs):
    """只使用合法动作掩码，禁止用目标真值替代策略选甲。"""
    legal = np.flatnonzero(obs["action_mask"])
    fire = [int(x) for x in legal if x > 0 and x % 2 == 0]
    track = [int(x) for x in legal if x % 2 == 1]
    return fire[0] if fire else track[0] if track else 0


def episode(env, scene, steps, seed=17):
    obs, info = env.reset(seed=seed, options={"scenario": scene})
    assert env.observation_space.contains(obs)
    initial_hp = env.debug_snapshot()["data"]["evaluation"]["robots"][1]["hp"]
    reward_sum = 0.0
    invalid = recovered = False
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(action_for(obs))
        assert env.observation_space.contains(obs)
        assert not info["action_masked"]
        reward_sum += reward
        if not obs["valid"][-1]:
            invalid = True
        elif invalid:
            recovered = True
        if terminated or truncated:
            break
    robots = env.debug_snapshot()["data"]["evaluation"]["robots"]
    own, target = robots
    assert abs(reward_sum - info["episode_damage"]) < 1e-9
    assert reward_sum == initial_hp - target["hp"]
    assert target["actual_shots"] == 0
    return dict(end_reason=info["end_reason"], steps=info["episode_steps"],
                damage=reward_sum, actual_shots=info["actual_shots"],
                initial_target_hp=initial_hp, final_target_hp=target["hp"],
                own_resources=own.get("resources"), target_profile=target.get("profile"),
                saw_invalid=invalid, recovered=recovered,
                observation_shapes={k: list(v.shape) for k, v in obs.items()},
                scene=info["scene"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--simulator-root", type=Path)
    parser.add_argument("--vision-root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/multirobot-check")
    parser.add_argument("--quick", action="store_true", help="only config/sampling and a 1s closed loop")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for role in ("infantry", "sentry"):
        config = read_config(ROOT / f"config/training/rmul_2027_{role}_curriculum_v2.json")
        assert config["curriculum"]["version"] == 2
        assert config["environment"]["scenario"]["duel"]["controlled"]["preset"] == (
            "sentry" if role == "sentry" else "infantry_health_cooling")
    kwargs = dict(action_mode="joint", selection_memory=True,
                  simulator_root=args.simulator_root, vision_root=args.vision_root,
                  log_dir=args.output / "episodes")
    sampler = MultiRobotFireEnv(**kwargs)
    first = sampler.prepare_scene(seed=17)[0]
    assert first == sampler.prepare_scene(seed=17)[0]
    seen = set()
    for seed in range(500):
        scene = sampler.prepare_scene(seed=seed)[0]["duel"]
        seen.add((scene["controlled"]["profile"], scene["target"]["profile"],
                  scene["target"].get("preset")))
    assert len(seen) == 36, len(seen)  # one own preset x four chassis x nine targets
    sampler.close()
    cases = [
        ("infantry_static", {"controlled": {"profile": "standard", "preset": "infantry_health_cooling", "motion": {"kind": "static"}},
                             "target": {"profile": "standard", "preset": "infantry_health_cooling", "motion": {"kind": "static"}}}, 3000),
        ("heavy_target", {"controlled": {"profile": "standard", "motion": {"kind": "static"}},
                          "target": {"profile": "heavy_target", "motion": {"kind": "static"}}}, 3000),
        ("sentry_low", {"controlled": {"profile": "low", "preset": "sentry", "motion": {"kind": "static"}},
                        "target": {"profile": "folding", "preset": "sentry", "motion": {"kind": "static"}}}, 600),
        ("moving_camera", {"controlled": {"profile": "wheel_leg", "motion": {"kind": "lift"}},
                           "target": {"profile": "standard", "motion": {"kind": "combined", "amplitude_m": 0.2, "speed_m_s": 0.3}}}, 500),
        ("own_fold_tunnel", {"controlled": {"profile": "folding", "motion": {"kind": "tunnel"}},
                             "target": {"profile": "standard", "motion": {"kind": "static"}}}, 900),
    ]
    if args.quick:
        cases = [(cases[0][0], cases[0][1], 100)]
    results = {}
    for name, duel, count in cases:
        env = MultiRobotFireEnv(episode_steps=3000, **kwargs)
        try:
            result = episode(env, {"duel": duel}, count)
            if not args.quick and name in ("infantry_static", "heavy_target"):
                assert result["end_reason"] == "target_destroyed", result
            if name == "sentry_low":
                assert result["own_resources"]["allowance"]["remaining"] <= 750
            if name == "own_fold_tunnel":
                assert result["saw_invalid"] and result["recovered"], result
            results[name] = result
            print(name, json.dumps({k: result[k] for k in ("end_reason", "steps", "damage", "actual_shots", "recovered")}), flush=True)
        finally:
            env.close()
        (args.output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if not args.quick:
        # Deliberately reduced one-shot allowance is a diagnostic fixture, never a new
        # training preset. Exercise the same finite-episode drain path against real IPC.
        root = MultiRobotFireEnv(**kwargs).simulator_root
        with tempfile.TemporaryDirectory(prefix="rmul-ammo-check-") as temp:
            config = Path(temp) / "one-shot.toml"
            text = (root / "config.toml").read_text()
            text, count = re.subn(r"(?m)^controlled_allowance\s*=.*$",
                                  'controlled_allowance = { mode="limited", initial=1 }',
                                  text, count=1)
            if count == 0:
                if "[combat]" in text:
                    raise ValueError("diagnostic fixture cannot locate controlled_allowance")
                text += '\n[combat]\ncontrolled_allowance = { mode="limited", initial=1 }\n'
            config.write_text(text)
            env = StaticFireEnv(episode_steps=3000, simulator_config=config, **kwargs)
            env.finite_episode = True
            try:
                result = episode(env, {}, 3000)
                assert result["end_reason"] == "ammo_exhausted", result
                assert result["actual_shots"] == 1 and result["damage"] > 0, result
                results["one_shot_drain_fixture"] = result
            finally:
                env.close()
        (args.output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("MULTIROBOT_GYM_OK", flush=True)


if __name__ == "__main__":
    main()
