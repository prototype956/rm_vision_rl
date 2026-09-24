"""在同一组合法旋转场景上成对评估联合策略与旧两动作策略。"""
import argparse
import copy
import csv
from html import escape
import json
from io import BytesIO
import hashlib
from pathlib import Path
import tempfile

import numpy as np

from src.environment.spawn import reset_spawn
from src.training.config import ROOT
from src.training.environment import make_environment
from src.training.models import read_checkpoint_metadata, load_model
from src.transport.processes import training_worker
from tools.training.view import _atomic_json, evaluate_checkpoint


def compare_checkpoints(checkpoint, baseline_checkpoint, *, output_dir=None, device=None):
    checkpoints = {"joint": Path(checkpoint).resolve(), "baseline": Path(baseline_checkpoint).resolve()}
    checkpoint_data = {key: path.read_bytes() for key, path in checkpoints.items()}
    metadata = {key: read_checkpoint_metadata(BytesIO(data)) for key, data in checkpoint_data.items()}
    configs = {key: item["config"] for key, item in metadata.items()}
    settings = {key: value["environment"] for key, value in configs.items()}
    if (settings["joint"].get("action_mode", "fire_only") != "joint"
            or settings["baseline"].get("action_mode", "fire_only") != "fire_only"):
        raise ValueError("comparison requires a joint checkpoint and a fire-only baseline")
    # 配置差异必须显式解决，不能悄悄替换检查点训练时的物理或视觉参数。
    for key in ("simulator_config", "vision_config"):
        if metadata["joint"]["fingerprints"][key] != metadata["baseline"]["fingerprints"][key]:
            raise ValueError(f"comparison {key} differs between checkpoints")
    for key in ("decision_clock", "scenario"):
        if settings["joint"].get(key) != settings["baseline"].get(key):
            raise ValueError(f"comparison environment.{key} differs between checkpoints")
    if any(v["episode_steps"] != 500 for v in settings.values()):
        raise ValueError("this benchmark requires 500-step (5 s) checkpoints")
    parent = Path(output_dir or ROOT / "artifacts/comparison").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="compare-", dir=parent))
    report = {"version": 1, "checkpoints": {k: str(v) for k, v in checkpoints.items()},
              "sha256": {key: hashlib.sha256(data).hexdigest() for key, data in checkpoint_data.items()},
              "cases": [], "aggregate": {}, "paired_aggregate": {}}
    for key, data in checkpoint_data.items():
        checkpoints[key] = output / f"{key}.zip"
        checkpoints[key].write_bytes(data)
    # 在抽取比较场景前，按原始契约校验两个模型；场景覆盖仅用于独立评估。
    for key, config in configs.items():
        env = make_environment(config, output / key)
        try:
            load_model(checkpoints[key], env, device=device)
        finally:
            env.close()
    env = make_environment(configs["joint"], output / "spawn")
    base = env.unwrapped
    try:
        with training_worker(base.simulator_root, base.simulator_binary, output / "spawn.log",
                             base.simulator_config) as client:
            for distance in (3, 5, 7):
                for speed in (-7, -3, -1, 1, 3, 7):
                    index = len(report["cases"])
                    scene = copy.deepcopy(settings["joint"]["scenario"])
                    for field in ("controlled_position_xz_m", "controlled_yaw_rad", "target_yaw_rad",
                                  "target_bearing_rad", "gimbal_yaw_rad", "gimbal_pitch_rad"):
                        scene.pop(field, None)
                    scene.update(target_distance_m=distance,
                                 motion={"kind": "rotation", "angular_speed_rad_s": speed})
                    case = {"scene_id": index, "sampling_seed": 10000 + index,
                            "distance_m": distance, "angular_speed_rad_s": speed,
                            "scenario": scene, "attempts": [], "results": {}}
                    report["cases"].append(case)
                    try:
                        initial = reset_spawn(client.reset, np.random.default_rng(case["sampling_seed"]),
                                              scene, case["attempts"])
                        case["spawn_seed"] = case["attempts"][-1]["seed"]
                        case["resolved_scenario"] = initial["data"]["evaluation"]["scenario"]
                    except Exception as error:
                        case["spawn_error"] = f"{type(error).__name__}: {error}"
                        _atomic_json(output / "comparison.json", report)
                        continue
                    override = {key: case[key] for key in ("scene_id", "scenario", "spawn_seed")}
                    for key, path in checkpoints.items():
                        print(f"Scene {index + 1}/18: {distance} m, {speed:+} rad/s, {key}", flush=True)
                        try:
                            replay_path = evaluate_checkpoint(path, output_dir=output / f"scene-{index:02d}" / key,
                                                              device=device, scene_override=override)
                            replay = json.loads(replay_path.read_text())
                            if replay["scenario"] != case["resolved_scenario"]:
                                raise RuntimeError("resolved scene differs from the shared spawn")
                            case["results"][key] = {**replay["summary"], "replay": str(replay_path)}
                        except Exception as error:
                            case["results"][key] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                        _atomic_json(output / "comparison.json", report)
    finally:
        env.close()
    paired = [case for case in report["cases"] if all(
        case["results"].get(key, {}).get("status") == "complete" for key in checkpoints)]
    for key in checkpoints:
        for name, cases in (("aggregate", report["cases"]), ("paired_aggregate", paired)):
            results = [case["results"][key] for case in cases
                       if case["results"].get(key, {}).get("status") == "complete"]
            shots = sum(r["score"]["eligible_shots"] for r in results)
            hits = sum(r["score"]["damaging_projectiles"] for r in results)
            report[name][key] = dict(completed=len(results), total_cases=18,
                                    damage=sum(r["score"]["official_damage"] for r in results),
                                    shots=shots, hits=hits, hit_rate=hits / shots if shots else None,
                                    slot_switches=sum(r["slot_switches"] for r in results))
    rows = []
    for case in report["cases"]:
        for key in checkpoints:
            result = case["results"].get(key, {})
            score = result.get("score", {})
            rows.append(dict(scene_id=case["scene_id"], distance_m=case["distance_m"],
                             speed_rad_s=case["angular_speed_rad_s"], policy=key,
                             status=result.get("status", "spawn_failed"), damage=score.get("official_damage"),
                             shots=score.get("eligible_shots"), hit_rate=result.get("hit_rate"),
                             slot_switches=result.get("slot_switches"), replay=result.get("replay", ""),
                             error=result.get("error", case.get("spawn_error", ""))))
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report["status"] = "complete" if len(paired) == 18 else "incomplete"
    _atomic_json(output / "comparison.json", report)
    header = "".join(f"<th>{escape(key)}</th>" for key in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row.values()) + "</tr>" for row in rows)
    (output / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>旋转靶成对评估</title>'
        '<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>'
        '<h1>旋转靶成对评估</h1><p>5 秒射击窗口及尾部结算；失败场景单独列出。'
        '汇总仅统计双方均完成的场景；短训练结果只用于链路验收。</p><pre>'
        + escape(json.dumps(report["paired_aggregate"], indent=2)) + f'</pre><table><tr>{header}</tr>{body}</table>')
    print(f"Comparison {report['status']}: {output / 'comparison.html'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--baseline-checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device')
    args = parser.parse_args()
    output = compare_checkpoints(args.checkpoint, args.baseline_checkpoint,
                                 output_dir=args.output_dir, device=args.device)
    return 0 if json.loads((output / 'comparison.json').read_text())['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
