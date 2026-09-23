"""按独立窗口伤害选择已保存策略，并生成报告和回放。"""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import webbrowser

from tools.training.view import _atomic_json, evaluate_checkpoint, open_replays


def _discover(run_dir, output, run):
    from src.training.models import read_checkpoint_metadata

    candidates, by_step = [], {}
    paths = sorted(run_dir.glob("checkpoint_*.zip")) + [run_dir / "final.zip"]
    snapshots = output / "models"
    snapshots.mkdir()
    for path in paths:
        item = {"name": path.name, "source": str(path), "status": "pending",
                "num_timesteps": None, "sha256": None, "replay": None,
                "official_damage": None, "hit_rate": None, "shots": None}
        candidates.append(item)
        try:
            data = path.read_bytes()
            item["sha256"] = hashlib.sha256(data).hexdigest()
            metadata = read_checkpoint_metadata(BytesIO(data))
            item["num_timesteps"] = metadata["num_timesteps"]
            # 不同环境配置的检查点不能直接参与同一次排名。
            for key in ("fingerprints", "observation"):
                if metadata[key] != run["metadata"][key]:
                    raise ValueError(f"checkpoint {key} differs from this training run")
            item["model"] = {key: metadata["config"][key] for key in ("algorithm", "policy_kind")}
            snapshot = snapshots / path.name
            snapshot.write_bytes(data)
            item["snapshot"] = str(snapshot)
            step = item["num_timesteps"]
            previous = by_step.get(step)
            if previous is not None:
                # 最后处理 final.zip，使其优先于同一步数的周期检查点。
                if path.name == "final.zip":
                    previous.update(status="duplicate", duplicate_of=path.name)
                else:
                    item.update(status="duplicate", duplicate_of=previous["name"])
                    continue
            by_step[step] = item
        except Exception as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}")
    return sorted(candidates, key=lambda item: (item["num_timesteps"] is None,
                                              item["num_timesteps"] or 0, item["name"]))


def _rank(candidates):
    return sorted((item for item in candidates if item["status"] == "complete"),
                  key=lambda item: (item["official_damage"], item["num_timesteps"]), reverse=True)


def analyze_run(run_dir, *, device=None):
    """分析已完成的训练，返回独立分析目录，不打开窗口。

    对每个待评估 ZIP 创建固定副本，使回放、摘要和最佳权重始终对应同一份字节，
    不受源文件后续替换影响。原运行目录的 run.json 保持只读。

    Args:
        run_dir: run.json 状态为 complete 的训练目录。
        device: 可选的推理设备覆盖。

    Returns:
        本次分析的唯一输出目录。
    """
    from src.training.models import publish_latest

    run_dir = Path(run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    if run.get("status") != "complete":
        raise ValueError("analysis requires a completed training run")
    parent = run_dir / "analysis"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="analysis-", dir=parent))
    state = {"version": 1, "run_dir": str(run_dir), "created_at": datetime.now(timezone.utc).isoformat(),
             "status": "running", "metric": "official_damage", "tie_break": "larger num_timesteps",
             "candidates": [], "best": None, "final": None, "report_status": "pending",
             "report": str(output / "report.html"), "display_errors": []}
    state_path = output / "analysis.json"
    _atomic_json(state_path, state)
    print(f"Analysis: {output}", flush=True)
    try:
        # 同时保留图表源数据、模型副本和评估结果，便于复查。
        (output / "logs").mkdir()
        for name in ("run.json", "episodes.monitor.csv", "logs/progress.csv"):
            source = run_dir / name
            if source.is_file():
                shutil.copyfile(source, output / name)
        state["candidates"] = _discover(run_dir, output, run)
        _atomic_json(state_path, state)
        pending = [item for item in state["candidates"] if item["status"] == "pending"]
        for index, item in enumerate(pending, 1):
            item["status"] = "evaluating"
            _atomic_json(state_path, state)
            print(f"Candidate {index}/{len(pending)}: {item['name']}", flush=True)
            try:
                path = evaluate_checkpoint(item["snapshot"], device=device, output_dir=output / "evaluation")
                replay = json.loads(path.read_text())
                item["replay"] = str(path)
                summary = replay["summary"]
                score = summary["score"]
                if replay["model"]["sha256"] != item["sha256"]:
                    raise ValueError("evaluation checkpoint digest differs from candidate snapshot")
                damage = score["official_damage"]
                item.update(evaluation_status=summary["status"], hit_rate=summary["hit_rate"],
                            shots=score["eligible_shots"], raw_damage=summary["raw_damage"])
                if (summary["status"] != "complete" or damage is None
                        or not math.isfinite(damage)):
                    item.update(status="incomplete", error="evaluation did not produce complete window damage")
                else:
                    item.update(status="complete", official_damage=damage)
            except Exception as error:
                item.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(f"Candidate failed: {item['name']}: {item['error']}", flush=True)
            _atomic_json(state_path, state)
        ranked = _rank(state["candidates"])
        state["final"] = next((item for item in state["candidates"] if item["name"] == "final.zip"), None)
        if ranked:
            for rank, item in enumerate(ranked, 1):
                item["rank"] = rank
            best = ranked[0]
            # 后续分析替换训练目录的 best.zip 时，本批次的模型副本保持不变。
            publish_latest(best["snapshot"], output / "best.zip")
            publish_latest(output / "best.zip", run_dir / "best.zip")
            state["best"] = best
            state["best_checkpoint"] = str(output / "best.zip")
            state["status"] = "partial" if any(
                item["status"] in ("failed", "incomplete") for item in state["candidates"]) else "complete"
            print(f"Best: {best['name']} at {best['num_timesteps']} steps; "
                  f"official damage={best['official_damage']}; saved {run_dir / 'best.zip'}", flush=True)
        else:
            state.update(status="failed", error="No candidate produced a complete evaluation; no new best.zip published")
            print(state["error"], flush=True)
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        for item in state["candidates"]:
            if item["status"] == "evaluating":
                item.update(status="interrupted", error=state["error"])
        raise
    finally:
        _atomic_json(state_path, state)
        try:
            from tools.training.report import write_report
            write_report(output, state, run)
            state["report_status"] = "complete"
            print(f"Report: {state['report']}", flush=True)
        except Exception as error:
            state.update(report_status="failed", report_error=f"{type(error).__name__}: {error}")
            print(f"Report failed; evaluation retained: {state['report_error']}", flush=True)
        _atomic_json(state_path, state)
    return output


def show_analysis(output, *, viewer_binary=None):
    """打开离线报告及 Final/Best 回放窗口，不修改评估状态。"""
    path = Path(output) / "analysis.json"
    state = json.loads(path.read_text())
    report = Path(state["report"])
    try:
        if state["report_status"] == "complete":
            print(f"Open report: {report.as_uri()}", flush=True)
            try:
                if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                    print("No graphical session; report retained for desktop viewing.", flush=True)
                elif not webbrowser.open(report.as_uri()):
                    state["display_errors"].append("No browser opened; open report.html manually")
            except Exception as error:
                state["display_errors"].append(f"Browser: {type(error).__name__}: {error}")
        final, best = state["final"], state["best"]
        same = final and best and final["sha256"] == best["sha256"]
        replays = [(item["replay"], role + (" (Final = Best)" if same else ""))
                   for role, item in (("Final", final), ("Best", best)) if item and item.get("replay")]
        try:
            if replays and not open_replays(replays, viewer_binary=viewer_binary):
                state["display_errors"].append("Replay windows unavailable; use the printed replay commands")
        except Exception as error:
            state["display_errors"].append(f"Replay: {type(error).__name__}: {error}")
    except KeyboardInterrupt:
        state["display_errors"].append("Viewing interrupted")
        raise
    finally:
        _atomic_json(path, state)
        for error in state["display_errors"]:
            print(f"Viewing: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--viewer-binary", type=Path)
    parser.add_argument("--no-view", action="store_true", help="save analysis without opening windows")
    args = parser.parse_args()

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        output = analyze_run(args.run_dir, device=args.device)
        if not args.no_view:
            show_analysis(output, viewer_binary=args.viewer_binary)
        state = json.loads((output / "analysis.json").read_text())
        return 1 if state["status"] == "failed" or state["report_status"] == "failed" else 0
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
