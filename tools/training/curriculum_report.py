"""Report v2/v3 curriculum progress with separate budget, combat and thermal gates."""
import argparse
import csv
import json
from pathlib import Path


def generate_report(root):
    """Read a role run or role group and write its derived report beside the inputs."""
    root = Path(root).resolve()
    if not (root/"run.json").is_file() and not any((root/role/"run.json").is_file() for role in ("infantry", "sentry")):
        raise ValueError("no curriculum run.json found")
    records = {}
    run_dirs = {root.name: root} if (root/"run.json").is_file() else {role: root/role for role in ("infantry", "sentry")}
    for role, directory in run_dirs.items():
        path = directory/"run.json"
        if not path.exists():continue
        run = json.loads(path.read_text())
        records[role] = dict(curriculum_version=run.get("config", {}).get("curriculum", {}).get("version"),status=run["status"],steps=run.get("last_completed_timesteps",0),
            budget=run["requested_timesteps"],budget_complete=run.get("training_budget_complete",False),
            combat_passed=run.get("curriculum_passed",False),thermal_passed=run.get("thermal_passed",False),
            stage=run.get("curriculum",{}).get("stage"),updates=run.get("completed_updates",0),
            final_validation=run.get("final_validation_reports"),final_test=run.get("final_course_reports"))
        metrics = directory/"logs/progress.csv"
        if metrics.exists():
            with metrics.open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            if rows:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                x = [float(row["time/total_timesteps"]) for row in rows]
                fig,axes = plt.subplots(3,1,figsize=(10,9),sharex=True)
                for axis,keys in zip(axes,(("rollout/damage","rollout/actual_shots"),
                    ("rollout/return",),("rollout/ordinary_locked_episode_fraction","rollout/round_locked_episode_fraction"))):
                    for key in keys:
                        if key in rows[0]:
                            axis.plot(x,[float(row[key]) for row in rows],label=key)
                    if axis.lines:axis.legend()
                    axis.grid(alpha=.2)
                axes[-1].set_xlabel("Cumulative training steps")
                fig.suptitle(f"{role} / curriculum v{records[role]['curriculum_version']}")
                fig.tight_layout()
                fig.savefig(root/(role+"-learning.png"),dpi=150)
                plt.close(fig)
    comparison = []
    if (root/"comparison/results.json").exists():
        from src.training.curriculum_v2 import STAGES
        rows = json.loads((root/"comparison/results.json").read_text())
        for role in ("infantry","sentry"):
            for c,stage in enumerate(STAGES):
                for controller in ("traditional","gru"):
                    group = [r for r in rows if r["role"] == role and r["course"] == c and r["controller"] == controller]
                    shots = sum(r.get("actual_shots",0) for r in group)
                    hits = sum(r.get("damaging_hits",0) for r in group)
                    times = [r["kill_time_s"] for r in group if r.get("kill_time_s") is not None]
                    comparison.append(dict(role=role,stage=stage,controller=controller,cases=len(group),
                        kill_rate=sum(r["end_reason"] == "target_destroyed" for r in group)/len(group) if group else 0.,
                        hit_efficiency=hits/shots if shots else 0.,actual_shots=shots,damaging_hits=hits,
                        mean_kill_time_s=sum(times)/len(times) if times else None,
                        locked_episode_fraction=sum(r.get("ordinary_heat_locks",0)>0 for r in group)/len(group) if group else 0.,
                        environment_errors=sum("error" in r for r in group)))
        (root/"comparison-summary.json").write_text(json.dumps(comparison,indent=2))
    (root/"report.json").write_text(json.dumps(records,indent=2))
    lines = ["# 课程训练实际结果", "", "预算、战斗门槛和热锁门槛分别统计；未完成项不记为通过。", "",
             "| 模型 | 状态 | 累计训练步数 | 更新数 | 预算完成 | 战斗通过 | 热锁通过 |", "|---|---|---:|---:|---|---|---|"]
    for role,r in records.items():
        lines.append(f"| {role} | {r['status']} | {r['steps']:,} / {r['budget']:,} | {r['updates']} | {r['budget_complete']} | {r['combat_passed']} | {r['thermal_passed']} |")
    if comparison:
        lines += ["", "## 固定最终场景的完整方案对照", "",
            "双方使用相同初始物理状态和射击机会序列，保留各自搜索和选板行为。不同控制器的选板奖励上下文不同，净回报不用于直接排名。", "",
            "| 模型 | 阶段 | 方案 | 击毁率 | 命中效率 | 普通热锁回合比例 | 环境错误 |",
            "|---|---|---|---:|---:|---:|---:|"]
        for row in comparison:
            lines.append(f"| {row['role']} | {row['stage']} | {row['controller']} | {row['kill_rate']:.1%} | {row['hit_efficiency']:.1%} | {row['locked_episode_fraction']:.1%} | {row['environment_errors']} |")
    lines += ["", "逐阶段原始指标见各角色 `curriculum/test/` 和 `report.json`；失败场景保留种子与冻结配置。", ""]
    (root/"RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate_report(args.run), indent=2))


if __name__ == "__main__":
    main()
