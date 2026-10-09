"""固定总 rollout 或每环境 n_steps，顺序比较独立环境数量的完整 PPO 训练吞吐。"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
from time import perf_counter

from src.training.config import DEFAULT_CONFIG, read_config, validate_config
from src.training.paths import discover_monitor_files
from tools.config import load_tool_config, write_effective_config


def benchmark(config, output, counts, repeats, timesteps, *, rollout_mode="total", updates=None):
    if rollout_mode not in ("total", "per-env"):
        raise ValueError("rollout_mode must be total or per-env")
    if updates is not None and (type(updates) is not int or updates < 1):
        raise ValueError("updates must be a positive integer")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # total 固定总样本数；per-env 保留输入配置的每环境采样长度。
    rollout = config["ppo"]["n_steps"] * config["n_envs"]
    configs = {}
    for count in counts:
        if count < 1 or (rollout_mode == "total" and rollout % count):
            raise ValueError("environment counts must be positive; total mode also requires divisors of the aggregate rollout")
        trial = copy.deepcopy(config)
        trial["n_envs"] = count
        if rollout_mode == "total":
            trial["ppo"]["n_steps"] = rollout // count
        trial["total_timesteps"] = (updates * count * trial["ppo"]["n_steps"]
                                    if updates is not None else timesteps)
        configs[count] = validate_config(trial)
    results = dict(created_at=datetime.now(timezone.utc).isoformat(),
                   cpu_count=os.cpu_count(), initial_load=os.getloadavg(),
                   rollout_mode=rollout_mode, requested_updates=updates,
                   aggregate_rollout=rollout if rollout_mode == "total" else None,
                   steps_per_environment=config["ppo"]["n_steps"] if rollout_mode == "per-env" else None,
                   trials=[])
    for repeat in range(repeats):
        # 交替执行顺序，减小机器温度、频率和后台负载随时间变化的影响。
        for count in counts if repeat % 2 == 0 else list(reversed(counts)):
            name = f"repeat-{repeat + 1}-envs-{count}"
            config_path = output / f"{name}.json"
            config_path.write_text(json.dumps(configs[count], indent=2))
            directory = output / name
            command = [sys.executable, "-m", "src.training.train", "--config", str(config_path),
                       "--output-dir", str(directory)]
            print(f"Starting {name}", flush=True)
            started = perf_counter()
            with (output / f"{name}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            elapsed = perf_counter() - started
            if code:
                raise RuntimeError(f"{name} failed ({code}); see {output / (name + '.log')}")
            manifest = json.loads((directory / "run.json").read_text())
            if manifest["status"] != "complete":
                raise RuntimeError(f"{name} did not complete")
            with (directory / "logs/progress.csv").open() as file:
                update_rows = list(csv.DictReader(file))
            sampling = sum(float(row["time/sampling_seconds"]) for row in update_rows)
            optimization = sum(float(row["time/optimization_seconds"]) for row in update_rows)
            samples = manifest["last_completed_timesteps"] - manifest["start_timesteps"]
            monitors = discover_monitor_files(directory)
            episodes = []
            for path in monitors:
                with path.open() as file:
                    episodes.extend(csv.DictReader(line for line in file if not line.startswith("#")))
            result = dict(name=name, n_envs=count, samples=samples, updates=len(update_rows),
                          steps_per_environment=configs[count]["ppo"]["n_steps"],
                          wall_seconds=elapsed, training_seconds=manifest["training_seconds"],
                          sampling_seconds=sampling, optimization_seconds=optimization,
                          training_fps=samples / manifest["training_seconds"],
                          sampling_fps=samples / sampling, wall_fps=samples / elapsed,
                          complete_episodes=len(episodes), monitor_files=len(monitors),
                          damage=sum(float(row["episode_damage"]) for row in episodes),
                          actual_shots=sum(int(row["actual_shots"]) for row in episodes))
            results["trials"].append(result)
            (output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
    mode_note = (f"固定每次更新总样本数 {rollout}。" if rollout_mode == "total" else
                 f"固定每环境 n_steps={config['ppo']['n_steps']}，总 rollout 随环境数增加。")
    lines = ["# 独立环境并行训练实测", "", mode_note,
             "包含实际 PPO 更新、自动重置和检查点保存。各组总样本量见 results.json；速率均按实际采样量计算。", "",
             "采样时间含策略推理、IPC、仿真及自动 reset；训练时间另含首次 reset、PPO 更新与日志。",
             "端到端时间还含 Python/worker 启动、最终保存及关闭。所有 fps 为聚合样本/秒。", "",
             "| 环境数 | 训练 fps（均值） | 训练加速比 | 采样 fps（均值） | 端到端 fps（均值） |",
             "|---:|---:|---:|---:|---:|"]
    baseline = statistics.mean(item["training_fps"] for item in results["trials"] if item["n_envs"] == counts[0])
    for count in counts:
        group = [item for item in results["trials"] if item["n_envs"] == count]
        average = lambda key: statistics.mean(item[key] for item in group)
        lines.append(f"| {count} | {average('training_fps'):.2f} | {average('training_fps') / baseline:.2f}× | "
                     f"{average('sampling_fps'):.2f} | {average('wall_fps'):.2f} |")
    lines += ["", "详见 results.json 及每次运行的 run.json、logs/progress.csv、Monitor 和原生日志。",
              "吞吐实验不证明最终收敛速度；应另行比较相同样本预算下的模型质量。", ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"Report: {output / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="tool configuration JSON")
    parser.add_argument("--training-config", type=Path, help="training configuration JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-envs", type=int, nargs="+")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--rollout-mode", choices=("total", "per-env"),
                        help="keep aggregate rollout size or preserve n_steps per environment")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--timesteps", type=int)
    budget.add_argument("--updates", type=int, help="complete rollout/update cycles per trial; total samples may differ")
    args = parser.parse_args()
    overrides = {key: getattr(args, key) for key in
                 ("output_dir", "n_envs", "repeats", "rollout_mode", "timesteps", "updates")
                 if getattr(args, key) is not None}
    if args.training_config is not None:
        overrides["config"] = str(args.training_config)
    config = load_tool_config("benchmark", args.config, overrides)
    if config["repeats"] < 1 or len(set(config["n_envs"])) != len(config["n_envs"]):
        parser.error("repeats must be positive and environment counts must be unique")
    def stop(signum, frame):
        raise KeyboardInterrupt(f"benchmark signal {signum}")

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        benchmark(read_config(config["config"]), config["output_dir"], config["n_envs"],
                  config["repeats"], config["timesteps"], rollout_mode=config["rollout_mode"],
                  updates=config["updates"])
        write_effective_config(Path(config["output_dir"]) / "tool_config.json", config)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
