"""NUC local simulator + CPU Actor loop. No Critic is loaded or evaluated."""
import argparse
import json
import signal
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from src.policy.recurrent_deployment import ActorRunner


def run(actor_dir, output, *, episodes=2, seed=1000, threads=1, warmup_steps=100):
    # Only the simulation harness depends on Gym/SB3 wrappers. ActorRunner itself
    # works with torch/numpy and an Actor packet, including on a real-device host.
    from src.training.environment import make_environment
    runner = ActorRunner(actor_dir, threads=threads)
    config = runner.manifest["simulation_config"]
    config["environment"]["scene_seed"] = seed
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = make_environment(config, output / "environment")
    results, latency, steps = [], [], 0
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                started = perf_counter_ns()
                action, _, _ = runner.predict(obs)
                elapsed = (perf_counter_ns()-started)*1e-6
                if steps >= warmup_steps:
                    latency.append(elapsed)
                steps += 1
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            results.append({k: info[k] for k in ("episode_damage", "actual_shots", "slot_switches",
                                                "episode_training_reward", "episode_steps", "end_reason")})
    finally:
        env.close()
    report = dict(actor_sha256=runner.manifest["sha256"], episodes=results, total_steps=steps,
                  warmup_steps=warmup_steps, cpu_threads=threads, critic_loaded=False,
                  latency_definition="ActorRunner.predict: input conversion, TorchScript forward, state/output handling")
    if latency:
        report.update(inference_ms=dict(p50=float(np.percentile(latency, 50)),
            p95=float(np.percentile(latency, 95)), p99=float(np.percentile(latency, 99)), maximum=max(latency)),
            latency_samples=len(latency), passes_100hz_target=bool(np.percentile(latency, 99) <= 10))
    else:
        report.update(latency_samples=0, passes_100hz_target=False)
    (output / "deployment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.episodes < 1 or args.threads < 1:
        parser.error("episodes and threads must be positive")
    report = run(args.actor, args.output, episodes=args.episodes, seed=args.seed, threads=args.threads)
    if not report["passes_100hz_target"]:
        raise SystemExit("deployment latency target was not met; see deployment.json")


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
