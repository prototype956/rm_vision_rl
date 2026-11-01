"""Fixed independent scenario seeds for a matched-budget GRU/MLP comparison."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import hashlib
import json
import multiprocessing as mp
import signal
from pathlib import Path

import numpy as np
import torch

from src.policy.recurrent_deployment import ActorRunner
from src.training.environment import make_environment
from src.training.models import read_checkpoint_metadata, MeasuredMaskablePPO


def _evaluate(job):
    kind, seed, config, path, output = job
    torch.set_num_threads(1)
    model = ActorRunner(path) if kind == "gru" else MeasuredMaskablePPO.load(path, device="cpu")
    config["environment"]["scene_seed"] = seed
    config["environment"]["decision_clock"]["seed"] = seed
    env = make_environment(config, output / "environment" / f"{kind}-{seed}")
    try:
        obs, _ = env.reset()
        done = False
        while not done:
            if kind == "gru":
                action = model.predict(obs)[0]
            else:
                action, _ = model.predict(obs, deterministic=True, action_masks=obs["action_mask"].astype(bool))
                action = int(action)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        robots = env.unwrapped.debug_snapshot()["data"]["evaluation"]["robots"]
        own = next(r for r in robots if r["robot_id"] == 1)
        hits = own["resources"]["damaging_hits"]
        shots, damage = info["actual_shots"], info["episode_damage"]
        return dict(model=kind, seed=seed, damage=damage, actual_shots=shots, damaging_hits=hits,
                    hit_efficiency=hits/shots if shots else 0., damage_per_shot=damage/shots if shots else 0.,
                    slot_switches=info["slot_switches"], episode_return=info["episode_training_reward"],
                    steps=info["episode_steps"], end_reason=info["end_reason"])
    finally:
        env.close()


def compare(gru, mlp, output, seeds=range(1000, 1020), workers=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    actor = ActorRunner(gru)
    metadata = read_checkpoint_metadata(mlp)
    if metadata["config"]["policy_kind"] != "asymmetric_mlp":
        raise ValueError("comparison requires the retained asymmetric MLP baseline")
    torch.set_num_threads(1)
    configs = {"gru": actor.manifest["simulation_config"], "mlp": copy.deepcopy(metadata["config"])}
    for config in configs.values():
        for key in ("simulator_root", "vision_root", "simulator_binary", "bridge_binary", "simulator_config"):
            config["environment"].pop(key, None)
    # Only observation/network modes may differ; rewards and physical environments must match.
    a, b = [copy.deepcopy(c["environment"]) for c in configs.values()]
    a.pop("observation_mode"); b.pop("observation_mode")
    if a != b or any(configs["gru"].get(k) != configs["mlp"].get(k)
                     for k in ("reward_scale", "selection_reward", "switch_penalty")):
        raise ValueError("GRU/MLP physical environment or reward mismatch")
    rows = []
    jobs = [(kind, seed, copy.deepcopy(configs[kind]), gru if kind == "gru" else mlp, output)
            for seed in seeds for kind in ("gru", "mlp")]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_evaluate, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: (r["seed"], r["model"]))
            print(json.dumps(row), flush=True)
            with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                writer.writeheader(); writer.writerows(rows)
    summary = dict(seed_range=list(seeds), gru_steps=actor.manifest["training_steps"],
                   mlp_steps=metadata["num_timesteps"],
                   gru_sha256=actor.manifest["sha256"],
                   mlp_checkpoint_sha256=hashlib.sha256(Path(mlp).read_bytes()).hexdigest(),
                   deterministic=True, control_step_ms=10,
                   efficiency="damaging_hits / actual emitted projectiles; damage_per_shot reported separately",
                   models={kind: {metric: float(np.mean([r[metric] for r in rows if r["model"] == kind]))
                                  for metric in ("damage", "actual_shots", "damaging_hits", "hit_efficiency", "damage_per_shot", "slot_switches", "episode_return")}
                           for kind in ("gru", "mlp")})
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru", required=True, type=Path, help="Actor directory")
    parser.add_argument("--mlp", required=True, type=Path, help="MLP checkpoint ZIP")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    compare(args.gru, args.mlp, args.output, workers=args.workers)


if __name__ == "__main__":
    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
