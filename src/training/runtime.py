"""Bounded diagnostics and versioned long-run storage, scoped to one run."""
import gzip
import json
import math
from pathlib import Path
import shutil

import numpy as np

DEFAULT = dict(streaming=True, bounded_logging=True, replay_batch_episodes=64,
               diagnostic_episodes=8, diagnostic_limit_gib=50, minimum_free_gib=50,
               keep_checkpoints=3, milestone_steps=10_000_000, simulator_threads=1)


def validate(value):
    if not isinstance(value,dict) or set(value)-DEFAULT.keys():
        raise ValueError("unsupported runtime options")
    result = {**DEFAULT,**value}
    for key in ("streaming","bounded_logging"):
        if type(result[key]) is not bool:
            raise ValueError(key+" must be boolean")
    for key in set(DEFAULT)-{"streaming","bounded_logging"}:
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(key+" must be a positive integer")
    return result


def low_disk(output, options):
    return shutil.disk_usage(output).free < options["minimum_free_gib"] * 2**30


def episode_summaries(episodes, output, update):
    with gzip.open(Path(output)/"episodes.jsonl.gz","at",encoding="utf-8") as stream:
        for ep in episodes:
            stream.write(json.dumps(dict(update=update,worker=ep["worker"],bootstrap=ep["bootstrap"],
                                         info=ep["info"]),allow_nan=False)+"\n")


def diagnostics(episodes, output, update, options, seed):
    from src.training.recurrent_ppo import save_rollout
    root = Path(output)/"rollouts"
    rng = np.random.default_rng(np.random.SeedSequence([seed,update,0xD1A6]))
    chosen = rng.choice(len(episodes),min(len(episodes),options["diagnostic_episodes"]),replace=False)
    save_rollout([episodes[int(i)] for i in chosen],root/f"update-{update:08d}")
    directories = sorted(root.glob("update-*"))
    sizes = {p:sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) for p in directories}
    native_bytes = sum(p.stat().st_size for subtree in (Path(output)/"environment",Path(output)/"curriculum")
                       for p in subtree.rglob("*") if p.is_file() and (p.suffix == ".log" or ".log." in p.name))
    total = sum(sizes.values()) + native_bytes
    for p in directories:
        if total <= options["diagnostic_limit_gib"]*2**30:
            break
        # Resolve and verify before recursively deleting only this run's diagnostic directory.
        if p.parent.resolve() != root.resolve() or not p.name.startswith("update-"):
            raise RuntimeError("diagnostic cleanup escaped run directory")
        shutil.rmtree(p)
        total -= sizes[p]
    if total > options["diagnostic_limit_gib"]*2**30:
        raise RuntimeError("native diagnostic logs exceed this run storage limit; checkpoint remains resumable")


def checkpoint(model, metadata, output, options, previous_steps, promoted=False):
    from src.policy.recurrent_deployment import export_actor
    from src.training.models import publish_latest
    output = Path(output)
    step = model.num_timesteps
    permanent = promoted or step//options["milestone_steps"] > previous_steps//options["milestone_steps"]
    path = output/(f"milestone_{step}.zip" if permanent else f"checkpoint_{step}.zip")
    model.save(path,metadata)
    publish_latest(path,output/"latest.zip")
    if permanent:
        export_actor(model.policy.actor,output/f"actor-{step}",config=model.config,training_steps=step)
    rolling = sorted(output.glob("checkpoint_*.zip"),key=lambda p:int(p.stem.split("_")[-1]))
    for old in rolling[:-options["keep_checkpoints"]]:
        old.unlink()
    return path
