"""Export a versioned CPU Actor from a GRU checkpoint, optionally verifying a real episode."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.policy.recurrent_deployment import export_actor, verify_export
from src.policy.recurrent_observations import ACTOR_KEYS
from src.training.recurrent_ppo import RecurrentPPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new Actor directory")
    parser.add_argument("--trajectory", type=Path, help="saved episode-*.npz from recurrent rollouts")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = RecurrentPPO.load(args.checkpoint, device="cpu")
    export_actor(model.policy.actor, args.output, config=model.config, training_steps=model.num_timesteps)
    if args.trajectory:
        with np.load(args.trajectory, allow_pickle=False) as data:
            observations = {k: data["obs_"+k] for k in ACTOR_KEYS}
        print(json.dumps(verify_export(model.policy.actor, args.output, observations), indent=2))
    print(f"Actor: {args.output}")


if __name__ == "__main__":
    main()
