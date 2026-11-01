"""Physical short/long detector loss regressions, outside curriculum promotion."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.training.config import ROOT,read_config
from src.training.environment import make_environment
from src.policy.recurrent_actor import RecurrentActor
from src.policy.recurrent_observations import ACTOR_KEYS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rows = []
    for role in ("infantry","sentry"):
        config = read_config(ROOT/f"config/training/rmul_2027_{role}_curriculum_v2.json")
        for profile in ("standard","wheel_leg","low","folding"):
            for kind,duration in (("short",120),("long",1000)):
                scenario = copy.deepcopy(config["environment"]["scenario"])
                scenario["duel"]["controlled"]["motion"] = dict(kind="static")
                scenario["duel"]["target"] = dict(profile=profile,preset="infantry_health_cooling",motion=dict(kind="static"))
                scenario["policy_start"] = dict(blackouts_ms=[[350,350+duration]])
                diagnostic_config = copy.deepcopy(config)
                diagnostic_config.pop("curriculum")
                diagnostic_config["environment"].update(scenario=scenario,scene_seed=92000)
                diagnostic_config["environment"]["decision_clock"]["seed"] = (92000 ^ 0x53484F54) % (2**32)
                env = make_environment(diagnostic_config,args.output/f"{role}-{profile}-{kind}")
                env.unwrapped._spawn_attempts = 1
                actor = RecurrentActor().eval()
                state = actor.initial_state(1)
                calls = dict(gimbal=0,resource=0)
                def count_gimbal(module, inputs, output):calls["gimbal"] += 1
                def count_resource(module, inputs, output):calls["resource"] += 1
                actor.gimbal_gru.register_forward_hook(count_gimbal)
                actor.resource_gru.register_forward_hook(count_resource)
                row = dict(role=role,profile=profile,kind=kind,duration_ms=duration)
                try:
                    obs,_ = env.reset()
                    resets = missing = 0
                    own_updates = resource_updates = 0
                    with torch.inference_mode():
                        for step in range(240):
                            tensors = {k:torch.as_tensor(np.asarray(obs[k])[None]) for k in ACTOR_KEYS}
                            previous = state
                            previous_calls = dict(calls)
                            _,_,actions,state = actor(tensors,state)
                            from src.training.curriculum_metrics import TrackingMetrics
                            if not TrackingMetrics.tracking(obs):
                                missing += 1
                                if any(obs["action_mask"][2::2]):raise AssertionError("missing-target fire mask leaked")
                                own_updates += calls["gimbal"]-previous_calls["gimbal"]
                                resource_updates += calls["resource"]-previous_calls["resource"]
                            if step > 0:resets += int(obs["target_reset"][0])
                            # This diagnostic deliberately tracks without shooting so the loss event finishes.
                            action = int(actions.item())
                            if action in (2,4,6,8):action -= 1
                            obs,_,term,trunc,info = env.step(action)
                            if term or trunc:break
                    row.update(target_resets=resets,missing_steps=missing,own_memory_updates=own_updates,
                               resource_memory_updates=resource_updates,actual_shots=info["actual_shots"])
                    if info["actual_shots"] != 0:raise AssertionError("diagnostic unexpectedly fired")
                    if missing == 0:raise AssertionError("blackout did not expose a non-tracking observation")
                    if kind == "short" and resets != 0:raise AssertionError("short occlusion reset target memory")
                    if kind == "long" and resets == 0:raise AssertionError("long loss did not reset target memory")
                    if missing and (own_updates == 0 or resource_updates == 0):raise AssertionError("own/resource memory stopped during loss")
                    row["passed"] = True
                except Exception as error:
                    row.update(passed=False,error=f"{type(error).__name__}: {error}")
                finally:
                    env.close()
                rows.append(row)
                print(json.dumps(row),flush=True)
    report = dict(passed=all(r["passed"] for r in rows),cases=rows,
                  purpose="adapter/memory/mask semantics only; no occlusion training benefit claim")
    (args.output/"results.json").write_text(json.dumps(report,indent=2))
    if not report["passed"]:raise SystemExit("loss regression failed; inspect results.json")


if __name__ == "__main__":main()
