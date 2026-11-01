"""Shared deterministic evaluation and natural settlement, independent of recording."""
import copy
import hashlib
import io
import json
from pathlib import Path


def predictor(source, kind, weights=None, device="cpu"):
    if kind == "traditional":
        return lambda obs: None
    if weights is None:
        from tools.training.foxglove_replay import make_predictor
        return make_predictor(source, kind, device)
    import numpy as np
    import torch
    from src.policy.recurrent_actor import RecurrentActor
    from src.policy.recurrent_observations import ACTOR_KEYS
    actor = RecurrentActor().to(device).eval()
    if isinstance(weights, bytes):
        weights = torch.load(io.BytesIO(weights), map_location=device, weights_only=True)
    actor.load_state_dict(weights)
    state = actor.initial_state(1)
    @torch.inference_mode()
    def predict(obs):
        nonlocal state
        tensors = {k:torch.as_tensor(np.asarray(obs[k])[None], device=device) for k in ACTOR_KEYS}
        _, _, actions, state = actor(tensors, state)
        return int(actions.item())
    return predict


def evaluate_scene(config, descriptor, directory, *, kind="checkpoint", source=None,
                   weights=None, model=None, record_mcap=False, device="cpu", combat_hud=True):
    from src.training.environment import make_environment
    from src.training.curriculum_metrics import TrackingMetrics
    from src.scoring.window import WindowScore
    from tools.training.view import _atomic_json
    settings = copy.deepcopy(config)
    if "curriculum" not in settings:
        from src.training.curriculum import implementation
        settings["curriculum"] = copy.deepcopy(implementation(descriptor).DEFAULT)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    env = make_environment(settings, directory / "environment",
                           controller="traditional" if kind == "traditional" else "policy")
    base = env.unwrapped
    base.collect_replay_events = True
    identity = dict(model=model or {"kind":kind}, scene=descriptor, source="synthetic_measurements")
    if model and kind != "traditional":
        from src.training.models import observation_contract, training_metadata
        runtime = training_metadata(settings, env)
        if model.get("observation_contract") is not None and model["observation_contract"] != observation_contract(env):
            env.close()
            raise ValueError("model observation/action contract differs")
        for key in ("simulator_config", "vision_config"):
            expected = model.get("fingerprints", {}).get(key)
            if expected is not None and expected != runtime["fingerprints"][key]:
                env.close()
                raise ValueError(f"model {key} fingerprint differs")
        identity["runtime_fingerprints"] = runtime["fingerprints"]
    if record_mcap:
        if kind == "traditional":
            from src.training.models import training_metadata
            identity["runtime_fingerprints"] = training_metadata(settings,env)["fingerprints"]
            identity["model"]["comparison_bridge_sha256"] = hashlib.sha256(base.bridge_binary.read_bytes()).hexdigest()
        base.replay_recording = (directory / ("replay.raw.mcap" if combat_hud else "replay.mcap"), identity)
    try:
        predict = predictor(source, kind, weights, device)
        obs, initial = env.reset(options={"curriculum_scene": descriptor})
        target_start_hp = next(r for r in base._response["data"]["evaluation"]["robots"] if r["robot_id"] == 2)["hp"]
        tracking = TrackingMetrics(obs)
        initial_thermal = copy.deepcopy(base._actor_semantic["thermal"])
        start = base._response["sim_time_ns"]
        score = WindowScore(base._response["round_id"], start, start+base.episode_steps*10_000_000)
        first = copy.deepcopy(base._response)
        first["data"]["events"] = list(base._bridge.event_history.values())+first["data"]["events"]
        score.ingest(first)
        action_digest, command_digest = hashlib.sha256(), hashlib.sha256()
        lost = requests = 0
        while True:
            action = predict(obs)
            untracked_request = action in (2,4,6,8) and not tracking.tracking(obs)
            obs, _, terminated, truncated, info = env.step(action)
            tracking.observe(obs)
            requests += int(info.get("shot_requested", False))
            lost += int(not tracking.tracking(obs))
            tracking.missing_requests += int(untracked_request and info.get("shot_requested",False))
            action_digest.update(json.dumps([action,info.get("executed_action"),info.get("selected_slot"),
                info.get("shot_requested"),info.get("shot_accepted")],separators=(",", ":")).encode())
            command_digest.update(json.dumps(base._last_replay_command,sort_keys=True,separators=(",", ":")).encode())
            score.ingest(base._response)
            if terminated or truncated:
                break
        end = base._response["sim_time_ns"]
        if end < score.end_ns:
            score.shorten(end)
        window = {k: info.get(k) for k in ("episode_damage", "actual_shots", "episode_steps",
                  "slot_switches", "episode_training_reward", "end_reason")}
        response = base._client.end_window(max_settle_steps=1000)
        score.ingest(response)
        if record_mcap:
            base._bridge.record_phase = "settling"
            base._bridge.record_snapshot(response)
        while response["data"]["settlement"]["status"] not in ("complete", "timed_out"):
            response = base._client.settle()
            score.ingest(response)
            if record_mcap:
                base._bridge.record_snapshot(response)
        final_score = score.finish(response)
        target = next(r for r in response["data"]["evaluation"]["robots"] if r["robot_id"] == 2)
        own = next(r for r in response["data"]["evaluation"]["robots"] if r["robot_id"] == 1)
        # Tail damage counts only when attributed to an eligible launch.
        killed = target["hp"] <= 0 and final_score["official_damage"] is not None and final_score["official_damage"] >= target_start_hp
        eligible_events = [s["damage_time_ns"] for s in final_score["projectiles"]
            if s["classification"] == "eligible" and s["damage"] > 0]
        kill_time = (max(eligible_events)-start)*1e-9 if killed and eligible_events else None
        summary = dict(window=window, settlement=final_score,
            action_sha256=action_digest.hexdigest(), command_sha256=command_digest.hexdigest(),
            start_ns=start, end_ns=end, settlement_end_ns=response["sim_time_ns"],
            status="complete" if final_score["status"] == "complete" else "settlement_timed_out")
        resource = own["resources"]
        row = {k:descriptor[k] for k in ("course", "group", "seed", "case", "scarce", "damage_capacity")}
        row.update(profile=descriptor["scenario"]["duel"]["target"]["profile"],
            motion_type=descriptor.get("motion_type"), controller=kind, status=summary["status"],
            admitted_scene=initial["curriculum_scene"], fireability=initial.get("fireability"),
            trackability=initial.get("trackability"), damage=final_score["official_damage"],
            actual_shots=final_score["eligible_shots"], damaging_hits=final_score["damaging_projectiles"],
            end_reason="target_destroyed" if killed else info["end_reason"], kill_time_s=kill_time,
            slot_switches=info["slot_switches"], episode_return=info["episode_training_reward"],
            steps=info["episode_steps"], shot_requests=requests, missing_steps=lost,
            tracking=tracking.result(), ordinary_heat_locks=resource.get("ordinary_heat_lock_count",0),
            round_heat_lock_count=resource.get("round_heat_lock_count",0),
            round_heat_lock=resource["round_locked"], heat_locked_s=own.get("heat_locked_s",0.),
            initial_actor_feedback=dict(thermal=initial_thermal), summary=summary)
        if "visibility_admission" in initial:row["visibility_admission"]=initial["visibility_admission"]
        if record_mcap:
            base._bridge.close_recording(summary)
            if combat_hud:
                from tools.training.combat_hud import augment_mcap
                augment_mcap(directory / "replay.raw.mcap", directory / "replay.mcap")
        _atomic_json(directory / "summary.json", summary)
        _atomic_json(directory / "result.json", row)
        return row
    finally:
        env.close()
