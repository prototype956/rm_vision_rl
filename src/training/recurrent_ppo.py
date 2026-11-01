"""Complete-episode masked PPO with epoch-refreshed state caches and TBPTT.

Caches are exact for parameters at the start of an epoch. Later minibatches use
an approximation, reduced by burn-in and the KL guard, not an exact replay of
all preceding history under each successive optimizer update.
"""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import torch

from src.policy.recurrent_actor import CONTRACT
from src.training.recurrent_policy import RecurrentPolicy, combine_states, select_state, detach_state
from src.training.recurrent_rollout import EpisodeWorkers, collect_episodes


def segments(episodes, length, burn_in):
    return [(e, start, max(0, start - burn_in), min(length, len(ep["actions"]) - start))
            for e, ep in enumerate(episodes) for start in range(0, len(ep["actions"]), length)]


def pack_episodes(episodes, device):
    """Time-major storage. Inactive positions repeat a safe observation but never update state."""
    length = max(len(ep["actions"]) for ep in episodes)
    observations = {}
    for key in episodes[0]["observations"]:
        columns = []
        for ep in episodes:
            source = ep["observations"][key][:-1]
            columns.append(np.concatenate((source, np.repeat(source[-1:], length-len(source), axis=0))))
        observations[key] = torch.as_tensor(np.stack(columns, axis=1), device=device)
    return observations


@torch.no_grad()
def state_cache(policy, packed, episodes, chunks):
    needed = {(e, burn) for e, _, burn, _ in chunks}
    cache = {}
    state = policy.initial_state(len(episodes))
    lengths = torch.tensor([len(ep["actions"]) for ep in episodes], device=packed["features"].device)
    for t in range(int(lengths.max().item())):
        for e in range(len(episodes)):
            if (e, t) in needed:
                cache[e, t] = select_state(state, slice(e, e+1))
        obs = {k: v[t] for k, v in packed.items()}
        # Replay only recurrent encoders; truth/value/distribution are unnecessary here.
        _, actor = policy.actor.logits(obs, state[0], (t < lengths)[:, None])
        critic = policy.critic.advance(obs, state[1], (t < lengths)[:, None])
        state = actor, critic
    return cache


class RecurrentPPO:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.device = torch.device(config["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(config["torch_threads"])
        torch.manual_seed(config["seed"])
        self.rng = np.random.default_rng(config["seed"])
        self.policy = RecurrentPolicy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config["ppo"]["learning_rate"])
        self.num_timesteps = self.completed_updates = 0
        self.curriculum_state = None
        self.pending_validation = False

    def update(self, episodes):
        p = self.config["ppo"]
        chunks = segments(episodes, p["sequence_length"], p["burn_in"])
        from src.training.sequence_storage import replay_cache, window, move_state
        all_adv = np.concatenate([ep["advantages"] for ep in episodes])
        adv_mean, adv_std = float(all_adv.mean()), float(all_adv.std()) + 1e-8
        metrics = []
        stopped = False
        self.policy.train()
        for epoch in range(p["n_epochs"]):
            epoch_started = perf_counter()
            cache = replay_cache(self.policy, episodes, chunks, self.config.get("runtime", {}).get("replay_batch_episodes", 64))
            order = self.rng.permutation(len(chunks))
            for begin in range(0, len(order), p["batch_size"]):
                batch = [chunks[i] for i in order[begin:begin+p["batch_size"]]]
                packed, targets, sizes, burn_sizes, burn_steps = window(
                    episodes, batch, self.device, adv_mean, adv_std)
                state = move_state(combine_states([cache[c[0], c[2]] for c in batch]), self.device)
                with torch.no_grad():
                    for t in range(burn_steps):
                        active = (t >= burn_steps-burn_sizes)[:, None]
                        obs = {k: v[t] for k, v in packed.items()}
                        _, actor = self.policy.actor.logits(obs, state[0], active)
                        critic = self.policy.critic.advance(obs, state[1], active)
                        state = actor, critic
                state = detach_state(state)
                logp, entropy, values, masks = [], [], [], []
                for t in range(int(sizes.max().item())):
                    active = t < sizes
                    obs = {k: v[t+burn_steps] for k, v in packed.items()}
                    dist, value, state = self.policy(obs, state, active[:, None])
                    logp.append(dist.log_prob(targets["actions"][t].long()))
                    entropy.append(dist.entropy())
                    values.append(value)
                    masks.append(active)
                valid = torch.stack(masks)
                old_logp = targets["log_probs"][valid]
                advantage = targets["advantages"][valid]
                returns = targets["returns"][valid]
                log_ratio = torch.stack(logp)[valid] - old_logp
                ratio = log_ratio.exp()
                kl = ((ratio - 1) - log_ratio).mean()
                if not torch.isfinite(kl):
                    raise FloatingPointError("non-finite recurrent PPO KL")
                if kl.item() > p["target_kl"]:
                    stopped = True
                    break
                policy_loss = -torch.minimum(advantage*ratio,
                    advantage*ratio.clamp(1-p["clip_range"], 1+p["clip_range"])).mean()
                value_loss = (torch.stack(values)[valid] - returns).square().mean()
                mean_entropy = torch.stack(entropy)[valid].mean()
                loss = policy_loss + p["vf_coef"]*value_loss - p["ent_coef"]*mean_entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite recurrent PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), p["max_grad_norm"], error_if_nonfinite=True)
                self.optimizer.step()
                metrics.append([loss.item(), policy_loss.item(), value_loss.item(), mean_entropy.item(), kl.item(), float(grad)])
            if "curriculum" in self.config:
                print(f"PPO update={self.completed_updates+1} epoch={epoch+1}/{p['n_epochs']} seconds={perf_counter()-epoch_started:.2f} kl_stop={stopped}", flush=True)
            if stopped:
                break
        if not metrics:
            raise RuntimeError("KL guard rejected the first minibatch; investigate state/log-probability parity")
        self.num_timesteps += sum(len(ep["actions"]) for ep in episodes)
        self.completed_updates += 1
        result = dict(zip(("loss", "policy_loss", "value_loss", "entropy", "approx_kl", "gradient_norm"),
                          np.mean(metrics, axis=0).tolist()))
        result.update(epochs=epoch+1, minibatches=len(metrics), kl_stopped=stopped,
                      samples=sum(len(ep["actions"]) for ep in episodes), episodes=len(episodes))
        return result

    def save(self, path, metadata):
        path = Path(path)
        data = copy.deepcopy(metadata)
        data.update(num_timesteps=self.num_timesteps, completed_updates=self.completed_updates,
                    policy_contract=CONTRACT, config=self.config, curriculum_state=self.curriculum_state,
                    pending_validation=self.pending_validation)
        stream = io.BytesIO()
        torch.save(dict(policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(), rng=self.rng.bit_generator.state,
                        cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None), stream)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".zip", delete=False) as f:
            temporary = Path(f.name)
        try:
            with ZipFile(temporary, "w") as archive:
                archive.writestr("rmvision.json", json.dumps(data, indent=2, allow_nan=False))
                archive.writestr("recurrent.pt", stream.getvalue())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, device=None):
        from src.training.models import read_checkpoint_metadata, _digest
        from src.policy.recurrent_observations import SCHEMA
        metadata = read_checkpoint_metadata(path)
        if metadata["policy_contract"] != CONTRACT:
            raise ValueError("recurrent model contract mismatch")
        if (metadata["observation"].get("version") != 5 or
                metadata["observation"].get("schema_sha256") != _digest(SCHEMA)):
            raise ValueError("recurrent observation schema mismatch; train a new model")
        config = copy.deepcopy(metadata["config"])
        if device is not None:
            config["device"] = str(device)
        result = cls(config)
        with ZipFile(path) as archive:
            data = torch.load(io.BytesIO(archive.read("recurrent.pt")), map_location=result.device, weights_only=True)
        result.policy.load_state_dict(data["policy"])
        result.optimizer.load_state_dict(data["optimizer"])
        torch.set_rng_state(data["torch_rng"].cpu())
        if result.device.type == "cuda" and data.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(data["cuda_rng"].cpu(), result.device)
        result.rng.bit_generator.state = data["rng"]
        result.num_timesteps = metadata["num_timesteps"]
        result.completed_updates = metadata["completed_updates"]
        result.curriculum_state = metadata.get("curriculum_state")
        result.pending_validation = metadata.get("pending_validation",False)
        # No live environment/hidden states are restored across a model/session boundary.
        return result


def save_rollout(episodes, directory):
    directory.mkdir(parents=True, exist_ok=False)
    for index, ep in enumerate(episodes):
        arrays = {"obs_"+k: v for k, v in ep["observations"].items()}
        arrays.update({k: ep[k] for k in ("actions", "log_probs", "values", "rewards", "terminated",
                                        "truncated", "advantages", "returns")})
        np.savez_compressed(directory / f"episode-{index:04d}.npz", **arrays)
        (directory / f"episode-{index:04d}.json").write_text(json.dumps(
            {k: ep[k] for k in ("bootstrap", "info", "worker", "execution")}, allow_nan=False), encoding="utf-8")


def export_reference(config, output, scene_descriptor=None):
    """Fresh evaluation packets when a resumed run has no remaining training.

    This does not collect a rollout or increment the cumulative training budget.
    """
    from src.training.environment import make_environment
    env = make_environment(config, Path(output)/"export-check")
    try:
        options = {"curriculum_scene":scene_descriptor} if scene_descriptor is not None else None
        observation, _ = env.reset(options=options)
        rows = [observation]
        for _ in range(2):
            observation, _, terminated, truncated, _ = env.step(0)
            rows.append(observation)
            if terminated or truncated:
                break
        return {key:np.stack([row[key] for row in rows]) for key in observation}
    finally:
        env.close()


def run_recurrent_training(config, resume=None, output_dir=None, *, stop_after_updates=None, frozen_scenarios=None):
    from src.training.environment import make_environment
    from src.training.models import training_metadata, read_checkpoint_metadata, publish_latest
    from src.training.train import _write_manifest
    from src.policy.recurrent_deployment import export_actor, verify_export
    from stable_baselines3.common.logger import configure
    from src.training.config import ROOT
    if config.get("curriculum", {}).get("version") == 3 and config["curriculum"].get("calibration") is None:
        raise ValueError("v3 training requires frozen traditional calibration")
    if output_dir is None:
        parent = ROOT / "artifacts/training"
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="gru-", dir=parent))
    else:
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
    probe = make_environment(config, output / "environment")
    try:
        metadata = training_metadata(config, probe)
    finally:
        probe.close()
    config = metadata["config"]
    if resume:
        previous = read_checkpoint_metadata(resume)
        if previous["observation"] != metadata["observation"] or previous["fingerprints"] != metadata["fingerprints"]:
            raise ValueError("resume observation/environment contract differs")
        model = RecurrentPPO.load(resume, device=config["device"])
        model.config = config
    else:
        model = RecurrentPPO(config)
    metadata["sampling"].update(start_method="spawn", barrier="all_workers_at_episode_boundary")
    metadata["execution"] = dict(cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_name=torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
        logical_cpus=os.cpu_count())
    manifest = dict(status="running", config=config, metadata=metadata, resume_from=str(resume) if resume else None,
                    requested_timesteps=config["total_timesteps"], start_timesteps=model.num_timesteps, last_completed_timesteps=model.num_timesteps, completed_updates=model.completed_updates)
    _write_manifest(output / "run.json", manifest)
    logger = configure(str(output / "logs"), ["stdout", "csv", "tensorboard"])
    curriculum = evaluator = None
    if "curriculum" in config:
        from src.training.curriculum import implementation
        Curriculum = implementation(config["curriculum"]).Curriculum
        from src.training.curriculum_evaluation import CurriculumEvaluator
        curriculum = Curriculum(config["curriculum"], config["seed"], model.curriculum_state)
        evaluator = CurriculumEvaluator(config, output, previous.get("curriculum_scenarios") if resume else frozen_scenarios)
    runtime = config.get("runtime")
    if runtime:
        os.environ["RM_TRAINING_THREADS"] = str(runtime["simulator_threads"])
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"
        os.environ["RUST_LOG"] = "error"
        os.environ["RM_BOUNDED_NATIVE_LOGS"] = "1"
    workers = None
    episodes = None
    started = perf_counter()
    initial_updates = model.completed_updates
    target = config["total_timesteps"] if curriculum else model.num_timesteps + config["total_timesteps"]
    print(f"GRU complete-episode training: {output}; {config['n_envs']} workers", flush=True)
    try:
        if evaluator is not None and resume is None and frozen_scenarios is None:
            evaluator.preflight()
        if evaluator is not None:
            metadata["curriculum_scenarios"] = evaluator.manifest
        if model.pending_validation and curriculum is not None:
            from src.training.curriculum import implementation
            validate_update = implementation(config["curriculum"]).validate_update
            validate_update(curriculum,evaluator,model.policy.actor,model.num_timesteps,model.completed_updates)
            model.pending_validation = False
            model.curriculum_state = curriculum.state()
        workers = EpisodeWorkers(config, output)
        while model.num_timesteps < target:
            if runtime:
                from src.training.runtime import low_disk
                if low_disk(output, runtime):
                    model.save(output / "disk-paused.zip", metadata)
                    manifest.update(status="disk_paused", last_completed_timesteps=model.num_timesteps)
                    return output
            previous_steps = model.num_timesteps
            promoted = False
            sample_start = perf_counter()
            episodes = collect_episodes(model.policy, workers, config, curriculum)
            sample_time = perf_counter() - sample_start
            update_start = perf_counter()
            metrics = model.update(episodes)
            optimize_time = perf_counter() - update_start
            if curriculum is not None and runtime:
                model.curriculum_state = curriculum.state()
                model.pending_validation = True
                model.save(output / "updated-pending-validation.zip",metadata)
            if curriculum is not None:
                from src.training.curriculum import implementation
                validate_update = implementation(config["curriculum"]).validate_update
                promoted = validate_update(curriculum, evaluator, model.policy.actor,
                                           model.num_timesteps, model.completed_updates)
                model.curriculum_state = curriculum.state()
                model.pending_validation = False
                logger.record("curriculum/stage", curriculum.stage)
                logger.record("curriculum/streak", curriculum.streak)
                manifest.update(curriculum=curriculum.state())
            if runtime:
                from src.training.runtime import diagnostics, episode_summaries
                episode_summaries(episodes, output, model.completed_updates)
                diagnostics(episodes, output, model.completed_updates, runtime, config["seed"])
            else:
                save_rollout(episodes, output / "rollouts" / f"update-{model.completed_updates:04d}")
            for key, value in metrics.items():
                logger.record("train/"+key, value)
            for name, key in (("damage", "episode_damage"), ("actual_shots", "actual_shots"),
                              ("slot_switches", "slot_switches"), ("return", "episode_training_reward")):
                logger.record("rollout/"+name, float(np.mean([ep["info"][key] for ep in episodes])))
            if "thermal_reward" in config:
                for key in ("ordinary_heat_penalty", "round_heat_penalty", "locked_time_penalty", "heat_locked_s"):
                    logger.record("rollout/"+key, float(np.mean([ep["info"]["episode_"+key] for ep in episodes])))
                for key in ("ordinary", "round"):
                    logger.record("rollout/"+key+"_locked_episode_fraction", float(np.mean([
                        ep["info"]["episode_"+key+"_heat_locks"] > 0 for ep in episodes])))
            logger.record("time/sampling_seconds", sample_time)
            logger.record("time/optimization_seconds", optimize_time)
            logger.record("time/total_timesteps", model.num_timesteps)
            logger.record("time/completed_updates", model.completed_updates)
            logger.dump(step=model.num_timesteps)
            if runtime:
                from src.training.runtime import checkpoint
                checkpoint(model, metadata, output, runtime, previous_steps, promoted)
            elif model.completed_updates % config["checkpoint_updates"] == 0:
                checkpoint = output / f"checkpoint_{model.num_timesteps}.zip"
                model.save(checkpoint, metadata)
                publish_latest(checkpoint, output / "latest.zip")
            manifest.update(last_completed_timesteps=model.num_timesteps, completed_updates=model.completed_updates)
            _write_manifest(output / "run.json", manifest)
            if stop_after_updates is not None and model.completed_updates - initial_updates >= stop_after_updates:
                model.save(output / "paused.zip", metadata)
                export_actor(model.policy.actor, output / "actor", config=config, training_steps=model.num_timesteps)
                verify_export(model.policy.actor, output / "actor", episodes[-1]["observations"])
                manifest.update(status="paused", training_seconds=perf_counter()-started)
                return output
        if evaluator is not None:
            final_validation = {str(c): evaluator.evaluate(model.policy.actor, c, model.completed_updates) for c in range(len(evaluator.validation))}
            final_reports = {str(c): evaluator.evaluate(model.policy.actor, c, model.completed_updates, test=True) for c in range(len(evaluator.validation))}
            manifest.update(final_course_reports=final_reports, final_validation_reports=final_validation, curriculum_passed=curriculum.stage == len(evaluator.validation)-1 and curriculum.mastered and all(r["passed"] for r in [*final_reports.values(), *final_validation.values()]), training_budget_complete=model.num_timesteps >= target,
                thermal_passed=all(r.get("thermal", {}).get("passed", True) for r in [*final_reports.values(), *final_validation.values()]))
        model.save(output / "final.zip", metadata)
        publish_latest(output / "final.zip", output / "latest.zip")
        export_actor(model.policy.actor, output / "actor", config=config, training_steps=model.num_timesteps)
        reference = episodes[-1]["observations"] if episodes else export_reference(
            config, output, evaluator.validation[0][0] if evaluator is not None else None)
        verify_export(model.policy.actor, output / "actor", reference)
        manifest.update(status="complete", training_seconds=perf_counter()-started)
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if workers is not None:
            workers.close()
        if evaluator is not None:
            evaluator.close()
        logger.close()
        _write_manifest(output / "run.json", manifest)
    return output
