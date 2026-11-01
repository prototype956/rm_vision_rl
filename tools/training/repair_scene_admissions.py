"""Repair only explicitly failed cases; preserve all accepted descriptors verbatim."""
import copy
import hashlib
from pathlib import Path


def repair(args):
    from src.training.curriculum_v2 import digest
    from src.training.environment import make_environment
    from src.training.scene_admission import difficulty_signature, validate_visibility_certificate
    from src.training.scene_placement import SAMPLER, identity
    from tools.training.curriculum_v3 import evaluate, manifest, read, write
    roles, audits = {}, []
    for role in args.roles:
        source = args.scenarios / role
        if not source.exists():
            source = args.scenarios / f"preflight-{role}" / role
        frozen = manifest(source / "curriculum/scenarios.json")
        report = read(source / "preflight.json")
        config = read(source / "config.json")
        if config["curriculum"].get("release_revision") != 3:
            raise ValueError("repair requires a revision-3 preflight")
        failed = {(f["split"], f["course"], f["case"]) for f in report["failures"]}
        if len(failed) != len(report["failures"]):
            raise ValueError("duplicate failure identity")
        probe = make_environment(config, args.output / role / "identity-probe")
        try:
            native = hashlib.sha256(Path(probe.unwrapped.simulator_binary).read_bytes()).hexdigest()
            simulation = hashlib.sha256(Path(probe.unwrapped.simulator_config).read_bytes()).hexdigest()
        finally:
            probe.close()
        previous = copy.deepcopy(frozen)
        preserved, jobs, keys = [], [], []
        known = set()
        for split in ("calibration", "validation", "test"):
            for course, scenes in enumerate(frozen[split]):
                for index, descriptor in enumerate(scenes):
                    key = (split, course, descriptor["case"])
                    known.add(key)
                    if key not in failed:
                        validate_visibility_certificate(descriptor, native, simulation)
                        preserved.append((key, digest(descriptor)))
                        continue
                    # The old blocked manifest may still contain admitted=True;
                    # only the source failure report determines what may change.
                    candidate = copy.deepcopy(descriptor)
                    candidate.pop("admitted", None)
                    candidate.pop("visibility_certificate", None)
                    candidate["placement_search"] = dict(sampler=SAMPLER, hints_sha256=identity())
                    jobs.append(candidate)
                    keys.append((split, course, index))
        if not failed <= known:
            raise ValueError("failure report refers to missing scene identities")
        # Group jobs by split/course: case IDs are only unique within that scope.
        results, failures = {}, []
        for split, course in dict.fromkeys((s, c) for s, c, _ in keys):
            batch = [(key, job) for key, job in zip(keys, jobs) if key[:2] == (split, course)]
            rows = evaluate(config, [d for _, d in batch],
                args.output / role / split / f"C{course}", args.workers, preflight=True)
            for (key, candidate), row in zip(batch, rows):
                _, _, index = key
                original = previous[split][course][index]
                entry = dict(role=role, split=split, course=course, case=original["case"],
                             original=original, original_sha256=digest(original))
                if "error" in row or "failure" in row:
                    replacement = dict(candidate, admitted=False)
                    failure = dict(split=split, course=course, case=original["case"],
                                   error=row.get("error", row.get("failure", "")))
                    failures.append(failure)
                    entry.update(passed=False, error=failure["error"])
                else:
                    replacement = dict(row["admitted_scene"], admitted=True)
                    validate_visibility_certificate(replacement, native, simulation)
                    entry.update(passed=True, replacement=replacement)
                if difficulty_signature(original) != difficulty_signature(replacement):
                    raise RuntimeError("repair changed fixed task difficulty")
                entry["difficulty_unchanged"] = True
                scenes = frozen[split][course]
                scenes[index] = replacement
                results[key] = row
                audits.append(entry)
            write(args.output / role / "repair-progress.json", dict(
                requested=len(jobs), completed=len(results), failed=len(failures)))
        for key, expected in preserved:
            split, course, case = key
            descriptor = next(d for d in frozen[split][course] if d["case"] == case)
            if digest(descriptor) != expected:
                raise RuntimeError("accepted descriptor changed during repair")
        frozen.pop("sha256")
        frozen["sha256"] = digest(frozen)
        write(args.output / role / "curriculum/scenarios.json", frozen)
        write(args.output / role / "config.json", config)
        write(args.output / role / "preserved.json", dict(
            source_manifest_sha256=previous["sha256"], count=len(preserved),
            descriptor_hashes=[dict(split=k[0], course=k[1], case=k[2], sha256=h) for k, h in preserved]))
        write(args.output / role / "preflight.json", dict(passed=not failures, failures=failures,
            historical_frozen_failures=report.get("historical_frozen_failures", []),
            repaired_from=str(source), preserved_cases=len(preserved), repaired_cases=len(jobs)-len(failures)))
        roles[role] = dict(preserved=len(preserved), requested=len(jobs),
                          repaired=len(jobs)-len(failures), failures=failures)
        write(args.output / "repair-audit.json", dict(sampler=SAMPLER, hints_sha256=identity(),
            native_sha256=native, changes=audits))
        write(args.output / "report.json", dict(status="running", roles={r:v["failures"] for r,v in roles.items()}))
    complete = not any(value["failures"] for value in roles.values())
    write(args.output / "repair-summary.json", dict(status="complete" if complete else "blocked",
        roles=roles, preserved=sum(v["preserved"] for v in roles.values()),
        requested=sum(v["requested"] for v in roles.values()),
        repaired=sum(v["repaired"] for v in roles.values()),
        native_attempt_limit=128, accepted_descriptors_unchanged=True,
        task_difficulty_unchanged=True, baseline_rebuilt=False, training_started=False))
    write(args.output / "report.json", dict(status="complete" if complete else "blocked",
        roles={r:v["failures"] for r,v in roles.items()}))
    if not complete:
        raise RuntimeError("failed-case repair did not pass; no release or training")
