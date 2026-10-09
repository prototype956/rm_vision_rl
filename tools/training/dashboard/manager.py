"""Process and artifact lifecycle for the local training dashboard."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any
from urllib.parse import quote

from src.training.config import ROOT, validate_config
from tools.training.dashboard.readers import list_runs, read_json, run_summary, tail_text
from tools.training.dashboard.schemas import (
    RequestError, TOOL_NAMES, validate_new_payload, validate_resume_payload,
    validate_tool_payload,
)


TERMINAL = {"complete", "interrupted", "failed", "stale", "unmanaged"}
TRAINING_MODES = {"new", "resume"}
TOOL_MODULES = {
    "analysis": "tools.training.analysis",
    "compare": "tools.training.compare",
    "diagnose": "tools.training.diagnose",
    "benchmark": "tools.training.benchmark",
    "view": "tools.training.view",
    "manual": "tools.training.manual",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _proc_starttime(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return fields[21]
    except (FileNotFoundError, OSError, IndexError, ValueError):
        return None


def _proc_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    except (FileNotFoundError, OSError, UnicodeError):
        return ""


class TrainingManager:
    """Own dashboard jobs while keeping the trainer in a separate process."""

    def __init__(self, root: Path = ROOT, *, poll_seconds: float = 2.0,
                 stop_timeout_seconds: float = 40.0, max_training_jobs: int = 1,
                 max_active_jobs: int | None = None):
        self.root = Path(root).resolve()
        self.training_root = (self.root / "artifacts" / "training").resolve()
        self.jobs_root = self.training_root / ".web-jobs"
        self.config_root = self.jobs_root / "configs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = max(0.2, float(poll_seconds))
        self.stop_timeout_seconds = max(5.0, float(stop_timeout_seconds))
        self.max_training_jobs = max(1, int(max_training_jobs))
        # A simulator-backed tool consumes the same host resources as a
        # training process.  Keep one shared capacity limit by default while
        # retaining max_training_jobs for compatibility with old sidecars.
        self.max_active_jobs = max(1, int(max_active_jobs if max_active_jobs is not None
                                          else max_training_jobs))
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._load_jobs()
        self._monitor = threading.Thread(target=self._monitor_loop, name="training-dashboard", daemon=True)
        self._monitor.start()

    def close(self) -> None:
        self._stop_event.set()
        self._monitor.join(timeout=max(1.0, self.poll_seconds * 2))

    def _load_jobs(self) -> None:
        for path in sorted(self.jobs_root.glob("*.json")):
            data = read_json(path)
            if isinstance(data, dict) and data.get("job_id"):
                self._jobs[data["job_id"]] = data
        self._refresh()

    def _record_path(self, job_id: str) -> Path:
        return self.jobs_root / f"{job_id}.json"

    def _save(self, record: dict[str, Any]) -> None:
        record["updated_at"] = _now()
        self._jobs[record["job_id"]] = copy.deepcopy(record)
        _atomic_json(self._record_path(record["job_id"]), record)

    def _pid_alive(self, record: dict[str, Any]) -> bool:
        pid = record.get("pid")
        if type(pid) is not int or pid <= 0 or _proc_starttime(pid) is None:
            return False
        expected = record.get("pid_starttime")
        if expected is not None and _proc_starttime(pid) != expected:
            return False
        command = _proc_command(pid)
        return "src.training.train" in command or "tools.training." in command

    def _active_training_count(self) -> int:
        count = 0
        for record in self._jobs.values():
            if record.get("mode") not in TRAINING_MODES:
                continue
            status = self._status(record)
            if status not in TERMINAL:
                count += 1
        return count

    def _active_job_count(self) -> int:
        count = 0
        for record in self._jobs.values():
            if self._status(record) not in TERMINAL:
                count += 1
        return count

    def _status(self, record: dict[str, Any]) -> str:
        process_alive = self._pid_alive(record)
        run_dir = Path(record.get("output_dir", ""))
        if record.get("mode") in {"analysis", "tool"}:
            if process_alive:
                return "stop_requested" if record.get("stop_requested") else "running"
            if record.get("returncode") is not None:
                return "complete" if record["returncode"] == 0 else "failed"
            # A dashboard restart loses the Popen object.  Recover a terminal
            # result from the tool's own atomically written manifest/report.
            if self._tool_terminal_state(record) is not None:
                return "complete" if self._tool_terminal_state(record) == "complete" else "failed"
            if record.get("pid") is None:
                return "unmanaged"
            return "stale"
        manifest = read_json(run_dir / "run.json", {}) if run_dir else {}
        manifest_status = manifest.get("status") if isinstance(manifest, dict) else None
        if record.get("stop_requested") and process_alive:
            return "stop_requested"
        if manifest_status in {"complete", "interrupted", "failed"}:
            return manifest_status
        if process_alive:
            return manifest_status if manifest_status in {"starting", "running"} else "starting"
        if record.get("returncode") is not None:
            return "failed" if record["returncode"] else (manifest_status or "complete")
        if manifest_status in {"starting", "running"}:
            return "stale"
        if record.get("pid") is None:
            return "unmanaged"
        return "failed"

    def _refresh_record(self, record: dict[str, Any]) -> None:
        process = self._processes.get(record.get("job_id"))
        if process is not None:
            returncode = process.poll()
            if returncode is not None and record.get("returncode") is None:
                record["returncode"] = returncode
                record["finished_at"] = _now()
                self._processes.pop(record.get("job_id"), None)
        pid_alive = self._pid_alive(record)
        if record.get("mode") in {"analysis", "tool"} and not pid_alive and record.get("returncode") is None:
            tool_state = self._tool_terminal_state(record)
            if tool_state is not None:
                record["returncode"] = 0 if tool_state == "complete" else 1
                record["finished_at"] = _now()
        if record.get("stop_requested") and pid_alive:
            deadline = record.get("stop_deadline")
            if deadline and time.time() >= deadline:
                try:
                    os.killpg(int(record["pgid"]), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, ValueError):
                    pass
                record["force_kill_sent_at"] = _now()
        if not pid_alive and record.get("pid") is not None and record.get("returncode") is None:
            # Popen is not available after a dashboard restart.  A dead process
            # is only considered failed when the run did not write a terminal state.
            run = read_json(Path(record.get("output_dir", "")) / "run.json", {}) or {}
            if run.get("status") in {"complete", "interrupted", "failed"}:
                record["returncode"] = 0 if run["status"] == "complete" else 1
                record["finished_at"] = _now()
        status = self._status(record)
        if record.get("status") != status:
            record["status"] = status
            self._save(record)
        elif record.get("stop_requested") and record.get("status") == "stop_requested":
            self._save(record)
        if (record.get("auto_analysis") and status == "complete" and not pid_alive
                and not record.get("auto_analysis_started")):
            record["auto_analysis_started"] = True
            self._save(record)
            try:
                self.start_analysis(record["output_dir"])
            except (RequestError, OSError):
                record["auto_analysis_error"] = "unable to start automatic analysis"
                self._save(record)

    def _refresh(self) -> None:
        with self._lock:
            for record in list(self._jobs.values()):
                self._refresh_record(record)

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self.poll_seconds):
            try:
                self._refresh()
            except Exception:
                # A malformed historical sidecar must not kill the monitor.
                continue

    def _tool_manifest_paths(self, record: dict[str, Any]) -> list[Path]:
        """Return known terminal artifact manifests for a managed tool job."""
        root = Path(record.get("output_dir", ""))
        if not root.is_dir():
            return []
        tool = record.get("tool") or ("analysis" if record.get("mode") == "analysis" else "")
        patterns = {
            "analysis": ("analysis/*/analysis.json",),
            "compare": ("compare-*/comparison.json", "comparison.json"),
            "diagnose": ("diagnose-*/run.json", "diagnostic-*/run.json", "diagnostic/run.json"),
            "benchmark": ("results.json",),
            "view": ("eval-*/replay.json", "eval-scenes-*/*/replay.json"),
            "manual": ("manual-*/run.json",),
        }.get(tool, ())
        try:
            created = datetime.fromisoformat(str(record.get("created_at", ""))).timestamp()
        except (TypeError, ValueError, OverflowError):
            created = 0.0
        paths = []
        for pattern in patterns:
            for path in root.glob(pattern):
                try:
                    if path.is_file() and path.stat().st_mtime + 1.0 >= created:
                        paths.append(path)
                except OSError:
                    continue
        return paths

    def _tool_terminal_state(self, record: dict[str, Any]) -> str | None:
        """Infer a completed/failed tool job after a service restart."""
        paths = self._tool_manifest_paths(record)
        if not paths:
            return None
        tool = record.get("tool") or ("analysis" if record.get("mode") == "analysis" else "")
        path = max(paths, key=lambda item: item.stat().st_mtime)
        state = read_json(path, {}) or {}
        if tool == "benchmark":
            return ("complete" if isinstance(state, dict) and state.get("trials")
                    and (Path(record.get("output_dir", "")) / "report.md").is_file() else None)
        if tool == "view":
            # replay.json and evaluation_scenes.json are terminal artifacts;
            # their payload shape intentionally has no status field.
            return "complete" if isinstance(state, dict) else None
        status = state.get("status") if isinstance(state, dict) else None
        if status in {"complete", "partial", "closed"}:
            return "complete"
        if status in {"failed", "interrupted", "incomplete"}:
            return "failed"
        return None

    def _output_dir(self) -> Path:
        self.training_root.mkdir(parents=True, exist_ok=True)
        for _ in range(10):
            name = f"web-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            path = self.training_root / name
            if not path.exists():
                return path
        raise RequestError("unable to allocate a unique training output directory")

    def _spawn(self, record: dict[str, Any], command: list[str], *, log_path: Path) -> dict[str, Any]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab")
        try:
            process = subprocess.Popen(
                command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                close_fds=True, start_new_session=True,
            )
        except OSError:
            log.close()
            raise
        finally:
            log.close()
        record.update(pid=process.pid, pgid=process.pid, pid_starttime=_proc_starttime(process.pid),
                      log_path=str(log_path), status="starting")
        self._processes[record["job_id"]] = process
        self._save(record)
        return record

    def _check_capacity(self) -> None:
        self._refresh()
        if self._active_training_count() >= self.max_training_jobs:
            raise RequestError("another training job is already active")

    def _check_active_capacity(self) -> None:
        self._refresh()
        if self._active_job_count() >= self.max_active_jobs:
            raise RequestError("another dashboard job is already active")

    def start_new(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_capacity()
            self._check_active_capacity()
            request = validate_new_payload(payload, self.root)
            job_id = uuid.uuid4().hex
            output = self._output_dir()
            config = copy.deepcopy(request["config"])
            config["total_timesteps"] = request["requested_timesteps"]
            config = validate_config(config)
            config_path = self.config_root / f"{job_id}.json"
            _atomic_json(config_path, config)
            record = {
                "job_id": job_id, "mode": "new", "status": "queued", "output_dir": str(output),
                "config_path": str(config_path), "preset": request["preset"],
                "requested_timesteps": request["requested_timesteps"],
                "actual_timesteps": request["actual_timesteps"],
                "auto_analysis": request["auto_analysis"], "created_at": _now(),
                "stop_requested": False,
            }
            command = [sys.executable, "-u", "-m", "src.training.train", "--config", str(config_path),
                       "--timesteps", str(request["requested_timesteps"]), "--output-dir", str(output)]
            record["command"] = command
            self._save(record)
            try:
                return self._spawn(record, command, log_path=self.jobs_root / f"{job_id}.log")
            except OSError as error:
                record.update(status="failed", returncode=127, error=str(error), finished_at=_now())
                self._save(record)
                raise RequestError(f"unable to start training: {error}") from error

    def start_resume(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._check_capacity()
            self._check_active_capacity()
            request = validate_resume_payload(payload, self.root)
            job_id = uuid.uuid4().hex
            output = self._output_dir()
            config = copy.deepcopy(request["config"])
            record = {
                "job_id": job_id, "mode": "resume", "status": "queued", "output_dir": str(output),
                "source_checkpoint": str(request["checkpoint"]),
                "requested_timesteps": request["requested_timesteps"],
                "actual_timesteps": request["actual_timesteps"],
                "auto_analysis": request["auto_analysis"], "created_at": _now(),
                "stop_requested": False, "effective_config": config,
            }
            command = [sys.executable, "-u", "-m", "src.training.train", "--resume",
                       str(request["checkpoint"]), "--timesteps", str(request["requested_timesteps"]),
                       "--device", config["device"], "--n-envs", str(config["n_envs"]),
                       "--output-dir", str(output)]
            record["command"] = command
            self._save(record)
            try:
                return self._spawn(record, command, log_path=self.jobs_root / f"{job_id}.log")
            except OSError as error:
                record.update(status="failed", returncode=127, error=str(error), finished_at=_now())
                self._save(record)
                raise RequestError(f"unable to start resume training: {error}") from error

    def start_analysis(self, run_dir: str | Path) -> dict[str, Any]:
        with self._lock:
            self._check_active_capacity()
            run_dir = Path(run_dir).resolve()
            try:
                run_dir.relative_to(self.training_root)
            except ValueError as error:
                raise RequestError("run directory must be below artifacts/training") from error
            manifest = read_json(run_dir / "run.json", {}) or {}
            if manifest.get("status") != "complete":
                raise RequestError("analysis requires a completed training run")
            self._refresh()
            for record in self._jobs.values():
                if record.get("mode") == "analysis" and Path(record.get("output_dir", "")).resolve() == run_dir:
                    if self._status(record) not in TERMINAL:
                        raise RequestError("analysis is already running for this run")
            job_id = uuid.uuid4().hex
            record = {
                "job_id": job_id, "mode": "analysis", "status": "queued", "output_dir": str(run_dir),
                "created_at": _now(), "stop_requested": False, "auto_analysis": False,
            }
            command = [sys.executable, "-u", "-m", "tools.training.analysis",
                       "--run-dir", str(run_dir), "--no-view"]
            record["command"] = command
            self._save(record)
            try:
                return self._spawn(record, command, log_path=self.jobs_root / f"{job_id}.log")
            except OSError as error:
                record.update(status="failed", returncode=127, error=str(error), finished_at=_now())
                self._save(record)
                raise RequestError(f"unable to start analysis: {error}") from error

    def tool_metadata(self) -> list[dict[str, Any]]:
        """Return the tools that can be submitted by the web console.

        The standalone JSON files remain the source of defaults.  Their
        values are returned for form prefilling, while request validation
        still applies the dashboard's path and field allowlists.
        """
        result = []
        config_root = self.root / "config" / "tools"
        for name, module in TOOL_MODULES.items():
            defaults = read_json(config_root / f"{name}.json", {})
            result.append({"name": name, "module": module,
                           "defaults": defaults if isinstance(defaults, dict) else {}})
        return result

    def start_tool(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Start one existing training tool in a managed process group.

        A generated config is passed with ``--config`` so the CLI keeps all
        domain logic and report formats.  The HTTP request never executes a
        tool synchronously and no shell is involved in process creation.
        """
        with self._lock:
            if tool not in TOOL_MODULES or tool not in TOOL_NAMES:
                raise RequestError(f"unknown tool: {tool}")
            self._check_active_capacity()
            config = validate_tool_payload(tool, payload, self.root)
            if tool == "analysis":
                manifest = read_json(Path(config["run_dir"]) / "run.json", {}) or {}
                if manifest.get("status") != "complete":
                    raise RequestError("analysis requires a completed training run")
            job_id = uuid.uuid4().hex
            if tool == "benchmark":
                # benchmark.py intentionally requires an empty output
                # directory.  Give each web submission an isolated child so
                # rerunning a benchmark never overwrites an earlier result.
                parent = Path(config["output_dir"]).resolve()
                parent.mkdir(parents=True, exist_ok=True)
                output = parent / f"web-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{job_id[:8]}"
                config["output_dir"] = str(output)
            config_path = self.config_root / f"{job_id}-{tool}.json"
            _atomic_json(config_path, config)

            # Analysis writes below its source run; other tools use their
            # configured artifact parent.  This path is informational for
            # artifact listing and is already constrained by validation.
            output_dir = config.get("run_dir") if tool == "analysis" else config.get("output_dir")
            record = {
                "job_id": job_id, "mode": "tool", "tool": tool,
                "status": "queued", "output_dir": output_dir,
                "config_path": str(config_path), "effective_config": config,
                "created_at": _now(), "stop_requested": False,
            }
            command = [sys.executable, "-u", "-m", TOOL_MODULES[tool],
                       "--config", str(config_path)]
            record["command"] = command
            self._save(record)
            try:
                return self._spawn(record, command, log_path=self.jobs_root / f"{job_id}.log")
            except OSError as error:
                record.update(status="failed", returncode=127, error=str(error), finished_at=_now())
                self._save(record)
                raise RequestError(f"unable to start {tool}: {error}") from error

    def get_job_artifact(self, job_id: str, artifact_path: str) -> Path:
        """Resolve an artifact below a tool job's configured output root."""
        if not isinstance(artifact_path, str) or not artifact_path or "\x00" in artifact_path:
            raise RequestError("invalid artifact path")
        record = self.get_job(job_id)
        output = record.get("output_dir")
        if not isinstance(output, str) or not output:
            raise RequestError("job has no artifact directory")
        root = Path(output).resolve()
        training_modes = TRAINING_MODES | {"analysis"}
        allowed_root = (self.training_root if record.get("mode") in training_modes
                        else (self.root / "artifacts").resolve())
        try:
            root.relative_to(allowed_root)
        except ValueError as error:
            raise RequestError("job artifact directory is outside the allowed artifact roots") from error
        path = (root / artifact_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RequestError("artifact path is outside the job output directory") from error
        if not path.is_file():
            raise RequestError("artifact not found")
        return path

    def _job_artifacts(self, record: dict[str, Any]) -> dict[str, Any]:
        """Expose a bounded set of report/replay links for tool jobs."""
        if record.get("mode") not in {"tool", "analysis"}:
            return {}
        output = record.get("output_dir")
        if not isinstance(output, str) or not output:
            return {}
        root = Path(output).resolve()
        if not root.is_dir():
            return {}
        training_modes = TRAINING_MODES | {"analysis"}
        allowed_root = (self.training_root if record.get("mode") in training_modes
                        else (self.root / "artifacts").resolve())
        try:
            root.relative_to(allowed_root)
        except ValueError:
            return {}
        tool = record.get("tool")
        search_root = root / "analysis" if tool == "analysis" else root
        if not search_root.is_dir():
            return {}
        try:
            created = datetime.fromisoformat(str(record.get("created_at", ""))).timestamp()
        except (TypeError, ValueError, OverflowError):
            created = 0.0
        names = {"report.html", "analysis.json", "comparison.html", "comparison.json",
                 "parity.json", "results.json", "report.md", "replay.json", "playlist.json"}
        files = []
        try:
            for path in search_root.rglob("*"):
                if not path.is_file() or path.name not in names:
                    continue
                try:
                    if path.stat().st_mtime + 1.0 < created:
                        continue
                    relative = path.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                files.append((relative, path.name))
        except OSError:
            return {}
        files.sort()
        files = files[:200]
        job_id = record.get("job_id", "")
        artifacts = [{"path": relative,
                      "url": f"/api/jobs/{job_id}/artifacts/{quote(relative, safe='/')}",
                      "label": relative} for relative, _ in files]
        result: dict[str, Any] = {"artifacts": artifacts}
        reports = [item for item in artifacts if item["path"].endswith(("report.html", "report.md"))]
        replays = [item for item in artifacts if item["path"].endswith(("replay.json", "playlist.json"))]
        results = [item for item in artifacts if item["path"].endswith(("results.json", "comparison.json", "analysis.json", "parity.json"))]
        if reports:
            result["report_url"], result["report_path"] = reports[0]["url"], reports[0]["path"]
        if replays:
            result["replay_url"] = replays[0]["url"]
        if results:
            result["results_url"] = results[0]["url"]
        return result

    def _job_view(self, record: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(record)
        result["status"] = self._status(record)
        result.update(self._job_artifacts(record))
        return result

    def stop(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise RequestError("unknown job")
            status = self._status(record)
            if status in TERMINAL:
                record["status"] = status
                self._save(record)
                return self._job_view(record)
            record["stop_requested"] = True
            record["stop_requested_at"] = _now()
            record["stop_deadline"] = time.time() + self.stop_timeout_seconds
            record["status"] = "stop_requested"
            self._save(record)
            if self._pid_alive(record):
                try:
                    os.killpg(int(record["pgid"]), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, ValueError) as error:
                    record["stop_error"] = str(error)
                    self._save(record)
            return self._job_view(record)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            self._refresh_record(self._jobs.get(job_id, {"job_id": job_id}))
            record = self._jobs.get(job_id)
            if record is None:
                raise RequestError("unknown job")
            return self._job_view(record)

    def get_log(self, job_id: str, limit: int = 12000) -> str:
        record = self.get_job(job_id)
        return tail_text(Path(record.get("log_path", "")), limit)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh()
            result = []
            for record in self._jobs.values():
                result.append(self._job_view(record))
            return sorted(result, key=lambda item: item.get("created_at", ""), reverse=True)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh()
            return list_runs(self.training_root)

    def get_run(self, run_id: str) -> dict[str, Any]:
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise RequestError("invalid run id")
        path = (self.training_root / run_id).resolve()
        try:
            path.relative_to(self.training_root)
        except ValueError as error:
            raise RequestError("run id is outside artifacts/training") from error
        if not path.is_dir():
            raise RequestError("unknown run")
        return run_summary(path)
