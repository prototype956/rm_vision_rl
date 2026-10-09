"""Small, tolerant readers for live training artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    """Read an atomically replaced JSON file, tolerating an in-progress write."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return default


def read_csv_tail(path: Path, limit: int = 200) -> list[dict[str, str]]:
    """Read complete CSV rows and ignore a concurrently written final row."""
    if not path.is_file():
        return []
    try:
        lines = path.read_text().splitlines()
    except (OSError, UnicodeError):
        return []
    if not lines:
        return []
    rows: list[dict[str, str]] = []
    try:
        for row in csv.DictReader(line for line in lines if not line.startswith("#")):
            if row and all(value is not None for value in row.values()):
                rows.append(dict(row))
    except csv.Error:
        return rows[-limit:]
    return rows[-limit:]


def tail_text(path: Path, limit: int = 12000) -> str:
    """Return a bounded log tail without failing a status request."""
    try:
        with path.open("rb") as file:
            file.seek(0, 2)
            size = file.tell()
            file.seek(max(0, size - limit))
            return file.read(limit).decode("utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return ""


def _number(value: Any) -> float | int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number == number or number in (float("inf"), float("-inf")):
        return None
    return int(number) if number.is_integer() else number


def run_summary(run_dir: Path) -> dict[str, Any]:
    """Build a JSON-safe summary for the dashboard list/detail views."""
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "run.json", {}) or {}
    progress = read_csv_tail(run_dir / "logs" / "progress.csv", 5)
    latest = progress[-1] if progress else {}
    summary = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": manifest.get("status", "unknown"),
        "requested_timesteps": manifest.get("requested_timesteps"),
        "rounded_timesteps": manifest.get("rounded_timesteps"),
        "start_timesteps": manifest.get("start_timesteps"),
        "last_completed_timesteps": manifest.get("last_completed_timesteps"),
        "completed_updates": manifest.get("completed_updates"),
        "resume_from": manifest.get("resume_from"),
        "actual_device": manifest.get("actual_device"),
        "config": manifest.get("config", {}),
        "metadata": manifest.get("metadata", {}),
        "latest_metrics": {
            key: _number(value) for key, value in latest.items()
            if key and value not in (None, "")
        },
        "checkpoints": sorted(path.name for path in run_dir.glob("*.zip") if path.is_file()),
        "analysis": sorted(str(path.relative_to(run_dir)) for path in run_dir.glob("analysis/*/analysis.json")),
    }
    return summary


def list_runs(training_root: Path) -> list[dict[str, Any]]:
    """List run directories without assuming every directory is complete."""
    if not training_root.is_dir():
        return []
    runs = [run_summary(path) for path in training_root.iterdir() if path.is_dir()
            and path.name != ".web-jobs" and (path / "run.json").is_file()]
    return sorted(runs, key=lambda item: item["run_id"], reverse=True)
