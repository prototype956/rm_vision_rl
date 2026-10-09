"""FastAPI application for the local RM Vision RL training console."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
except ImportError as error:  # pragma: no cover - exercised when optional deps are absent
    raise RuntimeError(
        "Web dependencies are missing; install requirements-web.txt before starting the dashboard"
    ) from error

from src.training.config import ROOT
from tools.training.dashboard.manager import TrainingManager
from tools.training.dashboard.readers import read_json
from tools.training.dashboard.schemas import RequestError, load_preset, training_config_paths


STATIC_ROOT = Path(__file__).with_name("static")


def _config(root: Path) -> dict[str, Any]:
    path = root / "config" / "tools" / "dashboard.json"
    value = read_json(path, {}) or {}
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid dashboard configuration: {path}")
    return value


def _error(error: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


def create_app(root: Path = ROOT) -> FastAPI:
    """Create an application with a manager rooted at the repository."""
    root = Path(root).resolve()
    settings = _config(root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = TrainingManager(
            root,
            poll_seconds=float(settings.get("poll_seconds", 2.0)),
            stop_timeout_seconds=float(settings.get("stop_timeout_seconds", 40.0)),
            max_training_jobs=int(settings.get("max_training_jobs", 1)),
            max_active_jobs=int(settings.get("max_active_jobs",
                                           settings.get("max_training_jobs", 1))),
        )
        app.state.manager = manager
        try:
            yield
        finally:
            manager.close()

    app = FastAPI(title="RM Vision RL Dashboard", lifespan=lifespan)

    def get_manager(request: Request) -> TrainingManager:
        return request.app.state.manager

    @app.get("/", response_class=FileResponse)
    async def index():
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/static/{asset:path}")
    async def static_asset(asset: str):
        path = (STATIC_ROOT / asset).resolve()
        try:
            path.relative_to(STATIC_ROOT.resolve())
        except ValueError as error:
            raise HTTPException(status_code=404, detail="asset not found") from error
        if not path.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(path)

    @app.get("/api/configs/training")
    async def training_configs():
        result = []
        for path in training_config_paths(root):
            try:
                name, config = load_preset(path.name, root)
                result.append({
                    "name": name,
                    "policy_kind": config["policy_kind"],
                    "task": config["environment"].get("task", "static_fire"),
                    "action_mode": config["environment"].get("action_mode", "fire_only"),
                    "n_envs": config["n_envs"],
                    "n_steps": config["ppo"]["n_steps"],
                    "batch_size": config["ppo"]["batch_size"],
                    "total_timesteps": config["total_timesteps"],
                })
            except RequestError as error:
                result.append({"name": path.name, "error": str(error)})
        return {"presets": result}

    @app.get("/api/tools")
    async def tools(request: Request):
        """Describe the standalone tools available in the web console."""
        return {"tools": get_manager(request).tool_metadata()}

    @app.get("/api/tools/{tool}")
    async def tool_detail(tool: str, request: Request):
        items = [item for item in get_manager(request).tool_metadata()
                 if item["name"] == tool]
        if not items:
            raise HTTPException(status_code=404, detail="unknown tool")
        return items[0]

    @app.get("/api/runs")
    async def runs(request: Request):
        manager = get_manager(request)
        return {"runs": manager.list_runs(), "jobs": manager.list_jobs()}

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str, request: Request):
        try:
            return get_manager(request).get_run(run_id)
        except RequestError as error:
            raise _error(error)

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str, request: Request):
        try:
            return get_manager(request).get_job(job_id)
        except RequestError as error:
            raise _error(error)

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    async def job_log(job_id: str, request: Request):
        try:
            return get_manager(request).get_log(job_id)
        except RequestError as error:
            raise _error(error)

    async def json_body(request: Request) -> dict[str, Any]:
        try:
            value = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise _error(RequestError(f"invalid JSON body: {error}"))
        if not isinstance(value, dict):
            raise _error(RequestError("request body must be a JSON object"))
        return value

    @app.post("/api/jobs/train")
    async def train(request: Request):
        try:
            return get_manager(request).start_new(await json_body(request))
        except RequestError as error:
            raise _error(error)

    @app.post("/api/jobs/resume")
    async def resume(request: Request):
        try:
            return get_manager(request).start_resume(await json_body(request))
        except RequestError as error:
            raise _error(error)

    @app.post("/api/jobs/{job_id}/stop")
    async def stop(job_id: str, request: Request):
        try:
            return get_manager(request).stop(job_id)
        except RequestError as error:
            raise _error(error)

    @app.get("/api/jobs/{job_id}/artifacts/{artifact_path:path}")
    async def job_artifact(job_id: str, artifact_path: str, request: Request):
        try:
            path = get_manager(request).get_job_artifact(job_id, artifact_path)
        except RequestError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return FileResponse(path)

    @app.post("/api/tools/{tool}/run")
    async def run_tool(tool: str, request: Request):
        try:
            return get_manager(request).start_tool(tool, await json_body(request))
        except RequestError as error:
            raise _error(error)

    # Keep the shorter route as a convenience for simple forms and older
    # clients; both routes use exactly the same allowlist and manager.
    @app.post("/api/tools/{tool}")
    async def run_tool_alias(tool: str, request: Request):
        try:
            return get_manager(request).start_tool(tool, await json_body(request))
        except RequestError as error:
            raise _error(error)

    @app.post("/api/runs/{run_id}/analyze")
    async def analyze(run_id: str, request: Request):
        try:
            run = get_manager(request).get_run(run_id)
            return get_manager(request).start_analysis(run["run_dir"])
        except RequestError as error:
            raise _error(error)

    @app.get("/api/artifacts/{run_id}/{artifact_path:path}")
    async def artifact(run_id: str, artifact_path: str, request: Request):
        try:
            run = get_manager(request).get_run(run_id)
            run_dir = Path(run["run_dir"]).resolve()
            path = (run_dir / artifact_path).resolve()
            path.relative_to(run_dir)
        except (RequestError, ValueError) as error:
            raise HTTPException(status_code=404, detail="artifact not found") from error
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(path)

    @app.exception_handler(RequestError)
    async def request_error_handler(_: Request, error: RequestError):
        return JSONResponse(status_code=400, content={"detail": str(error)})

    return app


app = create_app()
