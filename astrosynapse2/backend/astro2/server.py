"""Local-only FastAPI control plane for training, evaluation, and play."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .arena import (
    MAX_PAIRS,
    RECOMMENDED_PAIRS,
    ArenaConfig,
    ArenaManager,
    ModelResolutionError,
)
from .config import RunConfig, preset_config
from .hardware import system_snapshot
from .play import PlayManager
from .stats import elo_delta
from .storage import Store
from .supervisor import InvalidTransition, Supervisor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ASTRO2_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset: str = "astro4_m4"
    name: str | None = Field(default=None, max_length=80)
    overrides: dict[str, Any] = Field(default_factory=dict)
    start: bool = False


class ConfigPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    changes: dict[str, Any]


class CreateGameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str | None = None
    human_starts: bool = True
    seed: int | None = None


class GameChoiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: int = Field(ge=0)


class ModelPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pinned: bool


class CreateArenaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_a: str = Field(min_length=1, max_length=128)
    model_b: str = Field(min_length=1, max_length=128)
    pairs: int = Field(default=RECOMMENDED_PAIRS, ge=1, le=MAX_PAIRS)
    seed: int = 20260807
    max_turns: int = Field(default=180, ge=20, le=500)
    max_actions_per_turn: int = Field(default=160, ge=20, le=500)
    confidence: float = Field(default=0.95, ge=0.80, le=0.999)
    minimum_promotion_pairs: int = Field(
        default=RECOMMENDED_PAIRS, ge=RECOMMENDED_PAIRS, le=MAX_PAIRS
    )


def _build_config(request: CreateRunRequest) -> RunConfig:
    base = preset_config(request.preset).model_dump()
    if request.name:
        base["name"] = request.name
    base.update(request.overrides)
    if request.preset not in {"astro4_m4", "astro3_m4", "m4_24h", "quick"}:
        base["preset"] = "custom"
    return RunConfig.model_validate(base)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(DATA_DIR / "astrosynapse2.sqlite3")
    app.state.store = store
    app.state.arena = ArenaManager(store)
    app.state.supervisor = Supervisor(
        store,
        PROJECT_ROOT,
        evaluation_manager=app.state.arena,
    )
    app.state.play = PlayManager()
    try:
        yield
    finally:
        app.state.supervisor.shutdown()
        app.state.play.shutdown()
        app.state.arena.shutdown()


app = FastAPI(
    title="Astrosynapse 2",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _supervisor(request: Request) -> Supervisor:
    return request.app.state.supervisor


def _store(request: Request) -> Store:
    return request.app.state.store


def _play(request: Request) -> PlayManager:
    return request.app.state.play


def _arena(request: Request) -> ArenaManager:
    return request.app.state.arena


def _artifact_retention(checkpoint: dict[str, Any]) -> dict[str, Any]:
    evaluation = checkpoint.get("evaluation") or {}
    retention = evaluation.get("artifact_retention") if isinstance(evaluation, dict) else None
    return retention if isinstance(retention, dict) else {}


def _actor_unavailable_detail(checkpoint: dict[str, Any]) -> str:
    retention = _artifact_retention(checkpoint)
    removed = retention.get("removed_artifacts", [])
    if bool(retention.get("pruned")) and "actor" in removed:
        return "model actor snapshot was pruned by checkpoint retention"
    return "model actor snapshot is unavailable"


def _tainted_checkpoint_ids(checkpoints: list[dict[str, Any]]) -> set[str]:
    """Return legacy random-restart checkpoints and every descendant."""

    tainted = {
        checkpoint["id"]
        for checkpoint in checkpoints
        if (checkpoint.get("evaluation") or {}).get("reason") == "initial random model"
        and int(checkpoint["games"]) > 0
    }
    changed = True
    while changed:
        changed = False
        for checkpoint in checkpoints:
            if checkpoint.get("parent_id") in tainted and checkpoint["id"] not in tainted:
                tainted.add(checkpoint["id"])
                changed = True
    return tainted


def _model_document(checkpoint: dict[str, Any]) -> dict[str, Any]:
    result = dict(checkpoint)
    model_path = Path(str(checkpoint.get("path") or ""))
    actor_path = Path(str(checkpoint.get("actor_path") or ""))
    model_available = model_path.is_file() and Path(f"{model_path}.json").is_file()
    actor_available = actor_path.is_file()
    retention = _artifact_retention(checkpoint)
    retention_pruned = bool(retention.get("pruned"))
    size_bytes = 0
    for key in ("path", "actor_path"):
        value = checkpoint.get(key)
        if value and Path(value).is_file():
            size_bytes += Path(value).stat().st_size
    result["size_bytes"] = size_bytes
    result["size_mb"] = size_bytes / (1024 * 1024)
    result["artifact_availability"] = {
        "model": model_available,
        "actor": actor_available,
    }
    result["actor_available"] = actor_available
    result["model_available"] = model_available
    result["playable"] = actor_available
    result["actor_downloadable"] = actor_available
    result["artifact_state"] = (
        "available"
        if model_available and actor_available
        else "pruned"
        if retention_pruned and not model_available and not actor_available
        else "partial"
        if model_available or actor_available
        else "missing"
    )
    evaluation = checkpoint.get("evaluation") or {}
    latest = evaluation.get("latest_arena") if isinstance(evaluation, dict) else None
    result["evaluated"] = bool(latest)
    if isinstance(latest, dict):
        score = float(latest.get("model_a_score", 0.5))
        interval = latest.get("paired_interval") or {}
        result.update(
            score=score,
            ci_low=float(interval.get("low", 0.0)),
            ci_high=float(interval.get("high", 1.0)),
            elo_delta=elo_delta(score),
        )
    return result


@app.get("/api/health")
def health(request: Request) -> dict[str, Any]:
    return {
        "ok": True,
        "service": "Astrosynapse 2",
        "active_run_id": _supervisor(request).active_run_id(),
    }


@app.get("/api/system")
def system() -> dict[str, Any]:
    return system_snapshot()


@app.get("/api/presets")
def presets() -> dict[str, Any]:
    return {
        "astro4_m4": RunConfig.astro4_m4().model_dump(),
        "astro3_m4": RunConfig.astro3_m4().model_dump(),
        "m4_24h": RunConfig().model_dump(),
        "quick": RunConfig.quick().model_dump(),
    }


@app.get("/api/runs")
def runs(request: Request) -> list[dict[str, Any]]:
    store = _store(request)
    result = []
    for run in store.list_runs():
        item = dict(run)
        checkpoints = store.checkpoints(run["id"])
        tainted_ids = _tainted_checkpoint_ids(checkpoints)
        latest = next(
            (
                model
                for model in checkpoints
                if model["id"] not in tainted_ids
            ),
            None,
        )
        champion = next((model for model in checkpoints if model["is_champion"]), None)
        champion_evaluated = bool(
            champion and (champion.get("evaluation") or {}).get("latest_arena")
        )
        item["latest_model"] = (
            {
                "id": latest["id"],
                "label": latest["label"],
                "games": latest["games"],
            }
            if latest
            else None
        )
        item["deployment_model"] = (
            {
                "id": champion["id"],
                "label": (
                    champion["label"]
                    if champion_evaluated
                    else re.sub(r"^Champion\b", "Anchor", champion["label"], flags=re.IGNORECASE)
                ),
                "games": champion["games"],
                "evaluated": champion_evaluated,
            }
            if champion
            else None
        )
        result.append(item)
    return result


@app.post("/api/runs", status_code=201)
def create_run(payload: CreateRunRequest, request: Request) -> dict[str, Any]:
    try:
        run = _supervisor(request).create_run(_build_config(payload))
        return _supervisor(request).start(run["id"]) if payload.start else run
    except (ValueError, InvalidTransition) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/runs/{run_id}")
def run_status(run_id: str, request: Request) -> dict[str, Any]:
    try:
        return _supervisor(request).live_snapshot(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error


@app.patch("/api/runs/{run_id}/config")
def patch_config(run_id: str, payload: ConfigPatch, request: Request) -> dict[str, Any]:
    try:
        return _supervisor(request).update_config(run_id, payload.changes)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/runs/{run_id}/{command}")
def command(run_id: str, command: str, request: Request) -> dict[str, Any]:
    if command not in {"start", "pause", "resume", "stop", "checkpoint"}:
        raise HTTPException(status_code=404, detail="unknown command")
    try:
        return getattr(_supervisor(request), command)(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    except InvalidTransition as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/runs/{run_id}/metrics")
def metrics(
    run_id: str,
    request: Request,
    after: int = Query(default=-1, ge=-1),
    limit: int = Query(default=2_000, ge=1, le=20_000),
) -> list[dict[str, Any]]:
    try:
        _store(request).get_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    return _store(request).metrics(run_id, after=after, limit=limit)


@app.get("/api/runs/{run_id}/events")
def audit_events(
    run_id: str,
    request: Request,
    limit: int = Query(default=100, ge=1, le=2_000),
) -> list[dict[str, Any]]:
    try:
        _store(request).get_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    return _store(request).events(run_id, limit=limit)


@app.get("/api/models")
def models(
    request: Request,
    run_id: str | None = None,
    include_tainted: bool = False,
) -> list[dict[str, Any]]:
    checkpoints = _store(request).checkpoints(run_id)
    tainted_ids = _tainted_checkpoint_ids(checkpoints)
    visible = (
        checkpoints
        if include_tainted
        else [item for item in checkpoints if item["id"] not in tainted_ids]
    )
    result = []
    for item in visible:
        document = _model_document(item)
        document["integrity_status"] = (
            "tainted_random_restart" if item["id"] in tainted_ids else "verified_lineage"
        )
        result.append(document)
    return result


@app.patch("/api/models/{model_id}")
def patch_model(model_id: str, payload: ModelPatch, request: Request) -> dict[str, Any]:
    try:
        return _store(request).set_checkpoint_pinned(model_id, payload.pinned)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="model not found") from error


@app.get("/api/models/{model_id}/actor")
def download_actor(model_id: str, request: Request) -> FileResponse:
    try:
        checkpoint = _store(request).checkpoint(model_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="model not found") from error
    actor_path = checkpoint.get("actor_path")
    if not actor_path or not Path(actor_path).is_file():
        raise HTTPException(status_code=409, detail=_actor_unavailable_detail(checkpoint))
    return FileResponse(
        actor_path,
        media_type="application/octet-stream",
        filename=f"astrosynapse2-{model_id}.actor.npz",
    )


@app.post("/api/arena", status_code=201)
def create_arena(payload: CreateArenaRequest, request: Request) -> dict[str, Any]:
    try:
        return _arena(request).create(
            payload.model_a,
            payload.model_b,
            ArenaConfig(
                pairs=payload.pairs,
                seed=payload.seed,
                max_turns=payload.max_turns,
                max_actions_per_turn=payload.max_actions_per_turn,
                confidence=payload.confidence,
                minimum_promotion_pairs=payload.minimum_promotion_pairs,
                automatic_promotion=False,
            ),
        )
    except ModelResolutionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/arena")
def arena_jobs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    if run_id is None:
        return _arena(request).list(limit=limit)
    try:
        checkpoint_ids = {item["id"] for item in _store(request).checkpoints(run_id)}
        _store(request).get_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="run not found") from error
    jobs = _arena(request).list(limit=500)
    return [
        job for job in jobs if job["model_a"] in checkpoint_ids or job["model_b"] in checkpoint_ids
    ][:limit]


@app.get("/api/arena/{job_id}")
def arena_job(job_id: str, request: Request) -> dict[str, Any]:
    try:
        return _arena(request).get(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="arena job not found") from error


@app.get("/api/games")
def games(request: Request) -> list[dict[str, Any]]:
    return _play(request).list()


@app.post("/api/games", status_code=201)
def create_game(payload: CreateGameRequest, request: Request) -> dict[str, Any]:
    actor_path = None
    label = "Balanced baseline"
    if payload.model_id:
        try:
            checkpoint = _store(request).checkpoint(payload.model_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="model not found") from error
        actor_path = checkpoint.get("actor_path")
        label = checkpoint["label"]
        if not actor_path or not Path(actor_path).exists():
            raise HTTPException(status_code=409, detail=_actor_unavailable_detail(checkpoint))
    try:
        return _play(request).create(
            seed=payload.seed,
            human_starts=payload.human_starts,
            actor_path=actor_path,
            model_label=label,
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/games/{session_id}")
def game(session_id: str, request: Request) -> dict[str, Any]:
    try:
        return _play(request).get(session_id).snapshot()
    except KeyError as error:
        raise HTTPException(status_code=404, detail="game not found") from error


@app.post("/api/games/{session_id}/choice")
def game_choice(session_id: str, payload: GameChoiceRequest, request: Request) -> dict[str, Any]:
    try:
        return _play(request).get(session_id).choose(payload.action_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="game not found") from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/events")
async def events(
    request: Request,
    run_id: str | None = None,
    after: int = Query(default=-1, ge=-1),
) -> StreamingResponse:
    async def stream() -> AsyncIterator[str]:
        cursor = after
        while True:
            if await request.is_disconnected():
                return
            active = run_id or _supervisor(request).active_run_id()
            batch = _store(request).metrics(active, after=cursor, limit=100) if active else []
            if batch:
                for metric in batch:
                    cursor = int(metric["seq"])
                    yield f"id: {cursor}\nevent: metric\ndata: {json.dumps(metric)}\n\n"
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(stream(), media_type="text/event-stream")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "astro2.server:app",
        host=os.environ.get("ASTRO2_HOST", "127.0.0.1"),
        port=int(os.environ.get("ASTRO2_PORT", "8765")),
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
