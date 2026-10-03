"""GUI control plane for autonomous campaigns and mixed-architecture matches."""

import os
import subprocess
import sys
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .arena import resolve_model
from .autopilot import CampaignConfig, alive
from .autopilot_registry import ValuePredictor, read, resolve_critic
from .experiment_control import atomic_json

router = APIRouter(prefix="/api/autopilot", tags=["autopilot"])


def manager(request):
    return request.app.state.autopilot


@router.get("")
def campaigns(request: Request):
    return manager(request).list()


@router.get("/critics")
def critics(request: Request):
    return manager(request).critics()


@router.post("", status_code=201)
def create(payload: CampaignConfig, request: Request):
    try:
        return manager(request).create(payload)
    except (ValueError, KeyError) as error:
        raise HTTPException(422, str(error)) from error


class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_a: str
    model_b: str
    critic_a: str | None = None
    critic_b: str | None = None
    pairs: int = Field(default=2000, ge=1, le=20000)
    workers: int = Field(default=2, ge=1, le=6)
    seed: int = Field(default=20261004, ge=0, lt=2**53)
    mode: str = Field(default="arena", pattern="^(arena|selfplay)$")


@router.get("/matches")
def matches(request: Request):
    root = manager(request).root / "matches"
    result = []
    for path in root.glob("*/job.json"):
        job = read(path, {})
        job["pid"] = read(path.parent / "launch.json", {}).get("pid")
        if job.get("status") in ("running", "queued") and not alive(job.get("pid")):
            job["status"] = "interrupted"
        result.append(job)
    return sorted(result, key=lambda j: j["created_at"], reverse=True)


@router.post("/matches", status_code=201)
def create_match(payload: MatchRequest, request: Request):
    m = manager(request)
    try:
        if payload.mode == "selfplay" and payload.model_a != payload.model_b:
            raise ValueError("Self-play requires the same policy in both seats")
        a, b = resolve_model(m.store, payload.model_a), resolve_model(m.store, payload.model_b)
        if not a.actor_path or not b.actor_path:
            raise ValueError("Choose checkpoint policies")
        ca = resolve_critic(m.store.path.parent, payload.critic_a, m.models())
        cb = resolve_critic(m.store.path.parent, payload.critic_b, m.models())
        for c in (ca, cb):
            if c:
                ValuePredictor(c)
        if any(j["status"] in ("running", "queued") for j in matches(request)):
            raise ValueError("Pause or finish the active policy/critic match first")
        ident = uuid.uuid4().hex
        folder = m.root / "matches" / ident
        folder.mkdir(parents=True)
        job = dict(
            id=ident,
            **payload.model_dump(exclude={"critic_a", "critic_b"}),
            critic_a=ca,
            critic_b=cb,
            actor_a=a.actor_path,
            actor_b=b.actor_path,
            label_a=a.label,
            label_b=b.label,
            status="queued",
            created_at=time.time(),
            pairs_completed=0,
        )
        atomic_json(folder / "job.json", job)
        launch_match(m, folder, job)
        return job
    except (ValueError, KeyError) as error:
        raise HTTPException(422, str(error)) from error


def launch_match(m, folder, job):
    env = {
        **os.environ,
        "PYTHONPATH": str(m.project / "backend"),
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    with (folder / "worker.log").open("ab") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                str(m.project / "scripts/autopilot_match.py"),
                "--folder",
                str(folder),
            ],
            cwd=m.project,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    m.children.append(child)
    job["pid"] = child.pid
    atomic_json(folder / "launch.json", {"pid": child.pid})


@router.post("/matches/{ident}/{action}")
def control_match(ident: str, action: str, request: Request):
    m = manager(request)
    if not ident.isalnum() or len(ident) != 32:
        raise HTTPException(404, "Match not found")
    folder = m.root / "matches" / ident
    job = read(folder / "job.json")
    if not job:
        raise HTTPException(404, "Match not found")
    job["pid"] = read(folder / "launch.json", {}).get("pid")
    if action == "pause":
        (folder / "STOP").touch()
    elif action == "resume" and not alive(job.get("pid")) and job["status"] != "complete":
        (folder / "STOP").unlink(missing_ok=True)
        launch_match(m, folder, job)
    else:
        raise HTTPException(409, "Match cannot perform that action")
    return job


@router.get("/{ident}")
def get(ident: str, request: Request):
    try:
        return manager(request).get(ident)
    except KeyError as error:
        raise HTTPException(404, "Campaign not found") from error


@router.get("/{ident}/log")
def log(ident: str, request: Request):
    try:
        path = manager(request).folder(ident) / "worker.log"
        if not path.is_file():
            raise KeyError(ident)
        return FileResponse(path, media_type="text/plain")
    except KeyError as error:
        raise HTTPException(404, "Log not available") from error


@router.post("/{ident}/{action}")
def control(ident: str, action: str, request: Request):
    try:
        return manager(request).control(ident, action)
    except KeyError as error:
        raise HTTPException(404, "Campaign not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


class LimitsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_hours: float = Field(gt=0, le=720)
    storage_gb: float = Field(ge=1, le=500)
    workers: int = Field(ge=1, le=8)
    policy_fraction: float = Field(ge=0.5, le=0.95)


@router.patch("/{ident}/settings")
def settings(ident: str, payload: LimitsRequest, request: Request):
    m = manager(request)
    try:
        state = m.get(ident)
        if state["running"]:
            raise ValueError("Pause the campaign before changing resource limits")
        folder = m.folder(ident)
        config = CampaignConfig(**{**state["config"], **payload.model_dump()})
        atomic_json(folder / "config.json", config.model_dump())
        return m.get(ident)
    except KeyError as error:
        raise HTTPException(404, "Campaign not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy_id: str
    critic_id: str


@router.post("/{ident}/champions/restore")
def restore(ident: str, payload: RollbackRequest, request: Request):
    m = manager(request)
    try:
        state = m.get(ident)
        if state["running"] or state.get("pending"):
            raise ValueError("Use Stop after block, then restore champions at the block boundary")
        if not any(p["id"] == payload.policy_id and p.get("was_champion") for p in state["models"]):
            raise ValueError("Select a retained policy champion from this campaign")
        if not any(c["id"] == payload.critic_id for c in state["critics"]):
            raise ValueError("Select a retained accepted critic from this campaign")
        folder = m.folder(ident)
        saved = read(folder / "state.json")
        saved.update(champion_id=payload.policy_id, critic_id=payload.critic_id, tips={})
        saved["events"].append(
            dict(at=time.time(), kind="champions_restored", **payload.model_dump())
        )
        atomic_json(folder / "state.json", saved)
        return m.get(ident)
    except KeyError as error:
        raise HTTPException(404, "Campaign not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error
