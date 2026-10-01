"""Local API for independent critic training jobs."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .arena import ModelResolutionError
from .critic import CriticConfig

router = APIRouter(prefix="/api/critics", tags=["critics"])


class CreateCriticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=128)
    games: int = Field(default=1000, ge=20, le=10000)
    epochs: int = Field(default=20, ge=1, le=200)
    hidden_size: int = 128
    seed: int = Field(default=20261001, ge=0, lt=2**53)
    rules_version: int = Field(default=2, ge=1, le=2)
    positions_per_game: int = Field(default=64, ge=16, le=256)
    learning_rate: float = Field(default=0.001, gt=0, le=0.01, allow_inf_nan=False)


@router.get("")
def list_critics(request: Request):
    return request.app.state.critics.list()


@router.post("", status_code=201)
def create_critic(payload: CreateCriticRequest, request: Request):
    try:
        return request.app.state.critics.create(
            payload.model_id, CriticConfig(**payload.model_dump(exclude={"model_id"}))
        )
    except ModelResolutionError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/{job_id}")
def get_critic(job_id: str, request: Request):
    try:
        return request.app.state.critics.get(job_id)
    except KeyError as error:
        raise HTTPException(404, "Critic job not found") from error


@router.post("/{job_id}/{action}")
def control_critic(job_id: str, action: str, request: Request):
    try:
        return request.app.state.critics.control(job_id, action)
    except KeyError as error:
        raise HTTPException(404, "Critic job not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@router.get("/{job_id}/download/{kind}")
def download_critic(job_id: str, kind: str, request: Request):
    try:
        folder = request.app.state.critics.folder(job_id)
        filename = {
            "critic": "critic.npz",
            "report": "result.json",
            "log": "worker.log",
            "splits": "splits.json",
        }.get(kind)
        if not filename or not (folder / filename).is_file():
            raise KeyError(kind)
        return FileResponse(folder / filename, filename=f"critic-{job_id[:8]}-{filename}")
    except KeyError as error:
        raise HTTPException(404, "Artifact not available") from error
