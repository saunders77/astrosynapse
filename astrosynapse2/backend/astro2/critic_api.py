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
    games: int = Field(default=1000, ge=20, le=100000)
    epochs: int = Field(default=20, ge=1, le=2000)
    encoder_version: int | None = Field(default=None, ge=1, le=3)
    workers: int = Field(default=1, ge=1, le=8)
    batch_games: int = Field(default=1, ge=1, le=64)
    patience: int = Field(default=20, ge=1, le=2000)
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


class TrainPolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    opponent_ids: list[str] = Field(default_factory=list, max_length=32)
    iterations: int = Field(default=80, ge=1, le=10000)
    games: int = Field(default=256, ge=2, le=4096)
    workers: int = Field(default=6, ge=1, le=8)
    learning_rate: float = Field(default=0.000002, gt=0, le=0.0002)


@router.post("/{job_id}/policy/train", status_code=201)
def train_policy(job_id: str, payload: TrainPolicyRequest, request: Request):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from .arch3 import create_port
    from .arena import resolve_model
    from .experiment_control import atomic_json

    manager = request.app.state.critics
    try:
        job = manager.get(job_id)
        if job["status"] != "complete":
            raise ValueError("Finish critic training before starting policy training")
        from .critic import IndependentCritic

        _, meta, _ = IndependentCritic.load(manager.folder(job_id) / "critic.npz")
        if meta["encoder_version"] != 3:
            raise ValueError("Choose a completed arch3 critic")
        if any(
            run.get("status") == "running"
            for j in manager.list()
            for run in j.get("policy_runs", [])
        ):
            raise ValueError("Pause the active policy run before starting another")
        store = manager.store
        opponents = [resolve_model(store, i).actor_path for i in payload.opponent_ids]
        if any(p is None for p in opponents):
            raise ValueError("Choose checkpoint opponents")
        model = create_port(store, payload.model_id)
        folder = Path(model["path"]).parent / "policy"
        folder.mkdir()
        shutil.copyfile(manager.folder(job_id) / "critic.npz", folder / "independent-critic.npz")
        atomic_json(folder / "opponents.json", opponents)
        project = Path(__file__).resolve().parents[2]
        command = [
            sys.executable,
            str(project / "scripts/onpolicy_experiment.py"),
            "--model",
            model["path"],
            "--opponent",
            str(manager.folder(job_id) / "policy.actor.npz"),
            "--opponent-pool",
            str(folder / "opponents.json"),
            "--independent-critic",
            str(folder / "independent-critic.npz"),
            "--output",
            str(folder),
            "--iterations",
            str(payload.iterations),
            "--games",
            str(payload.games),
            "--workers",
            str(payload.workers),
            "--learning-rate",
            str(payload.learning_rate),
            "--rules-version",
            str(meta["config"]["rules_version"]),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": str(project / "backend"),
            "VECLIB_MAXIMUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        with (folder / "worker.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=project,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        manager.children.append(process)
        run = dict(
            model_id=model["id"],
            folder=str(folder),
            pid=process.pid,
            command=command,
            critic_id=job_id,
        )
        atomic_json(folder / "launch.json", run)
        job.setdefault("policy_runs", []).append(run)
        atomic_json(manager.folder(job_id) / "job.json", job)
        return run
    except KeyError as error:
        raise HTTPException(404, "Model or critic not found") from error
    except (ValueError, ModelResolutionError) as error:
        raise HTTPException(422, str(error)) from error


@router.post("/{job_id}/policy/{model_id}/{action}")
def control_policy(job_id: str, model_id: str, action: str, request: Request):
    import os
    import subprocess
    from pathlib import Path

    from .experiment_control import atomic_json

    manager = request.app.state.critics
    try:
        job = manager.get(job_id)
        run = next((r for r in job.get("policy_runs", []) if r["model_id"] == model_id), None)
        if run is None:
            raise KeyError(model_id)
        folder = Path(run["folder"])
        if action == "pause":
            if run["status"] != "running":
                raise ValueError("Only running policies can be paused")
            (folder / "STOP").touch()
        elif action == "resume":
            if run["status"] not in {"paused", "interrupted"}:
                raise ValueError("Only paused or interrupted policies can resume")
            if any(
                r.get("status") == "running"
                for j in manager.list()
                for r in j.get("policy_runs", [])
            ):
                raise ValueError("Another policy run is active")
            if not (folder / "state.json").exists():
                raise ValueError("No checkpoint exists yet; inspect the log and start a new run")
            project = Path(__file__).resolve().parents[2]
            (folder / "STOP").unlink(missing_ok=True)
            (folder / "complete.json").unlink(missing_ok=True)
            env = {
                **os.environ,
                "PYTHONPATH": str(project / "backend"),
                "VECLIB_MAXIMUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            with (folder / "worker.log").open("ab") as log:
                process = subprocess.Popen(
                    [*run["command"], "--resume"],
                    cwd=project,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            manager.children.append(process)
            run["pid"] = process.pid
            atomic_json(folder / "launch.json", run)
            atomic_json(manager.folder(job_id) / "job.json", job)
        else:
            raise ValueError("Unknown policy action")
        return manager.get(job_id)
    except KeyError as error:
        raise HTTPException(404, "Policy run not found") from error
    except ValueError as error:
        raise HTTPException(409, str(error)) from error


@router.get("/{job_id}/policy/{model_id}/log")
def policy_log(job_id: str, model_id: str, request: Request):
    from pathlib import Path

    try:
        job = request.app.state.critics.get(job_id)
        run = next((r for r in job.get("policy_runs", []) if r["model_id"] == model_id), None)
        if run is None:
            raise KeyError(model_id)
        return FileResponse(Path(run["folder"]) / "worker.log", media_type="text/plain")
    except KeyError as error:
        raise HTTPException(404, "Policy run not found") from error
