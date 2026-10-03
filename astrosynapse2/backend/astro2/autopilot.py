"""Durable autonomous campaigns; workers survive browser and API restarts."""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .autopilot_registry import (
    ValuePredictor,
    critic_inventory,
    read,
    resolve_critic,
)
from .experiment_control import atomic_json, code_identity, sha256


class CampaignConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Arch3 champion autopilot", min_length=1, max_length=80)
    policy_id: str
    critic_id: str
    opponent_ids: list[str] = Field(min_length=1, max_length=16)
    panel_ids: list[str] = Field(min_length=1, max_length=8)
    initial_candidate_ids: list[str] = Field(default_factory=list, max_length=8)
    policy_fraction: float = Field(default=0.8, ge=0.5, le=0.95)
    block_games: int = Field(default=20000, ge=20, le=100000)
    batch_games: int = Field(default=1000, ge=10, le=2000)
    seeds: list[int] = Field(
        default_factory=lambda: [2026100411, 2026100422], min_length=1, max_length=4
    )
    workers: int = Field(default=6, ge=1, le=8)
    learning_rate: float = Field(default=0.000002, gt=0, le=0.00002)
    screen_pairs: int = Field(default=512, ge=8, le=2000)
    gate_pairs: int = Field(default=12000, ge=2000, le=100000)
    panel_pairs: int = Field(default=4000, ge=1000, le=20000)
    panel_regression: float = Field(default=0.02, ge=0, le=0.05)
    probe_games: int = Field(default=2000, ge=20, le=10000)
    probe_pairs: int = Field(default=4000, ge=1000, le=20000)
    max_hours: float = Field(default=24, gt=0, le=720)
    storage_gb: float = Field(default=30, ge=1, le=500)
    max_blocks: int = Field(default=0, ge=0, le=10000)
    seed: int = Field(default=2026100401, ge=0, lt=2**53)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.block_games % self.batch_games or self.probe_games % self.batch_games:
            raise ValueError("Block and probe games must be multiples of batch games")
        if len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("Training seeds must be distinct nonnegative integers")
        return self


def alive(pid):
    if not pid:
        return False
    try:
        import psutil

        return (
            psutil.Process(pid).is_running()
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        )
    except (OSError, psutil.Error):
        return False


def worker_locked(folder):
    with (folder / "worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True


class AutopilotManager:
    def __init__(self, store, project):
        self.store, self.project = store, Path(project)
        self.root = store.path.parent / "autopilot"
        self.root.mkdir(exist_ok=True)
        self.children = []

    def folder(self, ident):
        if not ident.isalnum() or len(ident) != 32:
            raise KeyError(ident)
        folder = self.root / ident
        if not (folder / "state.json").is_file():
            raise KeyError(ident)
        return folder

    def models(self):
        from .progressive_models import progressive_models

        return self.store.checkpoints() + progressive_models(self.store.path.parent)

    def critics(self):
        return critic_inventory(self.store.path.parent, self.models())

    def get(self, ident):
        folder = self.folder(ident)
        state = read(folder / "state.json")
        launch = read(folder / "launch.json", {})
        running = worker_locked(folder)
        if (
            state["status"] in ("running", "queued")
            and launch.get("pid")
            and not running
            and not alive(launch.get("pid"))
        ):
            state["status"] = "interrupted"
        state["running"] = (
            running
            or alive(state.get("child_pid"))
            or (alive(launch.get("pid")) and state["status"] in ("queued", "running"))
        )
        state["pause_requested"] = (folder / "PAUSE").exists()
        state["stop_after_block"] = (folder / "DRAIN").exists()
        state["config"] = read(folder / "config.json")
        state["jobs"] = read(folder / "jobs.json", {})
        state["log_url"] = f"/api/autopilot/{ident}/log"
        # Training progress without having the API process own the learner.
        active = state.get("active_job")
        if active:
            job_folder = folder / "tasks" / active
            state["progress"] = read(job_folder / "state.json") or read(
                job_folder / "progress.json"
            )
        return state

    def list(self):
        runs = [self.get(p.parent.name) for p in self.root.glob("*/state.json")]
        return sorted(
            (run for run in runs if not run.get("diagnostic")),
            key=lambda run: run.get("created_at", 0),
            reverse=True,
        )

    def _snapshot(self, folder):
        shutil.copytree(
            self.project / "backend/astro2",
            folder / "runtime/astro2",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copytree(
            self.project / "scripts",
            folder / "runtime/scripts",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        atomic_json(
            folder / "code.json",
            code_identity(folder / "runtime/astro2", folder / "runtime/scripts"),
        )

    def create(self, config: CampaignConfig):
        from .arch3 import port_actor
        from .arena import resolve_model

        inventory = {m["id"]: m for m in self.models()}
        source = inventory.get(config.policy_id)
        if not source or not Path(source["path"]).is_file():
            raise ValueError("Select a trainable checkpoint")
        critic = resolve_critic(self.store.path.parent, config.critic_id, self.models())
        if critic["kind"] != "independent":
            raise ValueError("Autopilot training requires an independent critic")
        predictor = ValuePredictor(critic)
        if predictor.encoder.version != 3:
            raise ValueError("Autopilot requires an arch3 independent critic")
        refs = config.opponent_ids + config.panel_ids + config.initial_candidate_ids
        resolved = {ref: resolve_model(self.store, ref) for ref in refs}
        if any(r.actor_path is None for r in resolved.values()):
            raise ValueError("Select checkpoint opponents")
        ident = uuid.uuid4().hex
        folder = self.root / ident
        folder.mkdir()
        inputs = folder / "inputs"
        inputs.mkdir()
        lineage = port_actor(source["actor_path"], inputs / "policy")
        shutil.copyfile(critic["path"], inputs / "critic.npz")
        copied = {}
        for i, (ref, model) in enumerate(resolved.items()):
            destination = inputs / f"opponent-{i}.actor.npz"
            shutil.copyfile(model.actor_path, destination)
            copied[ref] = str(destination)
        champion_id = f"auto-{ident}-p0"
        critic_id = f"auto-{ident}-c0"
        model = dict(
            id=champion_id,
            label=f"{config.name} · arch3 · initial",
            path=lineage["model"],
            actor_path=lineage["actor"],
            encoder_version=3,
            architecture="arch3",
            games=source.get("games", 0),
            parent_id=source["id"],
            created_at=time.time(),
            checkpoint_name="initial",
            generation=0,
            was_champion=True,
            critic_id=critic_id,
        )
        initial = []
        for i, ref in enumerate(config.initial_candidate_ids):
            port = port_actor(resolved[ref].actor_path, inputs / f"candidate-{i}")
            initial.append(dict(model=port["model"], actor=port["actor"], source_id=ref))
        self._snapshot(folder)
        atomic_json(
            folder / "inputs.json",
            dict(
                opponents=[copied[r] for r in config.opponent_ids],
                panel=[copied[r] for r in config.panel_ids],
                candidates=initial,
                hashes={
                    str(p.relative_to(inputs)): sha256(p) for p in inputs.rglob("*") if p.is_file()
                },
            ),
        )
        atomic_json(folder / "config.json", config.model_dump())
        atomic_json(
            folder / "state.json",
            dict(
                id=ident,
                name=config.name,
                status="queued",
                phase="ready",
                created_at=time.time(),
                champion_id=champion_id,
                critic_id=critic_id,
                models=[model],
                critics=[
                    dict(
                        id=critic_id,
                        kind="independent",
                        path=str(inputs / "critic.npz"),
                        label=f"{config.name} · arch3 critic 0",
                        encoder_version=3,
                        created_at=time.time(),
                    )
                ],
                promotions=[],
                events=[],
                blocks=0,
                initial_index=0,
                attempt=0,
                critic_attempt=0,
                tips={},
                pending=None,
                active_job=None,
            ),
        )
        return self.get(ident)

    def launch(self, ident):
        folder = self.folder(ident)
        with (self.root / "manager.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if any(s["running"] for s in self.list()):
                raise ValueError("Pause the active Autopilot campaign first")
            state = read(folder / "state.json")
            if state["status"] == "complete":
                raise ValueError("Campaign budget is complete; create a continuation campaign")
            state["status"] = "queued"
            atomic_json(folder / "state.json", state)
            (folder / "PAUSE").unlink(missing_ok=True)
            (folder / "DRAIN").unlink(missing_ok=True)
            env = {
                **os.environ,
                "PYTHONPATH": str(folder / "runtime"),
                "VECLIB_MAXIMUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
            command = [
                sys.executable,
                str(folder / "runtime/scripts/autopilot_worker.py"),
                "--folder",
                str(folder),
            ]
            with (folder / "worker.log").open("ab") as log:
                child = subprocess.Popen(
                    command,
                    cwd=self.project,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            self.children.append(child)
            if sys.platform == "darwin" and shutil.which("caffeinate"):
                self.children.append(
                    subprocess.Popen(
                        ["caffeinate", "-i", "-w", str(child.pid)],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                )
            atomic_json(
                folder / "launch.json", dict(pid=child.pid, command=command, at=time.time())
            )
        return self.get(ident)

    def control(self, ident, action):
        folder = self.folder(ident)
        if action in ("start", "resume"):
            return self.launch(ident)
        if action not in ("pause", "drain"):
            raise ValueError("Unknown action")
        (folder / ("PAUSE" if action == "pause" else "DRAIN")).touch()
        if action == "pause":
            active = read(folder / "state.json", {}).get("active_job")
            if active and (folder / "tasks" / active).is_dir():
                (folder / "tasks" / active / "STOP").touch()
        return self.get(ident)
