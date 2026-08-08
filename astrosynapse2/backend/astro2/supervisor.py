"""Long-lived training process control, independent of browser connections."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import SAFE_LIVE_FIELDS, RunConfig
from .storage import Store


class InvalidTransition(RuntimeError):
    pass


@dataclass(slots=True)
class RunControl:
    pause_requested: threading.Event = field(default_factory=threading.Event)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    checkpoint_requested: threading.Event = field(default_factory=threading.Event)

    def checkpoint_due(self) -> bool:
        return self.checkpoint_requested.is_set()

    def consume_checkpoint(self) -> bool:
        if not self.checkpoint_requested.is_set():
            return False
        self.checkpoint_requested.clear()
        return True

    def should_stop(self) -> bool:
        return self.stop_requested.is_set()

    def wait_if_paused(self, on_state: Callable[[str], None] | None = None) -> bool:
        announced = False
        while self.pause_requested.is_set() and not self.stop_requested.is_set():
            if not announced and on_state:
                on_state("paused")
                announced = True
            self.stop_requested.wait(0.2)
        if announced and on_state and not self.stop_requested.is_set():
            on_state("running")
        return self.stop_requested.is_set()


@dataclass(slots=True)
class TrainingHandle:
    run_id: str
    control: RunControl
    thread: threading.Thread
    latest: dict[str, Any] = field(default_factory=dict)


class Supervisor:
    def __init__(self, store: Store, project_root: str | Path):
        self.store = store
        self.project_root = Path(project_root).resolve()
        self._lock = threading.RLock()
        self._handles: dict[str, TrainingHandle] = {}
        self._recover_interrupted_runs()

    def _recover_interrupted_runs(self) -> None:
        for run in self.store.list_runs(limit=500):
            if run["status"] in {"running", "pausing", "stopping"}:
                self.store.update_run(
                    run["id"],
                    status="interrupted",
                    phase="interrupted",
                    stopped_at=time.time(),
                    last_error="Backend exited while the run was active; resume from its latest checkpoint.",
                )
                self.store.event(
                    run["id"],
                    "run_interrupted",
                    "Recovered an unfinished run after backend restart",
                )

    def active_run_id(self) -> str | None:
        with self._lock:
            for run_id, handle in self._handles.items():
                # A stopping trainer still owns its process pool and Metal
                # learner until the safe boundary and final checkpoint finish.
                # Keep the slot occupied so another run cannot overlap it.
                if handle.thread.is_alive():
                    return run_id
        return None

    def start(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            current = self.active_run_id()
            if current and current != run_id:
                raise InvalidTransition(f"run {current} is already using the trainer")
            existing = self._handles.get(run_id)
            if existing and existing.thread.is_alive():
                if existing.control.stop_requested.is_set():
                    raise InvalidTransition("run is still stopping at a safe boundary")
                existing.control.pause_requested.clear()
                self.store.update_run(run_id, status="running", phase="self_play")
                self.store.event(run_id, "run_resumed", "Training resumed")
                return self.store.get_run(run_id)

            run = self.store.get_run(run_id)
            if run["status"] not in {"ready", "stopped", "interrupted", "failed", "complete"}:
                raise InvalidTransition(f"cannot start a run in state {run['status']}")
            control = RunControl()
            thread = threading.Thread(
                target=self._run_thread,
                args=(run_id, control),
                name=f"astro2-trainer-{run_id}",
                daemon=True,
            )
            handle = TrainingHandle(run_id=run_id, control=control, thread=thread)
            self._handles[run_id] = handle
            now = time.time()
            self.store.update_run(
                run_id,
                status="running",
                phase="initializing",
                started_at=run["started_at"] or now,
                stopped_at=None,
                last_error=None,
            )
            self.store.event(run_id, "run_started", "Training started")
            thread.start()
            return self.store.get_run(run_id)

    def _run_thread(self, run_id: str, control: RunControl) -> None:
        try:
            from .trainer import run_training

            def publish(payload: dict[str, Any]) -> None:
                with self._lock:
                    handle = self._handles.get(run_id)
                    if handle:
                        handle.latest = dict(payload)

            run_training(
                run_id=run_id,
                store=self.store,
                project_root=self.project_root,
                control=control,
                publish=publish,
            )
            final = self.store.get_run(run_id)
            if final["status"] != "failed":
                status = "stopped" if control.should_stop() else "complete"
                if final["status"] != status or final["stopped_at"] is None:
                    self.store.update_run(
                        run_id,
                        status=status,
                        phase=status,
                        stopped_at=time.time(),
                    )
                    self.store.event(run_id, f"run_{status}", f"Training {status}")
        except Exception as error:  # pragma: no cover - defensive process boundary
            self.store.update_run(
                run_id,
                status="failed",
                phase="failed",
                stopped_at=time.time(),
                last_error=f"{type(error).__name__}: {error}",
            )
            self.store.event(
                run_id,
                "run_failed",
                "Training stopped after an error",
                {"error": repr(error)},
            )

    def pause(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._handles.get(run_id)
            if not handle or not handle.thread.is_alive():
                run = self.store.get_run(run_id)
                if run["status"] == "paused":
                    return run
                raise InvalidTransition("run is not active")
            handle.control.pause_requested.set()
            self.store.update_run(run_id, status="pausing")
            self.store.event(run_id, "pause_requested", "Pause requested at the next safe boundary")
            return self.store.get_run(run_id)

    def resume(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._handles.get(run_id)
            if not handle or not handle.thread.is_alive():
                return self.start(run_id)
            if handle.control.stop_requested.is_set():
                raise InvalidTransition("run is still stopping at a safe boundary")
            handle.control.pause_requested.clear()
            self.store.update_run(run_id, status="running", phase="self_play")
            self.store.event(run_id, "run_resumed", "Training resumed")
            return self.store.get_run(run_id)

    def stop(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._handles.get(run_id)
            if handle and handle.thread.is_alive():
                handle.control.stop_requested.set()
                handle.control.pause_requested.clear()
                self.store.update_run(run_id, status="stopping")
                self.store.event(run_id, "stop_requested", "Stop requested at the next safe boundary")
            else:
                run = self.store.get_run(run_id)
                if run["status"] not in {"stopped", "complete", "failed"}:
                    self.store.update_run(run_id, status="stopped", phase="stopped", stopped_at=time.time())
            return self.store.get_run(run_id)

    def checkpoint(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._handles.get(run_id)
            if not handle or not handle.thread.is_alive():
                raise InvalidTransition("run is not active")
            handle.control.checkpoint_requested.set()
            self.store.event(run_id, "checkpoint_requested", "Checkpoint requested")
            return self.store.get_run(run_id)

    def live_snapshot(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        with self._lock:
            handle = self._handles.get(run_id)
            live = dict(handle.latest) if handle else {}
        metrics = self.store.metrics(run_id, after=-1, limit=2_000)
        return {
            "run": run,
            "live": live,
            "latest_metric": metrics[-1] if metrics else None,
            "active": bool(handle and handle.thread.is_alive()),
        }

    def update_config(self, run_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        invalid = set(changes) - SAFE_LIVE_FIELDS
        if invalid:
            raise ValueError(f"these settings require a new run: {sorted(invalid)}")
        run = self.store.get_run(run_id)
        merged = {**run["config"], **changes}
        config = RunConfig.model_validate(merged)
        updated = self.store.update_run(run_id, config_json=config.model_dump_json())
        self.store.event(run_id, "config_updated", "Queued live configuration update", changes)
        return updated

    def create_run(self, config: RunConfig) -> dict[str, Any]:
        return self.store.create_run(config)

    def status_document(self) -> str:
        active = self.active_run_id()
        return json.dumps({"active_run_id": active})

    def shutdown(self, timeout: float = 120.0) -> bool:
        """Request a checkpointed safe stop and wait for the rollout boundary.

        Returns ``True`` when every trainer stopped within the timeout.  A
        backend restart still marks an unfinished run interrupted defensively.
        """

        with self._lock:
            handles = [handle for handle in self._handles.values() if handle.thread.is_alive()]
            for handle in handles:
                handle.control.stop_requested.set()
                handle.control.pause_requested.clear()
                handle.control.checkpoint_requested.set()
                self.store.update_run(handle.run_id, status="stopping", phase="stopping")
                self.store.event(
                    handle.run_id,
                    "shutdown_stop_requested",
                    "Backend shutdown requested a final checkpoint",
                )
        deadline = time.monotonic() + max(0.0, timeout)
        for handle in handles:
            handle.thread.join(max(0.0, deadline - time.monotonic()))
        return all(not handle.thread.is_alive() for handle in handles)
