"""Long-lived training process control, independent of browser connections."""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Callable
from contextlib import suppress
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

    def wait_if_paused(
        self,
        on_state: Callable[[str], None] | None = None,
        while_paused: Callable[[], bool] | None = None,
        on_pause_start: Callable[[], None] | None = None,
        on_pause_end: Callable[[], None] | None = None,
    ) -> bool:
        """Wait at a safe boundary while still servicing durable work.

        ``while_paused`` runs before the first ``paused`` announcement and on
        later wait ticks.  Returning true means it changed the phase (for
        example while writing a requested checkpoint), so ``paused`` is
        announced again only after that work has completed.
        """

        announced = False
        pause_started = False
        try:
            while self.pause_requested.is_set() and not self.stop_requested.is_set():
                if not pause_started:
                    if on_pause_start:
                        on_pause_start()
                    pause_started = True
                did_work = bool(while_paused and while_paused())
                if (not announced or did_work) and on_state:
                    on_state("paused")
                    announced = True
                self.stop_requested.wait(0.2)
        finally:
            if pause_started and on_pause_end:
                on_pause_end()
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
    def __init__(
        self,
        store: Store,
        project_root: str | Path,
        *,
        evaluation_manager: Any | None = None,
    ):
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.evaluation_manager = evaluation_manager
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
            if run["status"] not in {
                "ready",
                "paused",
                "stopped",
                "interrupted",
                "failed",
                "complete",
            }:
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
            try:
                member = self.store.update_branch_member(run_id, status="running")
                self.store.update_branch_experiment(member["experiment_id"], status="running")
            except KeyError:
                pass
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
                evaluation_manager=self.evaluation_manager,
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
        finally:
            # The current thread owns the single accelerator slot until this
            # method returns. A short timer advances an experiment only after
            # the slot is observably free.
            timer = threading.Timer(0.1, self._advance_branch_queue, args=(run_id,))
            timer.daemon = True
            timer.start()

    def _advance_branch_queue(self, completed_run_id: str) -> None:
        try:
            member = self.store.branch_member(completed_run_id)
            experiment = self.store.branch_experiment(member["experiment_id"])
        except KeyError:
            if self.active_run_id() is None:
                queued_experiment = next(
                    (
                        item
                        for item in reversed(self.store.branch_experiments(limit=100))
                        if bool((item.get("config") or {}).get("auto_advance", True))
                        and any(member["status"] == "queued" for member in item["members"])
                    ),
                    None,
                )
                if queued_experiment is not None:
                    queued = next(
                        item
                        for item in queued_experiment["members"]
                        if item["status"] == "queued"
                    )
                    with suppress(InvalidTransition):
                        self.start(queued["run_id"])
            return
        run = self.store.get_run(completed_run_id)
        terminal = run["status"] if run["status"] in {"complete", "failed", "stopped"} else "complete"
        score: float | None = None
        for checkpoint in self.store.checkpoints(completed_run_id):
            latest_arena = (checkpoint.get("evaluation") or {}).get("latest_arena") or {}
            if "model_a_score" in latest_arena:
                value = float(latest_arena["model_a_score"])
                score = value if score is None else max(score, value)
        self.store.update_branch_member(completed_run_id, status=terminal, score=score)
        if terminal == "stopped":
            self.store.update_branch_experiment(member["experiment_id"], status="paused")
            return
        if not bool((experiment.get("config") or {}).get("auto_advance", True)):
            return
        queued = [item for item in experiment["members"] if item["status"] == "queued"]
        if not queued:
            statuses = {item["run_status"] for item in self.store.branch_experiment(member["experiment_id"])["members"]}
            final_status = "failed" if statuses == {"failed"} else "complete"
            self.store.update_branch_experiment(member["experiment_id"], status=final_status)
            return
        if self.active_run_id() is not None:
            return
        try:
            self.start(queued[0]["run_id"])
        except InvalidTransition:
            return

    @staticmethod
    def _copy_checkpoint_artifact(source: str | None, destination: Path) -> str | None:
        if not source:
            return None
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return str(destination)

    def create_branch_experiment(
        self,
        *,
        source_checkpoint_id: str,
        name: str,
        variants: list[dict[str, Any]],
        base_overrides: dict[str, Any] | None = None,
        auto_advance: bool = True,
        start: bool = False,
    ) -> dict[str, Any]:
        """Fork one compatible checkpoint into a sequential GPU experiment."""

        from .model import NumpyActor

        source = self.store.checkpoint(source_checkpoint_id)
        actor_path = source.get("actor_path")
        source_model = Path(source["path"])
        source_sidecar = source_model.with_suffix(source_model.suffix + ".json")
        if (
            not actor_path
            or not Path(actor_path).is_file()
            or not source_model.is_file()
            or not source_sidecar.is_file()
        ):
            raise ValueError("source checkpoint weights and actor must both be available")
        actor = NumpyActor.load(actor_path)
        if actor.spec.encoder_version != 2 or actor.spec.objective_version != 2:
            raise ValueError("Astro5 branches require a generation-4/5 policy checkpoint")
        if not variants:
            variants = [{}]
        experiment = self.store.create_branch_experiment(
            name=name,
            source_checkpoint_id=source_checkpoint_id,
            config={"auto_advance": auto_advance, "branch_count": len(variants)},
        )
        source_artifacts = (source.get("evaluation") or {}).get("artifacts") or {}
        for ordinal, variant in enumerate(variants):
            label = str(variant.get("label") or f"Branch {ordinal + 1}")[:80]
            overrides = {
                **(base_overrides or {}),
                **{key: value for key, value in variant.items() if key != "label"},
            }
            recipe = RunConfig.astro5_search(name=f"{name} · {label}").model_dump()
            recipe.update(
                hidden_size=actor.spec.hidden_size,
                residual_blocks=actor.spec.residual_blocks,
                bootstrap_heads=actor.spec.bootstrap_heads,
                initial_checkpoint_id=source_checkpoint_id,
                branch_experiment_id=experiment["id"],
            )
            recipe.update(overrides)
            # Give branches independent but reproducible game/RNG streams.
            experiment_seed = int(experiment["id"][:8], 16)
            recipe["seed"] = (
                int(recipe.get("seed", 0))
                + experiment_seed
                + ordinal * 10_000_019
            )
            config = RunConfig.model_validate(recipe)
            run = self.store.create_run(config)
            destination = self.store.path.parent / "checkpoints" / run["id"]
            model_destination = destination / "branch-root.safetensors"
            copied_model = self._copy_checkpoint_artifact(source["path"], model_destination)
            if copied_model is None:
                raise ValueError("source model disappeared while creating the branch")
            sidecar_source = str(Path(source["path"]).with_suffix(Path(source["path"]).suffix + ".json"))
            self._copy_checkpoint_artifact(
                sidecar_source,
                model_destination.with_suffix(model_destination.suffix + ".json"),
            )
            copied_actor = self._copy_checkpoint_artifact(
                actor_path, destination / "branch-root.actor.npz"
            )
            artifacts: dict[str, Any] = {"schema_version": 2, "replay_items": 0}
            for artifact_key, filename in (
                ("optimizer_path", "branch-root.optimizer.npz"),
                ("replay_path", "branch-root.replay.npz"),
                ("policy_replay_path", "branch-root.policy-replay.npz"),
                ("preference_replay_path", "branch-root.preference-replay.npz"),
            ):
                copied = self._copy_checkpoint_artifact(
                    source_artifacts.get(artifact_key), destination / filename
                )
                if copied:
                    artifacts[artifact_key] = copied
            for key in (
                "replay_items",
                "replay_capacity",
                "replay_format",
                "policy_replay_items",
                "policy_replay_format",
                "preference_replay_items",
            ):
                if key in source_artifacts:
                    artifacts[key] = source_artifacts[key]
            self.store.add_checkpoint(
                run_id=run["id"],
                parent_id=source_checkpoint_id,
                label=f"Branch root · {source['label']}",
                path=copied_model,
                actor_path=copied_actor,
                games=0,
                champion=True,
                evaluation={
                    "reason": "branch import",
                    "evaluated": True,
                    "source_checkpoint_id": source_checkpoint_id,
                    "source_run_id": source["run_id"],
                    "source_games": source["games"],
                    "artifacts": artifacts,
                    "training_state": {},
                },
            )
            self.store.set_checkpoint_pinned(source_checkpoint_id, True)
            self.store.add_branch_member(
                experiment_id=experiment["id"],
                run_id=run["id"],
                ordinal=ordinal,
                label=label,
                overrides=overrides,
            )
            self.store.event(
                run["id"],
                "branch_created",
                f"Forked from {source['label']}",
                {
                    "experiment_id": experiment["id"],
                    "source_checkpoint_id": source_checkpoint_id,
                    "ordinal": ordinal,
                },
            )
        result = self.store.branch_experiment(experiment["id"])
        if start and result["members"] and self.active_run_id() is None:
            self.start(result["members"][0]["run_id"])
            result = self.store.branch_experiment(experiment["id"])
        return result

    def start_branch_experiment(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.store.branch_experiment(experiment_id)
        queued = [item for item in experiment["members"] if item["status"] == "queued"]
        if not queued:
            raise InvalidTransition("branch experiment has no queued members")
        self.start(queued[0]["run_id"])
        return self.store.branch_experiment(experiment_id)

    def pause(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._handles.get(run_id)
            if not handle or not handle.thread.is_alive():
                run = self.store.get_run(run_id)
                if run["status"] == "paused":
                    return run
                raise InvalidTransition("run is not active")
            # A pause is durable: drain the current actor batch, snapshot the
            # learner/replay state, then enter the paused wait loop.
            handle.control.checkpoint_requested.set()
            # Publish the pause flag only after its checkpoint request.  The
            # trainer can therefore never observe "pause" and enter its wait
            # loop before the durable work is visible.
            handle.control.pause_requested.set()
            self.store.update_run(run_id, status="pausing")
            self.store.event(
                run_id,
                "pause_requested",
                "Pause requested; draining actors before a durable checkpoint",
            )
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
                # A stop is restart-safe for the same reason as a pause: drain
                # the current actor batch and persist a complete learner
                # boundary before releasing the trainer thread.
                run = self.store.get_run(run_id)
                if run["status"] != "paused":
                    handle.control.checkpoint_requested.set()
                handle.control.stop_requested.set()
                handle.control.pause_requested.clear()
                self.store.update_run(run_id, status="stopping")
                self.store.event(
                    run_id, "stop_requested", "Stop requested at the next safe boundary"
                )
            else:
                run = self.store.get_run(run_id)
                if run["status"] not in {"stopped", "complete", "failed"}:
                    self.store.update_run(
                        run_id, status="stopped", phase="stopped", stopped_at=time.time()
                    )
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
        metrics = self.store.metrics(run_id, after=-1, limit=1)
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
                # A reported paused state already follows a complete durable
                # checkpoint. Avoid serializing the same full replay twice on
                # the shutdown immediately following Pause & save.
                if self.store.get_run(handle.run_id)["status"] != "paused":
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
