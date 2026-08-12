"""Bounded, auditable retention for immutable checkpoint artifacts.

Checkpoint rows and their evaluation history remain in SQLite.  This module
only removes the large files belonging to old, unprotected checkpoints, and
records that transition in checkpoint metadata and the audit log.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import RunConfig
from .storage import Store

_ACTIVE_ARENA_STATUSES = frozenset({"queued", "running"})
_MODEL_SUFFIX = ".safetensors"
_ACTOR_SUFFIX = ".actor.npz"
_OPTIMIZER_SUFFIX = ".optimizer.npz"
_REPLAY_SUFFIX = ".replay.npz"
_MODEL_METADATA_SUFFIX = ".safetensors.json"


@dataclass(frozen=True, slots=True)
class _ArtifactReference:
    checkpoint_id: str
    kind: str
    path: Path
    expected_suffix: str


class RetentionSafetyError(RuntimeError):
    """Raised when a configured artifact is not safe to remove."""


def _artifact_references(checkpoint: dict[str, Any]) -> list[_ArtifactReference]:
    checkpoint_id = str(checkpoint["id"])
    result: list[_ArtifactReference] = []

    def append(kind: str, value: object, suffix: str) -> None:
        if isinstance(value, str) and value:
            result.append(
                _ArtifactReference(
                    checkpoint_id=checkpoint_id,
                    kind=kind,
                    path=Path(value).expanduser(),
                    expected_suffix=suffix,
                )
            )

    append("model", checkpoint.get("path"), _MODEL_SUFFIX)
    model_path = checkpoint.get("path")
    if isinstance(model_path, str) and model_path.endswith(_MODEL_SUFFIX):
        append("model_metadata", f"{model_path}.json", _MODEL_METADATA_SUFFIX)
    append("actor", checkpoint.get("actor_path"), _ACTOR_SUFFIX)
    artifacts = (checkpoint.get("evaluation") or {}).get("artifacts") or {}
    if isinstance(artifacts, dict):
        append("optimizer", artifacts.get("optimizer_path"), _OPTIMIZER_SUFFIX)
        append("replay", artifacts.get("replay_path"), _REPLAY_SUFFIX)
    return result


def _checkpoint_is_complete(checkpoint: dict[str, Any], config: RunConfig) -> bool:
    """Return whether a checkpoint still satisfies this run's resume boundary."""

    model_path = Path(str(checkpoint.get("path") or ""))
    actor_path = Path(str(checkpoint.get("actor_path") or ""))
    if not model_path.is_file() or not Path(f"{model_path}.json").is_file():
        return False
    if not actor_path.is_file():
        return False
    artifacts = (checkpoint.get("evaluation") or {}).get("artifacts") or {}
    if not isinstance(artifacts, dict):
        return not config.persist_optimizer_state and config.resume_replay_items <= 0
    if config.persist_optimizer_state:
        optimizer_path = artifacts.get("optimizer_path")
        if not isinstance(optimizer_path, str) or not Path(optimizer_path).is_file():
            return False
    if config.resume_replay_items <= 0:
        return True
    replay_items = artifacts.get("replay_items")
    if replay_items is None:
        return int(checkpoint.get("games", 0)) == 0
    try:
        replay_items = max(0, int(replay_items))
    except (TypeError, ValueError):
        return False
    if replay_items == 0:
        return True
    replay_path = artifacts.get("replay_path")
    return isinstance(replay_path, str) and Path(replay_path).is_file()


def _protected_checkpoint_reasons(
    store: Store,
    run_id: str,
    checkpoints: list[dict[str, Any]],
    *,
    keep_checkpoints: int,
    config: RunConfig,
) -> dict[str, set[str]]:
    reasons: dict[str, set[str]] = {}

    def protect(checkpoint_id: str | None, reason: str) -> None:
        if checkpoint_id:
            reasons.setdefault(checkpoint_id, set()).add(reason)

    for checkpoint in checkpoints[:keep_checkpoints]:
        protect(checkpoint["id"], "newest")

    run = store.get_run(run_id)
    protect(run.get("champion_id"), "current_champion")
    for checkpoint in checkpoints:
        if checkpoint["is_champion"]:
            protect(checkpoint["id"], "champion")
        if checkpoint["is_pinned"]:
            protect(checkpoint["id"], "pinned")
        latest_arena = (checkpoint.get("evaluation") or {}).get("latest_arena") or {}
        if isinstance(latest_arena, dict) and bool(latest_arena.get("promoted")):
            # _sync_league intentionally keeps accepted historical policies as
            # opponents after the champion flag moves.  Their actors are part
            # of the training population, not disposable diagnostics.
            protect(checkpoint["id"], "promoted_league_member")

    checkpoint_ids = {checkpoint["id"] for checkpoint in checkpoints}
    for job in store.arena_jobs(limit=1_000_000, include_internal=True):
        if job["status"] not in _ACTIVE_ARENA_STATUSES:
            continue
        for field in ("model_a", "model_b"):
            checkpoint_id = job.get(field)
            if checkpoint_id in checkpoint_ids:
                protect(checkpoint_id, "active_evaluation")

    # Full snapshots are self-contained, so lineage ancestors remain useful as
    # SQLite history but are not required as files.  Keep the newest complete
    # boundary explicitly so a partially written newest snapshot cannot remove
    # the trainer's last exact resume point.
    for checkpoint in checkpoints:
        if _checkpoint_is_complete(checkpoint, config):
            protect(checkpoint["id"], "latest_complete_resume_boundary")
            break
    return reasons


def _resolve_candidate(reference: _ArtifactReference, root: Path) -> Path:
    if not reference.path.is_absolute():
        raise RetentionSafetyError(
            f"{reference.checkpoint_id}:{reference.kind} is not an absolute path"
        )
    if reference.path.is_symlink():
        raise RetentionSafetyError(f"{reference.checkpoint_id}:{reference.kind} is a symbolic link")
    resolved = reference.path.resolve(strict=False)
    if resolved.parent != root:
        raise RetentionSafetyError(
            f"{reference.checkpoint_id}:{reference.kind} is outside the run artifact directory"
        )
    if not resolved.name.endswith(reference.expected_suffix):
        raise RetentionSafetyError(
            f"{reference.checkpoint_id}:{reference.kind} has an unexpected filename"
        )
    if resolved.exists() and not resolved.is_file():
        raise RetentionSafetyError(
            f"{reference.checkpoint_id}:{reference.kind} is not a regular file"
        )
    return resolved


def _retention_metadata(
    store: Store,
    checkpoint_id: str,
    *,
    operation: str,
) -> dict[str, Any]:
    try:
        checkpoint = store.checkpoint(checkpoint_id)
    except sqlite3.Error as error:
        raise RetentionSafetyError(
            f"SQLite failed while {operation} for checkpoint {checkpoint_id}: {error}"
        ) from error
    prior = (checkpoint.get("evaluation") or {}).get("artifact_retention") or {}
    if not isinstance(prior, dict):
        raise RetentionSafetyError(
            f"checkpoint {checkpoint_id} has malformed artifact retention metadata"
        )
    return dict(prior)


def _metadata_mapping(
    retention: dict[str, Any], field: str, checkpoint_id: str
) -> dict[str, dict[str, Any]]:
    value = retention.get(field, {})
    if not isinstance(value, dict) or any(not isinstance(item, dict) for item in value.values()):
        raise RetentionSafetyError(
            f"checkpoint {checkpoint_id} has malformed artifact retention {field}"
        )
    return {str(kind): dict(item) for kind, item in value.items()}


def _persist_retention_metadata(
    store: Store,
    checkpoint_id: str,
    retention: dict[str, Any],
    *,
    operation: str,
) -> None:
    try:
        store.update_checkpoint_evaluation(
            checkpoint_id,
            {"artifact_retention": retention},
        )
    except sqlite3.Error as error:
        raise RetentionSafetyError(
            f"SQLite failed while {operation} for checkpoint {checkpoint_id}: {error}"
        ) from error


def _record_pending_prune_intents(
    store: Store,
    references: list[_ArtifactReference],
    path: Path,
    *,
    size_bytes: int,
    device: int,
    inode: int,
    modified_ns: int,
    intended_at: float,
) -> None:
    """Persist authorization and provenance for every reference before unlink."""

    for reference in references:
        retention = _retention_metadata(
            store,
            reference.checkpoint_id,
            operation=f"reading the {reference.kind} pre-unlink journal",
        )
        pending = _metadata_mapping(
            retention,
            "pending_artifacts",
            reference.checkpoint_id,
        )
        prior = pending.get(reference.kind, {})
        prior_path = prior.get("path")
        if prior_path is not None and prior_path != str(path):
            raise RetentionSafetyError(
                f"{reference.checkpoint_id}:{reference.kind} has a pending intent for "
                "a different artifact"
            )
        try:
            attempts = max(0, int(prior.get("attempts", 0))) + 1
        except (TypeError, ValueError):
            attempts = 1
        try:
            first_intended_at = float(prior.get("intended_at", intended_at))
        except (TypeError, ValueError) as error:
            raise RetentionSafetyError(
                f"{reference.checkpoint_id}:{reference.kind} has an invalid pending intent time"
            ) from error
        entry = {
            **prior,
            "kind": reference.kind,
            "path": str(path),
            "status": "pending",
            "intended_at": first_intended_at,
            "last_attempt_at": intended_at,
            "attempts": attempts,
            "size_bytes": size_bytes,
            "device": device,
            "inode": inode,
            "modified_ns": modified_ns,
        }
        pending[reference.kind] = entry
        retention["pending_artifacts"] = pending
        retention["last_attempt_at"] = intended_at
        _persist_retention_metadata(
            store,
            reference.checkpoint_id,
            retention,
            operation=(
                f"persisting the {reference.kind} pending prune intent; "
                "the artifact was not unlinked"
            ),
        )


def _record_pending_failure(
    store: Store,
    references: list[_ArtifactReference],
    path: Path,
    error: OSError | RetentionSafetyError,
    *,
    failed_at: float,
) -> None:
    detail = f"{type(error).__name__}: {error}"
    for reference in references:
        retention = _retention_metadata(
            store,
            reference.checkpoint_id,
            operation=f"reading the failed {reference.kind} prune intent",
        )
        pending = _metadata_mapping(
            retention,
            "pending_artifacts",
            reference.checkpoint_id,
        )
        entry = pending.get(reference.kind)
        if not isinstance(entry, dict) or entry.get("path") != str(path):
            raise RetentionSafetyError(
                f"{reference.checkpoint_id}:{reference.kind} lost its pending prune intent"
            )
        entry = {
            **entry,
            "status": "pending",
            "last_failed_at": failed_at,
            "last_error": detail,
        }
        pending[reference.kind] = entry
        retention["pending_artifacts"] = pending
        retention["last_failure"] = {
            "kind": reference.kind,
            "path": str(path),
            "failed_at": failed_at,
            "error": detail,
        }
        _persist_retention_metadata(
            store,
            reference.checkpoint_id,
            retention,
            operation=f"recording the failed {reference.kind} unlink",
        )


def _mark_artifact_absent(
    store: Store,
    reference: _ArtifactReference,
    path: Path,
    removed_by_checkpoint: dict[str, list[str]],
    *,
    completed_at: float,
    confirmed_unlink: bool,
) -> None:
    """Complete one pending journal entry after unlink or crash reconciliation."""

    provenance = "unlink_confirmed" if confirmed_unlink else "missing_after_pending_intent"
    retention = _retention_metadata(
        store,
        reference.checkpoint_id,
        operation=f"reading the {reference.kind} post-unlink journal",
    )
    pending = _metadata_mapping(
        retention,
        "pending_artifacts",
        reference.checkpoint_id,
    )
    entry = pending.get(reference.kind)
    if not isinstance(entry, dict) or entry.get("path") != str(path):
        raise RetentionSafetyError(
            f"{reference.checkpoint_id}:{reference.kind} has no matching pending prune intent"
        )
    pending.pop(reference.kind)
    records = _metadata_mapping(
        retention,
        "artifact_records",
        reference.checkpoint_id,
    )
    record = {
        **entry,
        "status": "removed" if confirmed_unlink else "missing_after_pending_intent",
        "provenance": provenance,
    }
    if confirmed_unlink:
        record["removed_at"] = completed_at
    else:
        record["reconciled_at"] = completed_at
    records[reference.kind] = record
    prior_kinds = retention.get("removed_artifacts", [])
    if not isinstance(prior_kinds, list):
        raise RetentionSafetyError(
            f"checkpoint {reference.checkpoint_id} has malformed removed artifact metadata"
        )
    removed_kinds = {str(kind) for kind in prior_kinds}
    reconciled_kinds = retention.get("reconciled_missing_artifacts", [])
    if not isinstance(reconciled_kinds, list):
        raise RetentionSafetyError(
            f"checkpoint {reference.checkpoint_id} has malformed reconciled artifact metadata"
        )
    reconciled_missing = {str(kind) for kind in reconciled_kinds}
    if confirmed_unlink:
        removed_kinds.add(reference.kind)
        reconciled_missing.discard(reference.kind)
    else:
        reconciled_missing.add(reference.kind)
    retention.update(
        {
            "pruned": True,
            "pruned_at": retention.get("pruned_at", completed_at),
            "last_pruned_at": completed_at,
            "removed_artifacts": sorted(removed_kinds),
            "reconciled_missing_artifacts": sorted(reconciled_missing),
            "pending_artifacts": pending,
            "artifact_records": records,
        }
    )
    operation = (
        f"confirming the successful {reference.kind} unlink; its pending intent remains "
        "reconcilable"
        if confirmed_unlink
        else f"reconciling the missing {reference.kind} artifact from its pending intent"
    )
    _persist_retention_metadata(
        store,
        reference.checkpoint_id,
        retention,
        operation=operation,
    )
    removed_by_checkpoint.setdefault(reference.checkpoint_id, []).append(reference.kind)


def _reconcile_missing_pending_intents(
    store: Store,
    checkpoints: list[dict[str, Any]],
    root: Path,
    removed_by_checkpoint: dict[str, list[str]],
) -> tuple[int, int]:
    """Finish journals left between unlink and SQLite confirmation.

    Filesystem deletion and SQLite cannot share a transaction.  Recording each
    intent first means an absent file can be reconciled without pretending the
    current process observed a successful unlink.
    """

    reconciled_paths: set[Path] = set()
    reconciled_bytes: dict[Path, int] = {}
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint["id"])
        retention = (checkpoint.get("evaluation") or {}).get("artifact_retention") or {}
        if not isinstance(retention, dict):
            raise RetentionSafetyError(
                f"checkpoint {checkpoint_id} has malformed artifact retention metadata"
            )
        pending = _metadata_mapping(retention, "pending_artifacts", checkpoint_id)
        if not pending:
            continue
        references = {reference.kind: reference for reference in _artifact_references(checkpoint)}
        for kind, entry in pending.items():
            reference = references.get(kind)
            if reference is None:
                raise RetentionSafetyError(
                    f"{checkpoint_id}:{kind} pending intent no longer matches checkpoint metadata"
                )
            pending_path = entry.get("path")
            if not isinstance(pending_path, str):
                raise RetentionSafetyError(
                    f"{checkpoint_id}:{kind} pending intent has no valid path"
                )
            resolved = _resolve_candidate(reference, root)
            if pending_path != str(resolved):
                raise RetentionSafetyError(f"{checkpoint_id}:{kind} pending intent path changed")
            if resolved.exists():
                current_stat = resolved.stat()
                intended_identity = (
                    entry.get("device"),
                    entry.get("inode"),
                    entry.get("size_bytes"),
                    entry.get("modified_ns"),
                )
                current_identity = (
                    current_stat.st_dev,
                    current_stat.st_ino,
                    current_stat.st_size,
                    current_stat.st_mtime_ns,
                )
                if intended_identity != current_identity:
                    raise RetentionSafetyError(
                        f"{checkpoint_id}:{kind} artifact changed after its pending prune intent"
                    )
                continue
            _mark_artifact_absent(
                store,
                reference,
                resolved,
                removed_by_checkpoint,
                completed_at=time.time(),
                confirmed_unlink=False,
            )
            reconciled_paths.add(resolved)
            try:
                reconciled_bytes[resolved] = max(0, int(entry.get("size_bytes", 0)))
            except (TypeError, ValueError):
                reconciled_bytes[resolved] = 0
    return len(reconciled_paths), sum(reconciled_bytes.values())


def prune_checkpoint_artifacts(
    store: Store,
    run_id: str,
    *,
    keep_checkpoints: int,
    boundary_checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Prune old checkpoint files after a caller-established durable boundary.

    The artifact root is derived from the store and run ID; callers cannot
    supply a broader deletion target.  Every target is preflighted before the
    first unlink, and only explicit files referenced by checkpoint records are
    eligible.  The function never removes directories, database rows, metrics,
    evaluations, or runtime actor exports.
    """

    if keep_checkpoints < 1:
        raise ValueError("keep_checkpoints must be positive")
    run = store.get_run(run_id)
    config = RunConfig.model_validate(run["config"])
    checkpoint_parent = store.path.parent / "checkpoints"
    root = checkpoint_parent / run_id
    root = root.absolute()
    if checkpoint_parent.is_symlink() or root.is_symlink():
        raise RetentionSafetyError("run artifact directory must not be a symbolic link")
    root = root.resolve(strict=False)
    checkpoints = store.checkpoints(run_id)
    removed_by_checkpoint: dict[str, list[str]] = {}
    files_reconciled, bytes_reconciled = _reconcile_missing_pending_intents(
        store,
        checkpoints,
        root,
        removed_by_checkpoint,
    )
    # Reconciliation can update artifact-retention metadata.  Reload before
    # protection and candidate selection so this pass uses the persisted view.
    checkpoints = store.checkpoints(run_id)
    protected = _protected_checkpoint_reasons(
        store,
        run_id,
        checkpoints,
        keep_checkpoints=keep_checkpoints,
        config=config,
    )
    checkpoint_by_id = {checkpoint["id"]: checkpoint for checkpoint in checkpoints}

    protected_paths: set[Path] = set()
    for checkpoint_id in protected:
        checkpoint = checkpoint_by_id.get(checkpoint_id)
        if checkpoint is None:
            continue
        for reference in _artifact_references(checkpoint):
            try:
                protected_paths.add(_resolve_candidate(reference, root))
            except RetentionSafetyError:
                # A protected external/legacy path is never a deletion target.
                continue

    candidates: dict[Path, list[_ArtifactReference]] = {}
    for checkpoint in checkpoints:
        if checkpoint["id"] in protected:
            continue
        for reference in _artifact_references(checkpoint):
            resolved = _resolve_candidate(reference, root)
            if resolved in protected_paths:
                continue
            candidates.setdefault(resolved, []).append(reference)

    files_removed = 0
    bytes_removed = 0
    for path, references in candidates.items():
        if not path.exists():
            continue
        # Revalidate at the destructive boundary to catch a path swapped after
        # preflight.  A failure stops the pass; it never broadens the target.
        if path.is_symlink() or path.resolve(strict=True).parent != root or not path.is_file():
            raise RetentionSafetyError(f"artifact changed after retention preflight: {path.name}")
        intended_stat = path.stat()
        size = intended_stat.st_size
        intended_at = time.time()
        _record_pending_prune_intents(
            store,
            references,
            path,
            size_bytes=size,
            device=intended_stat.st_dev,
            inode=intended_stat.st_ino,
            modified_ns=intended_stat.st_mtime_ns,
            intended_at=intended_at,
        )
        try:
            # The journal commit can take arbitrarily long.  Verify that the
            # exact regular file authorized by the intent still occupies the
            # path immediately before unlinking it.
            if path.is_symlink():
                raise RetentionSafetyError(
                    f"artifact changed after its pending prune intent: {path.name}"
                )
            current_stat = path.stat()
            if (
                not path.is_file()
                or path.resolve(strict=True).parent != root
                or (
                    current_stat.st_dev,
                    current_stat.st_ino,
                    current_stat.st_size,
                    current_stat.st_mtime_ns,
                )
                != (
                    intended_stat.st_dev,
                    intended_stat.st_ino,
                    intended_stat.st_size,
                    intended_stat.st_mtime_ns,
                )
            ):
                raise RetentionSafetyError(
                    f"artifact changed after its pending prune intent: {path.name}"
                )
            path.unlink()
        except (OSError, RetentionSafetyError) as error:
            try:
                _record_pending_failure(
                    store,
                    references,
                    path,
                    error,
                    failed_at=time.time(),
                )
            except RetentionSafetyError as journal_error:
                error.add_note(f"retention failure journal also failed: {journal_error}")
            raise
        removed_at = time.time()
        for reference in references:
            _mark_artifact_absent(
                store,
                reference,
                path,
                removed_by_checkpoint,
                completed_at=removed_at,
                confirmed_unlink=True,
            )
        files_removed += 1
        bytes_removed += size

    completed_at = time.time()

    report: dict[str, Any] = {
        "status": "complete",
        "keep_checkpoints": keep_checkpoints,
        "checkpoints_considered": len(checkpoints),
        "checkpoints_protected": len(protected),
        "checkpoints_pruned": len(removed_by_checkpoint),
        "files_removed": files_removed,
        "bytes_removed": bytes_removed,
        "files_reconciled": files_reconciled,
        "bytes_reconciled": bytes_reconciled,
        "protected_reasons": {
            checkpoint_id: sorted(reasons) for checkpoint_id, reasons in protected.items()
        },
        "completed_at": completed_at,
    }
    if boundary_checkpoint_id is not None and boundary_checkpoint_id in checkpoint_by_id:
        store.update_checkpoint_evaluation(
            boundary_checkpoint_id,
            {"retention_boundary": report},
        )
    store.event(
        run_id,
        "checkpoint_retention_completed",
        (
            f"Checkpoint retention removed {files_removed:,} files "
            f"({bytes_removed / (1024 * 1024):.1f} MiB)"
        ),
        report,
    )
    return report


__all__ = [
    "RetentionSafetyError",
    "prune_checkpoint_artifacts",
]
