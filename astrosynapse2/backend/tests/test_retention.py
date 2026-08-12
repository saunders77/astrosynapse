from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from astro2.config import RunConfig
from astro2.retention import RetentionSafetyError, prune_checkpoint_artifacts
from astro2.storage import Store


def _add_artifact_checkpoint(
    store: Store,
    run_id: str,
    root: Path,
    index: int,
    *,
    champion: bool = False,
    complete: bool = True,
) -> tuple[dict[str, object], dict[str, Path]]:
    stem = f"g{index:010d}-test"
    paths = {
        "model": root / f"{stem}.safetensors",
        "model_metadata": root / f"{stem}.safetensors.json",
        "actor": root / f"{stem}.actor.npz",
        "optimizer": root / f"{stem}.optimizer.npz",
        "replay": root / f"{stem}.replay.npz",
    }
    for kind, path in paths.items():
        if complete or kind != "model_metadata":
            path.write_bytes(f"{index}:{kind}".encode())
    checkpoint = store.add_checkpoint(
        run_id=run_id,
        label=f"Checkpoint {index}",
        path=str(paths["model"]),
        actor_path=str(paths["actor"]),
        games=index * 100,
        champion=champion,
        evaluation={
            "diagnostics": {"historical_score": index / 10},
            "artifacts": {
                "optimizer_path": str(paths["optimizer"]),
                "replay_path": str(paths["replay"]),
                "replay_items": 10,
            },
        },
    )
    return checkpoint, paths


def test_retention_prunes_only_unprotected_checkpoint_files(tmp_path):
    store = Store(tmp_path / "astrosynapse2.sqlite3")
    run = store.create_run(RunConfig.quick())
    root = tmp_path / "checkpoints" / run["id"]
    root.mkdir(parents=True)
    checkpoints: list[dict[str, object]] = []
    artifacts: list[dict[str, Path]] = []
    for index in range(10):
        checkpoint, paths = _add_artifact_checkpoint(
            store,
            run["id"],
            root,
            index,
            champion=index == 0,
            complete=index < 8,
        )
        checkpoints.append(checkpoint)
        artifacts.append(paths)

    store.set_checkpoint_pinned(str(checkpoints[1]["id"]), True)
    store.update_checkpoint_evaluation(
        str(checkpoints[6]["id"]),
        {"latest_arena": {"promoted": True, "model_a_score": 0.53}},
    )
    active_job = store.create_arena_job(
        model_a=str(checkpoints[2]["id"]),
        model_b=str(checkpoints[3]["id"]),
        config={"pairs": 8},
    )
    runtime_actor = root / "runtime" / "current.actor.npz"
    runtime_actor.parent.mkdir()
    runtime_actor.write_bytes(b"runtime")

    report = prune_checkpoint_artifacts(
        store,
        run["id"],
        keep_checkpoints=2,
        boundary_checkpoint_id=str(checkpoints[9]["id"]),
    )

    assert report["checkpoints_pruned"] == 2
    assert report["files_removed"] == 10
    assert report["bytes_removed"] > 0
    assert not any(path.exists() for index in (4, 5) for path in artifacts[index].values())
    for index in (0, 1, 2, 3, 6, 7, 8, 9):
        expected = (
            path for kind, path in artifacts[index].items() if kind != "model_metadata" or index < 8
        )
        assert all(path.exists() for path in expected)
    assert runtime_actor.read_bytes() == b"runtime"

    # Rows, diagnostics, and lineage history survive file retention.
    pruned = store.checkpoint(str(checkpoints[4]["id"]))
    assert pruned["evaluation"]["diagnostics"]["historical_score"] == 0.4
    assert pruned["evaluation"]["artifact_retention"]["pruned"] is True
    assert set(pruned["evaluation"]["artifact_retention"]["removed_artifacts"]) == {
        "actor",
        "model",
        "model_metadata",
        "optimizer",
        "replay",
    }
    assert (
        store.checkpoint(str(checkpoints[6]["id"]))["evaluation"]["latest_arena"]["promoted"]
        is True
    )
    boundary = store.checkpoint(str(checkpoints[9]["id"]))
    assert boundary["evaluation"]["retention_boundary"]["files_removed"] == 10
    assert store.events(run["id"])[0]["kind"] == "checkpoint_retention_completed"

    # Finishing the arena releases its two snapshots at the next durable
    # retention boundary, while champion/pin/resume/newest protection remains.
    store.update_arena_job(active_job["id"], status="complete", result={})
    second = prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)
    assert second["checkpoints_pruned"] == 2
    assert not any(path.exists() for index in (2, 3) for path in artifacts[index].values())
    for index in (0, 1, 6, 7, 8, 9):
        assert artifacts[index]["actor"].is_file()


def test_retention_blocks_out_of_root_reference_before_deleting(tmp_path):
    store = Store(tmp_path / "astrosynapse2.sqlite3")
    run = store.create_run(RunConfig())
    root = tmp_path / "checkpoints" / run["id"]
    root.mkdir(parents=True)
    stale, stale_paths = _add_artifact_checkpoint(store, run["id"], root, 0)
    _add_artifact_checkpoint(store, run["id"], root, 1)
    _add_artifact_checkpoint(store, run["id"], root, 2)
    outside = tmp_path / "outside.actor.npz"
    outside.write_bytes(b"outside")
    with store._connect() as db:  # Deliberately simulate a corrupt legacy row.
        db.execute(
            "UPDATE checkpoints SET actor_path = ? WHERE id = ?",
            (str(outside), stale["id"]),
        )

    with pytest.raises(RetentionSafetyError, match="outside the run artifact directory"):
        prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)

    assert outside.read_bytes() == b"outside"
    assert all(path.is_file() for path in stale_paths.values())


def test_retention_never_follows_artifact_symlinks(tmp_path):
    store = Store(tmp_path / "astrosynapse2.sqlite3")
    run = store.create_run(RunConfig())
    root = tmp_path / "checkpoints" / run["id"]
    root.mkdir(parents=True)
    stale, stale_paths = _add_artifact_checkpoint(store, run["id"], root, 0)
    _add_artifact_checkpoint(store, run["id"], root, 1)
    _add_artifact_checkpoint(store, run["id"], root, 2)
    protected_target = tmp_path / "protected.actor.npz"
    protected_target.write_bytes(b"protected")
    stale_paths["actor"].unlink()
    stale_paths["actor"].symlink_to(protected_target)

    with pytest.raises(RetentionSafetyError, match="symbolic link"):
        prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)

    assert protected_target.read_bytes() == b"protected"
    assert stale_paths["model"].is_file()
    assert store.checkpoint(str(stale["id"]))["evaluation"].get("artifact_retention") is None


def test_retention_persists_exact_pending_intent_before_every_unlink(tmp_path, monkeypatch):
    store = Store(tmp_path / "astrosynapse2.sqlite3")
    run = store.create_run(RunConfig.quick())
    root = tmp_path / "checkpoints" / run["id"]
    root.mkdir(parents=True)
    checkpoint_by_path: dict[Path, tuple[str, str]] = {}
    for index in range(4):
        checkpoint, paths = _add_artifact_checkpoint(store, run["id"], root, index)
        if index < 2:
            checkpoint_by_path.update(
                {path: (str(checkpoint["id"]), kind) for kind, path in paths.items()}
            )

    original_unlink = Path.unlink
    observed: set[Path] = set()

    def assert_intent_then_unlink(path: Path, missing_ok: bool = False) -> None:
        if path in checkpoint_by_path:
            checkpoint_id, kind = checkpoint_by_path[path]
            entry = store.checkpoint(checkpoint_id)["evaluation"]["artifact_retention"][
                "pending_artifacts"
            ][kind]
            stat = path.stat()
            assert entry["status"] == "pending"
            assert entry["path"] == str(path)
            assert entry["size_bytes"] == stat.st_size
            assert (entry["device"], entry["inode"]) == (stat.st_dev, stat.st_ino)
            assert entry["modified_ns"] == stat.st_mtime_ns
            assert kind not in store.checkpoint(checkpoint_id)["evaluation"][
                "artifact_retention"
            ].get("removed_artifacts", [])
            observed.add(path)
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", assert_intent_then_unlink)
    report = prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)

    assert observed == set(checkpoint_by_path)
    assert report["files_removed"] == len(checkpoint_by_path)


def test_retention_refuses_a_file_replaced_after_pending_intent(tmp_path, monkeypatch):
    store = Store(tmp_path / "astrosynapse2.sqlite3")
    run = store.create_run(RunConfig.quick())
    root = tmp_path / "checkpoints" / run["id"]
    root.mkdir(parents=True)
    checkpoints: list[dict[str, object]] = []
    artifacts: list[dict[str, Path]] = []
    for index in range(4):
        checkpoint, paths = _add_artifact_checkpoint(store, run["id"], root, index)
        checkpoints.append(checkpoint)
        artifacts.append(paths)

    original_update = store.update_checkpoint_evaluation
    replaced_path: Path | None = None

    def replace_after_first_intent(checkpoint_id, evaluation, *, merge=True):
        nonlocal replaced_path
        result = original_update(checkpoint_id, evaluation, merge=merge)
        pending = evaluation.get("artifact_retention", {}).get("pending_artifacts", {})
        if replaced_path is None and pending:
            replaced_path = Path(next(iter(pending.values()))["path"])
            replaced_path.unlink()
            replaced_path.write_bytes(b"replacement must survive")
        return result

    monkeypatch.setattr(store, "update_checkpoint_evaluation", replace_after_first_intent)
    with pytest.raises(RetentionSafetyError, match="changed after its pending prune intent"):
        prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)

    assert replaced_path is not None
    assert replaced_path.read_bytes() == b"replacement must survive"
    retention = (
        store.checkpoint(str(checkpoints[1]["id"]))["evaluation"].get("artifact_retention")
        or store.checkpoint(str(checkpoints[0]["id"]))["evaluation"]["artifact_retention"]
    )
    pending_entry = next(iter(retention["pending_artifacts"].values()))
    assert pending_entry["status"] == "pending"
    assert "changed after its pending prune intent" in pending_entry["last_error"]
    assert retention.get("removed_artifacts", []) == []


def test_retention_journals_each_unlink_before_a_later_failure(tmp_path, monkeypatch):
    store = Store(tmp_path / "astrosynapse2.sqlite3")
    run = store.create_run(RunConfig.quick())
    root = tmp_path / "checkpoints" / run["id"]
    root.mkdir(parents=True)
    checkpoints: list[dict[str, object]] = []
    artifacts: list[dict[str, Path]] = []
    for index in range(4):
        checkpoint, paths = _add_artifact_checkpoint(store, run["id"], root, index)
        checkpoints.append(checkpoint)
        artifacts.append(paths)

    original_unlink = Path.unlink
    successful_unlinks = 0

    def fail_after_two_unlinks(path: Path, missing_ok: bool = False) -> None:
        nonlocal successful_unlinks
        if path.parent == root:
            if successful_unlinks == 2:
                raise OSError("injected retention failure")
            original_unlink(path, missing_ok=missing_ok)
            successful_unlinks += 1
            return
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_after_two_unlinks)
    with pytest.raises(OSError, match="injected retention failure"):
        prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)

    missing_kinds: dict[str, set[str]] = {}
    for index in (0, 1):
        checkpoint_id = str(checkpoints[index]["id"])
        missing_kinds[checkpoint_id] = {
            kind for kind, path in artifacts[index].items() if not path.exists()
        }
    assert sum(map(len, missing_kinds.values())) == 2
    pending_failures = []
    for checkpoint_id, kinds in missing_kinds.items():
        retention = store.checkpoint(checkpoint_id)["evaluation"].get("artifact_retention")
        if kinds:
            assert retention["pruned"] is True
            assert set(retention["removed_artifacts"]) == kinds
            pending_failures.extend(retention["pending_artifacts"].values())
        else:
            assert retention is None
    assert len(pending_failures) == 1
    assert pending_failures[0]["status"] == "pending"
    assert pending_failures[0]["last_error"] == "OSError: injected retention failure"

    # A later pass skips the two absent files and merges all remaining kinds
    # into the same durable record instead of losing the partial history.
    monkeypatch.setattr(Path, "unlink", original_unlink)
    report = prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)
    assert report["files_removed"] == 8
    for index in (0, 1):
        retention = store.checkpoint(str(checkpoints[index]["id"]))["evaluation"][
            "artifact_retention"
        ]
        assert set(retention["removed_artifacts"]) == set(artifacts[index])
        assert retention["pending_artifacts"] == {}


def test_retention_reconciles_a_crash_after_unlink_before_removed_marker(tmp_path, monkeypatch):
    store = Store(tmp_path / "astrosynapse2.sqlite3")
    run = store.create_run(RunConfig.quick())
    root = tmp_path / "checkpoints" / run["id"]
    root.mkdir(parents=True)
    checkpoints: list[dict[str, object]] = []
    artifacts: list[dict[str, Path]] = []
    for index in range(4):
        checkpoint, paths = _add_artifact_checkpoint(store, run["id"], root, index)
        checkpoints.append(checkpoint)
        artifacts.append(paths)

    original_update = store.update_checkpoint_evaluation
    injected = False

    def fail_first_removed_marker(checkpoint_id, evaluation, *, merge=True):
        nonlocal injected
        retention = evaluation.get("artifact_retention", {})
        records = retention.get("artifact_records", {})
        if not injected and any(record.get("status") == "removed" for record in records.values()):
            injected = True
            raise sqlite3.OperationalError("injected post-unlink SQLite failure")
        return original_update(checkpoint_id, evaluation, merge=merge)

    monkeypatch.setattr(store, "update_checkpoint_evaluation", fail_first_removed_marker)
    with pytest.raises(
        RetentionSafetyError,
        match="confirming the successful .* unlink.*pending intent remains reconcilable",
    ):
        prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)

    missing = [
        (index, kind, path)
        for index in (0, 1)
        for kind, path in artifacts[index].items()
        if not path.exists()
    ]
    assert len(missing) == 1
    missing_index, missing_kind, missing_path = missing[0]
    interrupted = store.checkpoint(str(checkpoints[missing_index]["id"]))["evaluation"][
        "artifact_retention"
    ]
    assert interrupted.get("pruned") is not True
    assert missing_kind not in interrupted.get("removed_artifacts", [])
    assert interrupted["pending_artifacts"][missing_kind]["path"] == str(missing_path)
    assert interrupted["pending_artifacts"][missing_kind]["status"] == "pending"

    monkeypatch.setattr(store, "update_checkpoint_evaluation", original_update)
    restarted_store = Store(tmp_path / "astrosynapse2.sqlite3")
    report = prune_checkpoint_artifacts(restarted_store, run["id"], keep_checkpoints=2)
    assert report["files_reconciled"] == 1
    assert report["bytes_reconciled"] > 0
    assert report["files_removed"] == 9
    reconciled = restarted_store.checkpoint(str(checkpoints[missing_index]["id"]))["evaluation"][
        "artifact_retention"
    ]
    assert reconciled["pending_artifacts"] == {}
    assert missing_kind not in reconciled["removed_artifacts"]
    assert missing_kind in reconciled["reconciled_missing_artifacts"]
    assert reconciled["artifact_records"][missing_kind]["status"] == (
        "missing_after_pending_intent"
    )
    assert reconciled["artifact_records"][missing_kind]["provenance"] == (
        "missing_after_pending_intent"
    )


def test_retention_never_unlinks_when_pending_intent_cannot_be_persisted(tmp_path, monkeypatch):
    store = Store(tmp_path / "astrosynapse2.sqlite3")
    run = store.create_run(RunConfig.quick())
    root = tmp_path / "checkpoints" / run["id"]
    root.mkdir(parents=True)
    checkpoints: list[dict[str, object]] = []
    artifacts: list[dict[str, Path]] = []
    for index in range(4):
        checkpoint, paths = _add_artifact_checkpoint(store, run["id"], root, index)
        checkpoints.append(checkpoint)
        artifacts.append(paths)

    def fail_intent_write(_checkpoint_id, _evaluation, *, merge=True):
        del merge
        raise sqlite3.OperationalError("injected intent SQLite failure")

    monkeypatch.setattr(store, "update_checkpoint_evaluation", fail_intent_write)
    with pytest.raises(
        RetentionSafetyError,
        match="pending prune intent; the artifact was not unlinked",
    ):
        prune_checkpoint_artifacts(store, run["id"], keep_checkpoints=2)

    assert all(path.is_file() for index in (0, 1) for path in artifacts[index].values())
    for index in (0, 1):
        retention = store.checkpoint(str(checkpoints[index]["id"]))["evaluation"].get(
            "artifact_retention"
        )
        assert retention is None
