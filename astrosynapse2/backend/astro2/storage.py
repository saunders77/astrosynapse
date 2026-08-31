"""Thread-safe local persistence for runs, metrics, models, and audit events."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import RunConfig

SCHEMA_VERSION = 3


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._metric_condition = threading.Condition()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open one transactional connection and always release its file handles.

        ``sqlite3.Connection`` commits or rolls back when used as a context
        manager, but it does not close itself. Store calls are frequent during
        training, so every transaction must explicitly close its connection.
        """

        connection = self._open_connection()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    stopped_at REAL,
                    games INTEGER NOT NULL DEFAULT 0,
                    decisions INTEGER NOT NULL DEFAULT 0,
                    updates INTEGER NOT NULL DEFAULT 0,
                    champion_id TEXT,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    seq INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE(run_id, seq)
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_run_seq ON metrics(run_id, seq);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    parent_id TEXT,
                    label TEXT NOT NULL,
                    path TEXT NOT NULL,
                    actor_path TEXT,
                    games INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    is_champion INTEGER NOT NULL DEFAULT 0,
                    is_pinned INTEGER NOT NULL DEFAULT 0,
                    eval_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_run_games ON checkpoints(run_id, games);
                CREATE TABLE IF NOT EXISTS arena_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    run_id TEXT,
                    promotion_tier TEXT,
                    trainer_scheduled INTEGER NOT NULL DEFAULT 0,
                    config_json TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    created_at REAL NOT NULL,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_audit_run_time ON audit_events(run_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS branch_experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_checkpoint_id TEXT NOT NULL REFERENCES checkpoints(id),
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS branch_members (
                    experiment_id TEXT NOT NULL REFERENCES branch_experiments(id) ON DELETE CASCADE,
                    run_id TEXT NOT NULL UNIQUE REFERENCES runs(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    overrides_json TEXT NOT NULL DEFAULT '{}',
                    score REAL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(experiment_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_branch_members_experiment
                    ON branch_members(experiment_id, ordinal);
                CREATE TABLE IF NOT EXISTS run_controller_state (
                    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                    state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                );
                """
            )
            # Version 3 materializes the trainer's arena predicates. Before
            # this, every ~128-game loop loaded and JSON-decoded the complete
            # arena history several times. ALTERs are intentionally discovered
            # from table_info so databases created by older releases migrate
            # safely without a separate bootstrap path.
            arena_columns = {
                str(row["name"])
                for row in db.execute("PRAGMA table_info(arena_jobs)").fetchall()
            }
            if "run_id" not in arena_columns:
                db.execute("ALTER TABLE arena_jobs ADD COLUMN run_id TEXT")
            if "promotion_tier" not in arena_columns:
                db.execute("ALTER TABLE arena_jobs ADD COLUMN promotion_tier TEXT")
            if "trainer_scheduled" not in arena_columns:
                db.execute(
                    "ALTER TABLE arena_jobs ADD COLUMN trainer_scheduled INTEGER NOT NULL DEFAULT 0"
                )
            db.execute(
                """UPDATE arena_jobs
                   SET run_id = (
                       SELECT run_id FROM checkpoints WHERE checkpoints.id = arena_jobs.model_a
                   )
                   WHERE run_id IS NULL"""
            )
            db.execute(
                """UPDATE arena_jobs
                   SET promotion_tier = COALESCE(
                       json_extract(config_json, '$.promotion_tier'),
                       CASE WHEN json_extract(config_json, '$.automatic_promotion') THEN 'full'
                            ELSE 'diagnostic' END
                   )
                   WHERE promotion_tier IS NULL"""
            )
            db.execute(
                """UPDATE arena_jobs
                   SET trainer_scheduled = 1
                   WHERE trainer_scheduled = 0
                     AND (json_extract(config_json, '$.trainer_scheduled') = 1
                          OR json_extract(config_json, '$.automatic_promotion') = 1)"""
            )
            db.execute(
                """CREATE INDEX IF NOT EXISTS idx_arena_trainer_run_status_tier
                   ON arena_jobs(run_id, trainer_scheduled, status, promotion_tier, created_at DESC)"""
            )
            db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def create_run(self, config: RunConfig) -> dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._connect() as db:
            db.execute(
                """INSERT INTO runs(
                    id, name, status, phase, config_json, created_at, updated_at
                ) VALUES(?, ?, 'ready', 'ready', ?, ?, ?)""",
                (run_id, config.name, config.model_dump_json(), now, now),
            )
        self.event(run_id, "run_created", f"Created {config.name}", config.model_dump())
        return self.get_run(run_id)

    def create_branch_experiment(
        self,
        *,
        name: str,
        source_checkpoint_id: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a durable collection of independently trainable branches."""

        # Resolve first so callers get a useful not-found failure without a
        # partially created experiment.
        self.checkpoint(source_checkpoint_id)
        experiment_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._connect() as db:
            db.execute(
                """INSERT INTO branch_experiments(
                    id, name, status, source_checkpoint_id, config_json, created_at, updated_at
                ) VALUES(?, ?, 'ready', ?, ?, ?, ?)""",
                (
                    experiment_id,
                    name,
                    source_checkpoint_id,
                    json.dumps(config or {}, separators=(",", ":")),
                    now,
                    now,
                ),
            )
        return self.branch_experiment(experiment_id)

    def add_branch_member(
        self,
        *,
        experiment_id: str,
        run_id: str,
        ordinal: int,
        label: str,
        overrides: dict[str, Any] | None = None,
        status: str = "queued",
    ) -> dict[str, Any]:
        now = time.time()
        with self._connect() as db:
            db.execute(
                """INSERT INTO branch_members(
                    experiment_id, run_id, ordinal, label, status,
                    overrides_json, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    run_id,
                    int(ordinal),
                    label,
                    status,
                    json.dumps(overrides or {}, separators=(",", ":")),
                    now,
                ),
            )
            db.execute(
                "UPDATE branch_experiments SET updated_at = ? WHERE id = ?",
                (now, experiment_id),
            )
        return self.branch_member(run_id)

    def branch_member(self, run_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM branch_members WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        item = dict(row)
        item["overrides"] = json.loads(item.pop("overrides_json"))
        return item

    def update_branch_member(
        self,
        run_id: str,
        *,
        status: str | None = None,
        score: float | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if status is not None:
            fields["status"] = status
        if score is not None:
            fields["score"] = float(score)
        if not fields:
            return self.branch_member(run_id)
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._connect() as db:
            cursor = db.execute(
                f"UPDATE branch_members SET {assignments} WHERE run_id = ?",
                [*fields.values(), run_id],
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)
            row = db.execute(
                "SELECT experiment_id FROM branch_members WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE branch_experiments SET updated_at = ? WHERE id = ?",
                    (time.time(), row["experiment_id"]),
                )
        return self.branch_member(run_id)

    def branch_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM branch_experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            members = db.execute(
                """SELECT bm.*, r.status AS run_status, r.games, r.updates,
                          r.champion_id, r.last_error
                   FROM branch_members bm JOIN runs r ON r.id = bm.run_id
                   WHERE bm.experiment_id = ? ORDER BY bm.ordinal""",
                (experiment_id,),
            ).fetchall()
        if row is None:
            raise KeyError(experiment_id)
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        item["members"] = []
        for member_row in members:
            member = dict(member_row)
            member["overrides"] = json.loads(member.pop("overrides_json"))
            item["members"].append(member)
        return item

    def branch_experiments(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            ids = [
                row["id"]
                for row in db.execute(
                    "SELECT id FROM branch_experiments ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
        return [self.branch_experiment(experiment_id) for experiment_id in ids]

    def update_branch_experiment(
        self, experiment_id: str, *, status: str
    ) -> dict[str, Any]:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE branch_experiments SET status = ?, updated_at = ? WHERE id = ?",
                (status, time.time(), experiment_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(experiment_id)
        return self.branch_experiment(experiment_id)

    def controller_state(self, run_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT state_json, updated_at FROM run_controller_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return {}
        return {**json.loads(row["state_json"]), "updated_at": row["updated_at"]}

    def set_controller_state(self, run_id: str, state: dict[str, Any]) -> dict[str, Any]:
        self.get_run(run_id)
        now = time.time()
        with self._connect() as db:
            db.execute(
                """INSERT INTO run_controller_state(run_id, state_json, updated_at)
                   VALUES(?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET
                       state_json=excluded.state_json, updated_at=excluded.updated_at""",
                (run_id, json.dumps(state, separators=(",", ":")), now),
            )
        return {**state, "updated_at": now}

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._decode_run(row)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_run(row) for row in rows]

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        return item

    def update_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        allowed = {
            "name",
            "status",
            "phase",
            "config_json",
            "started_at",
            "stopped_at",
            "games",
            "decisions",
            "updates",
            "champion_id",
            "last_error",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"unsupported run fields: {sorted(invalid)}")
        if not fields:
            return self.get_run(run_id)
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [run_id]
        with self._connect() as db:
            cursor = db.execute(f"UPDATE runs SET {assignments} WHERE id = ?", values)
            if cursor.rowcount != 1:
                raise KeyError(run_id)
        return self.get_run(run_id)

    def append_metric(self, run_id: str, seq: int, payload: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO metrics(run_id, seq, created_at, payload_json) VALUES(?, ?, ?, ?)",
                (run_id, seq, time.time(), json.dumps(payload, separators=(",", ":"))),
            )
        with self._metric_condition:
            self._metric_condition.notify_all()

    def metrics(self, run_id: str, after: int = -1, limit: int = 2_000) -> list[dict[str, Any]]:
        with self._connect() as db:
            if after < 0:
                rows = db.execute(
                    """SELECT seq, created_at, payload_json FROM metrics
                       WHERE run_id = ? ORDER BY seq DESC LIMIT ?""",
                    (run_id, limit),
                ).fetchall()[::-1]
            else:
                rows = db.execute(
                    """SELECT seq, created_at, payload_json FROM metrics
                       WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?""",
                    (run_id, after, limit),
                ).fetchall()
        return [
            {"seq": row["seq"], "created_at": row["created_at"], **json.loads(row["payload_json"])}
            for row in rows
        ]

    def wait_for_metrics(
        self,
        run_id: str,
        after: int,
        timeout: float = 15.0,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Block until telemetry advances, with a cross-process fallback poll."""

        with self._metric_condition:
            rows = self.metrics(run_id, after=after, limit=limit)
            if rows:
                return rows
            self._metric_condition.wait(timeout=max(0.0, float(timeout)))
        return self.metrics(run_id, after=after, limit=limit)

    def add_checkpoint(
        self,
        *,
        run_id: str,
        label: str,
        path: str,
        actor_path: str | None,
        games: int,
        parent_id: str | None = None,
        champion: bool = False,
        evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        checkpoint_id = uuid.uuid4().hex[:16]
        with self._connect() as db:
            if champion:
                db.execute("UPDATE checkpoints SET is_champion = 0 WHERE run_id = ?", (run_id,))
            db.execute(
                """INSERT INTO checkpoints(
                    id, run_id, parent_id, label, path, actor_path, games, created_at,
                    is_champion, eval_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    checkpoint_id,
                    run_id,
                    parent_id,
                    label,
                    path,
                    actor_path,
                    games,
                    time.time(),
                    int(champion),
                    json.dumps(evaluation or {}, separators=(",", ":")),
                ),
            )
            if champion:
                db.execute(
                    "UPDATE runs SET champion_id = ?, updated_at = ? WHERE id = ?",
                    (checkpoint_id, time.time(), run_id),
                )
        return self.checkpoint(checkpoint_id)

    def checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)).fetchone()
        if row is None:
            raise KeyError(checkpoint_id)
        item = dict(row)
        item["evaluation"] = json.loads(item.pop("eval_json"))
        item["is_champion"] = bool(item["is_champion"])
        item["is_pinned"] = bool(item["is_pinned"])
        return item

    def checkpoints(self, run_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM checkpoints"
        args: tuple[Any, ...] = ()
        if run_id:
            query += " WHERE run_id = ?"
            args = (run_id,)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, args).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evaluation"] = json.loads(item.pop("eval_json"))
            item["is_champion"] = bool(item["is_champion"])
            item["is_pinned"] = bool(item["is_pinned"])
            result.append(item)
        return result

    def update_checkpoint_evaluation(
        self,
        checkpoint_id: str,
        evaluation: dict[str, Any],
        *,
        merge: bool = True,
    ) -> dict[str, Any]:
        """Persist evaluation metadata without changing champion state."""

        with self._connect() as db:
            row = db.execute(
                "SELECT eval_json FROM checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
            if row is None:
                raise KeyError(checkpoint_id)
            value = json.loads(row["eval_json"]) if merge else {}
            value.update(evaluation)
            db.execute(
                "UPDATE checkpoints SET eval_json = ? WHERE id = ?",
                (json.dumps(value, separators=(",", ":")), checkpoint_id),
            )
        return self.checkpoint(checkpoint_id)

    def set_run_champion(self, run_id: str, checkpoint_id: str) -> dict[str, Any]:
        """Atomically restore a run's deployment anchor without inventing an arena result."""

        with self._connect() as db:
            row = db.execute(
                "SELECT run_id FROM checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
            if row is None or row["run_id"] != run_id:
                raise KeyError(checkpoint_id)
            db.execute("UPDATE checkpoints SET is_champion = 0 WHERE run_id = ?", (run_id,))
            db.execute("UPDATE checkpoints SET is_champion = 1 WHERE id = ?", (checkpoint_id,))
            db.execute(
                "UPDATE runs SET champion_id = ?, updated_at = ? WHERE id = ?",
                (checkpoint_id, time.time(), run_id),
            )
        return self.checkpoint(checkpoint_id)

    def finalize_checkpoint_arena(
        self,
        checkpoint_id: str,
        arena_evaluation: dict[str, Any],
        *,
        promote: bool = False,
    ) -> dict[str, Any]:
        """Atomically record an arena result and, if authorized, promote it.

        When ``promote`` is true, the candidate flag, every sibling champion
        flag, and ``runs.champion_id`` change in one SQLite transaction.
        Eligibility is deliberately checked by the arena layer before this
        low-level transaction helper is called.
        """

        with self._connect() as db:
            row = db.execute(
                "SELECT run_id, eval_json FROM checkpoints WHERE id = ?",
                (checkpoint_id,),
            ).fetchone()
            if row is None:
                raise KeyError(checkpoint_id)
            evaluation = json.loads(row["eval_json"])
            history = list(evaluation.get("arena_history", []))
            history.append(arena_evaluation)
            evaluation["arena_history"] = history[-20:]
            evaluation["latest_arena"] = arena_evaluation
            db.execute(
                "UPDATE checkpoints SET eval_json = ? WHERE id = ?",
                (json.dumps(evaluation, separators=(",", ":")), checkpoint_id),
            )
            if promote:
                run_id = row["run_id"]
                db.execute("UPDATE checkpoints SET is_champion = 0 WHERE run_id = ?", (run_id,))
                db.execute("UPDATE checkpoints SET is_champion = 1 WHERE id = ?", (checkpoint_id,))
                cursor = db.execute(
                    "UPDATE runs SET champion_id = ?, updated_at = ? WHERE id = ?",
                    (checkpoint_id, time.time(), run_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(run_id)
        return self.checkpoint(checkpoint_id)

    def set_checkpoint_pinned(self, checkpoint_id: str, pinned: bool) -> dict[str, Any]:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE checkpoints SET is_pinned = ? WHERE id = ?",
                (int(pinned), checkpoint_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(checkpoint_id)
        return self.checkpoint(checkpoint_id)

    def set_checkpoint_actor_path(
        self, checkpoint_id: str, actor_path: str
    ) -> dict[str, Any]:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE checkpoints SET actor_path = ? WHERE id = ?",
                (actor_path, checkpoint_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(checkpoint_id)
        return self.checkpoint(checkpoint_id)

    def create_arena_job(
        self,
        *,
        model_a: str,
        model_b: str,
        config: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a queued bounded arena comparison."""

        job_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._connect() as db:
            checkpoint = db.execute(
                "SELECT run_id FROM checkpoints WHERE id = ?", (model_a,)
            ).fetchone()
            run_id = str(checkpoint["run_id"]) if checkpoint is not None else None
            promotion_tier = str(
                config.get(
                    "promotion_tier",
                    "full" if config.get("automatic_promotion") else "diagnostic",
                )
            )
            trainer_scheduled = int(
                bool(config.get("trainer_scheduled") or config.get("automatic_promotion"))
            )
            db.execute(
                """INSERT INTO arena_jobs(
                    id, status, model_a, model_b, run_id, promotion_tier,
                    trainer_scheduled, config_json, result_json,
                    created_at, updated_at
                ) VALUES(?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    model_a,
                    model_b,
                    run_id,
                    promotion_tier,
                    trainer_scheduled,
                    json.dumps(config, separators=(",", ":")),
                    json.dumps(result or {}, separators=(",", ":")),
                    now,
                    now,
                ),
            )
        return self.arena_job(job_id)

    def arena_job(self, job_id: str, *, include_internal: bool = False) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM arena_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._decode_arena_job(row, include_internal=include_internal)

    def arena_jobs(
        self,
        *,
        limit: int = 50,
        include_internal: bool = False,
        run_id: str | None = None,
        statuses: Iterable[str] | None = None,
        promotion_tier: str | None = None,
        trainer_scheduled: bool | None = None,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        predicates: list[str] = []
        values: list[Any] = []
        if run_id is not None:
            predicates.append("run_id = ?")
            values.append(run_id)
        if statuses is not None:
            selected_statuses = tuple(dict.fromkeys(str(status) for status in statuses))
            if not selected_statuses:
                return []
            allowed = {"queued", "running", "complete", "failed", "cancelled"}
            if any(status not in allowed for status in selected_statuses):
                raise ValueError("unsupported arena status filter")
            predicates.append(f"status IN ({','.join('?' for _ in selected_statuses)})")
            values.extend(selected_statuses)
        if promotion_tier is not None:
            predicates.append("promotion_tier = ?")
            values.append(promotion_tier)
        if trainer_scheduled is not None:
            predicates.append("trainer_scheduled = ?")
            values.append(int(trainer_scheduled))
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        query = f"SELECT * FROM arena_jobs{where} ORDER BY created_at DESC LIMIT ?"
        values.append(int(limit))
        with self._connect() as db:
            rows = db.execute(query, values).fetchall()
        return [self._decode_arena_job(row, include_internal=include_internal) for row in rows]

    @staticmethod
    def _decode_arena_job(row: sqlite3.Row, *, include_internal: bool = False) -> dict[str, Any]:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        result = json.loads(item.pop("result_json"))
        if not include_internal:
            result = {key: value for key, value in result.items() if not key.startswith("_")}
        item["result"] = result
        return item

    def update_arena_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        allowed_statuses = {"queued", "running", "complete", "failed", "cancelled"}
        if status is not None and status not in allowed_statuses:
            raise ValueError(f"unsupported arena status: {status}")
        fields: dict[str, Any] = {"updated_at": time.time()}
        if status is not None:
            fields["status"] = status
        if result is not None:
            fields["result_json"] = json.dumps(result, separators=(",", ":"))
        if error is not None or status in {"running", "complete"}:
            fields["error"] = error
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = [*fields.values(), job_id]
        with self._connect() as db:
            cursor = db.execute(f"UPDATE arena_jobs SET {assignments} WHERE id = ?", values)
            if cursor.rowcount != 1:
                raise KeyError(job_id)
        return self.arena_job(job_id)

    # Verbose aliases make connector/API code self-documenting while preserving
    # the concise query names used elsewhere in Store.
    get_arena_job = arena_job
    list_arena_jobs = arena_jobs

    def delete_arena_job(self, job_id: str) -> None:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM arena_jobs WHERE id = ?", (job_id,))
            if cursor.rowcount != 1:
                raise KeyError(job_id)

    def event(
        self,
        run_id: str | None,
        kind: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO audit_events(run_id, created_at, kind, message, payload_json) VALUES(?, ?, ?, ?, ?)",
                (
                    run_id,
                    time.time(),
                    kind,
                    message,
                    json.dumps(payload or {}, separators=(",", ":")),
                ),
            )

    def events(self, run_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if run_id:
            query = "SELECT * FROM audit_events WHERE run_id = ? ORDER BY id DESC LIMIT ?"
            args: Iterable[Any] = (run_id, limit)
        else:
            query = "SELECT * FROM audit_events ORDER BY id DESC LIMIT ?"
            args = (limit,)
        with self._connect() as db:
            rows = db.execute(query, tuple(args)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result
