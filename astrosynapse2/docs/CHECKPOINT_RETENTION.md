# Checkpoint artifact retention

`keep_checkpoints` now bounds ordinary checkpoint artifacts for new training
runs. Retention executes only after a complete checkpoint has been persisted or
after the trainer has handled a completed arena evaluation. It does not run on
a timer or while a checkpoint is being written.

The configured count is a floor, not permission to remove protected models.
Astrosynapse keeps all of the following regardless of age:

- the newest `keep_checkpoints` records;
- the current champion and any checkpoint still marked champion;
- pinned checkpoints;
- former promoted champions that remain part of the population league;
- both inputs to queued or running arena jobs; and
- the newest complete optimizer/replay resume boundary.

Every model snapshot is self-contained, so parent checkpoints are not needed to
load a descendant. Lineage stays in SQLite even when an old snapshot is pruned.
Evaluation results, diagnostics, metrics, arena history, and audit events are
also retained.

Only explicit model, model-metadata, actor, optimizer, and replay files recorded
for a checkpoint are eligible. The cleanup root is derived from the database
location and run ID. Before the first deletion, every candidate must resolve to
a regular, non-symbolic-link file directly inside that run's checkpoint
directory and have the expected suffix. Any unsafe reference stops the pass;
the trainer records `checkpoint_retention_failed` and does not retry with a
broader target.

Successful passes record `checkpoint_retention_completed` with confirmed-unlink
and crash-reconciliation file and byte counts. Pruned checkpoint rows carry
`evaluation.artifact_retention`, while the models API exposes `artifact_state`,
`model_available`, `actor_available`, `playable`, and `actor_downloadable`. The
dashboard keeps pruned rows visible as history but disables actor download,
arena selection, and human play for unavailable files.

Filesystem removal and SQLite cannot share one transaction, so each file uses
a durable two-step journal:

1. Before unlink, retention writes a `pending_artifacts` entry containing the
   exact path, kind, byte size, file identity, attempt count, and time. If this
   SQLite write fails, the file is not unlinked.
2. Retention revalidates the same regular file and unlinks it, then moves that
   entry into `artifact_records` with `status: removed` and
   `provenance: unlink_confirmed`. Only this confirmed path is added to
   `removed_artifacts`.

An unlink failure leaves the pending entry with `last_error` and a timestamp,
then stops the pass. A later safe pass retries it. If the process or SQLite
fails after unlink but before the confirmation write, the durable pending entry
remains. On restart, an absent file with an exact matching intent is moved to
`artifact_records` as `missing_after_pending_intent` and listed separately in
`reconciled_missing_artifacts`; it is not falsely reported as a confirmed
unlink. A present file remains pending and is revalidated before any retry.
