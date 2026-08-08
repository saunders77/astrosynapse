# Local API

The FastAPI control server binds to `127.0.0.1:8765` by default. It has no cloud telemetry and no authentication because it is loopback-only. Do not bind it to a public interface without adding authentication and authorization.

All routes below are rooted at `/api`.

## System and runs

- `GET /health` — service and active-run ID.
- `GET /system` — CPU, memory, platform, and accelerator summary.
- `GET /presets` — validated `m4_24h` and `quick` configurations.
- `GET /runs` — recent persisted runs.
- `POST /runs` — create a run with `{preset, name?, overrides?, start?}`.
- `GET /runs/{id}` — persisted record, in-memory live snapshot, and latest metric.
- `PATCH /runs/{id}/config` — apply `{changes: {...}}` containing only safe live fields.
- `POST /runs/{id}/start` — start a ready/stopped/interrupted run.
- `POST /runs/{id}/pause` — request pause at the next actor-batch boundary.
- `POST /runs/{id}/resume` — resume a paused run.
- `POST /runs/{id}/stop` — request a final checkpoint and safe stop.
- `POST /runs/{id}/checkpoint` — request a checkpoint at the next boundary.
- `GET /runs/{id}/metrics?after={sequence}&limit={n}` — reconnectable time series.
- `GET /runs/{id}/events?limit={n}` — persisted audit trail.
- `GET /events?run_id={id}&after={sequence}` — Server-Sent Events metric stream with one-second keepalives. The dashboard currently uses incremental one-second polling so status, models, audit events, and metrics arrive together.

Configuration patches are validated against the full recipe. Architecture, replay allocation, process count, and batch shape require a new run. Duration, update ratio, exploration floor, opponent mix, checkpoint/evaluation intervals, pair budget, and telemetry cadence can change safely between batches.

## Models

- `GET /models?run_id={id}` — checkpoint lineage, paths, evaluation, champion, and pin metadata.
- `PATCH /models/{id}` with `{pinned: true|false}` — persist a registry pin.
- `GET /models/{id}/actor` — download the portable compressed NumPy actor.

Safetensor paths are intentionally not exposed as arbitrary file-download endpoints. The actor endpoint resolves only a checkpoint already recorded in SQLite.

## Arena

- `POST /arena` — create a manual paired job with `{model_a, model_b, pairs?, seed?, max_turns?, max_actions_per_turn?, confidence?, minimum_promotion_pairs?}`.
- `GET /arena?limit={n}&run_id={run}` — recent persistent jobs, optionally filtered to jobs involving that run's checkpoints.
- `GET /arena/{id}` — live or completed result.

Model references are checkpoint IDs or baseline names such as `baseline:balanced`, `baseline:economy`, and `baseline:aggressive`. Pair count defaults to 5,000 and is bounded at 20,000.

Every public/manual arena job is hard-coded to `automatic_promotion=false`, even if a client invents another field. Only the internal trainer can create an automatic job, and the arena layer independently rechecks the full 5,000-pair minimum and paired lower confidence bound before atomically changing champion state.

## Human games

- `GET /games` — active/recent in-memory game sessions.
- `POST /games` — create a game with `{model_id?, human_starts?, seed?}`. Omit `model_id` for the balanced baseline.
- `GET /games/{id}` — visible immutable state, pending legal actions, model values when available, result, and action log.
- `POST /games/{id}/choice` — submit `{action_id}` for the pending decision.

The server validates the submitted action against the exact pending legal-action tuple. Checkpoint games attach `model_value` and `model_recommended` to each human legal option; baseline games return null model values rather than fabricated scores.

## Persistence and reconnects

SQLite runs in WAL mode. Browser ownership is never used as a liveness signal. Runs, metrics, checkpoints, arena progress/results, and audit events survive browser reconnects. Completed arena pairs are retained and a clean backend restart requeues unfinished jobs. Human game sessions are intentionally process-local and are lost if the backend exits.
