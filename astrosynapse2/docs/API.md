# Local API

The FastAPI control server binds to `127.0.0.1:8765` by default. It has no cloud telemetry and no authentication because it is loopback-only. Do not bind it to a public interface without adding authentication and authorization.

All routes below are rooted at `/api`.

## System and runs

- `GET /health` — service and active-run ID.
- `GET /system` — CPU, memory, platform, and accelerator summary.
- `GET /presets` — validated `astro3_m4`, `m4_24h` compatibility, and `quick` configurations.
- `GET /runs` — recent persisted runs.
- `POST /runs` — create a run with `{preset, name?, overrides?, start?}`.
- `GET /runs/{id}` — persisted record, in-memory live snapshot, and latest metric.
- `PATCH /runs/{id}/config` — apply `{changes: {...}}` containing only safe live fields.
- `POST /runs/{id}/start` — start a ready, stopped, interrupted, failed, or complete run.
- `POST /runs/{id}/pause` — request a durable checkpoint followed by pause at the next actor-batch boundary, or cancel trainer evaluation at the next game boundary. For Astro3, restart-boundary checkpoints stream the complete stratified outcome replay buffer to disk in addition to model, optimizer, counters, league, and RNG state. The run reports `paused` only after persistence succeeds.
- `POST /runs/{id}/resume` — resume a paused run.
- `POST /runs/{id}/stop` — request the same restart-safe full checkpoint and then stop.
- `POST /runs/{id}/checkpoint` — request a checkpoint at the next boundary, including while already paused.
- `GET /runs/{id}/metrics?after={sequence}&limit={n}` — reconnectable time series.
- `GET /runs/{id}/events?limit={n}` — persisted audit trail.
- `GET /events?run_id={id}&after={sequence}` — Server-Sent Events metric stream with one-second keepalives. The dashboard currently uses incremental one-second polling so status, models, audit events, and metrics arrive together.

Configuration patches are validated against the full recipe. Architecture, training generation, encoder, replay allocation, process count, and learner batch shape require a new run. Duration, actor microtask shape, search fraction/width/rollouts/horizon, governor cadence, opponent mix, checkpoint/canary/evaluation intervals, pair budgets, and early-acceptance settings can change safely between batches.

## Models

- `GET /models?run_id={id}` — checkpoint lineage, evaluation, champion, pin, and artifact-availability metadata. `artifact_state`, `model_available`, `actor_available`, `playable`, and `actor_downloadable` distinguish retained history from files still present.
- `PATCH /models/{id}` with `{pinned: true|false}` — persist a registry pin.
- `GET /models/{id}/actor` — download the portable compressed NumPy actor.

Safetensor paths are intentionally not exposed as arbitrary file-download endpoints. The actor endpoint resolves only a checkpoint already recorded in SQLite and returns `409` with a retention-specific message when its actor artifact was pruned.

## Arena

- `POST /arena` — create a manual paired job with `{model_a, model_b, pairs?, seed?, max_turns?, max_actions_per_turn?, confidence?, minimum_promotion_pairs?}`.
- `GET /arena?limit={n}&run_id={run}` — recent persistent jobs, optionally filtered to jobs involving that run's checkpoints.
- `GET /arena/{id}` — live or completed result.

Model references are checkpoint IDs or baseline names such as `baseline:balanced`, `baseline:economy`, and `baseline:aggressive`. Pair count defaults to 5,000 and is bounded at 20,000.

Every public/manual arena job is hard-coded to `automatic_promotion=false`, even if a client invents another field. Only the internal trainer can create an automatic job. The arena layer independently rechecks that job's immutable tier/pair contract, completion state, distribution-free paired Hoeffding bounds, truncation eligibility, and current champion before atomically changing champion state. Astro5 can use fixed geometric early-acceptance looks beginning at 1,000 pairs; the one-sided bound is Bonferroni-corrected across every planned look and scores truncations as candidate losses. If that stricter proof does not clear the threshold, the job continues to the ordinary 5,000-pair gate.

Trainer cadence and plateau state count only complete, current evidence. A truncated arena can promote only after every truncated game is conservatively rescored as a candidate loss and the adjusted paired confidence interval still clears the promotion threshold; otherwise it remains retryable and does not create a false plateau. Natural training completion enters `finalizing_evaluation`, waits for that run's trainer job rather than globally draining every arena, and checks the newest due checkpoint before reporting complete. A trainer job can queue behind an occupied evaluator slot. Pause or stop cancels that one automatic arena at the next game boundary, or immediately before it starts if still queued, without cancelling the unrelated job.

## Card-choice Elo probes

- `POST /card-analysis` — queue a probe with `{model_id, kind: "scrap"|"acquire", games?, seed?, max_turns?, max_actions_per_turn?}`. The GUI always requests 1,000 games.
- `GET /card-analysis?limit={n}&model_id={checkpoint}&run_id={run}` — recent process-local jobs, optionally filtered to one checkpoint or run. The dashboard uses this to reconnect to an active job after a browser refresh.
- `GET /card-analysis/{id}` — live progress or the completed card leaderboard.

The selected checkpoint plays both seats using its greedy mean-head deployment policy. An acquire choice rates the selected purchase or free-acquire card against every other card legal in that decision. A scrap choice rates a selected hand/discard card against every other card legal in that decision; hand and discard evidence is combined into one score per card, and decline is not a card alternative. The analyzer groups events by player turn and rejects the complete turn unless exactly one card was acquired or exactly one card was scrapped; standard and free acquisitions both count, and hand/discard plus in-play scrap-for-ability actions all count toward the scrap filter. Acquire Elo uses the original multinomial/Plackett-Luce update and is normalized so Explorer is `2.0`; Scrap Elo uses the original pairwise update.

Completed text and JSON reports are written under `data/analysis/`. Job progress itself is process-local; restarting the backend clears the in-memory job list but does not remove completed reports.

## Human games

- `GET /games` — active/recent in-memory game sessions.
- `POST /games` — create a game with `{model_id?, human_starts?, seed?}`. Omit `model_id` for the balanced baseline.
- `GET /games/{id}` — visible immutable state, pending legal actions, model values when available, result, and action log.
- `POST /games/{id}/choice` — submit `{action_id}` for the pending decision.

The server validates the submitted action against the exact pending legal-action tuple. Checkpoint games attach `model_value` and `model_recommended` to each human legal option; baseline games return null model values rather than fabricated scores.

## Persistence and reconnects

SQLite runs in WAL mode. Browser ownership is never used as a liveness signal. Runs, metrics, checkpoints, arena progress/results, and audit events survive browser reconnects. Completed arena pairs are retained and a clean backend restart requeues unfinished jobs. Ordinary Astro3 checkpoints persist optimizer state, exact counters/elapsed time, RNG states, league state, and a bounded recent replay journal. Pause and Stop boundaries persist the complete stratified replay buffer using a streamed archive that avoids a second full-buffer RAM allocation. The run detail and metrics report the actual recovery coverage. Human game sessions and active card-analysis jobs are intentionally process-local and are lost if the backend exits; completed card-analysis report files remain on disk.
