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

Model references are checkpoint IDs or baseline names such as `baseline:balanced`, `baseline:economy`, and `baseline:aggressive`. Pair count defaults to and is capped at 2,000.

Every public/manual arena job is hard-coded to `automatic_promotion=false`, even if a client invents another field. Only the internal trainer can create an automatic job. The arena layer independently rechecks that job's immutable tier/pair contract, completion state, distribution-free paired Hoeffding bounds, truncation eligibility, and current champion before atomically changing champion state. Astro5 can use fixed geometric early-acceptance looks beginning at 1,000 pairs; the one-sided bound is Bonferroni-corrected across every planned look and scores truncations as candidate losses. If that stricter proof does not clear the threshold, the job continues to the ordinary 2,000-pair gate.

Trainer cadence and plateau state count only complete, current evidence. A truncated arena can promote only after every truncated game is conservatively rescored as a candidate loss and the adjusted paired confidence interval still clears the promotion threshold; otherwise it remains retryable and does not create a false plateau. Natural training completion enters `finalizing_evaluation`, waits for that run's trainer job rather than globally draining every arena, and checks the newest due checkpoint before reporting complete. A trainer job can queue behind an occupied evaluator slot. Pause or stop cancels that one automatic arena at the next game boundary, or immediately before it starts if still queued, without cancelling the unrelated job.

## Card-choice Elo probes

- `POST /card-analysis` — queue a probe with `{model_id, kind: "scrap"|"acquire", games?, seed?, max_turns?, max_actions_per_turn?}`. The GUI always requests 1,000 games.
- `GET /card-analysis?limit={n}&model_id={checkpoint}&run_id={run}` — recent process-local jobs, optionally filtered to one checkpoint or run. The dashboard uses this to reconnect to an active job after a browser refresh.
- `GET /card-analysis/{id}` — live progress or the completed card leaderboard.

The selected checkpoint plays both seats using its greedy mean-head deployment policy. An acquire choice rates the selected purchase or free-acquire card against every other card legal in that decision. A scrap choice rates a selected hand/discard card against every other card legal in that decision; hand and discard evidence is combined into one score per card, and decline is not a card alternative. The analyzer groups events by player turn and rejects the complete turn unless exactly one card was acquired or exactly one card was scrapped; standard and free acquisitions both count, and hand/discard plus in-play scrap-for-ability actions all count toward the scrap filter. Acquire Elo uses the original multinomial/Plackett-Luce update and is normalized so Explorer is `2.0`; Scrap Elo uses the original pairwise update.

Completed text and JSON reports are written under `data/analysis/`. Job progress itself is process-local; restarting the backend clears the in-memory job list but does not remove completed reports.

## Manual checkpoint advisor

- `GET /cards` — the 49 canonical Core Set card definitions, ordered by stable `card_id`. A React client may put these objects directly into advisor zones; the advisor trusts their ID and rehydrates the canonical server definition.
- `POST /advisor/evaluate` — score one manually tracked checkpoint decision without creating or mutating a server game.

The request body is:

```ts
type CardRef = {
  card_id: number; // 0..48; the other fields from GET /cards may remain present
};

type InPlayRef = {
  card: CardRef;
  activated: boolean;
  ally_triggered: boolean;
  copied_from_stealth_needle?: boolean;
};

type AdvisorObservation = {
  version: 2;
  player_id: 0 | 1;
  active_player: 0 | 1;       // must equal player_id
  starting_player: 0 | 1;
  is_starting_player: boolean;
  turn: number;
  action_number: number;
  own_authority: number;
  opponent_authority: number;
  opponent_pending_discard: number;
  combat: number;
  trade: number;
  pending_discard: number;
  hand: CardRef[];
  own_deck_count: number;      // own_deck.length + own_known_top.length
  own_deck: CardRef[];         // scrambled/unknown portion only
  own_known_top: CardRef[];    // top first
  own_discard: CardRef[];
  own_in_play: InPlayRef[];
  opponent_hand_count: number;
  opponent_known_hand: CardRef[];
  opponent_hidden: CardRef[];  // unknown hand + unknown deck, scrambled together
  opponent_deck_count: number;
  opponent_known_top: CardRef[];
  opponent_discard: CardRef[];
  opponent_in_play: InPlayRef[];
  trade_row: [CardRef, CardRef, CardRef, CardRef, CardRef];
  trade_deck_count: number;
  trade_deck: CardRef[];       // remaining trade-deck multiset, unordered
  explorers_remaining: number;
  explorer_supply: CardRef[];  // exactly that many Explorer objects
  scrap_heap: CardRef[];
  next_ship_to_top: boolean;
  blob_cards_played: number;
  all_allied: boolean;         // true exactly when own_in_play contains Mech World
  fleet_active: boolean;       // true exactly when own_in_play contains Fleet HQ
};

type SemanticAction = {
  kind: "play_card" | "activate_base" | "scrap_for_ability" |
        "attack_base" | "attack_player" | "acquire" | "end_turn" |
        "discard_card" | "scrap_card" | "choose_mode" | "copy_ship" |
        "destroy_base" | "scrap_trade_row" | "free_acquire" | "decline";
  card_id?: number;            // -1 means no source card
  target_card_id?: number;     // -1 means no target card
  ability?: string;
  source_zone?: string;
  amount?: number;
  amount2?: number;
};

type AdvisorRequest = {
  model_id: string;
  observation: AdvisorObservation;
  decision?: {
    family: "main" | "discard" | "scrap" | "ability_mode" |
            "copy_ship" | "destroy_base" | "scrap_trade_row" | "free_acquire";
    prompt?: string;
    actions?: SemanticAction[];
  };
};
```

Omit `decision` or set its family to `main` to have the server generate the same semantic main-phase legal set as the game engine. If a client includes actions with a `main` decision, they are ignored in favor of the server-generated legal set; its prompt is still preserved. Main generation requires `pending_discard: 0`; if a forced discard or another nested card choice is unresolved, the client supplies that decision instead. For a nested choice—discard, scrap, ability mode, copy, destroy, trade-row scrap, or free acquire—the client supplies the exact legal semantic actions shown by the physical card. Repeated physical copies with the same semantic action collapse to one checkpoint option, just as they do in the engine. The advisor scores them but never executes them.

Example nested decision fragment:

```json
{
  "family": "scrap",
  "prompt": "Missile Bot: scrap from hand or discard",
  "actions": [
    {"kind": "scrap_card", "card_id": 0, "source_zone": "hand"},
    {"kind": "scrap_card", "card_id": 1, "source_zone": "discard"},
    {"kind": "decline", "card_id": 20, "ability": "scrap_any"}
  ]
}
```

The response is deliberately flat for the action console:

```json
{
  "family": "main",
  "prompt": "Main phase",
  "score_semantics": "policy_probability",
  "expected_win_rate": 0.584,
  "actions": [
    {
      "id": 0,
      "label": "play card Scout",
      "kind": "play_card",
      "card_id": 0,
      "target_card_id": -1,
      "ability": "",
      "source_zone": "hand",
      "amount": 0,
      "amount2": 0,
      "model_value": 0.713,
      "model_recommended": true
    }
  ]
}
```

Astro4/Astro5 objective-version-2 checkpoints return normalized legal-action policy shares in `model_value`; `expected_win_rate` is the separate mean state-value estimate. Legacy outcome-head checkpoints return `score_semantics: "win_outcome"`, independent per-action outcome estimates, and `expected_win_rate: null` because they do not expose a separate state-value head.

Undefined cards are never silently encoded as blank slots. Missing/negative card IDs, a null/undefined trade-row card, inconsistent public zone counts, duplicate semantic nested actions, or a non-main decision without actions returns `422`. An unknown checkpoint returns `404`; a checkpoint whose actor was pruned or cannot be loaded returns `409`.

## Human games

- `GET /games` — active/recent in-memory game sessions.
- `POST /games` — create a game with `{model_id?, human_starts?, seed?}`. Omit `model_id` for the balanced baseline.
- `GET /games/{id}` — visible immutable state, pending legal actions, model values when available, result, and action log.
- `POST /games/{id}/choice` — submit `{action_id}` for the pending decision.

The server validates the submitted action against the exact pending legal-action tuple. Checkpoint games attach `model_value` and `model_recommended` to each human legal option; baseline games return null model values rather than fabricated scores.

## Persistence and reconnects

SQLite runs in WAL mode. Browser ownership is never used as a liveness signal. Runs, metrics, checkpoints, arena progress/results, and audit events survive browser reconnects. Completed arena pairs are retained and a clean backend restart requeues unfinished jobs. Ordinary Astro3 checkpoints persist optimizer state, exact counters/elapsed time, RNG states, league state, and a bounded recent replay journal. Pause and Stop boundaries persist the complete stratified replay buffer using a streamed archive that avoids a second full-buffer RAM allocation. The run detail and metrics report the actual recovery coverage. Human game sessions and active card-analysis jobs are intentionally process-local and are lost if the backend exits; completed card-analysis report files remain on disk.
