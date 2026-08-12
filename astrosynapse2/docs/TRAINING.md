# Training and evaluation

Astrosynapse has two explicit training contracts. Astro3 is recommended for new work. Astro2 remains available so legacy learner settings and persisted checkpoints remain interpretable.

| Capability | Astro3 (`astro3_m4`) | Astro2 compatibility (`m4_24h`) |
|---|---|---|
| Behavior policy | Exploratory heads plus direct deployment-policy data, league/baselines | Accepted champion |
| Rejected candidate | Learner continues; champion stays deployed | Learner rolls back |
| Outcome target | Terminal Monte Carlo | 60% terminal / 40% next-decision max |
| Tactical preference loss | Disabled | Enabled |
| Exploration schedule | 0.15 → 0.05 before the adaptive multiplier | 0.20 → 0.025, then scaled to an applied 0.020 → 0.0025 |
| Exploration support | Every eligible action | Current head's top 3 |
| Bootstrap heads | 5, lower overlap, fixed behavior-time perturbations | 3, high overlap, no perturbations |
| Replay | Natural profile with family-stratification correction | Rare-family-balanced, uncorrected |
| Encoder | v2 relational | v1 flat |
| Recovery | Optimizer, counters, RNG, league, bounded replay | Weights and persisted counters |

The compatibility mode preserves the legacy generation-2 learner configuration and checkpoint decoding, but it is not a bit-for-bit historical simulator: corrected shared engine, heuristic-baseline, evaluator, retention, and control-lifecycle behavior still applies. It is not the recommended route out of the measured plateau. See the [forensic analysis](PLATEAU_ANALYSIS_AND_ASTROSYNAPSE3.md) for evidence and the staged research plan.

## What Astro3 learns

For each non-forced action chosen by a player, replay stores the acting player's terminal result:

```text
win = 1.0
loss = 0.0
genuine completed draw = 0.5
safety truncation = no replay target
```

The Astro3 learner minimizes weighted binary cross-entropy on the selected action. Terminal Monte Carlo is deliberately simple: it has high variance and remains behavior-policy dependent, but it does not bootstrap the current policy's unsupported action estimates back into its own target.

Astro2's mixed next-decision target is still decoded for old runs. New Astro3 runs set `terminal_target_weight=1`, `use_bootstrap_targets=false`, and `tactical_preference_training=false`.

The engine's rules-level dominance mask is separate from the loss. It hides `END_TURN` while a card must still be played, positive combat can legally be spent, or a positive base can be activated. Acquisitions and in-play scrap abilities remain optional. Crucially, declining an optional scrap is not globally trained as an inferior `END_TURN` action.

## Information-state representation

Both encoders expose only information available to the acting player. They retain per-card counts for hand, unknown-order deck bags, discard, in-play cards, known top cards, the opponent's inferable hidden pool, trade row, remaining trade-deck multiset, scrap heap, and Explorer supply. Public resources, bases, effects, discard pressure, turn state, and starting seat are included.

Astro2 v1 uses:

- 1,292 state features;
- 203 semantic action features;
- 8 stable decision families.

Astro3 v2 preserves the v1 prefix and adds 16 action/state relational features. These include same-faction counts and ally potential, remaining trade, target-base defense and combat ratio, opponent known-top resources, lethal/breakpoint context, and the cost/horizon context of optional scrap. Observation-level relation context is computed once per decision, then combined with each candidate action.

Only genuinely ordered known-top slots are positional. Shuffled hand, market, deck, and option indices are excluded. Human play and old actors select the encoder version recorded in their model specification.

## Model and exploration

The default learner has separate state and action trunks, pre-normalized residual MLP blocks, a fused trunk, a family-specific output bank, and multiple action-value heads. A spatial convolution is not appropriate for unordered cards. A future entity-attention model is staged separately rather than being mixed into the policy-loop correction.

Astro3 normally samples one behavior head for a trajectory. A deterministic random projection specific to that head is added only to the nondeployment behavior policy's scores, affecting both greedy ranking and the epsilon candidate ranking. It is absent from the learned MLX logits, fitted loss, and deployed policy, so this is a behavior-time perturbation inspired by randomized-prior exploration—not a fully anchored randomized-prior function. Replay masks have lower overlap than Astro2, reducing the chance that every head sees every example. Epsilon exploration chooses among every deployment-eligible legal action (`exploration_top_k=0`). To reduce exploration/deployment mismatch, `deployment_policy_selfplay_fraction=0.20` makes a configured 20% of current-v-current self-play use the exact deployed policy: greedy mean-head logits with no behavior perturbation. Because current self-play is 60% of the configured schedule, this stream has a 12% overall scheduled share. Those trajectories still receive a sampled training head and required-head bootstrap mask, so they remain valid ensemble-training examples with explicit bootstrap ownership.

The GUI reports:

- scheduled and effective epsilon;
- mean ensemble probability dispersion;
- action-argmax disagreement over fixed diagnostic states;
- a head-collapse warning after sufficient updates;
- the current plateau exploration multiplier.

The fixed behavior perturbations and replay masks are exploration mechanisms, not uncertainty guarantees. A warning should trigger diagnosis, not automatic promotion or rejection by itself.

## Actor–learner pipeline

The parent process owns MLX and the Metal learner. Spawned workers use NumPy actors and never initialize MLX:

```text
current learner export
        |
        +--> CPU games versus learner / league / baseline
        |                    |
        |                    +--> compact chosen-action trajectories
        |
        +--> Metal replay updates
                             |
                             +--> immutable checkpoint + diagnostics
```

The Astro3 default opponent mix is:

- 60% current-learner self-play, of which 20% (12% of all scheduled games) directly follows the perturbation-free mean-head deployment policy;
- 30% accepted champion-history league and validated frozen anchors from other runs;
- 10% balanced/economy/aggressive baseline anchors.

For Astro3 only, other runs' current champions and explicitly pinned checkpoints are eligible frozen anchors. Their recorded encoder version and tensor dimensions are validated with a real forward pass before a worker can receive them. Astro2 compatibility runs remain isolated to their original per-run league semantics.

The champion is still the only automatically deployable model. Rejecting an arena candidate leaves that champion unchanged, but it no longer rewinds the learner. This keeps evaluation safety distinct from exploration and optimization.

Before ordinary self-play, corrected heuristic games populate replay until the warmup/curriculum requirements are met. Optional scrap now weighs immediate tactical value against retention cost; it no longer receives an unconditional bonus above ending the decision.

## Replay

Replay is preallocated and divided by stable decision family. The Astro3 natural profile follows the observed distribution much more closely than the legacy profile. If family stratification differs from cumulative behavior/write frequency, the batch receives normalized family-level weights based on `p_behavior / q_sample`. The controlled recent partition remains an intentional recency bias; it is reported but is not presented as fully importance-corrected.

Telemetry reports:

- family occupancy and cumulative writes;
- requested and realized sample shares;
- sample-to-write ratios;
- recent-partition use;
- importance-weight summaries and effective sample size when available;
- journal coverage on resume.

The replay buffer still contains chosen actions rather than full counterfactual legal-action targets. Search/reanalysis is the planned solution; family-level importance weighting corrects family-stratification bias, not recency bias or chosen-action confounding.

Safety-truncated trajectories and their preference pairs never enter replay. They are counted separately from genuine draws and contribute no league score, preventing a losing policy from improving its target by deliberately stalling.

## Recommended M4 / 16 GB recipe

| Setting | Astro3 default |
|---|---:|
| Duration | 1,440 minutes |
| CPU actors | up to 8 |
| Games per actor batch | 16 |
| Model width / residual blocks | 192 / 3 |
| Bootstrap heads | 5 |
| Learner batch | 2,048 |
| Learning rate | 2e-4, decaying cosine restarts, floor 4e-5 |
| Restart period / peak decay | 200,000 updates / 0.85 |
| Gradient clip | 5.0 |
| Updates per iteration | 32 |
| Replay capacity / warmup | 900,000 / 50,000 decisions |
| Heuristic curriculum | First 2,000 learner updates |
| Base epsilon | 0.15 → 0.05 over 2M games, all eligible actions |
| Bootstrap inclusion / behavior-perturbation scale | 0.35 / 0.25 |
| Direct deployment-policy data | 20% of current self-play (12% of scheduled games) |
| Checkpoint interval | 100,000 games |
| Mature evaluation interval | 500,000 games |
| Promotion evaluation | Adaptive, up to 5,000 paired seeds |
| Resume replay journal | Newest 100,000 decisions |

Cosine restart progress follows persisted update counts rather than wall time. Plateau response can temporarily multiply epsilon up to its configured ceiling after repeated clean automatic non-promotions. Invalid or stale evaluations do not count. The status payload always exposes both scheduled epsilon and the applied value.

The five-minute `quick` recipe uses the Astro3 contract with smaller model, replay, batch, and evaluation sizes. It proves wiring, persistence, and telemetry—not playing strength.

## Checkpoint quality gates

Before an automatic arena, a candidate runs deterministic and held-out diagnostics:

- deployment-masked tactical legality/dominance states;
- early optional high-cost scrap retention;
- game-grouped held-out BCE and Brier score;
- paired games against deterministic strategy baselines;
- all-family ensemble dispersion and argmax disagreement.

Astro2 compatibility can still require raw-logit `END_TURN` preference ordering. Astro3 gates deployed masked behavior instead; otherwise disabling the unsafe global preference loss would create an impossible raw-logit gate.

The small fixed-opponent sample and game-grouped Brier score remain checkpoint diagnostics in Astro3, but do not hard-block an arena by default: neither comparison has enough independent evidence to serve as a strength test. Generation 2 retains both legacy regression gates, and either gate can be explicitly enabled. Fixed-opponent diagnostic truncations and deterministic strategic violations hard-block that checkpoint; truncations in an arena invalidate the comparison and trigger the retry policy rather than counting as wins or losses. The paired arena is the statistical strength gate.

The strategic and ensemble diagnostic objects, pass/fail reasons, scrap rate/margin, and head statistics are stored with the immutable checkpoint and shown in Models & Arena.

## Paired arena evaluation

Each seed is run twice with the same isolated game randomness and exact seat swap. The seed pair—not two correlated games—is the statistical unit. The reported record contains overall and seat-split score, draws, truncations, throughput, job progress, and a distribution-free two-sided Hoeffding confidence interval for the bounded paired score.

Automatic promotion requires all of the following:

1. candidate and opponent are immutable checkpoint actors;
2. the trainer—not a public API caller—created the job;
3. the recorded quality gate passed;
4. the complete tier-specific required pair count finished (an early rejection is a terminal non-promotion, never an exception for promotion);
5. the paired confidence interval's lower bound exceeds 50% plus the configured margin;
6. the recorded opponent is still the current champion.

Astro3 may reject an obviously inferior candidate at predeclared geometric looks after at least the configured minimum number of pairs. Each one-sided Hoeffding upper bound uses a Bonferroni share of the configured family-wise error budget, so constant observations cannot create a false zero-width interval. Early acceptance is forbidden. A candidate that remains plausible completes the full fixed-size promotion test. Manual GUI/API arenas never promote models.

Any arena containing a safety truncation is explicitly ineligible for promotion. Truncations remain visible in the result instead of being silently counted as ordinary draws.

Only complete, clean comparisons against the still-current champion advance evaluation cadence or the adaptive plateau counter. Failed, cancelled, malformed, truncated, and stale jobs remain retryable; live training uses bounded exponential backoff and a new deterministic seed range. A tier smaller than the configured first early-rejection look disables early rejection instead of moving that look. At natural duration completion, the trainer waits for its own older arena, evaluates the newest due checkpoint, and makes at most three final scheduling attempts. It does not globally drain unrelated jobs, although its evaluation can queue behind a manual job already occupying the configured evaluator slot. Pause or stop cancels the trainer arena at the next game boundary, or immediately before it starts if still queued, without waiting for that manual job; resume reconsiders the still-due checkpoint. An explicit stop never schedules a final arena.

## Reading progress

- **Outcome BCE / Brier** — use held-out, game-grouped values. Training values on reused replay can look excellent while policy strength regresses.
- **Arena paired interval** — primary local evidence of improvement against one frozen opponent.
- **Opponent matrix** — necessary evidence against non-transitive regressions; one champion score is insufficient.
- **Strategic gates** — direct regression tests for known failures such as premium-card scrapping.
- **Head disagreement** — whether bootstrapped exploration retains meaningful policy diversity.
- **Effective epsilon** — the probability actually applied after schedule and plateau response.
- **Rollout mix** — realized learner/league/baseline games, not merely configured fractions.
- **Replay ratios / weights** — whether shared-network updates represent the behavior distribution.
- **Truncation rate** — sustained nonzero values invalidate ordinary win/loss interpretation.
- **Games/s and decisions/s, plus cumulative updates** — use all three views. A higher game count is not automatically more learning.
- **CPU/RAM/Metal** — resource pressure and learner utilization.

Do not select a model from training loss, raw game count, or a short unpaired match.

## Pause, checkpoint, stop, and recovery

Pause and stop take effect at safe actor-batch boundaries during learning and at a game boundary during trainer evaluation. Checkpoint requests are serviced at the next learner boundary or between final-evaluation attempts:

- **Checkpoint** queues a complete immutable learner boundary.
- **Pause** queues that checkpoint first and reports `paused` only after it is durable.
- A new checkpoint request can be serviced while already paused.
- **Stop** writes a final checkpoint when learner state advanced, then closes worker resources.

Natural duration completion enters the visible `finalizing_evaluation` phase and does not report `complete` until the newest due trainer comparison is resolved or its bounded retries are exhausted. Pause and stop remain responsive there: only the trainer-owned arena is cancelled, including when it is queued behind another evaluator job.

Astro3 checkpoint sidecars include model weights/specification, AdamW state, exact totals, seed cursor, active elapsed time, rollout and replay RNG states, league order/statistics, and a compressed bounded replay journal. Time spent paused is excluded even when a manual checkpoint or stop happens during the pause.

On restart, versioned complete training-state payloads are authoritative; partial legacy payloads cannot zero mature counters. If a newer Astro3 boundary is missing or has an unreadable optimizer/replay artifact, resume falls back to the previous complete boundary. If no complete boundary exists, it keeps the newest usable weights, restarts optimizer/replay deliberately, and emits explicit degraded-resume telemetry. Optimizer warmup origin is persisted. Legacy rollback checkpoints retain mature counters instead of rewinding game ranges while keeping an advanced update schedule. Arena work is recovered by the owning manager rather than being left as an orphaned `running` job.

Because only the newest configured replay slice is saved, an Astro3 resume is operationally durable but not bit-identical to a process that kept all 900,000 items resident. The GUI reports journal coverage rather than calling it full replay recovery.

After durable checkpoint and handled-evaluation boundaries, `keep_checkpoints` bounds ordinary artifact files while protecting champions, pins, promoted historical league members, active arena inputs, and the latest complete resume boundary. Pruned rows retain lineage, diagnostics, and arena history in SQLite and are labeled unavailable in the API and dashboard. The exact safety contract is documented in [checkpoint artifact retention](CHECKPOINT_RETENTION.md).

## Experimental discipline

Use at least three fresh seeds for major changes. Preserve common held-out scenario banks and arena seed lists. Compare against:

- the best Astro2 champion;
- several strategically diverse historical champions;
- every deterministic baseline;
- fresh exploiters when those are implemented.

Plot score and uncertainty against wall time, decisions, and updates. Require the improvement to repeat across seeds. The recommended release thresholds and future entity/search/population/native-engine stages are in the [plateau report](PLATEAU_ANALYSIS_AND_ASTROSYNAPSE3.md#promotion-and-release-gates).

## Current limitations

- The corrected system has unit/integration and deterministic-strategy validation, not a completed multi-million-game Astro3 strength result.
- Replay still lacks full legal-action counterfactual reanalysis.
- The model is a relational residual MLP, not yet an entity transformer or recurrent public-belief model.
- Population play uses accepted history, not yet dedicated main and league exploiters with a full payoff solver.
- Fixed random projections perturb behavior only; fitted prior terms or genuinely independent ensemble networks remain future experiments.
- The Python engine is correct-first. Native simulation is staged for search after differential parity testing.
