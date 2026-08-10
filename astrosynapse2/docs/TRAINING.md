# Training and evaluation

## What is learned

Astrosynapse 2 uses bootstrapped action values, inspired by [DouZero](https://proceedings.mlr.press/v139/zha21a.html), rather than the legacy PPO loop. For every non-forced action chosen by a player, replay retains that player's terminal result:

```text
win = 1.0
loss = 0.0
genuine draw or safety truncation = 0.5
```

For actor trajectories, the next decision's frozen-actor value is also retained. The learner uses a conservative 60% terminal / 40% next-decision target when that bootstrap is available, and the pure terminal result otherwise. This adds temporal credit assignment without discarding the unbiased game outcome anchor.

The learner minimizes masked binary cross-entropy on the selected action plus a small pairwise tactical loss. Exact base-set dominance pairs—attack, play, or activate before ending the turn—train the preferred action above `END_TURN` by a logit margin. These pairs come from rules, not card-quality guesses or authority-margin reward shaping.

Three bootstrap heads share the network. Each player samples one head and bootstrap mask for the entire game. That makes exploration coherent across a trajectory: one game follows one sampled value hypothesis instead of changing personality at every chooser call. A reduced per-decision epsilon may choose only among that head's top-scored eligible actions. Deployment averages the heads and selects greedily.

## Information-state representation

The base-set encoder currently produces:

- 1,292 state features;
- 203 semantic action features;
- 8 stable decision families;
- 3 bootstrap outcome heads by default.

State features retain exact per-card counts in public or inferable information-set zones, including hand, unknown-order deck bags, discard, in-play cards, known top cards, the opponent's inferable hidden pool, trade row, remaining trade-deck multiset, scrap heap, and Explorer supply. It also records authorities, available combat/trade, discard pressure, public continuous effects, turn state, and whether the acting player started.

Only genuinely ordered known-top slots are positional. Shuffled hand, market, deck, and option indices are excluded. Action features describe the verb, family, source and target card identities, zones, effect, and rule-relevant quantities. Forced one-option decisions are resolved by the engine and never enter inference or replay.

The engine retains every rules-legal option for human play and audits. Learned policies use a centralized dominance mask: `END_TURN` is not exposed while a card can be played, a positive base can be activated, or generated combat can legally be spent. Purchases and scrap abilities remain optional because those choices are genuinely strategic. The mask is applied before both greedy selection and exploration.

## Model

The default MLX model has:

- separate state and action input trunks;
- pre-normalized residual MLP blocks;
- a fused state/action trunk;
- a family-specific output bank;
- three bootstrapped logits per legal action.

A convolutional network is not used because Star Realms has no stable spatial grid. A large transformer is also a poor first-day trade on this hardware: the exact base-set counts already preserve card identity, while the smaller residual model permits substantially more games and replay updates in 24 hours.

## Actor–learner pipeline

The parent process owns MLX and the Metal learner. Spawned CPU workers never import MLX; they load a compact NumPy mirror of the same weights and run independent seeded games. This avoids one Metal context per worker and lets the M4 CPU and GPU operate concurrently:

```text
export immutable actor snapshot
          |
          +--> CPU worker games --------> compact float16 samples
          |
          +--> Metal replay updates from the previous data
                                      |
                                      +--> next actor snapshot
```

Each decision computes the state trunk once and reuses it across all legal actions. Worker output stays in contiguous float16 arrays through inter-process transfer and vectorized replay insertion.

Replay is preallocated and split by decision family. The default 900,000-decision outcome buffer plus the 50,000-pair tactical buffer use about 2.9 GB. Main-phase decisions receive 82% of outcome capacity and 72% of sampled updates. The remaining budget preserves explicit minimum coverage for discard, scrap, copy, destroy-base, trade-row-scrap, free-acquire, and modal decisions without allowing extremely rare families to consume equal learner time. Telemetry reports each family's write share, sample share, and sample-to-write ratio. A controlled recent fraction adapts to the latest accepted policy without discarding all older experience.

## Opponent schedule

The implemented schedule mixes trajectories from:

- 55% accepted-champion self-play;
- 30% prioritized accepted champion-history checkpoints, emphasizing opponents with a low smoothed current score;
- 15% balanced, economy, and aggressive deterministic baselines.

Only behavior-policy decisions are collected when playing a frozen or heuristic opponent. Both players are collected in champion self-play. Matchup estimates use a Beta prior to avoid overreacting to a handful of noisy games.

The accepted champion, rollout actor, and learner candidate are distinct roles. A candidate never controls ordinary self-play merely because it is the newest checkpoint. If an automatic arena rejects it while that same champion remains current, the learner is restored to the accepted champion with a fresh optimizer warmup. This prevents a failed candidate from redefining the data distribution for all later candidates.

The initial random model is saved as the lineage root. It is useful for reproducibility, not as a strength claim.

Before ordinary self-play, the replay warmup is filled with games between the balanced/economy/aggressive anchors, collecting both players' chosen actions with the same terminal win/loss targets. This demonstration curriculum lasts for 2,000 learner updates (256 in the quick preset), rather than ending as soon as one training batch fits in replay. It is not reward shaping or permanent imitation: it prevents a random untrained network from filling the first buffer with stalemates, then yields completely to the configured self-play/league mix. The zero-game random persistence checkpoint, unevaluated candidates, and rejected candidates are excluded from the opponent league.

## Recommended M4/16 GB preset

| Setting | Default |
|---|---:|
| Duration | 1,440 minutes |
| CPU actors | up to 8 |
| Games per actor batch | 16 |
| Model width / residual blocks | 192 / 3 |
| Bootstrap heads | 3 |
| Learner batch | 2,048 |
| AdamW learning rate | 3e-4 → 3e-5 over 400,000 persisted updates |
| Gradient clip | 5.0 |
| Updates per actor iteration | 32 |
| Replay capacity / warmup | 900,000 / 50,000 |
| Heuristic bootstrap | First 2,000 learner updates |
| Effective per-decision epsilon | 0.020 → 0.0025 over 1.5M games, top-3 only |
| Checkpoint interval | 100,000 games |
| Mature evaluation interval | 500,000 games |
| Promotion evaluation | Adaptive 200 → 1,000 → 5,000 paired seeds |

The learning-rate cosine follows persisted learner updates, not editable wall-clock duration. Extending a run therefore cannot rewind it to a larger learning rate. A short update warmup protects a fresh optimizer from correlated first batches and repeats when a resumed or rolled-back run recreates AdamW.

These values are defaults, not universal optima. Watch measured throughput and memory. On a 16 GB machine, reduce replay capacity or model width if macOS memory pressure becomes sustained; preserve paired evaluation size before spending that compute on a larger network.

## Automatic and manual evaluation

Before an automatic arena is scheduled, the immutable checkpoint must pass deterministic diagnostics: the raw-network tactical suite, held-out game-grouped outcome metrics, and paired games against the balanced, economy, and aggressive anchors. Any raw `END_TURN` dominance error, a held-out Brier regression, or a large fixed-opponent regression blocks the arena and is recorded in the checkpoint audit metadata. The hard action mask remains a separate deployment safeguard rather than making this learning test vacuous.

Every arena seed is run twice with the same isolated game randomness and exact model seat swap. Model-role RNG is also held stable across the pair. The seed pair—not two correlated games—is the statistical unit.

The recommended preset adapts evaluation to model maturity. Before 500,000 training games, every checkpoint is compared with 200 paired seeds. From 500,000 through 999,999 games, comparisons use 1,000 pairs and are spaced by 250,000 games (rounded up to a checkpoint boundary). Starting at 1,000,000 games, comparisons use the configured 5,000 pairs every configured 500,000 games. Early promotions are recorded as `provisional` or `development`; mature promotions are recorded as `full`. Adaptive evaluation can be disabled to use the configured interval and pair count from the beginning.

Every tier remains confidence-gated: a candidate is promoted only when its paired confidence interval's lower bound exceeds 50% (plus any configured promotion margin). A trainer-created job can promote automatically only if all of these hold:

1. both entries are immutable checkpoint actors;
2. the job was created internally as an automatic evaluation;
3. the complete pair count required by its recorded tier completed;
4. the full requested job completed;
5. the paired 95% confidence interval's lower bound is above 50%;
6. the recorded opponent is still the run's current champion when the job finishes.

If a newer evaluation already changed the champion, the older result remains useful evidence but is marked stale and cannot overwrite the newer champion.

The candidate's evaluation record, champion flags, and run champion ID are updated in one SQLite transaction. Small quick-run jobs and every job created manually in the GUI have `automatic: false` and cannot promote anything, regardless of observed win rate. Fixed job sizes are intentional: repeatedly checking an ordinary confidence interval and stopping as soon as it crosses the threshold would overstate confidence without a sequential-testing correction.

Arena output reports overall and seat-split score, Wilson and paired intervals, draws, truncations, games/s, ETA, and recent paired seeds. Elo difference is only a display transform of the measured score; it is not treated as an absolute global rating.

## Reading diagnostics

- **Outcome BCE** — masked blended-outcome cross-entropy. Lower is better only on a comparable held-out distribution.
- **Brier score** — squared error of the head-averaged win probability. It measures calibration as well as discrimination.
- **Explained variance** — how much target variation the current value estimate explains. Negative values are common during early noisy learning.
- **Bootstrap uncertainty** — average disagreement among heads. Falling uncertainty can mean convergence, but should be checked against arena strength and policy diversity.
- **Tactical preference loss/accuracy** — whether exact dominance pairs have the required ordering. The hard policy mask remains authoritative even while raw logits learn the invariant.
- **Held-out diagnostics** — game-grouped BCE/Brier and fixed-opponent scores computed from immutable seeds outside learner replay.
- **Replay utilization/families** — fill, write share, sample share, and sample-to-write ratio. Main decisions should dominate learner work while every rare ring still receives samples.
- **Games/s and decisions/s** — actor throughput measured over the latest telemetry interval.
- **Truncation rate** — safety-capped games. Sustained nonzero truncation should be investigated before trusting results.
- **Seat split and paired interval** — the primary evidence for model selection in this high-variance game.
- **CPU/RAM/Metal** — system memory, process RSS, active/cache/peak Metal allocation, and learner device.

Do not select a model because training BCE is lower or because an unpaired short match was lucky. Select it because it wins a sufficiently large paired evaluation against frozen opponents and does not regress badly against the broader league.

## Checkpoints and resume behavior

Checkpoints contain portable MLX safetensors, an architecture JSON sidecar, and a compressed NumPy actor. SQLite stores lineage, evaluation, champion, pin, and audit metadata.

On resume, the latest compatible model weights are loaded. This release deliberately does **not** persist replay contents or AdamW moments; replay refills and optimizer state restarts. That makes recovery reliable and compact but means repeatedly stopping a run reduces training efficiency. Prefer pause/resume within one backend session for short interruptions.

## Current limitations and next experiments

- The production engine is optimized Python, not the future Rust vector engine described as an extension path.
- The model is feed-forward over a maintained information state; it does not learn a recurrent history or public-belief search policy.
- Replay blends off-policy terminal Monte-Carlo and frozen next-decision targets, but does not yet perform full all-action counterfactual reanalysis or search.
- There is one base-set ruleset and no expansions or online multiplayer protocol.
- A 24-hour budget cannot guarantee “excellent” strength. Only evaluation can establish that.

High-value next experiments are differential-tested Rust simulation, centralized action-count-bucket inference, randomized prior functions, recurrent public-action history, entity attention, and ReBeL-style public-belief search. Each should earn its complexity through paired held-out comparisons, not training loss alone.
