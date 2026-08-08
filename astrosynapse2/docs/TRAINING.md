# Training and evaluation

## What is learned

Astrosynapse 2 uses Deep Monte-Carlo action values, inspired by [DouZero](https://proceedings.mlr.press/v139/zha21a.html), rather than the legacy PPO loop. For every non-forced action chosen by a player, the training target is that player's terminal result:

```text
win = 1.0
loss = 0.0
genuine draw or safety truncation = 0.5
```

The learner minimizes masked binary cross-entropy on the selected action. It does not use authority margin, hand-authored card quality, or the average result of a multi-game match. Each game's result labels only decisions from that game.

Three bootstrap heads share the network. Each player samples one head, epsilon, and bootstrap mask for the entire game. That makes exploration coherent across a trajectory: one game follows one sampled value hypothesis instead of changing personality at every chooser call. Deployment averages the heads and selects greedily.

## Information-state representation

The base-set encoder currently produces:

- 1,292 state features;
- 203 semantic action features;
- 8 stable decision families;
- 3 bootstrap outcome heads by default.

State features retain exact per-card counts in public or inferable information-set zones, including hand, unknown-order deck bags, discard, in-play cards, known top cards, the opponent's inferable hidden pool, trade row, remaining trade-deck multiset, scrap heap, and Explorer supply. It also records authorities, available combat/trade, discard pressure, public continuous effects, turn state, and whether the acting player started.

Only genuinely ordered known-top slots are positional. Shuffled hand, market, deck, and option indices are excluded. Action features describe the verb, family, source and target card identities, zones, effect, and rule-relevant quantities. Forced one-option decisions are resolved by the engine and never enter inference or replay.

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

Replay is preallocated and split by decision family. The default 900,000-decision buffer uses about 2.7 GB for state/action arrays and metadata. Balanced sampling prevents frequent main-phase choices from evicting discard, scrap, copy, destroy-base, trade-row-scrap, free-acquire, and modal decisions. A controlled recent fraction adapts to the latest policy without discarding older experience.

## Opponent schedule

The implemented schedule mixes:

- 55% current model self-play;
- 30% prioritized frozen checkpoints, emphasizing opponents with a low smoothed current score;
- 15% balanced, economy, and aggressive deterministic baselines.

Only current-policy decisions are collected when playing a frozen or heuristic opponent. Both players are collected in current self-play. Matchup estimates use a Beta prior to avoid overreacting to a handful of noisy games.

The initial random model is saved as the lineage root. It is useful for reproducibility, not as a strength claim.

Before ordinary self-play, the replay warmup is filled with games between the balanced/economy/aggressive anchors, collecting both players' chosen actions with the same terminal win/loss targets. This demonstration curriculum lasts for 2,000 learner updates (256 in the quick preset), rather than ending as soon as one training batch fits in replay. It is not reward shaping or permanent imitation: it prevents a random untrained network from filling the first buffer with stalemates, then yields completely to the configured self-play/league mix. The zero-game random persistence checkpoint is also excluded from the opponent league.

## Recommended M4/16 GB preset

| Setting | Default |
|---|---:|
| Duration | 1,440 minutes |
| CPU actors | up to 8 |
| Games per actor batch | 16 |
| Model width / residual blocks | 192 / 3 |
| Bootstrap heads | 3 |
| Learner batch | 2,048 |
| AdamW learning rate | 3e-4 → 3e-5 cosine |
| Gradient clip | 5.0 |
| Updates per actor iteration | 32 |
| Replay capacity / warmup | 900,000 / 50,000 |
| Heuristic bootstrap | First 2,000 learner updates |
| Epsilon | 0.20 → 0.025 over 1.5M games |
| Checkpoint interval | 100,000 games |
| Evaluation interval | 500,000 games |
| Promotion evaluation | 5,000 paired seeds / 10,000 games |

The learning-rate cosine follows active elapsed time. A short update warmup protects a fresh optimizer from the highly correlated first replay batches and is repeated when a resumed run recreates AdamW. Paused time and backend downtime are excluded; consumed active time is recovered from the latest persisted metric when a run resumes.

These values are defaults, not universal optima. Watch measured throughput and memory. On a 16 GB machine, reduce replay capacity or model width if macOS memory pressure becomes sustained; preserve paired evaluation size before spending that compute on a larger network.

## Automatic and manual evaluation

Every arena seed is run twice with the same isolated game randomness and exact model seat swap. Model-role RNG is also held stable across the pair. The seed pair—not two correlated games—is the statistical unit.

At the configured interval, the trainer checkpoints the candidate and queues it against the current champion. A job can promote automatically only if all of these hold:

1. both entries are immutable checkpoint actors;
2. the job was created internally as an automatic evaluation;
3. at least 5,000 seed pairs completed;
4. the full requested job completed;
5. the paired 95% confidence interval's lower bound is above 50%.

The candidate's evaluation record, champion flags, and run champion ID are updated in one SQLite transaction. Small quick-run jobs and every job created manually in the GUI have `automatic: false` and cannot promote anything, regardless of observed win rate.

Arena output reports overall and seat-split score, Wilson and paired intervals, draws, truncations, games/s, ETA, and recent paired seeds. Elo difference is only a display transform of the measured score; it is not treated as an absolute global rating.

## Reading diagnostics

- **Outcome BCE** — masked terminal-result cross-entropy. Lower is better only on a comparable held-out distribution.
- **Brier score** — squared error of the head-averaged win probability. It measures calibration as well as discrimination.
- **Explained variance** — how much target variation the current value estimate explains. Negative values are common during early noisy learning.
- **Bootstrap uncertainty** — average disagreement among heads. Falling uncertainty can mean convergence, but should be checked against arena strength and policy diversity.
- **Replay utilization/families** — fill and per-family occupancy. Rare rings should receive samples over time; a permanently empty family may indicate a rule/encoding issue.
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
- Replay is off-policy terminal Monte-Carlo data without counterfactual reanalysis.
- There is one base-set ruleset and no expansions or online multiplayer protocol.
- A 24-hour budget cannot guarantee “excellent” strength. Only evaluation can establish that.

High-value next experiments are differential-tested Rust simulation, centralized action-count-bucket inference, randomized prior functions, recurrent public-action history, entity attention, and ReBeL-style public-belief search. Each should earn its complexity through paired held-out comparisons, not training loss alone.
