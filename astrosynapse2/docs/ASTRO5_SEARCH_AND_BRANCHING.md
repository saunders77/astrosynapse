# Astro5 search, governor, and branching

Astro5 is the training path intended to replace the plateaued Astro4 recipe. It preserves
older run contracts; selecting Astro4, Astro3, or Astro2 still creates those exact generations.

## The ten changes

1. **Complete lineage boundaries.** Checkpoints persist weights, actor, optimizer, compact
   legal-set replay, preference replay, counters, RNG/league state, and branch-relative
   schedule origins. A configured restore hydrates the champion boundary or clears any artifact
   that is unavailable; it never mixes rejected post-champion replay into restored weights.
2. **Do not destroy performance valleys.** Astro5 defaults to `rejected_candidate_action=continue`.
   The champion remains the deployment anchor, while a rejected learner is quarantined and may
   continue until later canaries show whether it crossed the valley. Full restore and new-branch
   policies remain explicit alternatives.
3. **Learn calculated action sets.** Selected-action REINFORCE remains a broad background signal,
   but sampled main-phase states receive a searched target over up to four legal actions plus a
   searched state-value target. The M4 preset samples at most one state in 0.25% of games and uses
   one rollout per action, avoiding the former search budget's dominant CPU cost.
4. **Use public-belief search.** Every search rollout redeterminizes the observer's unknown deck,
   opponent hand/deck split, and future market order. Candidate actions at the same rollout index
   share that determinization and continuation randomness, reducing both information leakage and
   comparison variance. Reanalysis follows each candidate for two turns and then uses the
   acting model's state-value head; terminal outcomes still take precedence when a branch ends
   inside that horizon.
5. **Remember games, not repeated rows.** Workers retain a phase/family reservoir (12 decisions
   per player-game by default) before process-pool transfer. Search-labelled positions are kept
   first. Replay then samples player-games uniformly and balances phase/family within a game.
6. **Persist the long horizon without spending RAM.** The newest 250,000 policy decisions remain
   in the low-latency in-memory reservoir. Up to 5 million older decisions move by complete
   player-game into immutable float16 columnar SSD shards and are sampled through bounded NumPy
   memory maps; 30% of a batch comes from this tier by default. Checkpoint manifests cover both
   tiers, preserve referenced shards across rollback/branch creation, and reclaim obsolete shards
   only after no retained checkpoint references them.
7. **Measure natural behavior.** Checkpoint diagnostics use positions produced by the candidate
   playing fixed opponents. They report natural head disagreement, normalized entropy,
   own-policy value calibration, and KL from the current champion. The synthetic all-family suite
   remains a coverage check only.
8. **Separate optimization signals.** Learner telemetry includes behavior-policy, value, search
   policy, and search value losses; importance clipping; aggregate clipping frequency; and sampled
   actor/value/search gradient norms every 1,024 updates.
9. **Evaluate cheaply and often.** Every 10,000 games, training saves a checkpoint and runs a
   64-pair, seat-reversed canary. Every 50,000 games starts a 2,000-pair promotion test. Fixed geometric
   looks from 1,000 pairs can accept a clearly superior candidate using a Bonferroni-corrected one-sided Hoeffding
   lower bound; truncations are scored as candidate losses. Ambiguous evidence runs the full
   2,000 pairs, and the independent early-rejection boundary remains available. Natural completion
   forces a full evaluation of the newest checkpoint even when it falls between normal cadences.
10. **Adapt and branch.** A persisted realtime governor adjusts bounded learning-rate, update,
    entropy, and reanalysis multipliers from canary trend, entropy, gradient clipping, and search
    coverage. Optimization health is checked every 500 games; strategic state changes only when a
    new canary arrives. Each actor iteration is divided into four microtasks per worker, allowing
    the process pool to steal work around slow games while CPU collection overlaps Metal learning.
    Branch Lab can fork independent recipes from any compatible checkpoint and run them
    sequentially on the single Metal learner.

## Branch execution model

An experiment pins its source and creates one run per variant. Each run receives private copied
artifacts and an independent deterministic seed. All variants fork the selected source directly;
no queued variant inherits the preceding variant's final checkpoint or champion. Only one learner runs at a time. With auto-advance
enabled, queued branches start after the current branch—or an already-active ordinary run—releases
the trainer. Stopping a branch pauses its experiment instead of unexpectedly starting the next one.
The GUI's Branch runner controls resolve the backend's active run ID, so pause/stop remain enabled
and target the correct process even while the user is viewing a queued branch.

The built-in GUI variants include promotion-direction refinement, mature champion refinement, balanced search, search-heavy,
entropy recovery, value-first, fast exploitation, wide belief search, low-learning-rate long
memory, and explorer. They inherit model
architecture from the source checkpoint so the imported weights always remain compatible. The GUI
allows any subset of these recipes and can stop each branch by elapsed minutes, generated training
games, or the number of valid completed full promotion evaluations. Game budgets finish at a safe
actor-batch boundary and may therefore exceed the requested count by the final batch.

## Mature champion refinement

This mode is for a champion that already plays well and needs small, defensible improvements. At
the branch root it keeps the imported weights but intentionally clears imported optimizer momentum
and replay, avoiding continued training on a stale losing trajectory. It then runs conservative
local trials with lower exploration, more searched action-set supervision, a larger champion-history
share, and rollback to the champion boundary after a failed full evaluation. Pause/resume after the
branch has begun remains fully durable and restores the branch's own optimizer and replay.

The mature governor monitors clipping, normalized entropy, behavior-policy importance ratios,
searched-batch coverage, and rolling three-canary trends. It can cool learning-rate/update pressure
and increase searched supervision within configured bounds. It never weakens promotion confidence.

If a full evaluation is score-positive but its adjusted lower confidence bound still overlaps the
promotion threshold, the arena adds fixed-size blocks until the advantage is proven, the observed
score stops leaning positive, or the configured ceiling is reached. The default mature ceiling is
12,000 pairs and the GUI permits automatic ceilings up to 250,000. Confidence is Bonferroni-adjusted for every possible
extension boundary so optional repeated looks do not make promotion easier.

## Promotion-direction refinement

This branch-only mode adds a small weight-space prior to mature refinement. At branch creation it
finds the latest retained, completed full evaluations that actually promoted a candidate in the
source run. For each promotion it computes `candidate weights - defending champion weights`,
normalizes every tensor's transition by its RMS, and combines up to five transitions with recent
promotions weighted most strongly. A coordinate is guided only when at least 60% of the weighted
historical signs agree. The resulting artifact is frozen with the branch lineage and is auditable.

During learning, the ordinary minibatch gradient remains primary. The default adds a direction
component with RMS equal to 6% of that tensor's ordinary gradient RMS; because the optimizer
subtracts gradients, this nudges weights further along the successful consensus direction. The
strength is safely adjustable between batches and is reported with the number of guided tensors in
Diagnostics. Branch Lab exposes strength, history count, sign agreement, recency decay, initial
pair count, extension block, and maximum pair count before it freezes the branch artifact. This is
a prior, not a causal claim: a promoted checkpoint contains many correlated
SGD changes, so the normal canaries and arena still decide whether the reuse helped.

Directional full evaluations predeclare regular looks every 2,000 seed-pairs from pair 2,000 up to,
but not including, the configured ceiling. At each interim look, a two-sided Hoeffding bound shares a
1% familywise error budget across every look and both possible decisions: its upper bound must be
below 50% to reject early, or its lower bound must be above 50% to promote early. Unresolved evidence
continues to the ceiling—100,000 pairs by default—where a single final decision receives the remaining
4% error budget and therefore uses 96% standalone confidence. The complete sequential procedure has
at most 5% familywise decision error. The automatic ceiling can be configured as high as 250,000.
Evaluations retain only recent human-readable
pair diagnostics in memory and checkpoint their large resumable score arrays at a reduced cadence.

## Reading the result

Canaries estimate direction, not promotion certainty. Compare their slope alongside normalized
entropy, searched batch fraction, clipping frequency, and objective gradient ratios. Let the full
paired evaluation decide deployment. When the governor requests a branch after repeated canary
regressions, use Branch Lab to preserve the current checkpoint and test different recovery regimes
instead of overwriting the lineage.
