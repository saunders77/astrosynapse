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
6. **Persist the long horizon.** Scheduled checkpoints retain up to 500,000 policy decisions;
   pause/final checkpoints retain the full buffer. Policy archives are uncompressed for fast local
   save/load because disk, not CPU time, is the available resource. Up to 100 checkpoint families
   are retained by the Astro5 preset.
7. **Measure natural behavior.** Checkpoint diagnostics use positions produced by the candidate
   playing fixed opponents. They report natural head disagreement, normalized entropy,
   own-policy value calibration, and KL from the current champion. The synthetic all-family suite
   remains a coverage check only.
8. **Separate optimization signals.** Learner telemetry includes behavior-policy, value, search
   policy, and search value losses; importance clipping; aggregate clipping frequency; and sampled
   actor/value/search gradient norms every 1,024 updates.
9. **Evaluate cheaply and often.** Every 10,000 games, the latest checkpoint gets a 64-pair,
   seat-reversed canary. Every 50,000 games starts a 2,000-pair promotion test. Fixed geometric
   looks from 1,000 pairs can accept a clearly superior candidate using a Bonferroni-corrected one-sided Hoeffding
   lower bound; truncations are scored as candidate losses. Ambiguous evidence runs the full
   2,000 pairs, and the independent early-rejection boundary remains available. Natural completion
   forces a full evaluation of the newest checkpoint even when it falls between normal cadences.
10. **Adapt and branch.** A persisted realtime governor adjusts bounded learning-rate, update,
    entropy, and reanalysis multipliers from canary trend, entropy, gradient clipping, and search
    coverage. Optimization health is checked every 500 games; strategic state changes only when a
    new canary arrives. Each actor iteration is divided into four microtasks per worker, allowing
    the process pool to steal work around slow games while CPU collection overlaps Metal learning.
    Branch Lab can fork 1–8 independent recipes from any compatible checkpoint and run them
    sequentially on the single Metal learner.

## Branch execution model

An experiment pins its source and creates one run per variant. Each run receives private copied
artifacts and an independent deterministic seed. All variants fork the selected source directly;
no queued variant inherits the preceding variant's final checkpoint or champion. Only one learner runs at a time. With auto-advance
enabled, queued branches start after the current branch—or an already-active ordinary run—releases
the trainer. Stopping a branch pauses its experiment instead of unexpectedly starting the next one.
The GUI's Branch runner controls resolve the backend's active run ID, so pause/stop remain enabled
and target the correct process even while the user is viewing a queued branch.

The built-in GUI variants are balanced search, search-heavy, entropy recovery, value-first, fast
exploitation, wide belief search, low-learning-rate long memory, and explorer. They inherit model
architecture from the source checkpoint so the imported weights always remain compatible. The GUI
allows any subset of these recipes and can stop each branch by elapsed minutes, generated training
games, or the number of valid completed full promotion evaluations. Game budgets finish at a safe
actor-batch boundary and may therefore exceed the requested count by the final batch.

## Reading the result

Canaries estimate direction, not promotion certainty. Compare their slope alongside normalized
entropy, searched batch fraction, clipping frequency, and objective gradient ratios. Let the full
paired evaluation decide deployment. When the governor requests a branch after repeated canary
regressions, use Branch Lab to preserve the current checkpoint and test different recovery regimes
instead of overwriting the lineage.
