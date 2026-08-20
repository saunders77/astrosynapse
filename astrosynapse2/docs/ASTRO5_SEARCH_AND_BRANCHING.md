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
   but sampled main-phase states now receive a complete searched target over several legal
   actions plus a searched state-value target.
4. **Use public-belief search.** Every search rollout redeterminizes the observer's unknown deck,
   opponent hand/deck split, and future market order. Candidate actions at the same rollout index
   share that determinization and continuation randomness, reducing both information leakage and
   comparison variance.
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
9. **Evaluate cheaply and often.** Every 25,000-game checkpoint gets a 128-pair, seat-reversed
   canary. Every 200,000 games gets the conservative 5,000-pair promotion test. Natural completion
   forces a full evaluation of the newest checkpoint even when it falls between normal cadences.
   A failed quality gate still receives a non-promoting paired measurement.
10. **Adapt and branch.** A persisted realtime governor adjusts bounded learning-rate, update,
    entropy, and reanalysis multipliers from canary trend, entropy, gradient clipping, and search
    coverage. Branch Lab can fork 1–8 independent recipes from any compatible checkpoint and run
    them sequentially on the single Metal learner.

## Branch execution model

An experiment pins its source and creates one run per variant. Each run receives private copied
artifacts and an independent deterministic seed. Only one learner runs at a time. With auto-advance
enabled, queued branches start after the current branch—or an already-active ordinary run—releases
the trainer. Stopping a branch pauses its experiment instead of unexpectedly starting the next one.

The built-in GUI variants are balanced search, search-heavy, entropy recovery, value-first, fast
exploitation, wide belief search, low-learning-rate long memory, and explorer. They inherit model
architecture from the source checkpoint so the imported weights always remain compatible.

## Reading the result

Canaries estimate direction, not promotion certainty. Compare their slope alongside normalized
entropy, searched batch fraction, clipping frequency, and objective gradient ratios. Let the full
paired evaluation decide deployment. When the governor requests a branch after repeated canary
regressions, use Branch Lab to preserve the current checkpoint and test different recovery regimes
instead of overwriting the lineage.
