# Why Astrosynapse 2 plateaued, and the Astrosynapse 3 recovery plan

Audit date: 2026-08-11  
Audited run: `d5e31ed5aeb5`  
Live-run snapshot used below: approximately 4.15 million games, 998 million decisions, and 1.04 million learner updates

## Executive conclusion

The plateau is not primarily a lack of model parameters, a shortage of games, or Python's raw simulation speed. It is a policy-improvement failure caused by several mechanisms reinforcing one another:

1. The August 10 training-regime change nearly removed exploration while switching ordinary rollouts to a frozen champion.
2. Replay records only the chosen action. Alternatives therefore receive no counterfactual evidence, and late-game correlations can be learned as if they were causal.
3. The tactical preference objective taught a shared action encoder to suppress `END_TURN` globally. In optional scrap decisions, `END_TURN` means **keep the ship**, so this objective strongly and accidentally trained the model to scrap.
4. The heuristic bootstrap had the same directional defect: it scored `SCRAP_FOR_ABILITY` at `+610` and `END_TURN` at `-1000`, supplying a second source of poisoned targets.
5. Nominally independent bootstrap heads collapsed to almost the same policy, disabling the intended deep exploration.
6. Replay keeps only about 900,000 decisions—roughly several thousand games at current game lengths—and heavily reweights rare decision families without correcting their contribution to the shared network.
7. At the same regime boundary the learning rate reached its floor, candidates began rolling back to the champion, and the optimizer restarted. A rejected candidate could not improve its own future data distribution.
8. Evaluation became much better at detecting non-improvement, but it could only reject bad candidates; it could not repair the closed learning loop.

The result is predictable: millions of additional games are generated, but almost all are drawn from policies and actions the system already prefers. More compute makes the network more confident in the same biased objective.

Astrosynapse 3 is implemented as a new, explicit training generation. It keeps retained old checkpoint actors playable and the legacy learner configuration loadable, while changing the recommended learning contract: learner-driven rollouts, exploration across every deployment-eligible action, pure terminal Monte Carlo targets, family-corrected replay weighting, bootstrapped heads with fixed behavior-time perturbations, no global tactical preference loss, no rollback of rejected learners, stronger strategic diagnostics, resumable optimizer/replay state, and an information representation containing important card/action relationships. The compatibility preset is not a bit-for-bit historical simulator because corrected shared engine, baseline, evaluator, and lifecycle behavior applies to it too.

This is a corrected platform, not a claim that a new expert model has already been trained. Establishing a large strength gain requires fresh multi-seed runs and paired held-out evaluation.

## What happened at 2.4 million games

The run provides a useful natural experiment. A restart and training-system change occurred just before checkpoint 2,402,304. Several changes landed together:

| Mechanism | Before the change | After the change | Consequence |
|---|---:|---:|---|
| Effective late exploration | about 0.025 | 0.0025 | Only 1 in 400 eligible decisions explored |
| Exploration support | broader | top 3 only | Disfavored actions had zero discovery probability |
| Behavior actor | developing policy | frozen champion | Rejected improvements could not alter future experience |
| Learning rate | decaying | 3e-5 floor | Weak ability to escape the current basin |
| Rejection handling | continued learning | restore champion | Repeatedly returned learner weights to the same attractor |
| Tactical preferences | absent/different | shared global auxiliary | Suppressed `END_TURN` outside its intended context |

The post-change arena record is consistent with a stationary or regressing learner. The 3.303-million candidate narrowly promoted at score 0.5132 with paired interval approximately [0.5027, 0.5236]. Subsequent candidates at 3.403M, 3.503M, 3.703M, 3.803M, 4.003M, and 4.104M scored about 0.480, 0.485, 0.485, 0.488, 0.479, and 0.493 against the champion. The 3.903M point estimate was about 0.501, but its interval still included no improvement. Another candidate failed the held-out quality gate. Evaluation is therefore not merely unlucky: most candidates are measurably worse.

## The high-cost scrap failure is directly diagnosed

This behavior is not a vague strategic weakness. It has a traceable training cause.

In an optional in-play scrap decision, the legal choice is commonly:

```text
SCRAP_FOR_ABILITY(card)  versus  END_TURN
```

Here `END_TURN` has a local semantic meaning: decline the irreversible scrap and retain the purchased ship for future shuffles. The tactical preference dataset, however, used the same global action representation to train play, attack, and activate actions above `END_TURN`. That produces a shared negative `END_TURN` prior even in unrelated decision families.

On a fixed corpus of 359 natural early-game contexts with an optional scrap of a ship costing at least 6, the fraction of states in which checkpoints valued scrap above keep was:

| Checkpoint games | Scrap ranked above keep |
|---:|---:|
| 100,000 | 18% |
| 800,000 | 83% |
| 1,600,000 | 88% |
| 2,000,000 | 94% |
| 2,400,000 | 100% |
| Current | 100% |

The mean scrap-minus-keep logit margin jumped from about `+0.61` to `+8.55` at 2.4M and is now near `+10`. That discontinuity aligns with the new preference-training regime.

A separate 100-game behavior probe found that the current champion scrapped 17 of the 18 distinct cost-6-or-higher ships for which it received an optional scrap opportunity; 15 of 17 happened while the opponent still had at least 20 authority. All three heads agreed. The original heuristic baseline independently scrapped 60 of 63 such ships because of its fixed action score. Thus both learned supervision and bootstrap data pushed in the same wrong direction.

Astrosynapse 3 removes the global preference loss, makes optional scrap neutral in the baseline unless its immediate tactical benefit clears a retention threshold, and blocks promotion when a deterministic early-high-cost retention suite regresses. The deployment action mask still enforces truly dominated rules actions, but the network is no longer trained to generalize an `END_TURN` penalty across unrelated semantic decisions.

## Why the learner stopped discovering alternatives

### Chosen-action confounding

Replay contains `(information state, chosen action, terminal outcome)`. It does not contain a reliable value target for the legal actions that were not selected. In a late winning turn, scrapping an expensive ship may correlate with victory because the ship has already delivered value and the game is ending. A chosen-action value learner can incorrectly infer that the scrap caused the win. Once the model prefers scrap, it almost never observes the counterfactual keep trajectory from the same kind of state.

This is an especially serious problem in a deck builder. Acquisition and scrapping change the distribution of hands many turns later. The correct target has high variance, long delay, and strong policy dependence.

### Exploration existed in configuration more than in behavior

The configured epsilon decayed to 0.025, but a decision-scale multiplier of 0.10 made the effective probability 0.0025. Exploration then sampled only within the model's top three actions. Any strategically unusual action outside that set had exactly zero chance of producing corrective experience.

Bootstrap heads did not compensate. The behavior head was always included in the replay mask and each other head was included with probability 0.8, so about 64% of samples trained all three heads. They also shared nearly the entire network and had no independent prior. Fixed-state probes found less than 1% head argmax disagreement and probability standard deviation around 0.002–0.005. This is an ensemble in name, not a useful posterior sample.

### Frozen-champion rollouts plus rollback formed a closed loop

Ordinary games came from the accepted champion. A new candidate learned off that distribution, was evaluated, and—if rejected—was restored to the champion with a fresh optimizer. The candidate could neither visit new states through its own behavior nor preserve partial improvements that failed one global arena. This is safe deployment logic incorrectly used as learning logic.

Astrosynapse 3 separates the two roles. The champion remains the deployable and evaluation anchor; the learner remains the learner after a failed arena and generates most new trajectories. League and baseline opponents limit forgetting and cycling.

## Data use, replay, and target problems

The 900,000-decision replay capacity sounds large, but current games contain many decisions. It corresponds to only roughly five thousand recent games, not millions. The 35% recent partition can be drawn from only hundreds of games. Consecutive decisions from the same game are highly correlated, so the effective sample size is much smaller than the raw decision count.

The old family sampler also deliberately sampled rare decisions much more often than they were written. Observed sample-to-write ratios were roughly `2.3×` for main decisions but around `131–178×` for some copy, destroy, and free-acquire families. Protecting rare families is reasonable, but every sample updates the shared state/action trunk. Without importance correction, this changes the objective rather than merely reducing variance.

Finally, a target that was 60% terminal result and 40% the frozen actor's next-decision maximum bootstrapped the same policy's errors. The blend can reduce variance, but it is not an unbiased counterfactual estimate in this off-policy, partially observed setting.

Astrosynapse 3 therefore starts with a conservative target: the acting player's terminal result. It samples closer to the natural decision distribution and corrects family-level stratification toward cumulative behavior/write shares. The controlled recent partition remains an explicit recency bias. Terminal targets have higher variance, so this must be evaluated experimentally; it removes a more dangerous source of self-confirming bias. Search-based policy targets are a later, cleaner way to regain lower-variance credit assignment.

## Representation and capacity

The 1.26-million-parameter residual MLP is not obviously too small for the base set. It can fit the training data very well, and its confidence in the bad scrap policy increased dramatically. A larger MLP trained on the same data and loss would likely learn the same defect faster.

The important limitation is relational inductive bias. The legacy state contains most relevant facts, but it asks an MLP to infer interactions such as:

- the acquired card's faction matches cards already in the deck or hand;
- the purchase will trigger an ally effect this turn or on a likely future draw;
- a target base's defense exceeds available and likely next-turn combat;
- the opponent's public known-top cards cannot produce enough combat or trade to exploit that base;
- scrapping this specific card trades a small immediate reward for losing a costly future draw.

Astrosynapse 3 adds 16 explicit state/action cross-features for these relations while preserving the exact legacy prefix. The optimized encoder precomputes observation-level context once per decision and adds only candidate-specific fields per legal action. On a controlled 226-decision/1,415-action corpus, its median cost is about 0.649 ms per decision versus 0.594 ms for the legacy encoder—about 1.09×, down from the first implementation's roughly 4.15× overhead.

This relational bridge is intentionally modest. The longer-term model should encode cards, zones, bases, and legal actions as entities and use permutation-equivariant attention. A Set Transformer-style encoder is a natural fit for unordered hands, discard piles, trade rows, and decks, while positional encoding should be reserved for genuinely known ordered top cards.

## Star Realms-specific learning requirements

Star Realms is not just a small board game:

- It is partially observed. Opponent hand order and shuffled draws are hidden, but discard piles, purchases, scrap, and known-top effects create a reconstructible public belief state.
- It is stochastic. Trade-row replacement and shuffled deck order make single-trajectory targets noisy.
- A turn is a sequence of interdependent actions. Ordering play, ally activation, copying, scrapping, buying, targeting bases, and ending the turn matters.
- Deck-building actions have delayed, policy-dependent effects over future shuffle cycles.
- Scrapping is irreversible and becomes rational at different horizons: weak starter-card thinning is good; sacrificing a premium ship early is usually not.
- Legal-action counts vary widely, and many choices are semantically related objects rather than fixed action IDs.
- Strategies can be non-transitive. Economy, blob pressure, base control, and machine-cult thinning can form exploitable matchup cycles.

These properties favor information-state learning, population opponents, coherent episode-level exploration, and eventually public-belief search. They also make a single scalar training loss a poor progress signal.

## How the requested strategies become learnable

### Valuing acquisitions that trigger ally abilities

The minimum viable solution is now present in the Astro3 encoder: candidate faction, owned/hand/in-play same-faction counts, faction presence, ally-trigger potential, and remaining trade are represented as direct action-state relations. This removes the need for the MLP to discover equality and count interactions indirectly across thousands of unrelated inputs.

The stronger design is:

1. Represent every card and legal acquisition as an entity with faction and ability tokens.
2. Let action-query attention read the current deck, hand, in-play cards, known top, market, and remaining trade.
3. Add auxiliary, ground-truth heads for `ally triggers this turn`, expected ally triggers before the next shuffle, draw-cycle length, faction density after purchase, and expected time-to-redraw.
4. Train policy/value targets from shallow stochastic lookahead, so the same acquisition can receive different value depending on the deck and trade-row continuation.
5. Keep terminal win/loss as the final anchor; auxiliary targets improve representation, not the reward definition.

### Leaving an opponent base unattacked when it cannot matter next turn

The current bridge exposes target defense, combat-to-defense ratio, opponent known-top combat/trade, target faction, and lethal/breakpoint indicators. That gives a feed-forward model a direct route to the tactic when the relevant information is public.

For expert play, construct a public belief over the opponent's possible next hand from known top cards, discard, inferred hidden pool, and shuffle state. Sample determinizations consistent with that information, simulate the opponent's best response, and compare attacking the base with preserving combat for authority or another base. A learned next-turn resource head—combat, trade, draw, discard pressure, and base activation probability—can amortize much of this search. Search must never condition on the true hidden order unavailable to the acting player.

## Research-backed architecture path

The design draws on several established lines of work, but uses them according to this game's constraints:

- DouZero demonstrates that terminal-outcome action-value learning can be effective in a large imperfect-information card game, but it does not remove the need for adequate exploration and representative replay.
- Bootstrapped DQN and randomized prior functions motivate one hypothesis per trajectory and independently anchored ensemble heads. Astro3's implemented random projections perturb behavior choice only; fitted prior terms and independent networks remain stronger future variants.
- AlphaZero and Gumbel MuZero motivate search-improved policy targets and completed-policy evaluation. Because Star Realms is stochastic and partially observed, direct perfect-information MCTS would leak hidden state; search must operate over public beliefs or consistent determinizations.
- Deep CFR, NFSP, ReBeL, and related imperfect-information methods motivate learning from information states and best responses rather than treating the simulator's hidden state as observable.
- AlphaStar's league and PSRO-style population training motivate retaining champions, exploiters, and diverse historical policies rather than optimizing against one frozen opponent.
- Set Transformer motivates attention over unordered collections of card entities.
- Go-Explore-style self-play starts motivate curricula that deliberately revisit rare but strategically decisive states instead of waiting for them to occur naturally.
- Conservative offline-RL objectives such as CQL are useful for warm-starting from historical chosen-action data, but conservatism alone cannot invent values for unsupported alternatives. Historical Astro2 data should be treated as demonstrations and opponent material, not unquestioned ground truth.

Primary references are listed at the end of this report.

## What is implemented now

The repository now contains the new `astro3_m4` preset alongside the compatibility preset. Run configuration records `training_generation`, while each actor specification records its encoder version so retained old and new actors can be loaded correctly.

### Learning loop

- Learner-driven rollouts (`60%` current learner, `30%` accepted league/frozen anchors, `10%` corrected baselines). Within current-v-current games, a configured 20% uses the exact perturbation-free, greedy mean-head deployment policy—a 12% overall scheduled share—mitigating train/deploy mismatch while preserving real bootstrap-head masks. Astro3 automatically admits other runs' current champions and explicitly pinned checkpoints after validating their actor/encoder contract and a forward pass.
- Rejected candidates do not roll learner weights back; the champion remains unchanged for deployment.
- Pure terminal Monte Carlo targets by default.
- Natural replay profile with family-stratification correction and reported recency bias.
- Base epsilon `0.15 → 0.05`, applied directly before any adaptive plateau multiplier, with support over every eligible action rather than only top three.
- Five bootstrapped heads, lower mask overlap, and deterministic head-specific projections that perturb behavior action ranking only; they are not fitted prior terms in the MLX loss.
- Adaptive plateau response that increases exploration within a configured bound after repeated clean automatic non-promotions; invalid and stale evidence does not count.
- Cosine learning-rate restarts with decaying peaks instead of remaining permanently at the floor.
- The global tactical preference loss is disabled. Rules-level deployment masks remain.

### Representation and diagnostics

- Versioned relational action encoding; old actors continue to use the old dimensions.
- Fixed strategic retention states, including early optional high-cost scrap decisions.
- Ensemble diagnostics across all decision families, including argmax disagreement and probability dispersion.
- Larger held-out and baseline checkpoint diagnostics.
- Quality-gate reasons and strategic/ensemble metrics visible in the GUI.
- Baseline-score and held-out Brier regressions remain visible but are diagnostic-only for Astro3; the low-sample estimates no longer veto a candidate before the paired arena. Generation 2 keeps its legacy gates.
- Distribution-free paired Hoeffding confidence intervals for automatic promotion, plus Bonferroni-adjusted one-sided Hoeffding bounds at fixed early-rejection looks; early acceptance is forbidden.

### Recovery and operations

- Optimizer moments and warmup origin, versioned exact counters, seed cursor, rollout/replay RNG state, stable league order/statistics, active elapsed time, and a bounded recent replay journal are checkpointed for Astro3. Resume falls back past an incomplete artifact boundary or reports an explicit degraded weight-only recovery.
- Truncated self-play trajectories are excluded from replay and league rewards; any arena containing a truncation is explicitly ineligible for promotion instead of treating a stall as a draw.
- Pause requests first persist a checkpoint, then report `paused`; checkpoint requests can also be serviced while paused.
- Pause and stop states communicate queued/safe-boundary behavior truthfully.
- Natural completion evaluates the newest due immutable candidate before reporting complete. Only clean/current comparisons advance cadence or plateau state; invalid evidence is retried with new deterministic seed ranges. It waits for the trainer-owned job rather than globally draining every arena; a busy evaluator slot may still queue that job. Pause or stop cancels it at the next game boundary, or immediately before it starts if still queued, and cannot promote from a cancelled comparison.
- `keep_checkpoints` now prunes only old, unprotected artifact files after durable boundaries. Champions, pins, promoted league members, active arena inputs, the newest complete resume boundary, and the configured recent window remain available; SQLite lineage and diagnostics remain visible.
- GUI exposes training generation, behavior policy, target mode, effective exploration, plateau response, rollout mix, ensemble-collapse warnings, and resume durability.
- Human-versus-model play accepts both legacy and Astro3 actor encodings.

The replay journal is bounded to the newest configured 100,000 items, so recovery is operationally durable but is not a bit-for-bit serialization of the entire 900,000-item replay buffer. That distinction is intentional and visible in status metadata.

## Performance diagnosis

A controlled CPU profile separated simulation from learning-data preparation:

| Workload | Approximate rate / share |
|---|---:|
| Rules engine + heuristic, no neural encoding | 96.6 games/s |
| Legacy encoder path | 7.0 games/s; about 92.6% of chooser wall time in encoding |
| First relational encoder implementation | 1.7 games/s; about 4.15× legacy encoding cost |
| Optimized relational encoder microbenchmark | about 1.09× legacy encoding cost |
| Replay index selection, 32 × 2,048 samples | 5.368 s before; 0.111 s after (about 48× faster) |
| Runtime actor export | 229.5 ms compressed; 3.35 ms uncompressed (about 68.6× faster) |
| Runtime actor load | 19.7 ms compressed; 3.42 ms uncompressed (about 5.8× faster) |

The pure Python rules engine is therefore not the present root cause. Neural feature construction and replay sampling are the major Python/NumPy hotspots. A native, cloneable engine will become important when each real move launches many search simulations; porting first would only generate biased trajectories faster.

The engineering order is:

1. ~~remove redundant full-buffer sorting from replay sampling~~ — implemented and regression-tested;
2. batch/cache decision encoding and inference;
3. ~~remove compression from disposable per-iteration actor snapshots while keeping checkpoints compressed~~ — implemented and regression-tested;
4. batch inference across environments and profile inter-process serialization;
5. ~~enforce checkpoint artifact retention without deleting champion, pinned, league, arena, or resume evidence~~ — implemented with auditable availability metadata; metric compaction remains optional future work;
6. build a Rust or C++ cloneable simulator only behind deterministic differential replay tests;
7. use the native engine to buy belief-consistent search, not merely a larger raw game counter.

## Training and validation plan

### Phase A — establish that the corrected loop learns

Run at least three fresh Astro3 seeds. Do not warm-start the learner from the 2.4M+ Astro2 weights; keep those models as frozen opponents and regression benchmarks. Compare controlled ablations:

1. Astro3 default.
2. Default without relational features.
3. Default without fixed behavior perturbations.
4. Default with legacy replay sampling.
5. Default with champion-only behavior.

Plot strength against decisions, learner updates, and wall time—not just games. Use the same held-out seed banks and paired seat swaps for every condition.

### Phase B — better credit assignment

- Add game-grouped prioritized replay based on surprise and strategic rarity, with explicit probability logging and importance weights.
- Store compact legal-action sets at selected decisions and periodically reanalyze them with the current network or search.
- Add auxiliary factual targets: next-shuffle horizon, faction density, ally triggers, expected next-turn resources, base survival/activation, lethal distance, and card redraw probability.
- Sample targeted starts from valid engine states around rare scrap/copy/free-acquire/base decisions, while keeping ordinary full games as the main distribution.

### Phase C — entity model and public-belief search

- Replace the flat relational bridge with card/zone/action entities and action-query attention.
- Train a recurrent public-action-history or belief encoder.
- Implement shallow stochastic search over determinizations consistent with public information.
- Use a Gumbel-style root procedure or sampled policy improvement to give more than the chosen action a learning target.
- Distill the search policy into the fast actor for bulk self-play and human play.

### Phase D — open-ended population training

- Maintain champions, recent learners, historical strategic snapshots, main exploiters, and league exploiters.
- Choose opponents with a PSRO/PFSP-style mixture based on uncertainty and exploitability, not just low raw win rate.
- Periodically train fresh best responses from randomized initialization to detect blind spots.
- Track a payoff matrix and non-transitivity; do not compress all evidence into one Elo number.

### Phase E — native search engine

- Freeze a large corpus of serialized states, legal actions, state hashes, and seeded continuations.
- Differential-test the native engine against Python for at least 10,000 traces before use in training.
- Target at least a 10× simulation gain and cloneable state transitions.
- Spend the gain on 32–128 belief-consistent simulations at high-value decisions and fewer simulations at forced/obvious decisions.

## Promotion and release gates

The following gates should be automated before describing a model as substantially stronger:

| Gate | Required evidence |
|---|---|
| Rules parity | 10,000+ seeded differential traces; identical legal actions, state hashes, and outcomes |
| Information integrity | No feature or search path can access hidden deck/hand order unavailable to the actor |
| Resume integrity | Continuous and interrupted/resumed deterministic test agree on counters, RNG continuation, optimizer availability, and checkpoint lineage |
| Operations | Pause checkpoint completes before `paused`; pause/checkpoint/stop integration tests pass; bounded pause latency displayed |
| Basic strategy | 100% pass on deterministic dominance suite and early high-cost retention gate |
| Broader strategy | Held-out ally, acquisition, scrap, base-target, lethal, and known-top scenario banks show no material regression |
| Exploration | Every eligible action has nonzero configured support; head collapse warning remains off on held-out states |
| Replay | Family write/sample shares, applied weight range/effective sample size, recency share, and family drift are reported |
| Evaluation | Game-grouped held-out metrics and paired seat-swapped arenas use immutable seeds |
| Reproducibility | Improvement appears in at least 2 of 3 fresh seeds, preferably all 3 |
| Astro2 superiority | Paired lower confidence bound above 0.55 against the best Astro2 champion |
| Population robustness | Paired lower bound above 0.52 against each of the five strongest/diverse frozen opponents |
| Exploitability proxy | No newly trained held-out best response scores above 0.55 after its own validation |
| Requested concepts | Purpose-built ally-trigger and known-top/base benchmarks pass before relying on anecdotes |
| Native engine | At least 10× search-simulation throughput after parity is established |

An isolated lucky promotion is insufficient. The meaningful success criterion is a rising lower confidence envelope across seeds and opponents, plus disappearance of the diagnosed strategic failures.

## Recommended immediate operation

1. Let the current Astro2 process finish or stop it safely; preserve its database and actors as the forensic baseline.
2. Pin several strategically distinct, credible Astro2 checkpoints. The current champion is admitted automatically; pinning preserves useful older anchors when champion status changes.
3. Run the quick validation recipe after installing the updated code.
4. Launch three fresh `astro3_m4` runs with distinct numeric seeds (the GUI exposes **Training seed**, and the CLI accepts `--seed`). On a single machine, run them sequentially so they do not compete for Metal and memory.
5. Inspect early-high-cost retention, ensemble disagreement, effective epsilon, replay effective weights, rollout mix, truncations, and held-out Brier at every evaluated checkpoint.
6. Do not promote or warm-start from the current champion merely because it has four million games. Its measured scrap bias is strong prior evidence against using it as the new learner initialization.
7. After the first 500k–1M games per seed, run a common paired matrix against the current champion and diverse historical/baseline opponents before investing in longer runs.

## Honest remaining limitations

- No new long-duration Astro3 run has yet proved a large skill improvement. The implementation fixes identified causes and creates the measurement system needed to establish one.
- Terminal Monte Carlo targets are unbiased with respect to the behavior return but high variance and still policy dependent.
- The current model remains a shared-trunk ensemble. Its random projections perturb behavior but do not anchor the fitted values; fitted prior terms and genuinely independent ensemble networks remain future work.
- The relational feature bridge is not the final entity-attention architecture.
- Replay recovery is bounded, not bit-exact for the full buffer.
- Public-belief search, action-set reanalysis, population exploiters, and the native simulator are staged work, not silently claimed as complete.

## Primary research references

- Zha et al., [DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning](https://proceedings.mlr.press/v139/zha21a.html).
- Silver et al., [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270).
- Schrittwieser et al., [Mastering Atari, Go, chess and shogi by planning with a learned model](https://www.nature.com/articles/s41586-020-03051-4).
- Danihelka et al., [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO).
- Osband et al., [Deep Exploration via Bootstrapped DQN](https://proceedings.neurips.cc/paper/2016/hash/8d8818c8e140c64c743113f563cf750f-Abstract.html).
- Osband et al., [Randomized Prior Functions for Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2018/hash/5a7b238ba0f6502e5d6be14424b20ded-Abstract.html).
- Brown et al., [Combining Deep Reinforcement Learning and Search for Imperfect-Information Games (ReBeL)](https://arxiv.org/abs/2007.13544).
- Brown et al., [Deep Counterfactual Regret Minimization](https://proceedings.mlr.press/v97/brown19b.html).
- Heinrich and Silver, [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (NFSP)](https://arxiv.org/abs/1603.01121).
- Vinyals et al., [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z).
- Balduzzi et al., [Open-ended Learning in Symmetric Zero-sum Games](https://proceedings.mlr.press/v97/balduzzi19a.html).
- Lee et al., [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).
- Kumar et al., [Conservative Q-Learning for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html).
- Ecoffet et al., [First return, then explore](https://www.nature.com/articles/s41586-022-04460-3).
