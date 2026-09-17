# Astro6 review at 108.6 active hours

The run is ready to resume with operational fixes installed. Keep the current
learning parameters and eight CPU workers. The recorded progress is real under
the frozen historical game rules, but the current learner is behind its incumbent;
there is no evidence here that increasing the learning rate, adding threads, or
relaxing promotion thresholds would improve strength.

## Evidence from the campaign

- 108.584 active hours; 6,475,264 training games; 790 screens; 253 completed gates.
- Nine verified promotions. I recomputed all nine successful gates from their
  saved paired outcomes and checked their candidate actor hashes.
- Champion 9 scored **58.1787% against the original champion** over 2,048 paired
  seeds / 4,096 games. Its recorded fixed-sample 95% Hoeffding interval is
  **55.1777–61.1797%**. This does not establish the longer-term 70% target.
- The current stage has trained another 1,330,688 games without promotion. Its
  last ten gate scores average 47.7661% against champion 9. These are different
  candidates; the average describes recent results, not one model's strength.
- The last 500 training iterations average 29.78 seconds before evaluation,
  post-update KL 0.00324 versus a 0.015 target, entropy 0.0953, and clipping
  fraction 1.64%. No updates were rejected in that window; 20.6% stopped early
  at a minibatch KL check. This is not a numerical-instability signal.
- There were 44 truncated training games in the entire run. All stage metric
  iteration sequences agree with their saved resume states.

| Champion | Cumulative training games | Score against original champion |
|---|---:|---:|
| 1 | 8,192 | 52.73% |
| 2 | 1,032,192 | 52.86% |
| 3 | 1,343,488 | 52.81% |
| 4 | 1,957,888 | 55.30% |
| 5 | 2,441,216 | 55.66% |
| 6 | 2,498,560 | 56.03% |
| 7 | 2,695,168 | 56.30% |
| 8 | 3,096,576 | 55.81% |
| 9 | 5,144,576 | 58.18% |

These benchmarks have sampling uncertainty; neighboring point estimates do not
prove improvement or regression. The independent succession gates, rather than
these benchmark differences, establish the recorded promotions. Historical
rules and engine behavior differ from today's manual arena; these results must
not be pooled with corrected-rules results.

## Changes installed

1. **Resume budgets:** previously the UI resumed with the CLI's 48-hour default,
   even though the saved campaign budget is 144 hours. Omitted hours/workers now
   inherit the manifest. Explicit overrides persist. About **35.42 active hours**
   remain under the existing budget.
2. **Interrupted benchmarking:** a committed promotion with an incomplete
   original-champion benchmark now finishes that benchmark before more training.
3. **Decided gates:** resume recognizes an already-passed, futile, or exhausted
   gate from its saved outcomes before scheduling more games. Pausing at a
   decisive boundary no longer marks it as an unfinished gate.
4. **Useful performance measurements:** future learner records separate rollout,
   learning, checkpointing, evaluation, and total iteration wall time, plus
   worker count and committed updates. Gate reports include session wall time.
   Existing evaluator `seconds` values sum per-task durations, so dividing them
   by campaign elapsed time would overstate evaluation overhead.
5. **Numerical checks:** nonfinite gradient norms or policy diagnostics stop the
   learner before applying an invalid update; nonfinite post-update KL cannot
   reach checkpoint saving.

Both execution scripts were patched in the source tree **and in the runtime
actually used by Resume**. Maintenance acquired the campaign lock and verified
its original code identity. The full previous runtime, campaign manifest/state,
and active-stage manifest are backed up under
`data/progressive/20260910/runtime-revisions/1789535828976573000/`.
The campaign and active-stage manifests record the revision; completed-stage
manifests and historical gate evidence retain their old identities.

Model and optimizer artifacts, random states, counters, champion, temperature
0.03, two epochs, batch size 512, adaptive learning-rate cap 1e-5, target KL
0.015, screen schedule, game rules, and promotion error spending are unchanged.
A parameter or rules experiment would need separately measured strength evidence;
these infrastructure tests do not establish a stronger policy.

## Performance and validation

A repeated fixed-seed benchmark used the current learner and incumbent, with
64 trajectories and 128 paired evaluation seeds per repetition:

| Worker processes | Mean rollout seconds | Mean evaluation seconds |
|---|---:|---:|
| 4 | 2.734 | 10.537 |
| 8 | 2.172 | 7.946 |
| 10 | 2.196 | 7.975 |

All worker settings produced identical trajectory position counts, win totals,
and evaluation outcomes. Ten workers offered no measured advantage. Keep eight
processes and one Metal learner, with the existing BLAS thread caps. Python's
[process-based executor](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor)
allows these CPU tasks to bypass the interpreter lock; replacing them with
ordinary Python threads is not a supported optimization here. This short
benchmark does not predict sustained throughput under every thermal/load state.

Validation: **23 targeted tests passed**, covering supervisor resume, provenance
maintenance, promotion evidence, learner math, and progressive API behavior.
Four isolated Metal training executions checked original versus patched code
and split pause/resume. Final model tensors and optimizer arrays were **exactly
identical** across all three paths. The smoke test used 32 games per iteration;
it verifies behavior preservation, not long-run strength or a speedup.

Raw analysis, benchmark results, and smoke artifacts are in
`data/astro6-review-20260915/`. The campaign remains paused with its STOP markers
intact. Use the existing **Resume training** control; no dashboard restart is
required for the patched frozen scripts to take effect.
