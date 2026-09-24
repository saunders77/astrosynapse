# Astro6: replace noisy whole-game credit with action comparisons

The September 19 correction did not establish stronger play. The follow-up
contains 671,744 games, 1,312 learner iterations, and 11.286 learner hours
(evaluation adds further wall time). Its 82 screens average 50.212% against
champion 9; 43 completed gates average 50.050%. There are still nine promotions.
The old campaign has been safely paused, preserving its pending verification.

| Fresh-rollout diagnostic | First 300 corrected iterations | Last 300 |
|---|---:|---:|
| Overall Brier error | 0.17209 | 0.17394 |
| Forced-position Brier error | 0.20582 | 0.20267 |
| Choice-position Brier error | 0.17061 | 0.17260 |

Lower Brier error is better. These are different state samples from changing
policies, not a controlled causal comparison. They support neither a broad
calibration breakthrough nor a strength claim. Raw aggregates are saved in
`data/astro6-counterfactual-20260920/diagnosis.json`.

## A remaining coverage bug

The engine does not call a chooser when there is only one rules-legal action.
The previous fix captured decisions reduced to one option by the model's mask,
but still missed those engine-forced decisions. That was a gap in the previous
implementation and its mock test. Value recording now uses the engine's
decision hook, which sees both kinds of forced decision, and reuses an encoding
when the chooser already produced it. Opponent decisions are excluded. The
regression test now reproduces the engine bypass instead of always calling the
chooser. This fix is installed into the paused runtime only after validation.

## Why change the learning signal

The current policy receives about 107.6 trainable positions per game. Its
terminal Monte Carlo objective associates each selected action with the same
binary result, adjusted by a state baseline. This is a valid policy-gradient
estimator, but it provides weak evidence about which particular purchase or
scrap improved the game. At the existing low temperature, unlikely alternatives
also receive very few direct trials. A better critic cannot supply those
missing action comparisons by itself.

The next intervention is therefore a separate **paired terminal action-comparison
learner**, starting from verified champion 9 rather than continuing the stalled
optimizer. This is a hypothesis to test, not an established explanation of the
entire plateau.

## New branch

1. Play the frozen champion against its incumbent opponent with the deployed
   greedy policy. Uniform reservoir sampling retains two economic main-phase
   decisions per game, after routine card plays, and six ordinary positions for
   behavior preservation.
2. At each retained decision compare the actually selected action with up to
   three uniformly sampled distinct alternatives: acquire, scrap for ability,
   or end turn. Alternatives are not restricted to the model's top-ranked
   actions. Other decision families keep their existing behavior targets.
3. Sample 16 worlds consistent with the public information. For every action,
   copy the same world and reset the same per-seat random streams. Continue to
   the end with the correct frozen policy for each seat. No value-network leaf
   scores or source-game outcome labels are used. If any branch truncates, drop
   that chance sample for all actions.
4. Retain counts of paired wins and losses against the reference action. A tied
   outcome provides no preference gradient. Both directions of disagreement
   contribute to a logistic ranking objective; a noisy sample winner is never
   turned into a certain one-hot target. An alternative can receive a gradient
   even when its old sampling probability is effectively zero.
5. Fit only the final policy output layers, with a KL penalty against the frozen
   policy on comparison and ordinary positions. The critic, state features,
   action features, and all intermediate policy layers remain unchanged. This
   isolates the new signal and retains portable actor compatibility.
6. Split training and held-out data by whole source games, avoiding leakage
   between neighboring decisions. Use a fixed eight epochs and evaluate only
   that final checkpoint, not whichever checkpoint happens to win a screen.

The first trial uses 1,024 source games: at most 2,048 comparison roots and
131,072 terminal continuations. It has a three-hour cumulative budget and saves
dataset shards, models, optimizer state, manifests and progress. Resume retains
the sampled data and algorithm settings. Public-belief and continuation seeds
are reproducible; no simulator-hidden state is written to the dataset.

The campaign's exact historical rules-v1 runtime is frozen into the branch.
This avoids mixing a learner change with today's corrected engine or lethal
finisher. The existing campaign, champion and promotion error spending are
retained. This branch does not promote or replace anything automatically.

## Validation and next steps

`scripts/validate_counterfactual.py` first runs regression tests and a ten-game
real-engine smoke branch. It checks that learned checkpoints preserve all
non-output-layer tensors exactly. It then installs the forced-state recording
fix under the paused campaign lock and launches the bounded main trial.

The trial stops before arena evaluation if there is no paired training signal,
if the final held-out ranking loss fails to improve, or if held-out policy KL
exceeds 0.02. Otherwise it compares the deployed greedy candidate against
champion 9 on **4,096 fresh paired seeds / 8,192 games**. The fixed-pair interval
is exploratory evidence, not a campaign promotion certificate.

- Leave the original campaign paused while this runs.
- Read `data/astro6-counterfactual-20260920/validation.json` for job status and
  `trial/state.json` for live phase, dataset progress, held-out metrics and arena
  scores. Completion details go to `trial/result.json`.
- If the independent evaluation establishes an advantage, the next step is a
  fresh campaign promotion gate retaining the current incumbent's attempt
  count. Do not reset its error budget or reuse exploratory pairs.
- If the trial is negative or inconclusive, inspect label disagreements and
  held-out drift before spending another long training block. A successful
  software test alone is not evidence of stronger play.

To pause the separate trial, create `trial/STOP`. If it reaches its time budget
with useful work still pending, an explicit continuation is:

```sh
PYTHONPATH=backend ./.venv/bin/python scripts/counterfactual_experiment.py \
  --campaign data/progressive/20260910 \
  --output data/astro6-counterfactual-20260920/trial --resume --hours 6
```

This document is written before launching validation. No passing tests or new
strength gains are claimed yet.
