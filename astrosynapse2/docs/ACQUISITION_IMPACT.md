# Acquisition impact: current champion versus MiniAstro1000

Open `http://127.0.0.1:3000/acquisition-impact`, also linked from the control center
and Acquire students. The page contains signed effect charts with uncertainty,
frequency-versus-impact comparisons, a stage/budget heatmap, and a position
inspector with deck composition, candidate scores and paired rollout outcomes.

## Result of the completed experiment

Reference: Arch3 autopilot `policy-00004`, model
`auto-828813dc45d24cf086e019c6537d1b02-pd9dd811c376c`, frozen before collection.
Student: `669746289fef4e71ad5caf056559d2e6`, the 956-parameter MiniAstro continued
on 100,000 gen10 games. These are not the same reference models as the student's
original imitation task or the earlier 500-game arena.

The 1,000 source games contained 19,444 eligible acquisition decisions and 2,903
disagreements (14.93%). The uniform within-game reservoir selected 1,655 positions
for 64 paired continuations each: 105,920 pairs / 211,840 branches. No pairs were
discarded. Free-acquisition and mixed play/purchase decisions are outside this test.

| Comparison | Estimated champion-choice advantage | 95% game-bootstrap interval |
| --- | ---: | --- |
| All disagreements | +0.76 percentage points | +0.34 to +1.20 |
| Different purchases | +1.17 pp | +0.58 to +1.75 |
| Champion stops; student buys | +0.35 pp | −0.45 to +1.12 |
| Champion buys; student stops | −0.30 pp | −1.23 to +0.66 |
| Disagreements on turns 1–8 | +1.21 pp | +0.51 to +1.89 |

The measured average advantage is modest. Choosing between purchases has the
clearest aggregate positive signal. The two buy/stop categories have intervals
spanning zero; this does not establish equivalence or mean that all their cases
are unimportant.

A useful exploratory follow-up is **stop versus Federation Shuttle**. In 29 tested
positions from 27 games, when the champion stopped and MiniAstro wanted the
Shuttle, the mean advantage of stopping was +4.49 pp (interval +2.15 to +7.05).
Stop versus Trade Pod is another candidate: +2.41 pp (+0.39 to +4.52), from 32
positions in 28 games. These are selected findings among many comparisons, so
fresh positions are needed before promoting them to general strategy rules.

## Estimand and uncertainty

Each measured effect is `P(win | force champion choice, then champion play)` minus
`P(win | force student choice, then champion play)`, conditional on the sampled
public position and the engine's public-belief sampler. Both players use the
reference champion after the first action. The two branches share each sampled
world and random streams. All negative estimates are preserved.

The source distribution is champion self-play, restricted to MAIN decisions
whose exposed actions are only paid acquisitions or ending the turn, with at
least two choices. “Buy nothing” therefore has a clear executable meaning here.
It is not the student's original random-moment future-acquisition target.

Every sampled position is weighted by the inverse of its reservoir inclusion
probability. Games with many disagreements contribute correspondingly more total
weight even though only two positions per game were tested. Overall weighted
sample mass is exactly 2,903 disagreements. Frequencies come from all 19,444
eligible decisions, not just sampled roots. Subgroup means use weighted ratios;
estimated subgroup sample mass need not exactly equal its observed frequency.

Group intervals use 2,000 bootstrap resamples of whole source games. All roots,
both seats and paired outcomes from a selected game remain together. Per-position
intervals use approximate conservative paired bounds formed from Bonferroni
Wilson intervals for the two discordant outcome probabilities. No interval is
adjusted for selecting the most striking result among many comparisons.

Comparison tables are ordered by weighted signed local effect divided by total
eligible opportunities. This prioritizes common costly disagreements, but is
**not a prediction of full-game arena loss**. Local effects cannot simply be
added along a game, and the student visits a different state distribution when
it controls repeated choices.

## Reproduce and access

Raw experiment: `data/miniastro_regret/champion-pd9dd811c376c-1000games-64pairs/`.
The prepared `analysis.json` retains aggregate estimates and every sampled
position's summary. Raw game files retain detailed observations and paired
binary outcomes. The GUI downloads the prepared report.

Rebuild the report from the project directory:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=backend .venv/bin/python -m astro2.acquisition_impact data/miniastro_regret/champion-pd9dd811c376c-1000games-64pairs
```

Read-only endpoints:

- `GET /api/acquisition-impact`: prepared experiments.
- `GET /api/acquisition-impact/{id}`: aggregate estimates and sampled positions.
- `GET /api/acquisition-impact/{id}/positions/{game}/{root}`: readable position.
- `GET /api/acquisition-impact/{id}/download`: downloadable analysis JSON.
