# Astro6 autonomous improvement — September 20

The old campaign repeatedly extended one low-temperature, low-learning-rate
branch. Champion 9 survived 307 promotion attempts and more than 100 hours
without a verified successor. The critic fixes and the terminal counterfactual
experiment did not establish an improvement: the latter's two 8,192-pair tests
scored 50.40% and 49.87% against champion 9.

The replacement is a persistent portfolio supervisor, `scripts/autonomous_training.py`.
It starts from champion 9 with fresh optimizers and rotates four curricula:

| Curriculum | Sampling temperature | Initial learning rate | Policy advantage baseline |
| --- | ---: | ---: | --- |
| League exploration | 0.12 | 2e-6 | Separate critic |
| Outcome-only learning | 0.08 | 1e-6 | Constant 50% |
| Broad exploration | 0.25 | 4e-6 | Separate critic |
| Conservative policy heads | 0.05 | 2e-6 | Constant 50% |

All four use terminal outcomes from fresh complete games. The constant-baseline
branches remove the win-probability estimate from policy credit assignment.
The independent critic continues learning and reporting pre-fit calibration,
including forced decisions. Policy updates have categorical KL checks, rollback,
clipping, and adaptive learning rates. The full-policy recipes include a small
entropy bonus. Repeated portfolio cycles change seeds and initial learning rates.

Every branch trains in 4,096-game blocks. Two consecutive screens at or below
50% retire it early. The default cap is four blocks (16,384 games), adjustable
in the GUI. Subsequent cycles can seed a recipe from its strongest independently
confirmed checkpoint (at least 2,048 pairs and 50.5%, excluding candidates that
subsequently lost their promotion test). This lets promising learning accumulate
across trials without calling that checkpoint a champion. Every third cycle
returns to the verified incumbent. Each trial gets a fresh optimizer.

## Historical experience

The SQLite store contains 627,658 metric records, 1,889 checkpoint entries, and
1,924 arena-job entries, not complete game transcripts. Compatible retained
replay stores contain 10,201,296 observation rows in 1,265 live columnar shards
across three runs. These are retained rows, not a count of unique games.
Deleted shards referenced by old manifests are excluded.

The system samples 8,136 multi-action observations across runs and time, using
memory maps rather than loading the archive into RAM. The incumbent relabels
these legal choices. A small KL regularizer on this sample protects behavior
on historical positions while fresh on-policy returns provide reward gradients.
Old rewards and advantages are not treated as current-policy targets. Each new
champion gets a new hashed teacher dataset. Training opponents are 75% incumbent
and 25% past verified champions, balanced by seat.

## Promotion and recovery

A 512-pair greedy screen selects candidates for a separate 2,048-pair confirmation.
A confirmation score of at least 51% starts an entirely fresh promotion test.
Screen and confirmation results never count toward promotion confidence.

The incumbent's already-spent attempt count is inherited: the next test is 308,
not 1. The old partial attempt 307 is recorded as abandoned, with its budget
still spent and original evidence left intact. The promotion test preserves the
existing time-uniform confidence sequence and alpha spending
`0.05 / (attempt * (attempt + 1))`. Its larger cap, 131,072 pairs, gives small real
improvements more opportunity to establish evidence at this stricter threshold.
Predeclared futility stops are below 49% after 2,048 pairs, at or below 50% after
8,192, or below 50.5% after 32,768. A new incumbent resets the attempt counter.
Each accepted champion is also benchmarked over 2,048 pairs against the original.

The historical game engine and rules v1 are frozen and inherited. Manual arena
and Play remain on corrected rules v2; training promotions are historical-rules
evidence. Runtime identities, dataset/opponent hashes, optimizer/RNG state,
paired game outcomes, branch decisions, and evaluation identities are retained.
Atomic state transitions and contiguous evaluation prefixes support pause/resume.
A torn final evaluation append is recovered without replaying completed pairs.
Three consecutive learner failures stop visibly rather than silently consuming
resources. A 12 GiB free-disk guard bounds storage risk; unscreened intermediate
checkpoints are removed, while screened candidates and evidence are retained.

## GUI and operation

Open <http://127.0.0.1:3000/progressive>. It shows branches, decisions, screens,
independent gates, replay coverage, learning diagnostics, and critic calibration.
Controls support pause/save, resume, skipping a training branch, worker count,
total runtime budget, and maximum blocks per branch. Worker changes apply at
block boundaries. A promotion gate can be paused but cannot be skipped into a
new test with fresh seeds. Retained branch checkpoints appear in Models & Arena
and Play.

The new campaign uses eight workers and a 96-hour cumulative active-time budget.
Closing the dashboard does not stop it. Extend the budget from the GUI as needed.
The manager keeps the Mac awake while running, but a reboot requires Resume.

```sh
PYTHONPATH=backend .venv/bin/python scripts/autonomous_training.py \
  --output data/progressive/autonomous-20260920 --resume
```

Validation artifacts are in `data/autonomy-validation-20260920/`. Validation
includes replay gradients, seat-balanced curricula, inherited error spending,
durable gate completion, torn append recovery, branch retirement, GUI control
APIs, actual learner updates/resume, and a bounded end-to-end supervisor run.
The focused suite passes 34 tests. The real supervisor validation completed 272
training games across 17 bounded branches, including a fresh 256-pair validation
gate, with no learner errors. The full campaign's first 1,024 games also saved
and paused successfully. An audited supervisor-only revision adds the parent
selection rule; the runtime revision preserves its engine, checkpoint, optimizer,
random state, and promotion attempt count.
Small validation matches check execution, not strength. The new system still
needs independent wins to demonstrate improvement; a future promotion is not
assumed or guaranteed.
