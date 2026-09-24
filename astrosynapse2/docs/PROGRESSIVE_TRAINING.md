# Progressive champion training

The active replacement is the [September 23 direct policy search](astro6-evolution-2026-09-23.md).
It follows the unsuccessful [September 20 PPO supervisor](astro6-autonomous-2026-09-20.md)
and searches portable greedy policies with independent promotion certification.
It has produced the first new verified promotion to champion 10: 50.389% against
champion 9 over 130,944 paired seeds, with a time-uniform lower bound of 50.00197%.
The autonomous search continues against champion 10 under historical rules v1.
The same progress page supplies pause/resume and resource controls. Earlier campaigns described
below remain paused as historical evidence; their settings are not the active
evolution campaign's settings.

Open [Astro6 progress](http://127.0.0.1:3000/progressive) in the Astrosynapse 2 UI.
The learner and supervisor continue independently of the browser. **Pause &
save** finishes the current learning iteration or evaluation batch. **Resume
training** restores the frozen code, model, optimizer, counters, and random state.

The September 10 run starts from the best screened full-policy candidate in the
four-hour assessment campaign. Its saved cumulative budget is now 144 hours, with eight CPU
actors, one Metal learner, fresh trajectories, and no replay or search labels.
It screens every 8,192 games against the incumbent. Screening results are
exploratory; only a fresh independent verification can promote a checkpoint.

Promotion requires a time-uniform lower confidence bound above 50%. A bounded
betting mixture makes repeated looks valid. Error spending across candidates
against each incumbent preserves an overall 5% error budget for that succession
test. This is at least 95% confidence for each verified succession, including
small improvements. Gates stop on sufficient evidence, futility, or 32,768
paired seeds. Failure to promote means improvement was not established, not
that the model is necessarily worse.

Each accepted model becomes the next training opponent and the source of the
next learner stage. A separate 2,048-pair tournament tracks its strength against
the original champion. A succession of wins is not assumed to be transitive.
The eventual 70% target is judged against the original champions, separately
from these incremental promotion tests. Historical rules v1 remain active in the frozen training campaign. New manual
arena comparisons, Scrap/Acquire Elo probes (including bucketed Acquire), and
Play matches in the control center use corrected rules v2. Saved results retain
their original rules version; historical promotions are not corrected-rules evidence.

Artifacts are under `data/progressive/20260910/`. `state.json` records phase,
lineage, gates, and promotions. Each stage contains training metrics and retained
checkpoints. Each gate contains immutable model hashes, seed, paired outcomes,
and confidence evidence. `runtime/` is the frozen source used on resume.

Terminal controls from the project directory:

```sh
# Request a safe pause.
touch data/progressive/20260910/STOP

# Resume the same run using its saved cumulative budget (currently 144 hours).
PYTHONPATH=backend ./.venv/bin/python scripts/progressive_training.py \
  --output data/progressive/20260910 --resume
```

The dashboard's new champion lineage is maintained by this supervisor. Existing
database champions are not overwritten. Retained actors are discovered automatically in Models & Arena and Play, grouped
by Astro6 run with stage/checkpoint names and promoted generation numbers. The
progress page shows the current run, learner checkpoint, and champion checkpoint.
These models are read-only entries: the progressive supervisor owns retention,
and discovery does not add them to the older trainer’s league or change champions.
The assessment and all measured experiments are in
[the September report](assessment-2026-09-09.md).

The [108-hour review](astro6-review-2026-09-15.md) records the September 15
operational fixes. Resume now inherits saved hours and workers unless explicitly
overridden; an override is saved for subsequent resumes. Interrupted original-
champion benchmarks resume before learning, and already-decided gates do not
collect additional batches. Learner metrics separate rollout, learning,
checkpoint, and evaluation wall times. Historical evaluation `seconds` sums
worker task durations and is not elapsed wall time.

The paused campaign received an audited revision of its two execution scripts.
Its original runtime and manifests remain in `runtime-revisions/`; both the
supervisor and active learner manifests record the revision. Completed-stage
manifests retain their historical identities. Model, optimizer, seeds, rules,
and promotion thresholds were preserved. `scripts/maintain_progressive_runtime.py`
performs this restricted maintenance under the campaign lock, verifies the
existing identity, and refuses an unpaused or unexpectedly modified runtime.

The [September 19 critic correction](astro6-critic-2026-09-19.md) addresses
missing forced-action value examples and separates critic optimization from
policy clipping and KL control. Its validation job installs an explicit audited
revision only after checks complete; `data/astro6-critic-20260919/result.json`
records the outcome. Installation leaves the campaign paused and retains its
champion and promotion evidence. New learner checkpoints include a separate
critic optimizer sidecar and pre-fit calibration metrics.

The [September 20 follow-up](astro6-counterfactual-2026-09-20.md) records the
unsuccessful additional training block, corrects recording of engine-forced
positions, and introduces a separate terminal action-comparison branch from
champion 9. Its bounded trial and independent arena do not modify champion
succession. The original campaign is paused while this intervention is tested.
