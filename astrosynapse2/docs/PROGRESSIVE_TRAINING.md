# Progressive champion training

Open [Astro6 progress](http://127.0.0.1:3000/progressive) in the Astrosynapse 2 UI.
The learner and supervisor continue independently of the browser. **Pause &
save** finishes the current learning iteration or evaluation batch. **Resume
training** restores the frozen code, model, optimizer, counters, and random state.

The September 10 run starts from the best screened full-policy candidate in the
four-hour assessment campaign. It has a 48-hour cumulative budget, eight CPU
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

# Resume the same run; its original cumulative 48-hour limit remains active.
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
