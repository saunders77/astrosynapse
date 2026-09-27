# Astro6: restore the path to independent promotion testing

The September 26 investigation found **182 completed generations, 3,550,720
search games, 66 independently checked nominees, zero accepted steps and zero
promotion attempts against champion 10**. The campaign was safely paused in
generation 199. Champion 10's earlier verified promotion remains intact.

The failure was search stagnation compounded by the nomination funnel. It was
not a crashed worker, a lost promotion, or a demonstrated error in the betting
test. The 66 selected nominees averaged **+0.5371 percentage points** on selection
games and **−0.0405 points** on independent adoption games. All four mutation
families had nonpositive average independent gains. Short, noisy selection
produced optimistic winners; independent checks rejected them and the next
generation restarted from the same champion. There is no evidence that all those
rejected candidates deserved promotion.

Every nominee also had to pass a 4,096-pair adoption screen and a separate
4,096-pair confirmation screen before it could reach the much larger promotion
test. When the retained parent is the incumbent itself, these extra requirements
can exclude small improvements without allowing the certification test to resolve
them. Simply extending the number of unchanged generations had already failed.

The diagnosis and hashes of preserved champion and gate artifacts are in
`data/evolution-20260926/diagnosis.json`.

## Revision 3

- Increase population screening from 128 to **512 pairs** per actor and fresh
  finalist selection from 1,024 to **2,048 pairs**. These remain exploratory data.
- When the parent has the incumbent's exact actor hash, send one supported,
  frozen nominee directly from fresh selection to the separately seeded
  promotion test. Persist the spent attempt and pending test together. Neither
  champion nor learner changes on nomination; a failed test changes neither.
- Explore bounded continuations of the last verified policy direction and the
  latest accepted learner step, alongside existing mutations. Each direction is
  attempted once per parent and endpoint identity. Tensors, model spec and
  endpoint hashes are checked; critic tensors are preserved. These proposals
  remain hypotheses, not evidence of strength.
- If the learner differs from the champion, validate the nominated step on up to
  four independent 4,096-pair blocks. Stop on sufficient descriptive support or
  a nonpositive gain; an uncertain positive nominee can receive more data. The
  candidate cannot change between blocks. Resume reconstructs the same recorded
  prefix, and the 8,192-pair confirmation and promotion streams remain separate.
- Stop a promotion test with a nonpositive observed score after 16,384 pairs,
  in addition to the existing clearly negative early stop. Positive tests retain
  their full **131,072-pair** budget. This only changes futility stopping, not
  acceptance or the error budget.

The promotion calculation, alpha spending, seed derivation, historical game
engine and pairing rules are unchanged. Attempt `a` still spends
`0.05 / (a * (a + 1))` for the current incumbent. Search, pilot, selection,
adoption and confirmation data never count toward promotion evidence. A failed
attempt remains spent. The statistical guarantee is per incumbent's succession
test, not an unlimited succession of champions.

Direct nomination trades the former bottleneck for more certification work and
potentially more spent attempts. Later attempts consequently receive smaller
alpha. This does not relax the promotion threshold; the 24-hour run is still
an empirical test of whether the search can find enough signal.

## Validation and installation

All **472 backend tests** passed with native Metal access. Focused regression
tests cover independent nomination, atomic attempt spending, unchanged champions
before certification, direction reproducibility and portability, preserved
critics, no repeated line search for the same parent, bounded adoption,
pause/resume and separate block identities. The frontend build and five render
tests passed, along with lint and type checks. The progress-page explanation now
describes the new route; resource controls accept fractional cumulative hours.

A real historical-engine integration completed two generations and **448 games**
without changing the champion. Its result is
`data/evolution-20260926/integration-result.json`.

An additional isolated real-game control exercised the certification path with
identical policies: pause at 128 pairs, resume to 256, preserve the exact prefix
and contiguous unique indices, and retain the spent attempt. It correctly
declined promotion at 50.000% with a 48.348% lower bound and zero truncations.
This used 512 games and never touched the production champion or attempt budget.
The audit is `data/evolution-20260926/gate-resume-result.json`.

A separate real-game pilot tested eight policy-direction proposals on **1,024
paired seeds each (16,384 games total)** with no truncations. The best exploratory
score was 50.488%, with a paired standard error of 0.854 percentage points;
that is not persuasive evidence of improvement. A large extrapolation regressed
to 44.580%, motivating bounded smaller steps. The pilot establishes execution
and artifact validity, not a successful new champion. Its immutable outcomes and
manifest are in `data/evolution-20260926/lineage-screen/`.

Installation used `scripts/maintain_progressive_runtime.py` while paused and
under the campaign lock. Runtime revision **1790468167889308000** changed only
`astro2/evolution.py` and `scripts/evolution_training.py`. The old runtime,
manifest, state and resource settings are retained in that revision directory.
All 11 audited champion, promotion-test and anchor-test files remained byte
identical, and the state was unchanged during installation. The unfinished
version-2 generation 199 retains its original plan and samples; subsequent
generations use version 3. The installation and resource audit is
`data/evolution-20260926/installation-v3.json`.

## The next run

The same campaign was resumed through its local control API with **eight workers
and 24 additional active hours**. Because hours are cumulative, the saved total
is **60.990273970486335 hours**, starting from 36.990273970486335 elapsed hours.
The generation limit is 4,096, leaving the 24-hour time limit as the practical
stop. Pauses preserve pending tests and their evidence. The supervisor runs
independently of the browser and uses `caffeinate`.

The live supervisor completed the saved generation 199 and entered **generation
200 under revision 3**, with a fresh heartbeat and no error. The control API
confirmed it is running with the intended settings; the snapshot and estimated
finish time are in `data/evolution-20260926/launch-status.json`.

Watch [Astro6 progress](http://127.0.0.1:3000/progressive). Reaching a promotion
test is now possible without three successive short significance screens.
**New valid promotions still require the candidates to win the unchanged
certification test; this repair cannot promise one within 24 hours.** The active
campaign continues to use historical **rules v1**. Its results must not be pooled
with manual Play/Arena results under corrected rules v2.
