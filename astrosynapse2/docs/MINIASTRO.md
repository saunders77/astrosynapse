# MiniAstro acquisition students

MiniAstro predicts the next acquisition in the current turn, including Explorer and
None. It is a shallow neural candidate scorer designed to expose its arithmetic.
Open **Acquire students**, choose **MiniAstro**, select a checkpoint and parameter
budget, then train. The default is 10,000 parameters and 10,000 games. Existing
completed students can supply the same frozen dataset for comparisons without
simulating games again. Tree students remain available.

## Architecture

For every available candidate, a shared network computes:

```
score(candidate) = linear(candidate inputs)
                + Economy(candidate, economy inputs)
                + Deck composition(candidate, deck inputs)
                + Timing(candidate, timing inputs)
                + Matchup(candidate, matchup inputs)
                + Market(candidate, market inputs)
```

Each group is exactly one hidden layer: `sum(v * ReLU(x @ W + b))`. There are no
attention layers, residual blocks, learned embeddings, or recurrent state. Highest
score wins. The linear term and five group contributions sum exactly to the score.
All hidden-unit contributions and weights are accessible. Group names specify
input restrictions; they are **not discovered concepts or causal explanations**.
Several groups share inputs, so their allocations are not unique. A 1,000-parameter
model is easier to inspect, but even a tiny neural model need not yield a concise
English strategy.

Inputs extend the tree's human-readable feature vocabulary with exact public card
counts in each deck and the market, plus candidate copies in hand/discard/play.
Opponent composition means public card ownership, not hidden card locations or deck
order. The only hindsight input remains the explicitly permitted full-turn trade.
Inputs use signed log1p, training-only mean and standard deviation (floor 0.1), then
clipping to ±6. Normalization constants are stored separately from trainable weights.

The design uses the additive interpretability principle of
[Neural Additive Models](https://arxiv.org/abs/2004.13912), but groups related inputs
rather than assigning a separate function to every feature. This permits useful
candidate/context interactions while retaining exact group score decompositions.

## Training and evaluation

Sampling, targets, candidate feasibility and full-turn trade follow
[Acquire students](ACQUIRE_STUDENTS.md). One uniformly sampled decision moment per
turn receives the actual next acquisition as its target. Training uses masked
softmax cross-entropy over the available candidates, giving each moment equal
weight. These are behavioral labels, not teacher logits or win-value targets.

Whole games are split 70% training, 15% validation, 15% test. Reusing a dataset
preserves those exact assignments, regardless of the new job's seed. AdamW trains
for up to the requested epochs (default 40), with validation-based checkpoint
selection and patience 8. The test set does not select epochs. Inference is NumPy;
MLX is used only for fitting. Exported metrics use the same NumPy scorer as Play.

All-turn agreement counts unavailable future market targets as misses. Covered
agreement excludes these targets. Purchase agreement is measured on covered
non-None targets, reported separately so correct None predictions cannot disguise
weak purchase selection. Fidelity is agreement with the teacher's behavior, not
playing strength or proof that its choices are optimal. Live results also depend
on the accuracy of the user's full-turn trade estimate.

Three initial runs reused the gen10 10,000-game dataset, including 35,183 held-out
test moments from 1,500 games (99.15% candidate coverage):

| Actual trainable parameters | Hidden widths | All-turn test agreement | Covered purchase agreement |
| ---: | --- | ---: | ---: |
| 956 | 2 / 2 / 2 / 1 / 1 | 88.09% | 86.70% |
| 9,987 | 18 / 17 / 17 / 17 / 17 | 90.04% | 89.18% |
| 99,930 | 173 / 173 / 173 / 173 / 172 | 90.09% | 88.79% |

10,000 is the practical default: the larger run provided negligible extra overall
agreement and lower purchase agreement. These are single-seed results, not promises
for other checkpoints. A further 500 fresh games (11,851 sampled turns), with all
weights frozen, yielded 87.21%, 89.31%, and 89.38% all-turn agreement respectively;
covered purchase agreement was 85.58%, 88.17%, and 87.99%.
`scripts/benchmark_miniastro.py --source <student-id>` repeats
the comparison. `scripts/audit_miniastro.py --students <ids...>` evaluates frozen
students on new teacher-game seeds and attaches the results to their GUI reports.
Run either with `PYTHONPATH=backend .venv/bin/python` from the project directory.

## Inspection and later failure analysis

The Students page shows architecture, validation progress, every input weight for
a selected hidden unit, and held-out mistakes. Both Play experiences support
MiniAstro in the student selector, showing candidate scores and the contribution
differences between the top two choices. Softmax shares are model choice shares,
not calibrated confidence or win probabilities.

Each completed job retains `student.json`, `miniastro.npz` (weights, train-only
normalizers), `report.txt`, `errors.jsonl.gz`, and the source samples and frozen
teacher. The error set contains all held-out mistakes with the sampled observation,
true target, prediction, and exclusions for later concept discovery. Export the
JSON and weights together; `weights_path` in the JSON must point to the weights
file in its destination when using the Python recommendation function.

API additions:

- Create with `student_type: "miniastro"`, `parameter_budget: 1000|10000|100000`,
  `epochs: 1..200`, and optional `source_student_id`. Reuse requires the same frozen
  teacher, number of games, and rules.
- `GET /api/acquire-students/<id>/unit?group=0&unit=0` returns the exact selected
  hidden unit, all input weights, and normalization constants.
- Download kinds `weights`, `errors`, and `report` join `student` and `samples`.
- Existing recommendation routes automatically dispatch to the selected architecture.

Dataset features are cached in disk-backed arrays keyed by sample-file SHA-256 and
feature version. This keeps repeated budgets fast without loading all observations
as Python objects. Workers remain detached from the API and cancellation is
cooperative, as for tree students.
