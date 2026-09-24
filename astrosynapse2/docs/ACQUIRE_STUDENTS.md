# Acquire students

Open **Acquire students** from the control center or Astro6 progress page, or visit
`http://127.0.0.1:3000/students`. Select any retained playable checkpoint. The default
run is 1,000 games, corrected rules v2, and at most 11 total tree nodes (including
leaves). Historical rules v1 can be selected explicitly. An odd node budget up to
99 is supported; maximum path depth is seven tests. This is one weighted CART
regression tree, applied to every candidate, followed by a transparent maximum-score
selection. There are no neural scores or additional learned models at inference.

## Sampling and labels

Both seats use the frozen checkpoint's deployed greedy policy. A moment is the
instant immediately **before** an engine decision action, including forced actions
and nested ability decisions. Each turn uses independent uniform reservoir sampling
over all its decision moments. The selected action at the sampled boundary is
included in the future. Every recorded turn contributes exactly one sample; long
turns do not receive extra weight. Turns with multiple acquisitions remain eligible.

The target is the next actual paid or free acquisition at or after that boundary,
before the same turn ends. None means no further acquisition on that turn. Free
acquisitions use the engine action's target card, not its source card. An unfinished
final turn of a truncated game is retained but censored, never labeled as a completed
None example. Completed turns preceding a truncation remain eligible.

Candidates are distinct card identities in the visible market, Explorer if in supply,
and None. Paid cards require enough full-turn trade remaining after previous spending.
Row ships are also retained when the player's cards contain a free-ship ability:
this is a conservative forecast of possible acquisition later, not immediate legality.
Duplicates collapse because the target is a card identity, not a market slot.

A card first revealed by a later market replacement is impossible to name from the
sample's candidate set. Keep that true target and observation in the dataset, exclude
it from fitting, and count it as incorrect in **all-turn agreement**. Do not resample
the turn or silently convert it to None. Other uncovered targets are counted separately.

## The permitted future input

`total_trade` is total trade actually generated over the entire teacher turn, before
subtracting any purchases. It includes later draws and resource abilities. The recorder
tracks the maximum of current trade plus already paid purchase costs through the turn.
Trade gains are nonnegative in these rules. Features include the sampled moment's
current trade pool and spending **before** that moment, so money already spent cannot
be spent again. No future market, future hand, selected action, or outcome is a feature.

This is a hindsight-assisted imitation task. It does not claim to predict the unknown
future resource total. In both Play experiences, select a saved student and enter the
full-turn total or an estimate. The simulated game supplies spending automatically;
the tabletop companion has an explicit spending input. Inputs reset each turn, while
the selected student is remembered. Recommendations update at any decision moment,
including before all cards are played, and do not execute actions. The explanation
shows the winning candidate's tree path and every candidate's score and leaf.

## Fitting and interpretation

Human-readable features describe the candidate, current trade budget, deck composition,
factions, discard/draw counts, bases, authority, and market alternatives. A candidate's
target is one if it is the acquired card, zero otherwise. Candidate rows have weight
`1 / number_of_candidates`, giving each sampled turn total weight one.

The implementation uses deterministic best-first CART with weighted squared-error
reduction, at most 32 quantile thresholds per feature, a minimum leaf weight, and a
hard depth and total-node limit. Leaf scores are weighted imitation frequencies,
**not win probabilities**. The highest score wins; ties prefer None, then cheaper
cards, then alphabetical names. The entire selection rule accompanies the English
tree so the displayed explanation matches executable behavior.

Source games are split 70/15/15, keeping both seats and every turn of a game together.
Trees of up to 3, 7, 11, and the requested maximum nodes are fitted on training games;
validation agreement selects the size, with ties favoring smaller trees. The final test
games are never used for fitting or size selection. Reports separate purchase and None
agreement, candidate coverage, covered and all-turn agreement, and an always-None
baseline. These describe imitation fidelity, not strategic value or playing strength.

## Saved jobs and API

Jobs run as independent local worker processes, one at a time. Browser and API restarts
do not stop them. Cancel uses a cooperative stop file checked inside game decisions and
between fitting stages. Failed/interrupted jobs remain visible. A new run does not
replace or train the teacher. Its actor is copied and hashed before collection, so
checkpoint pruning cannot change a running job.

Artifacts live in `data/acquire_students/<id>/`: atomic `job.json` progress, immutable
`teacher.actor.npz`, `samples.jsonl.gz` with visible observations and sampled moment
indices, portable `student.json`, English `tree.txt`, and `worker.log`.

- `POST /api/acquire-students`: `model_id`, `games`, `seed`, `max_nodes`, `rules_version`.
- `GET /api/acquire-students` and `GET /api/acquire-students/<id>`: history and progress.
- `POST /api/acquire-students/<id>/cancel`: cooperative cancellation.
- `POST /api/acquire-students/<id>/recommend`: validated advisor `observation`,
  `total_trade`, and `spent`.
- `GET /api/acquire-students/<id>/download/{tree|student|samples}`: completed artifacts.
