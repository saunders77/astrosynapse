# Independent critic training

Open **Critic training** in the control center, or `/critics`. Choose a frozen
playing checkpoint, number of self-play games, epochs, network width and rules.
Rules v2 matches Play; v1 is available for historical campaign comparisons.
Start, pause and resume jobs from the page. Closing the browser or restarting
the API does not stop the detached worker. Only one critic job runs at a time.

The critic is a separate NumPy MLP: public state plus a decision-family one-hot,
two tanh hidden layers (width and width/2), and a scalar sigmoid output. All of
its parameters learn; it does not depend on the policy's learned features. The
128-wide configuration has 174,849 parameters with the current 1,292-state,
eight-family encoder. It trains on one CPU worker, without a Metal context.

## Data and measurement

Both seats use the frozen checkpoint's deployed greedy policy and the selected
engine rules. Seeds and seat starting positions are deterministic. Up to 64
uniformly sampled decision boundaries per game include forced choices. The
engine hook supplies the observation before the selected action is executed.
Labels are terminal results from the observed player's perspective. Truncated
games contribute no labels. No hidden state or future features enter inputs.

Whole games are split 70/15/15 before training. Input normalization is fitted
only on training games, with equal weight per game. Each game supplies one
mean-loss update per epoch. Validation log loss chooses the checkpoint; final
test games are evaluated only after training finishes and never choose an epoch.
Brier error, log loss and calibration bins compare the independent and original
critics on exactly the same sampled positions. Metrics weight each completed
game equally. Slice reports include forced/choice decisions, early/late global
turns (boundary 10), and decision families.

This first version measures frozen-policy self-play. It does not establish
calibration against human or different model opponents, tactical certainty, or
stronger play. Calibration bins contain correlated positions and are not
independent-sample confidence intervals. A tiny smoke run is an execution test,
not evidence of improved predictions. Selectively overrepresented tactical
positions or new opponent mixtures require separate evaluation.

## Saved artifacts and recovery

Jobs live under `data/critics/<id>/`. A copied policy and its SHA-256 identify
the data-generating policy. Individual game shards are saved atomically. The
checkpoint contains network parameters, Adam moments, normalization, completed
epoch, learning history and best validation weights. Pausing during an epoch
restarts that epoch from its last saved boundary, preserving deterministic
training. Completed games are never recollected. Resume rejects a changed
policy copy. A worker lock prevents duplicate execution; a dead worker becomes
interrupted and can resume. `worker.log` records failures.

The GUI provides downloads of `critic.npz` and `result.json`. Training does not
activate the critic in Play or modify actor checkpoints. Deployment should be
an explicit subsequent integration after reviewing held-out results.

For independent inference:

```python
import numpy as np
from astro2.critic import IndependentCritic
from astro2.engine_encoding import EngineEncoder

critic, metadata, _ = IndependentCritic.load("critic.npz")
encoder = EngineEncoder(version=metadata["encoder_version"])
encoded = encoder.encode_decision(decision.observation, decision)
family = np.eye(metadata["families"], dtype=np.float32)[int(encoded.family)]
x = np.concatenate((encoded.state, family))[None, :]
probability = float(critic.predict(x)[0])
```

API: `GET/POST /api/critics`, `GET /api/critics/{id}`, and
`POST /api/critics/{id}/pause` or `/resume`. Downloads use
`GET /api/critics/{id}/download/{critic|report|log|splits}`.
