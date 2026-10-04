# Autopilot plateau: credit assignment and checkpoint selection

This revision is **staged, not installed**. The running campaign and the user's
10k ELO job retain their existing processes, configuration and frozen runtime.
Installation must wait for the user's `continue`, then a completed Autopilot
block boundary and a control-service reload.

## Findings

The gates did not demonstrate that a stronger candidate was being suppressed.
After the October 3 recovery, one completed gate scored 50.353% over 50,000
paired seeds, with a 49.773% lower bound. Three critics improved held-out
prediction loss but their matched policy differences were +0.029%, −0.008%
and +0.069%; none established stronger play. These results do not authorize
relaxing the promotion test or declaring a new champion.

The investigation found three limitations in how candidates were produced:

1. **Terminal-only credit.** The learner computed `outcome - V(state)` at every
   choice. With binary outcomes and probability-valued predictions, all actions
   in won games receive nonnegative advantages and all actions in lost games
   receive nonpositive advantages. Improving the critic changes their magnitude,
   not their sign. This is a valid Monte Carlo policy-gradient estimator, not a
   sign bug; it is a noisy way to distinguish decisions in games with roughly
   100 learner choices. The claim that it caused all stagnation is a hypothesis,
   not a proven causal result.
2. **Entropy escalation.** The prior recovery increased temperature and added
   a persistent entropy reward. In the 0.25-temperature lane, entropy rose from
   0.301 at the beginning of policy-00031 to 0.802 at the end of policy-00035.
   Corresponding sampled-game scores fell from 43.3% to 27.3%. The greedy screen
   scores did not improve. These are different rollout batches, not a controlled
   ablation, but they show that increasing entropy was not restoring strength.
3. **Final-checkpoint-only nomination and unconditional continuation.** Every
   1,000-game checkpoint and its optimizer were saved, but only the 20,000-game
   endpoint could be nominated. That endpoint became the lane parent even after
   rejection. The system could neither detect an earlier peak nor keep the
   parent after a weaker endpoint. No claim is made that any ignored checkpoint
   has already earned promotion; they had not been independently tested.

The investigation also checked several suspected implementation faults. Every
rollout iteration uses a new actor snapshot path, so worker caching does not
reuse the first policy throughout a block. Rollout and PPO distributions use
the same mean-head logits, mask and temperature. Source and frozen-runtime
engine, encoders, native actor, model, arena, critic and sequential-test modules
were byte identical. The gate's alpha spending and certification logic were
not changed by this revision.

## Staged changes

`--gae-lambda` adds an optional generalized advantage estimate over consecutive
learner decisions in each complete game. The proposed recipe uses lambda 0.95,
gamma 1, temperature 0.1 and zero entropy reward. For a nonterminal choice,
`delta = V(next learner decision) - V(current)`; the final delta is
`outcome - V(current)`. A backward pass calculates
`advantage = delta + lambda * next advantage`. Forced actions and opponent
turns occur between those learner decisions; values always refer to the
learner's own perspective. No trace crosses a game boundary. Truncated games
remain excluded.

The independent critic is frozen before collection and advantage computation.
Lambda 1 exactly retains the old Monte Carlo calculation and remains the CLI
default. Smaller lambda introduces dependence on local critic accuracy; it is
a bias/variance tradeoff, not a guarantee of stronger play. This follows
[Schulman et al., Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
New learner manifests use version 5. Completed checkpoints preserve optimizer,
RNG and estimator settings for resume.

`temporal_credit=true` selects this recipe for both policy training and both
matched critic-probe arms. Lanes from a different recipe restart from the
verified champion instead of inheriting incompatible optimizer momentum.

`policy_checkpoint_games=[5000,10000,20000]` enables greedy checkpoint selection.
All three frozen candidates and, when different, the lane parent are evaluated
against the incumbent on common exploratory screen seeds. The best valid
checkpoint is saved in the pending task before any gate starts. Ties prefer
the earlier checkpoint. Resume uses that exact selection and existing evidence.
The gate uses an independently derived seed stream and retains the same alpha
spending, lower-bound requirement and historical-panel tests.

The lane advances only if the selected checkpoint beats both 50% and its parent
on the exploratory comparison without truncations. Otherwise it retains the
parent. This is a search decision, not certification. Optimizer and cumulative
game count follow the selected checkpoint; rejected endpoints no longer become
parents automatically. The three-block continuation limit still applies.

## Evidence and validation

An isolated 32-game audit collected 3,516 decisions. With the accepted critic,
temporal estimates changed the sign of 24.86% of advantages relative to Monte
Carlo. They assigned positive credit to 385 decisions in lost games and negative
credit to 489 decisions in won games. The cross-position standard deviation
changed from 0.4215 to 0.1630. This measures the changed learning signal; it does
not establish the true quality of those decisions, the variance of the gradient
estimator, or playing-strength improvement.

Regression tests cover exact lambda-1 compatibility, temporal arithmetic,
terminal boundaries, invalid/truncated inputs, recipe rebasing, intermediate
selection, optimizer pairing, selection persistence across resume, independent
gate seeds, failure to promote from favorable selection alone, and audited
runtime migration. An isolated real-game training/resume check exercises the
new estimator, checkpoints, optimizer, actor export and replay export using
one worker. These execution checks are not strength certification.

Audit results and isolated runs are in `data/autopilot-diagnosis-20261004/`.
`settings-staged.json` contains only the intended configuration changes.
`maintain_autopilot_runtime.py` now explicitly includes the learner module and
script in its backed-up migration set. It still refuses a running campaign or
unfinished block and preserves champion/evaluation hashes and spent attempts.

No installation, pause, drain, termination, API restart or live configuration
change was performed during this investigation. After authorization to continue,
finish the current block, install the audited revision, reload the API and run
fresh qualification. Further claims about successful promotion must wait for
those results.
