# Astro6 plateau and critic correction

The paused campaign has accumulated 200.669 active hours and 11,329,536 games.
Champion 9 was promoted at 5,144,576 games. The current learner has therefore
played another **6,184,960 games**, with **102.385 hours inside its learner**,
755 screens, and 262 completed unsuccessful verification attempts (attempt 263
is pending). The last 100 screens average **49.5137%** against champion 9; the
last 100 completed gates average **49.7215%**. These averages describe different
candidates and are not a pooled confidence interval for one model.

The last 500 learner iterations average value cross entropy 0.50428, policy
entropy 0.07879, post-update KL 0.00342, and learning rate 9.9867e-6. None of those
iterations was rejected. Numerical instability or an overly strict gate does
not explain these records by itself. Screens are near 50% before the gate.

## Identifiable weaknesses

`collect_trajectory` previously returned immediately when the dominance mask
left only one eligible action. Such positions supplied **neither policy nor
critic training examples**, although `ActorChooser.score` displays a win
estimate there. Forced plays can move between state encodings the critic has
not been trained to connect. Rare forced-only decision families can retain
stale value heads. This is a real training-coverage mismatch; it does not prove
that every observed swing is wrong. Draws and other newly revealed information
can legitimately change a win estimate.

The critic is a linear output bank on the policy's state features. Its gradients
cannot train those features. It also inherited the actor's learning-rate cap,
global gradient clipping, KL early stopping, and rollback schedule. The policy
can move its features while the critic gets only the same tiny optimization
steps. Old value loss was averaged over training minibatches and omitted forced
positions, so it was not a fresh-game calibration measurement for the UI.

The win estimate is an opponent-, policy-, and rules-dependent prediction. The
campaign learns against its incumbent with historical rules v1 and stochastic
actions; the UI can use a different opponent and corrected rules v2. A critic
correction cannot make an old checkpoint universally calibrated. The poor
baseline can increase policy-gradient noise, but terminal Monte Carlo returns
remain the training targets; these findings do **not** establish the critic as
the sole cause of the promotion plateau.

## Implemented correction

- Collect critic examples at every learner decision, including forced actions;
  continue to optimize policy only where a choice exists.
- Freeze the value bank for actor optimization, excluding even inherited Adam
  momentum and removing critic gradients from actor gradient clipping.
- Fit the value bank separately with Adam at 3e-4 for two epochs, using detached
  features from the accepted actor update. This rate is an explicit, persisted
  intervention to validate, not a measured optimal hyperparameter.
- Compute policy advantages **before** fitting the critic on the current games.
  This avoids leaking the current outcome into its own baseline. Critic fitting
  continues even if the policy stops early or rolls back.
- Persist the independent critic optimizer, update count, and random stream.
  A separate sidecar accompanies each model. Existing actor formats remain
  compatible, so new estimates are available through normal model loading.
- Record pre-fit Brier score, log loss, average prediction and outcome, split
  by family and forced versus choice positions, plus critic runtime.

Old experiments retain their previous behavior on resume unless explicitly
migrated. The optional `--no-separate-critic` path provides a control. No existing
checkpoint is rewritten, no champion is promoted by this change, and the
promotion error budget, seeds, and game rules remain intact. The critic still
has a linear head over policy-trained features; an independent representation
is a possible follow-up if the new calibration evidence warrants it.

## Validation and installation

`scripts/validate_astro6_critic.py` runs a bounded background job:

1. Regression tests for forced-state coverage, optimizer isolation, checkpoint
   state, native actor parity, and audited runtime maintenance.
2. Real Metal training comparisons requiring exactly equal model and optimizer
   arrays for uninterrupted versus resumed training, and for the historical
   learner versus the correction-disabled compatibility path.
3. Fresh-game audits of champion 9 and the stalled learner. The audit separately
   measures probability changes after forced Scout/Viper/Explorer plays within
   the same turn, where no new card information is revealed.
4. Two 4,096-game training arms with the same source and initial seed, one with
   the correction and one without, each screened on 1,024 paired seeds. These
   are exploratory measurements, not promotion evidence.
5. Only after successful completion, install the revision under the paused
   campaign lock, with a complete runtime/manifest backup and recorded source
   hashes. The campaign stays paused. An error or concurrent source change
   prevents installation.

Results, logs, audits and comparison checkpoints are written to
`data/astro6-critic-20260919/`. Read `result.json` for completion and installation
status. The job does not require an interactive agent to keep running. Test
results and a strength improvement are **not yet established** when this job is
launched.
