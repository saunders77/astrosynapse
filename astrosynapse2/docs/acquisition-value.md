# 10k Acquisition Value

The `acquire_bucketed` probe uses additive card values. Standard 1,000-game
acquire/scrap probes and historical Elo reports retain their existing semantics.
Run a new 10k test to obtain the new values: old Elo reports cannot be converted,
because their discarded multi-acquisition turns are unavailable.

## Choice model

At each purchase, free-acquisition, or eligible stopping decision, record the
currently visible legal cards, their multiplicities, current trade, and chosen
action. Paid bundles respect both supply and budget; free-acquisition bundles
contain at most one legal target. An empty bundle represents stopping/declining.

For each feasible multiset `B`, let `U(B) = sum(count(c, B) * value(c))` and
`P(B) = exp(U(B)) / sum(exp(U(B')))`. The empty bundle (No Card) is fixed at 0,
and the logit temperature at 1 for all raw fits. Every card, including Explorer,
is freely estimated and may be negative in any turn, other bucket, or overall.

After fitting, multiply every acquisition value and interval by **one shared
positive factor**, `2 / raw Explorer value at turn 3`. Explorer's turn-3 point
estimate is exactly 2; its estimates at other turns and in the overall list vary
naturally. No Card remains 0. A value of 4 means twice the fitted turn-3 Explorer
utility, not necessarily two Explorers at the current turn. This is a change of
display units; likelihood calculations and regularization use the raw scale.

If turn-3 Explorer is unsupported, nonfinite, nonpositive, or numerically zero,
calibration is marked unavailable and the report explicitly shows raw units.
Never substitute another turn or reverse the rankings with a negative factor.

The observed next purchase has likelihood
`P(next=c) = sum(P(B) * count(c, B) / size(B))`; stopping has likelihood
`P(empty)`. This marginalizes the possible remaining bundle rather than treating
the next purchased card as individually better than every other card. We assume
uniform purchase order within a proposed bundle. Recompute bundles after each
decision, so future refill cards never appear as earlier alternatives.

This is a **visible-market continuation approximation**. It does not estimate
future refill distributions, benefits from buying in a particular order, card
synergies, or win contribution. Card values describe the tested policy's
acquisition behavior averaged over observed states. Free-to-top placement is
also pooled into the acquired card's value.

## Turns, buckets, and overall values

All acquisition counts are retained, including repeated copies and free cards.
End-turn choices provide evidence when at least one affordable card remains.
Turns without any acquisition opportunity provide no preference evidence.
Each player-turn's captured decisions have total likelihood weight one, so
turns with many acquisitions do not automatically dominate the fit.

All decisions in a turn use the context at its first captured decision:
turn number, authority, previous acquisition count, and opponent color. Turn
30+ groups later turns. Each chart bucket is independently fitted with the same
zero baseline, temperature, and weak zero-centered Gaussian regularizer (precision
0.25). This version uses regularization, **not adjacent-turn smoothing**.
The overall list is a separate fit to all eligible turns, not an average of
bucket estimates. It includes opponent-color states omitted from that chart.

## Confidence intervals

Fit the turn-weighted marginal likelihood with BFGS. The observed Hessian
includes cross-card covariance and the regularizer. Sum score contributions
within each game, including both players and all their turns, to form a
game-cluster sandwich covariance. Include regularizer precision in the sandwich
meat and floor each reported marginal variance at the inverse-curvature
variance. The displayed interval is the estimate plus/minus 1.96 standard
errors: an approximate, regularized 95% interval conditional on the model.

Only No Card has a zero-width interval by definition. Explorer retains an
estimated interval, including at turn 3. Every interval and standard error is
multiplied by the same positive calibration factor. These are **conditional on
the fitted calibration factor**, not ratio confidence intervals that propagate
uncertainty in that factor; the reference's raw estimate and interval are saved
in calibration metadata. Unobserved cards, cards seen in fewer
than two independent games, and unconverged fits are marked unsupported, with
null values/intervals. Weakly identified estimates can have broad intervals.

Reports identify the model as `visible_bundle_acquisition_value_v2`, include
explicit `value`, `ci_lower`, `ci_upper`, and fit diagnostics, and use `_value_`
filenames. Legacy `elo` fields are compatibility aliases, not Elo units.
The UI uses model metadata to preserve the presentation of historical reports.
Version 1 reports keep their original Explorer = 2 at every turn labels; they
cannot be converted to version 2 without rerunning the acquisition test.
