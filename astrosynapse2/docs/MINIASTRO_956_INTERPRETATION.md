# Reading the continued 956-parameter MiniAstro

Inspected student: `669746289fef4e71ad5caf056559d2e6`, the saved epoch-38
continuation trained using 100,000 additional gen10 games. This document concerns
this student's executable acquisition policy, not every strategy in the teacher.

The analysis reads all 956 learned parameters and evaluates 20,000 randomly
selected covered test positions from its continuation dataset. Of these, 17,403
have more than one candidate. No weights were changed. The analysis is reproducible
with `PYTHONPATH=backend .venv/bin/python scripts/inspect_miniastro.py`.

## What the model computes

For each current candidate and None, it computes a score. Highest score wins.
There are 70 linear weights and eight hidden ReLU units. Each unit computes a
weighted sum of selected features plus a bias, clips negative sums to zero, and
multiplies by a learned output weight. Five units subtract penalties; three add
bonuses. Hidden units do not feed into one another.

Every feature is transformed as
`z = clip((sign(x) * log(1 + abs(x)) - mean) / std, -6, 6)`.
For this continuation, means and standard deviations remain those of the parent
student, preserving the meaning of its saved weights. Coefficients below multiply
these normalized features, not raw turns, cards, or authority points.

| Component | Inputs per unit | Units | Parameters |
| --- | ---: | ---: | ---: |
| Linear candidate terms | 70 | — | 70 |
| Economy | 89 | 2 | 182 |
| Deck composition | 134 | 2 | 272 |
| Timing | 79 | 2 | 162 |
| Matchup | 134 | 1 | 136 |
| Market | 132 | 1 | 134 |
| Total | | 8 | 956 |

Group names are architectural labels. Their inputs overlap. For example, the
Market unit sees both players' Trade Federation counts because those feature
names contain “trade.” It is not a pure market-only strategic concept.

## The eight units, translated cautiously

The descriptions below summarize dominant coefficients and actual activity. They
are useful interpretations, not exhaustive English rules equivalent to each unit.
All units also contain card-specific offsets and other features.

1. **Economy 1: a budget/alternative penalty with card-specific exemptions.**
   Its output multiplier is −0.635. More expensive affordable alternatives
   (+0.948), more starters (+0.742), more remaining trade (+0.662), and more
   trade left after this purchase (+0.463) activate the penalty. More printed
   trade already in the deck (−0.488) weakens it. Cutter, Battle Pod, Trade Bot,
   and many other cards have substantial negative offsets, which exempt them
   from some or all of the penalty. It contributes to relative card quality
   and budget use; it is not simply a rule to buy expensive cards.

2. **Economy 2: a conditional bonus for developing the deck.** Its multiplier
   is +0.851. Affordability (+1.347), starters (+1.092), remaining trade (+1.015),
   and total trade (+0.951) activate it. Existing deck trade (−0.643) and own
   Trade Federation count (−0.528) suppress it. This resembles an incentive to
   acquire useful affordable cards while the economy is immature. It includes
   different candidate offsets, so it is not an indiscriminate buy bonus.

3. **Deck composition 1: a developed-deck penalty, especially for trade cards.**
   Its multiplier is −0.770. Deck size (+1.028) and candidate printed trade
   (+0.970) activate it. Federation Shuttle (+0.834), Trade Bot (+0.729), and
   Trade Pod (+0.557) have extra positive activation weights. Many already-owned
   cards also activate it. In observed mature positions this can make another
   economy card worse than None. The unit does not explicitly compute deck
   dilution or number of future draws; that is an interpretation of the pattern.

4. **Deck composition 2: pressure to improve a small, healthy starter deck.**
   Its multiplier is −1.363. Higher own authority (+1.224), more unaligned cards
   (+0.735), and Scouts (+0.535) activate a penalty. Larger deck size (−1.955)
   and candidate affordability (−1.966) strongly suppress it. None has
   affordability zero, so it often suffers a large penalty while affordable
   purchases escape. The model thereby discourages passing in opening positions.
   Increasing deck size switches this pressure off, while Deck composition 1
   can then penalize additional purchases. Together they express a conditional
   preference to grow first and become more selective later.

5. **Timing 1: an early-game bonus with card-specific lifetimes.** Its multiplier
   is +1.228 and turn number has input weight −2.301. Its active score slope
   with respect to normalized turn is therefore −2.826. Cutter has a large
   positive offset (+0.639), while Survey Ship (−1.062), Battle Station (−0.726),
   and Mech World (−0.706) have negative offsets. These offsets let an early
   bonus persist longer for some cards. Two cards both inside the active region
   lose the same amount with advancing turn, so their relative timing score
   changes only when their activation regimes differ.

6. **Timing 2: a late-game bonus that often favors combat ships over economy
   and bases.** Multiplier +1.145, turn input +2.603, active normalized-turn slope
   +2.980. Candidate printed combat (+0.511) activates it; printed trade (−0.446),
   base status (−0.473), and base defense (−0.524) suppress it. War World and
   Royal Redoubt have additional negative offsets. The net score can still favor
   bases because the other terms matter. This unit and Timing 1 can both be
   active for the same candidate: they are not an exclusive early/late switch.

7. **Matchup 1: a penalty that distinguishes investments from fillers and fades
   as combat develops.** Multiplier −0.885. Opponent authority (+1.144) activates
   it, while own deck combat (−1.014) and opponent deck combat (−0.688) weaken it.
   Thinning ability (−0.957) and many engine-card identities strongly exempt a
   card. Explorer and unaligned status have positive weights, so Explorer often
   receives a particularly large penalty relative to Trade Bot or Cutter.
   Same-faction cards in your deck (−0.672) also reduce this penalty. This is
   compatible with favoring useful investments when the game still has runway,
   but its response to authority alone is not a complete strategic rule.

8. **Market 1: a smaller correction involving the row and both decks.** Multiplier
   −0.677. Leftover trade (+0.625) and candidate printed trade (+0.540) activate
   a penalty; candidate printed combat (−0.462) weakens it. Several cheap combat
   cards have large negative identity offsets. More blue cards in either deck
   weaken the penalty. Individual market cards shift it too: market Blob
   Destroyer (+0.682) raises activation, while Command Ship (−0.537) and Brain
   World (−0.515) lower it. This is a learned context correction; these weights
   alone do not establish strategic denial or a plan for future market replacements.

## Two particularly strong mathematical conclusions

**Same-faction ownership is rewarded.** For the feature “Candidate faction cards
in your deck,” the direct score coefficient is +0.1443. Seven unit contributions
have positive slopes when active; the only negative one is −0.0207. Consequently,
holding other inputs fixed, increasing this normalized feature always increases
the candidate's score until clipping makes it flat. The minimum unclipped slope
is +0.1236. This is a property of the entire network, not a conjecture from one
weight. Physically adding a card changes other features too, so the result does
not mean every additional card of that faction improves every future decision.
The separate same-faction-discard feature has mixed signs and no such guarantee.

**Unspent trade is penalized as a feature.** “Trade left after buying candidate”
has direct slope −0.2513, plus active-unit slopes −0.2941, −0.4833 and −0.4235.
All are negative. Holding the other features fixed, more leftover trade never
improves a candidate score. This is a pressure toward using the budget, not a
guarantee to buy the most expensive card or always buy something. Cost, identity,
and None's other terms can dominate.

## Worked positions

These are actual held-out recorded positions scored with the frozen model and
their recorded full-turn trade, including later trade gains when applicable.

**Opening investment:** sample 1,419,901, turn 3, authority 49–49, deck size 11,
ten starters, five total trade (four currently in pool). MiniAstro and the teacher
both choose Trade Bot. The model's decomposition is:

| Score term | Trade Bot | Explorer |
| --- | ---: | ---: |
| Linear | −1.055 | +0.321 |
| Economy | +2.155 | +0.177 |
| Deck composition | −0.699 | −1.821 |
| Timing | +6.618 | +2.539 |
| Matchup | 0 | −10.415 |
| Market | −1.561 | 0 |
| Total | **+5.458** | **−9.198** |

Barter World scores +3.965 and None −14.238. Trade Bot wins primarily through
avoiding the matchup penalty and receiving a larger early-game bonus, even
though its linear term is lower than Explorer's. This illustrates why reading
card identity weights as a global card ranking would be wrong.

**Stop adding economy:** sample 50,828, turn 22, authority 14–9, deck size 22,
deck printed combat 29, two trade remaining after six already spent. Both teacher
and student choose None. Scores are None −2.480, Explorer −3.160, Trade Bot −10.953.
Deck composition 1 alone charges Trade Bot −9.464, versus Explorer −5.549 and
None −4.163; Trade Bot has effectively lost its early-game timing bonus.

**Use the final two trade early:** sample 256,978, turn 7, deck size 13, ten
starters, authority 48–40, two trade remaining after six spent. Teacher and
student choose Explorer. Scores are Explorer −4.235 versus None −11.179.
Deck composition 2 penalizes None by −6.239 and Explorer by zero; the economy
terms also favor Explorer. Thus the model has not learned a universal aversion
to Explorer.

Across the inspected sample, Trade Bot beats Explorer in 42.3% of their 2,403
co-occurrences, while Cutter beats Explorer in all 416 co-occurrences. These are
conditional pair comparisons, not win rates or claims about every possible state.

## Where context actually changes preferences

A unit's state-only inputs shift the same activation sum for every candidate.
If both candidates remain active, that common shift cancels in their score
difference; if both remain inactive, neither score changes. The interaction
comes when the candidates cross different zero thresholds. Card-specific offsets
move those thresholds. Much of this small model's strategy is therefore a few
contextual switches between otherwise fairly stable card rankings.

For a diagnostic probe, holding every feature except turn number fixed in sample
695,082 changes the winner from Machine Base at turn 5 to Command Ship at turns
15, 30 and 50. In sample 2051597 it changes from Supply Bot at turn 5 to None at
15, 30 and 50. These are internal feature sensitivities, not legal game rollouts
or evidence of causal strategic value.

## Which units matter to actual choices?

Remove one unit's output without retraining and compare against the original
winner on the 17,403 inspected positions with multiple candidates:

| Unit removed | Recommendations changed |
| --- | ---: |
| Economy 1 | 9.6% |
| Economy 2 | 7.5% |
| Deck composition 1 | 16.8% |
| Deck composition 2 | 15.4% |
| Timing 1 | 16.5% |
| Timing 2 | 15.8% |
| Matchup 1 | 22.0% |
| Market 1 | 6.0% |

These percentages overlap and do not measure accuracy lost, unique information,
or strategic importance in winning games. They establish that all eight units
participate in the actual choice function.

## Scope and downloadable evidence

The model predicts the next acquisition this turn, including None. It does not
simulate future turns or explicitly optimize victory. Its approximately 88%
agreement is fidelity to gen10 behavior, not an endorsement of every rule above.
The named group structure was designed before training; the detailed preferences
and thresholds are learned. Weight correlations prevent a unique interpretation.

Inspection outputs in `data/acquire_students/669746289fef4e71ad5caf056559d2e6/`:

- `all-956-weights.csv`: every learned linear weight, hidden input weight, hidden
  bias, and hidden output weight, exactly 956 parameter rows.
- `feature-normalization.csv`: the constants needed to apply those weights.
- `weight-inspection.json`: ablations, feature slopes, exact worked-position
  decompositions, and turn-number probes.

The practical English summary is: develop a starter deck, favor useful engines
over filler when they have time to pay off, reward faction support, spend trade
effectively, shift preferences as the game develops, and stop adding weak economy
cards when their penalties exceed the model's incentive to acquire something.
