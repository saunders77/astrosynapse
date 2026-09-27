"""Additive, visible-market acquisition values (stop = 0; Explorer learned).

At each acquisition decision we enumerate affordable bundles in the *current*
market. A latent bundle has probability proportional to exp(sum(card values)).
Its first purchase is uniformly selected among its copies. Marginalizing that
first purchase gives a likelihood for the observed action, including stopping.
Thus B + Explorer can compete with A without declaring B alone better than A.
The market is re-observed after every action; unseen refill cards never enter a
counterfactual. This is a static-market continuation approximation, not a model
of refill expectations, purchase ordering, or downstream win probability.

Each player-turn has total likelihood weight one. Intervals use the numerical
observed Hessian and game-clustered scores with a weak Gaussian regularizer.
They are approximate 95% intervals conditional on this behavioral model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

from .cards import ALL_CARDS, CARD_BY_ID
from .engine import ActionKind, DecisionFamily

MODEL = "visible_bundle_acquisition_value_v2"
CARD_IDS = tuple(card.card_id for card in ALL_CARDS if card.card_id not in {0, 1})
FREE_IDS = CARD_IDS
INDEX = {card_id: index for index, card_id in enumerate(FREE_IDS)}
RIDGE = 0.25


@dataclass(frozen=True, slots=True)
class AcquisitionSample:
    game: int
    player: int
    context: Any
    # (card id, available copies, price); free acquisitions have price zero.
    market: tuple[tuple[int, int, int], ...]
    budget: int
    single: bool
    can_stop: bool
    chosen: int  # -1 means end turn / decline free acquisition
    weight: float = 1.0


def extract_acquisition_samples(events, game_index=0):
    # Import here to keep the existing Elo API independent of the new model.
    from .card_analysis import _FACTION_COLORS, AcquisitionContext, _opponent_top_acquired_color

    contexts = {}
    acquired = Counter()
    colors = {0: Counter(), 1: Counter()}
    recent = {0: {}, 1: {}}
    sequence = 0
    samples = []
    acquisition_count = 0
    for player, decision, selected in events:
        turn = int(decision.observation.turn)
        key = (player, turn)
        if key not in contexts:
            contexts[key] = AcquisitionContext(
                max(1, turn),
                int(decision.observation.own_authority),
                acquired[player],
                int(decision.observation.opponent_authority),
                _opponent_top_acquired_color(colors[1 - player], recent[1 - player]),
            )
        free = decision.family == DecisionFamily.FREE_ACQUIRE
        purchase = selected.kind in {ActionKind.ACQUIRE, ActionKind.FREE_ACQUIRE}
        stop = selected.kind == ActionKind.END_TURN or (
            free and selected.kind == ActionKind.DECLINE
        )
        if purchase or stop:
            available = Counter()
            prices = {}
            for action in decision.actions:
                if action.kind != (ActionKind.FREE_ACQUIRE if free else ActionKind.ACQUIRE):
                    continue
                card_id = action.target_card_id if free else action.card_id
                if card_id not in CARD_IDS:
                    continue
                price = 0 if free else CARD_BY_ID[card_id].cost
                if not free and price > decision.observation.trade:
                    continue
                # Actions are deduplicated by the engine; recover row multiplicity.
                count = (
                    decision.observation.explorers_remaining
                    if action.source_zone == "explorer_supply"
                    else sum(
                        card is not None and card.card_id == card_id
                        for card in decision.observation.trade_row
                    )
                )
                available[card_id] = max(available[card_id], count)
                prices[card_id] = price
            chosen = (selected.target_card_id if free else selected.card_id) if purchase else -1
            if purchase and available[chosen] < 1:
                raise ValueError(f"Acquired card {chosen} missing from its recorded legal market")
            if available and (chosen == -1 or available[chosen]):
                samples.append(
                    AcquisitionSample(
                        game_index,
                        player,
                        contexts[key],
                        tuple(
                            (cid, count, prices[cid]) for cid, count in sorted(available.items())
                        ),
                        max(0, int(decision.observation.trade)),
                        free,
                        any(
                            action.kind == (ActionKind.DECLINE if free else ActionKind.END_TURN)
                            for action in decision.actions
                        ),
                        chosen,
                    )
                )
            if purchase:
                acquisition_count += 1
                acquired[player] += 1
                sequence += 1
                card = CARD_BY_ID.get(chosen)
                color = _FACTION_COLORS.get(card.faction) if card else None
                if color:
                    colors[player][color] += 1
                    recent[player][color] = sequence
    counts = Counter((sample.player, sample.context.turn) for sample in samples)
    samples = [
        replace(sample, weight=1 / counts[sample.player, sample.context.turn]) for sample in samples
    ]
    return {
        "turns_observed": len(contexts),
        "single_card_turns": len(counts),
        "acquisitions_recorded": acquisition_count,
        "decisions": samples,
    }


@lru_cache(maxsize=4096)
def bundle_counts(market, budget, single, can_stop):
    """Unique multisets, not purchase permutations; bounded by visible supply."""
    rows = []

    def visit(index, remaining, counts):
        if index == len(market):
            if can_stop or any(counts):
                rows.append(tuple(counts))
            return
        _, supply, price = market[index]
        maximum = min(supply, remaining // price) if price else supply
        if single:
            maximum = min(maximum, 1 - sum(counts))
        for count in range(maximum + 1):
            visit(index + 1, remaining - count * price, [*counts, count])

    visit(0, budget, [])
    return np.asarray(rows, dtype=np.float64)


class _Likelihood:
    """Ragged bundle matrices using NumPy only; identical states share work."""

    def __init__(self, samples):
        grouped = {}
        for sample in samples:
            key = (sample.market, sample.budget, sample.single, sample.can_stop, sample.chosen)
            grouped.setdefault(key, []).append(sample)
        self.groups = list(grouped.values())
        matrices, indices, fractions, starts = [], [], [], []
        cursor = 0
        for group in self.groups:
            sample = group[0]
            counts = bundle_counts(sample.market, sample.budget, sample.single, sample.can_stop)
            ids = [card[0] for card in sample.market]
            sizes = counts.sum(axis=1)
            fraction = (
                (sizes == 0).astype(float)
                if sample.chosen == -1
                else counts[:, ids.index(sample.chosen)] / np.maximum(1, sizes)
            )
            # Padding to six visible card types makes vectorized reductions cheap.
            matrix = np.zeros((len(counts), 6))
            index = np.full(6, len(FREE_IDS), dtype=np.int32)
            for col, cid in enumerate(ids):
                matrix[:, col] = counts[:, col]
                index[col] = INDEX[cid]
            starts.append(cursor)
            cursor += len(counts)
            matrices.append(matrix)
            indices.append(np.broadcast_to(index, matrix.shape))
            fractions.append(fraction)
        self.x = np.concatenate(matrices)
        self.index = np.concatenate(indices)
        self.fraction = np.concatenate(fractions)
        self.starts = np.asarray(starts)
        self.group_index = np.repeat(np.arange(len(starts)), np.diff([*starts, cursor]))
        self.weights = np.array([sum(s.weight for s in group) for group in self.groups])

    def evaluate(self, values, individual=False):
        utility = (self.x * np.append(values, 0)[self.index]).sum(axis=1)
        maximum = np.maximum.reduceat(utility, self.starts)
        exp = np.exp(utility - maximum[self.group_index])
        denominator = np.add.reduceat(exp, self.starts)
        numerator = np.add.reduceat(exp * self.fraction, self.starts)
        numerator = np.maximum(numerator, 1e-300)
        loss = float(self.weights @ (np.log(denominator) - np.log(numerator)))
        delta = (
            exp / denominator[self.group_index] - exp * self.fraction / numerator[self.group_index]
        )
        if individual:
            scores = np.zeros((len(self.groups), len(FREE_IDS) + 1))
            for col in range(6):
                np.add.at(scores, (self.group_index, self.index[:, col]), delta * self.x[:, col])
            return scores[:, :-1]
        gradient = np.bincount(
            self.index.ravel(),
            weights=(self.x * (delta * self.weights[self.group_index])[:, None]).ravel(),
            minlength=len(FREE_IDS) + 1,
        )[:-1]
        return loss + RIDGE / 2 * float(values @ values), gradient + RIDGE * values


def _minimize(likelihood):
    """Small dense BFGS with Armijo backtracking, avoiding a runtime dependency."""
    values = np.zeros(len(FREE_IDS))
    inverse = np.eye(len(values))
    loss, gradient = likelihood.evaluate(values)
    converged = False
    for _iteration in range(200):
        if np.max(np.abs(gradient)) < 1e-5:
            converged = True
            break
        direction = -inverse @ gradient
        if gradient @ direction >= 0:
            inverse = np.eye(len(values))
            direction = -gradient
        step = min(1.0, 10 / max(10, np.max(np.abs(direction))))
        for _ in range(40):
            candidate = values + step * direction
            new_loss, new_gradient = likelihood.evaluate(candidate)
            if new_loss <= loss + 1e-4 * step * float(gradient @ direction):
                break
            step *= 0.5
        else:
            break
        s, y = candidate - values, new_gradient - gradient
        curvature = float(s @ y)
        if curvature > 1e-12:
            transform = np.eye(len(values)) - np.outer(s, y) / curvature
            inverse = transform @ inverse @ transform.T + np.outer(s, s) / curvature
        values, loss, gradient = candidate, new_loss, new_gradient
    return values, converged, _iteration + 1


def rate_acquisition_samples(samples):
    from .card_analysis import ChoiceOption, _card_color_for_option, _card_cost_for_option

    values = np.zeros(len(FREE_IDS))
    covariance = np.eye(len(values)) / RIDGE
    counts = Counter()
    evidence_games = {cid: set() for cid in CARD_IDS}
    wins = Counter()
    for sample in samples:
        counts.update({cid for cid, _, _ in sample.market})
        for cid, _, _ in sample.market:
            evidence_games[cid].add(sample.game)
        if sample.can_stop:
            counts[-1] += 1
        wins[sample.chosen] += 1
    converged, iterations = True, 0
    games = {sample.game for sample in samples}
    if samples:
        likelihood = _Likelihood(samples)
        values, converged, iterations = _minimize(likelihood)
        # Observed curvature includes cross-card covariance and the fixed anchors.
        hessian = np.empty((len(values), len(values)))
        for column in range(len(values)):
            perturb = np.zeros(len(values))
            perturb[column] = 1e-4
            hessian[:, column] = (
                likelihood.evaluate(values + perturb)[1] - likelihood.evaluate(values - perturb)[1]
            ) / 2e-4
        eigenvalues, eigenvectors = np.linalg.eigh((hessian + hessian.T) / 2)
        inverse = (eigenvectors / np.maximum(eigenvalues, RIDGE)) @ eigenvectors.T
        scores = likelihood.evaluate(values, individual=True)
        game_index = {game: index for index, game in enumerate(sorted(games))}
        clusters = np.zeros((len(games), len(values)))
        for score, group in zip(scores, likelihood.groups, strict=True):
            for sample in group:
                clusters[game_index[sample.game]] += sample.weight * score
        clusters -= clusters.mean(axis=0)
        meat = clusters.T @ clusters * (len(games) / max(1, len(games) - 1))
        covariance = inverse @ (meat + RIDGE * np.eye(len(values))) @ inverse
        # Never report less uncertainty than the model's curvature alone implies.
        diagonal = np.maximum(np.diag(covariance), np.diag(inverse))
    else:
        diagonal = np.diag(covariance)
    leaderboard = []
    for cid in (*CARD_IDS, -1):
        fixed = cid == -1
        value = 0.0 if fixed else float(values[INDEX[cid]])
        se = 0.0 if fixed else float(np.sqrt(max(0, diagonal[INDEX[cid]])))
        supported = fixed or (counts[cid] > 0 and len(evidence_games[cid]) >= 2 and converged)
        lower, upper = (value - 1.96 * se, value + 1.96 * se) if supported else (None, None)
        name = CARD_BY_ID[cid].name if cid != -1 else "No Card"
        option = ChoiceOption(f"card:{cid}" if cid != -1 else "no_card", name, "", name)
        leaderboard.append(
            {
                **option.to_dict(),
                "card_color": _card_color_for_option(option),
                "card_cost": _card_cost_for_option(option),
                "value": value if supported else None,
                # Compatibility aliases for saved-result consumers; model metadata disambiguates units.
                "elo": value,
                "raw_elo": value,
                "uncertainty": se if supported else None,
                "raw_uncertainty": se if supported else None,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": 0.95,
                "fixed_anchor": fixed,
                "supported": supported,
                "decision_count": counts[cid],
                "pairwise_comparisons": 0,
                "wins": wins[cid],
                "losses": counts[cid] - wins[cid],
                "next_k_factor": 0,
            }
        )
    leaderboard.sort(key=lambda entry: (not entry["supported"], -entry["elo"], entry["label"]))
    return {
        "leaderboard": leaderboard,
        "rating_model": MODEL,
        "interval_method": "regularized_game_cluster_sandwich_with_curvature_floor_95",
        "scored_decisions": len(samples),
        "pairwise_comparisons": 0,
        "eligible_turns": len({(s.game, s.player, s.context.turn) for s in samples}),
        "games_with_evidence": len(games),
        "fit_converged": converged,
        "fit_iterations": iterations,
        "normalization_factor": 1.0,
        "normalization_offset": 0.0,
        "explorer_raw_elo": float(values[INDEX[2]]),
    }


def calibrate_acquisition_report(result: dict[str, Any]) -> None:
    """Apply one positive turn-3 Explorer scale to every acquisition fit.

    Keep raw estimates for reproducibility. Intervals are expressed in the same
    units, conditional on the fitted calibration factor (not ratio intervals).
    Missing/nonpositive references cannot define this scale without reversing
    preference order; retain explicitly labeled raw units in that case.
    """
    charts = result.get("bucketed_charts", [])
    reference_bucket = next(
        (
            bucket
            for chart in charts
            if chart["key"] == "turn"
            for bucket in chart["buckets"]
            if bucket["key"] == "3"
        ),
        None,
    )
    reference = (
        next(
            (entry for entry in reference_bucket["leaderboard"] if entry["key"] == "card:2"),
            None,
        )
        if reference_bucket is not None
        else None
    )
    raw_value = reference.get("raw_elo") if reference else None
    supported = reference is not None and reference.get("supported", False)
    valid = supported and raw_value is not None and np.isfinite(raw_value) and raw_value > 1e-8
    factor = 2.0 / raw_value if valid else 1.0
    calibration = {
        "method": "explorer_turn_3",
        "status": "calibrated" if valid else "unavailable",
        "reference_card": "card:2",
        "reference_turn": 3,
        "target_value": 2.0,
        "reference_raw_value": raw_value,
        "reference_raw_ci_lower": reference.get("raw_ci_lower", reference.get("ci_lower"))
        if reference
        else None,
        "reference_raw_ci_upper": reference.get("raw_ci_upper", reference.get("ci_upper"))
        if reference
        else None,
        "factor": factor,
        "intervals_conditional_on_scale": True,
        "message": (
            "No Card = 0; one shared scale sets Explorer's turn-3 estimate to 2. "
            "Explorer is learned independently in other buckets and overall. "
            "95% intervals are conditional on this fitted scale."
            if valid
            else "Turn-3 Explorer has no positive supported estimate; calibration is unavailable. "
            "Showing raw model units with No Card = 0; Explorer remains freely estimated."
        ),
    }
    fits = [result, *(bucket for chart in charts for bucket in chart["buckets"])]
    for fit in fits:
        fit["calibration"] = calibration.copy()
        fit["normalization_factor"] = factor
        fit["normalization_offset"] = 0.0
        for entry in fit["leaderboard"]:
            # Derive from raw fields so applying the conversion again is harmless.
            raw = entry["raw_elo"]
            raw_se = entry["raw_uncertainty"]
            entry.setdefault("raw_ci_lower", entry["ci_lower"])
            entry.setdefault("raw_ci_upper", entry["ci_upper"])
            entry["elo"] = raw * factor
            entry["value"] = entry["elo"] if entry["supported"] else None
            entry["uncertainty"] = raw_se * factor if raw_se is not None else None
            for bound in ("lower", "upper"):
                raw_bound = entry[f"raw_ci_{bound}"]
                entry[f"ci_{bound}"] = raw_bound * factor if raw_bound is not None else None
            entry["calibration_reference"] = bool(valid and entry is reference)
    if valid:
        reference["value"] = reference["elo"] = 2.0
    for chart in charts:
        chart["calibration"] = calibration.copy()
