"""Opponent scheduling and conservative paired-match promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .stats import Interval, wilson_interval


@dataclass(slots=True)
class Opponent:
    id: str
    actor_path: str | None
    kind: Literal[
        "current",
        "checkpoint",
        "champion",
        "anchor",
        "exploiter",
        "baseline",
    ]
    label: str
    wins: float = 0.0
    games: int = 0
    pinned: bool = False

    @property
    def smoothed_score(self) -> float:
        # Beta(2, 2) keeps new noisy matchups away from the extremes.
        return (self.wins + 2.0) / (self.games + 4.0)


@dataclass(slots=True)
class PromotionDecision:
    promote: bool
    reason: str
    interval: Interval
    pairs: int
    first_seat_score: float
    second_seat_score: float
    paired_interval: Interval | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "promote": self.promote,
            "reason": self.reason,
            "interval": self.interval.as_dict(),
            "paired_interval": self.paired_interval.as_dict() if self.paired_interval else None,
            "pairs": self.pairs,
            "first_seat_score": self.first_seat_score,
            "second_seat_score": self.second_seat_score,
        }


@dataclass(slots=True)
class League:
    opponents: list[Opponent] = field(default_factory=list)

    def upsert(self, opponent: Opponent) -> None:
        for index, existing in enumerate(self.opponents):
            if existing.id == opponent.id:
                self.opponents[index] = opponent
                return
        self.opponents.append(opponent)

    def record(self, opponent_id: str, score: float, games: int = 1) -> None:
        target = next(item for item in self.opponents if item.id == opponent_id)
        target.wins += score
        target.games += games

    def snapshot(self) -> list[dict[str, object]]:
        """Persist matchup statistics without coupling to artifact paths."""

        return [
            {
                "id": item.id,
                "wins": float(item.wins),
                "games": int(item.games),
                "pinned": bool(item.pinned),
            }
            for item in self.opponents
        ]

    def restore(self, payload: list[dict[str, object]]) -> int:
        """Restore statistics and stable PFSP ordering for surviving opponents."""

        current = {opponent.id: opponent for opponent in self.opponents}
        ordered: list[Opponent] = []
        restored = 0
        for saved in payload:
            saved_id = str(saved.get("id") or "")
            opponent = current.pop(saved_id, None)
            if opponent is None:
                continue
            opponent.wins = max(0.0, float(saved.get("wins", 0.0)))
            opponent.games = max(0, int(saved.get("games", 0)))
            opponent.pinned = bool(saved.get("pinned", opponent.pinned))
            ordered.append(opponent)
            restored += 1
        # Checkpoints accepted after the durable snapshot have no saved
        # statistics yet. Keep their current discovery order after the exact
        # restored prefix.
        ordered.extend(opponent for opponent in self.opponents if opponent.id in current)
        self.opponents = ordered
        return restored

    def select(
        self,
        rng: np.random.Generator,
        *,
        mode: Literal["pfsp", "near_even", "uniform"] = "pfsp",
        kinds: set[str] | None = None,
    ) -> Opponent:
        candidates = [item for item in self.opponents if kinds is None or item.kind in kinds]
        if not candidates:
            raise LookupError("league has no eligible opponents")
        scores = np.asarray([item.smoothed_score for item in candidates], dtype=np.float64)
        if mode == "pfsp":
            weights = np.square(1.0 - scores)
        elif mode == "near_even":
            weights = scores * (1.0 - scores)
        else:
            weights = np.ones_like(scores)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones_like(scores)
        probabilities = weights / weights.sum()
        return candidates[int(rng.choice(len(candidates), p=probabilities))]


def paired_bootstrap_interval(
    pair_scores: np.ndarray,
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> Interval:
    values = np.asarray(pair_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        return Interval(0.5, 0.0, 1.0, 0)
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    # Chunk the resampling indices to bound temporary memory for 20k-pair finals.
    chunk = 250
    cursor = 0
    while cursor < samples:
        count = min(chunk, samples - cursor)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates[cursor : cursor + count] = values[indices].mean(axis=1)
        cursor += count
    alpha = (1.0 - confidence) / 2.0
    return Interval(
        estimate=float(values.mean()),
        low=float(np.quantile(estimates, alpha)),
        high=float(np.quantile(estimates, 1.0 - alpha)),
        samples=len(values),
    )


def decide_promotion(
    first_seat_results: np.ndarray,
    second_seat_results: np.ndarray,
    *,
    confidence: float = 0.95,
    margin: float = 0.0,
    minimum_pairs: int = 2_000,
    bootstrap_samples: int = 10_000,
) -> PromotionDecision:
    first = np.asarray(first_seat_results, dtype=np.float64)
    second = np.asarray(second_seat_results, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("paired seat result arrays must have the same one-dimensional shape")
    pairs = len(first)
    combined = np.concatenate([first, second]) if pairs else np.array([], dtype=np.float64)
    z = 1.959963984540054
    if confidence >= 0.99:
        z = 2.5758293035489004
    interval = wilson_interval(float(combined.sum()), len(combined), z=z)
    paired = paired_bootstrap_interval(
        (first + second) * 0.5,
        confidence=confidence,
        samples=bootstrap_samples,
    )
    threshold = 0.5 + margin
    if pairs < minimum_pairs:
        promote = False
        reason = f"inconclusive: {pairs:,} of {minimum_pairs:,} required seed pairs"
    elif paired.low <= threshold:
        promote = False
        reason = f"inconclusive: paired lower bound {paired.low:.3f} is not above {threshold:.3f}"
    else:
        promote = True
        reason = f"promote: paired lower bound {paired.low:.3f} exceeds {threshold:.3f}"
    return PromotionDecision(
        promote=promote,
        reason=reason,
        interval=interval,
        paired_interval=paired,
        pairs=pairs,
        first_seat_score=float(first.mean()) if pairs else 0.5,
        second_seat_score=float(second.mean()) if pairs else 0.5,
    )
