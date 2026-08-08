"""Small statistical helpers used by evaluation and the dashboard."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "samples": self.samples,
        }


def wilson_interval(wins: float, games: int, z: float = 1.959963984540054) -> Interval:
    """Wilson score interval; draws may be supplied as half a win."""

    if games <= 0:
        return Interval(0.5, 0.0, 1.0, 0)
    p = min(1.0, max(0.0, wins / games))
    z2 = z * z
    denominator = 1.0 + z2 / games
    centre = (p + z2 / (2.0 * games)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * games)) / games) / denominator
    return Interval(p, max(0.0, centre - radius), min(1.0, centre + radius), games)


def elo_delta(score: float, floor: float = 1e-4) -> float:
    """Convert a head-to-head score to an Elo difference for display only."""

    bounded = min(1.0 - floor, max(floor, score))
    return 400.0 * math.log10(bounded / (1.0 - bounded))
