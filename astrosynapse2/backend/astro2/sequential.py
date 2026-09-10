"""Time-uniform paired-mean bounds by mixtures of bounded betting processes.

For null mean <= m, E[1+b*(X/m-1) | past] <= 1 for b in (0,1), X in [0,1].
Each product, and their predeclared uniform mixture, is a nonnegative
supermartingale starting at 1. Ville's inequality bounds ever crossing 1/alpha
by alpha. Inverting these tests gives a one-sided lower confidence sequence.
Reference: Waudby-Smith & Ramdas, https://arxiv.org/abs/2010.09686.
"""

import math

import numpy as np

BET_FRACTIONS = np.array([0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])


def log_evidence(values, mean):
    values = np.asarray(values, dtype=float)
    if not 0 < mean <= 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("scores must be finite in [0,1] and null mean in (0,1]")
    # Pair scores have three values, but weighted bincount is unnecessary at
    # these bounded arena sizes. No outcome-dependent betting parameters.
    logs = np.log1p(BET_FRACTIONS[:, None] * (values[None, :] / mean - 1)).sum(axis=1)
    maximum = logs.max()
    return float(maximum + np.log(np.exp(logs - maximum).mean()))


def lower_sequence(values, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if not len(values):
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(35):
        midpoint = (low + high) / 2
        if log_evidence(values, midpoint) >= math.log(1 / alpha):
            low = midpoint
        else:
            high = midpoint
    return low


def promotion_evidence(rows, attempt):
    if attempt < 1 or len({r["pair"] for r in rows}) != len(rows):
        raise ValueError("invalid attempt or duplicate pairs")
    values = [sum(r["scores"]) / 2 for r in rows]
    if any(
        len(r["scores"]) != 2 or any(x not in (0.0, 0.5, 1.0) for x in r["scores"]) for r in rows
    ):
        raise ValueError("each pair must contain two seat-swapped games")
    alpha = 0.05 / (attempt * (attempt + 1))
    evidence = log_evidence(values, 0.5)
    return dict(
        pairs=len(rows),
        score=float(np.mean(values)) if values else None,
        lower=lower_sequence(values, alpha),
        alpha=alpha,
        confidence=1 - alpha,
        log_evidence=evidence,
        threshold=math.log(1 / alpha),
        passed=bool(values) and evidence >= math.log(1 / alpha),
        method="paired_betting_confidence_sequence",
        attempt=attempt,
        truncated=sum(r.get("truncated", 0) for r in rows),
    )
