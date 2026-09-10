"""Experiment provenance and fixed-budget certification, independent of ML."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity(package: Path, scripts: Path) -> dict[str, str]:
    return {
        f"{label}/{path.relative_to(root)}": sha256(path)
        for label, root in (("astro2", package), ("scripts", scripts))
        for path in sorted(root.rglob("*"))
        if path.suffix in {".py", ".cpp"}
    }


def certify(opponents: dict[str, list[dict]], *, pairs: int, attempt: int, target=0.70, alpha=0.05):
    """Family-wise fixed-sample certificate across opponents AND attempts.

    Each attempt receives alpha/[k(k+1)], whose infinite sum is alpha. Each
    opponent shares that attempt's budget. Pair scores are bounded independent
    observations; games within a pair need not be independent. Intermediate
    results never constitute acceptance. Seeds must be fresh for each attempt.
    """
    if pairs < 1 or attempt < 1 or not opponents or not 0 < alpha < 1 or not 0 <= target <= 1:
        raise ValueError("invalid certification design")
    budget = alpha / (attempt * (attempt + 1))
    radius = math.sqrt(math.log(2 * len(opponents) / budget) / (2 * pairs))
    reports = {}
    for opponent, rows in opponents.items():
        if len(rows) != pairs or {r["pair"] for r in rows} != set(range(pairs)):
            raise ValueError("certification requires the complete predeclared set of unique pairs")
        values = []
        for row in rows:
            if len(row["scores"]) != 2 or any(x not in (0.0, 0.5, 1.0) for x in row["scores"]):
                raise ValueError("invalid game score")
            values.append(sum(row["scores"]) / 2)
        score = sum(values) / pairs
        reports[opponent] = {
            "pairs": pairs,
            "score": score,
            "lower": max(0, score - radius),
            "upper": min(1, score + radius),
            "truncated": sum(r.get("truncated", 0) for r in rows),
        }
    return {
        "method": "fixed_paired_hoeffding_with_opponent_and_attempt_allocation",
        "target": target,
        "family_alpha": alpha,
        "attempt": attempt,
        "attempt_alpha": budget,
        "opponents": reports,
        "passed": all(r["lower"] >= target for r in reports.values()),
    }


def next_learning_rate(current, initial, maximum, target_kl, measured_kl, early_stop):
    if not math.isfinite(measured_kl) or measured_kl < -1e-5:
        raise ValueError("invalid measured KL")
    if measured_kl > target_kl or early_stop:
        return max(initial / 32, current / 1.5)
    if measured_kl < target_kl / 3:
        return min(maximum, current * 1.5)
    return current
