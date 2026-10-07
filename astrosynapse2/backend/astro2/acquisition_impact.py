"""Read-only analysis of paired MiniAstro acquisition comparisons.

Inverse inclusion weights undo the per-game disagreement reservoir. Uncertainty
resamples whole source games, keeping both seats, roots and paired draws together.
All reported effects remain signed: the reference can lose to the student.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np
import orjson

from .cards import CARD_BY_ID
from .experiment_control import atomic_json

SCHEMA = 1
STAGES = ["Turns 1–8", "Turns 9–16", "Turns 17–24", "Turns 25+"]
BUDGETS = ["0–2 trade", "3–4 trade", "5–6 trade", "7+ trade"]


def name(cid):
    return "Buy nothing" if cid == -1 else CARD_BY_ID[cid].name


def keys(event):
    teacher, student = event["teacher_card"], event["student_card"]
    category = (
        "Champion buys; student stops"
        if student == -1
        else "Champion stops; student buys"
        if teacher == -1
        else "Different purchases"
    )
    stage = STAGES[min((max(1, event["turn"]) - 1) // 8, 3)]
    trade = event["total_trade"] - event["spent"]
    budget = BUDGETS[0 if trade <= 2 else 1 if trade <= 4 else 2 if trade <= 6 else 3]
    return {
        "all": "All disagreements",
        "category": category,
        "pair": f"{teacher}:{student}",
        "stage": stage,
        "heat": f"{stage}|{budget}",
    }


def paired_interval(teacher_only, student_only, n):
    """Approximate conservative 95% interval from two Bonferroni Wilson intervals.

    Each discordance probability gets a 97.5% interval, then bounds subtract.
    This accounts for pairing and avoids zero-width intervals at zero discordance.
    These per-position intervals do not adjust for selecting extremes.
    """
    if not n:
        return [None, None]
    z = NormalDist().inv_cdf(0.9875)

    def wilson(k):
        p = k / n
        center = (p + z * z / (2 * n)) / (1 + z * z / n)
        radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return center - radius, center + radius

    plus, minus = wilson(teacher_only), wilson(student_only)
    return [max(-1, plus[0] - minus[1]), min(1, plus[1] - minus[0])]


def summarize_bucket(bucket, draws, eligible):
    numerator, denominator = bucket["num"], bucket["den"]
    weight = float(denominator.sum())
    samples = bucket["samples"]
    if not weight:
        return None
    boot_den = draws @ denominator
    valid = boot_den > 0
    boot_num = draws @ numerator
    boot_mean = boot_num[valid] / boot_den[valid]
    opportunity = draws @ eligible
    boot_burden = boot_num[opportunity > 0] / opportunity[opportunity > 0]
    return {
        "key": bucket["key"],
        "label": bucket["label"],
        "observed_disagreements": int(bucket["observed"]),
        "sampled_positions": samples,
        "sampled_games": int(np.count_nonzero(denominator)),
        "weighted_positions": weight,
        "effective_positions": weight * weight / bucket["weight_sq"],
        "frequency": bucket["observed"] / max(1, eligible.sum()),
        "effect": float(numerator.sum() / weight),
        "ci": np.quantile(boot_mean, [0.025, 0.975]).tolist(),
        "burden": float(numerator.sum() / max(1, eligible.sum())),
        "burden_ci": np.quantile(boot_burden, [0.025, 0.975]).tolist(),
        "supported": bool(np.count_nonzero(denominator) >= 20),
    }


def build_report(folder: Path, bootstraps=2000):
    progress = orjson.loads((folder / "progress.json").read_bytes())
    if progress["status"] != "complete":
        raise ValueError("Finish the rollout job before analyzing it")
    n = progress["games_completed"]
    eligible = np.zeros(n)
    buckets = {}
    roots = []
    discarded = truncated = all_disagreements = valid_pairs = 0
    visited = set()

    def bucket(kind, key):
        label = key
        if kind == "pair":
            a, b = map(int, key.split(":"))
            label = f"{name(a)} instead of {name(b)}"
        return buckets.setdefault(
            (kind, key),
            {
                "key": key,
                "label": label,
                "num": np.zeros(n),
                "den": np.zeros(n),
                "observed": 0,
                "samples": 0,
                "weight_sq": 0,
            },
        )

    for path in sorted(folder.glob("game-*.json")):
        game = orjson.loads(path.read_bytes())
        gi = game["game"]
        if gi in visited or not 0 <= gi < n:
            raise ValueError("Duplicate or invalid source game")
        visited.add(gi)
        eligible[gi] = game["counts"].get("eligible_positions", 0)
        truncated += int(game["source_truncated"])
        for event in game["events"]:
            if event["disagreement"]:
                all_disagreements += 1
                for kind, key in keys(event).items():
                    bucket(kind, key)["observed"] += 1
        for root in game["roots"]:
            outcomes = root["outcomes"]
            valid = [(a, b) for a, b in outcomes if a is not None and b is not None]
            discarded += len(outcomes) - len(valid)
            if not valid:
                continue
            if any(a not in (0, 1) or b not in (0, 1) for a, b in valid):
                raise ValueError("Non-binary terminal outcome")
            valid_pairs += len(valid)
            delta = float(np.mean([a - b for a, b in valid]))
            weight = 1 / root["inclusion_probability"]
            for kind, key in keys(root).items():
                group = bucket(kind, key)
                group["num"][gi] += weight * delta
                group["den"][gi] += weight
                group["samples"] += 1
                group["weight_sq"] += weight * weight
            plus = sum(a > b for a, b in valid)
            minus = sum(a < b for a, b in valid)
            o = root["observation"]
            feature = root["candidates"][0]["features"]
            roots.append(
                {
                    "id": f"{gi}/{root['root']}",
                    "game": gi,
                    "root": root["root"],
                    "turn": root["turn"],
                    "player": root["player_id"],
                    "teacher_card": root["teacher_card"],
                    "student_card": root["student_card"],
                    "teacher": name(root["teacher_card"]),
                    "student": name(root["student_card"]),
                    "category": keys(root)["category"],
                    "pair": keys(root)["pair"],
                    "heat": keys(root)["heat"],
                    "effect": delta,
                    "ci": paired_interval(plus, minus, len(valid)),
                    "weight": weight,
                    "valid_pairs": len(valid),
                    "teacher_wins": sum(a for a, b in valid),
                    "student_wins": sum(b for a, b in valid),
                    "teacher_only": plus,
                    "student_only": minus,
                    "both_win": sum(a == b == 1 for a, b in valid),
                    "both_lose": sum(a == b == 0 for a, b in valid),
                    "trade": root["total_trade"] - root["spent"],
                    "spent": root["spent"],
                    "own_authority": o["own_authority"],
                    "opponent_authority": o["opponent_authority"],
                    "deck_size": feature["Your total deck size (all zones)"],
                }
            )
    if len(visited) != n or sum(eligible) != progress["counts"]["eligible_positions"]:
        raise ValueError("Source-game coverage does not match the completed manifest")
    if all_disagreements != progress["counts"]["disagreements"]:
        raise ValueError("Disagreement frequency does not match the manifest")
    draws = (
        np.random.default_rng(20261006)
        .multinomial(n, np.full(n, 1 / n), size=bootstraps)
        .astype(float)
    )
    grouped = defaultdict(list)
    for (kind, _), b in buckets.items():
        summary = summarize_bucket(b, draws, eligible)
        if summary:
            grouped[kind].append(summary)
    histogram = []
    edges = [-1.01, -0.20, -0.10, -0.05, 0, 0.000001, 0.05, 0.10, 0.20, 1.01]
    labels = [
        "< −20",
        "−20 to −10",
        "−10 to −5",
        "−5 to <0",
        "Exactly 0",
        ">0 to 5",
        "5 to 10",
        "10 to 20",
        "≥20",
    ]
    for lo, hi, label in zip(edges[:-1], edges[1:], labels, strict=True):
        items = [r for r in roots if lo <= r["effect"] < hi]
        histogram.append(
            {"label": label, "count": len(items), "weighted_count": sum(r["weight"] for r in items)}
        )
    report = {
        "schema": SCHEMA,
        "id": folder.name,
        "reference": progress["reference_model"]["label"],
        "reference_id": progress["reference_model"]["ref"],
        "student_id": progress["settings"]["student"],
        "completed_at": progress["updated_at"],
        "reference_sha256": progress["reference_sha256"],
        "weights_sha256": progress["weights_sha256"],
        "games": n,
        "eligible_positions": int(sum(eligible)),
        "disagreements": all_disagreements,
        "disagreement_rate": all_disagreements / sum(eligible),
        "sampled_positions": len(roots),
        "valid_pairs": valid_pairs,
        "discarded_pairs": discarded,
        "truncated_source_games": truncated,
        "excluded_free_choices": progress["counts"].get("excluded_free_acquisition_decisions", 0),
        "rollouts_per_position": progress["settings"]["rollouts"],
        "bootstraps": bootstraps,
        "overall": grouped["all"][0],
        "categories": grouped["category"],
        "pairs": sorted(grouped["pair"], key=lambda x: -x["burden"]),
        "stages": sorted(grouped["stage"], key=lambda x: STAGES.index(x["key"])),
        "heat": grouped["heat"],
        "stage_labels": STAGES,
        "budget_labels": BUDGETS,
        "histogram": histogram,
        "positions": sorted(roots, key=lambda x: -x["effect"]),
        "method": {
            "effect": "Positive = champion choice increases rollout win probability relative to MiniAstro; negative = MiniAstro choice does better.",
            "weighting": "Inverse reservoir inclusion probability weights each sampled disagreement. Observed frequencies use every eligible source decision.",
            "interval": f"95% percentile intervals from {bootstraps:,} whole-source-game bootstrap resamples; exploratory, not adjusted for multiple comparisons.",
            "burden": "Weighted sum of signed local choice effects divided by all eligible acquisition decisions. This is a prioritization measure, not an additive prediction of full-game win-rate loss.",
            "scope": "Champion self-play; only acquisition-only MAIN decisions with at least two options. No free-acquisition or mixed play/purchase decisions. Both branches use the same public-belief draw and chance streams; champion plays both seats thereafter.",
            "position_interval": "Conservative paired 95% intervals using Bonferroni-adjusted Wilson bounds on the two discordant probabilities. Selected extreme examples need independent confirmation.",
        },
    }
    atomic_json(folder / "analysis.json", report)
    return report


def safe_folder(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", run_id):
        raise ValueError("Invalid analysis ID")
    folder = root / run_id
    if not (folder / "analysis.json").is_file():
        raise FileNotFoundError("Analysis not found")
    return folder


def position_detail(folder, game_id, root_id):
    game = orjson.loads((folder / f"game-{game_id:05d}.json").read_bytes())
    row = next(r for r in game["roots"] if r["root"] == root_id)
    o = row["observation"]

    def cards(items):
        return [
            {"card": name(cid), "count": count}
            for cid, count in sorted(Counter(c["card_id"] for c in items if c is not None).items())
        ]

    return {
        "trade_row": cards(o["trade_row"]),
        "hand": cards(o["hand"]),
        "own_deck": cards(
            [
                *o["hand"],
                *o["own_deck"],
                *o["own_known_top"],
                *o["own_discard"],
                *(e["card"] for e in o["own_in_play"]),
            ]
        ),
        "opponent_deck": cards(
            [
                *o["opponent_hidden"],
                *o["opponent_known_hand"],
                *o["opponent_known_top"],
                *o["opponent_discard"],
                *(e["card"] for e in o["opponent_in_play"]),
            ]
        ),
        "candidates": [
            {
                "card": name(c["card_id"]),
                "score": c["score"],
                "faction_support": c["features"]["Candidate faction cards in your deck"],
            }
            for c in row["candidates"]
        ],
        "current_trade": o["trade"],
        "total_trade": row["total_trade"],
        "spent": row["spent"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = build_report(args.folder)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "games",
                    "disagreements",
                    "sampled_positions",
                    "discarded_pairs",
                    "overall",
                    "categories",
                ]
            },
            indent=2,
        )
    )
