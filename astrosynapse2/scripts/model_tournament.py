#!/usr/bin/env python3
"""Run a balanced seat-swapped tournament and fit mean-500 Bradley-Terry Elo."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import time
from pathlib import Path
from statistics import NormalDist

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from astro2.arena import ArenaConfig, ResolvedModel, _play_pair


ELO_SCALE = 400.0 / math.log(10.0)


def fit_elo(scores: np.ndarray, *, ridge_sd_elo: float = 800.0) -> np.ndarray:
    """Penalized Bradley-Terry fit, centered to a mean rating of 500."""
    players = scores.shape[0]
    ratings = np.zeros(players, dtype=np.float64)
    precision = (ELO_SCALE / ridge_sd_elo) ** 2
    for _ in range(100):
        gradient = -precision * ratings
        information = np.eye(players, dtype=np.float64) * precision
        for first in range(players):
            for second in range(first + 1, players):
                values = scores[first, second]
                if not len(values):
                    continue
                games = 2 * len(values)
                points = float(values.sum() * 2.0)
                probability = 1.0 / (1.0 + math.exp(-(ratings[first] - ratings[second])))
                gradient[first] += points - games * probability
                gradient[second] -= points - games * probability
                weight = games * probability * (1.0 - probability)
                information[first, first] += weight
                information[second, second] += weight
                information[first, second] -= weight
                information[second, first] -= weight
        step = np.linalg.solve(information, gradient)
        ratings += step
        ratings -= ratings.mean()
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return 500.0 + ELO_SCALE * ratings


def laplace_intervals(
    scores: np.ndarray,
    ratings_elo: np.ndarray,
    *,
    confidence: float,
    ridge_sd_elo: float = 800.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior-curvature interval that remains nonzero under complete separation."""
    ratings = (ratings_elo - 500.0) / ELO_SCALE
    players = len(ratings)
    precision = (ELO_SCALE / ridge_sd_elo) ** 2
    information = np.eye(players, dtype=np.float64) * precision
    for first in range(players):
        for second in range(first + 1, players):
            games = 2 * len(scores[first, second])
            probability = 1.0 / (1.0 + math.exp(-(ratings[first] - ratings[second])))
            weight = games * probability * (1.0 - probability)
            information[first, first] += weight
            information[second, second] += weight
            information[first, second] -= weight
            information[second, first] -= weight
    covariance = np.linalg.inv(information)
    center = np.eye(players) - np.ones((players, players)) / players
    covariance = center @ covariance @ center
    radius = NormalDist().inv_cdf(0.5 + confidence / 2.0) * ELO_SCALE * np.sqrt(np.diag(covariance))
    return ratings_elo - radius, ratings_elo + radius


def bootstrap_intervals(
    scores: np.ndarray, *, samples: int, seed: int, confidence: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fits = np.empty((samples, scores.shape[0]), dtype=np.float64)
    for sample in range(samples):
        resampled = np.empty_like(scores)
        for first in range(scores.shape[0]):
            for second in range(first + 1, scores.shape[0]):
                values = scores[first, second]
                indices = rng.integers(0, len(values), size=len(values))
                resampled[first, second] = values[indices]
        fits[sample] = fit_elo(resampled)
    tail = (1.0 - confidence) / 2.0
    return np.quantile(fits, tail, axis=0), np.quantile(fits, 1.0 - tail, axis=0)


def play(task):
    first, second, pair_index, model_a, model_b, config = task
    result = _play_pair(model_a, model_b, config, pair_index)
    return first, second, pair_index, result


def matrix_from_records(records: list[dict], players: int) -> np.ndarray:
    scores = np.empty((players, players), dtype=object)
    for first in range(players):
        for second in range(players):
            scores[first, second] = np.array([], dtype=np.float64)
    grouped = {(i, j): [] for i in range(players) for j in range(i + 1, players)}
    for record in records:
        grouped[record["matchup"]].append(
            (record["pair_index"], (record["first_score"] + record["second_score"]) / 2.0)
        )
    for matchup, values in grouped.items():
        values.sort()
        scores[matchup] = np.asarray([value for _, value in values], dtype=np.float64)
    return scores


def report(registry, records, bootstrap_samples, bootstrap_seed, elapsed, confidence):
    scores = matrix_from_records(records, len(registry))
    ratings = fit_elo(scores)
    bootstrap_low, bootstrap_high = bootstrap_intervals(
        scores, samples=bootstrap_samples, seed=bootstrap_seed, confidence=confidence
    )
    laplace_low, laplace_high = laplace_intervals(scores, ratings, confidence=confidence)
    low = np.minimum(bootstrap_low, laplace_low)
    high = np.maximum(bootstrap_high, laplace_high)
    rows = []
    for index, entry in enumerate(registry):
        wins = draws = losses = 0
        for record in records:
            first, second = record["matchup"]
            if index not in (first, second):
                continue
            pair_scores = [record["first_score"], record["second_score"]]
            if index == second:
                pair_scores = [1.0 - value for value in pair_scores]
            wins += sum(value == 1.0 for value in pair_scores)
            draws += sum(value == 0.5 for value in pair_scores)
            losses += sum(value == 0.0 for value in pair_scores)
        rows.append(
            {
                "level": entry["level"],
                "elo": float(ratings[index]),
                "elo_interval_low": float(low[index]),
                "elo_interval_high": float(high[index]),
                "elo_uncertainty_minus": float(ratings[index] - low[index]),
                "elo_uncertainty_plus": float(high[index] - ratings[index]),
                "wins": wins,
                "draws": draws,
                "losses": losses,
            }
        )
    matchups = len(registry) * (len(registry) - 1) // 2
    ordered = sorted(rows, key=lambda row: row["elo"], reverse=True)
    adjacent = [
        {
            "higher_level": higher["level"],
            "lower_level": lower["level"],
            "gap": higher["elo_interval_low"] - lower["elo_interval_high"],
            "higher_interval_width": higher["elo_interval_high"] - higher["elo_interval_low"],
            "lower_interval_width": lower["elo_interval_high"] - lower["elo_interval_low"],
        }
        for higher, lower in zip(ordered, ordered[1:])
    ]
    overlapping = [item for item in adjacent if item["gap"] <= 0.0]
    return {
        "method": "Bradley-Terry maximum a posteriori Elo with an 800-point Gaussian separation regularizer",
        "normalization": "arithmetic mean Elo = 500",
        "confidence": confidence,
        "uncertainty": "conservative envelope of a whole-pair bootstrap interval and a penalized-likelihood curvature interval (the latter covers complete separation)",
        "bootstrap_samples": bootstrap_samples,
        "average_pairs_per_matchup": len(records) / matchups,
        "average_games_per_matchup": 2 * len(records) / matchups,
        "matchup_pair_counts": {
            f"{first + 1}-{second + 1}": len(scores[first, second])
            for first in range(len(registry))
            for second in range(first + 1, len(registry))
        },
        "total_games": 2 * len(records),
        "truncated_games": sum(record["truncated_games"] for record in records),
        "elapsed_seconds": elapsed,
        "ratings": ordered,
        "adjacent_interval_gaps": adjacent,
        "all_intervals_nonoverlapping": all(item["gap"] > 0.0 for item in adjacent),
        "widest_overlapping_interval": (
            max(
                max(item["higher_interval_width"], item["lower_interval_width"])
                for item in overlapping
            )
            if overlapping
            else 0.0
        ),
        "maximum_interval_half_width": max(
            max(row["elo_uncertainty_minus"], row["elo_uncertainty_plus"]) for row in rows
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("../php-game/models/registry.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--max-total-pairs", type=int, default=20_000)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--overlap-width-stop", type=float, default=20.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026093001)
    parser.add_argument("--replay-counts-from", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    models = [
        ResolvedModel(
            ref=entry["id"],
            label=entry["name"],
            kind="checkpoint",
            actor_path=str((args.registry.resolve().parents[2] / entry["source"]).resolve()),
            checkpoint_id=entry.get("checkpoint_id"),
        )
        for entry in registry
    ]
    records_path = args.output.with_suffix(".pairs.json")
    records = json.loads(records_path.read_text()) if args.resume and records_path.exists() else []
    for record in records:
        record["matchup"] = tuple(record["matchup"])
    replay_counts = None
    if args.replay_counts_from and not records:
        prior = json.loads(args.replay_counts_from.read_text())
        replay_counts = {
            tuple(int(value) - 1 for value in matchup.split("-")): count
            for matchup, count in prior["matchup_pair_counts"].items()
        }
    started = time.monotonic()
    if records:
        current = report(
            registry,
            records,
            args.bootstrap_samples,
            args.seed + len(records),
            0.0,
            args.confidence,
        )
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        while True:
            counts = {
                (first, second): sum(record["matchup"] == (first, second) for record in records)
                for first in range(len(models))
                for second in range(first + 1, len(models))
            }
            if not records:
                scheduled = list(counts)
            else:
                overlap = min(current["adjacent_interval_gaps"], key=lambda item: item["gap"])
                first, second = overlap["higher_level"] - 1, overlap["lower_level"] - 1
                overlap_width = max(
                    overlap["higher_interval_width"], overlap["lower_interval_width"]
                )
                scheduled = (
                    list(counts)
                    if current["widest_overlapping_interval"] > args.overlap_width_stop
                    else [(min(first, second), max(first, second))]
                )
            tasks = []
            for first, second in scheduled:
                matchup_seed = args.seed + first * 1009 + second * 9176
                matchup_config = ArenaConfig(seed=matchup_seed, extension_enabled=False)
                target = (
                    replay_counts[first, second]
                    if replay_counts is not None and not records
                    else counts[first, second] + args.batch_pairs
                )
                for pair_index in range(counts[first, second], target):
                    tasks.append((first, second, pair_index, models[first], models[second], matchup_config))
            for first, second, pair_index, result in pool.map(play, tasks, chunksize=1):
                records.append(
                    {
                        "matchup": (first, second),
                        "pair_index": pair_index,
                        "first_score": result["first_score"],
                        "second_score": result["second_score"],
                        "truncated_games": result["truncated_games"],
                    }
                )
            current = report(
                registry,
                records,
                args.bootstrap_samples,
                args.seed + len(records),
                time.monotonic() - started,
                args.confidence,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(current, indent=2) + "\n")
            records_path.write_text(json.dumps(records) + "\n")
            print(json.dumps({
                "minimum_pairs_per_matchup": min(current["matchup_pair_counts"].values()),
                "maximum_pairs_per_matchup": max(current["matchup_pair_counts"].values()),
                "total_games": current["total_games"],
                "minimum_adjacent_interval_gap": min(
                    item["gap"] for item in current["adjacent_interval_gaps"]
                ),
                "elapsed_seconds": current["elapsed_seconds"],
            }), flush=True)
            worst_overlap = min(current["adjacent_interval_gaps"], key=lambda item: item["gap"])
            narrow_overlap = current["widest_overlapping_interval"] <= args.overlap_width_stop
            if current["all_intervals_nonoverlapping"] or narrow_overlap:
                current["stop_reason"] = (
                    "all 90% intervals are nonoverlapping"
                    if current["all_intervals_nonoverlapping"]
                    else "every still-overlapping adjacent interval is at most 20 Elo points wide"
                )
                args.output.write_text(json.dumps(current, indent=2) + "\n")
                break
            if len(records) >= args.max_total_pairs:
                raise RuntimeError("maximum total pair budget reached before uncertainty target")


if __name__ == "__main__":
    main()
