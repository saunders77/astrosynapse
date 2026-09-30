#!/usr/bin/env python3
"""Measure two historical champions against the established ten-level Elo field."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from astro2.arena import ArenaConfig, ResolvedModel, _play_pair
from model_tournament import fit_elo, matrix_from_records


def play(task):
    first, second, pair_index, model_a, model_b, config = task
    result = _play_pair(model_a, model_b, config, pair_index)
    return {
        "matchup": (first, second),
        "pair_index": pair_index,
        "first_score": result["first_score"],
        "second_score": result["second_score"],
        "truncated_games": result["truncated_games"],
    }


def anchored_fit(scores):
    ratings = fit_elo(scores)
    return ratings + (500.0 - float(ratings[:10].mean()))


def main():
    root = Path(__file__).resolve().parents[1]
    registry_path = root.parent / "php-game/models/registry.json"
    base_pairs_path = root / "data/analysis/web-level-tournament-90-nonoverlap-20260930.pairs.json"
    output = root / "data/analysis/champion-candidate-measurement-20260930.json"
    pair_output = output.with_suffix(".pairs.json")
    registry = json.loads(registry_path.read_text())
    candidates = [
        {
            "id": "a4a66dd88a0a41e7",
            "name": "Astro4 champion at 202,368 games",
            "actor": root / "data/checkpoints/525e496fa93b/g0000202368-1786781135358940000-7160c08cd7e04943b2d4988bc64c10dd.actor.npz",
        },
        {
            "id": "8844ddc7295a4f60",
            "name": "Astro5 champion at 668,032 games",
            "actor": root / "data/analysis/astro5-champion-8844ddc7295a4f60.actor.npz",
        },
    ]
    models = [
        ResolvedModel(
            ref=entry["id"], label=entry["name"], kind="checkpoint",
            actor_path=str((registry_path.resolve().parents[2] / entry["source"]).resolve()),
            checkpoint_id=entry.get("checkpoint_id"),
        )
        for entry in registry
    ] + [
        ResolvedModel(ref=item["id"], label=item["name"], kind="checkpoint", actor_path=str(item["actor"]))
        for item in candidates
    ]
    seed = 2026093017
    pairs_per_matchup = 100
    tasks = []
    for existing in range(10):
        for candidate in (10, 11):
            config = ArenaConfig(
                seed=seed + existing * 1009 + candidate * 9176,
                extension_enabled=False,
            )
            for pair_index in range(pairs_per_matchup):
                tasks.append((existing, candidate, pair_index, models[existing], models[candidate], config))
    started = time.monotonic()
    if pair_output.exists():
        candidate_records = json.loads(pair_output.read_text())
        for record in candidate_records:
            record["matchup"] = tuple(record["matchup"])
    else:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=context) as pool:
            candidate_records = list(pool.map(play, tasks, chunksize=1))
        pair_output.write_text(json.dumps(candidate_records) + "\n")
    base_records = json.loads(base_pairs_path.read_text())
    for record in base_records:
        record["matchup"] = tuple(record["matchup"])
    records = base_records + candidate_records
    scores = matrix_from_records(records, 12)
    ratings = anchored_fit(scores)
    bootstrap_samples = 2000
    rng = np.random.default_rng(seed + 1)
    fits = np.empty((bootstrap_samples, 12), dtype=np.float64)
    for sample in range(bootstrap_samples):
        resampled = np.empty_like(scores)
        for first in range(12):
            for second in range(12):
                resampled[first, second] = np.array([], dtype=np.float64)
        for first in range(12):
            for second in range(first + 1, 12):
                values = scores[first, second]
                if not len(values):
                    continue
                indices = rng.integers(0, len(values), size=len(values))
                resampled[first, second] = values[indices]
        fits[sample] = anchored_fit(resampled)
    low, high = np.quantile(fits, [0.05, 0.95], axis=0)
    rows = []
    for index, item in enumerate([*registry, *candidates]):
        relevant = [record for record in candidate_records if index in record["matchup"]]
        points = games = 0.0
        for record in relevant:
            first, second = record["matchup"]
            values = [record["first_score"], record["second_score"]]
            if index == second:
                values = [1.0 - value for value in values]
            points += sum(values)
            games += 2
        rows.append(
            {
                "model": item.get("name") or f"Level {item['level']}",
                "id": item["id"],
                "level": item.get("level"),
                "elo": float(ratings[index]),
                "elo_90_low": float(low[index]),
                "elo_90_high": float(high[index]),
                "candidate_test_games": int(games),
                "candidate_test_score": points / games if games else None,
            }
        )
    report = {
        "method": "Bradley-Terry Elo fitted with the established 37,200-game field plus fresh candidate games; original ten models recentered to mean 500",
        "uncertainty": "90% percentile interval from 2,000 bootstrap resamples of whole seat-swapped pairs",
        "seed": seed,
        "candidate_matchups": 20,
        "pairs_per_candidate_level_matchup": pairs_per_matchup,
        "fresh_games": 2 * len(candidate_records),
        "truncated_games": sum(record["truncated_games"] for record in candidate_records),
        "elapsed_seconds": time.monotonic() - started,
        "ratings": sorted(rows, key=lambda row: row["elo"], reverse=True),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
