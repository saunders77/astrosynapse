"""Read-only forensic snapshot, explicitly excluding 2Champion branch lab runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from statistics import median


def main():
    out = Path("data/assessment_20260909")
    out.mkdir(exist_ok=True)
    db = sqlite3.connect("file:data/astrosynapse2.sqlite3?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    runs = {
        r["id"]: dict(r)
        for r in db.execute("SELECT * FROM runs WHERE lower(name) NOT LIKE '2champion branch lab%'")
    }
    jobs = json.loads((out / "historical_arenas.json").read_text())
    rows = []
    for rid, run in runs.items():
        config = json.loads(run.pop("config_json"))
        metrics = [
            json.loads(r[0])
            for r in db.execute(
                "SELECT payload_json FROM metrics WHERE run_id=? ORDER BY id DESC LIMIT 600", (rid,)
            )
        ]
        selected = {
            key
            for m in metrics
            for key in m
            if key.startswith(("objective_", "weighted_", "gradient_", "policy_", "search_"))
            or key
            in {
                "normalized_policy_entropy",
                "actor_sample_fraction",
                "importance_clipped_fraction",
                "reference_policy_kl",
                "brier",
                "replay_size",
                "games_per_second",
                "elapsed_seconds",
            }
        }
        aggregate = {}
        for key in selected:
            values = [m[key] for m in metrics if isinstance(m.get(key), (int, float))]
            if values:
                aggregate[key] = dict(median=median(values), last=values[0], samples=len(values))
        full = [
            j
            for j in jobs
            if j["run_id"] == rid
            and j["status"] == "complete"
            and j["promotion_tier"] not in ["canary", "diagnostic"]
        ]
        canaries = [
            j
            for j in jobs
            if j["run_id"] == rid and j["status"] == "complete" and j["promotion_tier"] == "canary"
        ]
        rows.append(
            dict(
                **run,
                config=config,
                recent_metrics=aggregate,
                latest_search_repeatability=next(
                    (m.get("search_repeatability") for m in metrics if "search_repeatability" in m),
                    None,
                ),
                full_evaluations=full,
                automatic_gates=[j for j in full if j.get("config", {}).get("automatic_promotion")],
                canaries=canaries,
                promotions=sum(bool((j.get("promotion") or {}).get("promoted")) for j in full),
                evaluation_pairs=sum(j.get("pairs_completed") or 0 for j in full),
                evaluation_seconds=sum(j.get("elapsed_seconds") or 0 for j in full),
            )
        )
    (out / "run_assessment.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "| Run | Training games | Automatic gates | Promotions | All comparison games | Last automatic score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["games"] < 10000:
            continue
        full = row["automatic_gates"]
        last = full[-1]["model_a_score"] if full else None
        lines.append(
            f"| {row['name']} (`{row['id']}`) | {row['games']:,} | {len(full)} | {row['promotions']} | {2 * row['evaluation_pairs']:,} | {last:.2%} |"
            if last is not None
            else f"| {row['name']} | {row['games']:,} | 0 | 0 | 0 | — |"
        )
    (out / "runs_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    recent = [r for r in rows if r["created_at"] >= 1787780000]
    print(
        "RECENT TOTAL",
        sum(r["games"] for r in recent),
        "training games",
        sum(2 * r["evaluation_pairs"] for r in recent),
        "evaluation games",
        sum(r["evaluation_seconds"] for r in recent) / 3600,
        "evaluation hours",
    )
    for r in recent:
        norms = {
            k: round(v["median"], 4)
            for k, v in r["recent_metrics"].items()
            if k.startswith("objective_") and any(s in k for s in ["norm", "cosine", "ratio"])
        }
        if norms:
            print(r["name"], norms)


if __name__ == "__main__":
    main()
