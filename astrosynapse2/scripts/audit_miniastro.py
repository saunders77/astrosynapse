"""Freeze trained MiniAstros and measure fidelity on fresh teacher-game seeds."""

import argparse
import json
from pathlib import Path

from astro2.acquire_student import TurnSampler
from astro2.arena import _derived_seed
from astro2.card_analysis import _GreedyChooser, _load_actor_encoder
from astro2.engine import Game, GameConfig
from astro2.experiment_control import atomic_json
from astro2.miniastro import recommend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", nargs="+", required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--root", type=Path, default=Path("data/acquire_students"))
    args = parser.parse_args()
    artifacts = [
        json.loads((args.root / sid / "student.json").read_text()) for sid in args.students
    ]
    assert len({a["teacher_sha256"] for a in artifacts}) == 1
    assert len({a["config"]["rules_version"] for a in artifacts}) == 1
    assert all(a["config"]["seed"] != args.seed for a in artifacts)
    actor, encoder = _load_actor_encoder(str(args.root / args.students[0] / "teacher.actor.npz"))
    counts = [
        {
            "turns": 0,
            "covered": 0,
            "correct": 0,
            "purchases": 0,
            "purchase_correct": 0,
            "none": 0,
            "none_correct": 0,
        }
        for _ in artifacts
    ]
    report_path = args.root / f"miniastro-fresh-audit-{args.seed}.json"
    for game_index in range(args.games):
        sampler = TurnSampler(_derived_seed(args.seed, game_index, "audit-moment"))
        chooser = _GreedyChooser(
            actor, encoder, _derived_seed(args.seed, game_index, "audit-policy")
        )
        game = Game(
            choosers=(chooser, chooser),
            config=GameConfig(
                seed=_derived_seed(args.seed, game_index, "audit-game"),
                rules_version=artifacts[0]["config"]["rules_version"],
                max_turns=240,
                max_actions_per_turn=220,
            ),
            decision_hook=sampler.observe,
        )
        result = game.run()
        for row in sampler.rows(game_index, truncated=result.truncated):
            if row["excluded_reason"] == "unfinished_turn":
                continue
            o = sampler.turns[(row["turn"], row["player_id"])]["observation"]
            for artifact, count in zip(artifacts, counts, strict=True):
                count["turns"] += 1
                if row["excluded_reason"]:
                    continue
                choice = recommend(artifact, o, row["total_trade"], row["spent"])["recommendation"][
                    "card_id"
                ]
                correct = int(choice == row["target"])
                count["covered"] += 1
                count["correct"] += correct
                kind = "purchases" if row["target"] != -1 else "none"
                count[kind] += 1
                count["purchase_correct" if kind == "purchases" else "none_correct"] += correct
        if (game_index + 1) % 25 == 0:
            atomic_json(
                report_path,
                {
                    "status": "running",
                    "games_completed": game_index + 1,
                    "seed": args.seed,
                    "counts": counts,
                },
            )
            print(f"Fresh games: {game_index + 1}/{args.games}", flush=True)
    results = []
    for sid, artifact, count in zip(args.students, artifacts, counts, strict=True):
        audit = {
            "games": args.games,
            "seed": args.seed,
            "turns": count["turns"],
            "all_turn_accuracy": count["correct"] / count["turns"],
            "accuracy": count["correct"] / count["covered"],
            "purchase_accuracy": count["purchase_correct"] / max(1, count["purchases"]),
            "none_accuracy": count["none_correct"] / max(1, count["none"]),
            "coverage": count["covered"] / count["turns"],
        }
        artifact["fresh_audit"] = audit
        atomic_json(args.root / sid / "student.json", artifact)
        job_path = args.root / sid / "job.json"
        job = json.loads(job_path.read_text())
        job["result"] = artifact
        atomic_json(job_path, job)
        results.append({"id": sid, "parameter_count": artifact["parameter_count"], **audit})
    atomic_json(report_path, {"status": "complete", "results": results})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
