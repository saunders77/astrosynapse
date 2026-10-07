"""Measure signed teacher-minus-student action value with paired terminal rollouts.

Fresh champion self-play supplies acquisition-only MAIN positions. Up to two
disagreements per game are uniformly reservoir sampled. Every legal comparison
shares a sampled public-information belief and chance streams; the champion continues
both seats after either forced action. No training or automatic model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astro2.arena import _derived_seed, resolve_model
from astro2.card_analysis import _GreedyChooser, _load_actor_encoder
from astro2.cards import CARD_BY_ID
from astro2.counterfactual import continuation_choosers
from astro2.engine import (
    ActionKind,
    DecisionFamily,
    Game,
    GameConfig,
    _TruncateGame,
    model_action_indices,
)
from astro2.experiment_control import atomic_json
from astro2.miniastro import feature_vector, forward, load_weights, transform
from astro2.planning import sample_belief
from astro2.storage import Store


def card_id(action):
    return action.card_id if action.kind == ActionKind.ACQUIRE else -1


def student_choice(artifact, decision, spent):
    actions = [decision.actions[i] for i in model_action_indices(decision)]
    if (
        decision.family != DecisionFamily.MAIN
        or len(actions) < 2
        or any(a.kind not in {ActionKind.ACQUIRE, ActionKind.END_TURN} for a in actions)
    ):
        return None
    # Same deterministic tie rule as the exported student.
    actions.sort(
        key=lambda a: (
            (-1, "")
            if card_id(a) == -1
            else (CARD_BY_ID[card_id(a)].cost, CARD_BY_ID[card_id(a)].name)
        )
    )
    total = decision.observation.trade + spent
    rows = [feature_vector(decision.observation, card_id(a), total, spent) for a in actions]
    if list(rows[0]) != artifact["feature_names"]:
        raise ValueError("Feature schema changed")
    weights = load_weights(artifact["weights_path"])
    x = transform(
        np.array([list(row.values()) for row in rows], dtype=np.float32),
        weights["mean"],
        weights["scale"],
    )
    scores = forward(weights, artifact["architecture"], x)[0]
    return (
        actions[int(scores.argmax())],
        total,
        [
            {"card_id": card_id(a), "score": float(s), "features": row}
            for a, s, row in zip(actions, scores, rows, strict=True)
        ],
    )


def paired_summary(outcomes):
    valid = [(a, b) for a, b in outcomes if a is not None and b is not None]
    counts = Counter(f"{a}{b}" for a, b in valid)
    n = len(valid)
    return {
        "valid_pairs": n,
        "discarded_pairs": len(outcomes) - n,
        "teacher_wins": sum(a for a, _ in valid),
        "student_wins": sum(b for _, b in valid),
        "teacher_only_wins": counts["10"],
        "student_only_wins": counts["01"],
        "both_win": counts["11"],
        "both_lose": counts["00"],
        "teacher_minus_student": (counts["10"] - counts["01"]) / n if n else None,
    }


def run_game(task):
    index, settings = task
    started = time.monotonic()
    artifact = json.loads(Path(settings["student_path"]).read_text())
    actor, encoder = _load_actor_encoder(settings["actor_path"])
    seed = settings["seed"]
    rng = np.random.default_rng(_derived_seed(seed, index, "regret-reservoir"))
    roots, events = [], []
    counts = Counter()
    spending = {}

    def observe(pid, decision, selected):
        counts["decisions"] += 1
        key = (decision.observation.turn, pid)
        spent = spending.get(key, 0)
        compared = student_choice(artifact, decision, spent)
        if decision.family == DecisionFamily.FREE_ACQUIRE:
            counts["excluded_free_acquisition_decisions"] += 1
        if compared:
            alternative, total, candidates = compared
            counts["eligible_positions"] += 1
            disagree = card_id(alternative) != card_id(selected)
            event = {
                "game": index,
                "turn": decision.observation.turn,
                "player_id": pid,
                "teacher_card": card_id(selected),
                "student_card": card_id(alternative),
                "disagreement": disagree,
                "total_trade": total,
                "spent": spent,
                "candidates": candidates,
            }
            events.append(event)
            if disagree:
                counts["disagreements"] += 1
                slot = (
                    len(roots)
                    if len(roots) < settings["roots_per_game"]
                    else int(rng.integers(counts["disagreements"]))
                )
                if slot < settings["roots_per_game"]:
                    snapshot = game.fork()
                    snapshot.decision_hook = None
                    record = (
                        snapshot,
                        selected,
                        alternative,
                        {**event, "observation": decision.observation.to_dict()},
                    )
                    if slot == len(roots):
                        roots.append(record)
                    else:
                        roots[slot] = record
        # Account for ALL actual paid acquisitions, including earlier mixed
        # play/purchase decisions where the student is not evaluated.
        if selected.kind == ActionKind.ACQUIRE:
            spending[key] = spent + selected.amount

    game = Game(
        choosers=tuple(
            _GreedyChooser(actor, encoder, _derived_seed(seed, index, f"source-{pid}"))
            for pid in (0, 1)
        ),
        config=GameConfig(
            seed=_derived_seed(seed, index, "regret-source-game"),
            starting_player=index % 2,
            rules_version=artifact["config"]["rules_version"],
            max_turns=240,
            max_actions_per_turn=220,
        ),
        decision_hook=observe,
    )
    source = game.run()
    results = []
    for ri, (snapshot, reference, alternative, event) in enumerate(roots):
        outcomes = []
        for draw in range(settings["rollouts"]):
            chance = _derived_seed(seed, index, f"regret-belief-{ri}-{draw}")
            belief = sample_belief(snapshot, event["player_id"], chance)
            pair = []
            for action in (reference, alternative):
                branch = belief.fork()
                branch.choosers = continuation_choosers(actor, actor, event["player_id"], chance)
                outcome = None
                try:
                    result = branch.continue_from_main_action(action)
                    if not result.truncated and result.winner is not None:
                        outcome = int(result.winner == event["player_id"])
                except _TruncateGame:
                    pass
                pair.append(outcome)
            outcomes.append(pair)
        results.append(
            {
                **event,
                "root": ri,
                "inclusion_probability": len(roots) / counts["disagreements"],
                "outcomes": outcomes,
                **paired_summary(outcomes),
            }
        )
    return {
        "game": index,
        "counts": dict(counts),
        "source_truncated": source.truncated,
        "seconds": time.monotonic() - started,
        "events": events,
        "roots": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="669746289fef4e71ad5caf056559d2e6")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--roots-per-game", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--reference-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.games, args.roots_per_game, args.rollouts, args.workers) < 1:
        raise ValueError("Budgets must be positive")
    folder = Path("data/acquire_students") / args.student
    artifact = json.loads((folder / "student.json").read_text())
    assert artifact["parameter_count"] == 956 and artifact["student_type"] == "miniastro"
    reference = resolve_model(Store(Path("data/astrosynapse2.sqlite3")), args.reference_model)
    args.output.mkdir(parents=True, exist_ok=False)
    actor_path = args.output / "reference.actor.npz"
    shutil.copyfile(reference.actor_path, actor_path)
    weights_path = args.output / "student.npz"
    shutil.copyfile(artifact["weights_path"], weights_path)
    artifact["weights_path"] = str(weights_path.resolve())
    atomic_json(args.output / "student.json", artifact)
    settings = {
        **vars(args),
        "output": str(args.output),
        "student_path": str((args.output / "student.json").resolve()),
        "actor_path": str(actor_path.resolve()),
    }
    progress = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": time.time(),
        "settings": settings,
        "reference_model": reference.to_dict(),
        "reference_sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
        "student_training_teacher_sha256": artifact["teacher_sha256"],
        "weights_sha256": hashlib.sha256(Path(artifact["weights_path"]).read_bytes()).hexdigest(),
        "games_completed": 0,
        "roots_completed": 0,
        "branches_completed": 0,
        "method": "Uniform disagreement reservoir per fresh reference-champion self-play game; paired public-belief terminal rollouts; reference champion continues both seats; paid acquisition-only MAIN decisions; signed reference-minus-student value (teacher fields mean reference champion); retain negative values.",
        "counts": {},
    }
    atomic_json(args.output / "progress.json", progress)
    started = time.monotonic()
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn")
        ) as pool:
            tasks = [pool.submit(run_game, (i, settings)) for i in range(args.games)]
            for future in as_completed(tasks):
                result = future.result()
                atomic_json(args.output / f"game-{result['game']:05d}.json", result)
                progress["games_completed"] += 1
                progress["roots_completed"] += len(result["roots"])
                progress["branches_completed"] += 2 * args.rollouts * len(result["roots"])
                counts = Counter(progress["counts"])
                counts.update(result["counts"])
                progress["counts"] = dict(counts)
                progress["elapsed_seconds"] = time.monotonic() - started
                progress["estimated_seconds_remaining"] = (
                    progress["elapsed_seconds"]
                    / progress["games_completed"]
                    * (args.games - progress["games_completed"])
                )
                progress["updated_at"] = time.time()
                atomic_json(args.output / "progress.json", progress)
                print(
                    json.dumps(
                        {
                            k: progress[k]
                            for k in [
                                "games_completed",
                                "roots_completed",
                                "branches_completed",
                                "elapsed_seconds",
                                "estimated_seconds_remaining",
                            ]
                        }
                    ),
                    flush=True,
                )
        progress["status"] = "complete"
    except BaseException as exc:
        progress.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        progress["updated_at"] = time.time()
        atomic_json(args.output / "progress.json", progress)


if __name__ == "__main__":
    main()
