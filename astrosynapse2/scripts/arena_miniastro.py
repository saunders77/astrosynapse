"""Seat-swapped arena between a MiniAstro acquisition policy and its teacher."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

import numpy as np
from astro2.arena import _derived_seed
from astro2.card_analysis import _GreedyChooser, _load_actor_encoder
from astro2.engine import (
    ActionKind,
    DecisionFamily,
    Game,
    GameConfig,
    Seating,
    model_action_indices,
)
from astro2.experiment_control import atomic_json
from astro2.miniastro import feature_vector, forward, load_weights, transform
from astro2.stats import elo_delta


class MiniAstroChooser:
    """Use MiniAstro for acquisition-only choices and gen10 everywhere else."""

    def __init__(self, artifact: dict, teacher, encoder, seed: int):
        self.artifact = artifact
        self.teacher = _GreedyChooser(teacher, encoder, seed)
        self.weights = load_weights(artifact["weights_path"])
        self.turn: int | None = None
        self.spent = 0
        self.decisions = 0

    def __call__(self, player_id, decision):
        if self.turn != decision.observation.turn:
            self.turn = decision.observation.turn
            self.spent = 0
        eligible = [decision.actions[i] for i in model_action_indices(decision)]
        paid = decision.family == DecisionFamily.MAIN and all(
            action.kind in {ActionKind.ACQUIRE, ActionKind.END_TURN} for action in eligible
        )
        free = decision.family == DecisionFamily.FREE_ACQUIRE and all(
            action.kind in {ActionKind.FREE_ACQUIRE, ActionKind.DECLINE} for action in eligible
        )
        if not (paid or free):
            return self.teacher(player_id, decision)

        choices = []
        for action in eligible:
            if action.kind in {ActionKind.END_TURN, ActionKind.DECLINE}:
                candidate = -1
            elif action.kind == ActionKind.FREE_ACQUIRE:
                candidate = action.target_card_id
            else:
                candidate = action.card_id
            choices.append((candidate, action))
        # At an acquisition-only MAIN decision, all resource-producing actions
        # have resolved. Current trade plus earlier purchase costs is therefore
        # the student's live full-turn-trade input.
        total_trade = decision.observation.trade + self.spent
        rows = [
            feature_vector(decision.observation, candidate, total_trade, self.spent)
            for candidate, _ in choices
        ]
        if list(rows[0]) != self.artifact["feature_names"]:
            raise RuntimeError("MiniAstro feature schema changed")
        x = transform(
            np.asarray([list(row.values()) for row in rows], dtype=np.float32),
            self.weights["mean"],
            self.weights["scale"],
        )
        scores = forward(self.weights, self.artifact["architecture"], x)[0]
        chosen = choices[int(np.argmax(scores))][1]
        self.decisions += 1
        if chosen.kind == ActionKind.ACQUIRE:
            self.spent += chosen.amount
        return chosen


def play_pair(task):
    pair, seed, artifact, actor_path, rules_version = task
    teacher, encoder = _load_actor_encoder(actor_path)
    game_seed = _derived_seed(seed, pair, "miniastro-arena-game")
    mini_seed = _derived_seed(seed, pair, "miniastro-arena-mini")
    teacher_seed = _derived_seed(seed, pair, "miniastro-arena-teacher")
    common = dict(
        config=GameConfig(
            rules_version=rules_version,
            seed=game_seed,
            seating=Seating.FIXED,
            starting_player=0,
            max_turns=240,
            max_actions_per_turn=220,
        )
    )
    mini_first = MiniAstroChooser(artifact, teacher, encoder, mini_seed)
    first = Game(
        player_names=("MiniAstro1000", "Gen10"),
        choosers=(mini_first, _GreedyChooser(teacher, encoder, teacher_seed)),
        **common,
    ).run()
    mini_second = MiniAstroChooser(artifact, teacher, encoder, mini_seed)
    second = Game(
        player_names=("Gen10", "MiniAstro1000"),
        choosers=(_GreedyChooser(teacher, encoder, teacher_seed), mini_second),
        **common,
    ).run()

    def score(result, mini_player):
        return 0.5 if result.winner is None else float(result.winner == mini_player)

    return {
        "pair": pair,
        "first": score(first, 0),
        "second": score(second, 1),
        "turns": first.turns + second.turns,
        "truncated": int(first.truncated) + int(second.truncated),
        "mini_decisions": mini_first.decisions + mini_second.decisions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--root", type=Path, default=Path("data/acquire_students"))
    parser.add_argument("--output", type=Path, default=Path("data/miniastro_arena_500.json"))
    args = parser.parse_args()
    if args.games <= 0 or args.games % 2:
        raise ValueError("games must be a positive even number for seat-swapped pairs")
    artifact = json.loads((args.root / args.student / "student.json").read_text())
    if artifact.get("student_type") != "miniastro":
        raise ValueError("student must be MiniAstro")
    actor_path = str(args.root / args.student / "teacher.actor.npz")
    pairs = args.games // 2
    completed = []
    tasks = [
        (pair, args.seed, artifact, actor_path, artifact["config"]["rules_version"])
        for pair in range(pairs)
    ]
    atomic_json(args.output, {"status": "running", "student": args.student, "games": args.games})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(play_pair, task) for task in tasks]
        for future in as_completed(futures):
            completed.append(future.result())
            if len(completed) % 10 == 0:
                atomic_json(
                    args.output,
                    {
                        "status": "running",
                        "student": args.student,
                        "pairs_completed": len(completed),
                        "games_completed": 2 * len(completed),
                    },
                )
                print(f"Games complete: {2 * len(completed)}/{args.games}", flush=True)
    scores = [score for row in completed for score in (row["first"], row["second"])]
    wins = sum(score == 1 for score in scores)
    losses = sum(score == 0 for score in scores)
    draws = len(scores) - wins - losses
    rate = sum(scores) / len(scores)
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z * z / len(scores)
    center = (rate + z * z / (2 * len(scores))) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / len(scores) + z * z / (4 * len(scores) ** 2))
        / denominator
    )
    result = {
        "status": "complete",
        "student": args.student,
        "teacher_model_id": artifact["model_id"],
        "games": len(scores),
        "pairs": pairs,
        "miniastro_wins": wins,
        "gen10_wins": losses,
        "draws": draws,
        "miniastro_score": rate,
        "wilson_95": {"low": center - radius, "high": center + radius},
        "elo_difference_miniastro_minus_gen10": elo_delta(rate),
        "miniastro_first_seat_score": sum(row["first"] for row in completed) / pairs,
        "miniastro_second_seat_score": sum(row["second"] for row in completed) / pairs,
        "truncated_games": sum(row["truncated"] for row in completed),
        "miniastro_acquisition_decisions": sum(row["mini_decisions"] for row in completed),
        "seed": args.seed,
        "method": "Gen10 controls non-acquisition decisions; MiniAstro controls paid/free acquisition and decline decisions. Paired common seeds with exact seat swap.",
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
