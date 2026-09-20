"""Fresh-game calibration and forced basic-card value swings for an Astro6 actor."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from astro2.arena import _ActorChooser, _derived_seed
from astro2.cards import CARD_BY_ID
from astro2.engine import ActionKind, Game, GameConfig, Seating, model_action_indices
from astro2.engine_encoding import EngineEncoder
from astro2.onpolicy import cached_actor, critic_calibration


def audit_game(task):
    actor_path, opponent_path, seed, index = task
    actor, opponent = cached_actor(actor_path), cached_actor(opponent_path)
    encoder = EngineEncoder(version=actor.spec.encoder_version)
    seat = index % 2
    rows, swings = [], []
    previous = None

    def observe(pid, decision, action):
        nonlocal previous
        if pid != seat:
            return
        encoded = encoder.encode_decision(decision.observation, decision)
        logits = actor.predict_values(encoded.state, np.array([int(encoded.family)]))[0]
        probability = float(np.mean(1 / (1 + np.exp(-np.clip(logits, -40, 40)))))
        forced = len(model_action_indices(decision)) == 1
        turn = decision.observation.turn
        rows.append((probability, int(encoded.family), forced))
        # Playing a forced Scout/Viper/Explorer supplies known resources, with
        # no draw or new trade-row information. Keep same-player, same-turn
        # transitions; other card plays can legitimately reveal information.
        if previous and previous["turn"] == turn and previous["basic_forced"]:
            swings.append(
                dict(
                    game=index,
                    turn=turn,
                    action=previous["action"],
                    before=previous["value"],
                    after=probability,
                    delta=probability - previous["value"],
                )
            )
        card = CARD_BY_ID.get(action.card_id)
        previous = dict(
            turn=turn,
            value=probability,
            action=action.label,
            basic_forced=forced
            and action.kind == ActionKind.PLAY_CARD
            and card is not None
            and card.name in {"Scout", "Viper", "Explorer"},
        )

    learner = _ActorChooser(actor, encoder, _derived_seed(seed, index, "learner"))
    other = _ActorChooser(
        opponent,
        EngineEncoder(version=opponent.spec.encoder_version),
        _derived_seed(seed, index, "opponent"),
    )
    result = Game(
        choosers=(learner, other) if seat == 0 else (other, learner),
        config=GameConfig(
            seed=_derived_seed(seed, index, "game"),
            seating=Seating.FIXED,
            starting_player=0,
            max_turns=180,
            max_actions_per_turn=160,
            rules_version=1,
        ),
        decision_hook=observe,
    ).run()
    return dict(
        rows=rows,
        swings=swings,
        truncated=result.truncated,
        target=0.5 if result.winner is None else float(result.winner == seat),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026091901)
    args = parser.parse_args()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = list(
            pool.map(
                audit_game,
                [(args.actor, args.opponent, args.seed, i) for i in range(args.games)],
                chunksize=2,
            )
        )
    valid = [r for r in results if not r["truncated"]]
    p, y, f, forced = [], [], [], []
    for result in valid:
        for probability, family, is_forced in result["rows"]:
            p.append(probability)
            y.append(result["target"])
            f.append(family)
            forced.append(is_forced)
    swings = [s for r in valid for s in r["swings"]]
    deltas = np.array([abs(s["delta"]) for s in swings])
    report = dict(
        actor=args.actor,
        opponent=args.opponent,
        rules_version=1,
        seed=args.seed,
        games=args.games,
        truncated=args.games - len(valid),
        policy="greedy mean-head deployment",
        calibration=critic_calibration(p, y, f, forced),
        forced_basic_card_transitions=len(swings),
        absolute_swing_quantiles=(
            dict(
                zip(
                    ["median", "p90", "p99", "maximum"],
                    np.quantile(deltas, [0.5, 0.9, 0.99, 1]).tolist(),
                    strict=True,
                )
            )
            if len(deltas)
            else {}
        ),
        largest_swings=sorted(swings, key=lambda s: abs(s["delta"]), reverse=True)[:20],
        scope="Calibration is specific to these policies, opponent and rules; positions within a game are correlated.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
