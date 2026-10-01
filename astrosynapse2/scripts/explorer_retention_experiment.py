"""Fixed-budget gen10 arena with an external Explorer-retention rule."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

for key in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from astro2.arena import _ActorChooser, _derived_seed
from astro2.engine import ActionKind, Game, GameConfig, Seating, model_action_indices
from astro2.engine_encoding import EngineEncoder
from astro2.experiment_control import atomic_json, code_identity, sha256
from astro2.native_actor import NativeActor

_actor = None


def explorer_scrap(action):
    return action.card_id == 2 and action.kind in {
        ActionKind.SCRAP_FOR_ABILITY, ActionKind.SCRAP_CARD,
    }


class RetentionChooser(_ActorChooser):
    def __init__(self, *args, enabled=True):
        super().__init__(*args)
        self.enabled = enabled
        self.overrides = Counter()

    def __call__(self, player, decision):
        selected = super().__call__(player, decision)
        turn = decision.observation.turn
        if not (self.enabled and 5 <= turn <= 8 and explorer_scrap(selected)):
            return selected
        # Preserve the original encoded legal set and state. Only remove banned
        # choices from ranking, falling back from a banned lethal override.
        eligible = [i for i in model_action_indices(replace(decision, opaque=None))
                    if not explorer_scrap(decision.actions[i])]
        if not eligible:
            raise RuntimeError("No allowed action under Explorer retention rule")
        encoded = self.encoder.encode_decision(decision.observation, decision)
        index, _ = self.actor.choose(encoded.state, encoded.actions[eligible],
                                     int(encoded.family), epsilon=0.0,
                                     head=None, rng=self.rng)
        self.overrides[str(turn)] += 1
        return decision.actions[eligible[index]]


def pair(task):
    global _actor
    path, seed, index, enabled = task
    if _actor is None:
        _actor = NativeActor.load(path)
    scores, overrides, stats = [], Counter(), Counter()
    for seat in (0, 1):
        a = RetentionChooser(_actor, EngineEncoder(version=_actor.spec.encoder_version),
                             _derived_seed(seed, index, "model_a"), enabled=enabled)
        b = _ActorChooser(_actor, EngineEncoder(version=_actor.spec.encoder_version),
                          _derived_seed(seed, index, "model_b"))
        opportunities, scraps = set(), set()

        def hook(player, decision, selected):
            label = "modified" if player == seat else "original"
            turn = decision.observation.turn
            if any(x.kind == ActionKind.SCRAP_FOR_ABILITY and x.card_id == 2
                   for x in decision.actions):
                opportunities.add((label, turn))
            if selected.kind == ActionKind.SCRAP_FOR_ABILITY and selected.card_id == 2:
                scraps.add((label, turn))
            if enabled and player == seat and 5 <= turn <= 8:
                assert not explorer_scrap(selected), "Explorer retention violated"

        game = Game(choosers=(a, b) if seat == 0 else (b, a), decision_hook=hook,
                    config=GameConfig(seed=_derived_seed(seed, index, "game"),
                                      seating=Seating.FIXED, starting_player=0,
                                      max_turns=400, max_actions_per_turn=200,
                                      rules_version=2))
        result = game.run()
        if result.truncated:
            raise RuntimeError(f"Truncated pair {index}: {result.truncation_reason}")
        scores.append(float(result.winner == seat))
        overrides.update(a.overrides)
        for label, turn in opportunities:
            stats[f"{label}:{turn}:eligible"] += 1
            stats[f"{label}:{turn}:scrapped"] += int((label, turn) in scraps)
    return dict(pair=index, scores=scores, overrides=dict(overrides), stats=dict(stats))


def summarize(rows):
    values = np.asarray([r["scores"] for r in rows])
    paired = values.mean(axis=1)
    score = float(paired.mean())
    se = float(paired.std(ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else 0.0
    radius = math.sqrt(math.log(40) / (2 * len(rows)))
    overrides, stats = Counter(), Counter()
    for row in rows:
        overrides.update(row["overrides"])
        stats.update(row["stats"])
    return dict(pairs=len(rows), games=2 * len(rows), modified_wins=int(values.sum()),
                original_wins=int(values.size-values.sum()), score=score,
                first_seat_score=float(values[:, 0].mean()),
                second_seat_score=float(values[:, 1].mean()),
                paired_standard_error=se,
                paired_normal95=[max(0, score-1.96*se), min(1, score+1.96*se)],
                hoeffding95=[max(0, score-radius), min(1, score+radius)],
                overrides=dict(overrides), stats=dict(stats), truncated=0,
                pairs_with_override=sum(bool(r["overrides"]) for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026093008)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(actor=str(args.actor.resolve()), actor_sha256=sha256(args.actor),
                    pairs=args.pairs, seed=args.seed, rules_version=2,
                    control=args.control, turn_counter="global, matching scrap chart",
                    intervention="Never scrap Explorer in global turns 5,6,7,8; otherwise gen10",
                    runtime=code_identity(Path(__file__).resolve().parents[1]/"backend/astro2",
                                          Path(__file__).resolve().parent))
    dest = args.output / "manifest.json"
    if dest.exists():
        assert json.loads(dest.read_text()) == manifest, "Experiment identity changed"
    else:
        atomic_json(dest, manifest)
    log_path = args.output / "pairs.jsonl"
    rows = [json.loads(x) for x in log_path.read_text().splitlines()] if log_path.exists() else []
    done = {r["pair"] for r in rows}
    assert len(done) == len(rows)
    tasks = [(str(args.actor.resolve()), args.seed, i, not args.control)
             for i in range(args.pairs) if i not in done]
    start = time.monotonic()
    with log_path.open("a", buffering=1) as log, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for row in pool.map(pair, tasks, chunksize=8):
            rows.append(row)
            log.write(json.dumps(row)+"\n")
            if len(rows) % 64 == 0 or len(rows) == args.pairs:
                summary = summarize(rows)
                summary.update(complete=len(rows) == args.pairs,
                               wall_seconds_this_session=time.monotonic()-start)
                atomic_json(args.output / "summary.json", summary)
                print(json.dumps({k:v for k,v in summary.items() if k != "stats"}), flush=True)
    assert sha256(args.actor) == manifest["actor_sha256"]


if __name__ == "__main__":
    main()
