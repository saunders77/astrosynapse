"""Isolated, resumable seat-paired experiments. Never promotes into the live DB."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import multiprocessing
import os
import sqlite3
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from astro2.arena import _ActorChooser, _derived_seed
from astro2.engine import Game, GameConfig, Seating
from astro2.engine_encoding import EngineEncoder
from astro2.experiment_control import code_identity
from astro2.model import NumpyActor
from astro2.planning import FrozenChooser, PlanningChooser, PlanningConfig

_actors = {}


def actor(path, native=False):
    key = (path, native)
    if key not in _actors:
        if native:
            from astro2.native_actor import NativeActor

            _actors[key] = NativeActor.load(path)
        else:
            _actors[key] = NumpyActor.load(path)
        while len(_actors) > 4:
            del _actors[next(iter(_actors))]
    return _actors[key]


def pair(task):
    a_path, b_path, options, seed, index = task
    start = time.monotonic()
    config = PlanningConfig(**options)
    a, b = actor(a_path, config.native), actor(b_path, config.native)
    scores = []
    searches = changes = branches = truncated = 0
    for seat in (0, 1):
        chooser = (PlanningChooser if config.rollouts else FrozenChooser)(
            a, _derived_seed(seed, index, "model_a"), config
        )
        if config.auto_actor_path:
            auto = actor(config.auto_actor_path, config.native)
            chooser.automatic = _ActorChooser(
                auto,
                EngineEncoder(version=auto.spec.encoder_version),
                _derived_seed(seed, index, "auto"),
            )
        opponent = _ActorChooser(
            b, EngineEncoder(version=b.spec.encoder_version), _derived_seed(seed, index, "model_b")
        )
        choosers = (chooser, opponent) if seat == 0 else (opponent, chooser)
        game = Game(
            choosers=choosers,
            config=GameConfig(
                seed=_derived_seed(seed, index, "game"),
                seating=Seating.FIXED,
                starting_player=0,
                max_turns=180,
                max_actions_per_turn=160,
                rules_version=config.rules_version,
            ),
        )
        if isinstance(chooser, PlanningChooser):
            chooser.game = game
        result = game.run()
        scores.append(float(result.winner == seat) if not result.truncated else 0.0)
        truncated += result.truncated
        searches += getattr(chooser, "searches", 0)
        changes += getattr(chooser, "changes", 0)
        branches += getattr(chooser, "branches", 0)
    return dict(
        pair=index,
        scores=scores,
        seconds=time.monotonic() - start,
        searches=searches,
        changes=changes,
        branches=branches,
        truncated=truncated,
    )


def summary(rows):
    values = np.asarray([r["scores"] for r in rows])
    n = len(values)
    mean = float(values.mean())
    radius = float(np.sqrt(np.log(40) / (2 * n)))
    paired = values.mean(axis=1)
    return dict(
        pairs=n,
        games=2 * n,
        score=mean,
        first=float(values[:, 0].mean()),
        second=float(values[:, 1].mean()),
        hoeffding95=[max(0, mean - radius), min(1, mean + radius)],
        paired_standard_error=float(paired.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        **{
            k: sum(r[k] for r in rows)
            for k in ["seconds", "searches", "changes", "branches", "truncated"]
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="5583687405f845fc")
    p.add_argument("--opponent", default="08aa018c672847d9")
    p.add_argument("--pairs", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026090901)
    p.add_argument("--config", default="{}")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    options = dataclasses.asdict(PlanningConfig(**json.loads(args.config)))
    c = sqlite3.connect("file:data/astrosynapse2.sqlite3?mode=ro", uri=True)

    def resolve(ref):
        row = c.execute("select actor_path from checkpoints where id=?", (ref,)).fetchone()
        return row[0] if row else str(Path(ref).resolve())

    a, b = resolve(args.model), resolve(args.opponent)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        model=args.model,
        opponent=args.opponent,
        model_sha256=hashlib.sha256(Path(a).read_bytes()).hexdigest(),
        opponent_sha256=hashlib.sha256(Path(b).read_bytes()).hexdigest(),
        seed=args.seed,
        code_identity=code_identity(
            Path(__import__("astro2").__file__).parent, Path(__file__).parent
        ),
        config=options,
        planned_pairs=args.pairs,
        auto_actor_sha256=(
            hashlib.sha256(Path(options["auto_actor_path"]).read_bytes()).hexdigest()
            if options["auto_actor_path"]
            else None
        ),
    )
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("cannot resume a different experiment")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    results = out / "pairs.jsonl"
    rows = (
        [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    )
    done = {r["pair"] for r in rows}
    tasks = [(a, b, options, args.seed, i) for i in range(args.pairs) if i not in done]
    started = time.monotonic()
    with (
        results.open("a", buffering=1) as f,
        concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
    ):
        pending = {pool.submit(pair, t) for t in tasks}
        while pending:
            completed, pending = concurrent.futures.wait(
                pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in completed:
                row = future.result()
                rows.append(row)
                f.write(json.dumps(row) + "\n")
            if completed:
                report = summary(rows)
                report["wall_seconds_this_session"] = time.monotonic() - started
                (out / "summary.json").write_text(json.dumps(report, indent=2))
                if len(rows) % 16 == 0 or not pending:
                    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
