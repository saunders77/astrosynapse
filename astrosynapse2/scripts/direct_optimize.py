"""Durable population search on full-game return, with fresh selection games.

This is a derivative-free training control: no critic, replay, or shaped reward
is involved. Every generation screens a population on common seed-paired games,
then ranks finalists on a new seed block. Final certification must use another
independent, fixed-size block; selection scores are not strength certificates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import multiprocessing
import sqlite3
import time
from pathlib import Path

import numpy as np
from astro2.experiment_control import code_identity
from astro2.planning import PlanningConfig
from planning_experiment import pair, summary


def evaluate(pool, model, opponent, vectors, seed, pairs, native=False, strategy_only=False):
    scores = [[] for _ in vectors]
    pending = {}
    for candidate, vector in enumerate(vectors):
        config = dataclasses.asdict(
            PlanningConfig(
                rollouts=0,
                residual=tuple(vector),
                native=native,
                residual_strategy_only=strategy_only,
            )
        )
        for index in range(pairs):
            task = (model, opponent, config, seed, index)
            pending[pool.submit(pair, task)] = candidate
    for future in concurrent.futures.as_completed(pending):
        scores[pending[future]].append(future.result())
    return [summary(rows) for rows in scores], scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="5583687405f845fc")
    p.add_argument("--opponent", default="08aa018c672847d9")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--screen-pairs", type=int, default=48)
    p.add_argument("--selection-pairs", type=int, default=192)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026090911)
    p.add_argument("--native", action="store_true")
    p.add_argument("--strategy-only", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)
    c = sqlite3.connect("file:data/astrosynapse2.sqlite3?mode=ro", uri=True)
    model = c.execute("select actor_path from checkpoints where id=?", (args.model,)).fetchone()[0]
    opponent = c.execute(
        "select actor_path from checkpoints where id=?", (args.opponent,)
    ).fetchone()[0]
    manifest = {
        **vars(args),
        "model_sha256": hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        "opponent_sha256": hashlib.sha256(Path(opponent).read_bytes()).hexdigest(),
        "code_identity": code_identity(
            Path(__import__("astro2").__file__).parent, Path(__file__).parent
        ),
    }
    manifest_path = out / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        ignored = {"generations", "workers", "resume"}
        if {k: v for k, v in previous.items() if k not in ignored} != {
            k: v for k, v in manifest.items() if k not in ignored
        }:
            raise ValueError("cannot resume changed inputs or code; start an explicit fork")
    elif manifest_path.exists():
        raise ValueError("use --resume or a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    state_path = out / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else dict(generation=0, center=[0.0] * 104, best=[0.0] * 104)
    )
    start = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for generation in range(state["generation"], args.generations):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, generation]))
            center = np.asarray(state["center"])
            best = np.asarray(state["best"])
            sigma = 0.08 * (0.97**generation)
            vectors = [best, center]
            for _ in range((args.population - 2 + 1) // 2):
                direction = rng.normal(size=104) * sigma
                # Correlate the two phase tables to search coherent card values.
                direction[49:98] = 0.75 * direction[:49] + 0.5 * direction[49:98]
                direction[:2] = 0
                direction[49:51] = 0
                vectors.extend([center + direction, center - direction])
            vectors = vectors[: args.population]
            screen_seed = int(rng.integers(2**62))
            screen, screen_rows = evaluate(
                pool,
                model,
                opponent,
                vectors,
                screen_seed,
                args.screen_pairs,
                args.native,
                args.strategy_only,
            )
            rank = np.argsort([-s["score"] for s in screen], kind="stable")
            finalists = [0] + [int(i) for i in rank if i != 0][:3]
            selection_seed = int(rng.integers(2**62))
            selection, selection_rows = evaluate(
                pool,
                model,
                opponent,
                [vectors[i] for i in finalists],
                selection_seed,
                args.selection_pairs,
                args.native,
                args.strategy_only,
            )
            selected = int(np.argmax([s["score"] for s in selection]))
            winner = finalists[selected]
            # Refine the population around a fresh-seed winner, retaining the
            # incumbent in every screen and every selection block.
            state["best"] = vectors[winner].tolist()
            elite = np.mean([vectors[int(i)] for i in rank[:4]], axis=0)
            state["center"] = (0.6 * vectors[winner] + 0.4 * elite).tolist()
            state["generation"] = generation + 1
            record = dict(
                generation=generation,
                sigma=sigma,
                screen_seed=screen_seed,
                selection_seed=selection_seed,
                screen=screen,
                finalists=finalists,
                selection=selection,
                winner=winner,
                best=state["best"],
                elapsed=time.monotonic() - start,
            )
            folder = out / f"g{generation:03d}"
            folder.mkdir(exist_ok=True)
            (folder / "results.json").write_text(json.dumps(record, indent=2))
            (folder / "pairs.json").write_text(
                json.dumps(dict(screen=screen_rows, selection=selection_rows))
            )
            (folder / "config.json").write_text(
                json.dumps(
                    dataclasses.asdict(
                        PlanningConfig(
                            rollouts=0,
                            residual=tuple(state["best"]),
                            native=args.native,
                            residual_strategy_only=args.strategy_only,
                        )
                    ),
                    indent=2,
                )
            )
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(state_path)
            print(
                json.dumps({k: v for k, v in record.items() if k not in ["best", "screen"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
