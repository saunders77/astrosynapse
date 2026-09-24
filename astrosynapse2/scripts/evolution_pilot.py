"""Bounded algorithm assay on the unchanged historical game runtime."""

from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import sys
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/progressive/autonomous-20260920/runtime"
sys.path.insert(0, str(ROOT / "backend/astro2"))
from evolution import mutate, read_actor, write_actor  # noqa: E402

sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(FROZEN / "scripts"))
from planning_experiment import pair, summary  # noqa: E402


def main():
    folder = ROOT / "data/evolution-20260923/pilot"
    parent = str(ROOT / "data/progressive/autonomous-20260920/source.actor.npz")
    weights = read_actor(parent)
    configs = [("parent", 0, 0, parent)]
    for operator, scale in [
        ("main_output", 0.003),
        ("main_output", 0.012),
        ("all_outputs", 0.006),
        ("head_mixture", 0.5),
        ("action_features", 0.002),
    ]:
        for sign in (1, -1):
            actor = str(folder / f"{operator}-{scale}-{sign}.actor.npz")
            write_actor(
                mutate(weights, seed=92371, operator=operator, scale=scale, sign=sign), actor
            )
            configs.append((operator, scale, sign, actor))
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for operator, scale, sign, actor in configs:
            path = folder / (Path(actor).stem + ".pairs.json")
            if path.exists():
                rows = json.loads(path.read_text())
            else:
                rows = list(
                    pool.map(
                        pair,
                        [
                            (
                                actor,
                                parent,
                                dict(rollouts=0, native=True, rules_version=1),
                                202609239271,
                                i,
                            )
                            for i in range(512)
                        ],
                        chunksize=4,
                    )
                )
                path.write_text(json.dumps(rows))
            result = dict(operator=operator, scale=scale, sign=sign, actor=actor, **summary(rows))
            results.append(result)
            print(json.dumps(result), flush=True)
            (folder / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
