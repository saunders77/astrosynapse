"""Reproducible lineage-search assay using a campaign's unchanged game runtime.

These games are exploratory training data, never promotion evidence. The
manifest freezes actors, code, seed and sample size before evaluating proposals.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
import sys
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=202609260100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--actors", type=Path)
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output.resolve()
    runtime = campaign / "runtime"
    # Loading the proposal code independently does not replace any game module.
    source = Path(__file__).resolve().parents[1] / "backend/astro2/evolution.py"
    spec = importlib.util.spec_from_file_location("proposals", source)
    proposals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proposals)
    sys.path[:0] = [str(runtime), str(runtime / "scripts")]
    from astro2.experiment_control import atomic_json, code_identity, sha256
    from planning_experiment import pair, summary

    manifest = json.loads((campaign / "manifest.json").read_text())
    identity = code_identity(runtime / "astro2", runtime / "scripts")
    if identity != manifest["code_identity"]:
        raise ValueError("campaign runtime identity changed")
    state = json.loads((campaign / "state.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    champion = state["champion"]
    parent = proposals.read_actor(champion)
    saved = output / "manifest.json"
    previous = json.loads(saved.read_text()) if saved.exists() else None
    candidates = []
    if previous is not None:
        # Never regenerate into an evidence directory before checking identity.
        # In particular, a promoted champion or revised runtime must not replace
        # the actors whose old outcomes are already recorded here.
        candidates = [{k: v for k, v in c.items() if k != "sha256"} for c in previous["candidates"]]
        if args.actors and json.loads(args.actors.read_text()) != candidates:
            raise ValueError("pilot candidate list changed")
    elif args.actors:
        candidates = json.loads(args.actors.read_text())
    else:
        last = next(b for b in state["branches"] if b.get("actor") == champion)
        for name, origin in [
            ("lineage", campaign / "source.actor.npz"),
            ("last_step", Path(last["parent"])),
        ]:
            base = proposals.read_actor(origin)
            for scale in (0.5, 1.0, 2.0, 4.0):
                weights = {k: v.copy() for k, v in parent.items()}
                for key, value in weights.items():
                    if key != "__spec_json__" and not key.startswith("value_"):
                        weights[key] = (value + scale * (value - base[key])).astype(value.dtype)
                actor = output / f"{name}-{scale}.actor.npz"
                model = proposals.write_actor(weights, actor)
                candidates.append(dict(name=f"{name}-{scale}", actor=str(actor), model=model))
    plan = dict(
        champion=champion,
        champion_sha256=sha256(champion),
        code_identity=identity,
        seed=args.seed,
        pairs=args.pairs,
        rules_version=1,
        candidates=[{**c, "sha256": sha256(c["actor"])} for c in candidates],
        purpose="exploratory search; independent promotion evidence still required",
    )
    if previous is not None and previous != plan:
        raise ValueError("pilot identity changed")
    atomic_json(saved, plan)
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for candidate in candidates:
            path = output / (candidate["name"] + ".pairs.jsonl")
            rows = (
                [json.loads(line) for line in path.read_text().splitlines()]
                if path.exists()
                else []
            )
            if [r["pair"] for r in rows] != list(range(len(rows))):
                raise ValueError("pilot rows must be contiguous")
            with path.open("a", buffering=1) as log:
                tasks = [
                    (
                        candidate["actor"],
                        champion,
                        dict(rollouts=0, native=True, rules_version=1),
                        args.seed,
                        i,
                    )
                    for i in range(len(rows), args.pairs)
                ]
                for row in pool.map(pair, tasks, chunksize=4):
                    rows.append(row)
                    log.write(json.dumps(row) + "\n")
            result = dict(**candidate, **summary(rows))
            results.append(result)
            print(json.dumps(result), flush=True)
            atomic_json(output / "results.json", results)


if __name__ == "__main__":
    main()
