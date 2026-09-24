"""Fixed-budget independent arena for the full update and a KL-limited blend."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026092099)
    args = parser.parse_args()
    trial, out = args.trial.resolve(), args.output.resolve()
    out.mkdir(exist_ok=True, parents=True)
    if (out / "manifest.json").exists():
        raise ValueError("evaluation already exists; use a new directory")
    runtime = trial / "runtime"
    sys.path[:0] = [str(runtime), str(runtime / "scripts")]
    from astro2.counterfactual import load_positions
    from astro2.experiment_control import atomic_json, code_identity, sha256
    from astro2.native_actor import NativeActor
    from planning_experiment import pair, summary
    from safetensors.numpy import save_file

    started = time.monotonic()
    state = dict(status="preparing_blend", pairs_per_model=args.pairs, completed={})

    def persist(**values):
        state.update(values, elapsed_seconds=time.monotonic() - started, heartbeat=time.time())
        atomic_json(out / "state.json", state)

    persist()
    try:
        manifest = json.loads((trial / "manifest.json").read_text())
        if code_identity(runtime / "astro2", runtime / "scripts") != manifest["code_identity"]:
            raise ValueError("trial runtime identity changed")
        checkpoint = json.loads((trial / "state.json").read_text())
        source = trial / "source.actor.npz"
        candidate = Path(checkpoint["model"]).with_suffix(".actor.npz")
        if sha256(source) != manifest["source_sha256"]:
            raise ValueError("source actor changed")
        with np.load(source, allow_pickle=False) as archive:
            original = {k: archive[k] for k in archive.files}
        with np.load(candidate, allow_pickle=False) as archive:
            learned = {k: archive[k] for k in archive.files}
        if original.keys() != learned.keys():
            raise ValueError("incompatible model tensors")
        for name in original:
            if original[name].shape != learned[name].shape:
                raise ValueError(f"shape changed: {name}")
            if not name.startswith("head_outputs."):
                np.testing.assert_array_equal(original[name], learned[name], err_msg=name)
        actor = NativeActor.load(candidate)
        references, deltas = [], []
        for shard in checkpoint["shards"]:
            for row in load_positions(Path(shard)):
                if int(row["game"]) % 5:
                    continue
                baseline = np.asarray(row["teacher_logits"], dtype=np.float64) / 0.03
                prediction = actor.predict_options(
                    row["state"], row["actions"], int(row["family"])
                ).mean(axis=1)
                references.append(baseline)
                deltas.append(np.asarray(prediction, dtype=np.float64) / 0.03 - baseline)
        if not references:
            raise ValueError("missing held-out positions for blend selection")

        def log_policy(logits):
            shifted = logits - logits.max()
            return shifted - np.log(np.exp(shifted).sum())

        old = [log_policy(v) for v in references]
        probabilities = [np.exp(v) for v in old]

        def kl(fraction):
            return float(
                np.mean(
                    [
                        np.sum(p * (logp - log_policy(ref + fraction * delta)))
                        for p, logp, ref, delta in zip(
                            probabilities, old, references, deltas, strict=True
                        )
                    ]
                )
            )

        low, high = 0.0, 1.0
        for _ in range(24):
            middle = (low + high) / 2
            if kl(middle) <= 0.0199:
                low = middle
            else:
                high = middle
        blended = {
            name: (
                value + np.float32(low) * (learned[name] - value)
                if name.startswith("head_outputs.")
                else value
            )
            for name, value in original.items()
        }
        blend_path = out / "conservative.actor.npz"
        np.savez_compressed(blend_path, **blended)
        save_file(
            {k: v for k, v in blended.items() if k != "__spec_json__"},
            str(out / "conservative.safetensors"),
        )
        (out / "conservative.safetensors.json").write_text(
            (trial / "source.safetensors.json").read_text()
        )
        NativeActor.load(blend_path)
        models = {"full_update": str(candidate), "conservative": str(blend_path)}
        atomic_json(
            out / "manifest.json",
            dict(
                pairs_per_model=args.pairs,
                seed=args.seed,
                workers=args.workers,
                rules_version=1,
                models={
                    name: dict(path=path, sha256=sha256(path)) for name, path in models.items()
                },
                opponent=dict(path=str(source), sha256=sha256(source)),
                blend_fraction=low,
                blend_heldout_kl=kl(low),
                full_heldout_kl=kl(1.0),
                blend_selection="Largest interpolation fraction below KL 0.0199 on saved held-out positions; no arena outcomes used",
                runtime_identity=manifest["code_identity"],
                interval="Fixed-sample paired Hoeffding intervals; Bonferroni across two models for simultaneous 95% coverage",
                deployment="No automatic promotion or training restart",
            ),
        )
        results = {name: [] for name in models}
        persist(status="evaluating", blend_fraction=low, blend_heldout_kl=kl(low))
        with (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool,
            (out / "pairs.jsonl").open("w", buffering=1) as log,
        ):
            for first in range(0, args.pairs, 128):
                names, tasks = [], []
                for index in range(first, min(first + 128, args.pairs)):
                    for name, path in models.items():
                        names.append(name)
                        tasks.append(
                            (
                                path,
                                str(source),
                                {"rollouts": 0, "native": True, "rules_version": 1},
                                args.seed,
                                index,
                            )
                        )
                for name, row in zip(names, pool.map(pair, tasks, chunksize=2), strict=True):
                    results[name].append(row)
                    log.write(json.dumps(dict(model=name, **row)) + "\n")
                persist(
                    completed={name: len(rows) for name, rows in results.items()},
                    scores={
                        name: float(np.mean([r["scores"] for r in rows]))
                        for name, rows in results.items()
                    },
                )
        radius = math.sqrt(math.log(2 * len(models) / 0.05) / (2 * args.pairs))
        reports = {}
        for name, rows in results.items():
            report = summary(rows)
            report["simultaneous95"] = [
                max(0.0, report["score"] - radius),
                min(1.0, report["score"] + radius),
            ]
            report["advantage_established"] = report["simultaneous95"][0] > 0.5
            reports[name] = report
        differences = np.array(
            [
                np.mean(a["scores"]) - np.mean(b["scores"])
                for a, b in zip(results["full_update"], results["conservative"], strict=True)
            ]
        )
        atomic_json(
            out / "result.json",
            dict(
                status="complete",
                reports=reports,
                blend_fraction=low,
                full_minus_conservative=float(differences.mean()),
                difference_standard_error=float(
                    differences.std(ddof=1) / np.sqrt(len(differences))
                ),
                wall_seconds=time.monotonic() - started,
                scope="Fresh fixed-budget evaluation, adjusted for both models; separate from campaign promotion evidence",
            ),
        )
        persist(status="complete", reports=reports)
    except BaseException:
        persist(status="failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
