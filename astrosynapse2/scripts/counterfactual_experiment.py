"""A resumable, bounded action-comparison branch from a verified champion."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import shutil
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from astro2.counterfactual import (
    ComparisonConfig,
    collect_comparisons,
    comparison_loss,
    load_positions,
    save_positions,
    training_batch,
)
from astro2.experiment_control import atomic_json, code_identity, sha256
from planning_experiment import pair, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--roots", type=int, default=2)
    parser.add_argument("--anchors", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--kl-weight", type=float, default=0.1)
    parser.add_argument("--max-kl", type=float, default=0.02)
    parser.add_argument("--eval-pairs", type=int, default=4096)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--seed", type=int, default=2026092007)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.games, args.workers, args.epochs, args.batch_size, args.eval_pairs) < 1:
        raise ValueError("budgets must be positive")
    if (
        args.games < 10
        or min(args.hours, args.learning_rate, args.max_kl) <= 0
        or args.kl_weight < 0
    ):
        raise ValueError("invalid training settings")
    config = ComparisonConfig(roots=args.roots, anchors=args.anchors, rollouts=args.rollouts)
    out, campaign = args.output.resolve(), args.campaign.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = out / "runtime"
    frozen = runtime / "scripts/counterfactual_experiment.py"
    if Path(__file__).resolve() != frozen:
        # Freeze the campaign's exact engine, masks and evaluation code. Only
        # this new learner is added; historical rules v1 are not silently mixed
        # with the current application's corrected engine implementation.
        with (out / "bootstrap.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not frozen.exists():
                if args.resume:
                    raise ValueError("missing frozen experiment runtime")
                original = json.loads((campaign / "manifest.json").read_text())
                identity = code_identity(campaign / "runtime/astro2", campaign / "runtime/scripts")
                if identity != original["code_identity"]:
                    raise ValueError("campaign source identity changed")
                source_state = json.loads((campaign / "state.json").read_text())
                champion = Path(source_state["champion"])
                model = champion.with_suffix("").with_suffix(".safetensors")
                shutil.copytree(
                    campaign / "runtime",
                    runtime,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".cache"),
                )
                project = Path(__file__).resolve().parents[1]
                shutil.copy2(
                    project / "backend/astro2/counterfactual.py",
                    runtime / "astro2/counterfactual.py",
                )
                shutil.copy2(__file__, frozen)
                shutil.copy2(champion, out / "source.actor.npz")
                shutil.copy2(model, out / "source.safetensors")
                shutil.copy2(
                    model.with_suffix(".safetensors.json"), out / "source.safetensors.json"
                )
                settings = {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                    if k not in {"resume", "output", "campaign"}
                }
                atomic_json(
                    out / "manifest.json",
                    dict(
                        settings=settings,
                        comparison=asdict(config),
                        source=str(champion),
                        source_sha256=sha256(out / "source.actor.npz"),
                        model_sha256=sha256(out / "source.safetensors"),
                        code_identity=code_identity(runtime / "astro2", runtime / "scripts"),
                        rules_version=1,
                        evaluation="fixed 4096 paired seeds by default; exploratory, not a promotion",
                    ),
                )
                atomic_json(
                    out / "state.json", dict(phase="created", elapsed=0, games=0, epoch=0, pairs=0)
                )
            elif not args.resume:
                raise ValueError("existing experiment requires --resume")
        os.execve(
            sys.executable,
            [sys.executable, str(frozen), *sys.argv[1:]],
            {**os.environ, "PYTHONPATH": str(runtime)},
        )
    lock = (out / "experiment.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((out / "manifest.json").read_text())
    if code_identity(runtime / "astro2", runtime / "scripts") != manifest["code_identity"]:
        raise ValueError("frozen runtime changed")
    if (
        sha256(out / "source.actor.npz") != manifest["source_sha256"]
        or sha256(out / "source.safetensors") != manifest["model_sha256"]
    ):
        raise ValueError("frozen source model changed")
    # Resume restores all algorithm settings. --hours is a cumulative budget
    # and can be explicitly extended, while workers change throughput only.
    for key, value in manifest["settings"].items():
        if key not in {"hours", "workers"}:
            setattr(args, key, value)
    config = ComparisonConfig(**manifest["comparison"])
    state = json.loads((out / "state.json").read_text())
    elapsed_before, started = state["elapsed"], time.monotonic()
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if args.resume:
        (out / "STOP").unlink(missing_ok=True)

    def persist(**values):
        state.update(
            values,
            elapsed=elapsed_before + time.monotonic() - started,
            heartbeat=time.time(),
            pid=os.getpid(),
        )
        atomic_json(out / "state.json", state)

    def halt():
        return (
            stopped
            or (out / "STOP").exists()
            or elapsed_before + time.monotonic() - started >= args.hours * 3600
        )

    source = str(out / "source.actor.npz")
    (out / "dataset").mkdir(exist_ok=True)
    try:
        if state["phase"] in {"complete", "no_heldout_improvement", "insufficient_signal"}:
            return
        if state["games"] < args.games:
            persist(phase="collecting_terminal_comparisons")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                while state["games"] < args.games and not halt():
                    first = state["games"]
                    end = min(args.games, first + args.workers * 2)
                    results = list(
                        pool.map(
                            collect_comparisons,
                            [
                                (source, source, args.seed, i, asdict(config))
                                for i in range(first, end)
                            ],
                            chunksize=1,
                        )
                    )
                    rows = [r for positions, _ in results for r in positions]
                    if not rows:
                        raise RuntimeError("no usable comparison or anchor positions")
                    shard = out / "dataset" / f"games-{first:06d}-{end:06d}.npz"
                    save_positions(shard, rows)
                    atomic_json(shard.with_suffix(".json"), [stats for _, stats in results])
                    persist(games=end, shards=[*state.get("shards", []), str(shard)])
            if halt():
                persist(phase="paused")
                return
        rows = [r for name in state["shards"] for r in load_positions(Path(name))]
        # All positions from one source game go to the same split. Nearby
        # decisions must not leak across training and validation.
        training = [r for r in rows if int(r["game"]) % 5 != 0]
        heldout = [r for r in rows if int(r["game"]) % 5 == 0]
        roots = sum(int(r["samples"] > 0) for r in rows)
        discordant = sum(float(r["wins"].sum() + r["losses"].sum()) for r in rows)
        persist(positions=len(rows), roots=roots, discordant=discordant)
        if (
            not training
            or not heldout
            or not any(r["samples"] > 0 for r in heldout)
            or discordant == 0
        ):
            persist(phase="insufficient_signal")
            return
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from astro2.model import (
            export_actor,
            load_model,
            load_optimizer_state,
            save_model,
            save_optimizer_state,
        )

        mx.random.seed(args.seed)
        mx.set_cache_limit(512 * 1024 * 1024)
        model, spec = load_model(state.get("model", str(out / "source.safetensors")))
        model.freeze()
        for head in model.head_outputs:
            head.unfreeze()
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        if state["epoch"] and not load_optimizer_state(optimizer, state["optimizer"]):
            raise RuntimeError("missing optimizer for resume")

        def loss(*arrays):
            return comparison_loss(model, *arrays, kl_weight=args.kl_weight)[0]

        gradient = nn.value_and_grad(model, loss)

        def measure(positions):
            ranking = kl = 0.0
            root_count = 0
            for offset in range(0, len(positions), args.batch_size):
                batch = positions[offset : offset + args.batch_size]
                _, metrics = comparison_loss(
                    model, *(mx.array(a) for a in training_batch(batch)), kl_weight=args.kl_weight
                )
                n = sum(int(r["samples"] > 0) for r in batch)
                ranking += float(metrics["ranking_loss"].item()) * n
                kl += float(metrics["anchor_kl"].item()) * len(batch)
                root_count += n
            result = dict(ranking_loss=ranking / max(root_count, 1), anchor_kl=kl / len(positions))
            if not all(np.isfinite(v) for v in result.values()):
                raise RuntimeError("nonfinite validation metrics")
            return result

        if "initial_heldout" not in state:
            persist(initial_heldout=measure(heldout))
        while state["epoch"] < args.epochs and not halt():
            epoch = state["epoch"]
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
            persist(phase="fitting_action_comparisons")
            # Each epoch has its own deterministic shuffle. Interrupted epochs
            # replay from the last committed model+optimizer, not partial work.
            order = rng.permutation(len(training))
            for offset in range(0, len(order), args.batch_size):
                if halt():
                    persist(phase="paused")
                    return
                batch = [training[int(i)] for i in order[offset : offset + args.batch_size]]
                value, grads = gradient(*(mx.array(a) for a in training_batch(batch)))
                grads, norm = optim.clip_grad_norm(grads, 1.0)
                mx.eval(value, norm)
                if not np.isfinite(float(value.item())) or not np.isfinite(float(norm.item())):
                    raise RuntimeError("nonfinite comparison objective or gradient")
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
            metrics = measure(heldout)
            checkpoint = out / f"epoch-{epoch + 1:03d}.safetensors"
            save_model(model, spec, checkpoint)
            save_optimizer_state(optimizer, checkpoint.with_suffix(".optimizer.npz"))
            export_actor(model, spec, checkpoint.with_suffix(".actor.npz"), compressed=False)
            persist(
                epoch=epoch + 1,
                model=str(checkpoint),
                optimizer=str(checkpoint.with_suffix(".optimizer.npz")),
                heldout=metrics,
                history=[*state.get("history", []), metrics],
            )
        if halt():
            persist(phase="paused")
            return
        if (
            state["heldout"]["ranking_loss"] >= state["initial_heldout"]["ranking_loss"]
            or state["heldout"]["anchor_kl"] > args.max_kl
        ):
            persist(phase="no_heldout_improvement")
            atomic_json(
                out / "result.json",
                dict(
                    status="no_heldout_improvement",
                    state=state,
                    next_step="Do not deploy; inspect action-label signal and heldout drift before increasing the budget.",
                ),
            )
            return
        # Only the final predeclared epoch is evaluated. No best-of-many arena
        # checkpoint selection, and no automatic writes to champion succession.
        actor_path = str(Path(state["model"]).with_suffix(".actor.npz"))
        record_path = out / "pairs.jsonl"
        pairs = (
            [json.loads(line) for line in record_path.read_text().splitlines()]
            if record_path.exists()
            else []
        )
        if [r["pair"] for r in pairs] != list(range(len(pairs))):
            raise ValueError("evaluation must be a contiguous pair prefix")
        persist(phase="evaluating_greedy_candidate")
        with (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool,
            record_path.open("a", buffering=1) as log,
        ):
            while len(pairs) < args.eval_pairs and not halt():
                results = list(
                    pool.map(
                        pair,
                        [
                            (
                                actor_path,
                                source,
                                {"rollouts": 0, "native": True, "rules_version": 1},
                                args.seed + 1000000000,
                                i,
                            )
                            for i in range(len(pairs), min(args.eval_pairs, len(pairs) + 128))
                        ],
                        chunksize=2,
                    )
                )
                for row in results:
                    log.write(json.dumps(row) + "\n")
                    pairs.append(row)
                persist(pairs=len(pairs), evaluation=summary(pairs))
        if len(pairs) < args.eval_pairs:
            persist(phase="paused")
            return
        report = summary(pairs)
        persist(phase="complete", evaluation=report)
        atomic_json(
            out / "result.json",
            dict(
                status="complete",
                evaluation=report,
                candidate=actor_path,
                evidence="exploratory independent fixed-pair evaluation; not an automatic promotion",
                next_step=(
                    "Submit the retained candidate to a fresh campaign promotion gate, preserving its existing attempt budget."
                    if report["hoeffding95"][0] > 0.5
                    else "Improvement is not established. Do not resume another long unchanged run; inspect heldout comparisons and the paired evaluation."
                ),
            ),
        )
    except BaseException as error:
        persist(phase="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
