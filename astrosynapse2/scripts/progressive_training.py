"""Successive independently verified champions, with a durable local UI feed."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from astro2.experiment_control import atomic_json, code_identity, sha256
from astro2.sequential import promotion_evidence
from planning_experiment import pair, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/progressive/20260910")
    p.add_argument(
        "--model",
        default="data/assessment_20260909/managed_campaign/full_policy/g00049152.safetensors",
    )
    p.add_argument("--champion", default="08aa018c672847d9")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--hours", type=float, default=48)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--round-iterations", type=int, default=16)
    p.add_argument("--screen-pairs", type=int, default=512)
    p.add_argument("--max-gate-pairs", type=int, default=32768)
    p.add_argument("--seed", type=int, default=2026091017)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if (
        min(
            args.workers,
            args.hours,
            args.games,
            args.round_iterations,
            args.screen_pairs,
            args.max_gate_pairs,
        )
        <= 0
    ):
        raise ValueError("budgets must be positive")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = out / "runtime"
    frozen = runtime / "scripts/progressive_training.py"
    if args.resume and frozen.exists() and Path(__file__).resolve() != frozen:
        os.execve(
            sys.executable,
            [sys.executable, str(frozen), *sys.argv[1:]],
            {**os.environ, "PYTHONPATH": str(runtime)},
        )
    lock = (out / "manager.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        manifest = json.loads((out / "manifest.json").read_text())
        for key, value in manifest["settings"].items():
            if key not in {"resume", "hours", "workers", "output"}:
                setattr(args, key, value)
        if code_identity(runtime / "astro2", runtime / "scripts") != manifest["code_identity"]:
            raise ValueError("frozen runtime changed")
        state = json.loads((out / "state.json").read_text())
        (out / "STOP").unlink(missing_ok=True)
        for path in out.glob("stage-*/STOP"):
            path.unlink()
    else:
        if (out / "manifest.json").exists():
            raise ValueError("use --resume or a new output directory")
        import astro2

        runtime.mkdir()
        for source, destination in [
            (Path(astro2.__file__).parent, runtime / "astro2"),
            (Path(__file__).parent, runtime / "scripts"),
        ]:
            shutil.copytree(
                source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
        db = sqlite3.connect("file:data/astrosynapse2.sqlite3?mode=ro", uri=True)
        row = db.execute(
            "select actor_path from checkpoints where id=?", (args.champion,)
        ).fetchone()
        champion = row[0] if row else str(Path(args.champion).resolve())
        anchor = out / "original-champion.actor.npz"
        shutil.copy2(champion, anchor)
        model = Path(args.model).resolve()
        source = out / "source.safetensors"
        shutil.copy2(model, source)
        shutil.copy2(
            model.with_suffix(model.suffix + ".json"), source.with_suffix(source.suffix + ".json")
        )
        optimizer = model.with_suffix(".optimizer.npz")
        if optimizer.exists():
            shutil.copy2(optimizer, source.with_suffix(".optimizer.npz"))
        manifest = dict(
            settings=vars(args),
            created_at=time.time(),
            source_sha256=sha256(source),
            original_champion_sha256=sha256(anchor),
            code_identity=code_identity(runtime / "astro2", runtime / "scripts"),
            promotion_contract="time-uniform paired-mean lower bound above 50%; 5% budget across attempts against each incumbent",
        )
        atomic_json(out / "manifest.json", manifest)
        state = dict(
            name="Astro6 · Progressive champion training",
            stage=0,
            round=0,
            attempt=0,
            model=str(source),
            champion=str(anchor),
            champion_label=args.champion,
            phase="starting",
            elapsed=0.0,
            history=[],
            promotions=[],
            gates=[],
            pending_gate=None,
            games_before_stage=0,
            games=0,
        )
    atomic_json(out.parent / "current.json", {"path": str(out)})
    started, elapsed_before = time.monotonic(), state["elapsed"]
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def persist():
        state.update(
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

    def matches(candidate, opponent, folder, seed, *, attempt=None, fixed_pairs=None):
        folder.mkdir(exist_ok=True)
        identity = dict(
            candidate_sha256=sha256(candidate),
            opponent_sha256=sha256(opponent),
            seed=seed,
            attempt=attempt,
            fixed_pairs=fixed_pairs,
            maximum_pairs=args.max_gate_pairs,
            rules_version=1,
        )
        path = folder / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("evaluation identity changed")
        atomic_json(path, identity)
        records = folder / "pairs.jsonl"
        rows = [json.loads(x) for x in records.read_text().splitlines()] if records.exists() else []
        if [r["pair"] for r in rows] != list(range(len(rows))):
            raise ValueError("evaluation must resume a contiguous prefix")
        budget = fixed_pairs or args.max_gate_pairs
        with (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool,
            records.open("a", buffering=1) as log,
        ):
            while len(rows) < budget and not halt():
                tasks = [
                    (
                        candidate,
                        opponent,
                        {"rollouts": 0, "native": True, "rules_version": 1},
                        seed,
                        i,
                    )
                    for i in range(len(rows), min(budget, len(rows) + 256))
                ]
                for result in pool.map(pair, tasks, chunksize=2):
                    rows.append(result)
                    log.write(json.dumps(result) + "\n")
                report = promotion_evidence(rows, attempt) if attempt else summary(rows)
                state["evaluation"] = report
                atomic_json(folder / "result.json", report)
                persist()
                if attempt and (
                    report["passed"]
                    or (len(rows) >= 2048 and report["score"] <= 0.50)
                    or (len(rows) >= 8192 and report["score"] < 0.51)
                ):
                    break
        report = promotion_evidence(rows, attempt) if attempt else (summary(rows) if rows else {})
        report["paused"] = halt() and len(rows) < budget and not report.get("passed")
        atomic_json(folder / "result.json", report)
        return report

    try:
        while not halt():
            if shutil.disk_usage(out).free < 12 * 1024**3:
                state["phase"] = "disk_budget_exhausted"
                break
            folder = out / f"stage-{state['stage']:03d}"
            folder.mkdir(exist_ok=True)
            if not state["pending_gate"]:
                state.update(phase="training", training_dir=str(folder), evaluation=None)
                persist()
                target = (state["round"] + 1) * args.round_iterations
                command = [
                    sys.executable,
                    str(runtime / "scripts/onpolicy_experiment.py"),
                    "--model",
                    state["model"],
                    "--opponent",
                    state["champion"],
                    "--iterations",
                    str(target),
                    "--games",
                    str(args.games),
                    "--workers",
                    str(args.workers),
                    "--batch-size",
                    "512",
                    "--epochs",
                    "2",
                    "--temperature",
                    "0.03",
                    "--learning-rate",
                    "0.0000003",
                    "--max-learning-rate",
                    "0.00001",
                    "--target-kl",
                    "0.015",
                    "--adaptive-kl",
                    "--eval-every",
                    "1000000",
                    "--eval-pairs",
                    str(args.screen_pairs),
                    "--seed",
                    str(args.seed + state["stage"] * 100003),
                    "--output",
                    str(folder),
                ]
                if (folder / "state.json").exists():
                    command.append("--resume")
                elif Path(state["model"]).with_suffix(".optimizer.npz").exists():
                    command += [
                        "--initial-optimizer",
                        str(Path(state["model"]).with_suffix(".optimizer.npz")),
                    ]
                with (folder / "training.log").open("a", buffering=1) as log:
                    child = subprocess.Popen(
                        command,
                        env={**os.environ, "PYTHONPATH": str(runtime)},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        while child.poll() is None:
                            if halt():
                                (folder / "STOP").touch()
                            persist()
                            time.sleep(3)
                        if child.returncode:
                            raise RuntimeError(
                                f"learner exited {child.returncode}; see {folder / 'training.log'}"
                            )
                    finally:
                        if child.poll() is None:
                            child.terminate()
                            child.wait()
                rows = [
                    json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
                ]
                latest = rows[-1]
                state["games"] = state["games_before_stage"] + latest["games"]
                if latest["iteration"] + 1 < target:
                    break
                candidate_model = latest["checkpoint"]
                candidate = str(Path(candidate_model).with_suffix(".actor.npz"))
                state["round"] += 1
                screen = dict(
                    stage=state["stage"],
                    round=state["round"],
                    games=state["games"],
                    checkpoint=candidate_model,
                    score=latest["evaluation"]["score"],
                    opponent=state["champion_label"],
                )
                state["history"].append(screen)
                # Only unscreened intermediate artifacts are disposable. Keep
                # every evaluated checkpoint, its optimizer, and the resume tip.
                keep = {Path(h["checkpoint"]).stem for h in state["history"]} | {
                    Path(candidate_model).stem
                }
                for artifact in folder.glob("g????????.*"):
                    if artifact.name.split(".")[0] not in keep:
                        artifact.unlink()
                if screen["score"] <= 0.50:
                    persist()
                    continue
                state["attempt"] += 1
                state["pending_gate"] = dict(
                    model=candidate_model,
                    actor=candidate,
                    stage=state["stage"],
                    attempt=state["attempt"],
                    round=state["round"],
                    seed=args.seed + 1000000000 + state["stage"] * 100000 + state["attempt"],
                )
                persist()
            gate = state["pending_gate"]
            state.update(phase="verifying_candidate", evaluation=None)
            persist()
            evidence = matches(
                gate["actor"],
                state["champion"],
                out / f"gate-{gate['stage']:03d}-{gate['attempt']:03d}",
                gate["seed"],
                attempt=gate["attempt"],
            )
            if evidence["paused"]:
                break
            record = {
                **gate,
                **evidence,
                "opponent": state["champion_label"],
                "games": state["games"],
            }
            state["gates"].append(record)
            if evidence["passed"]:
                state["promotions"].append(record)
                state.update(
                    model=gate["model"],
                    champion=gate["actor"],
                    champion_label=f"Astro6 champion {state['stage'] + 1}",
                    stage=state["stage"] + 1,
                    round=0,
                    attempt=0,
                    games_before_stage=state["games"],
                )
            state["pending_gate"] = None
            persist()
            if evidence["passed"]:
                state["phase"] = "benchmarking_original_champion"
                persist()
                benchmark = matches(
                    state["champion"],
                    str(out / "original-champion.actor.npz"),
                    out / f"anchor-{state['stage']:03d}",
                    args.seed + 2000000000 + state["stage"],
                    fixed_pairs=2048,
                )
                state["promotions"][-1]["original_champion_benchmark"] = benchmark
                persist()
        if state["phase"] != "disk_budget_exhausted":
            state["phase"] = (
                "paused" if stopped or (out / "STOP").exists() else "time_budget_complete"
            )
    except Exception as exc:
        state.update(phase="failed", error=str(exc))
        raise
    finally:
        persist()


if __name__ == "__main__":
    main()
