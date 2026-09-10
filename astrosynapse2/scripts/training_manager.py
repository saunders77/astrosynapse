"""Bounded experiment supervisor with immutable code and held-out certification.

Does not write the live champion registry. STOP finishes a training iteration;
--resume continues the same frozen runtime and experiment identities.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from astro2.experiment_control import atomic_json, certify, code_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--max-games", type=int, default=400_000)
    parser.add_argument("--games-per-iteration", type=int, default=512)
    parser.add_argument("--iterations-per-round", type=int, default=32)
    parser.add_argument("--screen-pairs", type=int, default=512)
    parser.add_argument("--certification-pairs", type=int, default=4096)
    parser.add_argument("--rules-version", type=int, choices=[1, 2], default=1)
    parser.add_argument("--seed", type=int, default=2026091001)
    args = parser.parse_args()
    if min(args.workers, args.max_hours, args.max_games, args.iterations_per_round) <= 0:
        raise ValueError("resource budgets must be positive")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = out / "runtime"
    manifest_path = out / "manifest.json"
    frozen_manager = runtime / "scripts/training_manager.py"
    if args.resume and frozen_manager.exists() and Path(__file__).resolve() != frozen_manager:
        # Resume the decision logic too, not only the learner subprocesses.
        os.execve(
            sys.executable,
            [sys.executable, str(frozen_manager), *sys.argv[1:]],
            {**os.environ, "PYTHONPATH": str(runtime)},
        )
    manager_lock = (out / "manager.lock").open("a+")
    try:
        fcntl.flock(manager_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("this experiment already has a running supervisor") from exc
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        saved = manifest["settings"]
        for key in vars(args):
            if key not in {"resume", "output", "max_hours", "max_games", "workers"}:
                setattr(args, key, saved[key])
        state = json.loads((out / "state.json").read_text())
        if state["status"] == "target_certified":
            print(json.dumps(state["winner"], indent=2))
            return
        state["status"] = "running"
        (out / "STOP").unlink(missing_ok=True)
        for trial_folder in out.iterdir():
            if trial_folder.is_dir() and (trial_folder / "state.json").exists():
                (trial_folder / "STOP").unlink(missing_ok=True)
        if code_identity(runtime / "astro2", runtime / "scripts") != manifest["code_identity"]:
            raise ValueError("the immutable experiment runtime has changed")
    else:
        if manifest_path.exists() or runtime.exists():
            raise ValueError("use --resume or a new output directory")
        import astro2

        runtime.mkdir()
        shutil.copytree(
            Path(astro2.__file__).parent,
            runtime / "astro2",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(
            Path(__file__).parent,
            runtime / "scripts",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        manifest = {
            "settings": vars(args),
            "created_at": time.time(),
            "code_identity": code_identity(runtime / "astro2", runtime / "scripts"),
            "source": "5583687405f845fc",
            "opponents": ["08aa018c672847d9", "5583687405f845fc"],
            "certification_trigger": 0.72,
            "trials": [
                {
                    "name": "strategic_heads",
                    "scope": "heads",
                    "strategic": True,
                    "lr": 2e-5,
                    "max_lr": 2e-4,
                },
                {
                    "name": "strategic_full",
                    "scope": "full",
                    "strategic": True,
                    "lr": 2e-6,
                    "max_lr": 2e-5,
                },
                {
                    "name": "full_policy",
                    "scope": "full",
                    "strategic": False,
                    "lr": 1e-6,
                    "max_lr": 1e-5,
                },
            ],
        }
        atomic_json(manifest_path, manifest)
        state = {"elapsed": 0, "attempts": 0, "active": None, "trials": {}, "status": "running"}
        atomic_json(out / "state.json", state)
    started = time.monotonic()
    previous_elapsed = state["elapsed"]
    env = {
        **os.environ,
        "PYTHONPATH": str(runtime),
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def elapsed():
        return previous_elapsed + time.monotonic() - started

    def persist():
        state["elapsed"] = elapsed()
        atomic_json(out / "state.json", state)

    def run(command, log_path, training_dir=None):
        with log_path.open("a", buffering=1) as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if training_dir is not None and (
                        stopped or (out / "STOP").exists() or elapsed() >= args.max_hours * 3600
                    ):
                        (training_dir / "STOP").touch()
                    # Fixed evaluation jobs finish their predeclared sample.
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f"child exited {child.returncode}; see {log_path}")
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()

    # Round-robin gives each architectural intervention a real learning budget.
    # KL adaptation occurs inside each fresh rollout, with every change logged.
    while not stopped and not (out / "STOP").exists():
        if elapsed() >= args.max_hours * 3600:
            state["status"] = "time_budget_exhausted"
            break
        if shutil.disk_usage(out).free < 16 * 1024**3:
            state["status"] = "disk_budget_exhausted"
            break
        available = [
            t for t in manifest["trials"] if not state["trials"].get(t["name"], {}).get("retired")
        ]
        if not available:
            state["status"] = "no_promising_trials"
            break
        trial = next((t for t in available if t["name"] == state["active"]), None)
        if trial is None:
            trial = min(
                available, key=lambda t: len(state["trials"].get(t["name"], {}).get("screens", []))
            )
        record = state["trials"].setdefault(trial["name"], {"screens": []})
        folder = out / trial["name"]
        folder.mkdir(exist_ok=True)
        state["active"] = trial["name"]
        round_number = len(record["screens"]) + 1
        iterations = round_number * args.iterations_per_round
        for name, trial_state in state["trials"].items():
            saved_state = out / name / "state.json"
            if saved_state.exists():
                trial_state["games"] = json.loads(saved_state.read_text())["games"]
        completed_games = sum(r.get("games", 0) for r in state["trials"].values())
        remaining_round_games = iterations * args.games_per_iteration - record.get("games", 0)
        if completed_games + remaining_round_games > args.max_games:
            state["status"] = "game_budget_exhausted"
            break
        persist()
        command = [
            sys.executable,
            str(runtime / "scripts/onpolicy_experiment.py"),
            "--model",
            manifest["source"],
            "--opponent",
            manifest["opponents"][0],
            "--iterations",
            str(iterations),
            "--games",
            str(args.games_per_iteration),
            "--workers",
            str(args.workers),
            "--batch-size",
            "512",
            "--epochs",
            "2",
            "--temperature",
            "0.03",
            "--learning-rate",
            str(trial["lr"]),
            "--max-learning-rate",
            str(trial["max_lr"]),
            "--target-kl",
            "0.015",
            "--adaptive-kl",
            "--train-scope",
            trial["scope"],
            "--rules-version",
            str(args.rules_version),
            "--eval-every",
            "1000000",
            "--eval-pairs",
            str(args.screen_pairs),
            "--seed",
            str(args.seed),
            "--output",
            str(folder),
        ]
        if trial["strategic"]:
            command.append("--strategic-only")
        if (folder / "manifest.json").exists():
            command.append("--resume")
        run(command, out / (trial["name"] + ".log"), folder)
        rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        latest = rows[-1]
        record["games"] = latest["games"]
        if latest["iteration"] + 1 < iterations:
            state["status"] = "stopped_at_checkpoint"
            break
        screen = {
            "round": round_number,
            "games": latest["games"],
            "checkpoint": latest["checkpoint"],
            "score": latest["evaluation"]["score"],
            "seed": latest["evaluation_seed"],
            "post_update_kl": latest["post_update_kl"],
            "learning_rate": latest["learning_rate"],
        }
        if any(
            row["truncated"] / args.games_per_iteration > 0.005
            for row in rows[-args.iterations_per_round :]
        ):
            record["retired"] = "excessive_truncations"
        # Four full rounds without approaching even 55% is a failed branch,
        # not a reason to spend a 100,000-pair promotion extension on it.
        if (
            round_number >= 4
            and max([s["score"] for s in record["screens"]] + [screen["score"]]) < 0.55
        ):
            record["retired"] = "no_substantial_progress_after_four_rounds"
        if screen["score"] >= manifest["certification_trigger"]:
            state["attempts"] += 1
            attempt = state["attempts"]
            persist()  # Commit the error-budget allocation before any test.
            completion = json.loads((folder / "complete.json").read_text())
            options = {
                "rollouts": 0,
                "native": True,
                "rules_version": args.rules_version,
                "auto_actor_path": completion["auto_actor_path"],
            }
            matches = {}
            for index, opponent in enumerate(manifest["opponents"]):
                destination = out / f"certificate-{attempt:03d}-{index}"
                run(
                    [
                        sys.executable,
                        str(runtime / "scripts/planning_experiment.py"),
                        "--model",
                        completion["checkpoint"],
                        "--opponent",
                        opponent,
                        "--pairs",
                        str(args.certification_pairs),
                        "--workers",
                        str(args.workers),
                        "--seed",
                        str(args.seed + 10_000_000 + attempt * 100 + index),
                        "--config",
                        json.dumps(options),
                        "--output",
                        str(destination),
                    ],
                    out / f"certificate-{attempt:03d}-{index}.log",
                )
                matches[opponent] = [
                    json.loads(line)
                    for line in (destination / "pairs.jsonl").read_text().splitlines()
                ]
            certificate = certify(matches, pairs=args.certification_pairs, attempt=attempt)
            atomic_json(out / f"certificate-{attempt:03d}.json", certificate)
            screen["certificate"] = certificate
            if certificate["passed"]:
                state["status"] = "target_certified"
                state["winner"] = {**completion, "options": options, "certificate": certificate}
                record["screens"].append(screen)
                persist()
                break
        record["screens"].append(screen)
        state["active"] = None
        print(json.dumps({"trial": trial["name"], **screen}), flush=True)
        persist()
    if state["status"] == "running":
        state["status"] = "stopped"
    persist()
    print(
        json.dumps(
            {
                "status": state["status"],
                "elapsed": state["elapsed"],
                "goal_met": state["status"] == "target_certified",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
