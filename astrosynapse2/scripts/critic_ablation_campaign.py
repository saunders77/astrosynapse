"""Run a frozen, resumable six-run arch3 critic ablation and fixed evaluations."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from astro2.experiment_control import atomic_json, code_identity, sha256

BRANCHES = ("frozen", "online", "constant")
SEEDS = (2026100311, 2026100322)


def prepare(folder, project):
    from astro2.progressive_models import progressive_models

    folder.mkdir(parents=True, exist_ok=False)
    inputs = folder / "inputs"
    inputs.mkdir()
    runtime = folder / "runtime"
    shutil.copytree(
        project / "backend/astro2", runtime / "astro2", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(
        project / "scripts", runtime / "scripts", ignore=shutil.ignore_patterns("__pycache__")
    )
    source = json.loads((project / "data/arch3/generation10.json").read_text())
    for original, name in [
        (source["path"], "initial.safetensors"),
        (source["path"] + ".json", "initial.safetensors.json"),
        (source["actor_path"], "initial.actor.npz"),
    ]:
        shutil.copyfile(original, inputs / name)
    critic_ref = json.loads((project / "data/arch3/generation10-critic-run.json").read_text())
    critic_folder = project / "data/critics" / critic_ref["id"]
    assert json.loads((critic_folder / "job.json").read_text())["status"] == "complete"
    shutil.copyfile(critic_folder / "critic.npz", inputs / "critic.npz")
    models = progressive_models(project / "data")
    inventory = {}
    for generation in (2, 4, 5, 6, 7, 8, 9, 10):
        model = next(m for m in models if m.get("generation") == generation)
        target = inputs / f"generation{generation}.actor.npz"
        shutil.copyfile(model["actor_path"], target)
        inventory[str(generation)] = dict(
            model_id=model["id"], path=str(target), sha256=sha256(target)
        )
    atomic_json(inputs / "opponents.json", [inventory[str(g)]["path"] for g in (2, 4, 6, 8)])
    runs = []
    for index, seed in enumerate(SEEDS):
        for branch in BRANCHES if index == 0 else BRANCHES[::-1]:
            runs.append(
                dict(
                    id=f"{branch}-seed{index + 1}",
                    branch=branch,
                    seed=seed,
                    folder=str(folder / f"{branch}-seed{index + 1}"),
                )
            )
    manifest = dict(
        project=str(project),
        created_at=time.time(),
        runs=runs,
        training_games_per_run=20000,
        total_training_games=120000,
        games_per_iteration=1000,
        iterations=20,
        workers=6,
        policy_epochs=2,
        policy_learning_rate=2e-6,
        temperature=0.1,
        critic_epochs=3,
        critic_learning_rate=0.0003,
        rules_version=2,
        train_opponents=[2, 4, 6, 8, 10],
        evaluation_panel=[10, 5, 7, 9],
        screen_pairs=512,
        screen_every_games=5000,
        final_pairs_per_opponent=2000,
        planned_evaluation_games=136576,
        planned_total_games=256576,
        final_eval_seed=202610039000,
        inventory=inventory,
        critic_source=critic_ref["id"],
        initial_policy=source["id"],
        input_hashes={p.name: sha256(p) for p in inputs.iterdir() if p.is_file()},
        code_identity=code_identity(runtime / "astro2", runtime / "scripts"),
        protocol="Fixed 20k endpoints, no checkpoint selection or automatic promotion. "
        "Identical rollout/opponent seed streams by iteration within training seed. "
        "Greedy mean-head evaluations with paired seeds and seat swaps. "
        "Fresh final evaluation seeds shared across candidates and initial-policy control. "
        "Final panel scores are equally weighted across opponents; only two independent "
        "training seeds. Raw pairs retained for paired uncertainty and multiple-comparison "
        "correction in later analysis. No repeated significance stopping.",
    )
    atomic_json(folder / "manifest.json", manifest)
    atomic_json(folder / "status.json", dict(status="queued", created_at=time.time()))
    return manifest


def read(path):
    return json.loads(path.read_text())


def campaign(folder):
    manifest = read(folder / "manifest.json")
    runtime = folder / "runtime"
    if code_identity(runtime / "astro2", runtime / "scripts") != manifest["code_identity"]:
        raise ValueError("Frozen runtime changed")
    for name, expected in manifest["input_hashes"].items():
        if sha256(folder / "inputs" / name) != expected:
            raise ValueError(f"Input changed: {name}")
    project = Path(manifest["project"])
    env = {
        **os.environ,
        "PYTHONPATH": str(runtime),
        "VECLIB_MAXIMUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    started = time.time()

    def update(**fields):
        atomic_json(
            folder / "status.json",
            dict(pid=os.getpid(), updated_at=time.time(), started_at=started, **fields),
        )

    def launch(command, log_path, phase, task):
        if (folder / "PAUSE").exists():
            update(status="paused", phase=phase, task=task)
            raise InterruptedError("Campaign pause requested")
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=project,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            update(status="running", phase=phase, task=task, child_pid=process.pid, command=command)
            if process.wait() != 0:
                raise RuntimeError(f"{task} failed; inspect {log_path}")

    try:
        for run in manifest["runs"]:
            out = Path(run["folder"])
            complete = out / "complete.json"
            if complete.exists() and read(complete).get("games") == 20000:
                continue
            out.mkdir(exist_ok=True)
            command = [
                sys.executable,
                str(runtime / "scripts/onpolicy_experiment.py"),
                "--model",
                str(folder / "inputs/initial.safetensors"),
                "--opponent",
                manifest["inventory"]["10"]["path"],
                "--opponent-pool",
                str(folder / "inputs/opponents.json"),
                "--output",
                str(out),
                "--seed",
                str(run["seed"]),
                "--iterations",
                "20",
                "--games",
                "1000",
                "--workers",
                "6",
                "--epochs",
                "2",
                "--learning-rate",
                "0.000002",
                "--temperature",
                "0.1",
                "--rules-version",
                "2",
                "--eval-every",
                "5",
                "--eval-pairs",
                "512",
                "--matched-rollout-seeds",
                "--skip-embedded-critic-training",
            ]
            if run["branch"] == "constant":
                command += ["--advantage-baseline", "constant"]
            else:
                command += ["--independent-critic", str(folder / "inputs/critic.npz")]
                if run["branch"] == "online":
                    command += [
                        "--update-independent-critic",
                        "--independent-critic-epochs",
                        "3",
                        "--independent-critic-learning-rate",
                        "0.0003",
                    ]
            if (out / "state.json").exists():
                (out / "STOP").unlink(missing_ok=True)
                complete.unlink(missing_ok=True)
                command += ["--resume"]
            launch(command, out / "worker.log", "training", run["id"])
            if read(complete)["games"] != 20000:
                raise InterruptedError("Training stopped before budget")
        final = folder / "evaluation"
        final.mkdir(exist_ok=True)
        candidates = [("initial-control", str(folder / "inputs/initial.actor.npz"))] + [
            (r["id"], str(Path(r["folder"]) / "g00020000.actor.npz")) for r in manifest["runs"]
        ]
        for name, actor in candidates:
            for index, opponent in enumerate(manifest["evaluation_panel"]):
                out = final / f"{name}-vs-generation{opponent}"
                result = out / "result.json"
                # planning_experiment writes summary.json, depending on runtime version.
                summaries = [out / "summary.json", result]
                if any(p.exists() and read(p).get("pairs") == 2000 for p in summaries):
                    continue
                out.mkdir(exist_ok=True)
                command = [
                    sys.executable,
                    str(runtime / "scripts/planning_experiment.py"),
                    "--model",
                    actor,
                    "--opponent",
                    manifest["inventory"][str(opponent)]["path"],
                    "--pairs",
                    "2000",
                    "--workers",
                    "6",
                    "--seed",
                    str(manifest["final_eval_seed"] + index),
                    "--config",
                    json.dumps(dict(rollouts=0, native=True, rules_version=2)),
                    "--output",
                    str(out),
                ]
                launch(command, out / "worker.log", "final_evaluation", out.name)
        # Record descriptive results only. Decisions and significance analysis wait for review.
        results = {}
        for name, _ in candidates:
            results[name] = {}
            for opponent in manifest["evaluation_panel"]:
                out = final / f"{name}-vs-generation{opponent}"
                summary = next(
                    (p for p in [out / "summary.json", out / "result.json"] if p.exists()), None
                )
                if summary is None:
                    raise RuntimeError(f"Missing evaluation: {out}")
                results[name][str(opponent)] = read(summary)
        timings = {}
        for run in manifest["runs"]:
            rows = [
                json.loads(line)
                for line in (Path(run["folder"]) / "metrics.jsonl").read_text().splitlines()
            ]
            timings[run["id"]] = {
                key: sum(row.get(key, 0) for row in rows)
                for key in (
                    "rollout_seconds",
                    "learning_seconds",
                    "critic_seconds",
                    "evaluation_seconds",
                    "checkpoint_seconds",
                )
            }
        atomic_json(
            folder / "results.json",
            dict(evaluation=results, timings=timings, protocol=manifest["protocol"]),
        )
        update(status="complete", elapsed_seconds=time.time() - started)
    except InterruptedError as error:
        update(status="paused", message=str(error))
    except Exception as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    folder = args.output.resolve()
    if args.prepare:
        prepare(folder, Path.cwd())
        print(folder)
        return
    with (folder / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign(folder)


if __name__ == "__main__":
    main()
