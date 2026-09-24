"""Bounded training portfolio with durable, independently verified promotions."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from astro2.autonomy import RECIPES, commit_gate, gate_finished, prepare_history, read_json
from astro2.experiment_control import atomic_json, code_identity, sha256
from astro2.sequential import promotion_evidence
from planning_experiment import pair, summary


def initialize(out, inherited, args):
    old = read_json(inherited / "state.json")
    if old is None or old["phase"] not in {"paused", "time_budget_complete", "failed"}:
        raise ValueError("the inherited campaign must be paused")
    with (inherited / "manager.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "manifest.json").exists():
            raise ValueError("use --resume for an existing campaign")
        runtime = out / "runtime"
        # Preserve the exact historical engine; update only the learner/supervision.
        shutil.copytree(
            inherited / "runtime", runtime, ignore=shutil.ignore_patterns("__pycache__")
        )
        project = Path.cwd()
        for name in ("autonomy.py", "onpolicy.py"):
            shutil.copy2(project / "backend/astro2" / name, runtime / "astro2" / name)
        for name in ("autonomous_training.py", "onpolicy_experiment.py"):
            shutil.copy2(project / "scripts" / name, runtime / "scripts" / name)
        source = out / "source.safetensors"
        shutil.copy2(old["model"], source)
        shutil.copy2(old["model"] + ".json", str(source) + ".json")
        champion = out / "source.actor.npz"
        shutil.copy2(old["champion"], champion)
        shutil.copy2(inherited / "original-champion.actor.npz", out / "original-champion.actor.npz")
        settings = dict(
            hours=args.hours,
            workers=args.workers,
            games=512,
            round_iterations=8,
            screen_pairs=512,
            confirm_pairs=2048,
            max_gate_pairs=131072,
            max_rounds=4,
            seed=secrets.randbits(48),
        )
        manifest = dict(
            mode="autonomous",
            version=1,
            inherited_from=str(inherited),
            created_at=time.time(),
            settings=settings,
            inherited_attempt=old["attempt"],
            inherited_champion_sha256=sha256(champion),
            inherited_runtime_identity=read_json(inherited / "manifest.json")["code_identity"],
            code_identity=code_identity(runtime / "astro2", runtime / "scripts"),
            promotion_contract="Fresh paired games; time-uniform lower bound above 50%; 5% error budget across attempts per incumbent, including inherited attempts",
        )
        state = dict(
            name="Astro6 · Autonomous improvement",
            mode="autonomous",
            phase="starting",
            stage=old["stage"],
            attempt=old["attempt"],
            champion=str(champion),
            champion_label=old["champion_label"],
            model=str(source),
            elapsed=0,
            games=old["games"],
            games_before_stage=old["games"],
            inherited_games=old["games"],
            inherited_promotions=len(old["promotions"]),
            history=old["history"],
            gates=old["gates"],
            promotions=old["promotions"],
            pending_gate=None,
            branches=[],
            active_branch=None,
            branch_counter=0,
            events=[],
            errors=0,
        )
        if old.get("pending_gate"):
            state["gates"].append(
                {
                    **old["pending_gate"],
                    **(old.get("evaluation") or {}),
                    "passed": False,
                    "abandoned": True,
                    "games": old["games"],
                    "opponent": old["champion_label"],
                    "reason": "Campaign replaced; attempt remains spent",
                    "evidence_path": str(inherited),
                }
            )
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "state.json", state)


class Campaign:
    def __init__(self, out):
        self.out = out
        self.runtime = out / "runtime"
        self.manifest = read_json(out / "manifest.json")
        if (
            code_identity(self.runtime / "astro2", self.runtime / "scripts")
            != self.manifest["code_identity"]
        ):
            raise ValueError("frozen runtime changed; explicit migration required")
        self.settings = self.manifest["settings"]
        self.state = read_json(out / "state.json")
        self.before = self.state["elapsed"]
        self.started = time.monotonic()
        self.stopped = False
        self.child = None
        # A killed manager may leave its learner alive. Reclaim only our recorded child.
        pid = self.state.pop("child_pid", None)
        if pid:
            import psutil

            try:
                process = psutil.Process(pid)
                if any(str(out / "branches") in arg for arg in process.cmdline()):
                    process.terminate()
                    process.wait(timeout=180)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        (out / "STOP").unlink(missing_ok=True)
        for path in out.glob("branches/*/STOP"):
            path.unlink()
        self.state.pop("error", None)
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

    def stop(self, *_):
        self.stopped = True

    def persist(self):
        self.state.update(
            elapsed=self.before + time.monotonic() - self.started,
            heartbeat=time.time(),
            pid=os.getpid(),
        )
        atomic_json(self.out / "state.json", self.state)

    def event(self, message):
        self.state["events"].append(dict(time=time.time(), message=message))
        self.state["events"] = self.state["events"][-100:]
        print(message, flush=True)
        self.persist()

    def controls(self):
        updates = read_json(self.out / "settings.json", {})
        for key in ("hours", "workers", "max_rounds", "max_stalled_generations"):
            if key in updates:
                self.settings[key] = updates[key]

    def halt(self):
        self.controls()
        return (
            self.stopped
            or (self.out / "STOP").exists()
            or self.before + time.monotonic() - self.started >= self.settings["hours"] * 3600
        )

    def skip(self, branch):
        request = read_json(self.out / "skip.json", {})
        return request.get("branch") == branch["id"]

    def gate_complete(self, report, maximum):
        return gate_finished(report, maximum)

    def matches(self, actor, opponent, folder, seed, *, attempt=None, pairs=None):
        folder.mkdir(parents=True, exist_ok=True)
        maximum = pairs or self.settings["max_gate_pairs"]
        identity = dict(
            candidate_sha256=sha256(actor),
            opponent_sha256=sha256(opponent),
            seed=seed,
            attempt=attempt,
            maximum_pairs=maximum,
            rules_version=1,
        )
        previous = read_json(folder / "manifest.json")
        if previous is not None and previous != identity:
            raise ValueError("evaluation identity changed")
        atomic_json(folder / "manifest.json", identity)
        path = folder / "pairs.jsonl"
        rows = []
        if path.exists():
            # Recover a torn final append, but never discard a complete recorded result.
            raw = path.read_bytes()
            lines = raw.splitlines(keepends=True)
            good = 0
            for i, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                    good += len(line)
                except ValueError:
                    if i != len(lines) - 1:
                        raise
                    with path.open("r+b") as file:
                        file.truncate(good)
            if rows and raw and not raw.endswith(b"\n") and good == len(raw):
                with path.open("ab") as file:
                    file.write(b"\n")
        if [r["pair"] for r in rows] != list(range(len(rows))):
            raise ValueError("evaluation must contain a contiguous prefix")

        def report():
            return promotion_evidence(rows, attempt) if attempt else (summary(rows) if rows else {})

        def done(r):
            return self.gate_complete(r, maximum) if attempt else r.get("pairs", 0) >= maximum

        result = report()
        self.state["evaluation"] = result
        self.persist()
        if not done(result) and not self.halt():
            with path.open("a", buffering=1) as log:
                while not done(result) and not self.halt():
                    workers = self.settings["workers"]
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=multiprocessing.get_context("spawn"),
                    ) as pool:
                        # Resource changes apply after a complete persisted pair
                        # block, even when a long promotion test is still active.
                        while (
                            not done(result)
                            and not self.halt()
                            and self.settings["workers"] == workers
                        ):
                            tasks = [
                                (
                                    actor,
                                    opponent,
                                    dict(rollouts=0, native=True, rules_version=1),
                                    seed,
                                    i,
                                )
                                for i in range(len(rows), min(maximum, len(rows) + 128))
                            ]
                            for row in pool.map(pair, tasks, chunksize=2):
                                rows.append(row)
                                log.write(json.dumps(row) + "\n")
                            result = report()
                            self.state["evaluation"] = result
                            atomic_json(folder / "result.json", result)
                            self.persist()
        result["paused"] = not done(result)
        atomic_json(folder / "result.json", result)
        return result

    def select_parent(self, recipe, cycle):
        # Keep regular fresh starts; other cycles can extend independently
        # promising checkpoints without declaring them champions.
        if cycle % 3 == 0:
            return self.state["model"], None
        candidates = []
        for branch in self.state["branches"]:
            model = branch.get("confirmation_model")
            confirmation = branch.get("confirmation", {})
            if (
                branch["stage"] != self.state["stage"]
                or branch["recipe"]["name"] != recipe["name"]
                or confirmation.get("pairs", 0) < 2048
                or confirmation.get("score", 0) < 0.505
                or not model
                or not Path(model).exists()
            ):
                continue
            gates = [g for g in self.state["gates"] if g.get("model") == model]
            evidence = gates[-1] if gates else confirmation
            if evidence.get("score", 0) > 0.5:
                candidates.append((evidence["score"], model, branch["id"]))
        if not candidates:
            return self.state["model"], None
        _, model, branch_id = max(candidates)
        return model, branch_id

    def new_branch(self):
        s = self.state
        s["branch_counter"] += 1
        index = s["branch_counter"] - 1
        recipe = dict(RECIPES[index % len(RECIPES)])
        source, parent = self.select_parent(recipe, index // len(RECIPES))
        # A fresh seed and modest LR alternation prevent replaying the same failed trial.
        recipe["learning_rate"] *= (1, 0.5, 1.5)[(index // len(RECIPES)) % 3]
        name = f"b{index + 1:04d}"
        folder = self.out / "branches" / name
        folder.mkdir(parents=True)
        branch = dict(
            id=name,
            recipe=recipe,
            stage=s["stage"],
            round=0,
            games=0,
            games_before=s["games"],
            status="training",
            best_score=0.0,
            source=source,
            parent_branch=parent,
            opponent=s["champion"],
            seed=self.settings["seed"] + index * 100003,
            folder=str(folder),
            screens=[],
            reason=(
                f"Extend promising {parent} with a fresh optimizer; champion remains the opponent"
                if parent
                else "Fresh branch from the verified champion"
            ),
        )
        s["branches"].append(branch)
        s["active_branch"] = name
        s["games_before_stage"] = s["games"]
        s["training_dir"] = str(folder)
        opponents = [p["actor"] for p in s["promotions"][-5:-1] if Path(p["actor"]).exists()]
        atomic_json(folder / "opponents.json", opponents)
        self.event(f"Started {name}: {recipe['name']} from {parent or s['champion_label']}")
        return branch

    def retire(self, branch, reason):
        branch.update(status="retired", reason=reason)
        self.state["active_branch"] = None
        self.state.pop("training_dir", None)
        self.event(f"Retired {branch['id']}: {reason}")

    def continue_or_retire(self, branch):
        if self.skip(branch):
            self.retire(branch, "Skipped from the control center")
        elif branch["round"] >= self.settings["max_rounds"]:
            self.retire(branch, "Bounded training budget reached; try a different curriculum")
        elif branch["round"] >= 2 and max(branch["screens"][-2:]) <= 0.5:
            self.retire(branch, "Two consecutive screens failed to beat the incumbent")
        else:
            branch.update(
                status="training",
                reason="Continue bounded learning; fresh screen at the next block",
            )
            self.persist()

    def train(self, branch):
        folder = Path(branch["folder"])
        target = (branch["round"] + 1) * self.settings["round_iterations"]
        self.state.update(phase="training", training_dir=str(folder), evaluation=None)
        self.persist()
        command = [
            sys.executable,
            str(self.runtime / "scripts/onpolicy_experiment.py"),
            "--model",
            branch["source"],
            "--opponent",
            branch["opponent"],
            "--iterations",
            str(target),
            "--games",
            str(self.settings["games"]),
            "--workers",
            str(self.settings["workers"]),
            "--batch-size",
            "256",
            "--epochs",
            "2",
            "--max-learning-rate",
            "0.00003",
            "--target-kl",
            ".015",
            "--adaptive-kl",
            "--eval-every",
            "1000000",
            "--eval-pairs",
            str(self.settings["screen_pairs"]),
            "--seed",
            str(branch["seed"]),
            "--output",
            str(folder),
            "--rules-version",
            "1",
            "--opponent-pool",
            str(folder / "opponents.json"),
            "--anchor-dataset",
            str(self.out / f"history-{branch['stage']:03d}.npz"),
        ]
        for key, value in branch["recipe"].items():
            if key != "name":
                command.extend(["--" + key.replace("_", "-"), str(value)])
        if (folder / "state.json").exists():
            command.append("--resume")
        with (folder / "training.log").open("a", buffering=1) as log:
            self.child = subprocess.Popen(
                command,
                env={**os.environ, "PYTHONPATH": str(self.runtime)},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            self.state["child_pid"] = self.child.pid
            self.persist()
            try:
                while self.child.poll() is None:
                    if self.halt() or self.skip(branch):
                        (folder / "STOP").touch()
                    self.persist()
                    time.sleep(2)
                if self.child.returncode:
                    raise RuntimeError(
                        f"Learner exited {self.child.returncode}; inspect {folder / 'training.log'}"
                    )
            finally:
                if self.child.poll() is None:
                    self.child.terminate()
                    self.child.wait()
                self.state.pop("child_pid", None)
                self.child = None
        learner = read_json(folder / "state.json", {})
        branch["games"] = learner.get("games", 0)
        self.state["games"] = branch["games_before"] + branch["games"]
        if self.halt():
            self.persist()
            return
        if self.skip(branch):
            self.retire(branch, "Skipped from the control center")
            return
        if learner.get("iteration", -1) + 1 < target:
            raise RuntimeError("learner stopped before the block completed")
        rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        latest = rows[-1]
        branch.update(
            round=branch["round"] + 1,
            model=latest["checkpoint"],
            actor=str(Path(latest["checkpoint"]).with_suffix(".actor.npz")),
        )
        score = latest["evaluation"]["score"]
        branch["screens"].append(score)
        branch["best_score"] = max(branch["best_score"], score)
        self.state["history"].append(
            dict(
                stage=self.state["stage"],
                games=self.state["games"],
                checkpoint=branch["model"],
                score=score,
                opponent=self.state["champion_label"],
                branch=branch["id"],
            )
        )
        keep = {Path(h["checkpoint"]).stem for h in self.state["history"]} | {
            Path(branch["model"]).stem
        }
        for path in folder.glob("g????????.*"):
            if path.name.split(".")[0] not in keep:
                path.unlink()
        self.state["errors"] = 0
        self.event(
            f"{branch['id']} block {branch['round']}: screen {score:.2%} over {self.settings['screen_pairs']} pairs"
        )
        if score >= 0.505:
            branch.update(
                status="confirming",
                reason="Promising screen; checking a fresh sample before spending a promotion attempt",
            )
            self.persist()
        else:
            self.continue_or_retire(branch)

    def confirm(self, branch):
        self.state.update(phase="confirming_candidate", evaluation=None)
        self.persist()
        evidence = self.matches(
            branch["actor"],
            branch["opponent"],
            Path(branch["folder"]) / f"confirm-{branch['round']:02d}",
            branch["seed"] + 10**9 + branch["round"],
            pairs=self.settings["confirm_pairs"],
        )
        if evidence["paused"]:
            return
        branch["confirmation"] = evidence
        branch["confirmation_model"] = branch["model"]
        if evidence["score"] >= 0.51:
            self.state["attempt"] += 1
            self.state["pending_gate"] = dict(
                model=branch["model"],
                actor=branch["actor"],
                stage=self.state["stage"],
                attempt=self.state["attempt"],
                branch=branch["id"],
                seed=self.settings["seed"]
                + 10**12
                + self.state["stage"] * 10**7
                + self.state["attempt"],
            )
            branch.update(
                status="verifying",
                reason="Independent confirmation passed; fresh promotion games now decide acceptance",
            )
            self.event(
                f"{branch['id']} confirmed at {evidence['score']:.2%}; promotion attempt {self.state['attempt']}"
            )
        else:
            self.event(
                f"{branch['id']} confirmation {evidence['score']:.2%}; no promotion attempt spent"
            )
            self.continue_or_retire(branch)

    def gate(self):
        s = self.state
        pending = s["pending_gate"]
        s.update(phase="verifying_candidate")
        self.persist()
        result = self.matches(
            pending["actor"],
            s["champion"],
            self.out / f"gate-{pending['stage']:03d}-{pending['attempt']:04d}",
            pending["seed"],
            attempt=pending["attempt"],
        )
        if result["paused"]:
            return
        record = commit_gate(s, result)
        branch = next(b for b in s["branches"] if b["id"] == pending["branch"])
        # Save promotion before any optional benchmark or cleanup.
        self.persist()
        if record["passed"]:
            branch.update(
                status="promoted", reason="Passed the independent time-uniform promotion gate"
            )
            s["active_branch"] = None
            s.pop("training_dir", None)
            self.event(
                f"Promoted {s['champion_label']}: {record['score']:.2%}, lower bound {record['lower']:.2%}"
            )
        else:
            self.event(
                f"{branch['id']} not proven stronger after {record['pairs']} pairs; gate budget remains spent"
            )
            self.continue_or_retire(branch)

    def run(self):
        self.state["phase"] = "starting"
        self.persist()
        try:
            while not self.halt():
                if shutil.disk_usage(self.out).free < 12 * 1024**3:
                    self.state["phase"] = "disk_budget_exhausted"
                    break
                if self.state["pending_gate"]:
                    self.gate()
                    continue
                # The transition above is idempotent across a crash before branch retirement.
                active = next(
                    (b for b in self.state["branches"] if b["id"] == self.state["active_branch"]),
                    None,
                )
                if active and active["stage"] != self.state["stage"]:
                    active.update(status="promoted", reason="Verified promotion committed")
                    self.state["active_branch"] = None
                    active = None
                if active and active["status"] == "verifying":
                    self.continue_or_retire(active)
                    continue
                if (
                    self.state["promotions"]
                    and len(self.state["promotions"]) > self.state["inherited_promotions"]
                ):
                    promotion = self.state["promotions"][-1]
                    if promotion.get("original_champion_benchmark", {}).get("pairs", 0) < 2048:
                        self.state["phase"] = "benchmarking_original_champion"
                        self.persist()
                        promotion["original_champion_benchmark"] = self.matches(
                            self.state["champion"],
                            str(self.out / "original-champion.actor.npz"),
                            self.out / f"anchor-{self.state['stage']:03d}",
                            self.settings["seed"] + 10**13 + self.state["stage"],
                            pairs=2048,
                        )
                        self.persist()
                        continue
                dataset = self.out / f"history-{self.state['stage']:03d}.npz"
                if not dataset.exists() or not dataset.with_suffix(".json").exists():
                    self.state["phase"] = "preparing_history"
                    self.persist()
                    self.state["historical_data"] = prepare_history(
                        Path.cwd() / "data",
                        self.state["champion"],
                        dataset,
                        seed=self.settings["seed"] + self.state["stage"],
                    )
                    self.event(
                        f"Prepared {self.state['historical_data']['sampled_positions']} historical anchors from {self.state['historical_data']['replay_positions']:,} retained positions"
                    )
                elif not self.state.get("historical_data"):
                    self.state["historical_data"] = read_json(dataset.with_suffix(".json"))
                branch = active or self.new_branch()
                if self.skip(branch) and branch["status"] != "verifying":
                    self.retire(branch, "Skipped from the control center")
                    continue
                try:
                    if branch["status"] == "training":
                        self.train(branch)
                    elif branch["status"] == "confirming":
                        self.confirm(branch)
                    else:
                        raise ValueError(f"unknown branch status {branch['status']}")
                except Exception as exc:
                    self.state["errors"] += 1
                    self.retire(branch, str(exc))
                    if self.state["errors"] >= 3:
                        raise RuntimeError(
                            "Three consecutive branch failures; inspect branch logs before resuming"
                        ) from exc
            if self.state["phase"] != "disk_budget_exhausted":
                self.state["phase"] = (
                    "paused"
                    if self.stopped or (self.out / "STOP").exists()
                    else "time_budget_complete"
                )
        except Exception as exc:
            self.state.update(phase="failed", error=str(exc))
            raise
        finally:
            self.persist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--inherit", default="data/progressive/20260910")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--hours", type=float, default=96)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.workers <= 16 or not 0 < args.hours <= 8760:
        raise ValueError("invalid worker or time budget")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "runtime/scripts/autonomous_training.py"
    if args.resume and frozen.exists() and Path(__file__).resolve() != frozen:
        os.execve(
            sys.executable,
            [sys.executable, str(frozen), *sys.argv[1:]],
            {**os.environ, "PYTHONPATH": str(out / "runtime")},
        )
    with (out / "manager.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume:
            initialize(out, Path(args.inherit).resolve(), args)
            if args.prepare_only:
                return
        elif args.prepare_only:
            return
    if Path(__file__).resolve() != frozen:
        os.execve(
            sys.executable,
            [sys.executable, str(frozen), "--output", str(out), "--resume"],
            {**os.environ, "PYTHONPATH": str(out / "runtime")},
        )
    with (out / "manager.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(out)
        atomic_json(out.parent / "current.json", dict(path=str(out)))
        campaign.run()


if __name__ == "__main__":
    main()
