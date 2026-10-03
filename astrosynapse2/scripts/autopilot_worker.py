"""Single-owner, resumable policy/critic training and promotion supervisor."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from astro2.autopilot import CampaignConfig
from astro2.autopilot_data import fit_candidate
from astro2.autopilot_registry import read
from astro2.experiment_control import atomic_json, code_identity, sha256
from astro2.sequential import lower_sequence
from planning_experiment import pair, summary


class Paused(Exception):
    pass


def stream_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % 2**62


def paired_difference(a, b, alpha):
    if len(a) != len(b) or [r["pair"] for r in a] != [r["pair"] for r in b]:
        raise ValueError("Matched evaluation cases required")
    diffs = np.array(
        [sum(x["scores"]) / 2 - sum(y["scores"]) / 2 for x, y in zip(a, b, strict=True)]
    )
    return dict(
        difference=float(diffs.mean()),
        lower=2 * lower_sequence((diffs + 1) / 2, alpha) - 1,
        pairs=len(diffs),
        truncated=sum(r.get("truncated", 0) for r in a + b),
    )


class Campaign:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.config = CampaignConfig(**read(self.folder / "config.json"))
        self.state = read(self.folder / "state.json")
        self.inputs = read(self.folder / "inputs.json")
        self.jobs = read(self.folder / "jobs.json", {})
        self.runtime = self.folder / "runtime"
        self.started = time.monotonic()
        self.elapsed_before = self.state.get("elapsed_seconds", 0)
        self.pool = None
        self.stopping = False
        self.child = None

    def save(self):
        self.state["elapsed_seconds"] = self.elapsed_before + time.monotonic() - self.started
        self.state["updated_at"] = time.time()
        self.state["budget"] = {
            k: sum(j.get("seconds", 0) for j in self.jobs.values() if j["category"] == k)
            for k in ("policy", "critic", "evaluation")
        }
        self.state["training_games"] = sum(j.get("games", 0) for j in self.jobs.values())
        atomic_json(self.folder / "state.json", self.state)
        atomic_json(self.folder / "jobs.json", self.jobs)

    def event(self, kind, **details):
        if self.state.get("pending") and kind.startswith(("policy_", "critic_")):
            self.state["pending"]["finished"] = True
        self.state["events"].append(dict(at=time.time(), kind=kind, **details))
        self.save()
        print(json.dumps(self.state["events"][-1]), flush=True)

    def check(self):
        if self.stopping or (self.folder / "PAUSE").exists():
            raise Paused("Pause requested")
        if self.elapsed_before + time.monotonic() - self.started >= self.config.max_hours * 3600:
            raise Paused("Time budget reached")
        if shutil.disk_usage(self.folder).free < 2 * 1024**3:
            raise Paused("Less than 2 GB free disk space")

    def model(self):
        return next(m for m in self.state["models"] if m["id"] == self.state["champion_id"])

    def critic(self):
        return next(c for c in self.state["critics"] if c["id"] == self.state["critic_id"])

    def begin_job(self, key, category):
        self.state.update(status="running", active_job=key, phase=category)
        self.jobs.setdefault(key, dict(category=category, seconds=0, status="running"))
        self.save()
        out = self.folder / "tasks" / key
        out.mkdir(parents=True, exist_ok=True)
        return out

    def train(
        self,
        key,
        model,
        critic,
        opponent,
        games,
        seed,
        category="policy",
        optimizer=None,
        export=False,
    ):
        out = self.begin_job(key, category)
        target = games // self.config.batch_games
        state = read(out / "state.json", {})
        if state.get("games", 0) >= games:
            self.jobs[key].update(status="complete", games=games)
            self.save()
            return state
        self.check()
        opponents = self.inputs["opponents"][:]
        # Previous campaign champions remain available as diverse opponents.
        opponents += [
            m["actor_path"]
            for m in self.state["models"]
            if m.get("was_champion") and m["actor_path"] != opponent
        ][-4:]
        if not (out / "opponents.json").exists():
            atomic_json(out / "opponents.json", list(dict.fromkeys(opponents)))
        command = [
            sys.executable,
            str(self.runtime / "scripts/onpolicy_experiment.py"),
            "--model",
            model,
            "--opponent",
            opponent,
            "--independent-critic",
            critic,
            "--opponent-pool",
            str(out / "opponents.json"),
            "--output",
            str(out),
            "--iterations",
            str(target),
            "--games",
            str(self.config.batch_games),
            "--workers",
            str(self.config.workers),
            "--seed",
            str(seed),
            "--epochs",
            "2",
            "--learning-rate",
            str(self.config.learning_rate),
            "--rules-version",
            "2",
            "--eval-every",
            str(target + 1),
            "--eval-pairs",
            "8",
            "--matched-rollout-seeds",
            "--skip-embedded-critic-training",
        ]
        if export:
            command.append("--export-critic-data")
        if optimizer:
            command += ["--initial-optimizer", optimizer]
        if state:
            command.append("--resume")
        elif (out / "manifest.json").exists():
            # No durable first iteration: deterministic restart, never fabricate completion.
            (out / "manifest.json").unlink()
        (out / "STOP").unlink(missing_ok=True)
        with (out / "worker.log").open("ab") as log:
            self.child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
            )
            self.state["child_pid"] = self.child.pid
            self.save()
            pause_reason = None
            charged_at = time.monotonic()
            while self.child.poll() is None:
                now = time.monotonic()
                self.jobs[key]["seconds"] += now - charged_at
                charged_at = now
                try:
                    self.check()
                except Paused as error:
                    pause_reason = str(error)
                    (out / "STOP").touch()
                progress = read(out / "state.json", {})
                self.jobs[key]["games"] = progress.get("games", 0)
                self.save()
                time.sleep(2)
            returncode = self.child.returncode
        self.jobs[key]["seconds"] += time.monotonic() - charged_at
        self.state.pop("child_pid", None)
        self.child = None
        state = read(out / "state.json", {})
        self.jobs[key]["games"] = state.get("games", 0)
        self.save()
        if returncode:
            raise RuntimeError(f"Training failed ({returncode}); see tasks/{key}/worker.log")
        if state.get("games", 0) < games:
            raise Paused(pause_reason or "Training stopped before block completion")
        self.jobs[key]["status"] = "complete"
        metrics_path = out / "metrics.jsonl"
        if metrics_path.is_file():
            embedded_eval = sum(
                json.loads(line).get("evaluation_seconds", 0)
                for line in metrics_path.read_text().splitlines()
                if line.strip()
            )
            charged = self.jobs[key].get("internal_evaluation_seconds", 0)
            self.jobs[key]["seconds"] -= max(0, embedded_eval - charged)
            self.jobs[key]["internal_evaluation_seconds"] = embedded_eval
            self.jobs[key + "-internal-evaluation"] = dict(
                category="evaluation", status="complete", seconds=embedded_eval
            )
        self.save()
        return state

    def evaluate(self, key, actor, opponent, pairs, seed, alpha=None):
        out = self.begin_job(key, "evaluation")
        rows = read(out / "pairs.json", [])
        if [r["pair"] for r in rows] != list(range(len(rows))):
            raise ValueError("Evaluation must resume an exact prefix")
        started = time.monotonic()
        try:
            while len(rows) < pairs:
                if (
                    alpha is not None
                    and len(rows) >= 2000
                    and lower_sequence([sum(r["scores"]) / 2 for r in rows], alpha) > 0.5
                ):
                    break
                self.check()
                tasks = [
                    (actor, opponent, dict(rollouts=0, native=True, rules_version=2), seed, i)
                    for i in range(len(rows), min(pairs, len(rows) + 256))
                ]
                rows.extend(self.pool.map(pair, tasks, chunksize=2))
                now = time.monotonic()
                self.jobs[key]["seconds"] += now - started
                started = now
                self.save()
                atomic_json(out / "pairs.json", rows)
                atomic_json(
                    out / "progress.json",
                    dict(
                        pairs=len(rows),
                        target=pairs,
                        score=float(np.mean([sum(r["scores"]) / 2 for r in rows])),
                    ),
                )
            report = summary(rows)
            if alpha is not None:
                report.update(
                    lower=lower_sequence([sum(r["scores"]) / 2 for r in rows], alpha), alpha=alpha
                )
                report["passed"] = report["lower"] > 0.5 and not report["truncated"]
            atomic_json(out / "result.json", report)
            self.jobs[key]["status"] = "complete"
            return rows, report
        finally:
            self.jobs[key]["seconds"] += time.monotonic() - started
            self.save()

    def register(self, model, actor, critic_id, source_id, games, label):
        ident = "auto-" + self.state["id"] + "-p" + hashlib.sha256(actor.encode()).hexdigest()[:12]
        if not any(m["id"] == ident for m in self.state["models"]):
            self.state["models"].append(
                dict(
                    id=ident,
                    label=f"{self.config.name} · arch3 · {label}",
                    path=model,
                    actor_path=actor,
                    encoder_version=3,
                    architecture="arch3",
                    games=games,
                    parent_id=source_id,
                    created_at=time.time(),
                    checkpoint_name=label,
                    generation=None,
                    was_champion=False,
                    critic_id=critic_id,
                )
            )
            self.save()
        return next(m for m in self.state["models"] if m["id"] == ident)

    def policy_gate(self, pending, candidate):
        if pending.get("finished"):
            return
        key = pending["key"]
        opponent = pending["champion_actor"]
        _, screen = self.evaluate(
            key + "-screen",
            candidate["actor_path"],
            opponent,
            self.config.screen_pairs,
            stream_seed(self.config.seed, key + "-screen"),
        )
        if screen["score"] <= 0.5 or screen["truncated"]:
            self.event(
                "policy_rejected",
                candidate=candidate["id"],
                reason="Screen did not qualify",
                score=screen["score"],
            )
            return
        if "attempt" not in pending:
            self.state["attempt"] += 1
            pending["attempt"] = self.state["attempt"]
            self.save()
        attempt = pending["attempt"]
        alpha = 0.025 / (attempt * (attempt + 1))
        _, gate = self.evaluate(
            key + "-gate",
            candidate["actor_path"],
            opponent,
            self.config.gate_pairs,
            stream_seed(self.config.seed, key + "-gate"),
            alpha,
        )
        candidate["evaluation"] = dict(
            model_a_score=gate["score"], paired_lower=gate["lower"], pairs=gate["pairs"]
        )
        if not gate["passed"]:
            self.event("policy_inconclusive", candidate=candidate["id"], gate=gate)
            return
        panels = []
        for i, reference in enumerate(self.inputs["panel"]):
            seed = stream_seed(self.config.seed, key + f"-panel-{i}")
            a, _ = self.evaluate(
                key + f"-panel-{i}-candidate",
                candidate["actor_path"],
                reference,
                self.config.panel_pairs,
                seed,
            )
            b, _ = self.evaluate(
                key + f"-panel-{i}-incumbent", opponent, reference, self.config.panel_pairs, seed
            )
            evidence = paired_difference(a, b, alpha / len(self.inputs["panel"]))
            panels.append(evidence)
        passed = all(
            p["lower"] > -self.config.panel_regression and not p["truncated"] for p in panels
        )
        if not passed:
            self.event(
                "policy_inconclusive",
                candidate=candidate["id"],
                reason="Panel non-regression not established",
                panel=panels,
                gate=gate,
            )
            return
        candidate.update(
            was_champion=True,
            generation=1 + sum(p["kind"] == "policy" for p in self.state["promotions"]),
        )
        self.state["champion_id"] = candidate["id"]
        record = dict(
            at=time.time(),
            kind="policy",
            model_id=candidate["id"],
            previous=pending["champion_id"],
            critic_id=candidate["critic_id"],
            gate=gate,
            panel=panels,
        )
        self.state["promotions"].append(record)
        self.event(
            "policy_promoted", **{k: v for k, v in record.items() if k not in ("at", "kind")}
        )

    def critic_stage(self, pending):
        if pending.get("finished"):
            return
        key = pending["key"]
        out = self.begin_job(key, "critic")
        result = read(out / "result.json")
        if result is None:
            shards = sorted(
                (self.folder / "tasks").glob("policy-*/critic-data/*.npz"),
                key=lambda p: p.stat().st_mtime,
            )
            started = time.monotonic()
            try:
                result = fit_candidate(
                    pending["critic_path"],
                    shards,
                    out,
                    seed=stream_seed(self.config.seed, key),
                    seconds=min(120, pending["allowance"] / 4),
                    check=self.check,
                )
                self.jobs[key]["status"] = "complete"
            finally:
                self.jobs[key]["seconds"] += time.monotonic() - started
                self.save()
        if not result["qualified"]:
            self.event("critic_rejected", reason="Prediction qualification failed", report=result)
            return
        candidate_path = str(out / "critic.npz")
        evidence = []
        for i, seed in enumerate(self.config.seeds[:2]):
            common = dict(
                model=pending["champion_model"],
                opponent=pending["champion_actor"],
                games=self.config.probe_games,
                seed=stream_seed(seed, key),
                category="critic",
            )
            old = self.train(key + f"-probe-{i}-old", critic=pending["critic_path"], **common)
            new = self.train(key + f"-probe-{i}-new", critic=candidate_path, **common)
            eval_seed = stream_seed(self.config.seed, key + f"-probe-evaluation-{i}")
            a, _ = self.evaluate(
                key + f"-probe-{i}-new-eval",
                str(Path(new["model"]).with_suffix(".actor.npz")),
                pending["champion_actor"],
                self.config.probe_pairs,
                eval_seed,
            )
            b, _ = self.evaluate(
                key + f"-probe-{i}-old-eval",
                str(Path(old["model"]).with_suffix(".actor.npz")),
                pending["champion_actor"],
                self.config.probe_pairs,
                eval_seed,
            )
            evidence.append((a, b))
        # Each training seed has independent evaluation cases; uncertainty conditional on probes.
        a, b = [], []
        for xs, ys in evidence:
            for x, y in zip(xs, ys, strict=True):
                a.append({**x, "pair": len(a)})
                b.append({**y, "pair": len(b)})
        attempt = pending["attempt"]
        report = paired_difference(a, b, 0.05 / (attempt * (attempt + 1)))
        if report["lower"] <= 0 or report["truncated"]:
            self.event("critic_inconclusive", prediction=result, playing_strength=report)
            return
        ident = f"auto-{self.state['id']}-c{attempt}"
        self.state["critics"].append(
            dict(
                id=ident,
                kind="independent",
                path=candidate_path,
                label=f"{self.config.name} · arch3 critic {attempt}",
                encoder_version=3,
                created_at=time.time(),
                prediction=result,
                playing_strength=report,
            )
        )
        previous = self.state["critic_id"]
        self.state["critic_id"] = ident
        self.state["promotions"].append(
            dict(
                at=time.time(),
                kind="critic",
                critic_id=ident,
                previous=previous,
                prediction=result,
                playing_strength=report,
            )
        )
        self.event("critic_promoted", critic_id=ident, playing_strength=report)

    def run(self):
        expected = read(self.folder / "code.json")
        if expected != code_identity(self.runtime / "astro2", self.runtime / "scripts"):
            raise ValueError("Frozen campaign runtime changed")
        for name, digest in self.inputs["hashes"].items():
            if sha256(self.folder / "inputs" / name) != digest:
                raise ValueError("Campaign input changed: " + name)
        self.state.update(status="running", pid=os.getpid())
        self.state.pop("error", None)
        self.save()
        with ProcessPoolExecutor(
            max_workers=self.config.workers, mp_context=mp.get_context("spawn")
        ) as pool:
            self.pool = pool
            while True:
                self.check()
                pending = self.state["pending"]
                if pending is None:
                    if (self.folder / "DRAIN").exists():
                        raise Paused("Stopped after completed block")
                    if self.config.max_blocks and self.state["blocks"] >= self.config.max_blocks:
                        self.state.update(
                            status="complete", phase="block_budget_complete", active_job=None
                        )
                        self.save()
                        return
                    size = sum(p.stat().st_size for p in self.folder.rglob("*") if p.is_file())
                    self.state["storage_bytes"] = size
                    if size > self.config.storage_gb * 1024**3:
                        raise Paused("Storage budget reached")
                    champion, critic = self.model(), self.critic()
                    budget = self.state["budget"]
                    allowance = (
                        budget["policy"]
                        * (1 - self.config.policy_fraction)
                        / self.config.policy_fraction
                        - budget["critic"]
                    )
                    pending = dict(
                        champion_id=champion["id"],
                        champion_model=champion["path"],
                        champion_actor=champion["actor_path"],
                        critic_id=critic["id"],
                        critic_path=critic["path"],
                    )
                    if self.state["initial_index"] < len(self.inputs["candidates"]):
                        index = self.state["initial_index"]
                        pending.update(
                            kind="initial",
                            key=f"initial-{index}",
                            candidate=self.inputs["candidates"][index],
                        )
                    elif (
                        allowance
                        >= max(
                            180,
                            budget["policy"]
                            / max(1, self.state["blocks"])
                            * self.config.probe_games
                            / self.config.block_games
                            * 2
                            * min(2, len(self.config.seeds)),
                        )
                        and self.state.get("critic_at_block") != self.state["blocks"]
                    ):
                        self.state["critic_attempt"] += 1
                        pending.update(
                            kind="critic",
                            key=f"critic-{self.state['critic_attempt']:04d}",
                            attempt=self.state["critic_attempt"],
                            allowance=allowance,
                        )
                    else:
                        block = self.state["blocks"]
                        lane = block % len(self.config.seeds)
                        tip = self.state["tips"].get(str(lane), {})
                        source = tip if tip.get("champion_id") == champion["id"] else {}
                        pending.update(
                            kind="policy",
                            key=f"policy-{block:05d}",
                            lane=lane,
                            source=source.get("model", champion["path"]),
                            source_games=source.get("games", champion.get("games", 0)),
                            optimizer=source.get("optimizer"),
                            seed=stream_seed(self.config.seeds[lane], f"block-{block}"),
                        )
                    self.state["pending"] = pending
                    self.save()
                if pending["kind"] == "critic":
                    self.critic_stage(pending)
                    self.state["critic_at_block"] = self.state["blocks"]
                else:
                    if pending["kind"] == "initial":
                        source = pending["candidate"]
                        candidate = self.register(
                            source["model"],
                            source["actor"],
                            pending["critic_id"],
                            source["source_id"],
                            20000,
                            pending["key"],
                        )
                    else:
                        trained = self.train(
                            pending["key"],
                            pending["source"],
                            pending["critic_path"],
                            pending["champion_actor"],
                            self.config.block_games,
                            pending["seed"],
                            optimizer=pending["optimizer"],
                            export=True,
                        )
                        candidate = self.register(
                            trained["model"],
                            str(Path(trained["model"]).with_suffix(".actor.npz")),
                            pending["critic_id"],
                            pending["champion_id"],
                            pending.get("source_games", 0) + self.config.block_games,
                            pending["key"],
                        )
                        self.state["tips"][str(pending["lane"])] = dict(
                            model=trained["model"],
                            optimizer=trained["optimizer"],
                            champion_id=pending["champion_id"],
                            games=candidate["games"],
                        )
                    self.policy_gate(pending, candidate)
                    if pending["kind"] == "initial":
                        self.state["initial_index"] += 1
                    else:
                        self.state["blocks"] += 1
                self.state.update(pending=None, active_job=None)
                self.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    with (args.folder / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(args.folder)

        def stop(*_):
            campaign.stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            campaign.run()
        except Paused as error:
            campaign.state.update(status="paused", phase=str(error))
            campaign.save()
        except Exception as error:
            campaign.state.update(status="failed", error=f"{type(error).__name__}: {error}")
            campaign.save()
            raise


if __name__ == "__main__":
    main()
