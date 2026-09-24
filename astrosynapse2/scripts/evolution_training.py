"""Persistent greedy-policy evolution with independent promotion certification.

Each generation races portable actors, rechecks finalists on fresh common seeds,
and validates one nominated step on another independent block before adoption.
Confirmation can then nominate it for the separately seeded promotion contract.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
from astro2.autonomy import commit_gate, read_json
from astro2.evolution import OPERATORS, choose_survivor, mutate, read_actor, recombine, write_actor
from astro2.experiment_control import atomic_json, code_identity, sha256
from autonomous_training import Campaign
from autonomous_training import initialize as initialize_portfolio


def initialize(out, inherited, args):
    inherited_manifest = read_json(inherited / "manifest.json")
    if (
        code_identity(inherited / "runtime/astro2", inherited / "runtime/scripts")
        != inherited_manifest["code_identity"]
    ):
        raise ValueError("inherited frozen runtime changed")
    initialize_portfolio(out, inherited, args)
    project = Path(__file__).resolve().parents[1]
    runtime = out / "runtime"
    shutil.copy2(project / "backend/astro2/evolution.py", runtime / "astro2/evolution.py")
    shutil.copy2(
        project / "scripts/evolution_training.py", runtime / "scripts/evolution_training.py"
    )
    manifest = read_json(out / "manifest.json")
    manifest.update(algorithm="greedy_evolution", manager="evolution_training.py", search_version=2)
    manifest["settings"].update(
        population=13,
        screen_pairs=128,
        selection_pairs=1024,
        adoption_pairs=4096,
        confirm_pairs=4096,
        max_rounds=16,
        max_stalled_generations=64,
        # Compatibility fields for the old progress reader. Evolution counts all
        # search/selection games explicitly, not fictitious PPO training blocks.
        games=0,
        round_iterations=0,
    )
    manifest["code_identity"] = code_identity(runtime / "astro2", runtime / "scripts")
    state = read_json(out / "state.json")
    state.update(
        name="Astro6 · Direct policy evolution",
        algorithm="greedy_evolution",
        history=[],
        learner=state["champion"],
        learner_model=state["model"],
        generation=0,
        accepted_steps=0,
        stalled_generations=0,
        generations_without_promotion=0,
        evaluated_games=0,
        search_games=0,
        verification_games=0,
        evaluation_counts={},
        search_summary=None,
    )
    if getattr(args, "continue_positive_gate", True):
        continue_positive_gate(out, inherited, state, manifest)
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "state.json", state)


def continue_positive_gate(out, inherited, state, manifest):
    """Continue an existing anytime-valid test without refunding or respending alpha.

    Choosing to continue after a futility stop is still optional sampling of
    the SAME predeclared e-process. No candidate, seed stream, alpha, or maximum
    changes. Old evidence stays immutable in the inherited campaign. The copied
    contiguous prefix is extended with previously unobserved pairs only.
    """
    eligible = [
        g
        for g in state["gates"]
        if (
            g.get("stage") == state["stage"]
            and not g.get("passed")
            and not g.get("abandoned")
            and g.get("method") == "paired_betting_confidence_sequence"
            and g.get("pairs", 0) >= 32768
            and 0.5 < g.get("score", 0) < 0.505
            and g.get("pairs", 0) < manifest["settings"]["max_gate_pairs"]
        )
    ]
    if not eligible:
        return
    gate = max(eligible, key=lambda g: g["log_evidence"])
    name = f"gate-{gate['stage']:03d}-{gate['attempt']:04d}"
    source = inherited / name
    identity = read_json(source / "manifest.json")
    if not identity or identity != dict(
        candidate_sha256=sha256(gate["actor"]),
        opponent_sha256=sha256(state["champion"]),
        seed=gate["seed"],
        attempt=gate["attempt"],
        maximum_pairs=manifest["settings"]["max_gate_pairs"],
        rules_version=1,
    ):
        raise ValueError("cannot continue a gate with different frozen inputs")
    rows = [json.loads(line) for line in (source / "pairs.jsonl").read_text().splitlines()]
    from astro2.sequential import promotion_evidence

    evidence = promotion_evidence(rows, gate["attempt"])
    if (
        [r["pair"] for r in rows] != list(range(gate["pairs"]))
        or evidence["score"] != gate["score"]
        or evidence["passed"]
    ):
        raise ValueError("saved gate evidence does not match its record")
    shutil.copytree(source, out / name)
    gate["continued_in"] = str(out / name)
    branch_id = "continued-gate"
    folder = out / "branches" / branch_id
    folder.mkdir(parents=True)
    # The complete player is byte-identical, but owned by this campaign so its
    # champion is discoverable and playable even after an old run is archived.
    local_actor = folder / "g00000001.actor.npz"
    local_model = folder / "g00000001.safetensors"
    for source_path, destination in (
        (gate["actor"], local_actor),
        (gate["model"], local_model),
        (gate["model"] + ".json", Path(str(local_model) + ".json")),
    ):
        shutil.copy2(source_path, destination)
    state["branches"].append(
        dict(
            id=branch_id,
            recipe=dict(name="continue_positive_evidence"),
            stage=state["stage"],
            round=0,
            games=0,
            best_score=evidence["score"],
            screens=[],
            status="verifying",
            source=state["model"],
            parent=state["champion"],
            folder=str(folder),
            candidates=[],
            actor=str(local_actor),
            model=str(local_model),
            reason="Continue the same frozen candidate and time-uniform test beyond the former futility cutoff",
        )
    )
    state.update(
        pending_gate={
            **{k: gate[k] for k in ("stage", "attempt", "seed")},
            "actor": str(local_actor),
            "model": str(local_model),
            "branch": branch_id,
            "continued_from": str(source),
        },
        active_branch=branch_id,
        learner=str(local_actor),
        learner_model=str(local_model),
    )
    state["evaluation_counts"][name] = 2 * len(rows)
    manifest["continued_gate"] = dict(
        path=str(source),
        original_pairs=len(rows),
        identity=identity,
        reason="Same anytime-valid test, unspent pair budget; all previously spent attempt alpha remains spent",
    )


class EvolutionCampaign(Campaign):
    def account_evaluation(self, result, key, category):
        if result is None:
            return
        counts = self.state["evaluation_counts"]
        count = 2 * result.get("pairs", 0)
        added = count - counts.get(key, 0)
        if added < 0:
            raise ValueError("evaluation prefix shrank")
        counts[key] = count
        self.state[category] += added
        self.state["evaluated_games"] += added
        self.state["games"] = self.state["inherited_games"] + self.state["search_games"]

    def persist(self):
        context = getattr(self, "evaluation_context", None)
        if context:
            self.account_evaluation(self.state.get("evaluation"), *context)
        super().persist()

    def gate_complete(self, report, maximum):
        # There is no minimum effect size in the promotion objective. In
        # particular, do not discard a positive 0.4% gain at 32,768 pairs just
        # because the previous supervisor imposed a 0.5% futility floor.
        return bool(
            report.get("passed")
            or report.get("pairs", 0) >= maximum
            or (report.get("pairs", 0) >= 8192 and report.get("score", 0.5) <= 0.4975)
        )

    def matches(self, actor, opponent, folder, seed, *, attempt=None, pairs=None):
        # A path is an immutable evaluation identity. Account partial evaluations
        # and resumes once, including the last block before a pause.
        key = str(folder.relative_to(self.out))
        category = "verification_games" if attempt or key.startswith("anchor-") else "search_games"
        self.evaluation_context = (key, category)
        self.state["evaluation"] = None
        try:
            result = super().matches(actor, opponent, folder, seed, attempt=attempt, pairs=pairs)
        finally:
            self.evaluation_context = None
        self.account_evaluation(result, key, category)
        self.persist()
        return result

    def new_branch(self):
        s = self.state
        index = s["generation"]
        operator = OPERATORS[index % len(OPERATORS)]
        # Revisit scales, not seeds. Useful steps survive across all operators.
        scale = dict(
            main_output=0.006, all_outputs=0.004, head_mixture=0.35, action_features=0.001
        )[operator]
        scale *= (1, 0.5, 2, 4)[(index // self.settings["max_rounds"]) % 4]
        name = f"b{index + 1:04d}"
        folder = self.out / "branches" / name
        folder.mkdir(parents=True, exist_ok=True)
        seed = self.settings["seed"] + index * 100003
        branch = dict(
            id=name,
            recipe=dict(name=operator, scale=scale),
            stage=s["stage"],
            round=index + 1,
            games=0,
            status="proposing",
            best_score=0.5,
            source=s["learner_model"],
            parent=s["learner"],
            opponent=s["champion"],
            seed=seed,
            folder=str(folder),
            screens=[],
            reason="Search the deployed greedy policy; retain the learner across generations",
            # Freeze sample sizes in the generation plan before any outcomes.
            screen_pairs=self.settings["screen_pairs"],
            selection_pairs=self.settings["selection_pairs"],
            adoption_pairs=self.settings.get("adoption_pairs", 4096),
            confirm_pairs=self.settings["confirm_pairs"],
            search_version=2,
            population=self.settings["population"],
        )
        s["branches"].append(branch)
        s.update(active_branch=name, branch_counter=index + 1)
        self.event(f"Generation {index + 1}: {operator}, mutation scale {scale:g}")
        return branch

    def propose(self, branch):
        s = self.state
        s.update(phase="evolving", evaluation=None)
        self.persist()
        folder = Path(branch["folder"])
        weights = read_actor(branch["parent"])
        candidates = [dict(actor=branch["parent"], model=branch["source"], kind="parent")]
        for index in range(branch["population"] - 1):
            seed = branch["seed"] + index // 2
            sign = 1 if index % 2 == 0 else -1
            actor = folder / f"g{index + 1:08d}.actor.npz"
            model = write_actor(
                mutate(
                    weights,
                    seed=seed,
                    sign=sign,
                    operator=branch["recipe"]["name"],
                    scale=branch["recipe"]["scale"],
                ),
                actor,
            )
            candidates.append(
                dict(actor=str(actor), model=model, kind="mutation", seed=seed, sign=sign)
            )
        if branch["parent"] != branch["opponent"]:
            candidates.append(dict(actor=branch["opponent"], model=s["model"], kind="champion"))
        for candidate in candidates:
            candidate["sha256"] = sha256(candidate["actor"])
        branch.update(
            candidates=candidates,
            status="screening",
            reason="Race candidates on the same paired game seeds",
        )
        self.persist()

    def evaluate_set(self, branch, indices, phase, pairs):
        self.state.update(
            phase=dict(screen="evolving", select="selecting_candidate", adopt="validating_step")[
                phase
            ]
        )
        self.persist()
        records, rows = [], []
        # Each stage has fresh seeds; each candidate shares seeds within a stage.
        seed = branch["seed"] + dict(screen=10**8, select=2 * 10**8, adopt=3 * 10**8)[phase]
        for index in indices:
            if self.halt() or self.skip(branch):
                return None
            candidate = branch["candidates"][index]
            if sha256(candidate["actor"]) != candidate["sha256"]:
                raise ValueError("candidate artifact changed")
            folder = Path(branch["folder"]) / f"{phase}-{index:02d}"
            self.state["search_summary"] = dict(
                generation=branch["round"],
                operator=branch["recipe"]["name"],
                phase=phase,
                candidate=indices.index(index) + 1,
                candidates=len(indices),
                pairs_per_candidate=pairs,
                accepted_steps=self.state["accepted_steps"],
                stalled_generations=self.state["stalled_generations"],
            )
            result = self.matches(candidate["actor"], branch["opponent"], folder, seed, pairs=pairs)
            if result["paused"]:
                return None
            records.append(result)
            rows.append(
                [json.loads(line) for line in (folder / "pairs.jsonl").read_text().splitlines()]
            )
        return records, rows

    def screen(self, branch):
        evaluated = self.evaluate_set(
            branch, list(range(len(branch["candidates"]))), "screen", branch["screen_pairs"]
        )
        if evaluated is None:
            return
        records, _ = evaluated
        ranking = sorted(range(1, len(records)), key=lambda i: records[i]["score"], reverse=True)
        finalists = [0, *ranking[:3]]
        # Always recheck the champion alongside the learner, so noisy accepted
        # steps cannot drift indefinitely away from the deployed baseline.
        for i, candidate in enumerate(branch["candidates"]):
            if candidate["kind"] == "champion" and i not in finalists:
                finalists.append(i)
        if branch.get("search_version", 1) >= 2:
            pairs = [(i, i + 1) for i in range(1, branch["population"] - 1, 2)]
            proposals = [
                tuple(read_actor(branch["candidates"][i]["actor"]) for i in pair) for pair in pairs
            ]
            weights, recipe = recombine(
                read_actor(branch["parent"]),
                proposals,
                [records[a]["score"] - records[b]["score"] for a, b in pairs],
            )
            if weights is not None:
                index = len(branch["candidates"])
                actor = Path(branch["folder"]) / f"g{index:08d}.actor.npz"
                model = write_actor(weights, actor)
                branch["candidates"].append(
                    dict(
                        actor=str(actor),
                        model=model,
                        kind="recombined",
                        sha256=sha256(actor),
                        components=[
                            {**item, "candidates": pairs[item["direction"]]} for item in recipe
                        ],
                    )
                )
                finalists.append(index)
        branch.update(
            screen_results=records,
            finalists=finalists,
            status="selecting",
            reason="Fresh matched games compare the finalists with the retained learner",
        )
        self.persist()

    def select(self, branch):
        evaluated = self.evaluate_set(
            branch, branch["finalists"], "select", branch["selection_pairs"]
        )
        if evaluated is None:
            return
        records, rows = evaluated
        winner, comparisons = choose_survivor(rows)
        index = branch["finalists"][winner]
        candidate = branch["candidates"][index]
        result = records[winner]
        s = self.state
        # Selecting the maximum creates optimism. Freeze one nominee, then use
        # untouched paired seeds before changing the persistent learner.
        branch.update(
            selection_results=records,
            comparisons=comparisons,
            winner=index,
            model=candidate["model"],
            actor=candidate["actor"],
            screens=[result["score"]],
            best_score=result["score"],
            accepted=False,
            gain=comparisons[winner]["gain"],
            status="adopting" if winner else "complete",
            reason=(
                f"Nominated step with selection gain {comparisons[winner]['gain']:.2%}; awaiting independent validation"
                if winner
                else "No supported step on fresh selection games; keep the learner"
            ),
        )
        if not winner:
            s["stalled_generations"] += 1
        s["history"].append(
            dict(
                stage=s["stage"],
                games=s["games"],
                checkpoint=branch["model"],
                score=result["score"],
                opponent=s["champion_label"],
                branch=branch["id"],
            )
        )
        self.event(f"{branch['id']}: {branch['reason']}; selection score {result['score']:.2%}")

    def adopt(self, branch):
        evaluated = self.evaluate_set(
            branch, [0, branch["winner"]], "adopt", branch.get("adoption_pairs", 4096)
        )
        if evaluated is None:
            return
        records, rows = evaluated
        winner, comparisons = choose_survivor(rows, standard_errors=2.0)
        comparison = comparisons[1]
        branch.update(
            adoption_results=records,
            adoption_comparison=comparison,
            accepted=bool(winner),
            status="confirming" if winner and records[1]["score"] > 0.5 else "complete",
            reason=(
                f"Accepted search step: independent matched gain {comparison['gain']:.2%}; promotion still unproven"
                if winner
                else f"Step failed independent validation (gain {comparison['gain']:.2%}); keep parent"
            ),
        )
        s = self.state
        if winner:
            s.update(learner=branch["actor"], learner_model=branch["model"])
            s["accepted_steps"] += 1
            s["stalled_generations"] = 0
        else:
            s["stalled_generations"] += 1
        # Acceptance and the next phase are one durable transition. Resume never
        # reuses this block to nominate a second candidate or accept twice.
        self.event(f"{branch['id']}: {branch['reason']}")

    def confirm(self, branch):
        self.state.update(phase="confirming_candidate")
        self.persist()
        result = self.matches(
            branch["actor"],
            branch["opponent"],
            Path(branch["folder"]) / "confirm",
            branch["seed"] + 10**9,
            pairs=branch.get("confirm_pairs", self.settings["confirm_pairs"]),
        )
        if result["paused"]:
            return
        branch["confirmation"] = result
        # A screening rule, not a significance claim. Certification starts anew.
        if result["score"] > 0.5 and result["score"] - 0.5 >= 2 * (
            result["paired_standard_error"] or 0
        ):
            s = self.state
            s["attempt"] += 1
            s["pending_gate"] = dict(
                model=branch["model"],
                actor=branch["actor"],
                stage=s["stage"],
                attempt=s["attempt"],
                branch=branch["id"],
                seed=self.settings["seed"] + 10**12 + s["stage"] * 10**7 + s["attempt"],
            )
            branch.update(
                status="verifying", reason="Frozen candidate qualified for fresh promotion evidence"
            )
            self.event(
                f"{branch['id']}: confirmation {result['score']:.2%}; promotion attempt {s['attempt']}"
            )
        else:
            branch.update(
                status="complete",
                reason=f"Confirmation {result['score']:.2%}; retain exploratory learner, no promotion claim",
            )
            self.event(f"{branch['id']}: {branch['reason']}")

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
        branch.update(
            status="complete",
            promotion_passed=record["passed"],
            reason="Promotion proven"
            if record["passed"]
            else "Promotion not proven; evidence and spent attempt retained",
        )
        if record["passed"]:
            s.update(
                learner=s["champion"],
                learner_model=s["model"],
                stalled_generations=0,
                generations_without_promotion=0,
            )
        self.event(
            f"{branch['id']}: {branch['reason']}; {record['score']:.3%}, lower bound {record['lower']:.3%}, {record['pairs']} pairs"
        )

    def finish(self, branch):
        s = self.state
        # Parent chains remain reconstructible from accepted portable actors.
        # Rejects retain their complete mutation recipe and paired evidence.
        keep = {s["learner"], s["champion"], branch["parent"]}
        keep.update(g["actor"] for g in s["gates"])
        if branch.get("accepted"):
            keep.add(branch["actor"])
        for candidate in branch.get("candidates", []):
            actor = Path(candidate["actor"])
            if str(actor) not in keep and actor.parent == Path(branch["folder"]):
                for path in (actor, Path(candidate["model"]), Path(candidate["model"] + ".json")):
                    path.unlink(missing_ok=True)
        branch.update(
            status="complete",
            games=sum(
                v
                for k, v in s["evaluation_counts"].items()
                if k.startswith(f"branches/{branch['id']}/")
            ),
        )
        s["generations_without_promotion"] = (
            0
            if branch.get("promotion_passed")
            else s.get("generations_without_promotion", 0)
            + max(0, branch["round"] - s["generation"])
        )
        s.update(active_branch=None, generation=branch["round"])
        self.persist()

    def run(self):
        self.state["phase"] = "starting"
        self.persist()
        try:
            while not self.halt():
                s = self.state
                if s["pending_gate"]:
                    self.gate()
                    continue
                if len(s["promotions"]) > s["inherited_promotions"]:
                    promotion = s["promotions"][-1]
                    if promotion.get("original_champion_benchmark", {}).get("pairs", 0) < 2048:
                        s["phase"] = "benchmarking_original_champion"
                        self.persist()
                        promotion["original_champion_benchmark"] = self.matches(
                            s["champion"],
                            str(self.out / "original-champion.actor.npz"),
                            self.out / f"anchor-{s['stage']:03d}",
                            self.settings["seed"] + 10**13 + s["stage"],
                            pairs=2048,
                        )
                        self.persist()
                        continue
                branch = next((b for b in s["branches"] if b["id"] == s["active_branch"]), None)
                if branch and (branch["stage"] != s["stage"] or branch["status"] == "complete"):
                    self.finish(branch)
                    continue
                if branch and self.skip(branch):
                    branch["reason"] = "Generation skipped; retained learner and all spent evidence"
                    self.finish(branch)
                    continue
                if not branch:
                    if (
                        max(s["stalled_generations"], s.get("generations_without_promotion", 0))
                        >= self.settings["max_stalled_generations"]
                    ):
                        s.update(
                            phase="search_stalled",
                            error=f"Search budget reached after {self.settings['max_stalled_generations']} generations without verified progress. Evidence retained. Increase the generation budget in Resource controls to continue.",
                        )
                        break
                    if shutil.disk_usage(self.out).free < 12 * 1024**3:
                        s["phase"] = "disk_budget_exhausted"
                        break
                    branch = self.new_branch()
                action = {
                    "proposing": self.propose,
                    "screening": self.screen,
                    "selecting": self.select,
                    "adopting": self.adopt,
                    "confirming": self.confirm,
                }.get(branch["status"])
                if action is None:
                    raise ValueError(f"unknown search state: {branch['status']}")
                action(branch)
            if self.state["phase"] not in {"disk_budget_exhausted", "search_stalled"}:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--inherit", default="data/progressive/autonomous-20260920")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--continue-positive-gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--hours", type=float, default=96)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not np.isfinite(args.hours) or not 0 < args.hours <= 8760:
        raise ValueError("invalid resource settings")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "runtime/scripts/evolution_training.py"
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
    if Path(__file__).resolve() != frozen:
        os.execve(
            sys.executable,
            [sys.executable, str(frozen), "--output", str(out), "--resume"],
            {**os.environ, "PYTHONPATH": str(out / "runtime")},
        )
    with (out / "manager.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = EvolutionCampaign(out)
        atomic_json(out.parent / "current.json", dict(path=str(out)))
        campaign.run()


if __name__ == "__main__":
    main()
