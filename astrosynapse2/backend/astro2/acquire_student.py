"""A single, bounded CART tree that ranks next-acquisition candidates.

Training uses one uniformly sampled pre-action decision boundary per player
turn. Labels are subsequent *actual* acquisitions, never teacher logits. The
only hindsight feature is total trade generated during that completed turn.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .arena import ModelResolutionError, _derived_seed, resolve_model
from .card_analysis import _GreedyChooser, _load_actor_encoder
from .cards import ALL_CARDS, CARD_BY_ID, EXPLORER, Faction
from .engine import ActionKind, Game, GameConfig, Observation
from .experiment_control import atomic_json

SCHEMA_VERSION = 1
NONE = -1
ACTIVE = {"queued", "collecting", "fitting", "evaluating"}
SAMPLING = (
    "One uniformly random pre-action decision boundary per player turn, including forced "
    "decisions. The action at that boundary is in the future. The label is the first paid "
    "or free acquisition at or after that boundary, or None if the turn ends first."
)
TRADE = (
    "Total trade actually generated over the entire teacher turn, including later draws "
    "and abilities, before subtracting purchases. This is an explicitly permitted hindsight "
    "input. Play requires your total or estimate; no hidden future is read."
)
SELECTION = (
    "Consider distinct cards currently in the trade row and Explorer when in supply. "
    "Remove cards costing more than total turn trade minus trade already spent, except "
    "row ships when your cards contain a free-ship ability. Always include None. "
    "Run this same tree for every candidate and choose the highest leaf score. "
    "Ties prefer None, then lower cost, "
    "then alphabetical card name. Scores measure imitation, not winning chances. Cards can "
    "be acquired later with unplayed trade or a free acquisition ability."
)


@dataclass(frozen=True)
class StudentConfig:
    games: int = 1000
    seed: int = 20260924
    max_nodes: int = 11
    rules_version: int = 2

    def __post_init__(self):
        if not 20 <= self.games <= 10_000:
            raise ValueError("games must be between 20 and 10,000")
        if not 3 <= self.max_nodes <= 99 or self.max_nodes % 2 != 1:
            raise ValueError("max_nodes must be an odd number between 3 and 99")
        if self.rules_version not in (1, 2):
            raise ValueError("rules_version must be 1 or 2")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be between 0 and 2^63 - 1")


def candidate_ids(observation: Observation, total_trade=None, spent=0) -> list[int]:
    ids = {card.card_id for card in observation.trade_row if card is not None}
    if total_trade is not None:
        own = [
            *observation.hand,
            *observation.own_deck,
            *observation.own_known_top,
            *observation.own_discard,
            *(e.card for e in observation.own_in_play),
        ]
        # Conservative future-turn feasibility, not a claim that a free ability
        # is immediately ready. Its source may still need to be drawn/played.
        free_ship = any(c.ally == "free_ship" or c.primary == "free_ship" for c in own)
        ids = {
            cid
            for cid in ids
            if CARD_BY_ID[cid].cost <= total_trade - spent
            or (free_ship and CARD_BY_ID[cid].is_ship)
        }
    if observation.explorers_remaining and (
        total_trade is None or EXPLORER.cost <= total_trade - spent
    ):
        ids.add(EXPLORER.card_id)
    return [NONE, *sorted(ids, key=lambda i: (CARD_BY_ID[i].cost, CARD_BY_ID[i].name))]


def features(observation: Observation, candidate: int, total_trade: float, spent: float):
    """Versioned human vocabulary shared verbatim by training and live advice."""
    o = observation
    card = CARD_BY_ID.get(candidate)
    own = [
        *o.hand,
        *o.own_deck,
        *o.own_known_top,
        *o.own_discard,
        *(entry.card for entry in o.own_in_play),
    ]
    opposing = [
        *o.opponent_hidden,
        *o.opponent_known_hand,
        *o.opponent_known_top,
        *o.opponent_discard,
        *(entry.card for entry in o.opponent_in_play),
    ]
    row = [c for c in o.trade_row if c is not None]
    remaining = max(0.0, total_trade - spent)
    values = {
        "Candidate is None": float(card is None),
        "Candidate cost": card.cost if card else 0,
        "Candidate printed trade": card.trade if card else 0,
        "Candidate printed combat": card.combat if card else 0,
        "Candidate printed authority": card.authority if card else 0,
        "Candidate base defense": card.defense if card else 0,
        "Candidate is a base": float(bool(card and card.is_base)),
        "Candidate has a draw ability": float(
            bool(card and ("draw" in card.primary or "draw" in card.ally))
        ),
        "Candidate has a scrap ability": float(bool(card and card.scrap)),
        "Candidate can thin your deck": float(
            bool(
                card
                and (
                    card.primary in {"scrap_hand", "scrap_discard", "scrap_any"}
                    or card.ally in {"scrap_hand", "scrap_discard", "scrap_any"}
                )
            )
        ),
        "Total trade generated this turn": total_trade,
        "Trade already spent this turn": spent,
        "Trade still available over this turn": remaining,
        "Trade currently in pool": o.trade,
        "Trade left after buying candidate": remaining - (card.cost if card else 0),
        "Candidate affordable from remaining turn trade": float(
            bool(card and card.cost <= remaining)
        ),
        "Row cards affordable from remaining turn trade": sum(c.cost <= remaining for c in row),
        "More expensive affordable row alternatives": sum(
            bool(card and card.cost < c.cost <= remaining) for c in row
        ),
        "Turn number": o.turn,
        "Your authority": o.own_authority,
        "Opponent authority": o.opponent_authority,
        "Your combat in pool": o.combat,
        "Your total deck size (all zones)": len(own),
        "Your undrawn cards": o.own_deck_count,
        "Your discard pile size": len(o.own_discard),
        "Your hand size": len(o.hand),
        "Your remaining starters": sum(c.card_id in (0, 1) for c in own),
        "Your deck printed trade": sum(c.trade for c in own),
        "Your deck printed combat": sum(c.combat for c in own),
        "Your deck printed authority": sum(c.authority for c in own),
        "Your deck draw cards": sum("draw" in c.primary or "draw" in c.ally for c in own),
        "Your bases in play": sum(e.card.is_base for e in o.own_in_play),
        "Opponent deck printed combat": sum(c.combat for c in opposing),
        "Opponent deck printed trade": sum(c.trade for c in opposing),
        "Opponent deck printed authority": sum(c.authority for c in opposing),
        "Opponent base defense in play": sum(e.card.defense for e in o.opponent_in_play),
        "Next acquired ship goes on top": float(o.next_ship_to_top),
        "Candidate faction cards in your deck": sum(
            bool(card and card.faction != Faction.UNALIGNED and c.faction == card.faction)
            for c in own
        ),
        "Candidate faction cards in your discard": sum(
            bool(card and card.faction != Faction.UNALIGNED and c.faction == card.faction)
            for c in o.own_discard
        ),
    }
    for faction in Faction:
        label = faction.value.replace("_", " ").title()
        values[f"Candidate faction is {label}"] = float(bool(card and card.faction == faction))
        values[f"Your {label} card count"] = sum(c.faction == faction for c in own)
        values[f"Opponent {label} card count"] = sum(c.faction == faction for c in opposing)
    for c in ALL_CARDS[2:]:
        values[f"Candidate is {c.name}"] = float(candidate == c.card_id)
    return values


class TurnSampler:
    """Reservoir sampling avoids weighting long turns more than short ones."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.turns = {}

    def observe(self, player_id, decision, selected):
        o = decision.observation
        key = (o.turn, player_id)
        turn = self.turns.setdefault(key, {"moments": 0, "spent": 0, "total_trade": 0})
        turn["moments"] += 1
        turn["total_trade"] = max(turn["total_trade"], o.trade + turn["spent"])
        if int(self.rng.integers(turn["moments"])) == 0:
            turn.update(
                observation=o,
                moment=turn["moments"] - 1,
                spent_at_moment=turn["spent"],
                target=NONE,
                found=False,
            )
        if selected.kind in (ActionKind.ACQUIRE, ActionKind.FREE_ACQUIRE):
            if not turn["found"]:
                turn.update(
                    target=selected.target_card_id
                    if selected.kind == ActionKind.FREE_ACQUIRE
                    else selected.card_id,
                    found=True,
                )
            if selected.kind == ActionKind.ACQUIRE:
                turn["spent"] += selected.amount

    def rows(self, game_index: int, *, truncated: bool = False):
        final_turn = max((key[0] for key in self.turns), default=-1)
        rows = []
        for (turn_number, player_id), turn in sorted(self.turns.items()):
            o = turn["observation"]
            ids = candidate_ids(o, turn["total_trade"], turn["spent_at_moment"])
            reason = (
                "unfinished_turn"
                if truncated and turn_number == final_turn
                else "future_market_card"
                if turn["target"] not in candidate_ids(o)
                else "unavailable_target"
                if turn["target"] not in ids
                else None
            )
            vectors = [
                features(o, cid, turn["total_trade"], turn["spent_at_moment"]) for cid in ids
            ]
            rows.append(
                {
                    "game": game_index,
                    "turn": turn_number,
                    "player_id": player_id,
                    "moment": turn["moment"],
                    "moments": turn["moments"],
                    "total_trade": turn["total_trade"],
                    "spent": turn["spent_at_moment"],
                    "candidates": ids,
                    "target": turn["target"],
                    "excluded_reason": reason,
                    "x": [list(v.values()) for v in vectors],
                    "observation": o.to_dict(),
                }
            )
        return rows


def tree_path(tree, x):
    path = []
    node = tree
    while "feature" in node:
        value = float(x[node["feature"]])
        left = value <= node["threshold"]
        path.append(
            {
                "node": node["id"],
                "feature": node["name"],
                "value": value,
                "threshold": node["threshold"],
                "answer": "yes" if left else "no",
            }
        )
        node = node["left"] if left else node["right"]
    return node["score"], path, node["id"]


def predict_row(tree, row):
    scores = [tree_path(tree, x)[0] for x in row["x"]]
    return row["candidates"][int(np.argmax(scores))]


def fit_tree(rows, names, max_nodes=11):
    """Deterministic weighted CART; one moment has total weight one.

    Histogram thresholds keep fitting bounded without another ML dependency.
    Best-first growth produces exactly one tree with at most max_nodes nodes.
    """
    rows = [r for r in rows if not r["excluded_reason"]]
    if not rows:
        raise ValueError("No usable training turns")
    x = np.asarray([v for r in rows for v in r["x"]], dtype=np.float32)
    y = np.asarray([float(c == r["target"]) for r in rows for c in r["candidates"]])
    weights = np.asarray([1 / len(r["candidates"]) for r in rows for _ in r["candidates"]])
    thresholds = [
        np.unique(np.quantile(x[:, j], np.linspace(0, 1, 33)))[:-1] for j in range(x.shape[1])
    ]
    binned = [np.searchsorted(t, x[:, j], side="left") for j, t in enumerate(thresholds)]
    min_weight = max(3.0, len(rows) * 0.002)

    def leaf(indices, node_id, depth):
        w = float(weights[indices].sum())
        return {
            "id": node_id,
            "score": float(np.dot(weights[indices], y[indices]) / w),
            "weight": w,
            "examples": len(indices),
            "depth": depth,
        }

    def split(indices, node):
        if node["depth"] >= 7:
            return None
        w = weights[indices]
        wy = w * y[indices]
        total, positive = w.sum(), wy.sum()
        best = None
        for j, cuts in enumerate(thresholds):
            if not len(cuts):
                continue
            bins = binned[j][indices]
            lw = np.cumsum(np.bincount(bins, weights=w, minlength=len(cuts) + 1))[:-1]
            lp = np.cumsum(np.bincount(bins, weights=wy, minlength=len(cuts) + 1))[:-1]
            rw, rp = total - lw, positive - lp
            gains = (
                lp**2 / np.maximum(lw, 1e-12) + rp**2 / np.maximum(rw, 1e-12) - positive**2 / total
            )
            gains[(lw < min_weight) | (rw < min_weight)] = -np.inf
            k = int(np.argmax(gains))
            gain = float(gains[k])
            if gain > 1e-9 and (best is None or gain > best[0]):
                best = (gain, j, float(cuts[k]))
        return best

    indices = np.arange(len(y))
    root = leaf(indices, 1, 0)
    leaves = [(root, indices, split(indices, root))]
    count = 1
    while count + 2 <= max_nodes:
        available = [(i, item[2][0]) for i, item in enumerate(leaves) if item[2] is not None]
        if not available:
            break
        index = max(available, key=lambda item: item[1])[0]
        node, ix, best = leaves.pop(index)
        _, j, cut = best
        left_ix = ix[x[ix, j] <= cut]
        right_ix = ix[x[ix, j] > cut]
        left = leaf(left_ix, count + 1, node["depth"] + 1)
        right = leaf(right_ix, count + 2, node["depth"] + 1)
        node.update(feature=j, name=names[j], threshold=cut, left=left, right=right)
        count += 2
        leaves.extend(
            (child, child_ix, split(child_ix, child))
            for child, child_ix in ((left, left_ix), (right, right_ix))
        )
    return root


def tree_size(tree):
    return 1 + tree_size(tree["left"]) + tree_size(tree["right"]) if "feature" in tree else 1


def english_tree(tree):
    lines = [SELECTION, "", "Each numbered line is one node. Start at 1 for every candidate."]

    def visit(node):
        if "feature" in node:
            lines.append(
                f"{node['id']}. Is {node['name']} ≤ {node['threshold']:.4g}? "
                f"Yes → {node['left']['id']}; No → {node['right']['id']}."
            )
            visit(node["left"])
            visit(node["right"])
        else:
            lines.append(f"{node['id']}. Give this candidate score {node['score']:.6f}.")

    visit(tree)
    return "\n".join(lines)


def metrics(tree, rows):
    complete = [r for r in rows if r["excluded_reason"] != "unfinished_turn"]
    usable = [r for r in complete if not r["excluded_reason"]]
    correct = sum(predict_row(tree, r) == r["target"] for r in usable)
    purchases = [r for r in usable if r["target"] != NONE]
    no_card = [r for r in usable if r["target"] == NONE]
    return {
        "turns": len(complete),
        "covered_turns": len(usable),
        "coverage": len(usable) / len(complete) if complete else None,
        "accuracy": correct / len(usable) if usable else None,
        "all_turn_accuracy": correct / len(complete) if complete else None,
        "purchase_accuracy": sum(predict_row(tree, r) == r["target"] for r in purchases)
        / len(purchases)
        if purchases
        else None,
        "none_accuracy": sum(predict_row(tree, r) == NONE for r in no_card) / len(no_card)
        if no_card
        else None,
        "always_none_accuracy": len(no_card) / len(usable) if usable else None,
    }


def split_games(games, seed):
    order = np.random.default_rng(seed).permutation(games)
    train_end, validation_end = int(0.7 * games), int(0.85 * games)
    return {
        int(g): "train" if i < train_end else "validation" if i < validation_end else "test"
        for i, g in enumerate(order)
    }


def recommend(artifact, observation, total_trade, spent):
    if not all(math.isfinite(v) and v >= 0 for v in (total_trade, spent)):
        raise ValueError("trade inputs must be finite and nonnegative")
    if total_trade < spent + observation.trade:
        raise ValueError("Total turn trade must cover trade already spent plus the current pool")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported student feature schema")
    candidates = []
    for cid in candidate_ids(observation, total_trade, spent):
        vector = features(observation, cid, total_trade, spent)
        if list(vector) != artifact["feature_names"]:
            raise ValueError("Student feature schema no longer matches this application")
        score, path, leaf = tree_path(artifact["tree"], list(vector.values()))
        candidates.append(
            {
                "card_id": cid,
                "name": CARD_BY_ID[cid].name if cid != NONE else "None",
                "score": score,
                "path": path,
                "leaf": leaf,
            }
        )
    selected = max(candidates, key=lambda c: c["score"])
    return {
        "recommendation": selected,
        "candidates": candidates,
        "total_trade": total_trade,
        "spent": spent,
        "selection_rule": SELECTION,
        "rules_version": artifact["config"]["rules_version"],
    }


def train_job(folder: Path):
    job_path = folder / "job.json"
    job = json.loads(job_path.read_text())
    config = StudentConfig(**job["config"])

    def update(**changes):
        job.update(changes, updated_at=time.time())
        atomic_json(job_path, job)

    def cancelled():
        return (folder / "STOP").exists()

    try:
        update(status="collecting", pid=os.getpid())
        actor, encoder = _load_actor_encoder(str(folder / "teacher.actor.npz"))
        splits = split_games(config.games, config.seed)
        partitions = {name: [] for name in ("train", "validation", "test")}
        feature_names = None
        counts = {
            "turns": 0,
            "future_market_card": 0,
            "unavailable_target": 0,
            "unfinished_turn": 0,
            "truncated_games": 0,
        }
        with gzip.open(folder / "samples.jsonl.gz", "wt", encoding="utf-8") as output:
            for game_index in range(config.games):
                if cancelled():
                    update(status="cancelled")
                    return
                sampler = TurnSampler(_derived_seed(config.seed, game_index, "student-moment"))
                chooser = _GreedyChooser(
                    actor, encoder, _derived_seed(config.seed, game_index, "student-policy")
                )
                game = Game(
                    choosers=(chooser, chooser),
                    config=GameConfig(
                        seed=_derived_seed(config.seed, game_index, "student-game"),
                        rules_version=config.rules_version,
                        max_turns=240,
                        max_actions_per_turn=220,
                    ),
                    decision_hook=sampler.observe,
                    cancel_hook=cancelled,
                )
                result = game.run()
                rows = sampler.rows(game_index, truncated=result.truncated)
                counts["truncated_games"] += int(result.truncated)
                for row in rows:
                    row["split"] = splits[game_index]
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    if feature_names is None:
                        o = next(iter(sampler.turns.values()))["observation"]
                        feature_names = list(features(o, NONE, 0, 0))
                    counts["turns"] += 1
                    if row["excluded_reason"]:
                        counts[row["excluded_reason"]] += 1
                    # Full observations remain in the audit dataset, not in fitting RAM.
                    row.pop("observation")
                    partitions[row["split"]].append(row)
                update(
                    games_completed=game_index + 1,
                    counts=counts,
                    progress=0.85 * (game_index + 1) / config.games,
                )
        if cancelled():
            update(status="cancelled")
            return
        update(status="fitting", progress=0.86)
        # Select tree size on separate source games; the test games stay untouched.
        sizes = sorted({3, min(7, config.max_nodes), min(11, config.max_nodes), config.max_nodes})
        choices = []
        selected_tree = None
        best_accuracy = -1.0
        for limit in sizes:
            if cancelled():
                update(status="cancelled")
                return
            tree = fit_tree(partitions["train"], feature_names, limit)
            validation = metrics(tree, partitions["validation"])
            accuracy = validation["all_turn_accuracy"] or 0.0
            choices.append({"nodes": tree_size(tree), "validation_accuracy": accuracy})
            if accuracy > best_accuracy + 1e-12:
                best_accuracy, selected_tree = accuracy, tree
            update(progress=0.86 + 0.09 * len(choices) / len(sizes), candidates=choices)
        update(status="evaluating", progress=0.96)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "id": job["id"],
            "model_id": job["model_id"],
            "model_label": job["model_label"],
            "teacher_sha256": job["teacher_sha256"],
            "config": asdict(config),
            "feature_names": feature_names,
            "tree": selected_tree,
            "node_count": tree_size(selected_tree),
            "tree_text": english_tree(selected_tree),
            "sampling": SAMPLING,
            "trade_definition": TRADE,
            "selection_rule": SELECTION,
            "counts": counts,
            "size_selection": choices,
            "split_games": {
                name: sorted(g for g, split in splits.items() if split == name)
                for name in partitions
            },
            "metrics": {name: metrics(selected_tree, rows) for name, rows in partitions.items()},
            "limitations": "Future row replacements not visible at the sampled moment are retained in the dataset, excluded from fitting, and counted as misses in all-turn accuracy. Unfinished final turns are censored. Accuracy is imitation fidelity, not playing strength.",
        }
        atomic_json(folder / "student.json", artifact)
        (folder / "tree.txt").write_text(artifact["tree_text"] + "\n", encoding="utf-8")
        update(status="complete", progress=1.0, result=artifact)
    except Exception as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


class StudentManager:
    """Durable detached jobs: browser/API restarts do not own training lifetime."""

    def __init__(self, store, output_dir: Path):
        self.store = store
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._children = []

    def folder(self, job_id):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise KeyError(job_id)
        folder = self.output_dir / job_id
        if not (folder / "job.json").is_file():
            raise KeyError(job_id)
        return folder

    def get(self, job_id):
        job = json.loads((self.folder(job_id) / "job.json").read_text())
        if job["status"] in ACTIVE and time.time() - job["updated_at"] > 10:
            try:
                if not job.get("pid"):
                    raise ProcessLookupError
                os.kill(job["pid"], 0)
            except ProcessLookupError:
                job.update(
                    status="interrupted",
                    error="Worker stopped before completing; start a new student. Results already saved remain available.",
                )
                atomic_json(self.folder(job_id) / "job.json", job)
        return job

    def list(self):
        self._children = [child for child in self._children if child.poll() is None]
        jobs = []
        for path in self.output_dir.glob("*/job.json"):
            job = self.get(path.parent.name)
            result = job.pop("result", None)
            if result:
                job["summary"] = {"node_count": result["node_count"], "metrics": result["metrics"]}
            jobs.append(job)
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)

    def create(self, model_id, config: StudentConfig):
        with (self.output_dir / "create.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if any(job["status"] in ACTIVE for job in self.list()):
                raise ValueError("An acquire student is already training; wait or cancel it first")
            resolved = resolve_model(self.store, model_id)
            if resolved.kind != "checkpoint":
                raise ModelResolutionError("Select a checkpoint to teach the acquire student")
            job_id = uuid.uuid4().hex
            folder = self.output_dir / job_id
            folder.mkdir()
            shutil.copyfile(resolved.actor_path, folder / "teacher.actor.npz")
            digest = hashlib.sha256((folder / "teacher.actor.npz").read_bytes()).hexdigest()
            job = {
                "id": job_id,
                "model_id": model_id,
                "model_label": resolved.label,
                "teacher_sha256": digest,
                "config": asdict(config),
                "status": "queued",
                "created_at": time.time(),
                "updated_at": time.time(),
                "progress": 0,
                "games_completed": 0,
                "error": None,
            }
            atomic_json(folder / "job.json", job)
            env = {
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
            try:
                with (folder / "worker.log").open("ab") as log:
                    child = subprocess.Popen(
                        [sys.executable, "-m", "astro2.acquire_student", str(folder)],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    self._children.append(child)
            except Exception as error:
                job.update(status="failed", error=str(error))
                atomic_json(folder / "job.json", job)
                raise
            return job

    def cancel(self, job_id):
        if self.get(job_id)["status"] in ACTIVE:
            (self.folder(job_id) / "STOP").touch()
        return self.get(job_id)

    def artifact(self, job_id):
        job = self.get(job_id)
        if job["status"] != "complete":
            raise ValueError("Student has not finished training")
        return job["result"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    train_job(parser.parse_args().folder)
