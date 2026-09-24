"""Portable curriculum, replay sampling, and durable promotion transitions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .experiment_control import atomic_json, sha256

RECIPES = [
    dict(
        name="League exploration",
        temperature=0.12,
        learning_rate=2e-6,
        entropy_weight=0.002,
        anchor_weight=0.10,
        advantage_baseline="critic",
        train_scope="full",
    ),
    dict(
        name="Outcome-only learning",
        temperature=0.08,
        learning_rate=1e-6,
        entropy_weight=0.001,
        anchor_weight=0.05,
        advantage_baseline="constant",
        train_scope="full",
    ),
    dict(
        name="Broad exploration",
        temperature=0.25,
        learning_rate=4e-6,
        entropy_weight=0.003,
        anchor_weight=0.10,
        advantage_baseline="critic",
        train_scope="full",
    ),
    dict(
        name="Conservative policy heads",
        temperature=0.05,
        learning_rate=2e-6,
        entropy_weight=0.0,
        anchor_weight=0.20,
        advantage_baseline="constant",
        train_scope="heads",
    ),
]


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def league_schedule(champion, opponents, seed, games):
    """Exact 75/25 incumbent/history mix, balanced by seat, stable on resume."""
    paths = [champion] * games
    if opponents:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1EA6]))
        for seat in (0, 1):
            indices = np.arange(seat, games, 2)
            for i in rng.choice(indices, size=len(indices) // 4, replace=False):
                paths[i] = opponents[int(rng.integers(len(opponents)))]
    return paths


def historical_inventory(data, spec):
    """Read retained data only. Stale manifest entries never count as usable data."""
    data = Path(data)
    report = dict(database={}, replay_shards=0, replay_positions=0, replay_sources=[])
    db_path = data / "astrosynapse2.sqlite3"
    if db_path.exists():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            for table in ("metrics", "checkpoints", "arena_jobs"):
                report["database"][table] = db.execute(f"select count(*) from {table}").fetchone()[
                    0
                ]
    shards, seen = [], set()
    for root in sorted((data / "checkpoints").glob("*")):
        manifests = sorted(
            root.glob("*.policy-replay.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not manifests:
            continue
        cold = read_json(manifests[0], {}).get("cold", {})
        if (cold.get("state_size"), cold.get("action_size")) != (spec.state_size, spec.action_size):
            continue
        retained = []
        for shard in cold.get("shards", []):
            path = Path(shard["path"]).resolve()
            if not path.is_relative_to(data.resolve()) or str(path) in seen:
                continue
            if all(
                (path / (key + ".npy")).exists()
                for key in ("states", "legal_actions", "action_offsets", "families")
            ):
                seen.add(str(path))
                retained.append(dict(path=str(path), decisions=shard["decisions"]))
        if retained:
            shards.append(retained)
            report["replay_sources"].append(root.name)
            report["replay_shards"] += len(retained)
            report["replay_positions"] += sum(s["decisions"] for s in retained)
    return report, shards


def prepare_history(data, actor_path, destination, *, seed, positions=8192):
    """Stratify across runs and time; relabel legal choices with the incumbent.

    Historical outcomes are not current-policy advantages. Their observations
    provide a fixed KL anchor, while all reward gradients use new complete games.
    Only selected rows are paged from the large, immutable replay archive.
    """
    from .native_actor import NativeActor

    destination = Path(destination)
    manifest = destination.with_suffix(".json")
    identity = sha256(actor_path)
    saved = read_json(manifest, {})
    if destination.exists() and saved.get("teacher_sha256") == identity:
        if saved.get("dataset_sha256") != sha256(destination):
            raise ValueError("history dataset changed")
        return saved
    actor = NativeActor.load(actor_path)
    report, groups = historical_inventory(data, actor.spec)
    if not groups:
        raise ValueError("no compatible retained replay shards")
    rng = np.random.default_rng(seed)
    states, actions, families, logits, offsets, origins = [], [], [], [], [0], []
    for group in groups:
        chosen = np.linspace(0, len(group) - 1, min(24, len(group)), dtype=int)
        per_shard = max(1, positions // len(groups) // len(chosen))
        for index in chosen:
            shard = Path(group[index]["path"])
            arrays = {
                key: np.load(shard / (key + ".npy"), mmap_mode="r", allow_pickle=False)
                for key in ("states", "legal_actions", "action_offsets", "families")
            }
            counts = np.diff(arrays["action_offsets"])
            eligible = np.flatnonzero((counts >= 2) & (counts <= 64))
            sampled = rng.choice(eligible, size=min(per_shard, len(eligible)), replace=False)
            for i in sampled:
                state = np.array(arrays["states"][i], dtype=np.float32)
                legal = np.array(
                    arrays["legal_actions"][
                        int(arrays["action_offsets"][i]) : int(arrays["action_offsets"][i + 1])
                    ],
                    dtype=np.float32,
                )
                family = int(arrays["families"][i])
                if (
                    not np.all(np.isfinite(state))
                    or not np.all(np.isfinite(legal))
                    or not 0 <= family < actor.spec.families
                ):
                    continue
                teacher = actor.predict_options(state, legal, family).mean(axis=1)
                if not np.all(np.isfinite(teacher)):
                    raise ValueError("nonfinite history teacher")
                states.append(state)
                actions.extend(legal)
                families.append(family)
                logits.extend(teacher)
                offsets.append(len(actions))
            origins.append(dict(path=str(shard), sampled=len(sampled)))
    if len(states) < 2:
        raise ValueError("no usable multi-action replay observations")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        states=np.asarray(states),
        actions=np.asarray(actions),
        families=np.asarray(families, dtype=np.int32),
        logits=np.asarray(logits),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    report.update(
        sampled_positions=len(states),
        teacher_sha256=identity,
        dataset_sha256=sha256(destination),
        seed=seed,
        sources=origins,
        purpose="Incumbent-policy KL anchor on retained historical observations",
    )
    atomic_json(manifest, report)
    return report


def anchor_batch(bank, indices, temperature=0.1):
    counts = bank["offsets"][indices + 1] - bank["offsets"][indices]
    count = int(max(counts))
    actions = np.zeros((len(indices), count, bank["actions"].shape[-1]), dtype=np.float32)
    mask = np.zeros((len(indices), count), dtype=np.float32)
    logp = np.zeros_like(mask)
    for j, i in enumerate(indices):
        a, b = bank["offsets"][i : i + 2]
        actions[j, : b - a] = bank["actions"][a:b]
        mask[j, : b - a] = 1
        values = bank["logits"][a:b].astype(np.float64) / temperature
        values -= values.max()
        logp[j, : b - a] = values - np.log(np.exp(values).sum())
    return bank["states"][indices], actions, mask, bank["families"][indices], logp


def anchor_loss(model, states, actions, mask, families, teacher, temperature=0.1):
    import mlx.core as mx

    from .onpolicy import masked_log_policy

    logp, _ = masked_log_policy(model, states, actions, mask, families, temperature)
    return mx.mean(mx.sum(mx.exp(teacher) * mask * (teacher - logp), axis=1))


def gate_finished(report, maximum):
    n, score = report.get("pairs", 0), report.get("score", 0.5)
    return bool(
        report.get("passed")
        or n >= maximum
        or (n >= 2048 and score < 0.49)
        or (n >= 8192 and score <= 0.5)
        or (n >= 32768 and score < 0.505)
    )


def commit_gate(state, evidence):
    """One state transaction owns gate completion, promotion, and branch retirement."""
    gate = state["pending_gate"]
    if gate is None:
        raise ValueError("no pending promotion gate")
    record = {**gate, **evidence, "games": state["games"], "opponent": state["champion_label"]}
    state["gates"].append(record)
    if evidence["passed"]:
        state["promotions"].append(record)
        state.update(
            model=gate["model"],
            champion=gate["actor"],
            stage=state["stage"] + 1,
            champion_label=f"Astro6 champion {state['stage'] + 1}",
            attempt=0,
        )
    state["pending_gate"] = None
    return record
