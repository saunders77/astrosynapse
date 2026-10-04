"""Whole-game replay and bounded independent critic fitting for Autopilot."""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import numpy as np

from .critic import IndependentCritic, save_npz
from .experiment_control import atomic_json, sha256


def game_partition(key):
    # Persistent assignment prevents a held-out game becoming training data next round.
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % 10


def export_trajectories(folder, iteration, trajectories, families, seed, encoder_version):
    folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    for i, t in enumerate(trajectories):
        if t.truncated or not t.value_states:
            continue
        indices = rng.choice(len(t.value_states), 32, replace=len(t.value_states) < 32)
        xs.append(
            np.concatenate(
                (
                    np.asarray(t.value_states)[indices],
                    np.eye(families, dtype=np.float32)[np.asarray(t.value_families)[indices]],
                ),
                axis=1,
            )
        )
        ys.append(t.target)
        ids.append(i)
    if xs:
        save_npz(
            folder / f"{iteration:05d}.npz",
            x=np.asarray(xs, dtype=np.float32),
            y=np.asarray(ys, dtype=np.float32),
            ids=np.asarray(ids),
            encoder_version=np.asarray(encoder_version),
        )


def fit_candidate(
    source,
    shards,
    output,
    *,
    seed,
    seconds,
    check,
    max_games=2500,
    epochs=30,
    recent_window=0,
    learning_rate=0.0003,
):
    """Recent plus historical games; train/validation/test disjoint across all rounds."""
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    model, meta, _ = IndependentCritic.load(source)
    original = copy.deepcopy(model)
    model.m = {k: np.zeros_like(v) for k, v in model.weights.items()}
    model.v = {k: np.zeros_like(v) for k, v in model.weights.items()}
    model.step = 0
    rng = np.random.default_rng(seed)
    # Select game IDs before loading feature arrays. 80% recent, 20% history.
    inventory = []
    for path in shards:
        check()
        with np.load(path) as z:
            if int(z["encoder_version"]) != meta["encoder_version"]:
                continue
            inventory.extend((str(path), int(i), j) for j, i in enumerate(z["ids"]))
    recent_count = min(int(max_games * 0.8), len(inventory))
    window = min(len(inventory), max(recent_count, recent_window))
    recent = inventory[-window:] if window else []
    if window == len(inventory):
        recent_count = min(max_games, len(inventory))
    selected = [recent[i] for i in rng.choice(len(recent), recent_count, replace=False)]
    history = inventory[:-window] if window else []
    if history:
        selected += [
            history[i]
            for i in rng.choice(
                len(history), min(max_games - recent_count, len(history)), replace=False
            )
        ]
    atomic_json(
        output / "replay.json",
        dict(
            source_sha256=sha256(source),
            seed=seed,
            games=[
                dict(path=p, game=g, index=i, partition=game_partition(f"{p}:{g}"))
                for p, g, i in selected
            ],
        ),
    )
    by_path = {}
    for path, gid, index in selected:
        by_path.setdefault(path, []).append((gid, index))
    datasets = {"train": [], "validation": [], "test": [], "historical_test": []}
    recent_keys = {(p, g) for p, g, _ in recent}
    for path, entries in by_path.items():
        check()
        with np.load(path) as z:
            shard_x, shard_y = z["x"], z["y"]
            for gid, index in entries:
                part = game_partition(f"{path}:{gid}")
                split = "train" if part < 7 else "validation" if part < 9 else "test"
                item = (shard_x[index].copy(), float(shard_y[index]))
                datasets[split].append(item)
                if split == "test" and (path, gid) not in recent_keys:
                    datasets["historical_test"].append(item)
    if min(len(datasets[k]) for k in ("train", "validation", "test")) < 10:
        report = dict(
            qualified=False,
            reason="Insufficient whole-game replay",
            counts={k: len(v) for k, v in datasets.items()},
        )
        atomic_json(output / "result.json", report)
        return report

    def metrics(critic, games):
        losses, briers = [], []
        for offset in range(0, len(games), 16):
            check()
            batch = games[offset : offset + 16]
            x = np.concatenate([g[0] for g in batch])
            y = np.repeat([g[1] for g in batch], 32)
            logits = critic.forward(x)[0]
            pred = critic.predict(x)
            losses.extend(
                (np.logaddexp(0, logits) - y * logits).reshape(-1, 32).mean(axis=1).tolist()
            )
            briers.extend(((pred - y) ** 2).reshape(-1, 32).mean(axis=1).tolist())
        return (
            dict(log_loss=float(np.mean(losses)), brier=float(np.mean(briers)), games=len(games))
            if games
            else None
        )

    before = metrics(original, datasets["validation"])
    best_loss, best, best_epoch, stale = before["log_loss"], copy.deepcopy(model), 0, 0
    started = time.monotonic()
    history_rows = []
    for epoch in range(1, epochs + 1):
        if time.monotonic() - started >= seconds:
            break
        check()
        order = rng.permutation(len(datasets["train"]))
        for offset in range(0, len(order), 16):
            check()
            batch = [datasets["train"][i] for i in order[offset : offset + 16]]
            model.train_batch(
                np.concatenate([g[0] for g in batch]),
                np.repeat([g[1] for g in batch], 32),
                learning_rate,
            )
        validation = metrics(model, datasets["validation"])
        history_rows.append(dict(epoch=epoch, **validation))
        if validation["log_loss"] < best_loss:
            best_loss, best, best_epoch, stale = (
                validation["log_loss"],
                copy.deepcopy(model),
                epoch,
                0,
            )
        else:
            stale += 1
        atomic_json(
            output / "progress.json", dict(epoch=epoch, best_epoch=best_epoch, history=history_rows)
        )
        if stale >= 3:
            break
    test_before, test_after = metrics(original, datasets["test"]), metrics(best, datasets["test"])
    old_history, new_history = (
        metrics(original, datasets["historical_test"]),
        metrics(best, datasets["historical_test"]),
    )
    qualified = (
        best_epoch > 0
        and test_after["log_loss"] < test_before["log_loss"]
        and test_after["brier"] <= test_before["brier"]
    )
    if old_history and new_history["log_loss"] > old_history["log_loss"] + 0.01:
        qualified = False
    report = dict(
        qualified=qualified,
        best_epoch=best_epoch,
        history=history_rows,
        validation_before=before,
        test_before=test_before,
        test_after=test_after,
        historical_before=old_history,
        historical_after=new_history,
        counts={k: len(v) for k, v in datasets.items()},
        settings=dict(
            max_games=max_games, recent_window=recent_window, learning_rate=learning_rate
        ),
        note="Prediction qualification only; fresh matched policy probes required for promotion.",
    )
    best.save(output / "critic.npz", {**meta, "autopilot_fit": report})
    atomic_json(output / "result.json", report)
    return report
