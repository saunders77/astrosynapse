"""MiniAstro: a budgeted additive, shallow acquisition policy.

Each named subnetwork sees a declared subset of public features and contributes
one scalar to a candidate's logit. Contributions sum exactly; they are model
arithmetic, not causal explanations. Inference uses NumPy; fitting uses MLX.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter
from functools import lru_cache

import numpy as np
import orjson

from .cards import ALL_CARDS, CARD_BY_ID
from .experiment_control import atomic_json

VERSION = 1
GROUP_NAMES = ("Economy", "Deck composition", "Timing", "Matchup", "Market")
SELECTION = (
    "Score each available acquisition candidate and None with MiniAstro. Its logit is the "
    "sum of linear candidate terms and five named, one-hidden-layer ReLU subnetworks. Choose the "
    "highest logit; ties prefer None, then lower cost, then alphabetical name. Softmax "
    "shares describe the student's choices, not win probability. Group names describe "
    "input restrictions, not discovered or proven strategic concepts."
)


class TrainingCancelled(Exception):
    pass


def extras(observation, candidate):
    """Only the sampled public observation. No future hand/order or outcome."""
    o = observation

    def count(cards):
        return Counter(c["card_id"] for c in cards if c is not None)

    hand = count(o["hand"])
    discard = count(o["own_discard"])
    played = count([e["card"] for e in o["own_in_play"]])
    own = hand + discard + played + count(o["own_deck"]) + count(o["own_known_top"])
    opponent = count(
        [
            *o["opponent_hidden"],
            *o["opponent_known_hand"],
            *o["opponent_known_top"],
            *o["opponent_discard"],
            *(e["card"] for e in o["opponent_in_play"]),
        ]
    )
    market = count(o["trade_row"])
    values = {f"Your copies of {c.name}": own[c.card_id] for c in ALL_CARDS}
    values.update({f"Opponent copies of {c.name}": opponent[c.card_id] for c in ALL_CARDS})
    values.update({f"Market copies of {c.name}": market[c.card_id] for c in ALL_CARDS[2:]})
    values.update(
        {
            "Candidate copies in your hand": hand[candidate],
            "Candidate copies in your discard": discard[candidate],
            "Candidate copies in play": played[candidate],
        }
    )
    return values


def feature_vector(observation, candidate, total_trade, spent):
    from .acquire_student import features

    values = features(observation, candidate, total_trade, spent)
    values.update(extras(observation.to_dict(), candidate))
    return values


def architecture(names, budget):
    """Allocate hidden units round-robin without exceeding the real parameter cap."""
    static = [i for i, name in enumerate(names) if name.startswith("Candidate ")]
    indices = []
    for group in GROUP_NAMES:
        selected = set(static)
        for i, name in enumerate(names):
            lower = name.lower()
            if group == "Economy" and (
                "trade" in lower or "affordable" in lower or "starter" in lower
            ):
                selected.add(i)
            if group == "Deck composition" and (name.startswith("Your ") or "faction" in lower):
                selected.add(i)
            if group == "Timing" and any(
                word in lower
                for word in ("turn", "hand", "discard", "undrawn", "top", "draw", "deck size")
            ):
                selected.add(i)
            if group == "Matchup" and (
                name.startswith("Opponent ")
                or any(word in lower for word in ("authority", "combat", "defense", "bases"))
            ):
                selected.add(i)
            if group == "Market" and (
                name.startswith("Market ") or "row" in lower or "trade" in lower
            ):
                selected.add(i)
        indices.append(sorted(selected))
    base = sorted(
        set(
            static
            + [
                i
                for i, n in enumerate(names)
                if n
                in {
                    "Trade left after buying candidate",
                    "More expensive affordable row alternatives",
                }
            ]
        )
    )
    widths = [1] * len(indices)
    used = len(base) + sum(len(ix) + 2 for ix in indices)
    if used > budget:
        raise ValueError(f"Feature schema needs at least {used} MiniAstro parameters")
    while True:
        affordable = [i for i, ix in enumerate(indices) if used + len(ix) + 2 <= budget]
        if not affordable:
            break
        i = min(affordable, key=lambda g: (widths[g], g))
        widths[i] += 1
        used += len(indices[i]) + 2
    return {
        "version": VERSION,
        "parameter_budget": budget,
        "parameter_count": used,
        "base_indices": base,
        "normalization": "signed log1p, train-only mean/std; clipped to ±6",
        "groups": [
            {"name": name, "indices": ix, "width": width, "parameters": (len(ix) + 2) * width}
            for name, ix, width in zip(GROUP_NAMES, indices, widths, strict=True)
        ],
    }


def initialize(spec, seed):
    rng = np.random.default_rng(seed)
    params = {"baseline": np.zeros(len(spec["base_indices"]), dtype=np.float32)}
    for i, group in enumerate(spec["groups"]):
        dim, width = len(group["indices"]), group["width"]
        params[f"w{i}"] = (rng.normal(size=(dim, width)) * np.sqrt(2 / dim)).astype(np.float32)
        params[f"b{i}"] = np.zeros(width, dtype=np.float32)
        params[f"v{i}"] = (rng.normal(size=width) * 0.03 / np.sqrt(width)).astype(np.float32)
    assert sum(p.size for p in params.values()) == spec["parameter_count"]
    return params


def transform(x, mean, scale):
    z = np.sign(x) * np.log1p(np.abs(x))
    return np.clip((z - mean) / scale, -6, 6).astype(np.float32)


def forward(params, spec, x, *, detailed=False):
    terms = [x[..., spec["base_indices"]] @ params["baseline"]]
    units = []
    for i, group in enumerate(spec["groups"]):
        activation = np.maximum(x[..., group["indices"]] @ params[f"w{i}"] + params[f"b{i}"], 0)
        contribution = activation * params[f"v{i}"]
        terms.append(contribution.sum(axis=-1))
        if detailed:
            units.append(contribution)
    components = np.stack(terms, axis=-1)
    return components.sum(axis=-1), components, units


def forward_mlx(params, spec, x):
    import mlx.core as mx

    scores = x[..., spec["base_indices"]] @ params["baseline"]
    for i, group in enumerate(spec["groups"]):
        h = mx.maximum(x[..., group["indices"]] @ params[f"w{i}"] + params[f"b{i}"], 0)
        scores = scores + mx.sum(h * params[f"v{i}"], axis=-1)
    return scores


def stream_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_arrays(folder, meta, update, cancelled):
    """Disk-backed raw features shared by every budget using the same dataset."""
    digest = stream_hash(folder / "samples.jsonl.gz")
    cache = folder.parent / "feature_cache" / f"{digest}-miniastro-v{VERSION}"
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (cache / "manifest.json").exists():
            names = None
            pointers, targets, splits, ids, reasons = [], [], [], [], []
            count = 0
            with (
                gzip.open(folder / "samples.jsonl.gz", "rb") as source,
                (cache / "features.bin").open("wb") as output,
            ):
                for index, line in enumerate(source):
                    if index % 1000 == 0:
                        if cancelled():
                            raise TrainingCancelled
                        update(
                            status="preparing",
                            progress=0.85 + 0.03 * index / max(1, meta["counts"]["turns"]),
                            prepared_turns=index,
                        )
                    row = orjson.loads(line)
                    candidates = row["candidates"]
                    extra = [extras(row["observation"], cid) for cid in candidates]
                    if names is None:
                        names = [*meta["feature_names"], *extra[0].keys()]
                    vectors = np.asarray(
                        [
                            base + list(add.values())
                            for base, add in zip(row["x"], extra, strict=True)
                        ],
                        dtype=np.float32,
                    )
                    output.write(vectors.tobytes())
                    pointers.append(
                        [*range(count, count + len(candidates)), *([-1] * (7 - len(candidates)))]
                    )
                    count += len(candidates)
                    targets.append(
                        candidates.index(row["target"]) if not row["excluded_reason"] else -1
                    )
                    ids.append([*candidates, *([-2] * (7 - len(candidates)))])
                    splits.append({"train": 0, "validation": 1, "test": 2}[row["split"]])
                    reasons.append(
                        2
                        if row["excluded_reason"] == "unfinished_turn"
                        else 1
                        if row["excluded_reason"]
                        else 0
                    )
            np.savez(
                cache / "choices.npz",
                pointers=np.asarray(pointers, dtype=np.int64),
                targets=np.asarray(targets, dtype=np.int32),
                splits=np.asarray(splits, dtype=np.int8),
                ids=np.asarray(ids, dtype=np.int16),
                reasons=np.asarray(reasons, dtype=np.int8),
            )
            atomic_json(
                cache / "manifest.json",
                {"shape": [count, len(names)], "feature_names": names, "dataset_sha256": digest},
            )
    manifest = json.loads((cache / "manifest.json").read_text())
    with np.load(cache / "choices.npz", allow_pickle=False) as choices:
        arrays = {key: choices[key] for key in choices.files}
    arrays["x"] = np.memmap(
        cache / "features.bin", mode="r", dtype=np.float32, shape=tuple(manifest["shape"])
    )
    return arrays, manifest


def normalization(data):
    ix = np.flatnonzero((data["splits"] == 0) & (data["targets"] >= 0))
    count = 0
    sums = np.zeros(data["x"].shape[1], dtype=np.float64)
    squares = sums.copy()
    for start in range(0, len(ix), 1024):
        rows = data["pointers"][ix[start : start + 1024]].ravel()
        x = data["x"][rows[rows >= 0]]
        x = np.sign(x) * np.log1p(np.abs(x))
        count += len(x)
        sums += x.sum(axis=0, dtype=np.float64)
        squares += (x.astype(np.float64) ** 2).sum(axis=0)
    if not count:
        raise ValueError("No usable MiniAstro training examples")
    mean = sums / count
    scale = np.maximum(np.sqrt(np.maximum(0, squares / count - mean**2)), 0.1)
    return mean.astype(np.float32), scale.astype(np.float32)


def batch(data, indices, mean, scale):
    pointers = data["pointers"][indices]
    x = transform(data["x"][np.maximum(pointers, 0)], mean, scale)
    return x, pointers >= 0, data["targets"][indices]


def evaluate(params, spec, data, mean, scale, split, *, score_fn=None):
    ix = np.flatnonzero((data["splits"] == split) & (data["reasons"] != 2))
    eligible = ix[data["targets"][ix] >= 0]
    predictions = np.full(len(data["targets"]), -1, dtype=np.int32)
    loss_sum = 0.0
    for start in range(0, len(eligible), 1024):
        chosen = eligible[start : start + 1024]
        x, mask, y = batch(data, chosen, mean, scale)
        scores = score_fn(x) if score_fn else forward(params, spec, x)[0]
        scores = np.where(mask, scores, -1e9)
        predictions[chosen] = scores.argmax(axis=1)
        shifted = scores - scores.max(axis=1, keepdims=True)
        loss_sum += float(
            (np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]).sum()
        )
    correct = predictions[eligible] == data["targets"][eligible]
    true_ids = data["ids"][eligible, data["targets"][eligible]]
    purchases = true_ids != -1

    def average(values):
        return float(np.mean(values)) if len(values) else None

    metrics = {
        "turns": len(ix),
        "covered_turns": len(eligible),
        "coverage": len(eligible) / len(ix) if len(ix) else None,
        "accuracy": average(correct),
        "all_turn_accuracy": int(correct.sum()) / len(ix) if len(ix) else None,
        "purchase_accuracy": average(correct[purchases]),
        "none_accuracy": average(correct[~purchases]),
        "always_none_accuracy": average(~purchases),
        "cross_entropy": loss_sum / len(eligible) if len(eligible) else None,
    }
    return metrics, predictions


def inspection(params, spec, names):
    groups = []
    for i, group in enumerate(spec["groups"]):
        units = []
        for unit in range(group["width"]):
            weights = params[f"w{i}"][:, unit]
            top = np.argsort(-np.abs(weights))[:6]
            units.append(
                {
                    "unit": unit,
                    "bias": float(params[f"b{i}"][unit]),
                    "output_weight": float(params[f"v{i}"][unit]),
                    "inputs": [
                        {"feature": names[group["indices"][j]], "weight": float(weights[j])}
                        for j in top
                    ],
                }
            )
        groups.append(
            {**group, "input_names": [names[j] for j in group["indices"]], "units": units}
        )
    return groups


def train(folder, config, meta, update, cancelled):
    import mlx.core as mx
    import mlx.optimizers as optim

    data, manifest = prepare_arrays(folder, meta, update, cancelled)
    names = manifest["feature_names"]
    spec = architecture(names, config.parameter_budget)
    if config.resume_student_id:
        metadata = json.loads((folder / "resume.json").read_text())
        if metadata["feature_names"] != names or metadata["architecture"] != spec:
            raise ValueError("Continuation feature schema or architecture changed")
        saved = load_weights(str(folder / "resume.npz"))
        mean, scale = saved["mean"], saved["scale"]
        initial = {k: saved[k].copy() for k in initialize(spec, config.seed)}
    else:
        mean, scale = normalization(data)
        initial = initialize(spec, config.seed)
    params = {k: mx.array(v) for k, v in initial.items()}
    optimizer = optim.AdamW(learning_rate=0.003, weight_decay=0.01)
    train_ix = np.flatnonzero((data["splits"] == 0) & (data["targets"] >= 0))
    rng = np.random.default_rng(config.seed)

    def loss(weights, x, mask, targets):
        scores = mx.where(mask, forward_mlx(weights, spec, x), -1e9)
        return mx.mean(mx.logsumexp(scores, axis=1) - scores[mx.arange(scores.shape[0]), targets])

    gradient = mx.value_and_grad(loss)
    history, best, best_epoch, best_score, stale = [], None, 0, -1.0, 0
    if config.resume_student_id:
        baseline, _ = evaluate(initial, spec, data, mean, scale, 1)
        best, best_score = initial, baseline["all_turn_accuracy"] or 0.0
        update(resumed_from=config.resume_student_id, initial_validation_accuracy=best_score)
    for epoch in range(1, config.epochs + 1):
        if cancelled():
            raise TrainingCancelled
        optimizer.learning_rate = 0.003 * (
            0.2 + 0.8 * (1 + np.cos(np.pi * (epoch - 1) / config.epochs)) / 2
        )
        order = rng.permutation(train_ix)
        total_loss = 0.0
        for start in range(0, len(order), 512):
            if cancelled():
                raise TrainingCancelled
            x, mask, y = batch(data, order[start : start + 512], mean, scale)
            value, gradients = gradient(params, mx.array(x), mx.array(mask), mx.array(y))
            params = optimizer.apply_gradients(gradients, params)
            mx.eval(params, optimizer.state, value)
            total_loss += float(value.item()) * len(y)
            if start % (512 * 30) == 0:
                update(
                    status="fitting",
                    epoch=epoch,
                    epochs=config.epochs,
                    parameter_count=spec["parameter_count"],
                    progress=0.88 + 0.09 * ((epoch - 1) + start / len(order)) / config.epochs,
                )

        def score_fn(x, weights=params):
            scores = forward_mlx(weights, spec, mx.array(x))
            mx.eval(scores)
            return np.asarray(scores)

        validation, _ = evaluate(None, spec, data, mean, scale, 1, score_fn=score_fn)
        score = validation["all_turn_accuracy"] or 0.0
        entry = {
            "epoch": epoch,
            "training_loss": total_loss / len(order),
            "validation_accuracy": score,
            "validation_purchase_accuracy": validation["purchase_accuracy"],
            "validation_loss": validation["cross_entropy"],
        }
        history.append(entry)
        if score > best_score + 1e-5:
            best = {k: np.asarray(v).copy() for k, v in params.items()}
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        update(history=history, best_epoch=best_epoch, epoch=epoch, validation_accuracy=score)
        if stale >= 8:
            break
    if best is None or cancelled():
        raise TrainingCancelled
    update(status="evaluating", progress=0.98)
    np.savez(folder / "miniastro.npz", mean=mean, scale=scale, **best)
    # Final metrics use NumPy, the deployed inference implementation.
    all_metrics, test_predictions = {}, None
    for split, name in enumerate(("train", "validation", "test")):
        all_metrics[name], prediction = evaluate(best, spec, data, mean, scale, split)
        if split == 2:
            test_predictions = prediction
    error_examples = []
    with (
        gzip.open(folder / "samples.jsonl.gz", "rb") as source,
        gzip.open(folder / "errors.jsonl.gz", "wb") as output,
    ):
        for index, line in enumerate(source):
            if data["splits"][index] != 2 or data["reasons"][index] == 2:
                continue
            row = orjson.loads(line)
            choice = int(test_predictions[index])
            predicted = int(data["ids"][index, choice]) if choice >= 0 else None
            if predicted == row["target"]:
                continue
            record = {
                "sample_index": index,
                "game": row["game"],
                "turn": row["turn"],
                "target": row["target"],
                "predicted": predicted,
                "excluded_reason": row["excluded_reason"],
                "total_trade": row["total_trade"],
                "spent": row["spent"],
                "observation": row["observation"],
            }
            output.write(orjson.dumps(record) + b"\n")
            if len(error_examples) < 30:
                error_examples.append({k: v for k, v in record.items() if k != "observation"})
    report = (
        f"MiniAstro v{VERSION}: {spec['parameter_count']:,} trainable parameters\n"
        "logit = linear candidate terms + economy + deck composition + timing + matchup + market\n"
        "Each subnetwork: output_weights · ReLU(input_weights · normalized_features + bias).\n"
        "Normalization: signed log1p, training-only mean/std, clipped to ±6.\n"
        f"Best validation epoch: {best_epoch}. Test games never select epochs.\n" + SELECTION
    )
    return {
        "student_type": "miniastro",
        "miniastro_version": VERSION,
        "architecture": spec,
        "parameter_count": spec["parameter_count"],
        "feature_names": names,
        "weights_path": str((folder / "miniastro.npz").resolve()),
        "dataset_sha256": manifest["dataset_sha256"],
        "metrics": all_metrics,
        "history": history,
        "best_epoch": best_epoch,
        "inspection": inspection(best, spec, names),
        "error_examples": error_examples,
        "report_text": report,
        "selection_rule": SELECTION,
    }


@lru_cache(maxsize=8)
def load_weights(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def recommend(artifact, observation, total_trade, spent):
    from .acquire_student import candidate_ids

    if artifact.get("miniastro_version") != VERSION:
        raise ValueError("Unsupported MiniAstro architecture version")
    weights = load_weights(artifact["weights_path"])
    ids = candidate_ids(observation, total_trade, spent)
    rows = [feature_vector(observation, cid, total_trade, spent) for cid in ids]
    if list(rows[0]) != artifact["feature_names"]:
        raise ValueError("MiniAstro feature schema no longer matches")
    x = transform(
        np.asarray([list(row.values()) for row in rows], dtype=np.float32),
        weights["mean"],
        weights["scale"],
    )
    scores, components, units = forward(weights, artifact["architecture"], x, detailed=True)
    probabilities = np.exp(scores - scores.max())
    probabilities /= probabilities.sum()
    candidates = []
    for index, cid in enumerate(ids):
        contributions = [
            {"name": name, "value": float(value)}
            for name, value in zip(("Card baseline", *GROUP_NAMES), components[index], strict=True)
        ]
        active_units = []
        for group, contribution in zip(GROUP_NAMES, units, strict=True):
            for unit in np.argsort(-np.abs(contribution[index]))[:3]:
                active_units.append(
                    {
                        "group": group,
                        "unit": int(unit),
                        "contribution": float(contribution[index, unit]),
                    }
                )
        candidates.append(
            {
                "card_id": cid,
                "name": CARD_BY_ID[cid].name if cid >= 0 else "None",
                "score": float(scores[index]),
                "policy_share": float(probabilities[index]),
                "contributions": contributions,
                "active_units": active_units,
            }
        )
    return {
        "student_type": "miniastro",
        "recommendation": candidates[int(scores.argmax())],
        "candidates": candidates,
        "selection_rule": SELECTION,
        "total_trade": total_trade,
        "spent": spent,
        "rules_version": artifact["config"]["rules_version"],
    }
