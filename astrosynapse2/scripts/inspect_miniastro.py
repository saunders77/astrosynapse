"""Export every MiniAstro parameter and inspect scores on held-out positions."""

import csv
import json
from pathlib import Path

import numpy as np
from astro2.cards import CARD_BY_ID
from astro2.miniastro import forward, transform


def main():
    root = Path("data/acquire_students")
    folder = root / "669746289fef4e71ad5caf056559d2e6"
    artifact = json.loads((folder / "student.json").read_text())
    spec, names = artifact["architecture"], artifact["feature_names"]
    cache = root / "feature_cache" / f"{artifact['dataset_sha256']}-miniastro-v1"
    manifest = json.loads((cache / "manifest.json").read_text())
    with np.load(folder / "miniastro.npz") as archive:
        weights = dict(archive)
    with np.load(cache / "choices.npz") as archive:
        data = dict(archive)
    raw = np.memmap(
        cache / "features.bin", mode="r", dtype=np.float32, shape=tuple(manifest["shape"])
    )
    test = np.flatnonzero((data["splits"] == 2) & (data["reasons"] == 0))
    selected = np.random.default_rng(20261005).choice(test, min(20000, len(test)), replace=False)
    pointers = data["pointers"][selected]
    mask = pointers >= 0
    x_raw = raw[np.maximum(pointers, 0)]
    x = transform(x_raw, weights["mean"], weights["scale"])
    scores, terms, units = forward(weights, spec, x, detailed=True)
    scores = np.where(mask, scores, -1e9)
    winners = scores.argmax(axis=1)
    runner = np.argsort(-scores, axis=1)[:, 1]
    multiple = mask.sum(axis=1) > 1
    ix = np.arange(len(selected))
    ids = data["ids"][selected]
    def label(cid):
        return CARD_BY_ID[int(cid)].name if cid >= 0 else "None"
    records = []
    for j, index in enumerate(spec["base_indices"]):
        records.append(["Linear", "", "input", names[index], float(weights["baseline"][j])])
    stats = []
    for gi, group in enumerate(spec["groups"]):
        for ui in range(group["width"]):
            w = weights[f"w{gi}"][:, ui]
            b, v = float(weights[f"b{gi}"][ui]), float(weights[f"v{gi}"][ui])
            for j, index in enumerate(group["indices"]):
                records.append([group["name"], ui + 1, "input", names[index], float(w[j])])
            records.extend(
                [[group["name"], ui + 1, "bias", "", b], [group["name"], ui + 1, "output", "", v]]
            )
            activation = x[..., group["indices"]] @ w + b
            contribution = units[gi][..., ui]
            removed = np.where(mask, scores - contribution, -1e9)
            both = (activation[ix, winners] > 0) & (activation[ix, runner] > 0)
            opposite = (activation[ix, winners] > 0) != (activation[ix, runner] > 0)
            stats.append(
                {
                    "name": f"{group['name']} {ui + 1}",
                    "bias": b,
                    "output": v,
                    "active_fraction": float((activation[mask] > 0).mean()),
                    "winner_flip_if_removed": float(
                        (removed.argmax(axis=1)[multiple] != winners[multiple]).mean()
                    ),
                    "winner_runner_both_active": float(both[multiple].mean()),
                    "winner_runner_opposite": float(opposite[multiple].mean()),
                    "mean_abs_winner_runner_contribution": float(
                        np.abs(contribution[ix, winners] - contribution[ix, runner])[
                            multiple
                        ].mean()
                    ),
                }
            )
    assert len(records) == artifact["parameter_count"] == 956
    with (folder / "all-956-weights.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "unit", "parameter", "feature", "value"])
        writer.writerows(records)
    with (folder / "feature-normalization.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["feature", "signed_log1p_mean", "signed_log1p_std", "clip_min", "clip_max"]
        )
        for name, mean, scale in zip(names, weights["mean"], weights["scale"], strict=True):
            writer.writerow([name, float(mean), float(scale), -6, 6])
    np.testing.assert_allclose(scores[mask], terms.sum(axis=-1)[mask], atol=1e-5)
    slopes = {}
    for feature in [
        "Turn number",
        "Opponent authority",
        "Your authority",
        "Your total deck size (all zones)",
        "Candidate faction cards in your deck",
        "Candidate faction cards in your discard",
        "More expensive affordable row alternatives",
        "Trade left after buying candidate",
    ]:
        index = names.index(feature)
        parts = []
        if index in spec["base_indices"]:
            parts.append(["Linear", float(weights["baseline"][spec["base_indices"].index(index)])])
        for gi, g in enumerate(spec["groups"]):
            if index not in g["indices"]:
                continue
            k = g["indices"].index(index)
            for ui in range(g["width"]):
                parts.append(
                    [
                        f"{g['name']} {ui + 1}",
                        float(weights[f"v{gi}"][ui] * weights[f"w{gi}"][k, ui]),
                    ]
                )
        slopes[feature] = parts

    def example(row):
        valid = mask[row]
        relevant = [
            "Turn number",
            "Your authority",
            "Opponent authority",
            "Total trade generated this turn",
            "Trade already spent this turn",
            "Trade currently in pool",
            "Your total deck size (all zones)",
            "Your remaining starters",
            "Your deck printed trade",
            "Your deck printed combat",
        ]
        return {
            "sample_index": int(selected[row]),
            "context": {name: float(x_raw[row, 0, names.index(name)]) for name in relevant},
            "teacher": label(ids[row, data["targets"][selected[row]]]),
            "choices": [
                {
                    "card": label(cid),
                    "score": round(float(score), 4),
                    "terms": [round(float(t), 4) for t in term],
                    "units": [round(float(z), 4) for u in units for z in u[row, col]],
                    "own_faction_cards": float(
                        x_raw[row, col, names.index("Candidate faction cards in your deck")]
                    ),
                }
                for col, (cid, score, term) in enumerate(
                    zip(ids[row], scores[row], terms[row], strict=True)
                )
                if valid[col]
            ],
        }

    pairs = {}
    for first, second in [
        ("Explorer", "None"),
        ("Trade Bot", "Explorer"),
        ("Cutter", "Explorer"),
        ("Blob Fighter", "Trade Bot"),
        ("Imperial Fighter", "Federation Shuttle"),
    ]:
        aid = next((cid for cid, c in CARD_BY_ID.items() if c.name == first), -1)
        bid = next((cid for cid, c in CARD_BY_ID.items() if c.name == second), -1)
        eligible = np.flatnonzero((ids == aid).any(axis=1) & (ids == bid).any(axis=1))
        if not len(eligible):
            continue
        ai = (ids[eligible] == aid).argmax(axis=1)
        bi = (ids[eligible] == bid).argmax(axis=1)
        delta = scores[eligible, ai] - scores[eligible, bi]
        pairs[f"{first} minus {second}"] = {
            "count": len(eligible),
            "first_preferred_fraction": float((delta > 0).mean()),
            "examples": [example(eligible[np.argmin(delta)]), example(eligible[np.argmax(delta)])],
        }
    # Pure turn-number sweep on a fixed observed candidate set: exact internal
    # sensitivity, explicitly not a physically replayed game counterfactual.
    sweeps = []
    for row in np.flatnonzero(multiple)[:1000]:
        probe = np.repeat(x_raw[row : row + 1], 4, axis=0)
        probe[:, :, names.index("Turn number")] = np.array([5, 15, 30, 50])[:, None]
        logits = forward(weights, spec, transform(probe, weights["mean"], weights["scale"]))[0]
        chosen = np.where(mask[row], logits, -1e9).argmax(axis=1)
        if len(set(ids[row, chosen])) > 1:
            sweeps.append(
                {
                    "original": example(row),
                    "turns": [5, 15, 30, 50],
                    "choices": [label(ids[row, j]) for j in chosen],
                }
            )
            if len(sweeps) == 8:
                break
    result = {
        "student": folder.name,
        "positions": len(selected),
        "multiple_choice_positions": int(multiple.sum()),
        "units": stats,
        "normalized_feature_slopes_when_active": slopes,
        "pairs": pairs,
        "turn_sweeps": sweeps,
    }
    (folder / "weight-inspection.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"pairs", "turn_sweeps"}}, indent=2)
    )
    for name, value in pairs.items():
        print(name, value["count"], value["first_preferred_fraction"])
        for e in value["examples"]:
            print(json.dumps(e))
    print("SWEEPS", json.dumps(sweeps))


if __name__ == "__main__":
    main()
