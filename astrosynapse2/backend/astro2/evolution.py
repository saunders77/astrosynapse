"""Direct search of portable greedy policies using paired full-game returns.

The search population is training data, never promotion evidence. Antithetic
mutations operate on the existing learned features and produce ordinary actors;
there is no special inference wrapper to forget when deploying a winner.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

OPERATORS = ("main_output", "all_outputs", "head_mixture", "action_features")


def read_actor(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_actor(weights, actor_path):
    """Publish model and actor with identical tensors, compatible with Play/Arena."""
    actor_path = Path(actor_path)
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads(bytes(weights["__spec_json__"]).decode())
    tensors = {key: value for key, value in weights.items() if key != "__spec_json__"}
    if not all(np.isfinite(value).all() for value in tensors.values()):
        raise ValueError("nonfinite policy mutation")
    model = actor_path.with_name(actor_path.name.removesuffix(".actor.npz") + ".safetensors")
    temporary = actor_path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez(file, **weights)
    temporary.replace(actor_path)
    save_file(tensors, str(model) + ".tmp")
    Path(str(model) + ".tmp").replace(model)
    Path(str(model) + ".json").write_text(json.dumps(spec, indent=2))
    return str(model)


def mutate(parent, *, seed, operator, scale, sign=1):
    """Matched +/- directions; family-independent biases are deliberately excluded.

    Output mutations give the final greedy objective a low dimensional starting
    point. A low-rank action-feature mutation can escape that representation
    when output search stalls. Critics and state encoders are always untouched.
    """
    if operator not in OPERATORS or not np.isfinite(scale) or scale <= 0 or sign not in (-1, 1):
        raise ValueError("invalid mutation")
    weights = {key: value.copy() for key, value in parent.items()}
    spec = json.loads(bytes(weights["__spec_json__"]).decode())
    rng = np.random.default_rng(seed)
    heads, families = spec["bootstrap_heads"], spec["families"]
    if operator in {"main_output", "all_outputs"}:
        # A coherent perturbation across heads survives ensemble averaging.
        shared = rng.normal(size=weights["head_outputs.0.weight"].shape)
        if operator == "main_output":
            shared[1:] = 0
        for head in range(heads):
            key = f"head_outputs.{head}.weight"
            weights[key] += (sign * scale * shared).astype(np.float32)
    elif operator == "head_mixture":
        shifts = rng.normal(size=(heads, families))
        shifts -= shifts.mean(axis=0, keepdims=True)
        factors = np.exp(sign * scale * shifts)
        factors /= factors.mean(axis=0, keepdims=True)
        for head in range(heads):
            for suffix in ("weight", "bias"):
                key = f"head_outputs.{head}.{suffix}"
                factor = factors[head, :, None] if suffix == "weight" else factors[head]
                weights[key] *= factor.astype(np.float32)
    else:
        key = "action_in.weight"
        rows, cols = weights[key].shape
        left, right = rng.normal(size=rows), rng.normal(size=cols)
        weights[key] += (sign * scale * np.outer(left, right)).astype(np.float32)
    return weights


def paired_difference(candidate, parent):
    """Compare candidate and parent on the same exogenous game seed block.

    Descriptive standard errors guide training only; selection and optional
    stopping prevent interpreting these as promotion confidence intervals.
    """
    if [r["pair"] for r in candidate] != [r["pair"] for r in parent] or len(candidate) < 2:
        raise ValueError("comparison requires matching paired seed indices")
    differences = np.array(
        [(sum(a["scores"]) - sum(b["scores"])) / 2 for a, b in zip(candidate, parent, strict=True)]
    )
    return dict(
        gain=float(differences.mean()),
        standard_error=float(differences.std(ddof=1) / np.sqrt(len(differences))),
        changed_pairs=int(np.count_nonzero(differences)),
        pairs=len(differences),
    )


def recombine(parent, directions, advantages, *, max_directions=3):
    """Combine favored sides of antithetic directions into a bounded proposal.

    Screening returns supply weights, never evidence of improvement. The convex
    combination stays inside the sampled perturbations, including nonlinear
    head mixtures. A fresh selection block evaluates the resulting policy.
    """
    advantages = np.asarray(advantages, dtype=float)
    if len(directions) != len(advantages) or not np.isfinite(advantages).all():
        raise ValueError("directions require finite paired advantages")
    if max_directions < 1:
        raise ValueError("at least one direction is required")
    ranked = sorted(range(len(directions)), key=lambda i: -abs(advantages[i]))
    selected = [i for i in ranked[:max_directions] if advantages[i] != 0]
    if not selected:
        return None, []
    total = sum(abs(advantages[i]) for i in selected)
    recipe = [
        dict(direction=i, sign=1 if advantages[i] > 0 else -1, weight=abs(advantages[i]) / total)
        for i in selected
    ]
    result = {key: value.copy() for key, value in parent.items()}
    for key, value in parent.items():
        if key == "__spec_json__":
            continue
        delta = np.zeros_like(value, dtype=np.float64)
        for item in recipe:
            proposal = directions[item["direction"]][0 if item["sign"] == 1 else 1]
            delta += item["weight"] * (proposal[key].astype(np.float64) - value)
        result[key] = (value + delta).astype(value.dtype)
    return result, recipe


def choose_survivor(rows, *, minimum_gain=0.0, standard_errors=1.0):
    """Nominate a step using matched returns; this is not promotion evidence."""
    comparisons = [paired_difference(item, rows[0]) for item in rows]
    eligible = [
        i
        for i, comparison in enumerate(comparisons)
        if i
        and comparison["gain"] > max(minimum_gain, standard_errors * comparison["standard_error"])
    ]
    winner = max(eligible, key=lambda i: comparisons[i]["gain"]) if eligible else 0
    return winner, comparisons
