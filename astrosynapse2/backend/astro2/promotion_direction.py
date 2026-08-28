"""Weight-space guidance derived only from verified champion promotions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from .storage import Store


def _promotion_transitions(
    store: Store,
    source_checkpoint_id: str,
    *,
    maximum: int,
    recent_decay: float,
) -> list[dict[str, Any]]:
    source = store.checkpoint(source_checkpoint_id)
    source_games = int(source["games"])
    transitions: list[dict[str, Any]] = []
    for job in store.arena_jobs(
        limit=20_000,
        include_internal=True,
        run_id=source["run_id"],
        statuses=("complete",),
        promotion_tier="full",
        trainer_scheduled=True,
    ):
        result = job.get("result") or {}
        if not bool((result.get("promotion") or {}).get("promoted")):
            continue
        candidate = store.checkpoint(job["model_a"])
        champion = store.checkpoint(job["model_b"])
        if int(candidate["games"]) > source_games:
            continue
        if not Path(candidate["path"]).is_file() or not Path(champion["path"]).is_file():
            continue
        transitions.append(
            {
                "candidate": candidate,
                "champion": champion,
                "score": float(result.get("model_a_score", 0.5)),
            }
        )
    transitions.sort(key=lambda item: int(item["candidate"]["games"]))
    selected = transitions[-maximum:]
    for age, transition in enumerate(reversed(selected)):
        # Arena advantage contributes modestly; recency is the primary weight.
        advantage = max(0.005, float(transition["score"]) - 0.5)
        transition["weight"] = advantage * recent_decay**age
    return selected


def build_promotion_direction(
    store: Store,
    source_checkpoint_id: str,
    *,
    maximum_transitions: int = 5,
    minimum_sign_agreement: float = 0.60,
    recent_decay: float = 0.75,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build per-tensor unit-RMS directions from successful transitions.

    Every transition is normalized independently per tensor. Coordinates are
    retained only when the weighted signs agree sufficiently, preventing one
    large historical update or one large layer from dominating guidance.
    """

    transitions = _promotion_transitions(
        store,
        source_checkpoint_id,
        maximum=maximum_transitions,
        recent_decay=recent_decay,
    )
    if not transitions:
        raise ValueError(
            "promotion-direction refinement requires retained promoted champion transitions"
        )
    loaded: list[
        tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, dict[str, Any]]
    ] = []
    for transition in transitions:
        candidate = load_file(transition["candidate"]["path"])
        champion = load_file(transition["champion"]["path"])
        if candidate.keys() != champion.keys():
            continue
        loaded.append((candidate, champion, float(transition["weight"]), transition))
    if not loaded:
        raise ValueError("promoted transition weight archives are incompatible")

    total_weight = sum(weight for _candidate, _champion, weight, _transition in loaded)
    directions: dict[str, np.ndarray] = {}
    retained_coordinates = 0
    total_coordinates = 0
    reference_keys = loaded[-1][0].keys()
    for name in reference_keys:
        normalized_deltas: list[tuple[np.ndarray, float]] = []
        shape = loaded[-1][0][name].shape
        if any(
            candidate[name].shape != shape
            for candidate, _champion, _weight, _transition in loaded
        ):
            continue
        for candidate, champion, weight, _transition in loaded:
            delta = candidate[name].astype(np.float32) - champion[name].astype(np.float32)
            rms = float(np.sqrt(np.mean(np.square(delta), dtype=np.float64)))
            if np.isfinite(rms) and rms > 1e-12:
                normalized_deltas.append((delta / rms, weight))
        if not normalized_deltas:
            continue
        effective_weight = sum(weight for _delta, weight in normalized_deltas)
        consensus = sum(delta * weight for delta, weight in normalized_deltas) / effective_weight
        sign_agreement = (
            sum(np.sign(delta) * weight for delta, weight in normalized_deltas)
            / effective_weight
        )
        mask = np.abs(sign_agreement) >= minimum_sign_agreement
        consensus = np.where(mask, consensus, 0.0).astype(np.float32)
        consensus_rms = float(np.sqrt(np.mean(np.square(consensus), dtype=np.float64)))
        total_coordinates += int(consensus.size)
        retained_coordinates += int(np.count_nonzero(mask))
        if np.isfinite(consensus_rms) and consensus_rms > 1e-12:
            directions[name] = consensus / consensus_rms

    if not directions:
        raise ValueError("promoted transitions have no sufficiently consistent weight directions")
    metadata = {
        "schema_version": 1,
        "source_checkpoint_id": source_checkpoint_id,
        "transition_count": len(loaded),
        "maximum_transitions": maximum_transitions,
        "minimum_sign_agreement": minimum_sign_agreement,
        "recent_decay": recent_decay,
        "retained_coordinate_fraction": retained_coordinates / max(1, total_coordinates),
        "transitions": [
            {
                "candidate_id": item["candidate"]["id"],
                "champion_id": item["champion"]["id"],
                "candidate_games": int(item["candidate"]["games"]),
                "score": float(item["score"]),
                "weight": float(item["weight"]),
            }
            for _candidate, _champion, _weight, item in loaded
        ],
        "total_transition_weight": total_weight,
    }
    return directions, metadata


def save_promotion_direction(
    store: Store,
    source_checkpoint_id: str,
    path: str | Path,
    *,
    maximum_transitions: int = 5,
    minimum_sign_agreement: float = 0.60,
    recent_decay: float = 0.75,
) -> tuple[str, dict[str, Any]]:
    directions, metadata = build_promotion_direction(
        store,
        source_checkpoint_id,
        maximum_transitions=maximum_transitions,
        minimum_sign_agreement=minimum_sign_agreement,
        recent_decay=recent_decay,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(directions)
    payload["__metadata_json__"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(target, **payload)
    return str(target), metadata


def load_promotion_direction(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["__metadata_json__"].tolist()).decode("utf-8"))
        directions = {
            name: np.asarray(archive[name], dtype=np.float32)
            for name in archive.files
            if name != "__metadata_json__"
        }
    return directions, metadata
