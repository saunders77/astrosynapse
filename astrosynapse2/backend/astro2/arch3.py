"""Lossless architecture migration and discovery of independent arch3 lineages."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .encoding import Encoder
from .experiment_control import atomic_json, sha256
from .model import NumpyActor


def port_actor(source: str | Path, destination: str | Path) -> dict:
    source, destination = Path(source), Path(destination)
    actor = NumpyActor.load(source)
    if actor.spec.encoder_version not in (2, 3):
        raise ValueError("Porting requires an arch2 or arch3 actor")
    old = Encoder(version=actor.spec.encoder_version)
    encoder = Encoder(version=3)
    if (actor.spec.state_size, actor.spec.action_size) != (old.state_size, old.action_size):
        raise ValueError("Source encoder dimensions do not match its architecture")
    spec = replace(actor.spec, state_size=encoder.state_size, encoder_version=3)
    weights = dict(actor.weights)
    weights["state_in.weight"] = np.pad(
        weights["state_in.weight"], ((0, 0), (0, spec.state_size - actor.spec.state_size))
    )
    destination.mkdir(parents=True, exist_ok=False)
    model = destination / "source.safetensors"
    save_file(weights, str(model))
    atomic_json(Path(str(model) + ".json"), spec.as_dict())
    arrays = {
        **weights,
        "__spec_json__": np.frombuffer(json.dumps(spec.as_dict()).encode(), dtype=np.uint8),
    }
    np.savez_compressed(destination / "source.actor.npz", **arrays)
    lineage = dict(
        id=destination.name,
        architecture="arch3",
        source_actor=str(source.resolve()),
        source_sha256=sha256(source),
        source_encoder_version=actor.spec.encoder_version,
        model=str(model.resolve()),
        actor=str((destination / "source.actor.npz").resolve()),
    )
    atomic_json(destination / "lineage.json", lineage)
    return lineage


def create_port(store, model_id):
    from .arena import resolve_model

    source = resolve_model(store, model_id)
    destination = store.path.parent / "arch3" / uuid.uuid4().hex
    destination.parent.mkdir(parents=True, exist_ok=True)
    lineage = port_actor(source.actor_path, destination)
    lineage.update(source_model_id=model_id, label=source.label + " · arch3")
    atomic_json(destination / "lineage.json", lineage)
    return next(m for m in arch3_models(store.path.parent) if m["actor_path"] == lineage["actor"])


def arch3_models(data_dir):
    result = []
    root = Path(data_dir) / "arch3"
    for path in root.glob("*/lineage.json"):
        lineage = json.loads(path.read_text())
        for actor in [path.parent / "source.actor.npz", *path.parent.glob("policy/g*.actor.npz")]:
            if not actor.is_file():
                continue
            checkpoint = actor.name.removesuffix(".actor.npz")
            model = actor.with_name(checkpoint + ".safetensors")
            result.append(
                dict(
                    id=f"arch3-{path.parent.name}-{checkpoint}",
                    run_id=f"arch3-{path.parent.name}",
                    run_name=lineage.get("label", "arch3"),
                    label=f"{lineage.get('label', 'arch3')} · {checkpoint}",
                    checkpoint_name=checkpoint,
                    generation=None,
                    path=str(model),
                    actor_path=str(actor),
                    games=int(checkpoint[1:])
                    if checkpoint.startswith("g") and checkpoint[1:].isdigit()
                    else 0,
                    created_at=actor.stat().st_mtime,
                    parent_id=lineage.get("source_model_id"),
                    is_champion=False,
                    was_champion=False,
                    is_pinned=True,
                    external=True,
                    training_generation=6,
                    encoder_version=3,
                    architecture="arch3",
                    evaluation={},
                )
            )
    return result
