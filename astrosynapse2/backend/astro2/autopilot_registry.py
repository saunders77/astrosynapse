"""Immutable policy/critic discovery shared by GUI, arena, and campaign workers."""

from __future__ import annotations

import json
from pathlib import Path

from .encoding import DecisionFamily, Encoder


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def campaign_models(data_dir):
    result = []
    for path in (Path(data_dir) / "autopilot").glob("*/state.json"):
        state = read(path, {})
        if state.get("diagnostic"):
            continue
        for model in state.get("models", []):
            if not Path(model["actor_path"]).is_file():
                continue
            result.append(
                {
                    **model,
                    "is_champion": model["id"] == state.get("champion_id"),
                    "was_champion": model.get("was_champion", False),
                    "run_id": f"autopilot-{path.parent.name}",
                    "run_name": state.get("name", "Autopilot"),
                    "external": True,
                    "is_pinned": True,
                    "training_generation": 6,
                    "training_rules_version": 2,
                    "evaluation": model.get("evaluation", {}),
                }
            )
    return result


def critic_inventory(data_dir, models=()):
    result = []
    root = Path(data_dir)
    for path in (root / "critics").glob("*/critic.npz"):
        job = read(path.parent / "job.json", {})
        if job.get("status") == "complete":
            result.append(
                dict(
                    id=f"critic-{path.parent.name}",
                    path=str(path),
                    kind="independent",
                    label=f"Independent critic · {path.parent.name[:8]}",
                    encoder_version=job.get("encoder_version")
                    or job.get("config", {}).get("encoder_version"),
                    is_champion=False,
                )
            )
    for path in (root / "autopilot").glob("*/state.json"):
        state = read(path, {})
        if state.get("diagnostic"):
            continue
        result.extend(
            {**c, "is_champion": c["id"] == state.get("critic_id")}
            for c in state.get("critics", [])
            if Path(c["path"]).is_file()
        )
    for model in models:
        if model.get("actor_path") and Path(model["actor_path"]).is_file():
            metadata = read(str(model.get("path", "")) + ".json", {})
            if metadata.get("objective_version", 2) < 2:
                continue
            result.append(
                dict(
                    id="embedded:" + model["id"],
                    path=model["actor_path"],
                    kind="embedded",
                    label="Embedded value · " + model["label"],
                    encoder_version=metadata.get("encoder_version", model.get("encoder_version")),
                    is_champion=False,
                )
            )
    return result


def resolve_critic(data_dir, reference, models=()):
    if not reference:
        return None
    found = next((c for c in critic_inventory(data_dir, models) if c["id"] == reference), None)
    if found is None:
        raise ValueError("Unknown critic reference")
    return found


class ValuePredictor:
    """Each critic gets its own encoder, independent of the playing policy."""

    def __init__(self, spec):
        from .critic import IndependentCritic
        from .engine_encoding import EngineEncoder
        from .model import NumpyActor

        self.kind = spec["kind"]
        if self.kind == "independent":
            self.model, meta, _ = IndependentCritic.load(spec["path"])
            version = meta["encoder_version"]
            if meta["config"]["rules_version"] != 2:
                raise ValueError("Critic rules must match rules version 2")
            if len(self.model.mean) != Encoder(version=version).state_size + len(DecisionFamily):
                raise ValueError("Critic dimensions do not match its encoder")
        else:
            self.model = NumpyActor.load(spec["path"])
            version = self.model.spec.encoder_version
            if self.model.spec.objective_version < 2:
                raise ValueError("This legacy policy has no state-value critic")
        self.encoder = EngineEncoder(version=version)

    def predict(self, decision):
        import numpy as np

        encoded = self.encoder.encode_decision(decision.observation, decision)
        if self.kind == "independent":
            x = np.concatenate(
                (encoded.state, np.eye(len(DecisionFamily), dtype=np.float32)[int(encoded.family)])
            )
            return float(self.model.predict(x[None])[0])
        logits = self.model.predict_values(encoded.state, np.asarray([int(encoded.family)]))
        return float(np.mean(1 / (1 + np.exp(-np.clip(logits, -40, 40)))))
