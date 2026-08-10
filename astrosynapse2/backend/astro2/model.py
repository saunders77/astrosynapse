"""Apple-silicon action-value network and a dependency-light actor runtime.

The learner uses MLX/Metal. Self-play workers use the exact same forward pass
implemented with NumPy, avoiding one Metal context (and hundreds of MB of
framework state) per CPU actor process.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelSpec:
    state_size: int
    action_size: int
    families: int
    hidden_size: int = 256
    action_hidden_size: int = 128
    residual_blocks: int = 3
    bootstrap_heads: int = 3
    layer_norm_eps: float = 1e-5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelSpec:
        return cls(**payload)


def _mlx_modules():
    # Import lazily: MLX initializes Metal at import time. Keeping it out of
    # web/actor imports lets the API and CPU workers stay lightweight.
    import mlx.core as mx
    import mlx.nn as nn

    return mx, nn


def build_model(spec: ModelSpec):
    """Create an MLX model without importing MLX in CPU actor processes."""

    mx, nn = _mlx_modules()

    class ResidualBlock(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.norm = nn.LayerNorm(width, eps=spec.layer_norm_eps)
            self.fc1 = nn.Linear(width, width * 2)
            self.fc2 = nn.Linear(width * 2, width)

        def __call__(self, value):
            branch = self.fc2(nn.silu(self.fc1(self.norm(value))))
            return (value + branch) * (1.0 / math.sqrt(2.0))

    class ActionValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            h = spec.hidden_size
            ah = spec.action_hidden_size
            self.state_in = nn.Linear(spec.state_size, h)
            self.state_norm = nn.LayerNorm(h, eps=spec.layer_norm_eps)
            self.state_blocks = [ResidualBlock(h) for _ in range(spec.residual_blocks)]

            self.action_in = nn.Linear(spec.action_size, ah)
            self.action_norm = nn.LayerNorm(ah, eps=spec.layer_norm_eps)
            self.action_blocks = [ResidualBlock(ah)]

            self.fusion_in = nn.Linear(h + ah, h)
            self.fusion_norm = nn.LayerNorm(h, eps=spec.layer_norm_eps)
            self.fusion_blocks = [ResidualBlock(h) for _ in range(spec.residual_blocks)]
            self.output = nn.Linear(h, spec.families * spec.bootstrap_heads)

        def __call__(self, states, actions, families=None):
            state = nn.silu(self.state_norm(self.state_in(states)))
            for block in self.state_blocks:
                state = block(state)

            action = nn.silu(self.action_norm(self.action_in(actions)))
            for block in self.action_blocks:
                action = block(action)

            value = nn.silu(self.fusion_norm(self.fusion_in(mx.concatenate([state, action], axis=-1))))
            for block in self.fusion_blocks:
                value = block(value)
            all_logits = self.output(value).reshape(
                (-1, spec.families, spec.bootstrap_heads)
            )
            if families is None:
                return all_logits
            indices = mx.broadcast_to(
                families.astype(mx.int32).reshape((-1, 1, 1)),
                (families.shape[0], 1, spec.bootstrap_heads),
            )
            return mx.take_along_axis(all_logits, indices, axis=1).squeeze(axis=1)

    return ActionValueNet()


def bootstrap_bce_loss(
    model,
    states,
    actions,
    families,
    targets,
    bootstrap_mask,
    sample_weights,
):
    """Masked binary outcome loss for bootstrapped Deep Monte-Carlo heads."""

    mx, nn = _mlx_modules()
    logits = model(states, actions, families)
    target_matrix = mx.broadcast_to(targets.reshape((-1, 1)), logits.shape)
    losses = nn.losses.binary_cross_entropy(
        logits, target_matrix, with_logits=True, reduction="none"
    )
    weights = bootstrap_mask * sample_weights.reshape((-1, 1))
    loss = mx.sum(losses * weights) / mx.maximum(mx.sum(weights), mx.array(1.0))
    probabilities = mx.sigmoid(logits)
    prediction = mx.mean(probabilities, axis=1)
    accuracy = mx.mean((prediction >= 0.5) == (targets >= 0.5))
    uncertainty = mx.mean(mx.std(probabilities, axis=1))
    brier = mx.mean(mx.square(prediction - targets))
    target_variance = mx.var(targets)
    explained_variance = 1.0 - mx.var(targets - prediction) / mx.maximum(
        target_variance, mx.array(1e-6)
    )
    return loss, {
        "accuracy": accuracy,
        "brier": brier,
        "explained_variance": explained_variance,
        "mean_prediction": mx.mean(prediction),
        "uncertainty": uncertainty,
    }


def preference_ranking_loss(
    model,
    states,
    preferred_actions,
    disfavored_actions,
    families,
    *,
    margin: float = 1.0,
):
    """Soft pairwise loss for exact rules-derived tactical preferences."""

    mx, _nn = _mlx_modules()
    preferred_logits = model(states, preferred_actions, families)
    disfavored_logits = model(states, disfavored_actions, families)
    differences = preferred_logits - disfavored_logits
    losses = mx.logaddexp(mx.array(0.0), mx.array(float(margin)) - differences)
    return mx.mean(losses), {
        "preference_accuracy": mx.mean(differences > 0),
        "preference_margin_mean": mx.mean(differences),
    }


def export_actor(model, spec: ModelSpec, path: str | Path) -> Path:
    """Export MLX weights into a compact NumPy archive for CPU actors."""

    from mlx.utils import tree_flatten

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in tree_flatten(model.parameters())}
    arrays["__spec_json__"] = np.frombuffer(
        json.dumps(spec.as_dict(), sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(target, **arrays)
    return target


def save_model(model, spec: ModelSpec, path: str | Path) -> Path:
    """Save portable safetensors plus a small architecture sidecar."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(target))
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(spec.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def load_model(path: str | Path):
    target = Path(path)
    spec = ModelSpec.from_dict(
        json.loads(target.with_suffix(target.suffix + ".json").read_text(encoding="utf-8"))
    )
    model = build_model(spec)
    model.load_weights(str(target))
    return model, spec


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


class NumpyActor:
    """Exact inference-only mirror of :func:`build_model`."""

    def __init__(self, spec: ModelSpec, weights: dict[str, np.ndarray]):
        self.spec = spec
        self.weights = weights

    @classmethod
    def load(cls, path: str | Path) -> NumpyActor:
        with np.load(path, allow_pickle=False) as archive:
            spec_json = bytes(archive["__spec_json__"].tolist()).decode("utf-8")
            weights = {
                key: np.array(archive[key], dtype=np.float32, copy=True)
                for key in archive.files
                if key != "__spec_json__"
            }
        return cls(ModelSpec.from_dict(json.loads(spec_json)), weights)

    def _linear(self, value: np.ndarray, prefix: str) -> np.ndarray:
        return value @ self.weights[f"{prefix}.weight"].T + self.weights[f"{prefix}.bias"]

    def _norm(self, value: np.ndarray, prefix: str) -> np.ndarray:
        mean = value.mean(axis=-1, keepdims=True)
        variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (value - mean) / np.sqrt(variance + self.spec.layer_norm_eps)
        return normalized * self.weights[f"{prefix}.weight"] + self.weights[f"{prefix}.bias"]

    def _residual(self, value: np.ndarray, prefix: str) -> np.ndarray:
        branch = self._norm(value, f"{prefix}.norm")
        branch = _silu(self._linear(branch, f"{prefix}.fc1"))
        branch = self._linear(branch, f"{prefix}.fc2")
        return (value + branch) * (1.0 / math.sqrt(2.0))

    def _state_features(self, states: np.ndarray) -> np.ndarray:
        state = _silu(self._norm(self._linear(states, "state_in"), "state_norm"))
        for index in range(self.spec.residual_blocks):
            state = self._residual(state, f"state_blocks.{index}")
        return state

    def _action_features(self, actions: np.ndarray) -> np.ndarray:
        action = _silu(self._norm(self._linear(actions, "action_in"), "action_norm"))
        return self._residual(action, "action_blocks.0")

    def _interaction_logits(
        self,
        state_features: np.ndarray,
        action_features: np.ndarray,
    ) -> np.ndarray:
        value = np.concatenate([state_features, action_features], axis=-1)
        value = _silu(self._norm(self._linear(value, "fusion_in"), "fusion_norm"))
        for index in range(self.spec.residual_blocks):
            value = self._residual(value, f"fusion_blocks.{index}")
        return self._linear(value, "output").reshape(
            (-1, self.spec.families, self.spec.bootstrap_heads)
        )

    def predict(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        families: np.ndarray,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        families = np.asarray(families, dtype=np.int64)
        if states.ndim == 1:
            states = states[None, :]
        if actions.ndim == 1:
            actions = actions[None, :]
        if families.ndim == 0:
            families = families[None]

        all_logits = self._interaction_logits(
            self._state_features(states), self._action_features(actions)
        )
        return all_logits[np.arange(all_logits.shape[0]), families]

    def predict_options(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        family: int,
    ) -> np.ndarray:
        """Score one decision while computing its expensive state trunk once."""

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        state_batch = np.asarray(state, dtype=np.float32).reshape((1, -1))
        state_features = self._state_features(state_batch)
        state_features = np.repeat(state_features, len(actions), axis=0)
        all_logits = self._interaction_logits(state_features, self._action_features(actions))
        return all_logits[:, family]

    def choose(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        family: int,
        *,
        head: int | None = None,
        epsilon: float = 0.0,
        exploration_top_k: int = 3,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, np.ndarray]:
        generator = rng or np.random.default_rng()
        logits = self.predict_options(state, actions, family)
        values = logits.mean(axis=1) if head is None else logits[:, head]
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
        if exploration_top_k < 1:
            raise ValueError("exploration_top_k must be positive")
        if len(actions) == 1:
            # This can happen after the model-policy dominance mask removes a
            # legal END_TURN.  The action choice is forced, but its estimated
            # outcome is not certainty and is used as a next-decision target.
            return 0, probabilities
        if epsilon > 0 and generator.random() < epsilon:
            # Explore among plausible actions instead of uniformly sampling a
            # potentially catastrophic tail action.  Bootstrap-head selection
            # supplies the broader, trajectory-coherent exploration signal.
            count = min(exploration_top_k, len(values))
            top = np.argpartition(values, -count)[-count:]
            return int(generator.choice(top)), probabilities
        best = np.flatnonzero(values == values.max())
        return int(generator.choice(best)), probabilities
