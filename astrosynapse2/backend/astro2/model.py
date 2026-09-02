"""Apple-silicon action-value network and a dependency-light actor runtime.

The learner uses MLX/Metal. Self-play workers use the exact same forward pass
implemented with NumPy, avoiding one Metal context (and hundreds of MB of
framework state) per CPU actor process.
"""

from __future__ import annotations

import json
import math
import uuid
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
    # Keep new optional fields after the original positional parameters so
    # third-party callers that constructed ModelSpec positionally remain
    # source-compatible. Internal code should still prefer keyword arguments.
    encoder_version: int = 1
    # Version 1 is the historical chosen-action outcome model. Version 2 uses
    # normalized legal-action policy logits plus a separate state-value head.
    objective_version: int = 1

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
            if spec.objective_version >= 2:
                # A small independent adapter/output path per head prevents the
                # shared trunk from collapsing every bootstrap hypothesis into
                # an effectively identical policy.
                self.head_blocks = [ResidualBlock(h) for _ in range(spec.bootstrap_heads)]
                self.head_outputs = [
                    nn.Linear(h, spec.families) for _ in range(spec.bootstrap_heads)
                ]
                self.value_output = nn.Linear(h, spec.families * spec.bootstrap_heads)
            else:
                self.output = nn.Linear(h, spec.families * spec.bootstrap_heads)

        def state_features(self, states):
            state = nn.silu(self.state_norm(self.state_in(states)))
            for block in self.state_blocks:
                state = block(state)
            return state

        def state_values_from_features(self, state, families):
            if spec.objective_version < 2:
                raise RuntimeError("separate state values require objective_version >= 2")
            values = self.value_output(state).reshape((-1, spec.families, spec.bootstrap_heads))
            indices = mx.broadcast_to(
                families.astype(mx.int32).reshape((-1, 1, 1)),
                (families.shape[0], 1, spec.bootstrap_heads),
            )
            return mx.take_along_axis(values, indices, axis=1).squeeze(axis=1)

        def state_values(self, states, families):
            return self.state_values_from_features(self.state_features(states), families)

        def action_logits_from_features(self, state, actions, families=None):
            action = nn.silu(self.action_norm(self.action_in(actions)))
            for block in self.action_blocks:
                action = block(action)

            value = nn.silu(
                self.fusion_norm(self.fusion_in(mx.concatenate([state, action], axis=-1)))
            )
            for block in self.fusion_blocks:
                value = block(value)
            if spec.objective_version >= 2:
                all_logits = mx.stack(
                    [
                        output(block(value))
                        for block, output in zip(self.head_blocks, self.head_outputs, strict=True)
                    ],
                    axis=-1,
                )
            else:
                all_logits = self.output(value).reshape((-1, spec.families, spec.bootstrap_heads))
            if families is None:
                return all_logits
            indices = mx.broadcast_to(
                families.astype(mx.int32).reshape((-1, 1, 1)),
                (families.shape[0], 1, spec.bootstrap_heads),
            )
            return mx.take_along_axis(all_logits, indices, axis=1).squeeze(axis=1)

        def __call__(self, states, actions, families=None):
            return self.action_logits_from_features(self.state_features(states), actions, families)

    return ActionValueNet()


def actor_critic_policy_loss(
    model,
    states,
    legal_actions,
    legal_mask,
    selected_indices,
    families,
    targets,
    behavior_probabilities,
    bootstrap_mask,
    sample_weights,
    *,
    value_loss_weight: float = 0.5,
    entropy_weight: float = 0.01,
    importance_clip: float = 2.0,
    search_policy_targets=None,
    search_mask=None,
    search_values=None,
    search_valid=None,
    search_policy_loss_weight: float = 0.0,
    search_value_loss_weight: float = 0.0,
    behavior_policy_loss_weight: float = 1.0,
    search_loss_reference_positions: int = 1,
    actor_sample_weights=None,
    actor_advantages=None,
    actor_advantage_valid=None,
    reference_model=None,
    reference_policy_kl_weight: float = 0.0,
    collection_policy_probabilities=None,
    behavior_heads=None,
    importance_groups: dict[str, Any] | None = None,
):
    """Game-balanced off-policy actor-critic loss over complete legal sets."""

    mx, nn = _mlx_modules()
    batch, options, action_size = legal_actions.shape
    state_features = model.state_features(states)
    repeated_state_features = mx.broadcast_to(
        state_features[:, None, :], (batch, options, state_features.shape[-1])
    ).reshape((batch * options, state_features.shape[-1]))
    repeated_families = mx.broadcast_to(families[:, None], (batch, options)).reshape(
        (batch * options,)
    )
    policy_logits = model.action_logits_from_features(
        repeated_state_features,
        legal_actions.reshape((batch * options, action_size)),
        repeated_families,
    ).reshape((batch, options, -1))
    masked_logits = mx.where(legal_mask[:, :, None] > 0, policy_logits, -1e9)
    log_probs = masked_logits - mx.logsumexp(masked_logits, axis=1, keepdims=True)
    probabilities = mx.exp(log_probs) * legal_mask[:, :, None]
    selected = (mx.arange(options)[None, :] == selected_indices.astype(mx.int32)[:, None]).astype(
        policy_logits.dtype
    )[:, :, None]
    chosen_log_probs = mx.sum(log_probs * selected, axis=1)
    chosen_probabilities = mx.sum(probabilities * selected, axis=1)

    value_logits = model.state_values_from_features(state_features, families)
    values = mx.sigmoid(value_logits)
    target_matrix = mx.broadcast_to(targets[:, None], values.shape)
    monte_carlo_advantages = target_matrix - values
    if actor_advantages is not None:
        supplied_advantages = mx.broadcast_to(actor_advantages[:, None], values.shape)
        valid_advantages = (
            mx.ones_like(actor_advantages)
            if actor_advantage_valid is None
            else actor_advantage_valid
        )
        advantages = mx.where(
            valid_advantages[:, None] > 0,
            supplied_advantages,
            monte_carlo_advantages,
        )
    else:
        valid_advantages = mx.zeros_like(targets)
        advantages = monte_carlo_advantages
    advantages = mx.stop_gradient(advantages)
    behavior = mx.maximum(behavior_probabilities[:, None], mx.array(1e-6))
    raw_ratios = chosen_probabilities / behavior
    ratios = mx.stop_gradient(mx.minimum(raw_ratios, mx.array(float(importance_clip))))
    weights = bootstrap_mask * sample_weights[:, None]
    denominator = mx.maximum(mx.sum(weights), mx.array(1.0))
    resolved_actor_sample_weights = (
        mx.ones_like(sample_weights) if actor_sample_weights is None else actor_sample_weights
    )
    actor_weights = weights * resolved_actor_sample_weights[:, None]
    actor_denominator = mx.maximum(mx.sum(actor_weights), mx.array(1.0))
    policy_loss = (
        -mx.sum(actor_weights * ratios * advantages * chosen_log_probs) / actor_denominator
    )
    value_losses = nn.losses.binary_cross_entropy(
        value_logits, target_matrix, with_logits=True, reduction="none"
    )
    value_loss = mx.sum(weights * value_losses) / denominator
    entropy_by_head = -mx.sum(probabilities * log_probs, axis=1)
    entropy = mx.sum(actor_weights * entropy_by_head) / actor_denominator
    legal_counts = mx.maximum(mx.sum(legal_mask, axis=1), mx.array(2.0))
    normalized_entropy = (
        mx.sum(actor_weights * (entropy_by_head / mx.log(legal_counts)[:, None]))
        / actor_denominator
    )
    reference_policy_kl = mx.array(0.0)
    reference_policy_head_kl = mx.array(0.0)
    if reference_model is not None and float(reference_policy_kl_weight) > 0:
        reference_state_features = reference_model.state_features(states)
        reference_repeated_state_features = mx.broadcast_to(
            reference_state_features[:, None, :],
            (batch, options, reference_state_features.shape[-1]),
        ).reshape((batch * options, reference_state_features.shape[-1]))
        reference_logits = reference_model.action_logits_from_features(
            reference_repeated_state_features,
            legal_actions.reshape((batch * options, action_size)),
            repeated_families,
        ).reshape((batch, options, -1))
        reference_masked_logits = mx.where(legal_mask[:, :, None] > 0, reference_logits, -1e9)
        reference_log_probs = reference_masked_logits - mx.logsumexp(
            reference_masked_logits, axis=1, keepdims=True
        )
        reference_probabilities = mx.stop_gradient(mx.exp(reference_log_probs))
        reference_policy_head_kl = (
            mx.sum(
                actor_weights
                * mx.sum(reference_probabilities * (reference_log_probs - log_probs), axis=1)
            )
            / actor_denominator
        )
        # Inference deploys the arithmetic mean of head logits, not a sampled
        # bootstrap head. Regularize that exact distribution so a harmless
        # permutation or specialization of individual heads is not penalized.
        deployment_logits = mx.mean(policy_logits, axis=2)
        deployment_masked_logits = mx.where(legal_mask > 0, deployment_logits, -1e9)
        deployment_log_probs = deployment_masked_logits - mx.logsumexp(
            deployment_masked_logits, axis=1, keepdims=True
        )
        reference_deployment_logits = mx.mean(reference_logits, axis=2)
        reference_deployment_masked_logits = mx.where(
            legal_mask > 0, reference_deployment_logits, -1e9
        )
        reference_deployment_log_probs = reference_deployment_masked_logits - mx.logsumexp(
            reference_deployment_masked_logits, axis=1, keepdims=True
        )
        reference_deployment_probabilities = mx.stop_gradient(
            mx.exp(reference_deployment_log_probs)
        )
        deployment_weights = sample_weights * resolved_actor_sample_weights
        deployment_denominator = mx.maximum(mx.sum(deployment_weights), mx.array(1.0))
        reference_policy_kl = (
            mx.sum(
                deployment_weights
                * mx.sum(
                    reference_deployment_probabilities
                    * (reference_deployment_log_probs - deployment_log_probs),
                    axis=1,
                )
            )
            / deployment_denominator
        )
    search_policy_loss = mx.array(0.0)
    search_value_loss = mx.array(0.0)
    searched_fraction = mx.array(0.0)
    search_weight_scale = mx.array(0.0)
    if search_policy_targets is not None and search_valid is not None:
        resolved_search_mask = legal_mask if search_mask is None else search_mask * legal_mask
        searched_logits = mx.where(resolved_search_mask[:, :, None] > 0, policy_logits, -1e9)
        searched_log_probs = searched_logits - mx.logsumexp(searched_logits, axis=1, keepdims=True)
        target_distribution = search_policy_targets[:, :, None]
        search_value_weights = bootstrap_mask * search_valid[:, None] * sample_weights[:, None]
        search_policy_weights = search_value_weights * resolved_actor_sample_weights[:, None]
        search_policy_denominator = mx.maximum(mx.sum(search_policy_weights), mx.array(1.0))
        search_value_denominator = mx.maximum(mx.sum(search_value_weights), mx.array(1.0))
        searched_count = mx.sum(search_valid * resolved_actor_sample_weights)
        search_weight_scale = mx.minimum(
            searched_count / mx.array(float(max(1, search_loss_reference_positions))),
            mx.array(1.0),
        )
        cross_entropy = -mx.sum(target_distribution * searched_log_probs, axis=1)
        search_policy_loss = (
            mx.sum(search_policy_weights * cross_entropy) / search_policy_denominator
        )
        if search_values is not None:
            searched_value_targets = mx.broadcast_to(search_values[:, None], value_logits.shape)
            searched_value_losses = nn.losses.binary_cross_entropy(
                value_logits,
                searched_value_targets,
                with_logits=True,
                reduction="none",
            )
            search_value_loss = (
                mx.sum(search_value_weights * searched_value_losses) / search_value_denominator
            )
        searched_fraction = mx.mean(search_valid)
    loss = (
        float(behavior_policy_loss_weight) * policy_loss
        + float(value_loss_weight) * value_loss
        - float(entropy_weight) * entropy
        + float(reference_policy_kl_weight) * reference_policy_kl
        + float(search_policy_loss_weight) * search_weight_scale * search_policy_loss
        + float(search_value_loss_weight) * search_weight_scale * search_value_loss
    )
    prediction = mx.mean(values, axis=1)
    diagnostics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "policy_entropy": entropy,
        "normalized_policy_entropy": normalized_entropy,
        "reference_policy_kl": reference_policy_kl,
        "reference_policy_head_kl": reference_policy_head_kl,
        "search_policy_loss": search_policy_loss,
        "search_value_loss": search_value_loss,
        "searched_fraction": searched_fraction,
        "searched_count": (mx.sum(search_valid) if search_valid is not None else mx.array(0.0)),
        "search_weight_scale": search_weight_scale,
        "weighted_policy_loss": float(behavior_policy_loss_weight) * policy_loss,
        "weighted_value_loss": float(value_loss_weight) * value_loss,
        "weighted_entropy_loss": -float(entropy_weight) * entropy,
        "weighted_reference_policy_kl": (float(reference_policy_kl_weight) * reference_policy_kl),
        "actor_sample_fraction": mx.mean(resolved_actor_sample_weights),
        "actor_supplied_advantage_fraction": mx.mean(valid_advantages),
        "actor_advantage_mean": (mx.sum(actor_weights * advantages) / actor_denominator),
        "actor_advantage_rms": mx.sqrt(
            mx.sum(actor_weights * mx.square(advantages)) / actor_denominator
        ),
        "weighted_search_policy_loss": (
            float(search_policy_loss_weight) * search_weight_scale * search_policy_loss
        ),
        "weighted_search_value_loss": (
            float(search_value_loss_weight) * search_weight_scale * search_value_loss
        ),
        "value_brier": mx.mean(mx.square(prediction - targets)),
        "value_accuracy": mx.mean((prediction >= 0.5) == (targets >= 0.5)),
        "mean_importance_ratio": mx.sum(weights * ratios) / denominator,
        "importance_clip_fraction": mx.sum(weights * (raw_ratios > float(importance_clip)))
        / denominator,
        "uncertainty": mx.sum(mx.std(probabilities, axis=2) * legal_mask)
        / mx.maximum(mx.sum(legal_mask), mx.array(1.0)),
    }
    collection_known = None
    collection_ratios = None
    collection_log_drift = None
    if collection_policy_probabilities is not None and behavior_heads is not None:
        resolved_heads = behavior_heads.astype(mx.int32)
        head_indices = mx.maximum(resolved_heads, mx.array(0))[:, None]
        owner_probabilities = mx.take_along_axis(
            chosen_probabilities, head_indices, axis=1
        ).squeeze(axis=1)
        current_collection_probabilities = mx.where(
            resolved_heads >= 0,
            owner_probabilities,
            mx.mean(chosen_probabilities, axis=1),
        )
        collection_known = collection_policy_probabilities > 0
        collection_denominator = mx.maximum(collection_policy_probabilities, mx.array(1e-6))
        collection_ratios = current_collection_probabilities / collection_denominator
        collection_log_drift = mx.abs(
            mx.log(mx.maximum(current_collection_probabilities, mx.array(1e-6)))
            - mx.log(collection_denominator)
        )
        known_weights = collection_known.astype(weights.dtype)
        known_denominator = mx.maximum(mx.sum(known_weights), mx.array(1.0))
        diagnostics["collection_policy_ratio"] = (
            mx.sum(known_weights * collection_ratios) / known_denominator
        )
        diagnostics["collection_policy_abs_log_drift"] = (
            mx.sum(known_weights * collection_log_drift) / known_denominator
        )
        diagnostics["collection_policy_samples"] = mx.sum(known_weights)
    for head in range(int(values.shape[1])):
        head_weights = weights[:, head]
        head_denominator = mx.maximum(mx.sum(head_weights), mx.array(1.0))
        diagnostics[f"importance_ratio_head_{head}"] = (
            mx.sum(head_weights * ratios[:, head]) / head_denominator
        )
    if importance_groups:
        for name, group_mask in importance_groups.items():
            resolved_group = group_mask.astype(weights.dtype)
            group_weights = weights * resolved_group[:, None]
            group_denominator = mx.maximum(mx.sum(group_weights), mx.array(1.0))
            group_actor_weights = actor_weights * resolved_group[:, None]
            group_actor_denominator = mx.maximum(mx.sum(group_actor_weights), mx.array(1.0))
            diagnostics[f"importance_ratio_{name}"] = (
                mx.sum(group_weights * ratios) / group_denominator
            )
            diagnostics[f"importance_samples_{name}"] = mx.sum(resolved_group)
            diagnostics[f"actor_samples_{name}"] = mx.sum(
                resolved_group * resolved_actor_sample_weights
            )
            diagnostics[f"policy_loss_{name}"] = (
                -mx.sum(group_actor_weights * ratios * advantages * chosen_log_probs)
                / group_actor_denominator
            )
            diagnostics[f"value_loss_{name}"] = (
                mx.sum(group_weights * value_losses) / group_denominator
            )
            diagnostics[f"advantage_{name}"] = (
                mx.sum(group_actor_weights * advantages) / group_actor_denominator
            )
            group_predictions = mx.broadcast_to(prediction[:, None], values.shape)
            diagnostics[f"value_brier_{name}"] = (
                mx.sum(group_weights * mx.square(group_predictions - target_matrix))
                / group_denominator
            )
            if collection_known is not None:
                collection_group_weights = resolved_group * collection_known.astype(weights.dtype)
                collection_group_denominator = mx.maximum(
                    mx.sum(collection_group_weights), mx.array(1.0)
                )
                diagnostics[f"collection_policy_ratio_{name}"] = (
                    mx.sum(collection_group_weights * collection_ratios)
                    / collection_group_denominator
                )
                diagnostics[f"collection_policy_abs_log_drift_{name}"] = (
                    mx.sum(collection_group_weights * collection_log_drift)
                    / collection_group_denominator
                )
                diagnostics[f"collection_policy_samples_{name}"] = mx.sum(collection_group_weights)
    return loss, diagnostics


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
    bootstrap_mask=None,
):
    """Soft pairwise loss for exact rules-derived tactical preferences."""

    mx, _nn = _mlx_modules()
    preferred_logits = model(states, preferred_actions, families)
    disfavored_logits = model(states, disfavored_actions, families)
    differences = preferred_logits - disfavored_logits
    losses = mx.logaddexp(mx.array(0.0), mx.array(float(margin)) - differences)
    weights = mx.ones_like(losses) if bootstrap_mask is None else bootstrap_mask
    denominator = mx.maximum(mx.sum(weights), mx.array(1.0))
    return mx.sum(weights * losses) / denominator, {
        "preference_accuracy": mx.sum(weights * (differences > 0)) / denominator,
        "preference_margin_mean": mx.sum(weights * differences) / denominator,
    }


def export_actor(
    model,
    spec: ModelSpec,
    path: str | Path,
    *,
    compressed: bool = True,
) -> Path:
    """Export MLX weights into a NumPy archive for CPU actors.

    Durable checkpoint actors default to compression.  Mutable runtime actors
    may opt out because they are rewritten and loaded by workers frequently,
    making compression CPU time more expensive than the temporary disk space.
    """

    from mlx.utils import tree_flatten

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in tree_flatten(model.parameters())}
    arrays["__spec_json__"] = np.frombuffer(
        json.dumps(spec.as_dict(), sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    writer = np.savez_compressed if compressed else np.savez
    writer(target, **arrays)
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


def regenerate_actor_snapshot(
    model_path: str | Path,
    actor_path: str | Path,
) -> Path:
    """Rebuild and validate a NumPy actor from a portable model checkpoint.

    Safetensors checkpoints and NumPy actors use the same flattened parameter
    names. Converting the arrays directly avoids initializing MLX/Metal in the
    API process when a user pins an older checkpoint.
    """

    from safetensors.numpy import load_file

    source = Path(model_path)
    sidecar = source.with_suffix(source.suffix + ".json")
    spec = ModelSpec.from_dict(json.loads(sidecar.read_text(encoding="utf-8")))
    arrays = {name: np.asarray(value) for name, value in load_file(source).items()}
    arrays["__spec_json__"] = np.frombuffer(
        json.dumps(spec.as_dict(), sort_keys=True).encode("utf-8"), dtype=np.uint8
    )

    target = Path(actor_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        NumpyActor.load(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def save_optimizer_state(optimizer: Any, path: str | Path) -> Path:
    """Atomically persist an MLX optimizer tree beside a model checkpoint."""

    from mlx.utils import tree_flatten

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.partial{target.suffix}")
    flattened = list(tree_flatten(optimizer.state))
    arrays = {f"value_{index}": np.asarray(value) for index, (_name, value) in enumerate(flattened)}
    arrays["__paths_json__"] = np.frombuffer(
        json.dumps([name for name, _value in flattened]).encode("utf-8"), dtype=np.uint8
    )
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target)
    return target


def load_optimizer_state(optimizer: Any, path: str | Path) -> bool:
    """Restore a state saved by :func:`save_optimizer_state` if it exists."""

    target = Path(path)
    if not target.is_file():
        return False
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    with np.load(target, allow_pickle=False) as archive:
        paths = json.loads(bytes(archive["__paths_json__"].tolist()).decode("utf-8"))
        flattened = [
            (str(name), mx.array(np.asarray(archive[f"value_{index}"])))
            for index, name in enumerate(paths)
        ]
    optimizer.state = tree_unflatten(flattened)
    leaves = [value for _name, value in tree_flatten(optimizer.state)]
    if leaves:
        mx.eval(*leaves)
    return True


def _silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


class NumpyActor:
    """Exact inference-only mirror of :func:`build_model`."""

    def __init__(self, spec: ModelSpec, weights: dict[str, np.ndarray]):
        self.spec = spec
        self.weights = weights
        self._prior_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

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
        *,
        head: int | None = None,
    ) -> np.ndarray:
        value = np.concatenate([state_features, action_features], axis=-1)
        value = _silu(self._norm(self._linear(value, "fusion_in"), "fusion_norm"))
        for index in range(self.spec.residual_blocks):
            value = self._residual(value, f"fusion_blocks.{index}")
        if getattr(getattr(self, "spec", None), "objective_version", 1) >= 2:
            if head is not None:
                if not 0 <= head < self.spec.bootstrap_heads:
                    raise ValueError(
                        f"head must be in [0, {self.spec.bootstrap_heads}), got {head}"
                    )
                return self._linear(
                    self._residual(value, f"head_blocks.{head}"),
                    f"head_outputs.{head}",
                )
            return np.stack(
                [
                    self._linear(
                        self._residual(value, f"head_blocks.{head}"),
                        f"head_outputs.{head}",
                    )
                    for head in range(self.spec.bootstrap_heads)
                ],
                axis=-1,
            )
        return self._linear(value, "output").reshape(
            (-1, self.spec.families, self.spec.bootstrap_heads)
        )

    def predict_values(self, states: np.ndarray, families: np.ndarray) -> np.ndarray:
        """Return state-value logits for generation-4 actors."""

        if self.spec.objective_version < 2:
            raise RuntimeError("separate state values require objective_version >= 2")
        states = np.asarray(states, dtype=np.float32)
        families = np.asarray(families, dtype=np.int64)
        if states.ndim == 1:
            states = states[None, :]
        if families.ndim == 0:
            families = families[None]
        values = self._linear(self._state_features(states), "value_output").reshape(
            (-1, self.spec.families, self.spec.bootstrap_heads)
        )
        return values[np.arange(len(values)), families]

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

    def predict_option_head(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        family: int,
        head: int,
    ) -> np.ndarray:
        """Score only one bootstrap head for a trajectory decision.

        Objective-v2 heads have independent residual adapters.  Rollout actors
        assign one head for an entire trajectory, so evaluating the other heads
        was pure work.  The public all-head methods remain unchanged for
        deployment ensembles, diagnostics, and checkpoint compatibility.
        """

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        state_batch = np.asarray(state, dtype=np.float32).reshape((1, -1))
        state_features = np.repeat(self._state_features(state_batch), len(actions), axis=0)
        all_logits = self._interaction_logits(
            state_features,
            self._action_features(actions),
            head=head if self.spec.objective_version >= 2 else None,
        )
        if self.spec.objective_version >= 2:
            return all_logits[:, family]
        return all_logits[:, family, head]

    def predict_option_value_batches(
        self,
        states: list[np.ndarray] | tuple[np.ndarray, ...],
        action_sets: list[np.ndarray] | tuple[np.ndarray, ...],
        families: list[int] | tuple[int, ...],
        heads: list[int | None] | tuple[int | None, ...],
    ) -> tuple[np.ndarray, ...]:
        """Vectorize several ragged decisions, grouped by behavior head.

        The result for each decision is the selected head's option vector, or
        the mean-head deployment vector when its head is ``None``. Grouping
        retains exact semantics while turning many tiny matrix multiplies into
        a few larger ones inside each rollout worker.
        """

        size = len(states)
        if not (len(action_sets) == len(families) == len(heads) == size):
            raise ValueError("batched option inputs must have equal lengths")
        if not size:
            return ()
        results: list[np.ndarray | None] = [None] * size
        groups: dict[int | None, list[int]] = {}
        for index, head in enumerate(heads):
            if head is not None and not 0 <= int(head) < self.spec.bootstrap_heads:
                raise ValueError(f"head must be in [0, {self.spec.bootstrap_heads}), got {head}")
            groups.setdefault(None if head is None else int(head), []).append(index)

        for head, indices in groups.items():
            grouped_states = np.stack(
                [np.asarray(states[index], dtype=np.float32) for index in indices]
            )
            grouped_actions = [
                np.asarray(action_sets[index], dtype=np.float32) for index in indices
            ]
            if any(actions.ndim != 2 or not len(actions) for actions in grouped_actions):
                raise ValueError("each batched decision needs a nonempty action matrix")
            counts = np.asarray([len(actions) for actions in grouped_actions], dtype=np.int64)
            state_features = np.repeat(self._state_features(grouped_states), counts, axis=0)
            action_features = self._action_features(np.concatenate(grouped_actions, axis=0))
            logits = self._interaction_logits(
                state_features,
                action_features,
                head=head if head is not None and self.spec.objective_version >= 2 else None,
            )
            row_families = np.repeat(
                np.asarray([families[index] for index in indices], dtype=np.int64), counts
            )
            if head is None:
                values = logits[np.arange(len(logits)), row_families].mean(axis=1)
            elif self.spec.objective_version >= 2:
                values = logits[np.arange(len(logits)), row_families]
            else:
                values = logits[np.arange(len(logits)), row_families, head]
            offsets = np.concatenate(([0], np.cumsum(counts)))
            for local_index, result_index in enumerate(indices):
                results[result_index] = values[offsets[local_index] : offsets[local_index + 1]]
        return tuple(value for value in results if value is not None)

    def _randomized_prior(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        family: int,
        head: int,
    ) -> np.ndarray:
        """Fixed, deterministic prior used only for trajectory exploration.

        A different random projection belongs to each bootstrap head.  It is
        never trained and is absent from deployment (`head=None`), so it keeps
        head policies distinct without changing checkpoint compatibility.
        """

        cached = self._prior_cache.get(head)
        if cached is None:
            seed = np.random.SeedSequence(
                [0xA5730, self.spec.state_size, self.spec.action_size, self.spec.families, head]
            )
            generator = np.random.default_rng(seed)
            state_weights = generator.normal(
                0.0, 1.0 / math.sqrt(self.spec.state_size), self.spec.state_size
            ).astype(np.float32)
            action_weights = generator.normal(
                0.0, 1.0 / math.sqrt(self.spec.action_size), self.spec.action_size
            ).astype(np.float32)
            family_bias = generator.normal(0.0, 0.35, self.spec.families).astype(np.float32)
            cached = (state_weights, action_weights, family_bias)
            self._prior_cache[head] = cached
        state_weights, action_weights, family_bias = cached
        state_term = float(np.asarray(state, dtype=np.float32).reshape(-1) @ state_weights)
        action_terms = np.asarray(actions, dtype=np.float32) @ action_weights
        return np.tanh(state_term + action_terms + float(family_bias[family])).astype(np.float32)

    def choose(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        family: int,
        *,
        head: int | None = None,
        epsilon: float = 0.0,
        exploration_top_k: int = 3,
        randomized_prior_scale: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, np.ndarray]:
        generator = rng or np.random.default_rng()
        # Keep the old path for lightweight test doubles and legacy actors.
        # Real objective-v2 rollout actors can skip every unassigned head.
        if (
            head is not None
            and hasattr(self, "weights")
            and getattr(getattr(self, "spec", None), "objective_version", 1) >= 2
        ):
            values = self.predict_option_head(state, actions, family, head)
        else:
            logits = self.predict_options(state, actions, family)
            values = logits.mean(axis=1) if head is None else logits[:, head]
        return self.choose_from_values(
            state,
            actions,
            family,
            values,
            head=head,
            epsilon=epsilon,
            exploration_top_k=exploration_top_k,
            randomized_prior_scale=randomized_prior_scale,
            rng=generator,
        )

    def choose_from_values(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        family: int,
        values: np.ndarray,
        *,
        head: int | None = None,
        epsilon: float = 0.0,
        exploration_top_k: int = 3,
        randomized_prior_scale: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, np.ndarray]:
        """Apply exploration to precomputed option values."""

        generator = rng or np.random.default_rng()
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (len(actions),):
            raise ValueError("option values must align with actions")
        if randomized_prior_scale < 0:
            raise ValueError("randomized_prior_scale must be nonnegative")
        if head is not None and randomized_prior_scale:
            values = values + randomized_prior_scale * self._randomized_prior(
                state, actions, family, head
            )
        if getattr(getattr(self, "spec", None), "objective_version", 1) >= 2:
            shifted = values - np.max(values)
            probabilities = np.exp(np.clip(shifted, -40.0, 0.0))
            probabilities /= probabilities.sum()
        else:
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
        if exploration_top_k < 0:
            raise ValueError("exploration_top_k must be nonnegative")
        if len(actions) == 1:
            # This can happen after the model-policy dominance mask removes a
            # legal END_TURN.  The action choice is forced, but its estimated
            # outcome is not certainty and is used as a next-decision target.
            return 0, probabilities
        if epsilon > 0 and generator.random() < epsilon:
            # Explore among plausible actions instead of uniformly sampling a
            # potentially catastrophic tail action.  Bootstrap-head selection
            # supplies the broader, trajectory-coherent exploration signal.
            count = len(values) if exploration_top_k == 0 else min(exploration_top_k, len(values))
            top = np.argpartition(values, -count)[-count:]
            return int(generator.choice(top)), probabilities
        best = np.flatnonzero(values == values.max())
        return int(generator.choice(best)), probabilities
