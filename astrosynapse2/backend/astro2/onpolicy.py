"""Fresh-trajectory policy optimization for the deployed mean-head actor.

This experimental learner deliberately has no replay, behavior-head mixtures,
critic gradients in the policy trunk, or bootstrapped search labels. Its actor
samples the exact masked distribution used by its PPO objective.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .arena import _ActorChooser, _derived_seed
from .engine import Game, GameConfig, Seating, model_action_indices
from .engine_encoding import EngineEncoder
from .native_actor import NativeActor
from .planning import strategic_decision

_ACTORS: OrderedDict[str, NativeActor] = OrderedDict()


def cached_actor(path):
    if path not in _ACTORS:
        _ACTORS[path] = NativeActor.load(path)
        while len(_ACTORS) > 3:
            _ACTORS.popitem(last=False)
    _ACTORS.move_to_end(path)
    return _ACTORS[path]


@dataclass
class Trajectory:
    states: list
    actions: list
    families: list
    selected: list
    log_probabilities: list
    log_policies: list
    target: float
    truncated: bool
    value_states: list = field(default_factory=list)
    value_families: list = field(default_factory=list)
    value_forced: list = field(default_factory=list)


def collect_trajectory(task):
    actor_path, opponent_path, temperature, seed, index, *extra = task
    actor, opponent = cached_actor(actor_path), cached_actor(opponent_path)
    encoder = EngineEncoder(version=actor.spec.encoder_version)
    rng = np.random.default_rng(_derived_seed(seed, index, "learner"))
    rows = Trajectory([], [], [], [], [], [], 0.0, False)
    latest_decision = latest_encoded = None
    automatic = None
    if extra and extra[0]:
        auto = cached_actor(extra[0])
        automatic = _ActorChooser(
            auto,
            EngineEncoder(version=auto.spec.encoder_version),
            _derived_seed(seed, index, "auto"),
        )

    def choose(pid, decision):
        nonlocal latest_decision, latest_encoded
        encoded = encoder.encode_decision(decision.observation, decision)
        latest_decision, latest_encoded = decision, encoded
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        if automatic is not None and not strategic_decision(decision):
            return automatic(pid, decision)
        if len(eligible) == 1:
            return decision.actions[int(eligible[0])]
        actions = encoded.actions[eligible]
        logits = actor.predict_options(encoded.state, actions, int(encoded.family)).mean(axis=1)
        logits = (logits - logits.max()) / temperature
        probabilities = np.exp(logits.astype(np.float64))
        probabilities /= probabilities.sum()
        chosen = int(rng.choice(len(eligible), p=probabilities))
        rows.states.append(encoded.state)
        rows.actions.append(actions)
        rows.families.append(int(encoded.family))
        rows.selected.append(chosen)
        rows.log_probabilities.append(float(np.log(probabilities[chosen])))
        rows.log_policies.append(
            (logits.astype(np.float64) - np.log(np.exp(logits.astype(np.float64)).sum())).astype(
                np.float32
            )
        )
        return decision.actions[int(eligible[chosen])]

    def record_value(pid, decision, _action):
        if pid != seat:
            return
        # The engine bypasses choose() when there is just one legal action.
        # Its decision hook still observes that state, before applying it.
        encoded = (
            latest_encoded
            if decision is latest_decision
            else encoder.encode_decision(decision.observation, decision)
        )
        rows.value_states.append(encoded.state)
        rows.value_families.append(int(encoded.family))
        rows.value_forced.append(
            len(decision.actions) == 1 or len(model_action_indices(decision)) == 1
        )

    other = _ActorChooser(
        opponent,
        EngineEncoder(version=opponent.spec.encoder_version),
        _derived_seed(seed, index, "opponent"),
    )
    seat = index % 2
    result = Game(
        choosers=(choose, other) if seat == 0 else (other, choose),
        decision_hook=record_value if len(extra) > 2 and extra[2] else None,
        config=GameConfig(
            seed=_derived_seed(seed, index, "game"),
            seating=Seating.FIXED,
            starting_player=0,
            max_turns=180,
            max_actions_per_turn=160,
            rules_version=extra[1] if len(extra) > 1 else 1,
        ),
    ).run()
    rows.truncated = result.truncated
    rows.target = float(result.winner == seat)
    return rows


def critic_logits(head, features, families, family_count, heads):
    """Select the value bank without involving any policy parameters."""
    import mlx.core as mx

    values = head(features).reshape((-1, family_count, heads))
    indices = mx.broadcast_to(families[:, None, None], (len(families), 1, heads))
    return mx.take_along_axis(values, indices, axis=1).squeeze(1)


def critic_calibration(probabilities, targets, families, forced):
    """Pre-fit, fresh-rollout diagnostics; never report training fit as calibration."""
    probabilities, targets = np.asarray(probabilities), np.asarray(targets)
    families, forced = np.asarray(families), np.asarray(forced, dtype=bool)

    def summary(mask):
        p, y = probabilities[mask], targets[mask]
        if not len(p):
            return {"positions": 0}
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        return dict(
            positions=len(p),
            prediction=float(p.mean()),
            outcome=float(y.mean()),
            brier=float(np.mean((p - y) ** 2)),
            log_loss=float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
        )

    return dict(
        all=summary(np.ones(len(targets), dtype=bool)),
        forced=summary(forced),
        choices=summary(~forced),
        families={str(f): summary(families == f) for f in np.unique(families)},
    )


def masked_log_policy(model, states, actions, mask, families, temperature):
    import mlx.core as mx

    batch, options, action_size = actions.shape
    features = model.state_features(states)
    repeated = mx.broadcast_to(features[:, None, :], (batch, options, features.shape[-1])).reshape(
        (batch * options, -1)
    )
    repeated_families = mx.broadcast_to(families[:, None], (batch, options)).reshape(-1)
    logits = (
        model.action_logits_from_features(
            repeated, actions.reshape((batch * options, action_size)), repeated_families
        )
        .reshape((batch, options, -1))
        .mean(axis=2)
        / temperature
    )
    logits = mx.where(mask > 0, logits, -1e9)
    return logits - mx.logsumexp(logits, axis=1, keepdims=True), features


def ppo_loss(
    model,
    states,
    actions,
    mask,
    families,
    selected,
    old_logp,
    advantages,
    targets,
    old_log_policies=None,
    *,
    temperature=0.1,
    clip=0.2,
    value_weight=0.5,
    entropy_weight=0.0,
):
    import mlx.core as mx
    import mlx.nn as nn

    logp, features = masked_log_policy(model, states, actions, mask, families, temperature)
    chosen = mx.take_along_axis(logp, selected[:, None], axis=1).squeeze(1)
    logratio = chosen - old_logp
    ratio = mx.exp(logratio)
    policy = -mx.mean(
        mx.minimum(ratio * advantages, mx.clip(ratio, 1 - clip, 1 + clip) * advantages)
    )
    # The critic has its own trainable output bank. It cannot move the shared
    # state representation and silently change action rankings.
    value_logits = model.state_values_from_features(mx.stop_gradient(features), families)
    value_targets = mx.broadcast_to(targets[:, None], value_logits.shape)
    value_loss = nn.losses.binary_cross_entropy(
        value_logits, value_targets, with_logits=True, reduction="mean"
    )
    entropy = -mx.mean(mx.sum(mx.exp(logp) * logp, axis=1))
    diagnostics = dict(
        policy_loss=policy,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=mx.mean((ratio - 1) - logratio),
        clip_fraction=mx.mean((mx.abs(ratio - 1) > clip).astype(mx.float32)),
        ratio_mean=mx.mean(ratio),
    )
    if old_log_policies is not None:
        old_probabilities = mx.exp(old_log_policies) * mask
        diagnostics["categorical_kl"] = mx.mean(
            mx.sum(old_probabilities * (old_log_policies - logp), axis=1)
        )
    return policy + value_weight * value_loss - entropy_weight * entropy, diagnostics
