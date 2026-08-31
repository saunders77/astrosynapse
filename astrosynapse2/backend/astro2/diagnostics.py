"""General checkpoint strength, calibration, and ensemble diagnostics."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import make_baseline
from .cards import CARD_BY_ID
from .encoding import DecisionFamily as ModelDecisionFamily
from .encoding import Encoder
from .engine import (
    Action,
    ActionKind,
    Decision,
    DecisionFamily,
    Game,
    GameConfig,
    InPlayObservation,
    model_action_indices,
)
from .engine_encoding import EngineEncoder
from .model import NumpyActor
from .replay import PolicyItem
from .selfplay import ActorPolicy, collect_game


def _actor_encoder(actor: Any) -> Encoder:
    version = int(getattr(getattr(actor, "spec", None), "encoder_version", 1))
    return EngineEncoder(version=version)


def all_family_decision_suite(*, seed: int) -> tuple[Decision, ...]:
    """Return a compact deterministic corpus containing every decision family."""

    base = Game(config=GameConfig(seed=seed, starting_player=0)).observation(0)
    scout, viper = CARD_BY_ID[0], CARD_BY_ID[1]
    battle_pod, blob_fighter = CARD_BY_ID[4], CARD_BY_ID[7]
    blob_wheel, battle_station = CARD_BY_ID[8], CARD_BY_ID[15]
    missile_mech, patrol_mech = CARD_BY_ID[21], CARD_BY_ID[22]
    stealth_needle, corvette = CARD_BY_ID[23], CARD_BY_ID[27]
    blob_carrier = CARD_BY_ID[5]

    main_observation = replace(
        base,
        turn=8,
        action_number=7,
        trade=3,
        hand=(),
        trade_row=(battle_pod, blob_fighter, blob_wheel, corvette, battle_station),
    )
    hand_observation = replace(base, turn=9, hand=(scout, viper))
    opponent_base_observation = replace(
        base,
        turn=10,
        opponent_in_play=(
            InPlayObservation(battle_station, True, False, False),
            InPlayObservation(blob_wheel, True, False, False),
        ),
    )
    row_observation = replace(
        base,
        turn=11,
        trade_row=(battle_pod, blob_fighter, corvette, blob_wheel, battle_station),
    )
    own_ship_observation = replace(
        base,
        turn=12,
        own_in_play=(
            InPlayObservation(stealth_needle, True, False, False),
            InPlayObservation(blob_fighter, True, False, False),
            InPlayObservation(corvette, True, False, False),
        ),
    )

    return (
        Decision(
            DecisionFamily.MAIN,
            main_observation,
            (
                Action(
                    ActionKind.ACQUIRE,
                    card_id=battle_pod.card_id,
                    source_zone="trade_row",
                    amount=battle_pod.cost,
                ),
                Action(
                    ActionKind.ACQUIRE,
                    card_id=blob_fighter.card_id,
                    source_zone="trade_row",
                    amount=blob_fighter.cost,
                ),
                Action(ActionKind.END_TURN),
            ),
            "Acquire a card or end the turn",
        ),
        Decision(
            DecisionFamily.DISCARD,
            hand_observation,
            (
                Action(ActionKind.DISCARD_CARD, card_id=scout.card_id, source_zone="hand"),
                Action(ActionKind.DISCARD_CARD, card_id=viper.card_id, source_zone="hand"),
            ),
            "Discard one card",
        ),
        Decision(
            DecisionFamily.SCRAP,
            hand_observation,
            (
                Action(ActionKind.SCRAP_CARD, card_id=scout.card_id, source_zone="hand"),
                Action(ActionKind.SCRAP_CARD, card_id=viper.card_id, source_zone="hand"),
                Action(ActionKind.DECLINE, card_id=missile_mech.card_id, ability="scrap_any"),
            ),
            "Scrap a card or decline",
        ),
        Decision(
            DecisionFamily.DESTROY_BASE,
            opponent_base_observation,
            (
                Action(
                    ActionKind.DESTROY_BASE,
                    card_id=missile_mech.card_id,
                    target_card_id=battle_station.card_id,
                    ability="destroy_base",
                    source_zone="opponent_in_play",
                ),
                Action(
                    ActionKind.DESTROY_BASE,
                    card_id=missile_mech.card_id,
                    target_card_id=blob_wheel.card_id,
                    ability="destroy_base",
                    source_zone="opponent_in_play",
                ),
                Action(ActionKind.DECLINE, card_id=missile_mech.card_id, ability="destroy_base"),
            ),
            "Destroy an opposing base or decline",
        ),
        Decision(
            DecisionFamily.SCRAP_TRADE_ROW,
            row_observation,
            (
                Action(
                    ActionKind.SCRAP_TRADE_ROW,
                    card_id=battle_pod.card_id,
                    target_card_id=blob_fighter.card_id,
                    ability="scrap_trade_row",
                    source_zone="trade_row",
                ),
                Action(
                    ActionKind.SCRAP_TRADE_ROW,
                    card_id=battle_pod.card_id,
                    target_card_id=corvette.card_id,
                    ability="scrap_trade_row",
                    source_zone="trade_row",
                ),
                Action(ActionKind.DECLINE, card_id=battle_pod.card_id, ability="scrap_trade_row"),
            ),
            "Scrap a trade-row card or decline",
        ),
        Decision(
            DecisionFamily.COPY_SHIP,
            own_ship_observation,
            (
                Action(
                    ActionKind.COPY_SHIP,
                    card_id=stealth_needle.card_id,
                    target_card_id=blob_fighter.card_id,
                    ability="copy_ship",
                    source_zone="in_play",
                ),
                Action(
                    ActionKind.COPY_SHIP,
                    card_id=stealth_needle.card_id,
                    target_card_id=corvette.card_id,
                    ability="copy_ship",
                    source_zone="in_play",
                ),
            ),
            "Copy an allied ship",
        ),
        Decision(
            DecisionFamily.FREE_ACQUIRE,
            row_observation,
            (
                Action(
                    ActionKind.FREE_ACQUIRE,
                    card_id=blob_carrier.card_id,
                    target_card_id=blob_fighter.card_id,
                    ability="free_ship_to_top",
                    source_zone="trade_row",
                ),
                Action(
                    ActionKind.FREE_ACQUIRE,
                    card_id=blob_carrier.card_id,
                    target_card_id=corvette.card_id,
                    ability="free_ship_to_top",
                    source_zone="trade_row",
                ),
                Action(
                    ActionKind.DECLINE,
                    card_id=blob_carrier.card_id,
                    ability="free_ship_to_top",
                ),
            ),
            "Acquire a free ship or decline",
        ),
        Decision(
            DecisionFamily.ABILITY_MODE,
            replace(base, turn=13),
            (
                Action(
                    ActionKind.CHOOSE_MODE,
                    card_id=patrol_mech.card_id,
                    ability="gain_combat",
                    amount=5,
                ),
                Action(
                    ActionKind.CHOOSE_MODE,
                    card_id=patrol_mech.card_id,
                    ability="gain_trade",
                    amount=3,
                ),
            ),
            "Choose an ability mode",
        ),
    )


def ensemble_metrics(actor: NumpyActor, decisions: tuple[Decision, ...]) -> dict[str, Any]:
    """Measure deployed-policy disagreement and probability spread across heads."""

    encoder = _actor_encoder(actor)
    disagreements = 0
    probability_stds: list[float] = []
    action_options = 0
    family_positions = {family.value: 0 for family in DecisionFamily}
    for decision in decisions:
        encoded = encoder.encode_decision(decision.observation, decision)
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        logits = actor.predict_options(
            encoded.state,
            encoded.actions[eligible],
            int(encoded.family),
        )
        if logits.ndim == 1:
            logits = logits[:, None]
        head_choices = np.argmax(logits, axis=0)
        disagreements += int(len(np.unique(head_choices)) > 1)
        if getattr(getattr(actor, "spec", None), "objective_version", 1) >= 2:
            shifted = logits - logits.max(axis=0, keepdims=True)
            probabilities = np.exp(np.clip(shifted, -40.0, 0.0))
            probabilities /= probabilities.sum(axis=0, keepdims=True)
        else:
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        probability_stds.extend(np.std(probabilities, axis=1).tolist())
        action_options += len(eligible)
        family_positions[decision.family.value] += 1
    return {
        "positions": len(decisions),
        "families": sum(count > 0 for count in family_positions.values()),
        "family_positions": family_positions,
        "action_options": action_options,
        "head_argmax_disagreements": int(disagreements),
        "head_argmax_disagreement_rate": float(disagreements / max(1, len(decisions))),
        "mean_probability_std": (float(np.mean(probability_stds)) if probability_stds else 0.0),
    }


def heldout_outcome_metrics(
    actor: NumpyActor,
    *,
    seed: int,
    games: int,
) -> dict[str, Any]:
    items = _natural_policy_items(actor, seed=seed, positions=max(64, games * 80))
    if not items:
        return {"games": 0, "samples": 0}
    states = np.stack([item.state for item in items])
    families = np.asarray([int(item.family) for item in items])
    targets = np.asarray([item.target for item in items], dtype=np.float32)
    logits = actor.predict_values(states, families)
    predictions = (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).mean(axis=1)
    clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    bce = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    game_briers: list[float] = []
    for game_id in sorted({int(item.game_id) for item in items}):
        indices = np.asarray(
            [index for index, item in enumerate(items) if int(item.game_id) == game_id]
        )
        game_briers.append(float(np.mean(np.square(predictions[indices] - targets[indices]))))
    return {
        "games": len(game_briers),
        "samples": int(len(targets)),
        "source": "candidate_policy_vs_fixed_opponents",
        "bce": float(bce),
        "brier": float(np.mean(np.square(predictions - targets))),
        "game_grouped_brier": float(np.mean(game_briers)),
        "accuracy": float(np.mean((predictions >= 0.5) == (targets >= 0.5))),
    }


def _natural_policy_items(
    actor: NumpyActor,
    *,
    seed: int,
    positions: int,
) -> list[PolicyItem]:
    """Collect decisions induced by the candidate, never baseline-only states."""

    encoder = _actor_encoder(actor)
    styles = ("balanced", "economy", "aggressive")
    policy = ActorPolicy(actor, encoder)
    items: list[PolicyItem] = []
    game_index = 0
    maximum_games = max(8, min(500, int(math.ceil(positions / 8))))
    while len(items) < positions and game_index < maximum_games:
        actor_player = game_index % 2
        baseline = make_baseline(styles[game_index % 3], seed + game_index * 17 + 1)
        policies: list[Any] = [baseline, baseline]
        policies[actor_player] = policy
        collect_players = [False, False]
        collect_players[actor_player] = True
        collected = collect_game(
            (policies[0], policies[1]),
            seed=seed + game_index,
            encoder=encoder,
            bootstrap_heads=actor.spec.bootstrap_heads,
            collect_players=(collect_players[0], collect_players[1]),
            deployment_policy=(actor_player == 0, actor_player == 1),
            collect_preferences=False,
            collect_policy_decisions=True,
            collect_outcome_decisions=False,
            max_turns=180,
            max_actions_per_turn=160,
        )
        items.extend(collected.policy_samples)
        game_index += 1
    return items[:positions]


def natural_policy_metrics(
    actor: NumpyActor,
    *,
    seed: int,
    positions: int,
    reference_actor: NumpyActor | None = None,
) -> dict[str, Any]:
    # When comparing with a champion, collect the deterministic state bank
    # with that reference policy. Every candidate is then measured on the same
    # natural positions rather than on a candidate-dependent distribution.
    collection_actor = reference_actor if reference_actor is not None else actor
    items = _natural_policy_items(collection_actor, seed=seed, positions=positions)
    disagreements = 0
    entropy_values: list[float] = []
    probability_stds: list[float] = []
    kl_values: list[float] = []
    action_flips = 0
    value_absolute_deltas: list[float] = []
    value_signed_deltas: list[float] = []
    family_positions = {family.name.lower(): 0 for family in ModelDecisionFamily}
    for item in items:
        logits = actor.predict_options(item.state, item.legal_actions, int(item.family))
        shifted = logits - logits.max(axis=0, keepdims=True)
        probabilities = np.exp(np.clip(shifted, -40.0, 0.0))
        probabilities /= probabilities.sum(axis=0, keepdims=True)
        disagreements += int(len(np.unique(np.argmax(probabilities, axis=0))) > 1)
        probability_stds.extend(np.std(probabilities, axis=1).tolist())
        deployed = probabilities.mean(axis=1)
        entropy_values.append(
            float(-np.sum(deployed * np.log(np.clip(deployed, 1e-9, 1.0))))
            / max(1e-9, math.log(len(deployed)))
        )
        family_positions[ModelDecisionFamily(int(item.family)).name.lower()] += 1
        if reference_actor is not None:
            reference_logits = reference_actor.predict_options(
                item.state, item.legal_actions, int(item.family)
            ).mean(axis=1)
            reference_probabilities = np.exp(reference_logits - np.max(reference_logits))
            reference_probabilities /= np.sum(reference_probabilities)
            kl_values.append(
                float(
                    np.sum(
                        deployed
                        * (
                            np.log(np.clip(deployed, 1e-9, 1.0))
                            - np.log(np.clip(reference_probabilities, 1e-9, 1.0))
                        )
                    )
                )
            )
            action_flips += int(int(np.argmax(deployed)) != int(np.argmax(reference_probabilities)))
            candidate_value = float(
                np.mean(
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -np.clip(
                                actor.predict_values(
                                    item.state,
                                    np.asarray([int(item.family)], dtype=np.int64),
                                ),
                                -40.0,
                                40.0,
                            )
                        )
                    )
                )
            )
            reference_value = float(
                np.mean(
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -np.clip(
                                reference_actor.predict_values(
                                    item.state,
                                    np.asarray([int(item.family)], dtype=np.int64),
                                ),
                                -40.0,
                                40.0,
                            )
                        )
                    )
                )
            )
            value_signed_deltas.append(candidate_value - reference_value)
            value_absolute_deltas.append(abs(candidate_value - reference_value))
    return {
        "positions": len(items),
        "family_positions": family_positions,
        "families": sum(value > 0 for value in family_positions.values()),
        "head_argmax_disagreements": disagreements,
        "head_argmax_disagreement_rate": disagreements / max(1, len(items)),
        "mean_probability_std": float(np.mean(probability_stds)) if probability_stds else 0.0,
        "mean_normalized_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "reference_policy_kl": float(np.mean(kl_values)) if kl_values else None,
        "reference_action_flips": action_flips if kl_values else None,
        "reference_action_flip_rate": (action_flips / len(kl_values) if kl_values else None),
        "reference_value_mean_absolute_delta": (
            float(np.mean(value_absolute_deltas)) if value_absolute_deltas else None
        ),
        "reference_value_mean_signed_delta": (
            float(np.mean(value_signed_deltas)) if value_signed_deltas else None
        ),
    }


def baseline_metrics(
    actor_path: str | Path,
    *,
    seed: int,
    pairs: int,
) -> dict[str, Any]:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    actor = NumpyActor.load(actor_path)
    encoder = _actor_encoder(actor)

    def choose(_player_id: int, decision: Decision) -> int:
        """Use a stable first-maximum tie break for exact paired games."""

        encoded = encoder.encode_decision(decision.observation, decision)
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        logits = actor.predict_options(
            encoded.state,
            encoded.actions[eligible],
            int(encoded.family),
        ).mean(axis=1)
        return int(eligible[int(np.argmax(logits))])

    scores: dict[str, float] = {}
    truncated = 0
    for style_index, style in enumerate(("balanced", "economy", "aggressive")):
        points = 0.0
        for pair in range(pairs):
            game_seed = seed + style_index * 10_000 + pair
            for model_player in (0, 1):
                baseline = make_baseline(style, game_seed + 50_000)
                policies = [baseline, baseline]
                policies[model_player] = choose
                result = Game(
                    choosers=(policies[0], policies[1]),
                    config=GameConfig(
                        seed=game_seed,
                        starting_player=0,
                        max_turns=180,
                        max_actions_per_turn=160,
                    ),
                ).run()
                truncated += int(result.truncated)
                points += 0.5 if result.winner is None else float(result.winner == model_player)
        scores[style] = points / (2 * pairs)
    return {
        "pairs_per_opponent": pairs,
        "scores": scores,
        "mean_score": float(np.mean(list(scores.values()))),
        "truncated_games": truncated,
    }


def checkpoint_diagnostics(
    actor_path: str | Path,
    *,
    seed: int,
    games: int,
    baseline_pairs: int,
    natural_positions: int = 2_000,
    reference_actor_path: str | Path | None = None,
) -> dict[str, Any]:
    actor = NumpyActor.load(actor_path)
    synthetic_ensemble = ensemble_metrics(
        actor,
        all_family_decision_suite(seed=seed + 720_000),
    )
    reference_actor = (
        NumpyActor.load(reference_actor_path)
        if reference_actor_path and Path(reference_actor_path).is_file()
        else None
    )
    try:
        ensemble = natural_policy_metrics(
            actor,
            seed=seed + 740_000,
            positions=natural_positions,
            reference_actor=reference_actor,
        )
    except (AttributeError, ValueError) as error:
        # Lightweight diagnostic doubles and legacy actors may not satisfy the
        # current self-play contract. Keep the all-family suite useful without
        # weakening real checkpoint diagnostics.
        ensemble = dict(synthetic_ensemble)
        ensemble["natural_diagnostics_error"] = f"{type(error).__name__}: {error}"
        ensemble["natural_diagnostics_fallback"] = True
    else:
        ensemble["natural_diagnostics_error"] = None
        ensemble["natural_diagnostics_fallback"] = False
    ensemble["synthetic_suite"] = synthetic_ensemble
    heldout = heldout_outcome_metrics(actor, seed=seed + 800_000, games=games)
    baselines = baseline_metrics(actor_path, seed=seed + 900_000, pairs=baseline_pairs)
    values = (ensemble, heldout, baselines)
    if not all(
        math.isfinite(float(value))
        for group in values
        for value in group.values()
        if isinstance(value, (int, float))
    ):
        raise RuntimeError("checkpoint diagnostics produced a non-finite metric")
    return {
        "ensemble": ensemble,
        "heldout": heldout,
        "baselines": baselines,
    }


__all__ = [
    "all_family_decision_suite",
    "baseline_metrics",
    "checkpoint_diagnostics",
    "ensemble_metrics",
    "heldout_outcome_metrics",
]
