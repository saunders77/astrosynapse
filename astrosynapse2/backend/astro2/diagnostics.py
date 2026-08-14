"""General checkpoint strength, calibration, and ensemble diagnostics."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import make_baseline
from .cards import CARD_BY_ID
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
from .model import NumpyActor
from .selfplay import collect_game

def _actor_encoder(actor: Any) -> Encoder:
    version = int(getattr(getattr(actor, "spec", None), "encoder_version", 1))
    return Encoder(version=version)


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
    encoder = _actor_encoder(actor)
    styles = ("balanced", "economy", "aggressive")
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    per_game_brier: list[float] = []
    for game_index in range(games):
        collected = collect_game(
            (
                make_baseline(styles[game_index % 3], seed + game_index * 2),
                make_baseline(styles[(game_index + 1) % 3], seed + game_index * 2 + 1),
            ),
            seed=seed + game_index,
            encoder=encoder,
            bootstrap_heads=actor.spec.bootstrap_heads,
            collect_players=(True, True),
            max_turns=180,
            max_actions_per_turn=160,
        )
        if not collected.samples:
            continue
        states = np.stack([sample.state for sample in collected.samples])
        actions = np.stack([sample.action for sample in collected.samples])
        families = np.asarray([int(sample.family) for sample in collected.samples])
        targets = np.asarray([sample.target for sample in collected.samples], dtype=np.float32)
        logits = (
            actor.predict_values(states, families)
            if getattr(actor.spec, "objective_version", 1) >= 2
            else actor.predict(states, actions, families)
        )
        predictions = (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).mean(axis=1)
        all_predictions.append(predictions)
        all_targets.append(targets)
        per_game_brier.append(float(np.mean(np.square(predictions - targets))))
    if not all_predictions:
        return {"games": 0, "samples": 0}
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    clipped = np.clip(predictions, 1e-6, 1.0 - 1e-6)
    bce = -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    return {
        "games": len(per_game_brier),
        "samples": int(len(targets)),
        "bce": float(bce),
        "brier": float(np.mean(np.square(predictions - targets))),
        "game_grouped_brier": float(np.mean(per_game_brier)),
        "accuracy": float(np.mean((predictions >= 0.5) == (targets >= 0.5))),
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
) -> dict[str, Any]:
    actor = NumpyActor.load(actor_path)
    ensemble = ensemble_metrics(
        actor,
        all_family_decision_suite(seed=seed + 720_000),
    )
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
