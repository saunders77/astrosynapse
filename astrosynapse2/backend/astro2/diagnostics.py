"""Deterministic held-out and tactical checkpoint diagnostics."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import make_baseline
from .cards import ALL_CARDS, CARD_BY_ID
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

EARLY_TURN_MAX = 18
HIGH_COST_MIN = 6


def _actor_encoder(actor: Any) -> Encoder:
    version = int(getattr(getattr(actor, "spec", None), "encoder_version", 1))
    return Encoder(version=version)


def _behavioral_suite(*, seed: int, games: int, limit: int = 512) -> tuple[Decision, ...]:
    decisions: list[Decision] = []
    styles = ("balanced", "economy", "aggressive")
    for game_index in range(games):
        first = make_baseline(styles[game_index % len(styles)], seed + 2 * game_index)
        second = make_baseline(styles[(game_index + 1) % len(styles)], seed + 2 * game_index + 1)

        def hook(_player: int, decision: Decision, _selected: Any) -> None:
            kinds = {action.kind for action in decision.actions}
            if ActionKind.END_TURN in kinds and kinds.intersection(
                {
                    ActionKind.PLAY_CARD,
                    ActionKind.ACTIVATE_BASE,
                    ActionKind.ATTACK_BASE,
                    ActionKind.ATTACK_PLAYER,
                }
            ):
                decisions.append(decision)

        Game(
            choosers=(first, second),
            config=GameConfig(
                seed=seed + game_index,
                starting_player=game_index % 2,
                max_turns=180,
                max_actions_per_turn=160,
            ),
            decision_hook=hook,
        ).run()
        if len(decisions) >= limit:
            break
    return tuple(decisions[:limit])


def tactical_metrics(actor: NumpyActor, decisions: tuple[Decision, ...]) -> dict[str, Any]:
    encoder = _actor_encoder(actor)
    raw_end = masked_end = attack_end = play_end = activate_end = 0
    margins: list[float] = []
    for decision in decisions:
        encoded = encoder.encode_decision(decision.observation, decision)
        logits = actor.predict_options(encoded.state, encoded.actions, int(encoded.family)).mean(
            axis=1
        )
        raw_index = int(np.argmax(logits))
        eligible = np.asarray(model_action_indices(decision), dtype=np.int64)
        masked_index = int(eligible[int(np.argmax(logits[eligible]))])
        raw_end += decision.actions[raw_index].kind == ActionKind.END_TURN
        masked_end += decision.actions[masked_index].kind == ActionKind.END_TURN
        end_index = next(
            index
            for index, action in enumerate(decision.actions)
            if action.kind == ActionKind.END_TURN
        )
        preferred = [
            index
            for index, action in enumerate(decision.actions)
            if action.kind
            in {
                ActionKind.PLAY_CARD,
                ActionKind.ACTIVATE_BASE,
                ActionKind.ATTACK_BASE,
                ActionKind.ATTACK_PLAYER,
            }
        ]
        margins.append(float(max(logits[preferred]) - logits[end_index]))
        if decision.actions[raw_index].kind == ActionKind.END_TURN:
            kinds = {decision.actions[index].kind for index in preferred}
            attack_end += bool(kinds & {ActionKind.ATTACK_BASE, ActionKind.ATTACK_PLAYER})
            play_end += ActionKind.PLAY_CARD in kinds
            activate_end += ActionKind.ACTIVATE_BASE in kinds
    return {
        "positions": len(decisions),
        "raw_end_turn_violations": int(raw_end),
        "masked_end_turn_violations": int(masked_end),
        "raw_attack_end_violations": int(attack_end),
        "raw_play_end_violations": int(play_end),
        "raw_activate_end_violations": int(activate_end),
        "mean_preference_logit_margin": float(np.mean(margins)) if margins else 0.0,
        "minimum_preference_logit_margin": float(np.min(margins)) if margins else 0.0,
    }


def strategic_decision_suite(*, seed: int) -> tuple[Decision, ...]:
    """Build exact optional-scrap positions where retaining the card is available.

    Each position occurs early, while both players have substantial authority,
    and contains only the card's optional scrap ability and END_TURN.  That
    makes END_TURN the semantically explicit "keep this card" action instead
    of relying on an inferred label from a rollout policy.
    """

    base = Game(config=GameConfig(seed=seed, starting_player=0)).observation(0)
    scrappable = tuple(card for card in ALL_CARDS if card.scrap)
    decisions: list[Decision] = []
    for index, card in enumerate(scrappable):
        observation = replace(
            base,
            turn=6 + index % 7,
            action_number=4 + index % 3,
            own_authority=44 - index % 4,
            opponent_authority=39 - index % 5,
            combat=0,
            trade=0,
            hand=(),
            own_in_play=(
                InPlayObservation(
                    card=card,
                    activated=True,
                    ally_triggered=False,
                    copied_from_stealth_needle=False,
                ),
            ),
        )
        decisions.append(
            Decision(
                DecisionFamily.MAIN,
                observation,
                (
                    Action(
                        ActionKind.SCRAP_FOR_ABILITY,
                        card_id=card.card_id,
                        ability=card.scrap,
                        source_zone="in_play",
                        amount=card.scrap_amount,
                        opaque=(index + 1,),
                    ),
                    Action(ActionKind.END_TURN),
                ),
                f"Retain {card.name} or use its optional scrap ability",
            )
        )
    return tuple(decisions)


def strategic_metrics(actor: NumpyActor, decisions: tuple[Decision, ...]) -> dict[str, Any]:
    """Measure whether optional scrap actions outrank the explicit keep action."""

    encoder = _actor_encoder(actor)
    margins: list[float] = []
    early_high_cost_margins: list[float] = []
    for decision in decisions:
        scrap_indices = [
            index
            for index, action in enumerate(decision.actions)
            if action.kind == ActionKind.SCRAP_FOR_ABILITY
        ]
        keep_indices = [
            index
            for index, action in enumerate(decision.actions)
            if action.kind == ActionKind.END_TURN
        ]
        if not scrap_indices or not keep_indices:
            continue
        encoded = encoder.encode_decision(decision.observation, decision)
        logits = actor.predict_options(
            encoded.state,
            encoded.actions,
            int(encoded.family),
        ).mean(axis=1)
        keep_logit = float(logits[keep_indices[0]])
        for scrap_index in scrap_indices:
            margin = float(logits[scrap_index] - keep_logit)
            margins.append(margin)
            card = CARD_BY_ID.get(decision.actions[scrap_index].card_id)
            if (
                card is not None
                and card.cost >= HIGH_COST_MIN
                and decision.observation.turn <= EARLY_TURN_MAX
            ):
                early_high_cost_margins.append(margin)

    scrap_over_keep = sum(margin > 0.0 for margin in margins)
    high_cost_scrap_over_keep = sum(margin > 0.0 for margin in early_high_cost_margins)
    return {
        "positions": len(decisions),
        "optional_scrap_positions": len(margins),
        "scrap_over_keep_count": int(scrap_over_keep),
        "scrap_over_keep_rate": float(scrap_over_keep / max(1, len(margins))),
        "mean_scrap_over_keep_logit_margin": (float(np.mean(margins)) if margins else 0.0),
        "early_high_cost_positions": len(early_high_cost_margins),
        "early_high_cost_scrap_over_keep_count": int(high_cost_scrap_over_keep),
        "early_high_cost_scrap_over_keep_rate": float(
            high_cost_scrap_over_keep / max(1, len(early_high_cost_margins))
        ),
        "early_high_cost_mean_scrap_over_keep_logit_margin": (
            float(np.mean(early_high_cost_margins)) if early_high_cost_margins else 0.0
        ),
        "early_high_cost_passed": bool(early_high_cost_margins and high_cost_scrap_over_keep == 0),
    }


def resource_efficiency_decision_suite(*, seed: int) -> tuple[Decision, ...]:
    """Early economic states where ending burns resources for no benefit."""

    base = Game(config=GameConfig(seed=seed, starting_player=0)).observation(0)
    explorer = CARD_BY_ID[2]
    expensive_row = tuple(CARD_BY_ID[card_id] for card_id in (5, 15, 21, 22, 23))
    decisions: list[Decision] = []
    for index in range(12):
        observation = replace(
            base,
            turn=2 + index % 4,
            action_number=5 + index % 3,
            own_authority=46 - index % 4,
            opponent_authority=45 - index % 5,
            combat=0,
            trade=2,
            hand=(),
            own_in_play=(),
            trade_row=expensive_row,
            explorers_remaining=10 - index % 3,
            explorer_supply=(explorer,) * (10 - index % 3),
        )
        decisions.append(
            Decision(
                DecisionFamily.MAIN,
                observation,
                (
                    Action(
                        ActionKind.ACQUIRE,
                        card_id=explorer.card_id,
                        source_zone="explorer_supply",
                        amount=explorer.cost,
                    ),
                    Action(ActionKind.END_TURN),
                ),
                "Spend otherwise-wasted early trade or end the turn",
            )
        )
    return tuple(decisions)


def resource_efficiency_metrics(
    actor: NumpyActor, decisions: tuple[Decision, ...]
) -> dict[str, Any]:
    encoder = _actor_encoder(actor)
    margins: list[float] = []
    for decision in decisions:
        encoded = encoder.encode_decision(decision.observation, decision)
        logits = actor.predict_options(encoded.state, encoded.actions, int(encoded.family)).mean(
            axis=1
        )
        acquire = next(
            index
            for index, action in enumerate(decision.actions)
            if action.kind == ActionKind.ACQUIRE
        )
        end = next(
            index
            for index, action in enumerate(decision.actions)
            if action.kind == ActionKind.END_TURN
        )
        margins.append(float(logits[acquire] - logits[end]))
    violations = sum(margin <= 0 for margin in margins)
    return {
        "positions": len(margins),
        "unused_resource_violations": int(violations),
        "unused_resource_violation_rate": float(violations / max(1, len(margins))),
        "mean_spend_over_end_logit_margin": float(np.mean(margins)) if margins else 0.0,
        "minimum_spend_over_end_logit_margin": float(np.min(margins)) if margins else 0.0,
        "passed": bool(margins and violations == 0),
    }


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
    suite = _behavioral_suite(seed=seed + 700_000, games=games)
    tactical = tactical_metrics(actor, suite)
    strategic = strategic_metrics(
        actor,
        strategic_decision_suite(seed=seed + 710_000),
    )
    resource_efficiency = resource_efficiency_metrics(
        actor,
        resource_efficiency_decision_suite(seed=seed + 715_000),
    )
    ensemble = ensemble_metrics(
        actor,
        all_family_decision_suite(seed=seed + 720_000),
    )
    heldout = heldout_outcome_metrics(actor, seed=seed + 800_000, games=games)
    baselines = baseline_metrics(actor_path, seed=seed + 900_000, pairs=baseline_pairs)
    values = (tactical, strategic, resource_efficiency, ensemble, heldout, baselines)
    if not all(
        math.isfinite(float(value))
        for group in values
        for value in group.values()
        if isinstance(value, (int, float))
    ):
        raise RuntimeError("checkpoint diagnostics produced a non-finite metric")
    return {
        "tactical": tactical,
        "strategic": strategic,
        "resource_efficiency": resource_efficiency,
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
    "resource_efficiency_decision_suite",
    "resource_efficiency_metrics",
    "strategic_decision_suite",
    "strategic_metrics",
    "tactical_metrics",
]
