"""Deterministic held-out and tactical checkpoint diagnostics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import make_baseline
from .encoding import Encoder
from .engine import ActionKind, Decision, Game, GameConfig, model_action_indices
from .model import NumpyActor
from .selfplay import collect_game


def _behavioral_suite(*, seed: int, games: int, limit: int = 512) -> tuple[Decision, ...]:
    decisions: list[Decision] = []
    styles = ("balanced", "economy", "aggressive")
    for game_index in range(games):
        first = make_baseline(styles[game_index % len(styles)], seed + 2 * game_index)
        second = make_baseline(
            styles[(game_index + 1) % len(styles)], seed + 2 * game_index + 1
        )

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
    encoder = Encoder()
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


def heldout_outcome_metrics(
    actor: NumpyActor,
    *,
    seed: int,
    games: int,
) -> dict[str, Any]:
    encoder = Encoder()
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
        logits = actor.predict(states, actions, families)
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
    encoder = Encoder()

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
    heldout = heldout_outcome_metrics(actor, seed=seed + 800_000, games=games)
    baselines = baseline_metrics(actor_path, seed=seed + 900_000, pairs=baseline_pairs)
    values = (tactical, heldout, baselines)
    if not all(
        math.isfinite(float(value))
        for group in values
        for value in group.values()
        if isinstance(value, (int, float))
    ):
        raise RuntimeError("checkpoint diagnostics produced a non-finite metric")
    return {"tactical": tactical, "heldout": heldout, "baselines": baselines}


__all__ = [
    "baseline_metrics",
    "checkpoint_diagnostics",
    "heldout_outcome_metrics",
    "tactical_metrics",
]
