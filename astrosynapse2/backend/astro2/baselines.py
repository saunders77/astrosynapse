"""Legal, reproducible reference choosers for evaluation and league diversity."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from .cards import CARD_BY_ID, Card, Faction
from .engine import Action, ActionKind, Decision, DecisionFamily


def _card(action: Action) -> Card | None:
    card_id = action.target_card_id if action.target_card_id >= 0 else action.card_id
    return CARD_BY_ID.get(card_id)


def _effect_value(effect: str, amount: int, style: str) -> float:
    weights = {
        "gain_combat": 1.2 if style == "aggressive" else 0.9,
        "gain_trade": 1.25 if style == "economy" else 0.9,
        "gain_authority": 0.7,
        "draw": 2.8,
        "draw_two": 5.6,
        "opponent_discard": 2.0,
        "ship_top": 1.8,
        "scrap_any": 2.2,
        "scrap_two_draw": 4.0,
        "draw_then_scrap": 2.8,
        "destroy_base": 2.5,
        "copy_ship": 2.0,
        "recycle": 2.0,
        "free_ship": 4.0,
        "scrap_trade_row": 0.8,
        "destroy_and_scrap": 3.2,
        "draw_destroy": 3.5,
        "all_ally": 2.5,
        "fleet_hq": 3.0,
    }
    return weights.get(effect, 0.5) + 0.35 * amount


def _card_value(card: Card | None, style: str, faction_bonus: float = 0.0) -> float:
    if card is None:
        return 0.0
    combat_weight = 1.45 if style == "aggressive" else 0.95
    trade_weight = 1.5 if style == "economy" else 1.0
    value = (
        combat_weight * card.combat
        + trade_weight * card.trade
        + 0.75 * card.authority
        + 0.45 * card.defense
        + _effect_value(card.primary, 0, style)
        + _effect_value(card.ally, card.ally_amount, style)
        + 0.45 * _effect_value(card.scrap, card.scrap_amount, style)
    )
    if card.faction != Faction.UNALIGNED:
        value += faction_bonus
    if card.is_base:
        value += 1.2
    return value


def _faction_counts(decision: Decision) -> dict[Faction, int]:
    observation = decision.observation
    cards = list(observation.hand)
    cards.extend(observation.own_deck)
    cards.extend(observation.own_discard)
    cards.extend(item.card for item in observation.own_in_play)
    counts = {faction: 0 for faction in Faction}
    for card in cards:
        counts[card.faction] += 1
    return counts


def _stable_best(actions: tuple[Action, ...], scores: list[float]) -> Action:
    best_score = max(scores)
    best = [action for action, score in zip(actions, scores, strict=True) if score == best_score]
    return min(best, key=lambda action: action.semantic_key)


@dataclass(frozen=True, slots=True)
class FirstLegalChooser:
    """A deterministic smoke-test opponent, independent of action ordering."""

    def __call__(self, _player_id: int, decision: Decision) -> Action:
        return min(decision.actions, key=lambda action: action.semantic_key)


class RandomChooser:
    """Uniform legal chooser with an isolated, reproducible RNG stream."""

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def __call__(self, _player_id: int, decision: Decision) -> Action:
        return decision.actions[self._rng.randrange(len(decision.actions))]


class HeuristicChooser:
    """Fast reference policy with economy, aggression, or balanced priorities."""

    def __init__(self, style: Literal["balanced", "economy", "aggressive"] = "balanced"):
        if style not in {"balanced", "economy", "aggressive"}:
            raise ValueError(f"unknown heuristic style: {style!r}")
        self.style = style

    def __call__(self, _player_id: int, decision: Decision) -> Action:
        faction_counts = _faction_counts(decision)
        scores = [self._score(decision, action, faction_counts) for action in decision.actions]
        return _stable_best(decision.actions, scores)

    def _score(
        self,
        decision: Decision,
        action: Action,
        faction_counts: dict[Faction, int],
    ) -> float:
        kind = action.kind
        card = _card(action)
        faction_bonus = 0.0
        if card is not None:
            faction_bonus = min(4.0, 0.35 * faction_counts.get(card.faction, 0))
        value = _card_value(card, self.style, faction_bonus)

        if decision.family == DecisionFamily.DISCARD:
            if kind == ActionKind.DECLINE:
                return 0.25
            return 20.0 - value
        if decision.family == DecisionFamily.SCRAP:
            if kind == ActionKind.DECLINE:
                return 0.0
            # Thinning Scouts and Vipers is normally useful; do not casually
            # delete a developed card just because an optional scrap exists.
            return 6.0 - value + (2.0 if action.card_id in {0, 1} else 0.0)
        if decision.family == DecisionFamily.COPY_SHIP:
            return value
        if decision.family == DecisionFamily.DESTROY_BASE:
            return -0.5 if kind == ActionKind.DECLINE else 10.0 + value
        if decision.family == DecisionFamily.SCRAP_TRADE_ROW:
            return -0.25 if kind == ActionKind.DECLINE else 0.5 + value
        if decision.family == DecisionFamily.FREE_ACQUIRE:
            return -0.25 if kind == ActionKind.DECLINE else value
        if decision.family == DecisionFamily.ABILITY_MODE:
            return self._mode_score(action)

        if kind in {ActionKind.PLAY_CARD, ActionKind.ACTIVATE_BASE}:
            return 1_000.0 + value
        if kind == ActionKind.SCRAP_FOR_ABILITY:
            return 610.0 + _effect_value(action.ability, action.amount, self.style) - 0.3 * value
        if kind == ActionKind.ATTACK_BASE:
            return 760.0 + 2.0 * action.amount + value
        if kind == ActionKind.ATTACK_PLAYER:
            multiplier = 2.1 if self.style == "aggressive" else 1.4
            return 800.0 + multiplier * action.amount
        if kind == ActionKind.ACQUIRE:
            return 500.0 + value - 0.05 * action.amount
        if kind == ActionKind.END_TURN:
            return -1_000.0
        return 0.0

    def _mode_score(self, action: Action) -> float:
        ability = action.ability
        amount = action.amount
        if ability == "gain_combat":
            return amount * (1.6 if self.style == "aggressive" else 1.0)
        if ability == "gain_trade":
            return amount * (1.7 if self.style == "economy" else 1.05)
        if ability == "gain_authority":
            return amount * 0.8
        if ability == "draw":
            return amount * 2.8
        if ability == "cycle":
            return amount * 1.5
        return _effect_value(ability, amount, self.style)


BASELINE_NAMES = ("first", "random", "balanced", "economy", "aggressive")


def make_baseline(name: str, seed: int = 0):
    """Create a top-level, pickle-friendly engine chooser by stable name."""

    normalized = name.strip().lower()
    if normalized == "first":
        return FirstLegalChooser()
    if normalized == "random":
        return RandomChooser(seed)
    if normalized in {"balanced", "economy", "aggressive"}:
        return HeuristicChooser(normalized)
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")


__all__ = [
    "BASELINE_NAMES",
    "FirstLegalChooser",
    "HeuristicChooser",
    "RandomChooser",
    "make_baseline",
]
