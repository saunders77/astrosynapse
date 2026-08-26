"""Stateless checkpoint advice for manually tracked tabletop games.

The browser's ordinary ``/games`` sessions own a complete simulated game.  A
manual tabletop session is different: the client is the source of truth for
the cards and submits one public information state at a time.  This module
validates that state, hydrates the engine's immutable observation/action types,
and scores the resulting decision with the same :class:`ActorChooser` used by
the deployed play surface.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cards import ALL_CARDS, CARD_BY_ID, EXPLORER, Card, CardType
from .engine import (
    Action,
    ActionKind,
    Decision,
    DecisionFamily,
    InPlayObservation,
    Observation,
)
from .play import ActorChooser


class AdvisorInputError(ValueError):
    """The supplied public state cannot describe a scoreable decision."""


class AdvisorModelError(RuntimeError):
    """The selected checkpoint actor cannot score the supplied decision."""


class CardReference(BaseModel):
    """A catalog card object; extra catalog fields are accepted and ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    card_id: int = Field(ge=0, le=len(ALL_CARDS) - 1)

    def card(self) -> Card:
        try:
            return CARD_BY_ID[self.card_id]
        except KeyError as error:  # Defensive if the catalog ever becomes sparse.
            raise ValueError(f"unknown card_id {self.card_id}") from error


class InPlayReference(BaseModel):
    """The status-bearing card shape emitted by ``InPlayObservation.to_dict``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    card: CardReference
    activated: bool
    ally_triggered: bool
    copied_from_stealth_needle: bool = False

    def in_play(self) -> InPlayObservation:
        card = self.card.card()
        requires_activation = (
            card.is_base
            and card.card_id not in (19, 29)
            and bool(card.combat or card.authority or card.trade or card.primary)
        )
        return InPlayObservation(
            card=card,
            # Continuous/no-resource bases are always active in engine
            # observations even if a manual client initializes every carried
            # base as "ready" at the start of a turn.
            activated=self.activated if requires_activation or card.is_ship else True,
            ally_triggered=self.ally_triggered,
            copied_from_stealth_needle=self.copied_from_stealth_needle,
        )


class AdvisorObservation(BaseModel):
    """Complete public checkpoint observation accepted by the advisor.

    The field names intentionally match :class:`astro2.engine.Observation` so
    the normal game payload can be copied directly.  Card values are the full
    objects returned by ``GET /api/cards``; only their stable ``card_id`` is
    trusted when hydrating the canonical immutable definitions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(default=2, ge=2, le=2)
    player_id: int = Field(ge=0, le=1)
    active_player: int = Field(ge=0, le=1)
    starting_player: int = Field(ge=0, le=1)
    is_starting_player: bool
    turn: int = Field(ge=0)
    action_number: int = Field(ge=0)
    own_authority: int = Field(ge=0)
    opponent_authority: int = Field(ge=0)
    opponent_pending_discard: int = Field(ge=0)
    combat: int = Field(ge=0)
    trade: int = Field(ge=0)
    pending_discard: int = Field(ge=0)
    hand: list[CardReference]
    own_deck_count: int = Field(ge=0)
    own_deck: list[CardReference]
    own_known_top: list[CardReference]
    own_discard: list[CardReference]
    own_in_play: list[InPlayReference]
    opponent_hand_count: int = Field(ge=0)
    opponent_known_hand: list[CardReference]
    opponent_hidden: list[CardReference]
    opponent_deck_count: int = Field(ge=0)
    opponent_known_top: list[CardReference]
    opponent_discard: list[CardReference]
    opponent_in_play: list[InPlayReference]
    trade_row: list[CardReference] = Field(min_length=5, max_length=5)
    trade_deck_count: int = Field(ge=0)
    trade_deck: list[CardReference]
    explorers_remaining: int = Field(ge=0, le=10)
    explorer_supply: list[CardReference]
    scrap_heap: list[CardReference]
    next_ship_to_top: bool
    blob_cards_played: int = Field(ge=0)
    all_allied: bool
    fleet_active: bool

    @model_validator(mode="after")
    def validate_public_counts(self) -> AdvisorObservation:
        if self.active_player != self.player_id:
            raise ValueError("active_player must equal player_id when requesting checkpoint advice")
        if self.is_starting_player != (self.player_id == self.starting_player):
            raise ValueError("is_starting_player does not match player_id and starting_player")

        known_own_top = len(self.own_known_top)
        if self.own_deck_count != len(self.own_deck) + known_own_top:
            raise ValueError(
                "own_deck_count must equal len(own_deck) + len(own_known_top)"
            )

        known_opponent = len(self.opponent_known_hand) + len(self.opponent_known_top)
        public_opponent_total = self.opponent_hand_count + self.opponent_deck_count
        if len(self.opponent_hidden) + known_opponent != public_opponent_total:
            raise ValueError(
                "opponent_hidden plus known hand/top cards must equal "
                "opponent_hand_count + opponent_deck_count"
            )
        if len(self.opponent_known_hand) > self.opponent_hand_count:
            raise ValueError("opponent_known_hand exceeds opponent_hand_count")
        if len(self.opponent_known_top) > self.opponent_deck_count:
            raise ValueError("opponent_known_top exceeds opponent_deck_count")
        if self.trade_deck_count != len(self.trade_deck):
            raise ValueError("trade_deck_count must equal len(trade_deck)")
        if len(self.explorer_supply) != self.explorers_remaining:
            raise ValueError("explorer_supply must contain explorers_remaining cards")
        if any(card.card_id != EXPLORER.card_id for card in self.explorer_supply):
            raise ValueError("explorer_supply may contain only Explorer cards")

        own_cards = [item.card.card() for item in self.own_in_play]
        derived_all_allied = any(card.card_id == 19 for card in own_cards)
        derived_fleet_active = any(card.card_id == 29 for card in own_cards)
        if self.all_allied != derived_all_allied:
            raise ValueError("all_allied must match whether Mech World is in own_in_play")
        if self.fleet_active != derived_fleet_active:
            raise ValueError("fleet_active must match whether Fleet HQ is in own_in_play")
        return self

    def observation(self) -> Observation:
        def cards(items: list[CardReference]) -> tuple[Card, ...]:
            return tuple(item.card() for item in items)

        return Observation(
            version=self.version,
            player_id=self.player_id,
            active_player=self.active_player,
            starting_player=self.starting_player,
            is_starting_player=self.is_starting_player,
            turn=self.turn,
            action_number=self.action_number,
            own_authority=self.own_authority,
            opponent_authority=self.opponent_authority,
            opponent_pending_discard=self.opponent_pending_discard,
            combat=self.combat,
            trade=self.trade,
            pending_discard=self.pending_discard,
            hand=cards(self.hand),
            own_deck_count=self.own_deck_count,
            own_deck=cards(self.own_deck),
            own_known_top=cards(self.own_known_top),
            own_discard=cards(self.own_discard),
            own_in_play=tuple(item.in_play() for item in self.own_in_play),
            opponent_hand_count=self.opponent_hand_count,
            opponent_known_hand=cards(self.opponent_known_hand),
            opponent_hidden=cards(self.opponent_hidden),
            opponent_deck_count=self.opponent_deck_count,
            opponent_known_top=cards(self.opponent_known_top),
            opponent_discard=cards(self.opponent_discard),
            opponent_in_play=tuple(item.in_play() for item in self.opponent_in_play),
            trade_row=cards(self.trade_row),
            trade_deck_count=self.trade_deck_count,
            trade_deck=cards(self.trade_deck),
            explorers_remaining=self.explorers_remaining,
            explorer_supply=cards(self.explorer_supply),
            scrap_heap=cards(self.scrap_heap),
            next_ship_to_top=self.next_ship_to_top,
            blob_cards_played=self.blob_cards_played,
            all_allied=self.all_allied,
            fleet_active=self.fleet_active,
        )


class AdvisorAction(BaseModel):
    """Stable semantic action fields accepted for a nested engine decision."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    kind: ActionKind
    card_id: int = Field(default=-1, ge=-1, le=len(ALL_CARDS) - 1)
    target_card_id: int = Field(default=-1, ge=-1, le=len(ALL_CARDS) - 1)
    ability: str = Field(default="", max_length=80)
    source_zone: str = Field(default="", max_length=40)
    amount: int = Field(default=0, ge=0)
    amount2: int = Field(default=0, ge=0)

    def action(self) -> Action:
        return Action(
            kind=self.kind,
            card_id=self.card_id,
            target_card_id=self.target_card_id,
            ability=self.ability,
            source_zone=self.source_zone,
            amount=self.amount,
            amount2=self.amount2,
        )


class AdvisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: DecisionFamily = DecisionFamily.MAIN
    prompt: str = Field(default="", max_length=240)
    actions: list[AdvisorAction] = Field(default_factory=list, max_length=256)


class AdvisorEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=128)
    observation: AdvisorObservation
    decision: AdvisorDecision | None = None


class AdvisorScoredAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int
    label: str
    kind: ActionKind
    card_id: int
    target_card_id: int
    ability: str
    source_zone: str
    amount: int
    amount2: int
    model_value: float
    model_recommended: bool


class AdvisorEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: DecisionFamily
    prompt: str
    score_semantics: Literal["policy_probability", "win_outcome"]
    expected_win_rate: float | None
    actions: list[AdvisorScoredAction]


def _deduplicate_actions(actions: list[Action]) -> tuple[Action, ...]:
    unique: dict[tuple[Any, ...], Action] = {}
    for action in actions:
        unique.setdefault(action.semantic_key, action)
    return tuple(unique.values())


def main_phase_actions(observation: Observation) -> tuple[Action, ...]:
    """Generate the engine-equivalent semantic main-phase legal action set.

    Locators in ``Action.opaque`` are intentionally absent: this stateless
    endpoint recommends an action but never mutates a game.  The fields visible
    to the model exactly match the live engine's legal actions.
    """

    if observation.active_player != observation.player_id:
        raise AdvisorInputError("the checkpoint player is not active")
    if observation.pending_discard > 0:
        raise AdvisorInputError(
            "pending_discard is unresolved; supply the legal discard decision actions"
        )
    actions: list[Action] = []
    for card in observation.hand:
        actions.append(Action(ActionKind.PLAY_CARD, card_id=card.card_id, source_zone="hand"))
    for item in observation.own_in_play:
        card = item.card
        if card.is_base and not item.activated:
            actions.append(
                Action(ActionKind.ACTIVATE_BASE, card_id=card.card_id, source_zone="in_play")
            )
        if card.scrap:
            actions.append(
                Action(
                    ActionKind.SCRAP_FOR_ABILITY,
                    card_id=card.card_id,
                    ability=card.scrap,
                    source_zone="in_play",
                    amount=card.scrap_amount,
                )
            )

    if observation.combat > 0:
        outposts = [
            item
            for item in observation.opponent_in_play
            if item.card.card_type == CardType.OUTPOST
        ]
        targets = outposts or [
            item
            for item in observation.opponent_in_play
            if item.card.card_type == CardType.BASE
        ]
        for item in targets:
            if observation.combat >= item.card.defense:
                actions.append(
                    Action(
                        ActionKind.ATTACK_BASE,
                        target_card_id=item.card.card_id,
                        source_zone="opponent_in_play",
                        amount=item.card.defense,
                        amount2=observation.combat,
                    )
                )
        if not outposts:
            actions.append(Action(ActionKind.ATTACK_PLAYER, amount=observation.combat))

    for card in observation.trade_row:
        if card is not None and card.cost <= observation.trade:
            actions.append(
                Action(
                    ActionKind.ACQUIRE,
                    card_id=card.card_id,
                    source_zone="trade_row",
                    amount=card.cost,
                )
            )
    if observation.explorers_remaining and observation.trade >= EXPLORER.cost:
        actions.append(
            Action(
                ActionKind.ACQUIRE,
                card_id=EXPLORER.card_id,
                source_zone="explorer_supply",
                amount=EXPLORER.cost,
            )
        )
    actions.append(Action(ActionKind.END_TURN))
    return _deduplicate_actions(actions)


def decision_from_request(request: AdvisorEvaluateRequest) -> Decision:
    observation = request.observation.observation()
    supplied = request.decision
    if supplied is None or supplied.family == DecisionFamily.MAIN:
        family = DecisionFamily.MAIN
        prompt = supplied.prompt if supplied and supplied.prompt else "Main phase"
        actions = main_phase_actions(observation)
    else:
        if not supplied.actions:
            raise AdvisorInputError("non-main decisions must supply at least one action")
        family = supplied.family
        prompt = supplied.prompt or family.value.replace("_", " ").title()
        actions = _deduplicate_actions([item.action() for item in supplied.actions])
        _validate_supplied_actions(observation, family, actions)
    try:
        return Decision(family=family, observation=observation, actions=actions, prompt=prompt)
    except ValueError as error:
        raise AdvisorInputError(str(error)) from error


_FAMILY_ACTION_KINDS: dict[DecisionFamily, frozenset[ActionKind]] = {
    DecisionFamily.MAIN: frozenset(
        {
            ActionKind.PLAY_CARD,
            ActionKind.ACTIVATE_BASE,
            ActionKind.SCRAP_FOR_ABILITY,
            ActionKind.ATTACK_BASE,
            ActionKind.ATTACK_PLAYER,
            ActionKind.ACQUIRE,
            ActionKind.END_TURN,
        }
    ),
    DecisionFamily.DISCARD: frozenset({ActionKind.DISCARD_CARD, ActionKind.DECLINE}),
    DecisionFamily.SCRAP: frozenset({ActionKind.SCRAP_CARD, ActionKind.DECLINE}),
    DecisionFamily.ABILITY_MODE: frozenset({ActionKind.CHOOSE_MODE}),
    DecisionFamily.COPY_SHIP: frozenset({ActionKind.COPY_SHIP}),
    DecisionFamily.DESTROY_BASE: frozenset({ActionKind.DESTROY_BASE, ActionKind.DECLINE}),
    DecisionFamily.SCRAP_TRADE_ROW: frozenset(
        {ActionKind.SCRAP_TRADE_ROW, ActionKind.DECLINE}
    ),
    DecisionFamily.FREE_ACQUIRE: frozenset({ActionKind.FREE_ACQUIRE, ActionKind.DECLINE}),
}

_SOURCE_CARD_REQUIRED = frozenset(
    {
        ActionKind.PLAY_CARD,
        ActionKind.ACTIVATE_BASE,
        ActionKind.SCRAP_FOR_ABILITY,
        ActionKind.ACQUIRE,
        ActionKind.DISCARD_CARD,
        ActionKind.SCRAP_CARD,
        ActionKind.CHOOSE_MODE,
        ActionKind.COPY_SHIP,
        ActionKind.DESTROY_BASE,
        ActionKind.SCRAP_TRADE_ROW,
        ActionKind.FREE_ACQUIRE,
        ActionKind.DECLINE,
    }
)

_TARGET_CARD_REQUIRED = frozenset(
    {
        ActionKind.ATTACK_BASE,
        ActionKind.COPY_SHIP,
        ActionKind.DESTROY_BASE,
        ActionKind.SCRAP_TRADE_ROW,
        ActionKind.FREE_ACQUIRE,
    }
)


def _validate_supplied_actions(
    observation: Observation,
    family: DecisionFamily,
    actions: tuple[Action, ...],
) -> None:
    allowed = _FAMILY_ACTION_KINDS[family]
    own_hand = {card.card_id for card in observation.hand}
    own_discard = {card.card_id for card in observation.own_discard}
    own_in_play = {item.card.card_id for item in observation.own_in_play}
    opponent_bases = {
        item.card.card_id for item in observation.opponent_in_play if item.card.is_base
    }
    trade_row = {card.card_id for card in observation.trade_row if card is not None}

    for action in actions:
        if action.kind not in allowed:
            raise AdvisorInputError(
                f"{action.kind.value} is not valid for decision family {family.value}"
            )
        if action.kind in _SOURCE_CARD_REQUIRED and action.card_id < 0:
            raise AdvisorInputError(f"{action.kind.value} requires a defined card_id")
        if action.kind in _TARGET_CARD_REQUIRED and action.target_card_id < 0:
            raise AdvisorInputError(
                f"{action.kind.value} requires a defined target_card_id"
            )
        if action.kind in {ActionKind.PLAY_CARD, ActionKind.DISCARD_CARD}:
            if action.card_id not in own_hand:
                raise AdvisorInputError(f"{action.label} does not reference a card in hand")
        elif action.kind == ActionKind.SCRAP_CARD:
            source = own_discard if action.source_zone == "discard" else own_hand
            if action.source_zone not in {"hand", "discard"} or action.card_id not in source:
                raise AdvisorInputError(
                    f"{action.label} does not reference its declared hand/discard zone"
                )
        elif action.kind in {
            ActionKind.ACTIVATE_BASE,
            ActionKind.SCRAP_FOR_ABILITY,
            ActionKind.CHOOSE_MODE,
            ActionKind.COPY_SHIP,
            ActionKind.DESTROY_BASE,
            ActionKind.SCRAP_TRADE_ROW,
            ActionKind.FREE_ACQUIRE,
            ActionKind.DECLINE,
        } and action.card_id not in own_in_play:
            raise AdvisorInputError(
                f"{action.label} does not reference a card in own_in_play"
            )

        if action.kind in {ActionKind.ATTACK_BASE, ActionKind.DESTROY_BASE}:
            if action.target_card_id not in opponent_bases:
                raise AdvisorInputError(
                    f"{action.label} does not target a base in opponent_in_play"
                )
        elif action.kind == ActionKind.COPY_SHIP:
            target = CARD_BY_ID[action.target_card_id]
            if action.target_card_id not in own_in_play or not target.is_ship:
                raise AdvisorInputError(
                    f"{action.label} does not target a ship in own_in_play"
                )
        elif action.kind in {ActionKind.SCRAP_TRADE_ROW, ActionKind.FREE_ACQUIRE}:
            target = CARD_BY_ID[action.target_card_id]
            if action.target_card_id not in trade_row:
                raise AdvisorInputError(
                    f"{action.label} does not target a card in the trade row"
                )
            if action.kind == ActionKind.FREE_ACQUIRE and not target.is_ship:
                raise AdvisorInputError(f"{action.label} must target a ship")


@dataclass(slots=True)
class _ChooserEntry:
    modified_ns: int
    size: int
    chooser: ActorChooser
    lock: threading.RLock


class CheckpointAdvisor:
    """Bounded, file-aware cache of deployment actor choosers."""

    def __init__(
        self,
        *,
        maximum_cached_actors: int = 8,
        chooser_factory: Callable[[str | Path], ActorChooser] = ActorChooser,
    ) -> None:
        if maximum_cached_actors < 1:
            raise ValueError("maximum_cached_actors must be positive")
        self.maximum_cached_actors = int(maximum_cached_actors)
        self.chooser_factory = chooser_factory
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _ChooserEntry] = OrderedDict()

    def _chooser(self, actor_path: str | Path) -> _ChooserEntry:
        path = Path(actor_path).expanduser().resolve()
        try:
            stat = path.stat()
        except OSError as error:
            raise AdvisorModelError(f"checkpoint actor snapshot is unavailable: {path}") from error
        key = str(path)
        with self._lock:
            entry = self._entries.get(key)
            signature = (stat.st_mtime_ns, stat.st_size)
            if entry is None or (entry.modified_ns, entry.size) != signature:
                try:
                    chooser = self.chooser_factory(path)
                except (OSError, ValueError, KeyError) as error:
                    raise AdvisorModelError(f"checkpoint actor could not be loaded: {error}") from error
                entry = _ChooserEntry(*signature, chooser, threading.RLock())
                self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.maximum_cached_actors:
                self._entries.popitem(last=False)
            return entry

    def evaluate(
        self,
        actor_path: str | Path,
        request: AdvisorEvaluateRequest,
    ) -> AdvisorEvaluation:
        decision = decision_from_request(request)
        entry = self._chooser(actor_path)
        try:
            with entry.lock:
                recommended, values, expected_win_rate = entry.chooser.score(decision)
        except (IndexError, RuntimeError, ValueError) as error:
            raise AdvisorModelError(f"checkpoint could not score this decision: {error}") from error

        option_values = np.asarray(values, dtype=np.float64)
        if option_values.shape != (len(decision.actions),):
            raise AdvisorModelError(
                "checkpoint returned an action-value vector with the wrong length"
            )
        if not np.all(np.isfinite(option_values)):
            raise AdvisorModelError("checkpoint returned a non-finite action value")
        if np.any(option_values < 0.0) or np.any(option_values > 1.0):
            raise AdvisorModelError("checkpoint returned an action value outside [0, 1]")
        if recommended < 0 or recommended >= len(decision.actions):
            raise AdvisorModelError("checkpoint recommended an action outside the legal set")
        if expected_win_rate is not None and not math.isfinite(expected_win_rate):
            raise AdvisorModelError("checkpoint returned a non-finite expected win rate")
        if expected_win_rate is not None and not 0.0 <= expected_win_rate <= 1.0:
            raise AdvisorModelError("checkpoint returned an expected win rate outside [0, 1]")

        score_semantics = (
            "policy_probability"
            if entry.chooser.actor.spec.objective_version >= 2
            else "win_outcome"
        )
        scored_actions = []
        for index, (action, value) in enumerate(
            zip(decision.actions, option_values, strict=True)
        ):
            scored_actions.append(
                AdvisorScoredAction(
                    id=index,
                    label=action.label,
                    kind=action.kind.value,
                    card_id=action.card_id,
                    target_card_id=action.target_card_id,
                    ability=action.ability,
                    source_zone=action.source_zone,
                    amount=action.amount,
                    amount2=action.amount2,
                    model_value=float(value),
                    model_recommended=index == recommended,
                )
            )
        return AdvisorEvaluation(
            family=decision.family.value,
            prompt=decision.prompt,
            score_semantics=score_semantics,
            expected_win_rate=(
                float(expected_win_rate) if expected_win_rate is not None else None
            ),
            actions=scored_actions,
        )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def card_catalog() -> list[dict[str, Any]]:
    """Return the stable catalog objects accepted in advisor observations."""

    return [card.to_dict() for card in ALL_CARDS]


__all__ = [
    "AdvisorEvaluateRequest",
    "AdvisorEvaluation",
    "AdvisorInputError",
    "AdvisorModelError",
    "CheckpointAdvisor",
    "card_catalog",
    "decision_from_request",
    "main_phase_actions",
]
