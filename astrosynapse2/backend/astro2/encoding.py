"""Stable, information-set encodings for Star Realms decisions.

The simulator is free to reorder a hand, draw pile, trade row, or collection of
identical cards.  Those positions are implementation details, not game facts.
This module therefore represents every zone as *exact per-card counts* and
every candidate as a semantic action.  In particular, action ``slot``,
``index`` and ``position`` fields are deliberately never read.

The small amount of duck typing at the edge is intentional.  It accepts both
the immutable Astro 2 engine objects and legacy ``sim.py`` observations while
keeping the numeric contract used by the model explicit and versionable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

# IDs are the canonical base-set order used by sim.py.  Names, unlike Python
# object hashes or list positions, are stable across processes and checkpoints.
BASE_CARD_NAMES = (
    "Scout",
    "Viper",
    "Explorer",
    "Battle Blob",
    "Battle Pod",
    "Blob Carrier",
    "Blob Destroyer",
    "Blob Fighter",
    "Blob Wheel",
    "Blob World",
    "Mothership",
    "Ram",
    "The Hive",
    "Trade Pod",
    "Battle Mech",
    "Battle Station",
    "Brain World",
    "Junkyard",
    "Machine Base",
    "Mech World",
    "Missile Bot",
    "Missile Mech",
    "Patrol Mech",
    "Stealth Needle",
    "Supply Bot",
    "Trade Bot",
    "Battlecruiser",
    "Corvette",
    "Dreadnaught",
    "Fleet HQ",
    "Imperial Fighter",
    "Imperial Frigate",
    "Recycling Station",
    "Royal Redoubt",
    "Space Station",
    "Survey Ship",
    "War World",
    "Barter World",
    "Central Office",
    "Command Ship",
    "Cutter",
    "Defense Center",
    "Embassy Yacht",
    "Federation Shuttle",
    "Flagship",
    "Freighter",
    "Port of Call",
    "Trade Escort",
    "Trading Post",
)


class DecisionFamily(IntEnum):
    """Checkpoint-stable heads for the game's heterogeneous chooser calls."""

    MAIN = 0
    DISCARD = 1
    SCRAP = 2
    DESTROY_BASE = 3
    SCRAP_TRADE_ROW = 4
    COPY_SHIP = 5
    FREE_ACQUIRE = 6
    ABILITY_MODE = 7


class ActionKind(IntEnum):
    UNKNOWN = 0
    PLAY = 1
    ACTIVATE_ABILITY = 2
    SCRAP_FROM_PLAY = 3
    ATTACK_BASE = 4
    ATTACK_PLAYER = 5
    ACQUIRE = 6
    END_TURN = 7
    DISCARD = 8
    DECLINE_DISCARD = 9
    SCRAP_CARD = 10
    DECLINE_SCRAP = 11
    DESTROY_BASE = 12
    DECLINE_DESTROY_BASE = 13
    SCRAP_TRADE_ROW = 14
    DECLINE_SCRAP_TRADE_ROW = 15
    COPY_SHIP = 16
    DECLINE_COPY = 17
    FREE_ACQUIRE = 18
    GAIN_ATTACK = 19
    GAIN_TRADE = 20
    GAIN_AUTHORITY = 21
    DRAW = 22
    RECYCLE = 23
    DECLINE = 24


class Zone(IntEnum):
    OWN_HAND = 0
    OWN_DRAW = 1
    OWN_KNOWN_TOP = 2
    OWN_DISCARD = 3
    OWN_IN_PLAY = 4
    OPPONENT_HIDDEN = 5
    OPPONENT_KNOWN_HAND = 6
    OPPONENT_KNOWN_TOP = 7
    OPPONENT_DISCARD = 8
    OPPONENT_IN_PLAY = 9
    TRADE_ROW = 10
    TRADE_DECK = 11
    SCRAP_HEAP = 12
    EXPLORER_SUPPLY = 13


class EffectKind(IntEnum):
    NONE = 0
    ATTACK = 1
    TRADE = 2
    AUTHORITY = 3
    DRAW = 4
    DISCARD = 5
    SCRAP = 6
    DESTROY_BASE = 7
    SCRAP_TRADE_ROW = 8
    FREE_ACQUIRE = 9
    COPY_SHIP = 10
    TOP_DECK = 11
    ALLY = 12
    OTHER = 13


FAMILY_COUNT = len(DecisionFamily)
ACTION_KIND_COUNT = len(ActionKind)
ZONE_COUNT = len(Zone)
EFFECT_COUNT = len(EffectKind)


_ACTION_ALIASES: dict[str, ActionKind] = {
    "play": ActionKind.PLAY,
    "play_card": ActionKind.PLAY,
    "ability_option": ActionKind.ACTIVATE_ABILITY,
    "activate": ActionKind.ACTIVATE_ABILITY,
    "activate_ability": ActionKind.ACTIVATE_ABILITY,
    "activate_base": ActionKind.ACTIVATE_ABILITY,
    "scrap_from_play": ActionKind.SCRAP_FROM_PLAY,
    "scrap_for_ability": ActionKind.SCRAP_FROM_PLAY,
    "attack": ActionKind.ATTACK_BASE,
    "attack_base": ActionKind.ATTACK_BASE,
    "attack_opponent": ActionKind.ATTACK_PLAYER,
    "attack_player": ActionKind.ATTACK_PLAYER,
    "acquire": ActionKind.ACQUIRE,
    "buy": ActionKind.ACQUIRE,
    "end_turn": ActionKind.END_TURN,
    "discard": ActionKind.DISCARD,
    "discard_card": ActionKind.DISCARD,
    "discard_normal": ActionKind.DISCARD,
    "discard_draw": ActionKind.DISCARD,
    "no_discard": ActionKind.DECLINE_DISCARD,
    "nodiscard": ActionKind.DECLINE_DISCARD,
    "scrap": ActionKind.SCRAP_CARD,
    "scrap_card": ActionKind.SCRAP_CARD,
    "scrap_from_hand": ActionKind.SCRAP_CARD,
    "scrap_from_hand_normal": ActionKind.SCRAP_CARD,
    "scrap_from_hand_draw": ActionKind.SCRAP_CARD,
    "scrap_from_discard": ActionKind.SCRAP_CARD,
    "scrap_from_discard_normal": ActionKind.SCRAP_CARD,
    "scrap_from_discard_draw": ActionKind.SCRAP_CARD,
    "no_scrap": ActionKind.DECLINE_SCRAP,
    "no_scrap_from_hand": ActionKind.DECLINE_SCRAP,
    "kill_base": ActionKind.DESTROY_BASE,
    "killbase": ActionKind.DESTROY_BASE,
    "destroy_base": ActionKind.DESTROY_BASE,
    "no_kill": ActionKind.DECLINE_DESTROY_BASE,
    "nokill": ActionKind.DECLINE_DESTROY_BASE,
    "row_scrap": ActionKind.SCRAP_TRADE_ROW,
    "rowscrap": ActionKind.SCRAP_TRADE_ROW,
    "scrap_trade_row": ActionKind.SCRAP_TRADE_ROW,
    "no_row_scrap": ActionKind.DECLINE_SCRAP_TRADE_ROW,
    "copy_ship": ActionKind.COPY_SHIP,
    "copyship": ActionKind.COPY_SHIP,
    "no_copy": ActionKind.DECLINE_COPY,
    "nocopy": ActionKind.DECLINE_COPY,
    "free_acquire": ActionKind.FREE_ACQUIRE,
    "free_buy": ActionKind.FREE_ACQUIRE,
    "gain_attack": ActionKind.GAIN_ATTACK,
    "gainattack": ActionKind.GAIN_ATTACK,
    "gain_trade": ActionKind.GAIN_TRADE,
    "trade": ActionKind.GAIN_TRADE,
    "gain_authority": ActionKind.GAIN_AUTHORITY,
    "authority": ActionKind.GAIN_AUTHORITY,
    "draw": ActionKind.DRAW,
    "switch": ActionKind.RECYCLE,
    "recycle": ActionKind.RECYCLE,
    "decline": ActionKind.DECLINE,
}

_FAMILY_ALIASES: dict[str, DecisionFamily] = {
    "main": DecisionFamily.MAIN,
    "main_phase": DecisionFamily.MAIN,
    "discard": DecisionFamily.DISCARD,
    "scrap": DecisionFamily.SCRAP,
    "destroy_base": DecisionFamily.DESTROY_BASE,
    "target_base": DecisionFamily.DESTROY_BASE,
    "scrap_trade_row": DecisionFamily.SCRAP_TRADE_ROW,
    "trade_row_scrap": DecisionFamily.SCRAP_TRADE_ROW,
    "copy_ship": DecisionFamily.COPY_SHIP,
    "free_acquire": DecisionFamily.FREE_ACQUIRE,
    "free_buy": DecisionFamily.FREE_ACQUIRE,
    "ability": DecisionFamily.ABILITY_MODE,
    "ability_mode": DecisionFamily.ABILITY_MODE,
}

_ZONE_ALIASES: dict[Zone, tuple[str, ...]] = {
    Zone.OWN_HAND: ("own_hand", "hand"),
    Zone.OWN_DRAW: ("own_draw", "own_deck", "draw_pile", "deck", "scramble_deck"),
    Zone.OWN_KNOWN_TOP: ("own_known_top", "known_top", "top_cards"),
    Zone.OWN_DISCARD: ("own_discard", "discard", "discard_pile"),
    Zone.OWN_IN_PLAY: ("own_in_play", "cards_in_play", "in_play"),
    Zone.OPPONENT_HIDDEN: (
        "opponent_hidden",
        "opponent_draw_and_hand",
        "opponent_scramble_deck_and_hand",
    ),
    Zone.OPPONENT_KNOWN_HAND: ("opponent_known_hand", "opponent_hand_cards"),
    Zone.OPPONENT_KNOWN_TOP: ("opponent_known_top", "opponent_top_cards"),
    Zone.OPPONENT_DISCARD: ("opponent_discard", "opponent_discard_pile"),
    Zone.OPPONENT_IN_PLAY: ("opponent_in_play", "opponent_cards_in_play"),
    Zone.TRADE_ROW: ("trade_row",),
    Zone.TRADE_DECK: ("trade_deck", "trade_draw", "ready_cards"),
    Zone.SCRAP_HEAP: ("scrap_heap", "scrapped", "removed_cards"),
    Zone.EXPLORER_SUPPLY: ("explorer_supply",),
}

_STATE_SCALARS = (
    # Canonical label, accepted observation aliases, scaling constant.
    ("is_starting_player", ("is_starting_player",), 1.0),
    ("authority", ("authority", "own_authority"), 50.0),
    ("opponent_authority", ("opponent_authority",), 50.0),
    ("attack", ("attack", "combat"), 20.0),
    ("trade", ("trade",), 15.0),
    ("must_discard", ("must_discard", "pending_discard"), 5.0),
    ("opponent_must_discard", ("opponent_must_discard", "opponent_pending_discard"), 5.0),
    ("next_ship_top", ("next_ship_top", "next_ship_to_top"), 1.0),
    ("blob_play_count", ("blob_play_count", "blob_cards_played"), 5.0),
    ("all_allied", ("all_allied",), 1.0),
    ("fleet_active", ("fleet_active",), 1.0),
    ("turn", ("turn", "turns"), 50.0),
    ("action_number", ("action_number", "decision_number"), 50.0),
    ("own_deck_count", ("own_deck_count",), 20.0),
    ("opponent_hand_count", ("opponent_hand_count",), 10.0),
    ("opponent_deck_count", ("opponent_deck_count",), 20.0),
    ("trade_deck_count", ("trade_deck_count",), 80.0),
    ("explorers_remaining", ("explorers_remaining",), 10.0),
)

# cost, combat, authority, trade, defense, faction(5), card type(3)
CARD_ATTRIBUTE_SIZE = 13

# amount, cost, attack available, target defense, trade available, draw count,
# discard count, scrap count, optional, required, top-deck, current authority
ACTION_NUMERIC_SIZE = 12
KNOWN_TOP_SLOTS = 3
_IN_PLAY_STATUS_FIELDS = (
    "ready",
    "ally_triggered",
    "copied_from_stealth_needle",
)
_IN_PLAY_STATUS_SIDES = ("own_in_play", "opponent_in_play")


def _snake(value: Any) -> str:
    if hasattr(value, "name"):
        value = value.name
    text = str(value).strip().replace("-", "_").replace(" ", "_")
    text = text.lower() if text.isupper() else re.sub(r"(?<!^)(?=[A-Z])", "_", text).lower()
    return re.sub(r"_+", "_", text).strip("_")


def _lookup(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        normalized = {_snake(key): item for key, item in value.items()}
        for name in names:
            if _snake(name) in normalized:
                return normalized[_snake(name)]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
        camel = name.split("_")[0] + "".join(part.title() for part in name.split("_")[1:])
        if hasattr(value, camel):
            return getattr(value, camel)
    return default


def _raw_action_kind(action: Any) -> Any:
    if isinstance(action, (tuple, list)) and action:
        return action[0]
    return _lookup(action, "kind", "action_kind", "type", "verb", default="unknown")


def action_kind(action: Any) -> ActionKind:
    raw = _raw_action_kind(action)
    if isinstance(raw, ActionKind):
        return raw
    name = _snake(raw)
    if name == "choose_mode":
        ability = _snake(_lookup(action, "ability", "effect", "mode", default=""))
        return {
            "gain_combat": ActionKind.GAIN_ATTACK,
            "gain_attack": ActionKind.GAIN_ATTACK,
            "gain_trade": ActionKind.GAIN_TRADE,
            "gain_authority": ActionKind.GAIN_AUTHORITY,
            "draw": ActionKind.DRAW,
            "cycle": ActionKind.RECYCLE,
            "recycle": ActionKind.RECYCLE,
        }.get(ability, ActionKind.UNKNOWN)
    if name in _ACTION_ALIASES:
        return _ACTION_ALIASES[name]
    # Legacy action names combine the verb and draw/normal mode.
    if name.startswith("discard_"):
        return ActionKind.DISCARD
    if name.startswith("scrap_from_hand") or name.startswith("scrap_from_discard"):
        return ActionKind.SCRAP_CARD
    return ActionKind.UNKNOWN


_KINDS_BY_FAMILY: dict[DecisionFamily, frozenset[ActionKind]] = {
    DecisionFamily.MAIN: frozenset(
        {
            ActionKind.PLAY,
            ActionKind.ACTIVATE_ABILITY,
            ActionKind.SCRAP_FROM_PLAY,
            ActionKind.ATTACK_BASE,
            ActionKind.ATTACK_PLAYER,
            ActionKind.ACQUIRE,
            ActionKind.END_TURN,
        }
    ),
    DecisionFamily.DISCARD: frozenset({ActionKind.DISCARD, ActionKind.DECLINE_DISCARD}),
    DecisionFamily.SCRAP: frozenset({ActionKind.SCRAP_CARD, ActionKind.DECLINE_SCRAP}),
    DecisionFamily.DESTROY_BASE: frozenset(
        {ActionKind.DESTROY_BASE, ActionKind.DECLINE_DESTROY_BASE}
    ),
    DecisionFamily.SCRAP_TRADE_ROW: frozenset(
        {ActionKind.SCRAP_TRADE_ROW, ActionKind.DECLINE_SCRAP_TRADE_ROW}
    ),
    DecisionFamily.COPY_SHIP: frozenset({ActionKind.COPY_SHIP, ActionKind.DECLINE_COPY}),
    DecisionFamily.FREE_ACQUIRE: frozenset({ActionKind.FREE_ACQUIRE}),
    DecisionFamily.ABILITY_MODE: frozenset(
        {
            ActionKind.GAIN_ATTACK,
            ActionKind.GAIN_TRADE,
            ActionKind.GAIN_AUTHORITY,
            ActionKind.DRAW,
            ActionKind.RECYCLE,
        }
    ),
}


def decision_family(decision_or_actions: Any) -> DecisionFamily:
    """Resolve a family explicitly, then fall back to the legal verbs.

    A mixed or unknown verb set is rejected instead of being assigned via a
    process-dependent hash.  This makes accidental new engine decisions fail
    loudly before corrupting a replay file.
    """

    explicit = _lookup(decision_or_actions, "family", "decision_family", default=None)
    if explicit is not None:
        if isinstance(explicit, DecisionFamily):
            return explicit
        if isinstance(explicit, int) and 0 <= explicit < FAMILY_COUNT:
            return DecisionFamily(explicit)
        name = _snake(explicit)
        if name in _FAMILY_ALIASES:
            return _FAMILY_ALIASES[name]

    actions = _lookup(decision_or_actions, "actions", "options", "legal_actions", default=None)
    if actions is None:
        actions = decision_or_actions
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        actions = (actions,)
    kinds = {action_kind(action) for action in actions}
    if ActionKind.UNKNOWN in kinds:
        raise ValueError(f"unknown action kind in decision: {sorted(int(kind) for kind in kinds)}")
    matches = [family for family, members in _KINDS_BY_FAMILY.items() if kinds <= members]
    if len(matches) != 1:
        raise ValueError(f"ambiguous decision family for kinds: {sorted(kind.name for kind in kinds)}")
    return matches[0]


def _effect_kind(value: Any) -> EffectKind:
    name = _snake(value or "none").lstrip("_")
    if name in {"none", "used", ""}:
        return EffectKind.NONE
    if "attack" in name or "combat" in name or name in {"5_or_0_or_3", "2_or_3_or_0"}:
        return EffectKind.ATTACK
    if "trade" in name or name in {"0_or_2_or_2", "0_or_1_or_1"}:
        return EffectKind.TRADE
    if "authority" in name:
        return EffectKind.AUTHORITY
    if "draw" in name or name in {"5_or_draws", "recycle"}:
        return EffectKind.DRAW
    if "discard" in name:
        return EffectKind.DISCARD
    if "scrap" in name and "row" not in name and "destroy" not in name:
        return EffectKind.SCRAP
    if "kill" in name or "destroy" in name:
        return EffectKind.DESTROY_BASE
    if "row" in name:
        return EffectKind.SCRAP_TRADE_ROW
    if "free" in name:
        return EffectKind.FREE_ACQUIRE
    if "copy" in name:
        return EffectKind.COPY_SHIP
    if "top" in name:
        return EffectKind.TOP_DECK
    if "ally" in name:
        return EffectKind.ALLY
    return EffectKind.OTHER


def _legacy_card(action: Sequence[Any], kind: ActionKind) -> Any:
    if kind in {
        ActionKind.PLAY,
        ActionKind.ACQUIRE,
        ActionKind.DISCARD,
        ActionKind.SCRAP_CARD,
        ActionKind.FREE_ACQUIRE,
        ActionKind.SCRAP_TRADE_ROW,
    }:
        return action[2] if len(action) > 2 else None
    if kind == ActionKind.SCRAP_FROM_PLAY:
        return action[3] if len(action) > 3 else None
    if kind == ActionKind.DESTROY_BASE:
        return action[3] if len(action) > 3 else None
    if kind == ActionKind.COPY_SHIP:
        return action[1] if len(action) > 1 else None
    return None


def _unwrap_card(card: Any) -> Any:
    nested = _lookup(card, "card", default=None)
    if nested is not None and nested is not card:
        return _unwrap_card(nested)
    # Legacy cards in play are [card_details, ally_used, option, active, copy].
    if (
        isinstance(card, (tuple, list))
        and card
        and isinstance(card[0], (tuple, list))
        and card[0]
    ):
        return card[0]
    return card


@dataclass(frozen=True, slots=True)
class SemanticAction:
    kind: ActionKind
    source_card: int = -1
    target_card: int = -1
    source_zone: int = -1
    target_zone: int = -1
    effect: EffectKind = EffectKind.NONE
    # Fixed order documented by ACTION_NUMERIC_SIZE above.
    numbers: tuple[float, ...] = (0.0,) * ACTION_NUMERIC_SIZE
    source_attributes: tuple[float, ...] = (0.0,) * CARD_ATTRIBUTE_SIZE
    target_attributes: tuple[float, ...] = (0.0,) * CARD_ATTRIBUTE_SIZE


@dataclass(frozen=True, slots=True)
class DecisionEncoding:
    state: np.ndarray
    actions: np.ndarray
    family: DecisionFamily


class Encoder:
    """Encode observations and candidates without positional information."""

    version = 1

    def __init__(
        self,
        card_names: Sequence[str] = BASE_CARD_NAMES,
        *,
        card_catalog: Mapping[Any, Any] | Sequence[Any] | None = None,
        strict: bool = True,
    ):
        if not card_names:
            raise ValueError("card_names must not be empty")
        self.card_names = tuple(card_names)
        self.card_count = len(self.card_names)
        self.strict = strict
        self._name_to_id = {_snake(name): index for index, name in enumerate(self.card_names)}
        self._catalog: dict[int, Any] = {}
        if card_catalog is None and self.card_names == BASE_CARD_NAMES:
            from .cards import ALL_CARDS

            card_catalog = ALL_CARDS
        if card_catalog is not None:
            items = card_catalog.items() if isinstance(card_catalog, Mapping) else enumerate(card_catalog)
            for key, value in items:
                card_id = self.card_id(key, allow_missing=True)
                if card_id is None:
                    card_id = self.card_id(value, allow_missing=True)
                if card_id is not None:
                    self._catalog[card_id] = value

    @classmethod
    def from_engine(cls, engine: Any, *, strict: bool = True) -> Encoder:
        catalog = _lookup(
            engine,
            "card_catalog",
            "card_specs",
            "cards",
            "CARD_CATALOG",
            "CARD_SPECS",
            "ALL_CARDS",
            default=None,
        )
        if catalog is None:
            return cls(strict=strict)
        values = list(catalog.values()) if isinstance(catalog, Mapping) else list(catalog)
        names = []
        for value in values:
            name = _lookup(value, "name", default=None)
            if name is None and isinstance(value, (tuple, list)) and value:
                name = value[0]
            names.append(str(name))
        return cls(names, card_catalog=catalog, strict=strict)

    @property
    def state_size(self) -> int:
        ordered_top = 2 * KNOWN_TOP_SLOTS * self.card_count
        in_play_status = (
            len(_IN_PLAY_STATUS_SIDES) * len(_IN_PLAY_STATUS_FIELDS) * self.card_count
        )
        return len(_STATE_SCALARS) + ZONE_COUNT * self.card_count + ordered_top + in_play_status

    @property
    def action_size(self) -> int:
        return (
            ACTION_KIND_COUNT
            + 2 * (self.card_count + CARD_ATTRIBUTE_SIZE)
            + 2 * ZONE_COUNT
            + EFFECT_COUNT
            + ACTION_NUMERIC_SIZE
        )

    def card_id(self, card: Any, *, allow_missing: bool = False) -> int | None:
        card = _unwrap_card(card)
        if card is None:
            return None
        if isinstance(card, (int, np.integer)):
            value = int(card)
            if 0 <= value < self.card_count:
                return value
        if hasattr(card, "value") and isinstance(card.value, int):
            value = int(card.value)
            if 0 <= value < self.card_count:
                return value

        raw_id = _lookup(card, "card_id", "definition_id", "kind", "id", default=None)
        if raw_id is not None and raw_id is not card:
            resolved = self.card_id(raw_id, allow_missing=True)
            if resolved is not None:
                return resolved

        name = _lookup(card, "name", "card_name", default=None)
        if name is None and isinstance(card, (tuple, list)) and card:
            name = card[0]
        if name is None and isinstance(card, str):
            name = card
        if name is not None and _snake(name) in self._name_to_id:
            return self._name_to_id[_snake(name)]
        if allow_missing or not self.strict:
            return None
        raise ValueError(f"unknown card: {card!r}")

    def _card_attributes(self, card: Any, card_id: int | None) -> tuple[float, ...]:
        card = _unwrap_card(card)
        if card_id is not None and card_id in self._catalog:
            card = self._catalog[card_id]
        if card is None:
            return (0.0,) * CARD_ATTRIBUTE_SIZE

        if isinstance(card, (tuple, list)) and len(card) >= 13:
            cost, combat, authority, trade = card[1], card[2], card[3], card[4]
            faction, card_type, defense = card[5], card[6], card[12]
        else:
            cost = _lookup(card, "cost", default=0)
            combat = _lookup(card, "combat", "attack", default=0)
            authority = _lookup(card, "authority", "health", default=0)
            trade = _lookup(card, "trade", default=0)
            faction = _lookup(card, "faction", "colour", "color", default="none")
            card_type = _lookup(card, "card_type", "type", default="ship")
            defense = _lookup(card, "defense", "shield", default=0)

        result = np.zeros(CARD_ATTRIBUTE_SIZE, dtype=np.float32)
        result[:5] = (float(cost), float(combat), float(authority), float(trade), float(defense))
        factions = {
            "red": 0,
            "machine_cult": 0,
            "green": 1,
            "blob": 1,
            "blue": 2,
            "trade_federation": 2,
            "yellow": 3,
            "star_empire": 3,
            "none": 4,
            "unaligned": 4,
        }
        result[5 + factions.get(_snake(faction), 4)] = 1.0
        types = {"ship": 0, "base": 1, "outp": 2, "outpost": 2}
        result[10 + types.get(_snake(card_type), 0)] = 1.0
        return tuple(float(item) for item in result)

    def _zone_value(self, observation: Any, zone: Zone) -> Any:
        zones = _lookup(observation, "zones", "zone_counts", default=None)
        if isinstance(zones, Mapping):
            normalized = {_snake(key): value for key, value in zones.items()}
            for alias in (zone.name, *_ZONE_ALIASES[zone]):
                if _snake(alias) in normalized:
                    return normalized[_snake(alias)]
        for alias in _ZONE_ALIASES[zone]:
            value = _lookup(observation, alias, default=None)
            if value is not None:
                return value
        return ()

    def _count_cards(self, value: Any, output: np.ndarray) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            # Faction -> cards is common for in-play.  Card -> integer is the
            # preferred pre-counted representation.
            if value and all(isinstance(count, (int, np.integer)) for count in value.values()):
                for card, count in value.items():
                    card_id = self.card_id(card, allow_missing=not self.strict)
                    if card_id is not None:
                        output[card_id] += int(count)
                return
            for nested in value.values():
                self._count_cards(nested, output)
            return
        if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == self.card_count:
            output += value.astype(np.float32, copy=False)
            return
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            # Canonical immutable-engine multisets are ``(card_id, count)`` pairs.
            if isinstance(value, (tuple, list)) and value and all(
                isinstance(pair, (tuple, list))
                and len(pair) == 2
                and isinstance(pair[0], (int, np.integer))
                and isinstance(pair[1], (int, np.integer))
                for pair in value
            ):
                for raw_card_id, count in value:
                    card_id = self.card_id(raw_card_id, allow_missing=not self.strict)
                    if card_id is not None:
                        output[card_id] += int(count)
                return
            # A single legacy card is itself a 14-item tuple.
            if isinstance(value, (tuple, list)) and value and isinstance(value[0], str):
                card_id = self.card_id(value, allow_missing=not self.strict)
                if card_id is not None:
                    output[card_id] += 1.0
                return
            for card in value:
                if card is None:
                    continue
                card_id = self.card_id(card, allow_missing=True)
                if card_id is not None:
                    output[card_id] += 1.0
                elif isinstance(card, (Mapping, Sequence)) and not isinstance(card, (str, bytes)):
                    self._count_cards(card, output)
                elif self.strict:
                    raise ValueError(f"unknown card in zone: {card!r}")
            return
        card_id = self.card_id(value, allow_missing=not self.strict)
        if card_id is not None:
            output[card_id] += 1.0

    def encode_state(self, observation: Any) -> np.ndarray:
        result = np.zeros(self.state_size, dtype=np.float32)
        for index, (_label, aliases, scale) in enumerate(_STATE_SCALARS):
            value = _lookup(observation, *aliases, default=0)
            result[index] = float(value or 0) / scale
        offset = len(_STATE_SCALARS)
        for zone in Zone:
            counts = result[
                offset + int(zone) * self.card_count : offset + (int(zone) + 1) * self.card_count
            ]
            self._count_cards(self._zone_value(observation, zone), counts)
        explorers = int(_lookup(observation, "explorers_remaining", default=0) or 0)
        if explorers:
            explorer_id = self._name_to_id.get("explorer")
            if explorer_id is not None:
                start = len(_STATE_SCALARS) + int(Zone.EXPLORER_SUPPLY) * self.card_count
                result[start + explorer_id] = explorers

        cursor = len(_STATE_SCALARS) + ZONE_COUNT * self.card_count
        for zone in (Zone.OWN_KNOWN_TOP, Zone.OPPONENT_KNOWN_TOP):
            ordered = self._zone_value(observation, zone)
            if isinstance(ordered, Sequence) and not isinstance(ordered, (str, bytes)):
                for slot, card in enumerate(ordered[:KNOWN_TOP_SLOTS]):
                    card_id = self.card_id(card, allow_missing=not self.strict)
                    if card_id is not None:
                        result[cursor + slot * self.card_count + card_id] = 1.0
            cursor += KNOWN_TOP_SLOTS * self.card_count

        for side in _IN_PLAY_STATUS_SIDES:
            cards = _lookup(observation, side, default=())
            for status in _IN_PLAY_STATUS_FIELDS:
                block = result[cursor : cursor + self.card_count]
                for item in cards:
                    card_id = self.card_id(item, allow_missing=not self.strict)
                    if card_id is None:
                        continue
                    if status == "ready":
                        activated = _lookup(item, "activated", default=None)
                        enabled = activated is not None and not bool(activated)
                    else:
                        enabled = bool(_lookup(item, status, default=False))
                    if enabled:
                        block[card_id] += 1.0
                cursor += self.card_count
        return result

    def zone_counts(self, encoded_state: np.ndarray, zone: Zone) -> np.ndarray:
        """Return a view of one exact count block (useful in tests/debug UIs)."""

        state = np.asarray(encoded_state)
        if state.shape != (self.state_size,):
            raise ValueError(f"expected state shape {(self.state_size,)}, got {state.shape}")
        start = len(_STATE_SCALARS) + int(zone) * self.card_count
        return state[start : start + self.card_count]

    def scalar_value(self, encoded_state: np.ndarray, name: str) -> float:
        """Read one normalized scalar by its stable canonical label."""

        normalized = _snake(name)
        labels = [label for label, _aliases, _scale in _STATE_SCALARS]
        if normalized not in labels:
            raise ValueError(f"unknown state scalar: {name!r}")
        return float(np.asarray(encoded_state)[labels.index(normalized)])

    def known_top_slot(
        self, encoded_state: np.ndarray, slot: int, *, opponent: bool = False
    ) -> np.ndarray:
        """Return a one-hot view for a meaningfully ordered known-top slot."""

        if not 0 <= slot < KNOWN_TOP_SLOTS:
            raise ValueError(f"slot must be in [0, {KNOWN_TOP_SLOTS})")
        state = np.asarray(encoded_state)
        cursor = len(_STATE_SCALARS) + ZONE_COUNT * self.card_count
        if opponent:
            cursor += KNOWN_TOP_SLOTS * self.card_count
        start = cursor + slot * self.card_count
        return state[start : start + self.card_count]

    def in_play_status_counts(
        self,
        encoded_state: np.ndarray,
        status: str,
        *,
        opponent: bool = False,
    ) -> np.ndarray:
        """Return exact per-card counts for a visible in-play status."""

        normalized = _snake(status)
        if normalized not in _IN_PLAY_STATUS_FIELDS:
            raise ValueError(f"unknown in-play status: {status!r}")
        state = np.asarray(encoded_state)
        cursor = (
            len(_STATE_SCALARS)
            + ZONE_COUNT * self.card_count
            + 2 * KNOWN_TOP_SLOTS * self.card_count
        )
        if opponent:
            cursor += len(_IN_PLAY_STATUS_FIELDS) * self.card_count
        cursor += _IN_PLAY_STATUS_FIELDS.index(normalized) * self.card_count
        return state[cursor : cursor + self.card_count]

    def semantic_action(self, action: Any) -> SemanticAction:
        kind = action_kind(action)
        legacy = isinstance(action, (tuple, list))
        legacy_card = _legacy_card(action, kind) if legacy else None

        source = (
            _lookup(action, "source_card", "card", "card_id", default=None)
            if not legacy
            else None
        )
        target = (
            _lookup(action, "target_card", "target", "target_card_id", default=None)
            if not legacy
            else None
        )
        if not legacy and kind == ActionKind.ACQUIRE:
            target, source = source, None
        if legacy_card is not None:
            if kind in {
                ActionKind.ATTACK_BASE,
                ActionKind.ACQUIRE,
                ActionKind.DESTROY_BASE,
                ActionKind.SCRAP_TRADE_ROW,
                ActionKind.COPY_SHIP,
                ActionKind.FREE_ACQUIRE,
            }:
                target = legacy_card
            else:
                source = legacy_card

        source_id = self.card_id(source, allow_missing=True)
        target_id = self.card_id(target, allow_missing=True)

        source_zone = _lookup(action, "source_zone", "from_zone", default=None)
        target_zone = _lookup(action, "target_zone", "to_zone", default=None)
        if not legacy and kind in {
            ActionKind.ATTACK_BASE,
            ActionKind.ACQUIRE,
            ActionKind.DESTROY_BASE,
            ActionKind.SCRAP_TRADE_ROW,
            ActionKind.COPY_SHIP,
            ActionKind.FREE_ACQUIRE,
        }:
            target_zone, source_zone = source_zone, None
        source_zone_id = self._zone_id(source_zone)
        target_zone_id = self._zone_id(target_zone)
        if source_zone_id < 0:
            if kind in {ActionKind.PLAY, ActionKind.DISCARD}:
                source_zone_id = int(Zone.OWN_HAND)
            elif kind == ActionKind.SCRAP_CARD:
                raw = _snake(_raw_action_kind(action))
                source_zone_id = int(
                    Zone.OWN_DISCARD if "discard" in raw else Zone.OWN_HAND
                )
            elif kind == ActionKind.SCRAP_FROM_PLAY:
                source_zone_id = int(Zone.OWN_IN_PLAY)
        if target_zone_id < 0:
            if kind in {ActionKind.ACQUIRE, ActionKind.FREE_ACQUIRE, ActionKind.SCRAP_TRADE_ROW}:
                target_zone_id = int(Zone.TRADE_ROW)
            elif kind in {ActionKind.ATTACK_BASE, ActionKind.DESTROY_BASE}:
                target_zone_id = int(Zone.OPPONENT_IN_PLAY)
            elif kind == ActionKind.COPY_SHIP:
                target_zone_id = int(Zone.OWN_IN_PLAY)

        effect = _lookup(action, "effect", "ability", "mode", default=None)
        if legacy:
            if kind == ActionKind.ACTIVATE_ABILITY and len(action) > 3:
                effect = action[3]
            elif kind == ActionKind.SCRAP_FROM_PLAY and len(action) > 4:
                effect = action[4]
            elif kind in {
                ActionKind.GAIN_ATTACK,
                ActionKind.GAIN_TRADE,
                ActionKind.GAIN_AUTHORITY,
                ActionKind.DRAW,
                ActionKind.RECYCLE,
            }:
                effect = action[0]

        amount = _lookup(action, "amount", "value", default=0) if not legacy else 0
        amount2 = _lookup(action, "amount2", default=0) if not legacy else 0
        cost = _lookup(action, "cost", "price", default=0) if not legacy else 0
        attack_available = _lookup(action, "attack_available", "combat_available", default=0)
        target_defense = _lookup(action, "target_defense", "defense", "shield", default=0)
        trade_available = _lookup(action, "trade_available", default=0)
        draw_count = _lookup(action, "draw_count", "cards_drawn", default=0)
        discard_count = _lookup(action, "discard_count", default=0)
        scrap_count = _lookup(action, "scrap_count", default=0)
        optional = _lookup(action, "optional", default=False)
        required = _lookup(action, "required", default=False)
        top_deck = _lookup(action, "top_deck", "to_top", default=False)
        current_authority = _lookup(action, "current_authority", default=0)

        if not legacy:
            if kind in {ActionKind.ACQUIRE, ActionKind.FREE_ACQUIRE}:
                cost = amount
            elif kind == ActionKind.ATTACK_BASE:
                target_defense = amount
                attack_available = amount2
            elif kind == ActionKind.ATTACK_PLAYER:
                attack_available = amount
            optional = kind == ActionKind.DECLINE
            top_deck = kind == ActionKind.FREE_ACQUIRE or bool(top_deck)

        if legacy:
            if kind in {
                ActionKind.GAIN_ATTACK,
                ActionKind.GAIN_TRADE,
                ActionKind.GAIN_AUTHORITY,
                ActionKind.DRAW,
            } and len(action) > 1:
                amount = action[1]
            elif kind == ActionKind.SCRAP_FROM_PLAY and len(action) > 5:
                amount = action[5]
            elif kind == ActionKind.ATTACK_BASE:
                target_defense = action[3] if len(action) > 3 else 0
                attack_available = action[4] if len(action) > 4 else 0
            elif kind == ActionKind.ATTACK_PLAYER and len(action) > 1:
                attack_available = action[1]
            card_details = _unwrap_card(legacy_card)
            if isinstance(card_details, (tuple, list)) and len(card_details) > 1:
                cost = card_details[1]
            optional = kind in {
                ActionKind.DECLINE_DISCARD,
                ActionKind.DECLINE_SCRAP,
                ActionKind.DECLINE_DESTROY_BASE,
                ActionKind.DECLINE_SCRAP_TRADE_ROW,
                ActionKind.DECLINE_COPY,
            }

        numbers = (
            float(amount or 0),
            float(cost or 0),
            float(attack_available or 0),
            float(target_defense or 0),
            float(trade_available or 0),
            float(draw_count or 0),
            float(discard_count or 0),
            float(scrap_count or 0),
            float(bool(optional)),
            float(bool(required)),
            float(bool(top_deck)),
            float(current_authority or 0),
        )
        return SemanticAction(
            kind=kind,
            source_card=-1 if source_id is None else source_id,
            target_card=-1 if target_id is None else target_id,
            source_zone=source_zone_id,
            target_zone=target_zone_id,
            effect=_effect_kind(effect),
            numbers=numbers,
            source_attributes=self._card_attributes(source, source_id),
            target_attributes=self._card_attributes(target, target_id),
        )

    @staticmethod
    def _zone_id(value: Any) -> int:
        if value is None:
            return -1
        if isinstance(value, Zone):
            return int(value)
        if isinstance(value, int) and 0 <= value < ZONE_COUNT:
            return value
        name = _snake(value)
        for zone, aliases in _ZONE_ALIASES.items():
            if name == _snake(zone.name) or any(name == _snake(alias) for alias in aliases):
                return int(zone)
        return -1

    def encode_action(self, action: Any) -> np.ndarray:
        semantic = self.semantic_action(action)
        result = np.zeros(self.action_size, dtype=np.float32)
        cursor = 0
        result[cursor + int(semantic.kind)] = 1.0
        cursor += ACTION_KIND_COUNT

        if semantic.source_card >= 0:
            result[cursor + semantic.source_card] = 1.0
        cursor += self.card_count
        result[cursor : cursor + CARD_ATTRIBUTE_SIZE] = semantic.source_attributes
        cursor += CARD_ATTRIBUTE_SIZE

        if semantic.target_card >= 0:
            result[cursor + semantic.target_card] = 1.0
        cursor += self.card_count
        result[cursor : cursor + CARD_ATTRIBUTE_SIZE] = semantic.target_attributes
        cursor += CARD_ATTRIBUTE_SIZE

        if semantic.source_zone >= 0:
            result[cursor + semantic.source_zone] = 1.0
        cursor += ZONE_COUNT
        if semantic.target_zone >= 0:
            result[cursor + semantic.target_zone] = 1.0
        cursor += ZONE_COUNT

        result[cursor + int(semantic.effect)] = 1.0
        cursor += EFFECT_COUNT
        result[cursor : cursor + ACTION_NUMERIC_SIZE] = semantic.numbers
        return result

    def encode_decision(
        self,
        observation: Any,
        decision_or_actions: Any,
        *,
        family: DecisionFamily | int | str | None = None,
    ) -> DecisionEncoding:
        actions = _lookup(
            decision_or_actions, "actions", "options", "legal_actions", default=None
        )
        if actions is None:
            actions = decision_or_actions
        actions = tuple(actions)
        if not actions:
            raise ValueError("cannot encode a decision with no legal actions")
        resolved_family = (
            decision_family(decision_or_actions)
            if family is None
            else self._coerce_family(family)
        )
        return DecisionEncoding(
            state=self.encode_state(observation),
            actions=np.stack([self.encode_action(action) for action in actions]),
            family=resolved_family,
        )

    @staticmethod
    def _coerce_family(value: DecisionFamily | int | str) -> DecisionFamily:
        if isinstance(value, DecisionFamily):
            return value
        if isinstance(value, int):
            return DecisionFamily(value)
        name = _snake(value)
        if name not in _FAMILY_ALIASES:
            raise ValueError(f"unknown decision family: {value!r}")
        return _FAMILY_ALIASES[name]


# More descriptive alias for callers that prefer it.
ObservationEncoder = Encoder
