"""Deterministic, information-safe Star Realms Core Set engine.

The engine is intentionally callback driven.  A policy sees one immutable
``Decision`` at a time and returns either an ``Action`` or its integer index.
There is no global RNG and no dominance pruning: legal actions are deduplicated
only when their complete semantic encoding is identical.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from typing import Any

from .cards import (
    CARD_BY_ID,
    EXPLORER,
    SCOUT,
    VIPER,
    Card,
    CardType,
    Faction,
    build_trade_deck,
    card_counts,
)


class Seating(StrEnum):
    FIXED = "fixed"
    RANDOM = "random"


class DecisionFamily(StrEnum):
    MAIN = "main"
    DISCARD = "discard"
    SCRAP = "scrap"
    ABILITY_MODE = "ability_mode"
    COPY_SHIP = "copy_ship"
    DESTROY_BASE = "destroy_base"
    SCRAP_TRADE_ROW = "scrap_trade_row"
    FREE_ACQUIRE = "free_acquire"


class ActionKind(StrEnum):
    PLAY_CARD = "play_card"
    ACTIVATE_BASE = "activate_base"
    SCRAP_FOR_ABILITY = "scrap_for_ability"
    ATTACK_BASE = "attack_base"
    ATTACK_PLAYER = "attack_player"
    ACQUIRE = "acquire"
    END_TURN = "end_turn"
    DISCARD_CARD = "discard_card"
    SCRAP_CARD = "scrap_card"
    CHOOSE_MODE = "choose_mode"
    COPY_SHIP = "copy_ship"
    DESTROY_BASE = "destroy_base"
    SCRAP_TRADE_ROW = "scrap_trade_row"
    FREE_ACQUIRE = "free_acquire"
    DECLINE = "decline"


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Card):
        return value.to_dict()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items() if str(key) != "opaque"}
    return value


def _card_multiset(cards: Iterable[Card]) -> tuple[Card, ...]:
    """An immutable, order-invariant multiset that generic encoders can count."""

    return tuple(sorted(cards, key=lambda card: card.card_id))


class _JsonMixin:
    def to_dict(self) -> dict[str, Any]:
        return {key: _json_value(value) for key, value in asdict(self).items() if key != "opaque"}

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class Action(_JsonMixin):
    """A stable, fixed-width semantic action description.

    ``opaque`` is an engine-local locator.  It is excluded from equality,
    hashing, repr, and serialization; model code should use only the seven
    semantic fields preceding it.
    """

    kind: ActionKind
    card_id: int = -1
    target_card_id: int = -1
    ability: str = ""
    source_zone: str = ""
    amount: int = 0
    amount2: int = 0
    opaque: tuple[Any, ...] = field(default=(), compare=False, hash=False, repr=False)

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.kind.value,
            self.card_id,
            self.target_card_id,
            self.ability,
            self.source_zone,
            self.amount,
            self.amount2,
        )

    @property
    def label(self) -> str:
        card = CARD_BY_ID.get(self.card_id)
        target = CARD_BY_ID.get(self.target_card_id)
        bits = [self.kind.value.replace("_", " ")]
        if card is not None:
            bits.append(card.name)
        if self.kind == ActionKind.SCRAP_CARD and self.source_zone:
            zone_label = "discard pile" if self.source_zone == "discard" else self.source_zone
            bits.append("from " + zone_label.replace("_", " "))
        if target is not None:
            bits.append("-> " + target.name)
        if self.ability:
            bits.append("(" + self.ability.replace("_", " ") + ")")
        if self.amount or self.amount2:
            bits.append(f"[{self.amount},{self.amount2}]")
        return " ".join(bits)


@dataclass(frozen=True)
class InPlayObservation(_JsonMixin):
    card: Card
    activated: bool
    ally_triggered: bool
    copied_from_stealth_needle: bool


@dataclass(frozen=True)
class Observation(_JsonMixin):
    """Immutable view containing only information available to ``player_id``."""

    version: int
    player_id: int
    active_player: int
    starting_player: int
    is_starting_player: bool
    turn: int
    action_number: int
    own_authority: int
    opponent_authority: int
    opponent_pending_discard: int
    combat: int
    trade: int
    pending_discard: int
    hand: tuple[Card, ...]
    own_deck_count: int
    own_deck: tuple[Card, ...]
    own_known_top: tuple[Card, ...]
    own_discard: tuple[Card, ...]
    own_in_play: tuple[InPlayObservation, ...]
    opponent_hand_count: int
    opponent_known_hand: tuple[Card, ...]
    opponent_hidden: tuple[Card, ...]
    opponent_deck_count: int
    opponent_known_top: tuple[Card, ...]
    opponent_discard: tuple[Card, ...]
    opponent_in_play: tuple[InPlayObservation, ...]
    trade_row: tuple[Card | None, ...]
    trade_deck_count: int
    trade_deck: tuple[Card, ...]
    explorers_remaining: int
    explorer_supply: tuple[Card, ...]
    scrap_heap: tuple[Card, ...]
    next_ship_to_top: bool
    blob_cards_played: int
    all_allied: bool
    fleet_active: bool


@dataclass(frozen=True)
class Decision(_JsonMixin):
    family: DecisionFamily
    observation: Observation
    actions: tuple[Action, ...]
    prompt: str = ""

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("a Decision must contain at least one legal action")
        if len({action.semantic_key for action in self.actions}) != len(self.actions):
            raise ValueError("Decision contains semantically duplicate actions")


def model_action_indices(decision: Decision) -> tuple[int, ...]:
    """Return the strategically meaningful subset exposed to learned policies.

    The game engine keeps every rules-legal action for human play and auditing.
    Learned actors, however, should not spend capacity rediscovering exact
    dominance invariants.  Ending a turn while a card can still be played, a
    positive base can still be activated, or generated combat can legally be
    spent is strictly dominated in the base set: those actions do not prevent
    a later purchase or end-turn choice and their resources do not carry over.

    Optional purchases and scrap abilities deliberately remain choices.
    """

    indices = tuple(range(len(decision.actions)))
    if decision.family != DecisionFamily.MAIN:
        return indices
    dominated_end = any(
        action.kind
        in {
            ActionKind.PLAY_CARD,
            ActionKind.ACTIVATE_BASE,
            ActionKind.ATTACK_BASE,
            ActionKind.ATTACK_PLAYER,
        }
        for action in decision.actions
    )
    if not dominated_end:
        return indices
    filtered = tuple(
        index for index, action in enumerate(decision.actions) if action.kind != ActionKind.END_TURN
    )
    return filtered or indices


@dataclass(frozen=True)
class GameConfig(_JsonMixin):
    seed: int = 0
    seating: Seating = Seating.FIXED
    starting_player: int | None = None
    max_turns: int = 400
    max_actions_per_turn: int = 200
    explorer_supply: int = 10
    initial_authority: int = 50

    def __post_init__(self) -> None:
        if isinstance(self.seating, str):
            object.__setattr__(self, "seating", Seating(self.seating))
        if self.starting_player not in (None, 0, 1):
            raise ValueError("starting_player must be 0, 1, or None")
        if self.max_turns <= 0 or self.max_actions_per_turn <= 0:
            raise ValueError("turn and action safeguards must be positive")
        if self.explorer_supply < 0 or self.initial_authority <= 0:
            raise ValueError("invalid supply or starting authority")


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


@dataclass(frozen=True)
class RNGStreams:
    """Independent injected streams keep unrelated random events decoupled."""

    seating: random.Random
    market: random.Random
    player_0: random.Random
    player_1: random.Random

    @classmethod
    def from_seed(cls, seed: int) -> RNGStreams:
        values = []
        value = seed & 0xFFFFFFFFFFFFFFFF
        for _ in range(4):
            value = _splitmix64(value)
            values.append(value)
        return cls(*(random.Random(item) for item in values))


@dataclass(frozen=True)
class GameResult(_JsonMixin):
    winner: int | None
    turns: int
    decisions: int
    forced_choices: int
    truncated: bool
    truncation_reason: str | None
    seed: int
    starting_player: int

    @property
    def is_draw(self) -> bool:
        return self.winner is None

    @property
    def winner_player_id(self) -> int | None:
        return self.winner


Chooser = Callable[[int, Decision], int | Action]
CancelHook = Callable[[], bool]
DecisionHook = Callable[[int, Decision, Action], None]


def first_chooser(_player_id: int, _decision: Decision) -> int:
    return 0


def make_random_chooser(seed: int) -> Chooser:
    rng = random.Random(seed)

    def choose(_player_id: int, decision: Decision) -> int:
        return rng.randrange(len(decision.actions))

    return choose


@dataclass
class _InPlay:
    uid: int
    card: Card
    original_card: Card
    activated: bool = False
    ally_triggered: bool = False

    @property
    def is_stealth_copy(self) -> bool:
        return self.original_card.card_id == 23 and self.card.card_id != 23

    def has_faction(self, faction: Faction) -> bool:
        return self.card.faction == faction or (
            self.original_card.card_id == 23 and faction == Faction.MACHINE_CULT
        )


@dataclass
class _Player:
    player_id: int
    name: str
    rng: random.Random
    authority: int
    deck: list[Card] = field(default_factory=list)
    hand: list[Card] = field(default_factory=list)
    discard: list[Card] = field(default_factory=list)
    in_play: list[_InPlay] = field(default_factory=list)
    must_discard: int = 0
    known_top: list[Card] = field(default_factory=list)
    revealed_hand: list[Card] = field(default_factory=list)
    combat: int = 0
    trade: int = 0
    next_ship_top: bool = False
    blob_cards_played: int = 0


class _TruncateGame(RuntimeError):
    pass


class Game:
    """One reusable deterministic two-player Core Set game."""

    def __init__(
        self,
        player_names: Sequence[str] = ("player_0", "player_1"),
        choosers: Sequence[Chooser] | Mapping[int, Chooser] | None = None,
        config: GameConfig | None = None,
        rng_streams: RNGStreams | None = None,
        cancel_hook: CancelHook | None = None,
        decision_hook: DecisionHook | None = None,
    ) -> None:
        if len(player_names) != 2:
            raise ValueError("Star Realms requires exactly two players")
        self.config = config or GameConfig()
        self.rng_streams = rng_streams or RNGStreams.from_seed(self.config.seed)
        self.cancel_hook = cancel_hook
        self.decision_hook = decision_hook
        self.choosers = self._normalize_choosers(choosers)
        self.players = [
            _Player(
                0, str(player_names[0]), self.rng_streams.player_0, self.config.initial_authority
            ),
            _Player(
                1, str(player_names[1]), self.rng_streams.player_1, self.config.initial_authority
            ),
        ]
        self.scrap_heap: list[Card] = []
        self.trade_deck = build_trade_deck()
        self.rng_streams.market.shuffle(self.trade_deck)
        self.trade_row: list[Card | None] = [self.trade_deck.pop() for _ in range(5)]
        self.explorers_remaining = self.config.explorer_supply
        self.turns = 0
        self.decisions = 0
        self.forced_choices = 0
        self._turn_actions = 0
        self._uid = 0
        self._winner: int | None = None
        self._truncation_reason: str | None = None
        self.result: GameResult | None = None

        for player in self.players:
            player.deck = [SCOUT] * 8 + [VIPER] * 2
            player.rng.shuffle(player.deck)

        if self.config.starting_player is not None:
            self.starting_player = self.config.starting_player
        elif self.config.seating == Seating.RANDOM:
            order = [0, 1]
            self.rng_streams.seating.shuffle(order)
            self.starting_player = order[0]
        else:
            self.starting_player = 0
        self.active_player = self.starting_player
        self._draw(self.players[self.starting_player], 3)
        self._draw(self.players[1 - self.starting_player], 5)

    @staticmethod
    def _normalize_choosers(
        choosers: Sequence[Chooser] | Mapping[int, Chooser] | None,
    ) -> dict[int, Chooser]:
        if choosers is None:
            return {0: first_chooser, 1: first_chooser}
        if isinstance(choosers, Mapping):
            return {0: choosers.get(0, first_chooser), 1: choosers.get(1, first_chooser)}
        if len(choosers) != 2:
            raise ValueError("provide one chooser per player")
        return {0: choosers[0], 1: choosers[1]}

    def fork(self) -> Game:
        """Clone a live game for counterfactual search without generic deepcopy.

        Card definitions and configuration are immutable and are deliberately
        shared. Only mutable zones, in-play flags, counters, and the four RNG
        streams are copied. Search creates several forks from the same decision;
        teaching ``deepcopy`` the whole object graph about this distinction was
        a substantial and avoidable allocator/GC cost.
        """

        def clone_random(source: random.Random) -> random.Random:
            cloned = random.Random()
            cloned.setstate(source.getstate())
            return cloned

        forked = object.__new__(Game)
        forked.config = self.config
        forked.rng_streams = RNGStreams(
            seating=clone_random(self.rng_streams.seating),
            market=clone_random(self.rng_streams.market),
            player_0=clone_random(self.rng_streams.player_0),
            player_1=clone_random(self.rng_streams.player_1),
        )
        forked.cancel_hook = self.cancel_hook
        forked.decision_hook = self.decision_hook
        forked.choosers = dict(self.choosers)
        player_rngs = (forked.rng_streams.player_0, forked.rng_streams.player_1)
        forked.players = []
        for player, player_rng in zip(self.players, player_rngs, strict=True):
            forked.players.append(
                _Player(
                    player_id=player.player_id,
                    name=player.name,
                    rng=player_rng,
                    authority=player.authority,
                    deck=list(player.deck),
                    hand=list(player.hand),
                    discard=list(player.discard),
                    in_play=[
                        _InPlay(
                            uid=item.uid,
                            card=item.card,
                            original_card=item.original_card,
                            activated=item.activated,
                            ally_triggered=item.ally_triggered,
                        )
                        for item in player.in_play
                    ],
                    must_discard=player.must_discard,
                    known_top=list(player.known_top),
                    revealed_hand=list(player.revealed_hand),
                    combat=player.combat,
                    trade=player.trade,
                    next_ship_top=player.next_ship_top,
                    blob_cards_played=player.blob_cards_played,
                )
            )
        forked.scrap_heap = list(self.scrap_heap)
        forked.trade_deck = list(self.trade_deck)
        forked.trade_row = list(self.trade_row)
        forked.explorers_remaining = self.explorers_remaining
        forked.turns = self.turns
        forked.decisions = self.decisions
        forked.forced_choices = self.forced_choices
        forked._turn_actions = self._turn_actions
        forked._uid = self._uid
        forked._winner = self._winner
        forked._truncation_reason = self._truncation_reason
        forked.result = self.result
        forked.starting_player = self.starting_player
        forked.active_player = self.active_player
        return forked

    def run(self) -> GameResult:
        if self.result is not None:
            return self.result
        try:
            while self._winner is None and self.turns < self.config.max_turns:
                self._take_turn(self.players[self.active_player])
                if self._winner is None:
                    self.active_player = 1 - self.active_player
            if self._winner is None and self._truncation_reason is None:
                self._truncation_reason = "max_turns"
        except _TruncateGame:
            pass

        self.result = GameResult(
            winner=self._winner if self._truncation_reason is None else None,
            turns=self.turns,
            decisions=self.decisions,
            forced_choices=self.forced_choices,
            truncated=self._truncation_reason is not None,
            truncation_reason=self._truncation_reason,
            seed=self.config.seed,
            starting_player=self.starting_player,
        )
        return self.result

    def continue_from_main_action(self, action: Action) -> GameResult:
        """Continue a cloned game from a main-phase decision already emitted.

        The live engine calls decision hooks before applying the selected
        action. Counterfactual training can therefore clone that exact state,
        force a different legal action here, and roll both branches forward
        with identical hidden state and RNG streams.
        """

        if self.result is not None:
            raise RuntimeError("cannot continue a completed game")
        player = self.players[self.active_player]
        ended = self._apply_main_action(player, action)
        try:
            while not ended and self._winner is None:
                selected = self._choose(
                    player, DecisionFamily.MAIN, self._main_actions(player), "Main phase"
                )
                ended = self._apply_main_action(player, selected)
            if self._winner is None:
                self._cleanup_and_draw(player)
                self.active_player = 1 - self.active_player
            while self._winner is None and self.turns < self.config.max_turns:
                self._take_turn(self.players[self.active_player])
                if self._winner is None:
                    self.active_player = 1 - self.active_player
            if self._winner is None and self._truncation_reason is None:
                self._truncation_reason = "max_turns"
        except _TruncateGame:
            pass
        self.result = GameResult(
            winner=self._winner if self._truncation_reason is None else None,
            turns=self.turns,
            decisions=self.decisions,
            forced_choices=self.forced_choices,
            truncated=self._truncation_reason is not None,
            truncation_reason=self._truncation_reason,
            seed=self.config.seed,
            starting_player=self.starting_player,
        )
        return self.result

    def observation(self, player_id: int) -> Observation:
        if player_id not in (0, 1):
            raise ValueError("unknown player")
        own = self.players[player_id]
        opponent = self.players[1 - player_id]
        own_unknown_deck = own.deck[: -len(own.known_top)] if own.known_top else list(own.deck)
        opponent_unknown_deck = (
            opponent.deck[: -len(opponent.known_top)] if opponent.known_top else list(opponent.deck)
        )
        unrevealed_hand = list(opponent.hand)
        for known in opponent.revealed_hand:
            for index, card in enumerate(unrevealed_hand):
                if card.card_id == known.card_id:
                    unrevealed_hand.pop(index)
                    break
        return Observation(
            version=2,
            player_id=player_id,
            active_player=self.active_player,
            starting_player=self.starting_player,
            is_starting_player=player_id == self.starting_player,
            turn=self.turns,
            action_number=self._turn_actions,
            own_authority=own.authority,
            opponent_authority=opponent.authority,
            opponent_pending_discard=opponent.must_discard,
            combat=own.combat if self.active_player == player_id else 0,
            trade=own.trade if self.active_player == player_id else 0,
            pending_discard=own.must_discard,
            hand=tuple(own.hand),
            own_deck_count=len(own.deck),
            own_deck=_card_multiset(own_unknown_deck),
            own_known_top=tuple(reversed(own.known_top)),
            own_discard=_card_multiset(own.discard),
            own_in_play=self._in_play_observation(own.in_play),
            opponent_hand_count=len(opponent.hand),
            opponent_known_hand=tuple(opponent.revealed_hand),
            opponent_hidden=_card_multiset(opponent_unknown_deck + unrevealed_hand),
            opponent_deck_count=len(opponent.deck),
            opponent_known_top=tuple(reversed(opponent.known_top)),
            opponent_discard=_card_multiset(opponent.discard),
            opponent_in_play=self._in_play_observation(opponent.in_play),
            trade_row=tuple(self.trade_row),
            trade_deck_count=len(self.trade_deck),
            trade_deck=_card_multiset(self.trade_deck),
            explorers_remaining=self.explorers_remaining,
            explorer_supply=(EXPLORER,) * self.explorers_remaining,
            scrap_heap=_card_multiset(self.scrap_heap),
            next_ship_to_top=own.next_ship_top,
            blob_cards_played=own.blob_cards_played,
            all_allied=any(entry.card.card_id == 19 for entry in own.in_play),
            fleet_active=self._fleet_hq_active(own),
        )

    def resample_public_belief(self, observer: int, seed: int) -> None:
        """Redeterminize hidden zones without changing any public quantity.

        Search training must not turn the learner into an oracle for the exact
        opponent hand or future market order present in a worker process. Each
        rollout therefore samples a state from the current public information
        set, while paired actions use the same seed (common random numbers).
        """

        if observer not in (0, 1):
            raise ValueError("unknown observer")
        rng = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
        own = self.players[observer]
        opponent = self.players[1 - observer]

        known_count = min(len(own.known_top), len(own.deck))
        if known_count:
            unknown = own.deck[:-known_count]
            rng.shuffle(unknown)
            own.deck = unknown + own.deck[-known_count:]
        else:
            rng.shuffle(own.deck)

        # Revealed cards must remain in the opponent's hand, and a publicly
        # known top-deck suffix must remain on top. Only the complementary
        # hidden cards may move between hand and deck.
        hand_slots: list[Card | None] = list(opponent.hand)
        unmatched_indices = list(range(len(hand_slots)))
        for known in opponent.revealed_hand:
            matched = next(
                (
                    index
                    for index in unmatched_indices
                    if hand_slots[index] is not None
                    and hand_slots[index].card_id == known.card_id
                ),
                None,
            )
            if matched is not None:
                unmatched_indices.remove(matched)
        hidden_hand = [hand_slots[index] for index in unmatched_indices]
        for index in unmatched_indices:
            hand_slots[index] = None

        opponent_known_count = min(len(opponent.known_top), len(opponent.deck))
        if opponent_known_count:
            unknown_deck = list(opponent.deck[:-opponent_known_count])
            known_deck = list(opponent.deck[-opponent_known_count:])
        else:
            unknown_deck = list(opponent.deck)
            known_deck = []
        hidden_opponent = [*hidden_hand, *unknown_deck]
        rng.shuffle(hidden_opponent)
        hidden_iter = iter(hidden_opponent[: len(unmatched_indices)])
        opponent.hand = [
            card if card is not None else next(hidden_iter) for card in hand_slots
        ]
        opponent.deck = hidden_opponent[len(unmatched_indices) :] + known_deck

        # The remaining trade deck order is hidden. The visible row and every
        # zone/card count stay untouched.
        rng.shuffle(self.trade_deck)

    def unordered_card_zones(self, player_id: int) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Return inspectable hidden cards without revealing any card order or opponent split.

        This is intended for the local human-play inspector. Sorting each zone
        by card id makes the payload a multiset rather than a draw-order leak;
        the opponent's hand and deck are also merged into one hidden pool.
        """

        if player_id not in (0, 1):
            raise ValueError("unknown player")
        own = self.players[player_id]
        opponent = self.players[1 - player_id]

        def zones(player: _Player) -> dict[str, list[dict[str, Any]]]:
            return {
                "hand": [card.to_dict() for card in _card_multiset(player.hand)],
                "deck": [card.to_dict() for card in _card_multiset(player.deck)],
            }

        # The opponent's hand size and deck size are public, but the assignment
        # of cards between those zones is not.  Expose one shuffled information-
        # set pool so inspector clients cannot recover that hidden assignment.
        opponent_hidden = _card_multiset(opponent.hand + opponent.deck)
        return {
            "own": zones(own),
            "opponent": {
                "hidden": [card.to_dict() for card in opponent_hidden],
            },
        }

    @staticmethod
    def _in_play_observation(cards: Iterable[_InPlay]) -> tuple[InPlayObservation, ...]:
        return tuple(
            InPlayObservation(item.card, item.activated, item.ally_triggered, item.is_stealth_copy)
            for item in cards
        )

    def state_dict(self, *, include_hidden: bool = False) -> dict[str, Any]:
        """Human-readable state; hidden orders require explicit opt-in."""

        data: dict[str, Any] = {
            "active_player": self.active_player,
            "turns": self.turns,
            "trade_row": [card.name if card else None for card in self.trade_row],
            "trade_deck_count": len(self.trade_deck),
            "explorers_remaining": self.explorers_remaining,
            "scrap_heap": [
                (CARD_BY_ID[card_id].name, count) for card_id, count in card_counts(self.scrap_heap)
            ],
            "players": [],
            "result": self.result.to_dict() if self.result else None,
        }
        for player in self.players:
            item: dict[str, Any] = {
                "player_id": player.player_id,
                "name": player.name,
                "authority": player.authority,
                "hand_count": len(player.hand),
                "deck_count": len(player.deck),
                "discard": [card.name for card in player.discard],
                "in_play": [entry.card.name for entry in player.in_play],
                "combat": player.combat,
                "trade": player.trade,
                "must_discard": player.must_discard,
            }
            if include_hidden:
                item["hand"] = [card.name for card in player.hand]
                item["deck_top_to_bottom"] = [card.name for card in reversed(player.deck)]
            data["players"].append(item)
        if include_hidden:
            data["trade_deck_top_to_bottom"] = [card.name for card in reversed(self.trade_deck)]
        return data

    def to_json(self, *, include_hidden: bool = False, indent: int | None = 2) -> str:
        return json.dumps(
            self.state_dict(include_hidden=include_hidden), indent=indent, sort_keys=True
        )

    def card_conservation(self) -> dict[int, int]:
        """Count every physical card in every zone (useful for engine audits)."""

        cards: list[Card] = list(self.trade_deck) + list(self.scrap_heap)
        cards.extend(card for card in self.trade_row if card is not None)
        cards.extend([EXPLORER] * self.explorers_remaining)
        for player in self.players:
            cards.extend(player.deck)
            cards.extend(player.hand)
            cards.extend(player.discard)
            cards.extend(entry.original_card for entry in player.in_play)
        return dict(card_counts(cards))

    def _take_turn(self, player: _Player) -> None:
        self.turns += 1
        self._turn_actions = 0
        player.combat = 0
        player.trade = 0
        player.next_ship_top = False
        player.blob_cards_played = 0
        for item in player.in_play:
            item.ally_triggered = False
            item.activated = not self._base_requires_activation(item.card)

        # Discard pressure resolves before the player can draw/cycle via a base.
        while player.must_discard > 0 and player.hand:
            actions = [
                Action(
                    ActionKind.DISCARD_CARD,
                    card_id=card.card_id,
                    source_zone="hand",
                    opaque=(index,),
                )
                for index, card in enumerate(player.hand)
            ]
            action = self._choose(
                player, DecisionFamily.DISCARD, actions, "Choose a card to discard"
            )
            self._discard_from_hand(player, int(action.opaque[0]))
            player.must_discard -= 1
        if not player.hand:
            player.must_discard = 0

        ended = False
        while not ended and self._winner is None:
            action = self._choose(
                player, DecisionFamily.MAIN, self._main_actions(player), "Main phase"
            )
            ended = self._apply_main_action(player, action)

        if self._winner is None:
            self._cleanup_and_draw(player)

    def _check_safeguards(self) -> None:
        if self.cancel_hook is not None and self.cancel_hook():
            self._truncation_reason = "cancelled"
            raise _TruncateGame()
        if self._turn_actions >= self.config.max_actions_per_turn:
            self._truncation_reason = "max_actions_per_turn"
            raise _TruncateGame()

    @staticmethod
    def _deduplicate(actions: Iterable[Action]) -> tuple[Action, ...]:
        result: list[Action] = []
        seen = set()
        for action in actions:
            if action.semantic_key not in seen:
                seen.add(action.semantic_key)
                result.append(action)
        return tuple(result)

    def _choose(
        self,
        player: _Player,
        family: DecisionFamily,
        actions: Iterable[Action],
        prompt: str,
    ) -> Action:
        self._check_safeguards()
        options = self._deduplicate(actions)
        if not options:
            raise RuntimeError("engine generated an empty decision: " + prompt)
        decision = Decision(family, self.observation(player.player_id), options, prompt)
        self.decisions += 1
        self._turn_actions += 1
        if len(options) == 1:
            selected = options[0]
            self.forced_choices += 1
        else:
            raw = self.choosers[player.player_id](player.player_id, decision)
            if isinstance(raw, bool):
                raise TypeError("chooser returned bool; return an action or integer index")
            if isinstance(raw, int):
                if raw < 0 or raw >= len(options):
                    raise IndexError(f"chooser selected illegal action index {raw}")
                selected = options[raw]
            elif isinstance(raw, Action):
                try:
                    selected = options[options.index(raw)]
                except ValueError as exc:
                    raise ValueError("chooser returned an action not in Decision.actions") from exc
            else:
                raise TypeError("chooser must return Action or integer index")
        if self.decision_hook is not None:
            self.decision_hook(player.player_id, decision, selected)
        return selected

    def _main_actions(self, player: _Player) -> tuple[Action, ...]:
        opponent = self.players[1 - player.player_id]
        actions: list[Action] = []
        for index, card in enumerate(player.hand):
            actions.append(
                Action(
                    ActionKind.PLAY_CARD, card_id=card.card_id, source_zone="hand", opaque=(index,)
                )
            )
        for item in player.in_play:
            if item.card.is_base and not item.activated:
                actions.append(
                    Action(
                        ActionKind.ACTIVATE_BASE,
                        card_id=item.card.card_id,
                        source_zone="in_play",
                        opaque=(item.uid,),
                    )
                )
            if item.card.scrap:
                actions.append(
                    Action(
                        ActionKind.SCRAP_FOR_ABILITY,
                        card_id=item.card.card_id,
                        ability=item.card.scrap,
                        source_zone="in_play",
                        amount=item.card.scrap_amount,
                        opaque=(item.uid,),
                    )
                )
        if player.combat > 0:
            outposts = [
                item for item in opponent.in_play if item.card.card_type == CardType.OUTPOST
            ]
            targets = outposts or [
                item for item in opponent.in_play if item.card.card_type == CardType.BASE
            ]
            for item in targets:
                if player.combat >= item.card.defense:
                    actions.append(
                        Action(
                            ActionKind.ATTACK_BASE,
                            target_card_id=item.card.card_id,
                            source_zone="opponent_in_play",
                            amount=item.card.defense,
                            amount2=player.combat,
                            opaque=(item.uid,),
                        )
                    )
            if not outposts:
                actions.append(Action(ActionKind.ATTACK_PLAYER, amount=player.combat))
        for slot, card in enumerate(self.trade_row):
            if card is not None and card.cost <= player.trade:
                actions.append(
                    Action(
                        ActionKind.ACQUIRE,
                        card_id=card.card_id,
                        source_zone="trade_row",
                        amount=card.cost,
                        opaque=(slot,),
                    )
                )
        if self.explorers_remaining and player.trade >= EXPLORER.cost:
            actions.append(
                Action(
                    ActionKind.ACQUIRE,
                    card_id=EXPLORER.card_id,
                    source_zone="explorer_supply",
                    amount=EXPLORER.cost,
                )
            )
        # Ending is always legal; unplayed cards go to the discard pile.
        actions.append(Action(ActionKind.END_TURN))
        return self._deduplicate(actions)

    def _apply_main_action(self, player: _Player, action: Action) -> bool:
        kind = action.kind
        if kind == ActionKind.PLAY_CARD:
            self._play_card(player, int(action.opaque[0]))
        elif kind == ActionKind.ACTIVATE_BASE:
            item = self._find_in_play(player, int(action.opaque[0]))
            self._activate_card(player, item)
        elif kind == ActionKind.SCRAP_FOR_ABILITY:
            item = self._find_in_play(player, int(action.opaque[0]))
            self._scrap_in_play(player, item)
        elif kind == ActionKind.ATTACK_BASE:
            opponent = self.players[1 - player.player_id]
            item = self._find_in_play(opponent, int(action.opaque[0]))
            player.combat -= item.card.defense
            opponent.in_play.remove(item)
            opponent.discard.append(item.original_card)
        elif kind == ActionKind.ATTACK_PLAYER:
            opponent = self.players[1 - player.player_id]
            opponent.authority -= player.combat
            player.combat = 0
            if opponent.authority <= 0:
                self._winner = player.player_id
        elif kind == ActionKind.ACQUIRE:
            if action.source_zone == "explorer_supply":
                self._acquire_explorer(player, action.amount)
            else:
                self._acquire_market(player, int(action.opaque[0]), action.amount)
        elif kind == ActionKind.END_TURN:
            return True
        else:
            raise RuntimeError(f"unexpected main action {kind}")
        return False

    def _play_card(self, player: _Player, hand_index: int) -> None:
        card = player.hand.pop(hand_index)
        self._forget_revealed(player, card)
        self._uid += 1
        item = _InPlay(self._uid, card, card, activated=False)
        player.in_play.append(item)
        if card.card_id == 23:
            self._copy_stealth_needle(player, item)
        if item.card.faction == Faction.BLOB and item.original_card.card_id != 23:
            player.blob_cards_played += 1
        if item.card.is_ship:
            self._activate_card(player, item)
        elif not self._base_requires_activation(item.card):
            # Continuous/no-resource bases have no once-per-turn activation to
            # schedule (Mech World, Fleet HQ, and Battle Station).
            item.activated = True
            self._trigger_available_allies(player)
        else:
            # A base enters play immediately, but its once-per-turn ability is
            # deliberately scheduled as a main-phase action.  This is crucial
            # for cards such as Blob World, Recycling Station, and Central
            # Office whose correct timing can change later decisions.
            self._trigger_available_allies(player)

    def _activate_card(self, player: _Player, item: _InPlay) -> None:
        if item.activated:
            raise RuntimeError("card activated twice")
        item.activated = True
        card = item.card
        player.combat += card.combat
        player.authority += card.authority
        player.trade += card.trade
        if card.is_ship and self._fleet_hq_active(player):
            player.combat += 1
        if card.primary:
            self._execute_effect(player, card.primary, 0, item)
        self._trigger_available_allies(player)

    def _trigger_available_allies(self, player: _Player) -> None:
        # Effects can draw or alter state, but cannot add in-play cards without
        # returning to the main loop.  Iterate to cover all newly enabled allies.
        progress = True
        while progress:
            progress = False
            all_ally = any(entry.card.card_id == 19 for entry in player.in_play)
            for item in list(player.in_play):
                if item.ally_triggered or not item.card.ally:
                    continue
                faction = item.card.faction
                allied = all_ally or any(
                    other.uid != item.uid and other.has_faction(faction) for other in player.in_play
                )
                if allied:
                    item.ally_triggered = True
                    self._execute_effect(player, item.card.ally, item.card.ally_amount, item)
                    progress = True

    def _execute_effect(self, player: _Player, effect: str, amount: int, source: _InPlay) -> None:
        opponent = self.players[1 - player.player_id]
        if effect == "gain_combat":
            player.combat += amount
        elif effect == "gain_trade":
            player.trade += amount
        elif effect == "gain_authority":
            player.authority += amount
        elif effect == "draw":
            self._draw(player, 1)
        elif effect == "draw_two":
            self._draw(player, 2)
        elif effect in ("all_ally", "fleet_hq"):
            pass  # continuous while the base remains in play
        elif effect == "opponent_discard":
            opponent.must_discard += 1
        elif effect == "ship_top":
            player.next_ship_top = True
        elif effect == "embassy_yacht":
            if sum(1 for item in player.in_play if item.card.is_base) >= 2:
                self._draw(player, 2)
        elif effect == "scrap_trade_row":
            self._choose_trade_row_scrap(player, source)
        elif effect == "blob_world":
            action = self._choose(
                player,
                DecisionFamily.ABILITY_MODE,
                (
                    Action(
                        ActionKind.CHOOSE_MODE,
                        card_id=source.card.card_id,
                        ability="gain_combat",
                        amount=5,
                    ),
                    Action(
                        ActionKind.CHOOSE_MODE,
                        card_id=source.card.card_id,
                        ability="draw",
                        amount=player.blob_cards_played,
                    ),
                ),
                "Blob World: gain 5 combat or draw per Blob played",
            )
            if action.ability == "gain_combat":
                player.combat += 5
            else:
                self._draw(player, action.amount)
        elif effect == "scrap_any":
            self._scrap_any(player, source, required=False)
        elif effect == "scrap_two_draw":
            count = 0
            for _ in range(2):
                count += self._scrap_any(player, source, required=False)
            self._draw(player, count)
        elif effect == "draw_then_scrap":
            self._draw(player, 1)
            self._scrap_from_hand(player, source)
        elif effect == "destroy_base":
            self._choose_destroy_base(player, source)
        elif effect == "patrol_mech":
            action = self._choose(
                player,
                DecisionFamily.ABILITY_MODE,
                (
                    Action(
                        ActionKind.CHOOSE_MODE,
                        card_id=source.card.card_id,
                        ability="gain_combat",
                        amount=5,
                    ),
                    Action(
                        ActionKind.CHOOSE_MODE,
                        card_id=source.card.card_id,
                        ability="gain_trade",
                        amount=3,
                    ),
                ),
                "Patrol Mech: gain 5 combat or 3 trade",
            )
            if action.ability == "gain_combat":
                player.combat += 5
            else:
                player.trade += 3
        elif effect == "copy_ship":
            # Copy selection happens before activation, in _play_card.
            pass
        elif effect == "recycle":
            self._recycle(player, source)
        elif effect == "barter_world":
            self._choose_resource(player, source, "gain_authority", 2, "gain_trade", 2)
        elif effect == "defense_center":
            self._choose_resource(player, source, "gain_combat", 2, "gain_authority", 3)
        elif effect == "trading_post":
            self._choose_resource(player, source, "gain_authority", 1, "gain_trade", 1)
        elif effect == "free_ship":
            self._free_ship(player, source)
        elif effect == "destroy_and_scrap":
            self._choose_destroy_base(player, source)
            self._choose_trade_row_scrap(player, source)
        elif effect == "draw_destroy":
            # Official order matters: the card draw can change later choices.
            self._draw(player, 1)
            self._choose_destroy_base(player, source)
        else:
            raise ValueError(f"unknown card effect {effect!r}")

    def _choose_resource(
        self,
        player: _Player,
        source: _InPlay,
        first: str,
        first_amount: int,
        second: str,
        second_amount: int,
    ) -> None:
        action = self._choose(
            player,
            DecisionFamily.ABILITY_MODE,
            (
                Action(
                    ActionKind.CHOOSE_MODE,
                    card_id=source.card.card_id,
                    ability=first,
                    amount=first_amount,
                ),
                Action(
                    ActionKind.CHOOSE_MODE,
                    card_id=source.card.card_id,
                    ability=second,
                    amount=second_amount,
                ),
            ),
            source.card.name + ": choose mode",
        )
        if action.ability == "gain_combat":
            player.combat += action.amount
        elif action.ability == "gain_authority":
            player.authority += action.amount
        else:
            player.trade += action.amount

    def _scrap_any(self, player: _Player, source: _InPlay, required: bool) -> int:
        actions: list[Action] = []
        for index, card in enumerate(player.discard):
            actions.append(
                Action(
                    ActionKind.SCRAP_CARD,
                    card_id=card.card_id,
                    source_zone="discard",
                    opaque=(index,),
                )
            )
        for index, card in enumerate(player.hand):
            actions.append(
                Action(
                    ActionKind.SCRAP_CARD, card_id=card.card_id, source_zone="hand", opaque=(index,)
                )
            )
        if not required:
            actions.append(
                Action(ActionKind.DECLINE, card_id=source.card.card_id, ability="scrap_any")
            )
        if not actions:
            return 0
        action = self._choose(
            player, DecisionFamily.SCRAP, actions, "Scrap a card from hand or discard"
        )
        if action.kind == ActionKind.DECLINE:
            return 0
        if action.source_zone == "hand":
            card = player.hand.pop(int(action.opaque[0]))
            self._forget_revealed(player, card)
        else:
            card = player.discard.pop(int(action.opaque[0]))
        self.scrap_heap.append(card)
        return 1

    def _scrap_from_hand(self, player: _Player, source: _InPlay) -> None:
        if not player.hand:
            return
        actions = [
            Action(ActionKind.SCRAP_CARD, card_id=card.card_id, source_zone="hand", opaque=(index,))
            for index, card in enumerate(player.hand)
        ]
        action = self._choose(
            player, DecisionFamily.SCRAP, actions, source.card.name + ": scrap from hand"
        )
        card = player.hand.pop(int(action.opaque[0]))
        self._forget_revealed(player, card)
        self.scrap_heap.append(card)

    def _recycle(self, player: _Player, source: _InPlay) -> None:
        action = self._choose(
            player,
            DecisionFamily.ABILITY_MODE,
            (
                Action(
                    ActionKind.CHOOSE_MODE,
                    card_id=source.card.card_id,
                    ability="gain_trade",
                    amount=1,
                ),
                Action(
                    ActionKind.CHOOSE_MODE, card_id=source.card.card_id, ability="cycle", amount=2
                ),
            ),
            "Recycling Station: gain trade or cycle up to two cards",
        )
        if action.ability == "gain_trade":
            player.trade += 1
            return
        discarded = 0
        for _ in range(2):
            if not player.hand:
                break
            actions = [
                Action(
                    ActionKind.DISCARD_CARD,
                    card_id=card.card_id,
                    source_zone="hand",
                    opaque=(index,),
                )
                for index, card in enumerate(player.hand)
            ]
            actions.append(Action(ActionKind.DECLINE, card_id=source.card.card_id, ability="cycle"))
            selected = self._choose(
                player, DecisionFamily.DISCARD, actions, "Discard a card to replace"
            )
            if selected.kind == ActionKind.DECLINE:
                break
            self._discard_from_hand(player, int(selected.opaque[0]))
            discarded += 1
        self._draw(player, discarded)

    def _copy_stealth_needle(self, player: _Player, needle: _InPlay) -> None:
        candidates = [
            item for item in player.in_play if item.uid != needle.uid and item.card.is_ship
        ]
        if not candidates:
            return
        actions = [
            Action(
                ActionKind.COPY_SHIP,
                card_id=needle.original_card.card_id,
                target_card_id=item.card.card_id,
                ability="copy_ship",
                source_zone="in_play",
                opaque=(item.uid,),
            )
            for item in candidates
        ]
        action = self._choose(
            player, DecisionFamily.COPY_SHIP, actions, "Stealth Needle: copy a ship"
        )
        target = self._find_in_play(player, int(action.opaque[0]))
        needle.card = target.card

    def _choose_destroy_base(self, player: _Player, source: _InPlay) -> None:
        opponent = self.players[1 - player.player_id]
        outposts = [item for item in opponent.in_play if item.card.card_type == CardType.OUTPOST]
        targets = outposts or [
            item for item in opponent.in_play if item.card.card_type == CardType.BASE
        ]
        if not targets:
            return
        actions = [
            Action(
                ActionKind.DESTROY_BASE,
                card_id=source.card.card_id,
                target_card_id=item.card.card_id,
                ability="destroy_base",
                source_zone="opponent_in_play",
                opaque=(item.uid,),
            )
            for item in targets
        ]
        actions.append(
            Action(ActionKind.DECLINE, card_id=source.card.card_id, ability="destroy_base")
        )
        action = self._choose(
            player, DecisionFamily.DESTROY_BASE, actions, "Optionally destroy a base"
        )
        if action.kind != ActionKind.DECLINE:
            item = self._find_in_play(opponent, int(action.opaque[0]))
            opponent.in_play.remove(item)
            opponent.discard.append(item.original_card)

    def _choose_trade_row_scrap(self, player: _Player, source: _InPlay) -> None:
        # Explorer is a separate supply, never a trade-row scrap target.
        actions = [
            Action(
                ActionKind.SCRAP_TRADE_ROW,
                card_id=source.card.card_id,
                target_card_id=card.card_id,
                ability="scrap_trade_row",
                source_zone="trade_row",
                opaque=(slot,),
            )
            for slot, card in enumerate(self.trade_row)
            if card is not None
        ]
        if not actions:
            return
        actions.append(
            Action(ActionKind.DECLINE, card_id=source.card.card_id, ability="scrap_trade_row")
        )
        action = self._choose(
            player, DecisionFamily.SCRAP_TRADE_ROW, actions, "Optionally scrap a trade-row card"
        )
        if action.kind != ActionKind.DECLINE:
            slot = int(action.opaque[0])
            card = self.trade_row[slot]
            if card is None:
                raise RuntimeError("trade-row target disappeared")
            self.scrap_heap.append(card)
            self._refill_market_slot(slot)

    def _free_ship(self, player: _Player, source: _InPlay) -> None:
        # All five market slots are checked; Explorer is not part of the row.
        actions = [
            Action(
                ActionKind.FREE_ACQUIRE,
                card_id=source.card.card_id,
                target_card_id=card.card_id,
                ability="free_ship_to_top",
                source_zone="trade_row",
                opaque=(slot,),
            )
            for slot, card in enumerate(self.trade_row)
            if card is not None and card.is_ship
        ]
        if not actions:
            return
        actions.append(
            Action(
                ActionKind.DECLINE,
                card_id=source.card.card_id,
                ability="free_ship_to_top",
            )
        )
        action = self._choose(
            player,
            DecisionFamily.FREE_ACQUIRE,
            actions,
            "Optionally acquire a ship free onto your deck",
        )
        if action.kind != ActionKind.DECLINE:
            self._acquire_market(player, int(action.opaque[0]), 0, force_top=True)

    def _scrap_in_play(self, player: _Player, item: _InPlay) -> None:
        player.in_play.remove(item)
        self.scrap_heap.append(item.original_card)
        self._execute_effect(player, item.card.scrap, item.card.scrap_amount, item)

    def _acquire_market(
        self, player: _Player, slot: int, cost: int, force_top: bool = False
    ) -> None:
        card = self.trade_row[slot]
        if card is None:
            raise RuntimeError("cannot acquire an empty market slot")
        player.trade -= cost
        self._place_acquired(player, card, force_top=force_top)
        self._refill_market_slot(slot)

    def _acquire_explorer(self, player: _Player, cost: int) -> None:
        if self.explorers_remaining <= 0:
            raise RuntimeError("Explorer supply exhausted")
        player.trade -= cost
        self.explorers_remaining -= 1
        self._place_acquired(player, EXPLORER)

    def _place_acquired(self, player: _Player, card: Card, force_top: bool = False) -> None:
        if card.is_ship and (force_top or player.next_ship_top):
            player.deck.append(card)
            player.known_top.append(card)
            player.next_ship_top = False
        else:
            player.discard.append(card)

    def _refill_market_slot(self, slot: int) -> None:
        self.trade_row[slot] = self.trade_deck.pop() if self.trade_deck else None

    def _draw(self, player: _Player, count: int) -> None:
        for _ in range(count):
            if not player.deck:
                if not player.discard:
                    break
                player.deck.extend(player.discard)
                player.discard.clear()
                player.rng.shuffle(player.deck)
                player.known_top.clear()
            card = player.deck.pop()
            player.hand.append(card)
            if player.known_top and player.known_top[-1].card_id == card.card_id:
                player.known_top.pop()
                player.revealed_hand.append(card)

    def _cleanup_and_draw(self, player: _Player) -> None:
        for card in player.hand:
            player.discard.append(card)
        player.hand.clear()
        player.revealed_hand.clear()
        ships = [item for item in player.in_play if item.card.is_ship]
        for item in ships:
            player.in_play.remove(item)
            player.discard.append(item.original_card)
        player.combat = 0
        player.trade = 0
        player.next_ship_top = False
        player.blob_cards_played = 0
        self._draw(player, 5)

    @staticmethod
    def _find_in_play(player: _Player, uid: int) -> _InPlay:
        for item in player.in_play:
            if item.uid == uid:
                return item
        raise RuntimeError("in-play target no longer exists")

    @staticmethod
    def _forget_revealed(player: _Player, card: Card) -> None:
        for index, known in enumerate(player.revealed_hand):
            if known.card_id == card.card_id:
                player.revealed_hand.pop(index)
                break

    def _discard_from_hand(self, player: _Player, index: int) -> None:
        card = player.hand.pop(index)
        self._forget_revealed(player, card)
        player.discard.append(card)

    @staticmethod
    def _fleet_hq_active(player: _Player) -> bool:
        return any(item.card.card_id == 29 for item in player.in_play)

    @staticmethod
    def _base_requires_activation(card: Card) -> bool:
        return (
            card.is_base
            and card.card_id not in (19, 29)
            and bool(card.combat or card.authority or card.trade or card.primary)
        )


def play_game(
    chooser_0: Chooser = first_chooser,
    chooser_1: Chooser = first_chooser,
    *,
    seed: int = 0,
    seating: Seating = Seating.FIXED,
    max_turns: int = 400,
    max_actions_per_turn: int = 200,
    cancel_hook: CancelHook | None = None,
    decision_hook: DecisionHook | None = None,
) -> GameResult:
    """Convenience function used by CPU self-play workers."""

    game = Game(
        choosers=(chooser_0, chooser_1),
        config=GameConfig(
            seed=seed,
            seating=seating,
            max_turns=max_turns,
            max_actions_per_turn=max_actions_per_turn,
        ),
        cancel_hook=cancel_hook,
        decision_hook=decision_hook,
    )
    return game.run()


__all__ = [
    "Action",
    "ActionKind",
    "Decision",
    "DecisionFamily",
    "Game",
    "GameConfig",
    "GameResult",
    "InPlayObservation",
    "Observation",
    "RNGStreams",
    "Seating",
    "first_chooser",
    "make_random_chooser",
    "play_game",
]
