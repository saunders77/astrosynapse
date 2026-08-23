"""Fast, exact encoder for the typed Astro 2 engine.

The generic :mod:`astro2.encoding` adapter deliberately accepts legacy maps,
tuples, aliases, and camel-cased fields. Production self-play has a much
narrower contract: immutable ``Observation`` and ``Action`` dataclasses with
stable integer card IDs. This encoder exploits that contract while retaining
the generic encoder as a fallback and differential-test oracle.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from .cards import ALL_CARDS
from .encoding import (
    _IN_PLAY_STATUS_FIELDS,
    _STATE_SCALARS,
    ACTION_KIND_COUNT,
    ACTION_NUMERIC_SIZE,
    CARD_ATTRIBUTE_SIZE,
    EFFECT_COUNT,
    KNOWN_TOP_SLOTS,
    RELATION_FEATURE_SIZE,
    ZONE_COUNT,
    ActionKind,
    DecisionEncoding,
    DecisionFamily,
    EffectKind,
    Encoder,
    SemanticAction,
    Zone,
    _effect_kind,
    _RelationContext,
)
from .engine import (
    Action,
    Decision,
    InPlayObservation,
    Observation,
)
from .engine import ActionKind as EngineActionKind
from .engine import DecisionFamily as EngineDecisionFamily

_ENGINE_ACTION_KINDS = {
    EngineActionKind.PLAY_CARD: ActionKind.PLAY,
    EngineActionKind.ACTIVATE_BASE: ActionKind.ACTIVATE_ABILITY,
    EngineActionKind.SCRAP_FOR_ABILITY: ActionKind.SCRAP_FROM_PLAY,
    EngineActionKind.ATTACK_BASE: ActionKind.ATTACK_BASE,
    EngineActionKind.ATTACK_PLAYER: ActionKind.ATTACK_PLAYER,
    EngineActionKind.ACQUIRE: ActionKind.ACQUIRE,
    EngineActionKind.END_TURN: ActionKind.END_TURN,
    EngineActionKind.DISCARD_CARD: ActionKind.DISCARD,
    EngineActionKind.SCRAP_CARD: ActionKind.SCRAP_CARD,
    EngineActionKind.COPY_SHIP: ActionKind.COPY_SHIP,
    EngineActionKind.DESTROY_BASE: ActionKind.DESTROY_BASE,
    EngineActionKind.SCRAP_TRADE_ROW: ActionKind.SCRAP_TRADE_ROW,
    EngineActionKind.FREE_ACQUIRE: ActionKind.FREE_ACQUIRE,
    EngineActionKind.DECLINE: ActionKind.DECLINE,
}

_ENGINE_MODE_KINDS = {
    "gain_combat": ActionKind.GAIN_ATTACK,
    "gain_attack": ActionKind.GAIN_ATTACK,
    "gain_trade": ActionKind.GAIN_TRADE,
    "gain_authority": ActionKind.GAIN_AUTHORITY,
    "draw": ActionKind.DRAW,
    "cycle": ActionKind.RECYCLE,
    "recycle": ActionKind.RECYCLE,
}

_ENGINE_FAMILIES = {
    EngineDecisionFamily.MAIN: DecisionFamily.MAIN,
    EngineDecisionFamily.DISCARD: DecisionFamily.DISCARD,
    EngineDecisionFamily.SCRAP: DecisionFamily.SCRAP,
    EngineDecisionFamily.DESTROY_BASE: DecisionFamily.DESTROY_BASE,
    EngineDecisionFamily.SCRAP_TRADE_ROW: DecisionFamily.SCRAP_TRADE_ROW,
    EngineDecisionFamily.COPY_SHIP: DecisionFamily.COPY_SHIP,
    EngineDecisionFamily.FREE_ACQUIRE: DecisionFamily.FREE_ACQUIRE,
    EngineDecisionFamily.ABILITY_MODE: DecisionFamily.ABILITY_MODE,
}

_ENGINE_ZONES = {
    "hand": int(Zone.OWN_HAND),
    "own_hand": int(Zone.OWN_HAND),
    "discard": int(Zone.OWN_DISCARD),
    "own_discard": int(Zone.OWN_DISCARD),
    "in_play": int(Zone.OWN_IN_PLAY),
    "own_in_play": int(Zone.OWN_IN_PLAY),
    "opponent_in_play": int(Zone.OPPONENT_IN_PLAY),
    "trade_row": int(Zone.TRADE_ROW),
    "explorer_supply": int(Zone.EXPLORER_SUPPLY),
}

_TARGET_ZONE_KINDS = frozenset(
    {
        ActionKind.ATTACK_BASE,
        ActionKind.ACQUIRE,
        ActionKind.DESTROY_BASE,
        ActionKind.SCRAP_TRADE_ROW,
        ActionKind.COPY_SHIP,
        ActionKind.FREE_ACQUIRE,
    }
)


class EngineEncoder(Encoder):
    """Typed, incrementally cached encoder for engine observations/actions.

    ``max_count_cache`` bounds reusable multiset encodings across games. The
    per-player state cache additionally keeps the previous encoded state and
    replaces only zone, known-top, and in-play-status blocks whose stable card
    ID signatures changed.
    """

    def __init__(
        self,
        *,
        strict: bool = True,
        version: int = 1,
        max_count_cache: int = 4_096,
    ) -> None:
        if max_count_cache < 1:
            raise ValueError("max_count_cache must be positive")
        super().__init__(card_catalog=ALL_CARDS, strict=strict, version=version)
        self.max_count_cache = int(max_count_cache)
        self._count_cache: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        self._state_cache: dict[
            int,
            tuple[
                np.ndarray,
                tuple[tuple[int, ...], ...],
                tuple[tuple[int, ...], tuple[int, ...]],
                tuple[tuple[int, ...], ...],
            ],
        ] = {}
        self._action_templates: dict[tuple[int, int, int, int, int, int], np.ndarray] = {}
        self._effect_cache: dict[str, EffectKind] = {"": EffectKind.NONE}
        self._card_attribute_table = tuple(
            self._card_attributes(card, card.card_id) for card in ALL_CARDS
        )
        # Engine actions use -1 rather than None for an absent card. Preserve
        # the established encoding contract, whose generic path represents
        # that sentinel as the default unaligned/ship attribute pair.
        self._missing_card_attributes = self._card_attributes(-1, None)
        self._zero_card_attributes = (0.0,) * CARD_ATTRIBUTE_SIZE
        self._numeric_start = (
            ACTION_KIND_COUNT
            + 2 * (self.card_count + CARD_ATTRIBUTE_SIZE)
            + 2 * ZONE_COUNT
            + EFFECT_COUNT
        )

    @staticmethod
    def _card_ids(cards: Sequence[object], *, ordered: bool = False) -> tuple[int, ...]:
        ids = tuple(
            item.card.card_id if isinstance(item, InPlayObservation) else item.card_id
            for item in cards
            if item is not None
        )
        return ids if ordered else tuple(sorted(ids))

    def _counts(self, signature: tuple[int, ...]) -> np.ndarray:
        cached = self._count_cache.get(signature)
        if cached is None:
            cached = np.bincount(signature, minlength=self.card_count).astype(np.float32)
            cached.flags.writeable = False
            self._count_cache[signature] = cached
            if len(self._count_cache) > self.max_count_cache:
                self._count_cache.popitem(last=False)
        else:
            self._count_cache.move_to_end(signature)
        return cached

    @staticmethod
    def _zone_values(observation: Observation) -> tuple[Sequence[object], ...]:
        return (
            observation.hand,
            observation.own_deck,
            observation.own_known_top,
            observation.own_discard,
            observation.own_in_play,
            observation.opponent_hidden,
            observation.opponent_known_hand,
            observation.opponent_known_top,
            observation.opponent_discard,
            observation.opponent_in_play,
            observation.trade_row,
            observation.trade_deck,
            observation.scrap_heap,
            observation.explorer_supply,
        )

    @staticmethod
    def _state_scalars(observation: Observation) -> tuple[float, ...]:
        return (
            float(observation.is_starting_player),
            observation.own_authority / 50.0,
            observation.opponent_authority / 50.0,
            observation.combat / 20.0,
            observation.trade / 15.0,
            observation.pending_discard / 5.0,
            observation.opponent_pending_discard / 5.0,
            float(observation.next_ship_to_top),
            observation.blob_cards_played / 5.0,
            float(observation.all_allied),
            float(observation.fleet_active),
            observation.turn / 50.0,
            observation.action_number / 50.0,
            observation.own_deck_count / 20.0,
            observation.opponent_hand_count / 10.0,
            observation.opponent_deck_count / 20.0,
            observation.trade_deck_count / 80.0,
            observation.explorers_remaining / 10.0,
        )

    @staticmethod
    def _status_signatures(observation: Observation) -> tuple[tuple[int, ...], ...]:
        result: list[tuple[int, ...]] = []
        for cards in (observation.own_in_play, observation.opponent_in_play):
            for status in _IN_PLAY_STATUS_FIELDS:
                if status == "ready":
                    ids = (item.card.card_id for item in cards if not item.activated)
                else:
                    ids = (
                        item.card.card_id
                        for item in cards
                        if bool(getattr(item, status))
                    )
                result.append(tuple(sorted(ids)))
        return tuple(result)

    def encode_state(self, observation: object) -> np.ndarray:
        if not isinstance(observation, Observation):
            return super().encode_state(observation)

        zone_signatures = tuple(self._card_ids(zone) for zone in self._zone_values(observation))
        ordered_signatures = (
            self._card_ids(observation.own_known_top, ordered=True),
            self._card_ids(observation.opponent_known_top, ordered=True),
        )
        status_signatures = self._status_signatures(observation)
        cached = self._state_cache.get(observation.player_id)
        if cached is None:
            result = np.zeros(self.state_size, dtype=np.float32)
            previous_zones: tuple[tuple[int, ...], ...] = ()
            previous_ordered: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
            previous_statuses: tuple[tuple[int, ...], ...] = ()
        else:
            cached_state, previous_zones, previous_ordered, previous_statuses = cached
            result = cached_state.copy()

        scalars = self._state_scalars(observation)
        if len(scalars) != len(_STATE_SCALARS):
            raise RuntimeError("typed state scalar table is out of sync")
        result[: len(scalars)] = scalars

        zone_start = len(_STATE_SCALARS)
        for zone_index, signature in enumerate(zone_signatures):
            if zone_index >= len(previous_zones) or signature != previous_zones[zone_index]:
                start = zone_start + zone_index * self.card_count
                result[start : start + self.card_count] = self._counts(signature)

        ordered_start = zone_start + ZONE_COUNT * self.card_count
        for side, signature in enumerate(ordered_signatures):
            if signature != previous_ordered[side]:
                start = ordered_start + side * KNOWN_TOP_SLOTS * self.card_count
                block = result[start : start + KNOWN_TOP_SLOTS * self.card_count]
                block.fill(0.0)
                for slot, card_id in enumerate(signature[:KNOWN_TOP_SLOTS]):
                    block[slot * self.card_count + card_id] = 1.0

        status_start = ordered_start + 2 * KNOWN_TOP_SLOTS * self.card_count
        for index, signature in enumerate(status_signatures):
            if index >= len(previous_statuses) or signature != previous_statuses[index]:
                start = status_start + index * self.card_count
                result[start : start + self.card_count] = self._counts(signature)

        cached_state = result.copy()
        cached_state.flags.writeable = False
        self._state_cache[observation.player_id] = (
            cached_state,
            zone_signatures,
            ordered_signatures,
            status_signatures,
        )
        return result

    def _engine_kind(self, action: Action) -> ActionKind:
        if action.kind == EngineActionKind.CHOOSE_MODE:
            return _ENGINE_MODE_KINDS.get(action.ability, ActionKind.UNKNOWN)
        return _ENGINE_ACTION_KINDS.get(action.kind, ActionKind.UNKNOWN)

    def _engine_effect(self, ability: str) -> EffectKind:
        effect = self._effect_cache.get(ability)
        if effect is None:
            effect = _effect_kind(ability)
            self._effect_cache[ability] = effect
        return effect

    def semantic_action(self, action: object) -> SemanticAction:
        if not isinstance(action, Action):
            return super().semantic_action(action)

        kind = self._engine_kind(action)
        source_card = action.card_id if 0 <= action.card_id < self.card_count else -1
        target_card = (
            action.target_card_id if 0 <= action.target_card_id < self.card_count else -1
        )
        source_zone = _ENGINE_ZONES.get(action.source_zone, -1)
        target_zone = -1

        if kind == ActionKind.ACQUIRE:
            target_card, source_card = source_card, -1
        if kind in _TARGET_ZONE_KINDS:
            target_zone, source_zone = source_zone, -1
        if source_zone < 0:
            if kind in {ActionKind.PLAY, ActionKind.DISCARD}:
                source_zone = int(Zone.OWN_HAND)
            elif kind == ActionKind.SCRAP_CARD:
                source_zone = int(
                    Zone.OWN_DISCARD if action.source_zone == "discard" else Zone.OWN_HAND
                )
            elif kind == ActionKind.SCRAP_FROM_PLAY:
                source_zone = int(Zone.OWN_IN_PLAY)
        if target_zone < 0:
            if kind in {ActionKind.ACQUIRE, ActionKind.FREE_ACQUIRE, ActionKind.SCRAP_TRADE_ROW}:
                target_zone = int(Zone.TRADE_ROW)
            elif kind in {ActionKind.ATTACK_BASE, ActionKind.DESTROY_BASE}:
                target_zone = int(Zone.OPPONENT_IN_PLAY)
            elif kind == ActionKind.COPY_SHIP:
                target_zone = int(Zone.OWN_IN_PLAY)

        amount = float(action.amount)
        cost = amount if kind in {ActionKind.ACQUIRE, ActionKind.FREE_ACQUIRE} else 0.0
        attack_available = 0.0
        target_defense = 0.0
        if kind == ActionKind.ATTACK_BASE:
            target_defense = amount
            attack_available = float(action.amount2)
        elif kind == ActionKind.ATTACK_PLAYER:
            attack_available = amount
        numbers = (
            amount,
            cost,
            attack_available,
            target_defense,
            0.0,
            0.0,
            0.0,
            0.0,
            float(kind == ActionKind.DECLINE),
            0.0,
            float(kind == ActionKind.FREE_ACQUIRE),
            0.0,
        )
        if source_card >= 0:
            source_attributes = self._card_attribute_table[source_card]
        elif kind == ActionKind.ACQUIRE:
            source_attributes = self._zero_card_attributes
        else:
            source_attributes = self._missing_card_attributes
        target_attributes = (
            self._card_attribute_table[target_card]
            if target_card >= 0
            else self._missing_card_attributes
        )
        return SemanticAction(
            kind=kind,
            source_card=source_card,
            target_card=target_card,
            source_zone=source_zone,
            target_zone=target_zone,
            effect=self._engine_effect(action.ability),
            numbers=numbers,
            source_attributes=source_attributes,
            target_attributes=target_attributes,
        )

    def _encode_engine_action(self, action: Action) -> tuple[SemanticAction, np.ndarray]:
        semantic = self.semantic_action(action)
        key = (
            int(semantic.kind),
            semantic.source_card,
            semantic.target_card,
            semantic.source_zone,
            semantic.target_zone,
            int(semantic.effect),
        )
        template = self._action_templates.get(key)
        if template is None:
            template = super()._encode_semantic_action(
                SemanticAction(
                    kind=semantic.kind,
                    source_card=semantic.source_card,
                    target_card=semantic.target_card,
                    source_zone=semantic.source_zone,
                    target_zone=semantic.target_zone,
                    effect=semantic.effect,
                    source_attributes=semantic.source_attributes,
                    target_attributes=semantic.target_attributes,
                )
            )
            template.flags.writeable = False
            self._action_templates[key] = template
        result = template.copy()
        result[self._numeric_start : self._numeric_start + ACTION_NUMERIC_SIZE] = semantic.numbers
        return semantic, result

    def encode_action(self, action: object) -> np.ndarray:
        if not isinstance(action, Action):
            return super().encode_action(action)
        return self._encode_engine_action(action)[1]

    def _engine_relation_context(
        self,
        observation: Observation,
        encoded_state: np.ndarray,
    ) -> _RelationContext:
        own_zones = (
            Zone.OWN_HAND,
            Zone.OWN_DRAW,
            Zone.OWN_KNOWN_TOP,
            Zone.OWN_DISCARD,
            Zone.OWN_IN_PLAY,
        )
        draw_zones = (Zone.OWN_DRAW, Zone.OWN_KNOWN_TOP, Zone.OWN_DISCARD)
        own_counts = {zone: self.zone_counts(encoded_state, zone) for zone in own_zones}

        def faction_counts(zones: Sequence[Zone]) -> dict[str, float]:
            return {
                faction: sum(float(own_counts[zone][ids].sum()) for zone in zones)
                for faction, ids in self._relation_faction_ids.items()
            }

        def ally_faction_counts(zones: Sequence[Zone]) -> dict[str, float]:
            return {
                faction: sum(float(own_counts[zone][ids].sum()) for zone in zones)
                if len(ids)
                else 0.0
                for faction, ids in self._relation_ally_faction_ids.items()
            }

        known_cards = observation.opponent_known_top
        return _RelationContext(
            owned_total=sum(float(own_counts[zone].sum()) for zone in own_zones),
            owned_factions=faction_counts(own_zones),
            owned_ally_factions=ally_faction_counts(own_zones),
            draw_factions=faction_counts(draw_zones),
            in_play_factions=faction_counts((Zone.OWN_IN_PLAY,)),
            known_card_count=len(known_cards),
            known_combat=sum(float(card.combat) for card in known_cards),
            known_trade=sum(float(card.trade) for card in known_cards),
            known_draws=sum(
                card.primary in {"draw", "draw_two"} or card.ally in {"draw", "draw_two"}
                for card in known_cards
            ),
            known_factions=frozenset(card.faction.value for card in known_cards),
            combat=float(observation.combat),
            opponent_authority=float(observation.opponent_authority),
            trade=float(observation.trade),
            turn=observation.turn,
        )

    def encode_decision(
        self,
        observation: object,
        decision_or_actions: object,
        *,
        family: DecisionFamily | int | str | None = None,
    ) -> DecisionEncoding:
        if not isinstance(observation, Observation) or not isinstance(
            decision_or_actions, Decision
        ):
            return super().encode_decision(
                observation, decision_or_actions, family=family
            )

        resolved_family = (
            _ENGINE_FAMILIES[decision_or_actions.family]
            if family is None
            else self._coerce_family(family)
        )
        state = self.encode_state(observation)
        semantics: list[SemanticAction] = []
        encoded_actions = np.empty(
            (len(decision_or_actions.actions), self.action_size), dtype=np.float32
        )
        for index, action in enumerate(decision_or_actions.actions):
            semantic, encoded_action = self._encode_engine_action(action)
            semantics.append(semantic)
            encoded_actions[index] = encoded_action
        if self.version >= 2:
            context = self._engine_relation_context(observation, state)
            for index, semantic in enumerate(semantics):
                encoded_actions[index, -RELATION_FEATURE_SIZE:] = (
                    self._relation_features_from_semantic(semantic, context)
                )
        return DecisionEncoding(
            state=state,
            actions=encoded_actions,
            family=resolved_family,
        )
