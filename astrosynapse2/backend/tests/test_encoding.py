from dataclasses import dataclass, replace

import numpy as np
import pytest
from astro2.cards import CARD_BY_ID
from astro2.encoding import (
    ActionKind,
    DecisionFamily,
    Encoder,
    Zone,
    action_kind,
    decision_family,
)
from astro2.engine import Action, Game, GameConfig, InPlayObservation
from astro2.engine import ActionKind as EngineActionKind


def card(name, cost=0, attack=0, authority=0, trade=0, faction="none", kind="ship", shield=0):
    return (
        name,
        cost,
        attack,
        authority,
        trade,
        faction,
        kind,
        "none",
        "none",
        0,
        "none",
        0,
        shield,
        0,
    )


SCOUT = card("Scout", trade=1)
VIPER = card("Viper", attack=1)
EXPLORER = card("Explorer", cost=2, trade=2)


def test_zone_encoding_is_exact_counts_and_order_invariant():
    encoder = Encoder()
    first = {
        "authority": 47,
        "hand": [SCOUT, VIPER, SCOUT],
        "scrambleDeck": [VIPER, SCOUT],
        "discardPile": [EXPLORER],
        "cardsInPlay": {"none": [[SCOUT, False, None, True, False]]},
        "tradeRow": [EXPLORER, VIPER],
    }
    second = {
        **first,
        "hand": [SCOUT, SCOUT, VIPER],
        "scrambleDeck": [SCOUT, VIPER],
        "tradeRow": [VIPER, EXPLORER],
    }

    first_encoded = encoder.encode_state(first)
    second_encoded = encoder.encode_state(second)
    np.testing.assert_array_equal(first_encoded, second_encoded)
    hand = encoder.zone_counts(first_encoded, Zone.OWN_HAND)
    assert hand[0] == 2
    assert hand[1] == 1
    assert hand.sum() == 3
    assert encoder.zone_counts(first_encoded, Zone.OWN_IN_PLAY)[0] == 1


def test_candidate_encoding_never_uses_raw_list_position():
    encoder = Encoder()
    at_position_zero = ("play", 0, SCOUT)
    at_arbitrary_position = ("play", 987_654, SCOUT)
    np.testing.assert_array_equal(
        encoder.encode_action(at_position_zero), encoder.encode_action(at_arbitrary_position)
    )
    assert not np.array_equal(
        encoder.encode_action(at_position_zero), encoder.encode_action(("play", 0, VIPER))
    )


@dataclass(frozen=True)
class Candidate:
    kind: str
    card: int
    index: int
    source_zone: str = "own_hand"


def test_structured_candidate_also_ignores_index():
    encoder = Encoder()
    first = Candidate("play_card", card=0, index=2)
    second = Candidate("play_card", card=0, index=200)
    np.testing.assert_array_equal(encoder.encode_action(first), encoder.encode_action(second))


def test_decision_families_have_explicit_stable_ids():
    assert [int(family) for family in DecisionFamily] == list(range(8))
    assert decision_family([("play", 0, SCOUT), ("endTurn",)]) == DecisionFamily.MAIN
    assert decision_family([("discardNormal", 0, SCOUT), ("nodiscard",)]) == DecisionFamily.DISCARD
    assert decision_family([("gainattack", 5), ("draw", 2)]) == DecisionFamily.ABILITY_MODE
    assert action_kind(("freeAcquire", 5, EXPLORER)) == ActionKind.FREE_ACQUIRE


def test_actual_engine_observation_aliases_order_and_status_are_encoded():
    encoder = Encoder()
    game = Game(config=GameConfig(seed=9))
    observation = replace(
        game.observation(0),
        own_authority=37,
        combat=9,
        pending_discard=2,
        next_ship_to_top=True,
        blob_cards_played=3,
        own_known_top=(CARD_BY_ID[3], CARD_BY_ID[4]),
        opponent_known_top=(CARD_BY_ID[7],),
        own_in_play=(
            InPlayObservation(CARD_BY_ID[9], False, True, False),
            InPlayObservation(CARD_BY_ID[23], True, False, True),
        ),
    )
    state = encoder.encode_state(observation)

    assert encoder.scalar_value(state, "is_starting_player") == 1
    assert encoder.scalar_value(state, "authority") == pytest.approx(37 / 50)
    assert encoder.scalar_value(state, "attack") == pytest.approx(9 / 20)
    assert encoder.scalar_value(state, "must_discard") == pytest.approx(2 / 5)
    assert encoder.scalar_value(state, "next_ship_top") == 1
    assert encoder.scalar_value(state, "blob_play_count") == pytest.approx(3 / 5)
    assert encoder.known_top_slot(state, 0)[3] == 1
    assert encoder.known_top_slot(state, 1)[4] == 1
    assert encoder.known_top_slot(state, 0, opponent=True)[7] == 1
    assert encoder.in_play_status_counts(state, "ready")[9] == 1
    assert encoder.in_play_status_counts(state, "ally_triggered")[9] == 1
    assert encoder.in_play_status_counts(state, "copied_from_stealth_needle")[23] == 1
    assert encoder.zone_counts(state, Zone.OWN_DRAW).sum() == observation.own_deck_count
    assert encoder.zone_counts(state, Zone.OPPONENT_HIDDEN).sum() == (
        observation.opponent_hand_count + observation.opponent_deck_count
    )
    assert encoder.zone_counts(state, Zone.TRADE_DECK).sum() == observation.trade_deck_count


def test_actual_engine_actions_use_semantics_but_never_opaque_locator():
    encoder = Encoder()
    first = Action(EngineActionKind.PLAY_CARD, card_id=3, source_zone="hand", opaque=(0,))
    relocated = Action(
        EngineActionKind.PLAY_CARD, card_id=3, source_zone="hand", opaque=(999_999,)
    )
    np.testing.assert_array_equal(encoder.encode_action(first), encoder.encode_action(relocated))
    semantic = encoder.semantic_action(first)
    assert semantic.source_card == 3
    assert semantic.source_attributes[0] == 6  # cost
    assert semantic.source_attributes[1] == 8  # combat
    assert semantic.source_attributes[6] == 1  # Blob/green faction channel


def test_every_real_engine_decision_encodes_without_unknown_family_or_action():
    encoder = Encoder()
    rng = np.random.default_rng(81)
    encoded_decisions = 0

    def chooser(_player_id, decision):
        nonlocal encoded_decisions
        encoded = encoder.encode_decision(decision.observation, decision)
        assert encoded.actions.shape == (len(decision.actions), encoder.action_size)
        encoded_decisions += 1
        return int(rng.integers(len(decision.actions)))

    result = Game(
        choosers=(chooser, chooser),
        config=GameConfig(seed=81, max_turns=80, max_actions_per_turn=160),
    ).run()
    assert encoded_decisions == result.decisions - result.forced_choices
    assert encoded_decisions > 20
