from dataclasses import dataclass, replace

import numpy as np
import pytest
from astro2.cards import CARD_BY_ID, CARD_BY_NAME
from astro2.encoding import (
    RELATION_FEATURE_SIZE,
    ActionKind,
    DecisionFamily,
    Encoder,
    Zone,
    action_kind,
    decision_family,
)
from astro2.engine import Action, Game, GameConfig, InPlayObservation
from astro2.engine import ActionKind as EngineActionKind
from astro2.engine_encoding import EngineEncoder


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
    relocated = Action(EngineActionKind.PLAY_CARD, card_id=3, source_zone="hand", opaque=(999_999,))
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


@pytest.mark.parametrize("version", [1, 2])
def test_engine_encoder_exactly_matches_generic_encoder_across_random_games(version):
    reference = Encoder(version=version)
    native = EngineEncoder(version=version)
    rng = np.random.default_rng(8_100 + version)
    decisions = 0

    def chooser(_player_id, decision):
        return int(rng.integers(len(decision.actions)))

    def audit(_player_id, decision, _selected):
        nonlocal decisions
        expected = reference.encode_decision(decision.observation, decision)
        actual = native.encode_decision(decision.observation, decision)
        repeated = native.encode_decision(decision.observation, decision)
        assert actual.family == expected.family
        np.testing.assert_array_equal(actual.state, expected.state)
        np.testing.assert_array_equal(actual.actions, expected.actions)
        np.testing.assert_array_equal(repeated.state, expected.state)
        np.testing.assert_array_equal(repeated.actions, expected.actions)
        decisions += 1

    for seed in range(81, 85):
        Game(
            choosers=(chooser, chooser),
            config=GameConfig(seed=seed, max_turns=60, max_actions_per_turn=160),
            decision_hook=audit,
        ).run()
    assert decisions > 100


@pytest.mark.parametrize("version", [1, 2])
def test_engine_encoder_matches_every_engine_decision_family(version):
    from astro2.diagnostics import all_family_decision_suite

    reference = Encoder(version=version)
    native = EngineEncoder(version=version)
    decisions = all_family_decision_suite(seed=113)

    assert {decision.family for decision in decisions} == set(type(decisions[0].family))
    for decision in decisions:
        expected = reference.encode_decision(decision.observation, decision)
        actual = native.encode_decision(decision.observation, decision)
        assert actual.family == expected.family
        np.testing.assert_array_equal(actual.state, expected.state)
        np.testing.assert_array_equal(actual.actions, expected.actions)


def test_engine_encoder_incremental_cache_invalidates_changed_state_blocks():
    reference = Encoder(version=2)
    native = EngineEncoder(version=2)
    game = Game(config=GameConfig(seed=127))
    base = game.observation(0)
    changed = replace(
        base,
        own_authority=31,
        combat=7,
        hand=(CARD_BY_ID[0], CARD_BY_ID[1], CARD_BY_ID[3]),
        own_discard=(CARD_BY_ID[4], CARD_BY_ID[4]),
        own_known_top=(CARD_BY_ID[7], CARD_BY_ID[10]),
        opponent_known_top=(CARD_BY_ID[27],),
        own_in_play=(
            InPlayObservation(CARD_BY_ID[9], False, True, False),
            InPlayObservation(CARD_BY_ID[23], True, False, True),
        ),
        opponent_in_play=(
            InPlayObservation(CARD_BY_ID[15], False, False, False),
        ),
    )
    reordered = replace(
        changed,
        own_known_top=tuple(reversed(changed.own_known_top)),
        own_in_play=(
            InPlayObservation(CARD_BY_ID[9], True, False, False),
            InPlayObservation(CARD_BY_ID[23], False, True, True),
        ),
    )

    for observation in (base, changed, game.observation(1), reordered, base):
        np.testing.assert_array_equal(
            native.encode_state(observation), reference.encode_state(observation)
        )


def test_astro3_relational_features_expose_ally_acquisition_context():
    game = Game(config=GameConfig(seed=91))
    observation = game.observation(0)
    acquire_blob_fighter = Action(
        EngineActionKind.ACQUIRE,
        card_id=CARD_BY_NAME["Blob Fighter"].card_id,
        source_zone="trade_row",
        amount=1,
    )
    decision = (
        acquire_blob_fighter,
        Action(EngineActionKind.END_TURN),
    )
    unsupported = replace(observation, own_discard=(), own_deck=(), hand=())
    supported = replace(
        unsupported,
        own_discard=(CARD_BY_NAME["Battle Pod"], CARD_BY_NAME["Ram"]),
    )

    legacy = Encoder(version=1)
    np.testing.assert_array_equal(
        legacy.encode_decision(unsupported, decision).actions,
        legacy.encode_decision(supported, decision).actions,
    )

    astro3 = Encoder(version=2)
    assert astro3.action_size == legacy.action_size + RELATION_FEATURE_SIZE
    unsupported_relation = astro3.encode_decision(unsupported, decision).actions[
        0, -RELATION_FEATURE_SIZE:
    ]
    supported_relation = astro3.encode_decision(supported, decision).actions[
        0, -RELATION_FEATURE_SIZE:
    ]
    assert unsupported_relation[3] == 1  # The candidate has an ally ability.
    assert unsupported_relation[4] == 0  # No other Blob card is owned.
    assert supported_relation[0] == pytest.approx(0.2)
    assert supported_relation[4] == 1
    assert supported_relation[5] > unsupported_relation[5]

    # Blob Wheel has no ally text itself, but pairing it with an owned Blob
    # Fighter can trigger the Fighter's ally ability on a future draw.
    acquire_blob_wheel = Action(
        EngineActionKind.ACQUIRE,
        card_id=CARD_BY_NAME["Blob Wheel"].card_id,
        source_zone="trade_row",
        amount=3,
    )
    wheel_relations = astro3.encode_decision(
        replace(unsupported, own_discard=(CARD_BY_NAME["Blob Fighter"],)),
        (acquire_blob_wheel, Action(EngineActionKind.END_TURN)),
    ).actions[0, -RELATION_FEATURE_SIZE:]
    assert wheel_relations[3] == 0
    assert wheel_relations[4] == 1


def test_astro3_batched_relation_context_matches_per_action_reference():
    game = Game(config=GameConfig(seed=92))
    observation = replace(
        game.observation(0),
        combat=7,
        trade=5,
        turn=11,
        own_discard=(CARD_BY_NAME["Battle Pod"], CARD_BY_NAME["Blob Fighter"]),
        opponent_known_top=(CARD_BY_NAME["Corvette"], CARD_BY_NAME["Survey Ship"]),
    )
    actions = (
        Action(
            EngineActionKind.ACQUIRE,
            card_id=CARD_BY_NAME["Blob Carrier"].card_id,
            source_zone="trade_row",
            amount=6,
        ),
        Action(
            EngineActionKind.ATTACK_BASE,
            target_card_id=CARD_BY_NAME["Space Station"].card_id,
            amount=4,
        ),
        Action(
            EngineActionKind.SCRAP_FOR_ABILITY,
            card_id=CARD_BY_NAME["Battlecruiser"].card_id,
            source_zone="own_in_play",
            ability="draw",
            amount=1,
        ),
        Action(EngineActionKind.END_TURN),
    )

    legacy = Encoder(version=1).encode_decision(observation, actions)
    encoder = Encoder(version=2)
    encoded = encoder.encode_decision(observation, actions)
    state = encoder.encode_state(observation)
    per_action_reference = np.stack(
        [encoder._relation_features(observation, action, state) for action in actions]
    )

    np.testing.assert_array_equal(encoded.state, legacy.state)
    np.testing.assert_array_equal(encoded.actions[:, :-RELATION_FEATURE_SIZE], legacy.actions)
    np.testing.assert_array_equal(
        encoded.actions[:, -RELATION_FEATURE_SIZE:],
        per_action_reference,
    )
    attack_relations = encoded.actions[1, -RELATION_FEATURE_SIZE:]
    assert attack_relations[10] == pytest.approx(0.4)  # Target defense / 10.
    assert attack_relations[11] == pytest.approx(1.75)  # Current combat / defense.
    assert attack_relations[7] > 0  # Opponent known-top combat remains separate.
