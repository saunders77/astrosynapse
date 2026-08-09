from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest
from astro2.cards import (
    ALL_CARDS,
    CARD_BY_NAME,
    EXPLORER,
    SCOUT,
    VIPER,
    CardType,
    build_trade_deck,
)
from astro2.engine import (
    Action,
    ActionKind,
    Decision,
    DecisionFamily,
    Game,
    GameConfig,
    Seating,
    _InPlay,
    make_random_chooser,
)


def end_turn_chooser(_player_id, decision):
    for index, action in enumerate(decision.actions):
        if action.kind == ActionKind.END_TURN:
            return index
    return 0


def test_card_catalog_and_canonical_market_shape():
    assert len(ALL_CARDS) == 49
    assert len(build_trade_deck()) == 80
    game = Game(config=GameConfig(seed=4))
    assert len(game.trade_row) == 5
    assert EXPLORER not in game.trade_row
    assert game.explorers_remaining == 10


def test_scrap_action_labels_identify_the_source_zone():
    assert (
        Action(ActionKind.SCRAP_CARD, card_id=SCOUT.card_id, source_zone="hand").label
        == "scrap card Scout from hand"
    )
    assert (
        Action(ActionKind.SCRAP_CARD, card_id=SCOUT.card_id, source_zone="discard").label
        == "scrap card Scout from discard pile"
    )


def test_public_types_are_immutable_and_human_serializable():
    game = Game(config=GameConfig(seed=2))
    observation = game.observation(0)
    action = Action(ActionKind.END_TURN)
    with pytest.raises(FrozenInstanceError):
        action.amount = 2
    with pytest.raises(FrozenInstanceError):
        observation.trade = 99
    with pytest.raises(FrozenInstanceError):
        SCOUT.cost = 8
    assert '"kind": "end_turn"' in action.to_json()
    assert '"trade_row"' in observation.to_json()
    decision = Decision(DecisionFamily.MAIN, observation, (action,))
    assert "opaque" not in decision.to_json()
    assert "hand" not in game.state_dict()["players"][1]
    assert "hand" in game.state_dict(include_hidden=True)["players"][1]


def test_same_seed_and_policy_streams_reproduce_exactly():
    def run_once():
        game = Game(
            choosers=(make_random_chooser(101), make_random_chooser(202)),
            config=GameConfig(seed=999, max_turns=40),
        )
        result = game.run()
        return result, game.to_json(include_hidden=True)

    assert run_once() == run_once()


def test_random_seating_is_seeded_and_fixed_seating_is_explicit():
    fixed = Game(config=GameConfig(seed=8, seating=Seating.FIXED))
    random_a = Game(config=GameConfig(seed=8, seating=Seating.RANDOM))
    random_b = Game(config=GameConfig(seed=8, seating=Seating.RANDOM))
    explicit = Game(config=GameConfig(seed=8, starting_player=1))
    assert fixed.starting_player == 0
    assert random_a.starting_player == random_b.starting_player
    assert explicit.starting_player == 1
    assert explicit.observation(1).is_starting_player is True
    assert explicit.observation(0).is_starting_player is False
    assert explicit.observation(0).starting_player == 1


def test_hidden_order_permutations_produce_identical_observations():
    game = Game(config=GameConfig(seed=6))
    before = game.observation(0)
    game.players[1].deck.reverse()
    game.trade_deck.reverse()
    after = game.observation(0)
    assert before == after
    assert before.opponent_hidden
    assert before.trade_deck


def test_opponent_hand_deck_assignment_is_not_in_policy_observation():
    game = Game(config=GameConfig(seed=6))
    opponent = game.players[1]
    before = game.observation(0)
    opponent.hand[0], opponent.deck[0] = opponent.deck[0], opponent.hand[0]
    after = game.observation(0)
    assert before == after


def test_known_top_cards_are_not_duplicated_in_hidden_multisets():
    game = Game(config=GameConfig(seed=61))
    opponent = game.players[1]
    opponent.deck.append(EXPLORER)
    opponent.known_top.append(EXPLORER)
    observed = game.observation(0)
    hidden = Counter(card.card_id for card in observed.opponent_hidden)
    assert observed.opponent_known_top == (EXPLORER,)
    assert hidden.get(EXPLORER.card_id, 0) == 0


def test_semantic_dedup_keeps_distinct_cards_without_dominance_pruning():
    game = Game(config=GameConfig(seed=3))
    player = game.players[game.active_player]
    player.hand = [SCOUT, SCOUT, VIPER, VIPER]
    actions = game._main_actions(player)
    play_actions = [action for action in actions if action.kind == ActionKind.PLAY_CARD]
    assert [action.card_id for action in play_actions] == [SCOUT.card_id, VIPER.card_id]


def test_exactly_one_action_is_forced_without_calling_chooser():
    def forbidden(_player_id, _decision):
        raise AssertionError("chooser must not be called for a forced choice")

    game = Game(choosers=(forbidden, forbidden), config=GameConfig(seed=1, max_turns=1))
    player = game.players[game.active_player]
    player.discard.extend(player.hand)
    player.hand.clear()
    result = game.run()
    assert result.decisions == 1
    assert result.forced_choices == 1


def test_blob_carrier_checks_fifth_market_slot_and_places_ship_on_top():
    game = Game(config=GameConfig(seed=12))
    player = game.players[0]
    bases = [card for card in ALL_CARDS if card.card_type != CardType.SHIP]
    game.trade_row = bases[:4] + [CARD_BY_NAME["Battle Blob"]]
    source = _InPlay(100, CARD_BY_NAME["Blob Carrier"], CARD_BY_NAME["Blob Carrier"], True)
    old_top = player.deck[-1]
    game._free_ship(player, source)
    assert player.deck[-1].name == "Battle Blob"
    assert player.deck[-2] == old_top
    assert len(game.trade_row) == 5


def test_blob_carrier_free_acquisition_is_optional():
    def decline(_player_id, decision):
        return next(
            index
            for index, action in enumerate(decision.actions)
            if action.kind == ActionKind.DECLINE
        )

    game = Game(choosers=(decline, decline), config=GameConfig(seed=121))
    player = game.players[0]
    source = _InPlay(100, CARD_BY_NAME["Blob Carrier"], CARD_BY_NAME["Blob Carrier"], True)
    row_before = tuple(game.trade_row)
    deck_before = tuple(player.deck)
    game._free_ship(player, source)
    assert tuple(game.trade_row) == row_before
    assert tuple(player.deck) == deck_before


def test_stealth_needle_copying_blob_does_not_count_as_blob_play():
    game = Game(config=GameConfig(seed=13))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Battle Blob"]]
    game._play_card(player, 0)
    assert player.blob_cards_played == 1
    player.hand = [CARD_BY_NAME["Stealth Needle"]]
    game._play_card(player, 0)
    assert player.in_play[-1].card.name == "Battle Blob"
    assert player.blob_cards_played == 1


def test_draw_destroy_draws_before_presenting_destroy_choice():
    seen_hand_sizes = []

    def hook(_player_id, decision, _action):
        if decision.family == DecisionFamily.DESTROY_BASE:
            seen_hand_sizes.append(len(decision.observation.hand))

    game = Game(config=GameConfig(seed=14), decision_hook=hook)
    player = game.players[0]
    opponent = game.players[1]
    player.deck = [SCOUT]
    player.hand.clear()
    source_card = CARD_BY_NAME["Battlecruiser"]
    source = _InPlay(101, source_card, source_card, True)
    player.in_play = [source]
    base = CARD_BY_NAME["Blob Wheel"]
    opponent.in_play = [_InPlay(102, base, base, True)]
    game._scrap_in_play(player, source)
    assert seen_hand_sizes == [1]


def test_forced_discard_precedes_base_activation():
    families = []

    def chooser(_player_id, decision):
        families.append(decision.family)
        return end_turn_chooser(_player_id, decision)

    game = Game(choosers=(chooser, chooser), config=GameConfig(seed=15, max_turns=1))
    player = game.players[game.active_player]
    base = CARD_BY_NAME["Machine Base"]
    player.in_play.append(_InPlay(103, base, base))
    player.must_discard = 1
    game.run()
    assert families[0] == DecisionFamily.DISCARD
    assert families[1] == DecisionFamily.MAIN


def test_new_base_ability_can_be_scheduled_later_in_main_phase():
    game = Game(config=GameConfig(seed=151))
    player = game.players[0]
    blob_world = CARD_BY_NAME["Blob World"]
    player.hand = [blob_world]
    decisions_before = game.decisions
    game._play_card(player, 0)
    item = player.in_play[-1]
    assert not item.activated
    assert game.decisions == decisions_before
    assert any(
        action.kind == ActionKind.ACTIVATE_BASE and action.card_id == blob_world.card_id
        for action in game._main_actions(player)
    )


def test_scrap_only_base_does_not_emit_a_no_op_activation():
    game = Game(config=GameConfig(seed=152))
    player = game.players[0]
    station = CARD_BY_NAME["Battle Station"]
    player.hand = [station]
    game._play_card(player, 0)
    actions = game._main_actions(player)
    assert not any(action.kind == ActionKind.ACTIVATE_BASE for action in actions)
    assert any(action.kind == ActionKind.SCRAP_FOR_ABILITY for action in actions)


def test_end_turn_is_legal_with_cards_remaining_and_discards_them():
    game = Game(
        choosers=(end_turn_chooser, end_turn_chooser),
        config=GameConfig(seed=16, max_turns=1),
    )
    player = game.players[game.active_player]
    opening_hand = list(player.hand)
    result = game.run()
    assert result.truncation_reason == "max_turns"
    assert len(player.hand) == 5
    assert all(card in player.discard for card in opening_hand)


def test_safeguards_are_truncated_draws_not_fabricated_winners():
    turn_cap = Game(config=GameConfig(seed=20, max_turns=1)).run()
    action_cap = Game(config=GameConfig(seed=20, max_actions_per_turn=1)).run()
    cancelled = Game(config=GameConfig(seed=20), cancel_hook=lambda: True).run()
    assert (turn_cap.winner, turn_cap.truncation_reason) == (None, "max_turns")
    assert (action_cap.winner, action_cap.truncation_reason) == (
        None,
        "max_actions_per_turn",
    )
    assert (cancelled.winner, cancelled.truncation_reason) == (None, "cancelled")


def test_scrap_heap_and_card_conservation_cover_all_physical_cards():
    game = Game(config=GameConfig(seed=23))
    expected = game.card_conservation()
    player = game.players[0]
    game.explorers_remaining -= 1
    player.hand.append(EXPLORER)
    game._uid += 1
    game._play_card(player, len(player.hand) - 1)
    item = player.in_play[-1]
    game._scrap_in_play(player, item)
    assert game.scrap_heap[-1] == EXPLORER
    assert game.card_conservation() == expected


@pytest.mark.parametrize("card", ALL_CARDS, ids=lambda card: card.name)
def test_every_card_effect_is_implemented(card):
    game = Game(config=GameConfig(seed=card.card_id + 100, max_actions_per_turn=100))
    player = game.players[0]
    source = _InPlay(1000 + card.card_id, card, card, True)
    # Direct effect smoke tests isolate the effect dispatch table from draw and
    # market luck while still exercising every primary, ally, and scrap opcode.
    for effect, amount in (
        (card.primary, 0),
        (card.ally, card.ally_amount),
        (card.scrap, card.scrap_amount),
    ):
        if effect:
            game._execute_effect(player, effect, amount, source)
