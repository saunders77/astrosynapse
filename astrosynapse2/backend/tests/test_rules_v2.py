"""Official rule examples, separately versioned from historical training games."""

import numpy as np
import pytest
from astro2.cards import CARD_BY_NAME, EXPLORER, SCOUT
from astro2.encoding import Encoder
from astro2.engine import ActionKind, Decision, DecisionFamily, Game, GameConfig, _InPlay
from astro2.engine_encoding import EngineEncoder


@pytest.mark.parametrize("zone", ["hand", "discard", "in_play"])
@pytest.mark.parametrize("version", [1, 2])
def test_explorer_scrap_destination_for_every_player_zone(zone, version):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.hand = []
    player.discard = []
    player.in_play = []
    source = _InPlay(100, CARD_BY_NAME["Trade Bot"], CARD_BY_NAME["Trade Bot"])
    game.explorers_remaining = 9
    if zone == "in_play":
        item = _InPlay(101, EXPLORER, EXPLORER)
        player.in_play.append(item)
        game._scrap_in_play(player, item)
        assert player.combat == 2
    else:
        getattr(player, zone).append(EXPLORER)
        game.choosers[0] = lambda _, decision: next(
            a for a in decision.actions if a.kind == ActionKind.SCRAP_CARD
        )
        game._scrap_any(player, source, required=False)
        assert player.combat == 0
    assert game.explorers_remaining == (10 if version == 2 else 9)
    assert (EXPLORER in game.scrap_heap) == (version == 1)


@pytest.mark.parametrize("version", [1, 2])
def test_ally_draw_can_wait_until_after_a_purchase_and_only_runs_once(version):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Blob Fighter"], CARD_BY_NAME["Trade Pod"]]
    player.deck = [SCOUT]
    player.in_play = []
    game._play_card(player, 0)
    game._play_card(player, 0)
    assert player.hand == []  # Ally draw is available, not automatically used.
    ally = next(
        a
        for a in game._main_actions(player)
        if a.kind == ActionKind.ACTIVATE_ALLY and a.card_id == 7
    )
    observation = game.observation(0)
    decision = Decision(DecisionFamily.MAIN, observation, game._main_actions(player))
    fast = EngineEncoder(version=2).encode_decision(observation, decision)
    generic = Encoder(version=2).encode_decision(observation, decision)
    np.testing.assert_allclose(fast.actions, generic.actions)
    # Using an unrelated available action does not consume the ally draw.
    game._acquire_explorer(player, 2)
    assert ally in game._main_actions(player)
    game._apply_main_action(player, ally)
    assert player.hand == [SCOUT]
    assert ally not in game._main_actions(player)
    with pytest.raises(RuntimeError, match="not available"):
        game._apply_main_action(player, ally)


@pytest.mark.parametrize("version", [1, 2])
def test_resource_allies_trigger_immediately_without_a_policy_action(version):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Trade Pod"], CARD_BY_NAME["Battle Pod"]]
    player.in_play = []

    game._play_card(player, 0)
    assert player.trade == 3
    assert player.combat == 0
    game._play_card(player, 0)

    assert player.combat == 8  # 4 printed + 2 from each now-allied card.
    assert all(item.ally_triggered for item in player.in_play)
    assert not any(
        action.kind == ActionKind.ACTIVATE_ALLY for action in game._main_actions(player)
    )


def test_base_resources_are_immediate_but_choice_and_draw_abilities_are_manual():
    game = Game(config=GameConfig(rules_version=2))
    player = game.players[0]
    player.hand = [
        CARD_BY_NAME["Central Office"],
        CARD_BY_NAME["Defense Center"],
        CARD_BY_NAME["Port of Call"],
    ]
    player.in_play = []

    game._play_card(player, 0)
    central = player.in_play[-1]
    assert player.trade == 2
    assert not central.activated
    central_action = next(
        action for action in game._main_actions(player) if action.card_id == central.card.card_id
    )
    game._apply_main_action(player, central_action)
    assert player.trade == 2  # Activating ship-to-top does not grant trade twice.

    game._play_card(player, 0)
    defense = player.in_play[-1]
    assert player.combat == 2  # Its resource ally fired immediately via Central Office.
    assert defense.ally_triggered
    assert not defense.activated
    assert any(
        action.kind == ActionKind.ACTIVATE_BASE and action.card_id == defense.card.card_id
        for action in game._main_actions(player)
    )

    game._play_card(player, 0)
    port = player.in_play[-1]
    assert player.trade == 5
    assert port.activated
    assert not any(
        action.kind == ActionKind.ACTIVATE_BASE and action.card_id == port.card.card_id
        for action in game._main_actions(player)
    )


@pytest.mark.parametrize("version", [1, 2])
def test_ship_draw_primary_is_manual_while_printed_resources_are_immediate(version):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Flagship"]]
    player.deck = [SCOUT]
    player.in_play = []

    game._play_card(player, 0)
    flagship = player.in_play[-1]
    assert player.combat == 5
    assert player.hand == []
    activate = next(
        action
        for action in game._main_actions(player)
        if action.kind == ActionKind.ACTIVATE_BASE and action.card_id == flagship.card.card_id
    )
    assert activate.ability == "draw"
    assert activate.label.startswith("activate ability Flagship")
    game._apply_main_action(player, activate)
    assert player.hand == [SCOUT]


def test_authority_allies_trigger_immediately():
    game = Game(config=GameConfig(rules_version=2))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Federation Shuttle"], CARD_BY_NAME["Flagship"]]
    player.in_play = []

    game._play_card(player, 0)
    game._play_card(player, 0)

    assert player.authority == 59
    assert player.trade == 2
    assert player.combat == 5
    assert all(item.ally_triggered for item in player.in_play)


def test_stealth_needle_waits_for_a_copy_target_then_gains_copied_resources():
    game = Game(config=GameConfig(rules_version=2))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Stealth Needle"], CARD_BY_NAME["Viper"]]
    player.in_play = []

    game._play_card(player, 0)
    assert not any(
        action.kind == ActionKind.ACTIVATE_BASE for action in game._main_actions(player)
    )
    game._play_card(player, 0)
    activate = next(
        action
        for action in game._main_actions(player)
        if action.kind == ActionKind.ACTIVATE_BASE
    )
    game._apply_main_action(player, activate)

    assert player.in_play[0].card.name == "Viper"
    assert player.combat == 2


def test_surviving_bases_gain_resources_and_resource_allies_at_turn_start():
    seen = []

    def choose_end(_player_id, decision):
        seen.append(decision.observation)
        return next(action for action in decision.actions if action.kind == ActionKind.END_TURN)

    game = Game(choosers=(choose_end, choose_end), config=GameConfig(rules_version=2))
    player = game.players[0]
    player.hand = []
    player.in_play = [
        _InPlay(100, CARD_BY_NAME["Central Office"], CARD_BY_NAME["Central Office"]),
        _InPlay(101, CARD_BY_NAME["Defense Center"], CARD_BY_NAME["Defense Center"]),
        _InPlay(102, CARD_BY_NAME["Port of Call"], CARD_BY_NAME["Port of Call"]),
    ]

    game._take_turn(player)

    assert seen[0].trade == 5
    assert seen[0].combat == 2
    assert seen[0].own_in_play[1].ally_triggered


def test_mech_world_enables_other_factions_and_ally_needs_a_card_in_play():
    game = Game(config=GameConfig(rules_version=2))
    player = game.players[0]
    fighter = _InPlay(100, CARD_BY_NAME["Blob Fighter"], CARD_BY_NAME["Blob Fighter"])
    world = _InPlay(101, CARD_BY_NAME["Mech World"], CARD_BY_NAME["Mech World"])
    player.in_play = [fighter]
    assert not game._ally_available(player, fighter)
    player.in_play.append(world)
    assert game._ally_available(player, fighter)
    player.in_play.remove(world)
    assert not game._ally_available(player, fighter)


def test_scrapping_copied_explorer_returns_needle_not_an_extra_explorer():
    game = Game(config=GameConfig(rules_version=2))
    player = game.players[0]
    item = _InPlay(100, EXPLORER, CARD_BY_NAME["Stealth Needle"])
    player.in_play = [item]
    game._scrap_in_play(player, item)
    assert game.explorers_remaining == 10
    assert game.scrap_heap == [CARD_BY_NAME["Stealth Needle"]]
