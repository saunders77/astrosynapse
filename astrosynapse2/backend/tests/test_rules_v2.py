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
def test_ally_draw_is_immediate_and_only_runs_once(version):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Blob Fighter"], CARD_BY_NAME["Trade Pod"]]
    player.deck = [SCOUT]
    player.in_play = []
    game._play_card(player, 0)
    game._play_card(player, 0)
    assert player.hand == [SCOUT]
    assert not any(a.kind == ActionKind.ACTIVATE_ALLY for a in game._main_actions(player))
    observation = game.observation(0)
    decision = Decision(DecisionFamily.MAIN, observation, game._main_actions(player))
    fast = EngineEncoder(version=2).encode_decision(observation, decision)
    generic = Encoder(version=2).encode_decision(observation, decision)
    np.testing.assert_allclose(fast.actions, generic.actions)
    game._acquire_explorer(player, 2)
    game._trigger_automatic_allies(player)
    assert player.hand == [SCOUT]


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
    assert central.activated
    assert player.next_ship_top
    assert not any(
        a.kind == ActionKind.ACTIVATE_BASE and a.card_id == central.card.card_id
        for a in game._main_actions(player)
    )

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
def test_ship_draw_primary_and_printed_resources_are_immediate(version):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Flagship"]]
    player.deck = [SCOUT]
    player.in_play = []

    game._play_card(player, 0)
    flagship = player.in_play[-1]
    assert player.combat == 5
    assert player.hand == [SCOUT]
    assert flagship.activated
    assert not any(a.kind == ActionKind.ACTIVATE_BASE for a in game._main_actions(player))


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


def test_stealth_needle_without_a_target_cannot_copy_a_later_ship():
    game = Game(config=GameConfig(rules_version=2))
    player = game.players[0]
    player.hand = [CARD_BY_NAME["Stealth Needle"], CARD_BY_NAME["Viper"]]
    player.in_play = []

    game._play_card(player, 0)
    assert not any(
        action.kind == ActionKind.ACTIVATE_BASE for action in game._main_actions(player)
    )
    game._play_card(player, 0)
    assert not any(a.kind == ActionKind.ACTIVATE_BASE for a in game._main_actions(player))
    assert player.in_play[0].card.name == "Stealth Needle"
    assert player.in_play[0].activated
    assert player.combat == 1


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


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("late_base", [False, True])
def test_embassy_yacht_draws_automatically_once_two_bases_are_present(version, late_base):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    base = CARD_BY_NAME["Battle Station"]
    player.in_play = [_InPlay(100, base, base, True)]
    if not late_base:
        player.in_play.append(_InPlay(101, base, base, True))
    player.deck = [SCOUT, EXPLORER, SCOUT]
    player.hand = [CARD_BY_NAME["Embassy Yacht"]]
    game._play_card(player, 0)
    if late_base:
        assert not player.hand
        assert not any(a.kind == ActionKind.ACTIVATE_BASE for a in game._main_actions(player))
        player.hand.append(base)
        game._play_card(player, 0)
    assert player.hand == [SCOUT, EXPLORER]
    game._trigger_automatic_allies(player)
    assert player.hand == [SCOUT, EXPLORER]


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("name", ["Central Office", "Freighter"])
def test_next_ship_top_is_automatic_mandatory_and_consumed_once(version, name):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.in_play = []
    player.hand = [CARD_BY_NAME[name], CARD_BY_NAME["Federation Shuttle"]]
    game._play_card(player, 0)
    assert player.next_ship_top == (name == "Central Office")
    game._play_card(player, 0)
    assert player.next_ship_top
    assert not any(a.ability == "ship_top" for a in game._main_actions(player))
    game._place_acquired(player, CARD_BY_NAME["Battle Station"])
    assert player.next_ship_top
    game._acquire_explorer(player, 2)
    assert player.deck[-1] == EXPLORER
    assert not player.next_ship_top
    game._trigger_automatic_allies(player)
    game._place_acquired(player, SCOUT)
    assert player.discard[-1] == SCOUT


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("name,family", [
    ("Trade Bot", DecisionFamily.SCRAP),
    ("Battle Pod", DecisionFamily.SCRAP_TRADE_ROW),
    ("Missile Mech", DecisionFamily.DESTROY_BASE),
    ("Patrol Mech", DecisionFamily.ABILITY_MODE),
])
def test_ship_primary_prompts_happen_inside_play(version, name, family):
    seen = []

    def choose(_, decision):
        seen.append(decision.family)
        return 0

    game = Game(choosers=(choose, choose), config=GameConfig(rules_version=version))
    player = game.players[0]
    player.in_play = []
    player.hand = [CARD_BY_NAME[name], SCOUT]
    base = CARD_BY_NAME["Battle Station"]
    game.players[1].in_play = [_InPlay(100, base, base, True)]
    game._play_card(player, 0)
    assert seen == [family]
    assert not any(a.kind == ActionKind.ACTIVATE_BASE for a in game._main_actions(player))


@pytest.mark.parametrize("version", [1, 2])
def test_needle_copied_draw_resolves_immediately_and_fleet_hq_counts_once(version):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.in_play = [
        _InPlay(i + 100, CARD_BY_NAME[name], CARD_BY_NAME[name], True)
        for i, name in enumerate(["Fleet HQ", "Corvette"])
    ]
    player.hand = [CARD_BY_NAME["Stealth Needle"]]
    player.deck = [SCOUT]
    game._play_card(player, 0)
    assert player.hand == [SCOUT]
    assert player.combat == 6  # Copied combat + one Fleet HQ bonus + both Corvette allies.
    assert player.in_play[-1].activated


@pytest.mark.parametrize("version", [1, 2])
def test_surviving_central_office_arms_top_and_draws_after_forced_discard(version):
    seen = []

    def choose(_, decision):
        seen.append(decision)
        if decision.family == DecisionFamily.MAIN:
            return next(a for a in decision.actions if a.kind == ActionKind.END_TURN)
        return 0

    game = Game(choosers=(choose, choose), config=GameConfig(rules_version=version))
    player = game.players[0]
    player.in_play = [
        _InPlay(i + 100, CARD_BY_NAME[name], CARD_BY_NAME[name], True, True)
        for i, name in enumerate(["Central Office", "Port of Call"])
    ]
    player.hand = [SCOUT, EXPLORER]
    player.deck = [CARD_BY_NAME["Viper"]]
    player.must_discard = 1
    game._take_turn(player)
    assert seen[0].family == DecisionFamily.DISCARD
    assert CARD_BY_NAME["Viper"] not in seen[0].observation.hand
    assert seen[1].family == DecisionFamily.MAIN
    assert CARD_BY_NAME["Viper"] in seen[1].observation.hand
    assert seen[1].observation.next_ship_to_top
    assert not any(a.ability in {"draw", "ship_top"} for a in seen[1].actions)


@pytest.mark.parametrize("version", [1, 2])
def test_mech_world_immediately_triggers_ship_ally_draw(version):
    game = Game(config=GameConfig(rules_version=version))
    player = game.players[0]
    player.in_play = []
    player.hand = [CARD_BY_NAME["Blob Fighter"], CARD_BY_NAME["Mech World"]]
    player.deck = [SCOUT]
    game._play_card(player, 0)
    game._play_card(player, 0)
    assert player.hand == [SCOUT]
    assert player.in_play[0].ally_triggered
