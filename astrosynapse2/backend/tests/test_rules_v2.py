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


def test_ally_draw_can_wait_until_after_a_purchase_and_only_runs_once():
    game = Game(config=GameConfig(rules_version=2))
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
